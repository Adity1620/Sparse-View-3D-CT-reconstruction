"""
R-VGGT Phase 1: Complete Medical Data + Physics Grounding
Generates physics-correct DRRs with proper camera geometry and ground truth volumes.

Usage:
    python phase1_complete.py --input_path /path/to/ct.nii --output_dir data/patient_001
"""

import argparse
from pathlib import Path
from typing import Tuple, Dict

import imageio.v2 as imageio
import nibabel as nib
import numpy as np
import pydicom
import torch
import torchio as tio
from diffdrr.drr import DRR
from diffdrr.pose import convert


# ============================================================================
# DATA LOADING
# ============================================================================

def load_nifti(nifti_path: Path) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """Load NIfTI volume and extract spacing."""
    nii = nib.load(str(nifti_path))
    volume = nii.get_fdata().astype(np.float32)
    zooms = nii.header.get_zooms()
    if len(zooms) < 3:
        raise ValueError("NIfTI spacing must include 3 spatial dimensions.")
    spacing_xyz = (float(zooms[0]), float(zooms[1]), float(zooms[2]))
    return volume, spacing_xyz


def load_dicom_series(dicom_dir: Path) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """Load DICOM series and convert to HU volume."""
    files = [p for p in dicom_dir.iterdir() if p.is_file()]
    slices = []
    for fp in files:
        try:
            ds = pydicom.dcmread(str(fp))
            slices.append(ds)
        except Exception:
            continue
    
    if not slices:
        raise RuntimeError(f"No readable DICOM slices found in {dicom_dir}")

    def sort_key(ds):
        if "ImagePositionPatient" in ds:
            return float(ds.ImagePositionPatient[2])
        if "InstanceNumber" in ds:
            return int(ds.InstanceNumber)
        return 0

    slices.sort(key=sort_key)
    first = slices[0]
    sx = float(first.PixelSpacing[1])
    sy = float(first.PixelSpacing[0])
    sz = float(getattr(first, "SliceThickness", 1.0))

    volume = []
    for ds in slices:
        arr = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        hu = arr * slope + intercept
        volume.append(hu)

    volume = np.stack(volume, axis=0)  # [Z, Y, X]
    volume = np.transpose(volume, (2, 1, 0))  # [X, Y, Z]
    return volume, (sx, sy, sz)


# ============================================================================
# PREPROCESSING
# ============================================================================

def window_and_normalize(
    volume_hu: np.ndarray, 
    hu_min: float = -1000.0, 
    hu_max: float = 1000.0
) -> np.ndarray:
    """Window HU values and normalize to [0, 1]."""
    vol = np.clip(volume_hu, hu_min, hu_max)
    vol = (vol - hu_min) / (hu_max - hu_min)
    return vol.astype(np.float32)


def build_subject(volume_xyz: np.ndarray, spacing_xyz: Tuple[float, float, float]) -> tio.Subject:
    """Build TorchIO subject for DiffDRR."""
    vol_tensor = torch.from_numpy(volume_xyz).unsqueeze(0)  # [1, X, Y, Z]
    sx, sy, sz = spacing_xyz

    # Build affine transformation
    affine = np.eye(4, dtype=np.float32)
    affine[0, 0], affine[1, 1], affine[2, 2] = sx, sy, sz
    xdim, ydim, zdim = volume_xyz.shape
    affine[0, 3] = -(xdim - 1) * sx / 2.0
    affine[1, 3] = -(ydim - 1) * sy / 2.0
    affine[2, 3] = -(zdim - 1) * sz / 2.0

    img = tio.ScalarImage(tensor=vol_tensor, affine=affine)
    subject = tio.Subject(
        volume=img, 
        density=tio.ScalarImage(tensor=vol_tensor.clone(), affine=affine)
    )

    if not hasattr(subject, "mask"):
        subject.mask = None
    subject.reorient = torch.eye(4).unsqueeze(0)
    return subject


# ============================================================================
# CAMERA GEOMETRY
# ============================================================================

def make_sparse_poses(
    n_views: int, 
    sdd: float, 
    device: torch.device, 
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Generate sparse viewing poses uniformly distributed around the volume.
    
    Returns:
        poses: [n_views, 4, 4] camera-to-world transformation matrices
    """
    angles = torch.linspace(0.0, 2.0 * np.pi, n_views + 1, device=device, dtype=dtype)[:-1]
    sod = sdd / 2.0  # Source-to-object distance
    
    # Source position along Z-axis
    translation = torch.tensor([0.0, 0.0, sod], device=device, dtype=dtype)
    translation = translation.unsqueeze(0).repeat(n_views, 1)
    
    # Rotation around Y-axis (standard circular orbit)
    rotation = torch.stack(
        [
            torch.zeros_like(angles),
            angles,
            torch.zeros_like(angles),
        ],
        dim=-1,
    )
    
    return convert(
        rotation, 
        translation, 
        parameterization="euler_angles", 
        convention="XYZ", 
        degrees=False
    )


# ============================================================================
# X-RAY RENDERING
# ============================================================================

def line_integrals_to_intensity(
    line_integrals: torch.Tensor, 
    target_at_p99: float = 0.25,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    Convert line integrals to X-ray intensities using Beer-Lambert law.
    Calibrates per-view so 99th percentile attenuation maps to target intensity.
    
    Args:
        line_integrals: [V, 1, H, W] accumulated attenuation
        target_at_p99: Target intensity (0-1) for 99th percentile attenuation
        
    Returns:
        intensities: [V, 1, H, W] X-ray intensities in [0, 1]
    """
    flat = line_integrals.flatten(start_dim=2)
    p99 = torch.quantile(flat, 0.99, dim=2, keepdim=True).clamp_min(eps)
    
    # Calculate attenuation coefficient: I = exp(-alpha * mu)
    # If p99 attenuation should give target_at_p99 intensity:
    # target_at_p99 = exp(-alpha * p99)
    # alpha = -log(target_at_p99) / p99
    alpha = (-np.log(target_at_p99)) / p99
    alpha = alpha.unsqueeze(-1)  # [V, 1, 1, 1]
    
    return torch.exp(-line_integrals * alpha)


def add_poisson_noise(projections: torch.Tensor, dose_factor: float) -> torch.Tensor:
    """Add realistic Poisson noise to simulate X-ray dose."""
    scaled = torch.clamp(projections, min=0.0) * dose_factor
    return torch.poisson(scaled) / dose_factor


def auto_exposure(img: np.ndarray, low: float = 1.0, high: float = 99.0, eps: float = 1e-6) -> np.ndarray:
    """Apply percentile-based auto-exposure for better visualization."""
    lo = float(np.percentile(img, low))
    hi = float(np.percentile(img, high))
    if hi - lo < eps:
        return np.clip(img, 0.0, 1.0)
    return np.clip((img - lo) / (hi - lo), 0.0, 1.0)


# ============================================================================
# SAVING UTILITIES
# ============================================================================

def save_xrays(
    projections: torch.Tensor, 
    out_dir: Path, 
    use_auto_exposure: bool = True,
    prefix: str = "xray_view"
) -> None:
    """Save X-ray projections as PNG images."""
    out_dir.mkdir(parents=True, exist_ok=True)
    views = projections.detach().cpu().numpy()[:, 0]  # [V, H, W]
    
    for i, img in enumerate(views):
        if use_auto_exposure:
            img = auto_exposure(img)
        else:
            img = np.clip(img, 0.0, 1.0)
        
        out_path = out_dir / f"{prefix}_{i:03d}.png"
        imageio.imwrite(out_path, (img * 255.0).astype(np.uint8))
    
    print(f"✓ Saved {views.shape[0]} X-rays to: {out_dir}")


def save_training_data(
    xrays: torch.Tensor,
    poses,  # Can be RigidTransform or Tensor
    volume_gt: np.ndarray,
    spacing_xyz: Tuple[float, float, float],
    camera_params: Dict,
    out_dir: Path
) -> None:
    """Save complete training dataset."""
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert poses to tensor if it's a RigidTransform object
    if hasattr(poses, 'matrix'):
        # DiffDRR RigidTransform object - extract the transformation matrix
        poses_tensor = poses.matrix.cpu()
    elif isinstance(poses, torch.Tensor):
        poses_tensor = poses.cpu()
    else:
        raise TypeError(f"Unsupported poses type: {type(poses)}")
    
    # Save everything in one file for easy loading
    data = {
        'xrays': xrays.cpu(),              # [1, V, 3, H, W]
        'poses': poses_tensor,             # [V, 4, 4]
        'volume_gt': torch.from_numpy(volume_gt),  # [X, Y, Z]
        'spacing': spacing_xyz,            # (sx, sy, sz)
        'camera': camera_params,
        'metadata': {
            'n_views': xrays.shape[1],
            'resolution': (xrays.shape[3], xrays.shape[4]),
            'volume_shape': volume_gt.shape,
        }
    }
    
    out_path = out_dir / 'training_data.pt'
    torch.save(data, out_path)
    print(f"✓ Saved training data to: {out_path}")
    
    # Print summary
    print("\n=== Dataset Summary ===")
    print(f"X-rays shape: {tuple(data['xrays'].shape)}")
    print(f"Camera poses: {tuple(data['poses'].shape)}")
    print(f"Ground truth volume: {tuple(data['volume_gt'].shape)}")
    print(f"Spacing (mm): {spacing_xyz}")
    print(f"SDD: {camera_params['sdd']} mm")


# ============================================================================
# MAIN PROCESSING
# ============================================================================

def process_phase1(
    input_path: str,
    output_dir: str,
    n_views: int = 4,
    height: int = 256,
    width: int = 256,
    delx: float = 1.5,
    sdd: float = 1020.0,
    hu_min: float = -1000.0,
    hu_max: float = 1000.0,
    dose_factor: float = 20000.0,
    target_at_p99: float = 0.25,
    device: torch.device = None,
) -> Dict:
    """
    Complete Phase 1 processing pipeline.
    
    Returns:
        Dictionary containing all processed data
    """
    print("=" * 70)
    print("PHASE 1: Medical Data + Physics Grounding")
    print("=" * 70)
    
    # Setup
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    if device is None:
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = 'cpu'
    print(f"\nDevice: {device}")
    
    # Load volume
    print(f"\n[1/5] Loading volume from: {input_path}")
    if input_path.is_dir():
        volume_hu_xyz, spacing_xyz = load_dicom_series(input_path)
        print(f"  ✓ Loaded DICOM series")
    else:
        volume_hu_xyz, spacing_xyz = load_nifti(input_path)
        print(f"  ✓ Loaded NIfTI")
    
    print(f"  Volume shape [X,Y,Z]: {volume_hu_xyz.shape}")
    print(f"  Spacing [sx,sy,sz] mm: {spacing_xyz}")
    
    # Preprocess
    print(f"\n[2/5] Preprocessing (HU window: [{hu_min}, {hu_max}])")
    volume_norm_xyz = window_and_normalize(volume_hu_xyz, hu_min=hu_min, hu_max=hu_max)
    subject = build_subject(volume_norm_xyz, spacing_xyz)
    print(f"  ✓ Normalized to [0, 1]")
    
    # Setup DRR renderer
    print(f"\n[3/5] Setting up virtual X-ray scanner")
    print(f"  Views: {n_views}")
    print(f"  Detector: {height}×{width} pixels, {delx} mm/pixel")
    print(f"  SDD: {sdd} mm")
    drr = DRR(subject, sdd=sdd, height=height, width=width, delx=delx).to(device)
    poses = make_sparse_poses(n_views=n_views, sdd=sdd, device=device)
    print(f"  ✓ Camera poses generated")
    
    # Render X-rays
    print(f"\n[4/5] Rendering synthetic X-rays (Beer-Lambert law)")
    with torch.no_grad():
        line_integrals = drr(poses)  # [V, 1, H, W]
        clean_projections = line_integrals_to_intensity(
            line_integrals, 
            target_at_p99=target_at_p99
        )
        noisy_projections = add_poisson_noise(clean_projections, dose_factor=dose_factor)
    
    print(f"  ✓ Line integrals: {tuple(line_integrals.shape)}")
    print(f"  ✓ Applied Beer-Lambert attenuation")
    print(f"  ✓ Added Poisson noise (dose factor: {dose_factor})")
    
    # Prepare final format [1, V, 3, H, W] for VGGT
    eps = 1e-6
    atten = -torch.log(noisy_projections + eps)
    atten = (atten - atten.mean(dim=(2,3), keepdim=True)) / (atten.std(dim=(2,3), keepdim=True) + 1e-6)
    final_xrays = atten.repeat(1, 3, 1, 1).unsqueeze(0)
    final_xrays = torch.clamp(final_xrays, 0.0, 1.0)
    
    print(f"\n  Final X-rays: {tuple(final_xrays.shape)}")
    print(f"  Intensity range: [{float(final_xrays.min()):.4f}, {float(final_xrays.max()):.4f}]")
    
    # Save everything
    print(f"\n[5/5] Saving outputs to: {output_dir}")
    
    # Save visualizations
    save_xrays(
        noisy_projections, 
        output_dir / 'xrays', 
        use_auto_exposure=True
    )
    
    # Save training data
    camera_params = {
        'sdd': sdd,
        'height': height,
        'width': width,
        'delx': delx,
    }
    
    save_training_data(
        xrays=final_xrays,
        poses=poses,
        volume_gt=volume_norm_xyz,
        spacing_xyz=spacing_xyz,
        camera_params=camera_params,
        out_dir=output_dir
    )
    
    print("\n" + "=" * 70)
    print("PHASE 1 COMPLETE ✓")
    print("=" * 70)
    print("\nNext step: Run Phase 2 to extract VGGT features")
    print(f"  python phase2_vggt_features.py --data_path {output_dir}/training_data.pt")
    
    return {
        'xrays': final_xrays,
        'poses': poses,
        'volume_gt': volume_norm_xyz,
        'spacing': spacing_xyz,
        'camera': camera_params,
    }


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Phase 1: Generate physics-correct X-ray projections from CT volumes"
    )
    
    # I/O
    parser.add_argument(
        "--input_path", 
        type=str, 
        required=True,
        help="Path to .nii/.nii.gz file or DICOM directory"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="data/phase1_v2_output",
        help="Output directory for processed data"
    )
    
    # Camera geometry
    parser.add_argument("--n_views", type=int, default=4, help="Number of sparse views")
    parser.add_argument("--height", type=int, default=256, help="Detector height (pixels)")
    parser.add_argument("--width", type=int, default=256, help="Detector width (pixels)")
    parser.add_argument("--delx", type=float, default=1.5, help="Detector pixel size (mm)")
    parser.add_argument("--sdd", type=float, default=1020.0, help="Source-to-detector distance (mm)")
    
    # Preprocessing
    parser.add_argument("--hu_min", type=float, default=-1000.0, help="HU window minimum")
    parser.add_argument("--hu_max", type=float, default=1000.0, help="HU window maximum")
    
    # X-ray simulation
    parser.add_argument("--dose_factor", type=float, default=20000.0, help="Poisson noise dose factor")
    parser.add_argument(
        "--target_at_p99", 
        type=float, 
        default=0.25,
        help="Beer-Lambert calibration: intensity for 99th percentile attenuation (higher = brighter)"
    )
    
    args = parser.parse_args()
    
    process_phase1(
        input_path=args.input_path,
        output_dir=args.output_dir,
        n_views=args.n_views,
        height=args.height,
        width=args.width,
        delx=args.delx,
        sdd=args.sdd,
        hu_min=args.hu_min,
        hu_max=args.hu_max,
        dose_factor=args.dose_factor,
        target_at_p99=args.target_at_p99,
    )


if __name__ == "__main__":
    main()