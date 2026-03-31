import argparse
from pathlib import Path
import numpy as np
import pydicom
import torch
import torchio as tio
from diffdrr.drr import DRR
from diffdrr.pose import RigidTransform
import imageio.v2 as imageio

def euler_xyz_to_matrix(angles):
    rx, ry, rz = angles[:, 0], angles[:, 1], angles[:, 2]
    cx, sx = torch.cos(rx), torch.sin(rx)
    cy, sy = torch.cos(ry), torch.sin(ry)
    cz, sz = torch.cos(rz), torch.sin(rz)
    r = torch.zeros((angles.shape[0], 3, 3), device=angles.device)
    r[:, 0, 0] = cy * cz
    r[:, 0, 1] = -cy * sz
    r[:, 0, 2] = sy
    r[:, 1, 0] = sx * sy * cz + cx * sz
    r[:, 1, 1] = -sx * sy * sz + cx * cz
    r[:, 1, 2] = -sx * cy
    r[:, 2, 0] = -cx * sy * cz + sx * sz
    r[:, 2, 1] = cx * sy * sz + sx * cz
    r[:, 2, 2] = cx * cy
    return r

def load_dicom_series(dicom_dir: Path):
    files = [p for p in dicom_dir.iterdir() if p.is_file()]
    slices = []
    for fp in files:
        try:
            ds = pydicom.dcmread(str(fp))
            slices.append(ds)
        except: continue
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]) if "ImagePositionPatient" in x else int(x.InstanceNumber))
    first = slices[0]
    spacing = (float(first.SliceThickness), float(first.PixelSpacing[0]), float(first.PixelSpacing[1]))
    volume = []
    for ds in slices:
        arr = ds.pixel_array.astype(np.float32)
        hu = arr * float(getattr(ds, "RescaleSlope", 1.0)) + float(getattr(ds, "RescaleIntercept", 0.0))
        volume.append(hu)
    return np.stack(volume, axis=0), spacing

def window_and_normalize(volume, hu_min=-1000.0, hu_max=1000.0):
    vol = np.clip(volume, hu_min, hu_max)
    vol = (vol - hu_min) / (hu_max - hu_min)
    return vol.astype(np.float32)

def auto_exposure(img, low=1.0, high=99.0, eps=1e-6):
    lo = np.percentile(img, low)
    hi = np.percentile(img, high)
    if hi - lo < eps: return np.clip(img, 0.0, 1.0)
    return np.clip((img - lo) / (hi - lo), 0.0, 1.0)

def build_subject_from_volume(volume, spacing):
    # 1. Transpose and Tensorize: diffdrr expects [X, Y, Z]
    vol_xyz = np.transpose(volume, (2, 1, 0))
    vol_tensor = torch.from_numpy(vol_xyz).unsqueeze(0) # [1, X, Y, Z]

    # 2. Setup Physical Geometry (Affine)
    sz, sy, sx = spacing
    affine = np.eye(4, dtype=np.float32)
    affine[0, 0], affine[1, 1], affine[2, 2] = sx, sy, sz
    
    # Centering logic to keep the patient in the crosshairs
    xdim, ydim, zdim = vol_xyz.shape
    affine[0, 3] = -(xdim - 1) * sx / 2.0
    affine[1, 3] = -(ydim - 1) * sy / 2.0
    affine[2, 3] = -(zdim - 1) * sz / 2.0

    # 3. Create the Image container
    volume_img = tio.ScalarImage(tensor=vol_tensor, affine=affine)
    
    # 4. Build the Subject and Inject Missing Attributes Manually
    subject = tio.Subject(
        volume=volume_img,
        # 'density' is used by diffdrr for the actual ray-tracing
        density=tio.ScalarImage(tensor=vol_tensor.clone(), affine=affine)
    )
    
    # --- SURGICAL FIXES FOR DIFFDRR ---
    # Many versions of diffdrr will crash if these aren't exactly like this:
    if not hasattr(subject, 'mask'):
        subject.mask = None
        
    # Standard 4x4 Identity for camera alignment
    subject.reorient = torch.eye(4).unsqueeze(0) 
    
    # Ensure physical spacing is accessible
    subject.spacing = torch.tensor([sx, sy, sz]).unsqueeze(0)
    
    return subject

def render_drr(subject, angles, height, width, delx, sdd, device):
    drr = DRR(subject, sdd=sdd, height=height, width=width, delx=delx).to(device)
    sod = sdd / 2.0
    n_views = len(angles)
    
    # Fix UserWarning: Ensure angles are handled as tensors correctly
    phi = angles.to(device).clone().detach().unsqueeze(-1)
    zeros = torch.zeros_like(phi)
    euler_angles = torch.cat([zeros, phi, zeros], dim=-1) # Rotate around Y-axis
    r_matrices = euler_xyz_to_matrix(euler_angles)
    
    # Fix Matrix error: Build explicit 4x4 matrix [R|t]
    full_matrices = torch.eye(4, device=device).repeat(n_views, 1, 1)
    full_matrices[:, :3, :3] = r_matrices
    full_matrices[:, :3, 3] = torch.tensor([sod, 0, 0], device=device)
    
    # Use explicit matrix initialization
    poses = RigidTransform(full_matrices)

    with torch.no_grad():
        # Differentiable Radiative Rasterization (DRR)
        line_integrals = drr(poses)
        # Apply Beer-Lambert for medical-grade contrast
        img = torch.exp(-line_integrals * 0.02) 
        
    return img.detach().cpu().numpy()[:, 0]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dicom_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="block0_out")
    parser.add_argument("--num_views", type=int, default=4)
    parser.add_argument("--dose", type=float, default=2000.0)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--delx", type=float, default=1.5)
    parser.add_argument("--sdd", type=float, default=1000.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vol_hu, spacing = load_dicom_series(Path(args.dicom_dir))
    vol_norm = window_and_normalize(vol_hu)
    subject = build_subject_from_volume(vol_norm, spacing)
    
    angles = torch.linspace(0, 2*np.pi, args.num_views)
    clean_np = render_drr(subject, angles, args.height, args.width, args.delx, args.sdd, device)

    # Poisson noise (Low-dose simulation)
    noisy = np.random.poisson(clean_np * args.dose) / args.dose

    for i in range(noisy.shape[0]):
        img = auto_exposure(noisy[i])
        imageio.imwrite(out_dir / f"drr_{i:03d}.png", (img * 255).astype(np.uint8))
    
    print(f"BLOCK 0 COMPLETE: Saved to {out_dir}")

if __name__ == "__main__":
    main()