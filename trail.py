# import argparse
# from pathlib import Path

# import imageio.v2 as imageio
# import nibabel as nib
# import numpy as np
# import pydicom
# import torch
# import torchio as tio
# from diffdrr.drr import DRR
# from diffdrr.pose import convert


# def load_nifti(nifti_path: Path):
#     nii = nib.load(str(nifti_path))
#     volume = nii.get_fdata().astype(np.float32)
#     zooms = nii.header.get_zooms()
#     if len(zooms) < 3:
#         raise ValueError("NIfTI spacing must include 3 spatial dimensions.")
#     spacing_xyz = (float(zooms[0]), float(zooms[1]), float(zooms[2]))
#     return volume, spacing_xyz


# def load_dicom_series(dicom_dir: Path):
#     files = [p for p in dicom_dir.iterdir() if p.is_file()]
#     slices = []
#     for fp in files:
#         try:
#             ds = pydicom.dcmread(str(fp))
#             slices.append(ds)
#         except Exception:
#             continue
#     if not slices:
#         raise RuntimeError(f"No readable DICOM slices found in {dicom_dir}")

#     def sort_key(ds):
#         if "ImagePositionPatient" in ds:
#             return float(ds.ImagePositionPatient[2])
#         if "InstanceNumber" in ds:
#             return int(ds.InstanceNumber)
#         return 0

#     slices.sort(key=sort_key)
#     first = slices[0]
#     sx = float(first.PixelSpacing[1])
#     sy = float(first.PixelSpacing[0])
#     sz = float(getattr(first, "SliceThickness", 1.0))

#     volume = []
#     for ds in slices:
#         arr = ds.pixel_array.astype(np.float32)
#         slope = float(getattr(ds, "RescaleSlope", 1.0))
#         intercept = float(getattr(ds, "RescaleIntercept", 0.0))
#         hu = arr * slope + intercept
#         volume.append(hu)

#     volume = np.stack(volume, axis=0)  # [Z, Y, X]
#     volume = np.transpose(volume, (2, 1, 0))  # [X, Y, Z]
#     return volume, (sx, sy, sz)


# def window_and_normalize(volume_hu, hu_min=-1000.0, hu_max=1000.0):
#     vol = np.clip(volume_hu, hu_min, hu_max)
#     vol = (vol - hu_min) / (hu_max - hu_min)
#     return vol.astype(np.float32)


# def build_subject(volume_xyz, spacing_xyz):
#     vol_tensor = torch.from_numpy(volume_xyz).unsqueeze(0)  # [1, X, Y, Z]
#     sx, sy, sz = spacing_xyz

#     affine = np.eye(4, dtype=np.float32)
#     affine[0, 0], affine[1, 1], affine[2, 2] = sx, sy, sz
#     xdim, ydim, zdim = volume_xyz.shape
#     affine[0, 3] = -(xdim - 1) * sx / 2.0
#     affine[1, 3] = -(ydim - 1) * sy / 2.0
#     affine[2, 3] = -(zdim - 1) * sz / 2.0

#     img = tio.ScalarImage(tensor=vol_tensor, affine=affine)
#     subject = tio.Subject(volume=img, density=tio.ScalarImage(tensor=vol_tensor.clone(), affine=affine))

#     if not hasattr(subject, "mask"):
#         subject.mask = None
#     subject.reorient = torch.eye(4).unsqueeze(0)
#     return subject


# def make_sparse_poses(n_views, sdd, device, dtype=torch.float32):
#     angles = torch.linspace(0.0, 2.0 * np.pi, n_views + 1, device=device, dtype=dtype)[:-1]
#     sod = sdd / 2.0
#     # For diffdrr==0.6.0 default detector frame, source offset is along Z.
#     translation = torch.tensor([0.0, 0.0, sod], device=device, dtype=dtype).unsqueeze(0).repeat(n_views, 1)
#     rotation = torch.stack(
#         [
#             torch.zeros_like(angles),
#             angles,
#             torch.zeros_like(angles),
#         ],
#         dim=-1,
#     )
#     return convert(rotation, translation, parameterization="euler_angles", convention="XYZ", degrees=False)


# def add_poisson_noise(projections, dose_factor):
#     scaled = torch.clamp(projections, min=0.0) * dose_factor
#     return torch.poisson(scaled) / dose_factor


# def enhance_view_for_vggt(img, eps=1e-6):
#     # 1) Log transform to linear attenuation-like space
#     x = -np.log(np.clip(img, eps, 1.0))
#     # 2) Normalize to [0, 1] per image
#     x_min = float(x.min())
#     x_max = float(x.max())
#     if x_max - x_min < eps:
#         x_norm = np.zeros_like(x, dtype=np.float32)
#     else:
#         x_norm = (x - x_min) / (x_max - x_min)
#     # 3) Standardize per image
#     mean = float(x_norm.mean())
#     std = float(x_norm.std())
#     if std < eps:
#         x_std = np.zeros_like(x_norm, dtype=np.float32)
#     else:
#         x_std = (x_norm - mean) / std
#     return x_std.astype(np.float32)


# def enhance_batch_for_vggt(noisy_projections):
#     views = noisy_projections.detach().cpu().numpy()[:, 0]  # [V,H,W]
#     enhanced = [enhance_view_for_vggt(v) for v in views]
#     enhanced = np.stack(enhanced, axis=0)[:, None, :, :]  # [V,1,H,W]
#     return torch.from_numpy(enhanced)


# def line_integrals_to_intensity(line_integrals, target_at_p99=0.15, eps=1e-6):
#     # Calibrate Beer-Lambert per view so the 99th percentile maps to target intensity.
#     # This avoids near-black DRRs when attenuation scale varies across scans.
#     flat = line_integrals.flatten(start_dim=2)
#     p99 = torch.quantile(flat, 0.99, dim=2, keepdim=True).clamp_min(eps)
#     alpha = (-np.log(target_at_p99)) / p99
#     alpha = alpha.unsqueeze(-1)  # [V,1,1,1]
#     return torch.exp(-line_integrals * alpha)


# def auto_exposure(img, low=1.0, high=99.0, eps=1e-6):
#     lo = float(np.percentile(img, low))
#     hi = float(np.percentile(img, high))
#     if hi - lo < eps:
#         return np.clip(img, 0.0, 1.0)
#     return np.clip((img - lo) / (hi - lo), 0.0, 1.0)


# def standard_to_display(img, low=1.0, high=99.0, eps=1e-6):
#     lo = float(np.percentile(img, low))
#     hi = float(np.percentile(img, high))
#     if hi - lo < eps:
#         return np.zeros_like(img, dtype=np.float32)
#     return np.clip((img - lo) / (hi - lo), 0.0, 1.0)


# def save_xrays(enhanced_projections, out_dir: Path, use_auto_exposure=False):
#     out_dir.mkdir(parents=True, exist_ok=True)
#     views = enhanced_projections.detach().cpu().numpy()[:, 0]  # [V,H,W], standardized
#     for i, img in enumerate(views):
#         # Standardized tensors are not directly viewable; remap for PNG display.
#         img = standard_to_display(img)
#         if use_auto_exposure:
#             img = auto_exposure(img)
#         img = np.clip(img, 0.0, 1.0)
#         imageio.imwrite(out_dir / f"xray_view_{i:03d}.png", (img * 255.0).astype(np.uint8))
#     print(f"Saved {views.shape[0]} X-rays to: {out_dir}")


# def process_block0(
#     input_path,
#     n_views=4,
#     height=256,
#     width=256,
#     delx=1.5,
#     sdd=1020.0,
#     hu_min=-1000.0,
#     hu_max=1000.0,
#     dose_factor=20000.0,
#     target_at_p99=0.15,
#     device=None,
# ):
#     print("=== BLOCK 0: DATA PREPARATION ===")
#     input_path = Path(input_path)
#     if device is None:
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Device: {device}")

#     if input_path.is_dir():
#         print(f"Loading DICOM series from: {input_path}")
#         volume_hu_xyz, spacing_xyz = load_dicom_series(input_path)
#     else:
#         print(f"Loading NIfTI from: {input_path}")
#         volume_hu_xyz, spacing_xyz = load_nifti(input_path)

#     print(f"Volume shape [X,Y,Z]: {volume_hu_xyz.shape}")
#     print(f"Spacing [sx,sy,sz] mm: {spacing_xyz}")

#     volume_norm_xyz = window_and_normalize(volume_hu_xyz, hu_min=hu_min, hu_max=hu_max)
#     subject = build_subject(volume_norm_xyz, spacing_xyz)

#     drr = DRR(subject, sdd=sdd, height=height, width=width, delx=delx).to(device)
#     poses = make_sparse_poses(n_views=n_views, sdd=sdd, device=device)

#     with torch.no_grad():
#         line_integrals = drr(poses)  # [V,1,H,W]
#         clean_projections = line_integrals_to_intensity(line_integrals, target_at_p99=target_at_p99)
#         noisy_projections = add_poisson_noise(clean_projections, dose_factor=dose_factor)
#     enhanced_projections = enhance_batch_for_vggt(noisy_projections).to(noisy_projections.dtype)

#     # [V,1,H,W] -> [V,3,H,W] -> [1,V,3,H,W]
#     final_input = enhanced_projections.repeat(1, 3, 1, 1).unsqueeze(0)

#     print(f"Final Input Tensor Shape: {tuple(final_input.shape)}")
#     print(f"Min Value: {float(final_input.min()):.6f}")
#     print(f"Max Value: {float(final_input.max()):.6f}")
#     return final_input


# def main():
#     parser = argparse.ArgumentParser(description="Block 0: Input + Virtual Scanner with DiffDRR")
#     parser.add_argument("--input_path", type=str, required=True, help="Path to .nii/.nii.gz or DICOM directory")
#     parser.add_argument("--n_views", type=int, default=4)
#     parser.add_argument("--height", type=int, default=256)
#     parser.add_argument("--width", type=int, default=256)
#     parser.add_argument("--delx", type=float, default=1.5)
#     parser.add_argument("--sdd", type=float, default=1020.0)
#     parser.add_argument("--hu_min", type=float, default=-1000.0)
#     parser.add_argument("--hu_max", type=float, default=1000.0)
#     parser.add_argument("--dose_factor", type=float, default=20000.0)
#     parser.add_argument(
#         "--target_at_p99",
#         type=float,
#         default=0.15,
#         help="Beer-Lambert calibration target: intensity assigned to per-view 99th percentile attenuation",
#     )
#     parser.add_argument("--auto_exposure", action="store_true", help="Apply per-view percentile auto-exposure before saving PNGs")
#     parser.add_argument("--save_xrays_dir", type=str, default="images", help="Directory to save DRR X-ray PNGs")
#     parser.add_argument("--save_pt", type=str, default=None, help="Optional output .pt path")
#     args = parser.parse_args()

#     final_input = process_block0(
#         input_path=args.input_path,
#         n_views=args.n_views,
#         height=args.height,
#         width=args.width,
#         delx=args.delx,
#         sdd=args.sdd,
#         hu_min=args.hu_min,
#         hu_max=args.hu_max,
#         dose_factor=args.dose_factor,
#         target_at_p99=args.target_at_p99,
#     )

#     save_xrays(
#         final_input[0, :, :1, :, :],
#         Path(args.save_xrays_dir),
#         use_auto_exposure=args.auto_exposure,
#     )

#     if args.save_pt:
#         out_path = Path(args.save_pt)
#         out_path.parent.mkdir(parents=True, exist_ok=True)
#         torch.save(final_input.cpu(), out_path)
#         print(f"Saved tensor to: {out_path}")


# if __name__ == "__main__":
#     main()
import torch
d = torch.load('data/phase2_output/phase2_features.pt', map_location='cpu')

# See what's inside vggt_features
print(type(d['vggt_features']))
print(d['vggt_features'].keys())

# Then check each value's shape
for k, v in d['vggt_features'].items():
    if isinstance(v, torch.Tensor):
        print(f"  {k}: {tuple(v.shape)}")
    else:
        print(f"  {k}: {type(v)} = {v}")