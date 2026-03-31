import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import pydicom
from scipy.ndimage import rotate
from tqdm import tqdm


def load_dicom_series(dicom_dir: Path):
    files = sorted([p for p in dicom_dir.iterdir() if p.is_file()])
    if not files:
        raise FileNotFoundError(f"No DICOM files found in {dicom_dir}")

    slices = []
    for fp in files:
        try:
            ds = pydicom.dcmread(str(fp), stop_before_pixels=False)
            slices.append(ds)
        except Exception:
            continue

    if not slices:
        raise RuntimeError("No readable DICOM slices found.")

    def sort_key(ds):
        if "ImagePositionPatient" in ds:
            return float(ds.ImagePositionPatient[2])
        if "InstanceNumber" in ds:
            return int(ds.InstanceNumber)
        return 0

    slices.sort(key=sort_key)
    first = slices[0]

    pixel_spacing = getattr(first, "PixelSpacing", [1.0, 1.0])
    slice_thickness = float(getattr(first, "SliceThickness", 1.0))
    spacing = (slice_thickness, float(pixel_spacing[0]), float(pixel_spacing[1]))

    volume = []
    for ds in slices:
        arr = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        hu = arr * slope + intercept
        volume.append(hu)

    volume = np.stack(volume, axis=0)
    return volume, spacing, slices


def window_and_normalize(volume, hu_min=-1000.0, hu_max=1000.0):
    volume = np.clip(volume, hu_min, hu_max)
    volume = (volume - hu_min) / (hu_max - hu_min)
    return volume.astype(np.float32)


def project_volume(volume, spacing, angle_deg, axis=0):
    # Rotate around z-axis (axial), then sum along depth axis.
    rotated = rotate(
        volume,
        angle=angle_deg,
        axes=(1, 2),
        reshape=False,
        order=1,
        mode="nearest",
    )
    depth_spacing = spacing[axis]
    proj = rotated.sum(axis=axis) * depth_spacing
    return proj


def trilinear_sample(volume, pts):
    # volume: (Z, Y, X), pts: (N, 3) in voxel index space (z, y, x)
    z = pts[:, 0]
    y = pts[:, 1]
    x = pts[:, 2]

    z0 = np.floor(z).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x0 = np.floor(x).astype(np.int32)
    z1 = z0 + 1
    y1 = y0 + 1
    x1 = x0 + 1

    inside = (
        (x >= 0)
        & (x <= volume.shape[2] - 1)
        & (y >= 0)
        & (y <= volume.shape[1] - 1)
        & (z >= 0)
        & (z <= volume.shape[0] - 1)
    )

    z0c = np.clip(z0, 0, volume.shape[0] - 1)
    y0c = np.clip(y0, 0, volume.shape[1] - 1)
    x0c = np.clip(x0, 0, volume.shape[2] - 1)
    z1c = np.clip(z1, 0, volume.shape[0] - 1)
    y1c = np.clip(y1, 0, volume.shape[1] - 1)
    x1c = np.clip(x1, 0, volume.shape[2] - 1)

    dz = z - z0
    dy = y - y0
    dx = x - x0

    c000 = volume[z0c, y0c, x0c]
    c001 = volume[z0c, y0c, x1c]
    c010 = volume[z0c, y1c, x0c]
    c011 = volume[z0c, y1c, x1c]
    c100 = volume[z1c, y0c, x0c]
    c101 = volume[z1c, y0c, x1c]
    c110 = volume[z1c, y1c, x0c]
    c111 = volume[z1c, y1c, x1c]

    c00 = c000 * (1 - dx) + c001 * dx
    c01 = c010 * (1 - dx) + c011 * dx
    c10 = c100 * (1 - dx) + c101 * dx
    c11 = c110 * (1 - dx) + c111 * dx

    c0 = c00 * (1 - dy) + c01 * dy
    c1 = c10 * (1 - dy) + c11 * dy

    c = c0 * (1 - dz) + c1 * dz
    c = np.where(inside, c, 0.0)
    return c


def project_cone_beam(volume, spacing, angle_deg, det_shape, det_spacing, src_dist, det_dist, ray_steps):
    # Simple cone-beam DRR with fixed rotation about z-axis.
    # World coordinates centered at volume center, units: mm.
    zdim, ydim, xdim = volume.shape
    sz, sy, sx = spacing
    cx = (xdim - 1) * sx / 2.0
    cy = (ydim - 1) * sy / 2.0
    cz = (zdim - 1) * sz / 2.0

    theta = math.radians(angle_deg)
    # Source and detector center in rotated frame (x', y', z)
    src = np.array([src_dist, 0.0, 0.0], dtype=np.float32)
    det_center = np.array([-det_dist, 0.0, 0.0], dtype=np.float32)

    # Rotate around z-axis
    rot = np.array(
        [
            [math.cos(theta), -math.sin(theta), 0.0],
            [math.sin(theta), math.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    src = rot @ src
    det_center = rot @ det_center

    # Detector plane basis vectors (u along y, v along z)
    u_dir = rot @ np.array([0.0, 1.0, 0.0], dtype=np.float32)
    v_dir = rot @ np.array([0.0, 0.0, 1.0], dtype=np.float32)

    H, W = det_shape
    du, dv = det_spacing
    u_coords = (np.arange(W) - (W - 1) / 2.0) * du
    v_coords = (np.arange(H) - (H - 1) / 2.0) * dv
    uu, vv = np.meshgrid(u_coords, v_coords)

    det_points = (
        det_center[None, None, :]
        + uu[..., None] * u_dir[None, None, :]
        + vv[..., None] * v_dir[None, None, :]
    )

    # Ray from source to each detector point
    rays = det_points - src[None, None, :]
    ray_len = np.linalg.norm(rays, axis=-1)
    ray_dir = rays / np.maximum(ray_len[..., None], 1e-6)

    # Sample along rays
    t_vals = np.linspace(0.0, 1.0, ray_steps, dtype=np.float32)
    # Accumulate in world space
    acc = np.zeros((H, W), dtype=np.float32)
    for t in t_vals:
        pts = src[None, None, :] + ray_dir * (ray_len[..., None] * t)
        # Convert world mm -> voxel index space (z, y, x)
        xw = pts[..., 0] + cx
        yw = pts[..., 1] + cy
        zw = pts[..., 2] + cz
        xi = xw / sx
        yi = yw / sy
        zi = zw / sz
        samp = trilinear_sample(volume, np.stack([zi.ravel(), yi.ravel(), xi.ravel()], axis=1))
        acc += samp.reshape(H, W)

    # Approximate integral by step length in mm
    acc *= (ray_len / max(ray_steps - 1, 1)).astype(np.float32)
    return acc


def beer_lambert(proj, k=1.0, normalize=False, eps=1e-6):
    if normalize:
        denom = float(np.percentile(proj, 99.5))
        proj = proj / max(denom, eps)
    return np.exp(-k * proj)


def auto_exposure(img, low=1.0, high=99.0, eps=1e-6):
    lo = float(np.percentile(img, low))
    hi = float(np.percentile(img, high))
    if hi - lo < eps:
        return np.clip(img, 0.0, 1.0)
    img = (img - lo) / (hi - lo)
    return np.clip(img, 0.0, 1.0)


def add_poisson_noise(image, i0=1e4, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    noisy = rng.poisson(i0 * image) / i0
    return noisy.astype(np.float32)


def save_png(path, image):
    import imageio.v2 as imageio

    img = np.clip(image, 0.0, 1.0)
    img = (img * 255.0).astype(np.uint8)
    imageio.imwrite(path, img)


def main():
    parser = argparse.ArgumentParser(description="Block 0: Virtual X-Ray Scanner (DRR)")
    parser.add_argument("--dicom_dir", type=str, default="full_3mm")
    parser.add_argument("--out_dir", type=str, default="block0_out")
    parser.add_argument("--num_views", type=int, default=4)
    parser.add_argument("--angles", type=float, nargs="*", default=None)
    parser.add_argument("--hu_min", type=float, default=-1000.0)
    parser.add_argument("--hu_max", type=float, default=1000.0)
    parser.add_argument("--k", type=float, default=None)
    parser.add_argument("--normalize_proj", action="store_true")
    parser.add_argument("--auto_exposure", action="store_true")
    parser.add_argument("--cone_beam", action="store_true")
    parser.add_argument("--det_h", type=int, default=0)
    parser.add_argument("--det_w", type=int, default=0)
    parser.add_argument("--det_spacing", type=float, nargs=2, default=None)
    parser.add_argument("--src_dist", type=float, default=1000.0)
    parser.add_argument("--det_dist", type=float, default=500.0)
    parser.add_argument("--ray_steps", type=int, default=256)
    parser.add_argument("--noise", action="store_true")
    parser.add_argument("--i0", type=float, default=1e4)
    parser.add_argument("--save_npy", action="store_true")
    args = parser.parse_args()

    dicom_dir = Path(args.dicom_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    volume_hu, spacing, slices = load_dicom_series(dicom_dir)
    volume = window_and_normalize(volume_hu, args.hu_min, args.hu_max)

    if args.angles is not None and len(args.angles) > 0:
        angles = args.angles
    else:
        step = 360.0 / args.num_views
        angles = [i * step for i in range(args.num_views)]

    rng = np.random.default_rng(0)
    projections = []
    for idx, ang in enumerate(tqdm(angles, desc="Generating DRRs")):
        if args.cone_beam:
            if args.det_spacing is None:
                # Detector u along Y, v along Z
                det_spacing = (spacing[1], spacing[0])
            else:
                det_spacing = (float(args.det_spacing[0]), float(args.det_spacing[1]))
            det_h = args.det_h if args.det_h > 0 else volume.shape[0]
            det_w = args.det_w if args.det_w > 0 else volume.shape[1]
            proj = project_cone_beam(
                volume,
                spacing,
                ang,
                det_shape=(det_h, det_w),
                det_spacing=det_spacing,
                src_dist=args.src_dist,
                det_dist=args.det_dist,
                ray_steps=args.ray_steps,
            )
        else:
            proj = project_volume(volume, spacing, ang)
        if args.k is None:
            # Choose k so that p99 -> exp(-2) ~ 0.135 (good contrast)
            p99 = float(np.percentile(proj, 99.0))
            k = 2.0 / max(p99, 1e-6)
        else:
            k = args.k
        img = beer_lambert(proj, k=k, normalize=args.normalize_proj)
        if args.auto_exposure:
            img = auto_exposure(img)
        if args.noise:
            img = add_poisson_noise(img, i0=args.i0, rng=rng)
        projections.append(img)
        save_png(out_dir / f"drr_{idx:03d}_angle_{int(ang):03d}.png", img)

    meta = {
        "dicom_dir": str(dicom_dir),
        "num_slices": int(volume.shape[0]),
        "volume_shape": list(volume.shape),
        "spacing": list(spacing),
        "hu_min": args.hu_min,
        "hu_max": args.hu_max,
        "angles": [float(a) for a in angles],
        "k": args.k,
        "noise": bool(args.noise),
        "i0": args.i0,
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    if args.save_npy:
        np.save(out_dir / "volume_hu.npy", volume_hu.astype(np.float32))
        np.save(out_dir / "volume_norm.npy", volume.astype(np.float32))
        np.save(out_dir / "projections.npy", np.stack(projections, axis=0))


if __name__ == "__main__":
    main()
