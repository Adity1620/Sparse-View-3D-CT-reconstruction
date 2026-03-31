import argparse
import csv
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from diffdrr.drr import DRR

from block0_input import (
    build_subject,
    line_integrals_to_intensity,
    load_dicom_series,
    load_nifti,
    make_sparse_poses,
    window_and_normalize,
)

try:
    from skimage.metrics import structural_similarity as ssim_fn
except Exception:
    ssim_fn = None


def load_gray(path: Path):
    img = imageio.imread(path)
    if img.ndim == 3:
        img = img[..., :3].mean(axis=2)
    img = img.astype(np.float32)
    if img.max() > 1.0:
        img /= 255.0
    return np.clip(img, 0.0, 1.0)


def match_size(pred, gt):
    if pred.shape == gt.shape:
        return pred, gt
    h = min(pred.shape[0], gt.shape[0])
    w = min(pred.shape[1], gt.shape[1])
    return pred[:h, :w], gt[:h, :w]


def metrics(pred, gt):
    diff = pred - gt
    mse = float(np.mean(diff * diff))
    mae = float(np.mean(np.abs(diff)))
    psnr = float(20.0 * np.log10(1.0 / np.sqrt(max(mse, 1e-12))))
    out = {"mse": mse, "mae": mae, "psnr": psnr}
    if ssim_fn is not None:
        out["ssim"] = float(ssim_fn(gt, pred, data_range=1.0))
    else:
        out["ssim"] = None
    return out


def make_panel(pred, gt):
    err = np.abs(pred - gt)
    pred_u8 = (np.clip(pred, 0.0, 1.0) * 255.0).astype(np.uint8)
    gt_u8 = (np.clip(gt, 0.0, 1.0) * 255.0).astype(np.uint8)
    err_u8 = (np.clip(err / max(err.max(), 1e-6), 0.0, 1.0) * 255.0).astype(np.uint8)
    return np.concatenate([pred_u8, gt_u8, err_u8], axis=1)


def generate_clean_gt_views(
    input_path,
    n_views=4,
    height=256,
    width=256,
    delx=1.5,
    sdd=1020.0,
    hu_min=-1000.0,
    hu_max=1000.0,
    target_at_p99=0.15,
    device=None,
):
    input_path = Path(input_path)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if input_path.is_dir():
        volume_hu_xyz, spacing_xyz = load_dicom_series(input_path)
    else:
        volume_hu_xyz, spacing_xyz = load_nifti(input_path)

    volume_norm_xyz = window_and_normalize(volume_hu_xyz, hu_min=hu_min, hu_max=hu_max)
    subject = build_subject(volume_norm_xyz, spacing_xyz)

    drr = DRR(subject, sdd=sdd, height=height, width=width, delx=delx).to(device)
    poses = make_sparse_poses(n_views=n_views, sdd=sdd, device=device)

    with torch.no_grad():
        line_integrals = drr(poses)  # [V,1,H,W]
        clean = line_integrals_to_intensity(line_integrals, target_at_p99=target_at_p99)
    return clean.detach().cpu().numpy()[:, 0]  # [V,H,W]


def main():
    parser = argparse.ArgumentParser(description="Compare generated noisy DRRs with clean GT DRRs generated from DICOM/NIfTI")
    parser.add_argument("--pred_dir", type=str, required=True, help="Directory with generated noisy PNGs (e.g., xray_view_000.png)")
    parser.add_argument("--input_path", type=str, required=True, help="Path to source DICOM folder or NIfTI used to generate DRRs")
    parser.add_argument("--pattern", type=str, default="xray_view_*.png", help="Glob for predicted images")
    parser.add_argument("--n_views", type=int, default=4)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--delx", type=float, default=1.5)
    parser.add_argument("--sdd", type=float, default=1020.0)
    parser.add_argument("--hu_min", type=float, default=-1000.0)
    parser.add_argument("--hu_max", type=float, default=1000.0)
    parser.add_argument("--target_at_p99", type=float, default=0.15)
    parser.add_argument("--out_dir", type=str, default="block0_compare_out", help="Directory for metrics and panels")
    parser.add_argument("--allow_invert", action="store_true", help="Use inverted prediction if it matches GT better")
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    out_dir = Path(args.out_dir)
    panel_dir = out_dir / "panels"
    gt_dir = out_dir / "gt_generated"
    out_dir.mkdir(parents=True, exist_ok=True)
    panel_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    pred_files = sorted(pred_dir.glob(args.pattern))
    if not pred_files:
        raise RuntimeError("No predicted images found. Check --pred_dir and --pattern.")

    gt_views = generate_clean_gt_views(
        input_path=args.input_path,
        n_views=args.n_views,
        height=args.height,
        width=args.width,
        delx=args.delx,
        sdd=args.sdd,
        hu_min=args.hu_min,
        hu_max=args.hu_max,
        target_at_p99=args.target_at_p99,
    )
    for i, gt in enumerate(gt_views):
        gt_u8 = (np.clip(gt, 0.0, 1.0) * 255.0).astype(np.uint8)
        imageio.imwrite(gt_dir / f"gt_view_{i:03d}.png", gt_u8)

    n = min(len(pred_files), len(gt_views))
    rows = []
    for i in range(n):
        pred_path = pred_files[i]
        pred = load_gray(pred_path)
        gt = gt_views[i]
        pred, gt = match_size(pred, gt)

        chosen = "normal"
        m = metrics(pred, gt)
        if args.allow_invert:
            m_inv = metrics(1.0 - pred, gt)
            if m_inv["mse"] < m["mse"]:
                pred = 1.0 - pred
                m = m_inv
                chosen = "inverted"

        panel = make_panel(pred, gt)
        panel_path = panel_dir / f"compare_{i:03d}.png"
        imageio.imwrite(panel_path, panel)

        rows.append(
            {
                "index": i,
                "pred_file": pred_path.name,
                "gt_file": f"gt_view_{i:03d}.png",
                "mode": chosen,
                "mse": m["mse"],
                "mae": m["mae"],
                "psnr": m["psnr"],
                "ssim": m["ssim"],
                "panel": str(panel_path),
            }
        )

    with open(out_dir / "metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["index", "pred_file", "gt_file", "mode", "mse", "mae", "psnr", "ssim", "panel"],
        )
        writer.writeheader()
        writer.writerows(rows)

    valid_ssim = [r["ssim"] for r in rows if r["ssim"] is not None]
    summary = {
        "num_pairs": n,
        "avg_mse": float(np.mean([r["mse"] for r in rows])),
        "avg_mae": float(np.mean([r["mae"] for r in rows])),
        "avg_psnr": float(np.mean([r["psnr"] for r in rows])),
        "avg_ssim": float(np.mean(valid_ssim)) if valid_ssim else None,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Compared {n} pairs")
    print(f"Summary saved: {out_dir / 'summary.json'}")
    print(f"Per-view metrics: {out_dir / 'metrics.csv'}")
    print(f"Generated clean GT views: {gt_dir}")
    print(f"Panels (pred | gt | abs error): {panel_dir}")


if __name__ == "__main__":
    main()
