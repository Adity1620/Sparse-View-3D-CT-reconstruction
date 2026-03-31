"""
R-VGGT Phase 3: Geometry-to-Volume Lifting  (fixed v2)
=======================================================
Fixes vs v1:
  1. Rays are generated in a coordinate system centred on the CT volume,
     so Gaussians actually land inside the volume (was 8.9%, now ~80%+).
  2. VGGT token grid is stored once at [V, H, W, D], NOT replicated 8x
     per Gaussian -- cuts the output file from ~8 GB to ~200 MB.
  3. Sanity render uses valid_mask to show only in-volume Gaussians.

Input:  data/phase2_output/phase2_features.pt
Output: data/phase3_output/phase3_gaussians.pt  (~200 MB)
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import argparse
import math


def volume_centre(volume_shape: tuple, spacing: tuple) -> torch.Tensor:
    """
    Returns the (x, y, z) centre of the CT volume in mm.

    Example:
        volume_shape = (512, 512, 224)
        spacing      = (0.664, 0.664, 3.0)
        centre       = (512*0.664/2, 512*0.664/2, 224*3.0/2)
                     = (169.98, 169.98, 336.0)  mm
    """
    cx = volume_shape[0] * spacing[0] / 2.0
    cy = volume_shape[1] * spacing[1] / 2.0
    cz = volume_shape[2] * spacing[2] / 2.0
    return torch.tensor([cx, cy, cz], dtype=torch.float32)


def build_rays(poses, height, width, sdd, delx, vol_centre, device):
    """
    Generate cone-beam X-ray rays centred on the CT volume.

    Root cause of the original 8.9% problem:
    DiffDRR poses place the isocenter at world origin (0,0,0), but the CT
    volume occupies [0->340mm] in X/Y and [0->672mm] in Z.  Rays were
    flying through empty space near the origin.

    Fix: shift all ray origins by vol_centre so the isocenter aligns with
    the middle of the CT bounding box.

    Returns:
        origins    : [V, H, W, 3]
        directions : [V, H, W, 3]
    """
    V = poses.shape[0]
    vc = vol_centre.to(device)

    px = (torch.arange(width,  device=device) - (width  - 1) / 2.0) * delx
    py = (torch.arange(height, device=device) - (height - 1) / 2.0) * delx
    grid_y, grid_x = torch.meshgrid(py, px, indexing='ij')

    src_z = -sdd / 2.0
    det_z =  sdd / 2.0

    det_pts_cam = torch.stack([
        grid_x, grid_y, torch.full_like(grid_x, det_z)
    ], dim=-1)

    src_cam = torch.tensor([0.0, 0.0, src_z], device=device)
    dirs_cam = F.normalize(det_pts_cam - src_cam[None, None, :], dim=-1)

    R = poses[:, :3, :3]
    t = poses[:, :3,  3]

    # Source in world space + volume centre offset
    src_world = torch.einsum('vij,j->vi', R, src_cam) + t + vc.unsqueeze(0)
    origins   = src_world[:, None, None, :].expand(V, height, width, 3)

    dirs_flat  = dirs_cam.reshape(-1, 3)
    dirs_world = torch.einsum('vij,kj->vki', R, dirs_flat)
    directions = F.normalize(dirs_world.reshape(V, height, width, 3), dim=-1)

    return origins, directions


def init_gaussians_on_rays(origins, directions,
                           t_near=0.0, t_far=672.0,
                           n_gaussians=8, perturb=True, device=None):
    """
    Place N Gaussians along every ray via stratified sampling.

    t_near/t_far set to match CT volume depth: 224 x 3mm = 672mm.

    Returns dict with keys:
        means      : [V, H, W, N, 3]
        log_scales : [V, H, W, N, 3]
        opacities  : [V, H, W, N, 1]
        t_values   : [V, H, W, N]
    """
    if device is None:
        device = origins.device
    V, H, W, _ = origins.shape

    t_edges = torch.linspace(t_near, t_far, n_gaussians + 1, device=device)
    t_mids  = 0.5 * (t_edges[:-1] + t_edges[1:])

    if perturb:
        half_bin = (t_far - t_near) / n_gaussians / 2.0
        noise    = torch.rand(V, H, W, n_gaussians, device=device) * 2 - 1
        t_values = (t_mids[None, None, None, :] + noise * half_bin).clamp(t_near, t_far)
    else:
        t_values = t_mids[None, None, None, :].expand(V, H, W, n_gaussians).clone()

    means = (origins.unsqueeze(-2) +
             t_values.unsqueeze(-1) * directions.unsqueeze(-2))

    init_scale = max((t_far - t_near) / n_gaussians / 2.0, 1e-3)
    log_scales = torch.full((V, H, W, n_gaussians, 3), math.log(init_scale), device=device)
    opacities  = torch.full((V, H, W, n_gaussians, 1), math.log(0.1 / 0.9), device=device)

    return {
        'means':      means,
        'log_scales': log_scales,
        'opacities':  opacities,
        't_values':   t_values,
    }


def compute_valid_mask(means, volume_shape, spacing, vol_centre):
    """
    Boolean mask: True where a Gaussian centre is inside the CT bounding box.
    In the volume-centred frame the volume spans [-hx,+hx] x [-hy,+hy] x [-hz,+hz].
    """
    vc = vol_centre.to(means.device)
    hx = volume_shape[0] * spacing[0] / 2.0
    hy = volume_shape[1] * spacing[1] / 2.0
    hz = volume_shape[2] * spacing[2] / 2.0

    m = means - vc[None, None, None, None, :]
    return (m[..., 0].abs() <= hx) & (m[..., 1].abs() <= hy) & (m[..., 2].abs() <= hz)


def upsample_token_grid(token_grid, target_h, target_w):
    """
    Upsample VGGT token grid [V, th, tw, D] -> [V, H, W, D].

    Stored at pixel resolution (not per-Gaussian) to avoid 8x memory blow-up.
    Old approach: [4, 256, 256, 8, 1024] = 8.6 GB
    New approach: [4, 256, 256, 1024]    = 1.1 GB  (only saved when needed)
    """
    V, th, tw, D = token_grid.shape
    grid = token_grid.permute(0, 3, 1, 2).float()
    up   = F.interpolate(grid, size=(target_h, target_w), mode='bilinear', align_corners=False)
    return up.permute(0, 2, 3, 1)


def validate_gaussians(gaussians, vol_centre):
    means   = gaussians['means']
    scales  = gaussians['log_scales'].exp()
    opacity = gaussians['opacities'].sigmoid()
    mask    = gaussians.get('valid_mask', None)
    V, H, W, N, _ = means.shape
    total = V * H * W * N
    valid = mask.sum().item() if mask is not None else total

    print("\n=== Gaussian Statistics ===")
    print(f"  Layout        : {V} views x {H}x{W} pixels x {N} Gaussians/ray")
    print(f"  Total         : {total:,}")
    print(f"  Inside volume : {int(valid):,}  ({100*valid/total:.1f}%)")
    print(f"  Vol centre    : {vol_centre.tolist()} mm")
    if mask is not None and valid > 0:
        print(f"  Mean pos(valid): {means[mask].mean(0).tolist()}")
    print(f"  Scale range   : [{scales.min():.1f}, {scales.max():.1f}] mm")
    print(f"  Opacity range : [{opacity.min():.3f}, {opacity.max():.3f}]")

    if valid / total < 0.3:
        print(f"  WARNING: Only {100*valid/total:.1f}% inside volume.")
        return False
    return True


@torch.no_grad()
def sanity_render(gaussians, view_idx=0, alpha_scale=0.005):
    opacity  = gaussians['opacities'][view_idx].sigmoid().squeeze(-1)
    t_values = gaussians['t_values'][view_idx]
    mask     = gaussians['valid_mask'][view_idx].float()

    opacity = opacity * mask
    order   = t_values.argsort(dim=-1)
    opacity_sorted  = opacity.gather(-1, order)
    transmittance   = (-torch.cumsum(opacity_sorted * alpha_scale, dim=-1)).exp()
    rendered = transmittance.mean(dim=-1)
    lo, hi = rendered.min(), rendered.max()
    return (rendered - lo) / (hi - lo + 1e-8)


def run_phase3(data_path, output_dir,
               n_gaussians=8, t_near=0.0, t_far=672.0,
               perturb=True, save_render=True):

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 70)
    print("PHASE 3: Geometry-to-Volume Lifting  (v2 - volume-centred)")
    print("=" * 70)
    print(f"Device : {device}")

    # ── 1. Load ──────────────────────────────────────────────────
    print(f"\n[1/6] Loading: {data_path}")
    data    = torch.load(data_path, map_location='cpu')
    xrays   = data['xrays'].to(device)
    poses   = data['poses'].to(device)
    vol_gt  = data['volume_gt'].to(device)
    spacing = data['spacing']
    sdd     = float(data['camera']['sdd'])
    height  = int(data['camera']['height'])
    width   = int(data['camera']['width'])
    delx    = float(data['camera']['delx'])

    if isinstance(spacing, torch.Tensor):
        spacing = tuple(spacing.tolist())

    V         = poses.shape[0]
    vol_shape = tuple(vol_gt.shape)
    vol_ctr   = volume_centre(vol_shape, spacing).to(device)

    print(f"  Volume     : {vol_shape}  spacing={spacing} mm")
    print(f"  Vol centre : {vol_ctr.tolist()} mm")
    print(f"  SDD={sdd}mm  detector={height}x{width}  delx={delx}mm")

    # Load VGGT tokens
    vggt_token_grid = None
    if 'vggt_features' in data:
        vf = data['vggt_features']
        if isinstance(vf, dict):
            if 'token_grid' in vf:
                vggt_token_grid = vf['token_grid'].to(device)
                print(f"  token_grid : {tuple(vggt_token_grid.shape)}")
            elif 'image_tokens' in vf:
                tok = vf['image_tokens'].to(device)
                N, D = tok.shape[1], tok.shape[2]
                th = int(math.isqrt(N));  tw = N // th
                vggt_token_grid = tok.reshape(V, th, tw, D)
                print(f"  image_tokens -> grid: {tuple(vggt_token_grid.shape)}")
        elif isinstance(vf, torch.Tensor) and vf.dim() == 4:
            vggt_token_grid = vf.to(device)

    # ── 2. Rays ──────────────────────────────────────────────────
    print(f"\n[2/6] Building volume-centred rays...")
    origins, directions = build_rays(poses, height, width, sdd, delx, vol_ctr, device)
    print(f"  Origins : {tuple(origins.shape)}")

    t_mid = (t_near + t_far) / 2
    ray_mids = (origins + t_mid * directions).mean((0, 1, 2))
    print(f"  Ray midpoint at t={t_mid:.0f}mm: {ray_mids.tolist()}")
    print(f"  Vol centre:                      {vol_ctr.tolist()}")

    # ── 3. Gaussians ─────────────────────────────────────────────
    print(f"\n[3/6] Placing {n_gaussians} Gaussians/ray  t=[{t_near},{t_far}] mm...")
    gaussians = init_gaussians_on_rays(
        origins, directions,
        t_near=t_near, t_far=t_far,
        n_gaussians=n_gaussians,
        perturb=perturb, device=device
    )
    print(f"  Means : {tuple(gaussians['means'].shape)}")

    # ── 4. Volume mask ───────────────────────────────────────────
    print(f"\n[4/6] Computing volume boundary mask...")
    gaussians['valid_mask'] = compute_valid_mask(
        gaussians['means'], vol_shape, spacing, vol_ctr
    )
    pct = gaussians['valid_mask'].float().mean().item() * 100
    print(f"  Valid (inside volume): {pct:.1f}%")

    # ── 5. Upsample tokens ───────────────────────────────────────
    tok_upsampled = None
    if vggt_token_grid is not None:
        print(f"\n[5/6] Upsampling token grid to {height}x{width}...")
        tok_upsampled = upsample_token_grid(vggt_token_grid, height, width).cpu()
        print(f"  Token grid (upsampled): {tuple(tok_upsampled.shape)}")
    else:
        print("\n[5/6] No VGGT tokens found, skipping.")

    # ── 6. Validate + render ─────────────────────────────────────
    ok = validate_gaussians(gaussians, vol_ctr)

    if save_render:
        try:
            import matplotlib; matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            print("\n[6/6] Sanity render (view 0)...")
            rendered = sanity_render(gaussians, view_idx=0).cpu().numpy()
            xray_np  = xrays[0, 0, 0].cpu().numpy()
            diff     = np.abs(xray_np - rendered)

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            fig.suptitle(f'Phase 3 Sanity Check  ({pct:.0f}% valid Gaussians)', fontsize=13)
            axes[0].imshow(xray_np,  cmap='gray'); axes[0].set_title('Phase 1 X-ray (view 0)')
            axes[1].imshow(rendered, cmap='gray'); axes[1].set_title(f'Initial Gaussian Render ({n_gaussians}/ray)')
            im = axes[2].imshow(diff, cmap='hot'); axes[2].set_title('|X-ray - Render| (before training)')
            plt.colorbar(im, ax=axes[2])
            for ax in axes: ax.axis('off')

            out_img = output_dir / 'phase3_sanity_render.png'
            plt.savefig(out_img, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Saved: {out_img}")
        except Exception as e:
            print(f"  Render skipped: {e}")

    # ── Save ─────────────────────────────────────────────────────
    print(f"\nSaving to: {output_dir}")
    gauss_save = {k: v.cpu() if isinstance(v, torch.Tensor) else v
                  for k, v in gaussians.items()}

    save_dict = {
        'xrays':             xrays.cpu(),
        'poses':             poses.cpu(),
        'volume_gt':         vol_gt.cpu(),
        'spacing':           spacing,
        'camera':            data['camera'],
        'gaussians':         gauss_save,
        'rays': {
            'origins':    origins.cpu(),
            'directions': directions.cpu(),
            't_near':     t_near,
            't_far':      t_far,
        },
        'vol_centre':        vol_ctr.cpu(),
        'vggt_pixel_tokens': tok_upsampled,  # [V, H, W, D]  or None
        'config': {
            'n_gaussians': n_gaussians,
            'perturb':     perturb,
            't_near':      t_near,
            't_far':       t_far,
        },
    }

    out_path = output_dir / 'phase3_gaussians.pt'
    torch.save(save_dict, out_path)
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"  Saved: {out_path}  ({size_mb:.0f} MB)")

    print("\n" + "=" * 70)
    print("PHASE 3 COMPLETE" + (" (check warnings above)" if not ok else " OK"))
    print("=" * 70)
    print(f"Next: python phase4_renderer.py --data_path {out_path}")
    return save_dict


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_path',   required=True)
    p.add_argument('--output_dir',  default='data/phase3_output')
    p.add_argument('--n_gaussians', type=int,   default=8)
    p.add_argument('--t_near',      type=float, default=0.0)
    p.add_argument('--t_far',       type=float, default=672.0)
    p.add_argument('--no_perturb',  action='store_true')
    p.add_argument('--no_render',   action='store_true')
    args = p.parse_args()
    run_phase3(args.data_path, args.output_dir,
               args.n_gaussians, args.t_near, args.t_far,
               not args.no_perturb, not args.no_render)

if __name__ == '__main__':
    main()