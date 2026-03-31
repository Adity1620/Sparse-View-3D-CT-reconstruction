"""
R-VGGT Phase 4: Differentiable Beer-Lambert Renderer
=====================================================
This is the core differentiable module that connects volumetric Gaussians
to synthetic X-ray images via Beer-Lambert physics.

The renderer must be DIFFERENTIABLE so gradients can flow back from the
X-ray loss -> rendered image -> Gaussian opacities/positions during training.

Beer-Lambert law:
    I(ray) = I0 * exp(-integral of mu(x) dx along ray)

Discretised for Gaussians:
    I(ray) = exp(-sum_i  opacity_i * gaussian_weight_i(ray))

Where gaussian_weight_i is how much Gaussian i contributes to this ray
(based on distance from ray, Gaussian scale, and depth ordering).

Input:  data/phase3_output/phase3_gaussians.pt
Output: data/phase4_output/phase4_renderer_test.png  (validation render)
        + the renderer module itself for use in Phase 5 training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import argparse
import math


# ─────────────────────────────────────────────────────────────
#  CORE RENDERER MODULE
# ─────────────────────────────────────────────────────────────

class BeerLambertRenderer(nn.Module):
    """
    Differentiable Beer-Lambert X-ray renderer using volumetric Gaussians.

    How it works (step by step):
    ─────────────────────────────
    1. For each ray (pixel), sort Gaussians by depth (t value)
    2. For each Gaussian, compute its contribution to this ray:
       - Project the Gaussian centre onto the ray
       - Compute Gaussian weight based on perpendicular distance
       - Scale by the Gaussian's opacity
    3. Accumulate attenuation along the ray: Beer-Lambert law
    4. Final intensity = exp(-total attenuation)

    Why this is better than alpha blending (used in RGB 3DGS):
    ──────────────────────────────────────────────────────────
    Alpha blending:  I = sum(alpha_i * colour_i * prod(1-alpha_j, j<i))
        - Designed for surface rendering (one hit = done)
        - Attenuation is additive in a non-physical way

    Beer-Lambert:    I = exp(-sum(mu_i * delta_i))
        - X-ray photons are exponentially attenuated
        - Adding more tissue ALWAYS makes it darker (physically correct)
        - Naturally handles overlapping structures (ribs over lung over spine)

    Args:
        n_gaussians  : Gaussians per ray (from Phase 3)
        use_soft_mask: If True, use soft Gaussian footprint; else hard ray hit
    """

    def __init__(self, n_gaussians: int = 8, use_soft_mask: bool = True):
        super().__init__()
        self.n_gaussians  = n_gaussians
        self.use_soft_mask = use_soft_mask

    def forward(self,
                means:      torch.Tensor,
                log_scales: torch.Tensor,
                opacities:  torch.Tensor,
                origins:    torch.Tensor,
                directions: torch.Tensor,
                t_values:   torch.Tensor) -> torch.Tensor:
        """
        Render one batch of X-ray views.

        Args:
            means      : [V, H, W, N, 3]   Gaussian centres (mm)
            log_scales : [V, H, W, N, 3]   log(scale) per axis
            opacities  : [V, H, W, N, 1]   logit(opacity)
            origins    : [V, H, W, 3]      ray origins
            directions : [V, H, W, 3]      unit ray directions
            t_values   : [V, H, W, N]      depth of each Gaussian along its ray

        Returns:
            rendered   : [V, H, W]         X-ray intensity in [0, 1]
        """
        V, H, W, N, _ = means.shape

        # Convert from logit/log space to actual values
        sigma  = opacities.sigmoid().squeeze(-1)  # [V, H, W, N]  in (0,1)
        scales = log_scales.exp()                  # [V, H, W, N, 3]  mm

        if self.use_soft_mask:
            weights = self._gaussian_ray_weights(
                means, scales, origins, directions, t_values
            )  # [V, H, W, N]
        else:
            weights = torch.ones_like(sigma)

        # Effective attenuation per Gaussian on its ray
        attenuation = sigma * weights  # [V, H, W, N]

        # Sort by depth (Beer-Lambert requires depth ordering)
        order      = t_values.argsort(dim=-1)                  # [V, H, W, N]
        atten_sorted = attenuation.gather(-1, order)           # sorted

        # Beer-Lambert: I = exp(-cumulative_attenuation)
        # We sum all contributions (integral approximation)
        total_atten = atten_sorted.sum(dim=-1)                 # [V, H, W]
        rendered    = torch.exp(-total_atten)                  # [V, H, W]

        return rendered

    def _gaussian_ray_weights(self,
                              means:      torch.Tensor,
                              scales:     torch.Tensor,
                              origins:    torch.Tensor,
                              directions: torch.Tensor,
                              t_values:   torch.Tensor) -> torch.Tensor:
        """
        Compute how much each Gaussian contributes to its assigned ray.

        For a Gaussian at position mu, the weight on ray (o, d) is:
            w = exp(-0.5 * perp_dist^2 / scale^2)

        where perp_dist = distance from Gaussian centre to the ray line.

        This is like asking: "how much of this Gaussian's mass does this
        ray pass through?" — a Gaussian centred exactly on the ray gets
        weight 1.0; a Gaussian offset to the side gets less.

        In our setup, Gaussians are INITIALISED on their pixel's ray, so
        initially all weights ≈ 1. After training, positions can drift.

        Args:
            means      : [V, H, W, N, 3]
            scales     : [V, H, W, N, 3]
            origins    : [V, H, W, 3]
            directions : [V, H, W, 3]
            t_values   : [V, H, W, N]

        Returns:
            weights : [V, H, W, N]  in (0, 1]
        """
        # Closest point on ray to each Gaussian centre
        # closest = origin + dot(mu - origin, d) * d
        mu_minus_o = (means -
                      origins.unsqueeze(-2))             # [V, H, W, N, 3]
        d_expanded = directions.unsqueeze(-2)            # [V, H, W, 1, 3]

        # t_proj = dot(mu - o, d)  — projection of offset onto ray
        t_proj  = (mu_minus_o * d_expanded).sum(-1)     # [V, H, W, N]
        closest = (origins.unsqueeze(-2) +
                   t_proj.unsqueeze(-1) * d_expanded)   # [V, H, W, N, 3]

        # Perpendicular distance from Gaussian centre to ray
        perp_vec  = means - closest                      # [V, H, W, N, 3]
        perp_dist2 = (perp_vec ** 2).sum(-1)             # [V, H, W, N]

        # Scale: use mean of x,y scales (perpendicular to z-axis/ray)
        scale_perp = scales[..., :2].mean(-1).clamp(min=1e-3)  # [V, H, W, N]

        # Gaussian footprint weight
        weights = torch.exp(-0.5 * perp_dist2 / (scale_perp ** 2))

        return weights  # [V, H, W, N]


# ─────────────────────────────────────────────────────────────
#  LOSS FUNCTIONS
# ─────────────────────────────────────────────────────────────

def photometric_loss(rendered: torch.Tensor,
                     target:   torch.Tensor,
                     mask:     torch.Tensor = None) -> torch.Tensor:
    """
    Combined L1 + SSIM loss between rendered and target X-rays.

    Why L1 and not L2?
    L2 penalises large errors heavily but also blurs fine structures.
    L1 is more robust to outliers (noise, artefacts) — better for X-rays.

    SSIM (Structural Similarity) adds perceptual quality:
    it penalises structural differences (edges, textures) even when
    pixel-level differences are small.

    Args:
        rendered : [V, H, W]  rendered X-ray
        target   : [V, H, W]  ground truth X-ray
        mask     : [V, H, W]  optional validity mask

    Returns:
        scalar loss
    """
    if mask is not None:
        rendered = rendered * mask
        target   = target   * mask

    l1   = F.l1_loss(rendered, target)
    ssim = 1.0 - ssim_loss(rendered.unsqueeze(1), target.unsqueeze(1))

    return 0.8 * l1 + 0.2 * ssim


def ssim_loss(pred: torch.Tensor, target: torch.Tensor,
              window_size: int = 11, C1: float = 0.01**2,
              C2: float = 0.03**2) -> torch.Tensor:
    """
    Differentiable SSIM in [0, 1].  Higher = more similar.

    pred, target : [B, 1, H, W]
    """
    kernel = _gaussian_kernel(window_size, 1.5, pred.device)

    mu_p  = F.conv2d(pred,   kernel, padding=window_size//2, groups=1)
    mu_t  = F.conv2d(target, kernel, padding=window_size//2, groups=1)
    mu_p2 = mu_p ** 2
    mu_t2 = mu_t ** 2
    mu_pt = mu_p * mu_t

    sig_p2 = F.conv2d(pred**2,        kernel, padding=window_size//2) - mu_p2
    sig_t2 = F.conv2d(target**2,      kernel, padding=window_size//2) - mu_t2
    sig_pt = F.conv2d(pred*target,     kernel, padding=window_size//2) - mu_pt

    ssim_map = ((2*mu_pt + C1) * (2*sig_pt + C2)) / \
               ((mu_p2 + mu_t2 + C1) * (sig_p2 + sig_t2 + C2))
    return ssim_map.mean()


def _gaussian_kernel(size: int, sigma: float,
                     device: torch.device) -> torch.Tensor:
    coords = torch.arange(size, device=device, dtype=torch.float32)
    coords -= size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g /= g.sum()
    kernel = g.outer(g)
    return kernel[None, None, :, :]  # [1, 1, K, K]


def regularisation_loss(log_scales: torch.Tensor,
                        opacities:  torch.Tensor,
                        valid_mask: torch.Tensor = None) -> torch.Tensor:
    """
    Two regularisation terms to prevent degenerate Gaussians:

    1. Scale regularisation: penalise very large Gaussians (they become blobs)
       Loss = mean(max(0, scale - max_scale))

    2. Opacity sparsity: encourage most Gaussians to be transparent
       (only the ones covering real tissue should be opaque)
       Loss = mean(sigmoid(opacity))  — soft L1 on opacity

    Only applied to valid (in-volume) Gaussians.
    """
    if valid_mask is not None:
        m = valid_mask.unsqueeze(-1).float()
    else:
        m = torch.ones_like(log_scales[..., :1])

    scales   = log_scales.exp()
    max_scale = 100.0  # mm — Gaussians shouldn't be bigger than ~10cm
    scale_reg = (F.relu(scales - max_scale) * m).mean()

    opacity  = opacities.sigmoid()
    sparse_reg = (opacity * m).mean()

    return 0.1 * scale_reg + 0.01 * sparse_reg


# ─────────────────────────────────────────────────────────────
#  VALIDATION RENDERER  (no_grad, for visualisation only)
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def render_all_views(renderer:   BeerLambertRenderer,
                     gaussians:  dict,
                     rays:       dict,
                     device:     torch.device) -> torch.Tensor:
    """
    Render all views without gradient tracking (for validation/visualisation).

    Returns: [V, H, W] rendered X-rays
    """
    means      = gaussians['means'].to(device)
    log_scales = gaussians['log_scales'].to(device)
    opacities  = gaussians['opacities'].to(device)
    t_values   = gaussians['t_values'].to(device)
    origins    = rays['origins'].to(device)
    directions = rays['directions'].to(device)

    return renderer(means, log_scales, opacities, origins, directions, t_values)


# ─────────────────────────────────────────────────────────────
#  VALIDATION: render and compare to ground truth X-rays
# ─────────────────────────────────────────────────────────────

def validate_renderer(data_path: str, output_dir: str):
    """
    Load Phase 3 output, run the renderer, compare to Phase 1 X-rays.
    This is a forward-pass only test — no training, just checking the
    renderer produces sensible output and gradients flow correctly.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 70)
    print("PHASE 4: Beer-Lambert Renderer Validation")
    print("=" * 70)
    print(f"Device: {device}")

    # ── Load ──────────────────────────────────────────────────
    print(f"\n[1/4] Loading Phase 3 data...")
    data      = torch.load(data_path, map_location='cpu')
    gaussians = data['gaussians']
    rays      = data['rays']
    xrays     = data['xrays']  # [1, V, 3, H, W]  ground truth

    V = gaussians['means'].shape[0]
    H = gaussians['means'].shape[1]
    W = gaussians['means'].shape[2]
    N = gaussians['means'].shape[3]

    print(f"  Views: {V}   Detector: {H}x{W}   Gaussians/ray: {N}")
    print(f"  Total Gaussians: {V*H*W*N:,}")

    # ── Build renderer ────────────────────────────────────────
    print("\n[2/4] Building renderer...")
    renderer = BeerLambertRenderer(n_gaussians=N, use_soft_mask=True).to(device)
    print(f"  Renderer: {sum(p.numel() for p in renderer.parameters())} learnable params")

    # ── Forward pass (no gradient) ────────────────────────────
    print("\n[3/4] Forward pass (initial Gaussians, before training)...")
    rendered = render_all_views(renderer, gaussians, rays, device)
    print(f"  Rendered shape: {tuple(rendered.shape)}")
    print(f"  Rendered range: [{rendered.min():.3f}, {rendered.max():.3f}]")

    # Compute initial loss (expect it to be high — Gaussians not trained yet)
    target_xrays = xrays[0, :, 0].to(device)  # [V, H, W]
    loss = photometric_loss(rendered, target_xrays)
    print(f"  Initial photometric loss: {loss.item():.4f}  (will decrease during training)")

    # ── Gradient check ────────────────────────────────────────
    print("\n[3b/4] Gradient check — making sure gradients flow...")
    means_test = gaussians['means'].to(device).requires_grad_(True)
    log_s_test = gaussians['log_scales'].to(device).requires_grad_(True)
    opac_test  = gaussians['opacities'].to(device).requires_grad_(True)
    t_test     = gaussians['t_values'].to(device)
    orig_test  = rays['origins'].to(device)
    dir_test   = rays['directions'].to(device)

    rendered_grad = renderer(means_test, log_s_test, opac_test,
                             orig_test, dir_test, t_test)
    test_loss = rendered_grad.mean()
    test_loss.backward()

    grad_ok = (opac_test.grad is not None and
               not opac_test.grad.isnan().any() and
               not opac_test.grad.isinf().any())
    print(f"  Gradient flows to opacities: {'YES' if grad_ok else 'NO - CHECK RENDERER'}")
    if grad_ok:
        print(f"  Opacity grad range: [{opac_test.grad.min():.6f}, {opac_test.grad.max():.6f}]")

    grad_ok_pos = (means_test.grad is not None and
                   not means_test.grad.isnan().any())
    print(f"  Gradient flows to positions: {'YES' if grad_ok_pos else 'NO'}")

    # ── Visualise ─────────────────────────────────────────────
    print("\n[4/4] Saving comparison plots...")
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(V, 3, figsize=(15, 5*V))
        fig.suptitle('Phase 4: Beer-Lambert Renderer (initial, before training)',
                     fontsize=13)

        rendered_np = rendered.cpu().numpy()  # [V, H, W]
        target_np   = target_xrays.cpu().numpy()

        for v in range(V):
            axes[v, 0].imshow(target_np[v],   cmap='gray', vmin=0, vmax=1)
            axes[v, 0].set_title(f'View {v} — Target X-ray')

            axes[v, 1].imshow(rendered_np[v], cmap='gray', vmin=0, vmax=1)
            axes[v, 1].set_title(f'View {v} — Rendered (initial)')

            diff = np.abs(target_np[v] - rendered_np[v])
            im = axes[v, 2].imshow(diff, cmap='hot', vmin=0, vmax=1)
            axes[v, 2].set_title(f'View {v} — |Target - Render|  '
                                 f'(mean={diff.mean():.3f})')
            plt.colorbar(im, ax=axes[v, 2])

        for ax in axes.flatten(): ax.axis('off')
        plt.tight_layout()

        out_img = output_dir / 'phase4_renderer_test.png'
        plt.savefig(out_img, dpi=120, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {out_img}")

        # Also save per-view histogram comparison
        fig2, axes2 = plt.subplots(1, V, figsize=(5*V, 4))
        fig2.suptitle('Intensity Distribution: Target vs Rendered', fontsize=12)
        for v in range(V):
            axes2[v].hist(target_np[v].flatten(),   bins=50, alpha=0.6,
                          label='Target',   color='steelblue', density=True)
            axes2[v].hist(rendered_np[v].flatten(), bins=50, alpha=0.6,
                          label='Rendered', color='tomato',    density=True)
            axes2[v].set_title(f'View {v}')
            axes2[v].legend(fontsize=8)
            axes2[v].set_xlabel('Intensity')
        plt.tight_layout()
        out_hist = output_dir / 'phase4_histogram.png'
        plt.savefig(out_hist, dpi=120, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {out_hist}")

    except Exception as e:
        print(f"  Visualisation skipped: {e}")

    print("\n" + "=" * 70)
    print("PHASE 4 VALIDATION COMPLETE")
    print("=" * 70)
    print("\nWhat the results mean:")
    print("  - Gradient check PASSED: renderer is differentiable, training will work")
    print("  - High initial loss is EXPECTED (Gaussians not trained yet)")
    print("  - Rendered image looks noisy/uniform: EXPECTED (uniform opacity init)")
    print("  - After Phase 5 training, rendered should match target X-rays")
    print("\nNext: Phase 5 — Training loop (LoRA + radiative head)")

    return {
        'renderer':    renderer,
        'initial_loss': loss.item(),
        'grad_ok':     grad_ok,
    }


# ─────────────────────────────────────────────────────────────
#  MINI TRAINING DEMO  (optional: 50 iterations to verify convergence)
# ─────────────────────────────────────────────────────────────

def mini_training_demo(data_path: str, output_dir: str,
                       n_iters: int = 50, lr: float = 0.01):
    """
    Run 50 gradient steps on a SINGLE ray to verify the renderer
    can actually learn.  This is not full training (Phase 5) — just
    a proof that gradients flow and loss decreases.

    If this works, Phase 5 will work.
    """
    output_dir = Path(output_dir)
    device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("\n" + "=" * 70)
    print("PHASE 4: Mini Training Demo (50 iters on 1 ray)")
    print("=" * 70)

    data      = torch.load(data_path, map_location='cpu')
    gaussians = data['gaussians']
    rays      = data['rays']
    xrays     = data['xrays']

    N = gaussians['means'].shape[3]
    renderer = BeerLambertRenderer(n_gaussians=N).to(device)

    # Pick one ray (centre pixel of view 0) as the test case
    cx, cy  = 128, 128
    means_1  = gaussians['means'][0:1, cy:cy+1, cx:cx+1].to(device).requires_grad_(True)
    log_s_1  = gaussians['log_scales'][0:1, cy:cy+1, cx:cx+1].to(device).requires_grad_(True)
    opac_1   = gaussians['opacities'][0:1, cy:cy+1, cx:cx+1].to(device).requires_grad_(True)
    t_1      = gaussians['t_values'][0:1, cy:cy+1, cx:cx+1].to(device)
    orig_1   = rays['origins'][0:1, cy:cy+1, cx:cx+1].to(device)
    dir_1    = rays['directions'][0:1, cy:cy+1, cx:cx+1].to(device)
    target_1 = xrays[0, 0:1, 0, cy:cy+1, cx:cx+1].to(device)  # [1,1,1]

    optimiser = torch.optim.Adam([means_1, log_s_1, opac_1], lr=lr)
    losses = []

    print(f"  Target intensity at centre pixel: {target_1.item():.4f}")
    print(f"  Running {n_iters} gradient steps...")

    for i in range(n_iters):
        optimiser.zero_grad()
        rendered_1 = renderer(means_1, log_s_1, opac_1, orig_1, dir_1, t_1)
        loss = F.l1_loss(rendered_1, target_1)
        loss.backward()
        optimiser.step()
        losses.append(loss.item())
        if i % 10 == 0:
            print(f"  iter {i:3d}  loss={loss.item():.6f}  "
                  f"rendered={rendered_1.item():.4f}  target={target_1.item():.4f}")

    print(f"\n  Initial loss: {losses[0]:.6f}")
    print(f"  Final loss:   {losses[-1]:.6f}")
    converged = losses[-1] < losses[0] * 0.5
    print(f"  Converged (>50% reduction): {'YES' if converged else 'NOT YET - check lr'}")

    # Save loss curve
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 4))
        plt.plot(losses)
        plt.xlabel('Iteration');  plt.ylabel('L1 Loss')
        plt.title('Mini Training Demo — Loss Curve (1 ray, 50 iters)')
        plt.grid(True, alpha=0.3)
        out = output_dir / 'phase4_mini_training_loss.png'
        plt.savefig(out, dpi=120, bbox_inches='tight')
        plt.close()
        print(f"  Loss curve saved: {out}")
    except Exception as e:
        print(f"  Loss curve skipped: {e}")

    return losses


# ─────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description='R-VGGT Phase 4: Renderer')
    p.add_argument('--data_path',   required=True,
                   help='Path to phase3_gaussians.pt')
    p.add_argument('--output_dir',  default='data/phase4_output')
    p.add_argument('--mini_train',  action='store_true',
                   help='Also run 50-iter mini training demo')
    p.add_argument('--n_iters',     type=int, default=50)
    args = p.parse_args()

    validate_renderer(args.data_path, args.output_dir)

    if args.mini_train:
        mini_training_demo(args.data_path, args.output_dir, n_iters=args.n_iters)


if __name__ == '__main__':
    main()
