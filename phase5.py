"""
R-VGGT Phase 5: Training Loop
==============================
Trains the volumetric Gaussians to reproduce the 4 sparse X-ray views
using differentiable Beer-Lambert rendering.

Training strategy:
  - Optimise Gaussian parameters: means, log_scales, opacities
  - A lightweight "radiative head" MLP refines opacity predictions
    using VGGT token features (the LoRA-style adaptation)
  - Loss = photometric (L1 + SSIM) + regularisation
  - ~500-2000 iterations depending on convergence

Input:  data/phase3_output/phase3_gaussians.pt
        data/phase4_output/  (renderer code imported)
Output: data/phase5_output/trained_gaussians.pt
        data/phase5_output/training_curves.png
        data/phase5_output/final_renders.png
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import argparse
import math
import time

# Import renderer from Phase 4
# (place phase4_renderer.py in same directory)
from phase4 import BeerLambertRenderer, photometric_loss, regularisation_loss


# ─────────────────────────────────────────────────────────────
#  RADIATIVE HEAD  (the LoRA-style adaptation from VGGT tokens)
# ─────────────────────────────────────────────────────────────

class RadiativeHead(nn.Module):
    """
    Lightweight MLP that predicts opacity corrections from VGGT features.

    This is the key innovation connecting VGGT geometric priors to
    volumetric radiative rendering.

    Instead of learning opacity from scratch, we:
      1. Start with uniform initial opacity (Phase 3)
      2. Use VGGT's geometric understanding to predict CORRECTIONS
         to opacity based on what the transformer "knows" about the geometry
      3. Add the correction to the base opacity

    This is analogous to LoRA (Low-Rank Adaptation):
      base_opacity + delta_opacity(vggt_features) = final_opacity

    Why this works:
      VGGT was trained on multi-view images and learned geometric structure.
      Even though it was trained on natural images (not X-rays), its features
      encode depth, surface normals, and geometric relationships.
      The radiative head translates these geometric features into
      X-ray attenuation predictions.

    Architecture:
      Input:  VGGT feature vector (1024-dim) + depth position (1-dim)
      Hidden: 256 -> 128
      Output: opacity correction (1-dim, tanh-scaled)
    """

    def __init__(self, feature_dim: int = 1024, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim + 1, hidden_dim),  # +1 for depth encoding
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh()   # output in (-1, +1): additive correction to base opacity
        )
        # Initialise near-zero so corrections start small
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)

    def forward(self,
                pixel_features: torch.Tensor,
                t_values:       torch.Tensor,
                t_near:         float,
                t_far:          float) -> torch.Tensor:
        """
        Predict opacity corrections for all Gaussians.

        Args:
            pixel_features : [V, H, W, D]     VGGT feature per pixel
            t_values       : [V, H, W, N]     depth of each Gaussian
            t_near, t_far  : depth range for normalisation

        Returns:
            corrections    : [V, H, W, N, 1]  opacity corrections in (-1, +1)
        """
        V, H, W, D = pixel_features.shape
        N = t_values.shape[-1]

        # Normalise depth to [0, 1]
        t_norm = ((t_values - t_near) / (t_far - t_near + 1e-8)).unsqueeze(-1)  # [V,H,W,N,1]

        # Expand pixel features to all N Gaussians on each ray
        feat_expanded = pixel_features.unsqueeze(-2).expand(V, H, W, N, D)       # [V,H,W,N,D]

        # Concatenate: [feature, normalised_depth]
        inp = torch.cat([feat_expanded, t_norm], dim=-1)  # [V, H, W, N, D+1]

        # Flatten for batch MLP, then reshape back
        inp_flat = inp.reshape(-1, D + 1)
        out_flat = self.net(inp_flat)                     # [-1, 1]
        corrections = out_flat.reshape(V, H, W, N, 1)

        return corrections


# ─────────────────────────────────────────────────────────────
#  TRAINABLE GAUSSIAN PARAMETERS
# ─────────────────────────────────────────────────────────────

class TrainableGaussians(nn.Module):
    """
    Wraps Gaussian parameters as nn.Parameters for optimisation.

    We optimise:
      - means      (positions can drift slightly from initial ray placement)
      - log_scales (allow Gaussians to grow/shrink to fit anatomy)
      - opacities  (main training signal: high in tissue, low in air)

    We do NOT optimise:
      - ray origins / directions (fixed camera geometry)
      - t_values (depth ordering stays fixed; positions update via means)
    """

    def __init__(self, gaussians: dict, valid_mask: torch.Tensor = None):
        super().__init__()
        self.means      = nn.Parameter(gaussians['means'].clone())
        self.log_scales = nn.Parameter(gaussians['log_scales'].clone())
        self.opacities  = nn.Parameter(gaussians['opacities'].clone())

        # Fixed (not trained)
        self.register_buffer('t_values',   gaussians['t_values'].clone())
        if valid_mask is not None:
            self.register_buffer('valid_mask', valid_mask.clone())
        else:
            self.register_buffer('valid_mask', torch.ones_like(gaussians['t_values'], dtype=torch.bool))

    def get_params(self):
        return self.means, self.log_scales, self.opacities, self.t_values

    def apply_radiative_correction(self, corrections: torch.Tensor) -> torch.Tensor:
        """Add MLP corrections to base opacity logits."""
        return self.opacities + corrections  # [V, H, W, N, 1]


# ─────────────────────────────────────────────────────────────
#  TRAINING LOOP
# ─────────────────────────────────────────────────────────────

def train(data_path:      str,
          output_dir:     str,
          n_iters:        int   = 1000,
          lr_gaussians:   float = 0.005,
          lr_head:        float = 0.001,
          log_every:      int   = 50,
          save_every:     int   = 250,
          use_head:       bool  = True,
          valid_only:     bool  = True):
    """
    Main training loop.

    Optimisation strategy:
      - Adam optimiser, separate learning rates for Gaussians and radiative head
      - Gaussians: higher lr (0.005) — they need to move quickly
      - Radiative head: lower lr (0.001) — MLP should learn slowly and stably
      - Gradient clipping to prevent explosions early in training

    Args:
        n_iters      : total gradient steps (500 = fast test, 2000 = full)
        lr_gaussians : learning rate for Gaussian parameters
        lr_head      : learning rate for radiative head MLP
        log_every    : print loss every N iters
        save_every   : save checkpoint every N iters
        use_head     : whether to use radiative head (False = Gaussians only)
        valid_only   : only compute loss on in-volume Gaussians
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 70)
    print("PHASE 5: Training Loop")
    print("=" * 70)
    print(f"Device     : {device}")
    print(f"Iterations : {n_iters}")
    print(f"Use head   : {use_head}")

    # ── Load data ─────────────────────────────────────────────
    print(f"\n[1/5] Loading Phase 3 data...")
    data       = torch.load(data_path, map_location='cpu')
    gaussians  = data['gaussians']
    rays       = data['rays']
    xrays      = data['xrays']        # [1, V, 3, H, W]
    tok_grid   = data.get('vggt_pixel_tokens', None)   # [V, H, W, 1024] or None
    t_near     = float(data['config']['t_near'])
    t_far      = float(data['config']['t_far'])

    V, H, W, N, _ = gaussians['means'].shape
    target = xrays[0, :, 0].to(device)   # [V, H, W]  use channel 0 (grayscale)

    print(f"  Views: {V}  Detector: {H}x{W}  Gaussians/ray: {N}")
    print(f"  t range: [{t_near}, {t_far}] mm")
    print(f"  Target range: [{target.min():.3f}, {target.max():.3f}]")

    # ── Build models ──────────────────────────────────────────
    print("\n[2/5] Building models...")
    gauss_model = TrainableGaussians(gaussians, gaussians.get('valid_mask')).to(device)
    renderer    = BeerLambertRenderer(n_gaussians=N, use_soft_mask=True).to(device)

    param_groups = [
        {'params': [gauss_model.means,
                    gauss_model.log_scales,
                    gauss_model.opacities],
         'lr': lr_gaussians,
         'name': 'gaussians'},
    ]

    head = None
    if use_head and tok_grid is not None:
        feature_dim = tok_grid.shape[-1]
        head        = RadiativeHead(feature_dim=feature_dim, hidden_dim=256).to(device)
        param_groups.append({'params': head.parameters(),
                             'lr': lr_head, 'name': 'head'})
        pixel_features = tok_grid.to(device)  # [V, H, W, D]
        print(f"  RadiativeHead: {sum(p.numel() for p in head.parameters()):,} params")
        print(f"  VGGT feature dim: {feature_dim}")
    elif use_head and tok_grid is None:
        print("  WARNING: use_head=True but no vggt_pixel_tokens found.")
        print("  Training Gaussians only (no radiative head).")
        use_head = False

    optimiser = torch.optim.Adam(param_groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=n_iters, eta_min=1e-6)

    gauss_params = sum(p.numel() for p in gauss_model.parameters())
    print(f"  Trainable Gaussian params: {gauss_params:,}")
    print(f"  Renderer params: 0 (fixed physics)")

    # Fixed ray buffers
    origins    = rays['origins'].to(device)
    directions = rays['directions'].to(device)

    # ── Training loop ─────────────────────────────────────────
    print(f"\n[3/5] Training for {n_iters} iterations...")
    print("-" * 60)
    print(f"{'Iter':>6}  {'Photo':>8}  {'Reg':>8}  {'Total':>8}  {'Time':>6}")
    print("-" * 60)

    losses_photo = []
    losses_reg   = []
    losses_total = []
    t_start      = time.time()
    best_loss    = float('inf')

    for it in range(n_iters):
        optimiser.zero_grad()

        # Get current Gaussian parameters
        means, log_scales, opacities, t_vals = gauss_model.get_params()

        # Apply radiative head corrections if available
        if use_head and head is not None:
            corrections    = head(pixel_features, t_vals, t_near, t_far)
            opacities_eff  = gauss_model.apply_radiative_correction(corrections)
        else:
            opacities_eff  = opacities

        # Render all views
        rendered = renderer(means, log_scales, opacities_eff,
                            origins, directions, t_vals)   # [V, H, W]

        # Photometric loss
        if valid_only and gauss_model.valid_mask is not None:
            # Aggregate valid mask to pixel level:
            # a pixel is valid if ANY of its Gaussians are inside the volume
            pixel_valid = gauss_model.valid_mask.any(dim=-1).float()  # [V, H, W]
        else:
            pixel_valid = None

        loss_photo = photometric_loss(rendered, target, mask=pixel_valid)

        # Regularisation
        loss_reg = regularisation_loss(log_scales, opacities_eff,
                                       gauss_model.valid_mask)

        loss = loss_photo + loss_reg

        # Backward
        loss.backward()

        # Gradient clipping (prevents early instability)
        nn.utils.clip_grad_norm_(gauss_model.parameters(), max_norm=1.0)
        if head is not None:
            nn.utils.clip_grad_norm_(head.parameters(), max_norm=1.0)

        optimiser.step()
        scheduler.step()

        # Logging
        losses_photo.append(loss_photo.item())
        losses_reg.append(loss_reg.item())
        losses_total.append(loss.item())

        if loss.item() < best_loss:
            best_loss = loss.item()

        if it % log_every == 0 or it == n_iters - 1:
            elapsed = time.time() - t_start
            rendered_range = f"[{rendered.min():.3f},{rendered.max():.3f}]"
            print(f"{it:6d}  {loss_photo.item():8.5f}  {loss_reg.item():8.5f}  "
                  f"{loss.item():8.5f}  {elapsed:5.1f}s  render={rendered_range}")

        # Checkpoint
        if (it + 1) % save_every == 0:
            ckpt = {
                'iter':        it + 1,
                'means':       means.detach().cpu(),
                'log_scales':  log_scales.detach().cpu(),
                'opacities':   opacities_eff.detach().cpu(),
                'loss':        loss.item(),
            }
            ckpt_path = output_dir / f'checkpoint_iter{it+1}.pt'
            torch.save(ckpt, ckpt_path)
            print(f"  --> Checkpoint saved: {ckpt_path}")

    print("-" * 60)
    print(f"Training complete. Best loss: {best_loss:.5f}")

    # ── Save final Gaussians ──────────────────────────────────
    print("\n[4/5] Saving trained Gaussians...")
    means_f, log_s_f, opac_f, t_f = gauss_model.get_params()
    if use_head and head is not None:
        with torch.no_grad():
            corrections_f = head(pixel_features, t_f, t_near, t_far)
            opac_final    = gauss_model.apply_radiative_correction(corrections_f)
    else:
        opac_final = opac_f

    save_dict = {
        'gaussians': {
            'means':      means_f.detach().cpu(),
            'log_scales': log_s_f.detach().cpu(),
            'opacities':  opac_final.detach().cpu(),
            't_values':   t_f.cpu(),
            'valid_mask': gauss_model.valid_mask.cpu(),
        },
        'rays':      rays,
        'xrays':     data['xrays'],
        'spacing':   data['spacing'],
        'camera':    data['camera'],
        'vol_centre': data.get('vol_centre'),
        'config':    data['config'],
        'training': {
            'n_iters':       n_iters,
            'final_loss':    losses_total[-1],
            'best_loss':     best_loss,
            'losses_photo':  losses_photo,
            'losses_total':  losses_total,
        }
    }
    out_path = output_dir / 'trained_gaussians.pt'
    torch.save(save_dict, out_path)
    print(f"  Saved: {out_path}")

    # ── Final render + plots ──────────────────────────────────
    print("\n[5/5] Generating final renders and training curves...")
    _save_training_plots(losses_photo, losses_reg, losses_total, output_dir)
    _save_final_renders(renderer, save_dict, target, output_dir, device)

    print("\n" + "=" * 70)
    print("PHASE 5 COMPLETE")
    print("=" * 70)
    print(f"  Final photometric loss : {losses_photo[-1]:.5f}")
    print(f"  Best loss              : {best_loss:.5f}")
    print(f"  Loss reduction         : {(1 - losses_total[-1]/losses_total[0])*100:.1f}%")
    print(f"\nNext: Phase 6 — 3D volume reconstruction from trained Gaussians")
    print(f"  python phase6_reconstruct.py --data_path {out_path}")

    return save_dict


# ─────────────────────────────────────────────────────────────
#  VISUALISATION HELPERS
# ─────────────────────────────────────────────────────────────

def _save_training_plots(losses_photo, losses_reg, losses_total, output_dir):
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Phase 5 Training Curves', fontsize=13)

        iters = list(range(len(losses_total)))

        # Loss curves
        axes[0].plot(iters, losses_total, label='Total', color='navy',    lw=2)
        axes[0].plot(iters, losses_photo, label='Photo', color='steelblue', lw=1.5, linestyle='--')
        axes[0].plot(iters, losses_reg,   label='Reg',   color='tomato',    lw=1, linestyle=':')
        axes[0].set_xlabel('Iteration'); axes[0].set_ylabel('Loss')
        axes[0].set_title('Loss Curves')
        axes[0].legend(); axes[0].grid(True, alpha=0.3)
        axes[0].set_yscale('log')

        # Loss reduction bar
        pct = (1 - losses_total[-1] / (losses_total[0] + 1e-8)) * 100
        axes[1].bar(['Initial', 'Final'],
                    [losses_total[0], losses_total[-1]],
                    color=['tomato', 'steelblue'])
        axes[1].set_ylabel('Total Loss')
        axes[1].set_title(f'Loss Reduction: {pct:.1f}%')
        axes[1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        out = output_dir / 'training_curves.png'
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {out}")
    except Exception as e:
        print(f"  Training plot skipped: {e}")


@torch.no_grad()
def _save_final_renders(renderer, data, target, output_dir, device):
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        gaussians = data['gaussians']
        rays      = data['rays']

        means      = gaussians['means'].to(device)
        log_scales = gaussians['log_scales'].to(device)
        opacities  = gaussians['opacities'].to(device)
        t_values   = gaussians['t_values'].to(device)
        origins    = rays['origins'].to(device)
        directions = rays['directions'].to(device)

        rendered = renderer(means, log_scales, opacities,
                            origins, directions, t_values)  # [V, H, W]

        V = rendered.shape[0]
        fig, axes = plt.subplots(V, 3, figsize=(15, 5*V))
        if V == 1: axes = axes[None]
        fig.suptitle('Phase 5: Final Renders vs Target X-rays', fontsize=13)

        rendered_np = rendered.cpu().numpy()
        target_np   = target.cpu().numpy()

        for v in range(V):
            axes[v, 0].imshow(target_np[v],   cmap='gray', vmin=0, vmax=1)
            axes[v, 0].set_title(f'View {v} — Target')
            axes[v, 1].imshow(rendered_np[v], cmap='gray', vmin=0, vmax=1)
            axes[v, 1].set_title(f'View {v} — Rendered (trained)')
            diff = np.abs(target_np[v] - rendered_np[v])
            im = axes[v, 2].imshow(diff, cmap='hot', vmin=0, vmax=0.5)
            axes[v, 2].set_title(f'View {v} — Error (mean={diff.mean():.3f})')
            plt.colorbar(im, ax=axes[v, 2])

        for ax in axes.flatten(): ax.axis('off')
        plt.tight_layout()
        out = output_dir / 'final_renders.png'
        plt.savefig(out, dpi=120, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {out}")

        # PSNR metric
        for v in range(V):
            mse  = ((rendered_np[v] - target_np[v]) ** 2).mean()
            psnr = -10 * np.log10(mse + 1e-8)
            print(f"  View {v} — MSE: {mse:.5f}   PSNR: {psnr:.2f} dB")

    except Exception as e:
        print(f"  Final render plot skipped: {e}")


# ─────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description='R-VGGT Phase 5: Training')
    p.add_argument('--data_path',    required=True,
                   help='Path to phase3_gaussians.pt')
    p.add_argument('--output_dir',   default='data/phase5_output')
    p.add_argument('--n_iters',      type=int,   default=1000,
                   help='Training iterations (500=quick test, 2000=full)')
    p.add_argument('--lr_gaussians', type=float, default=0.005)
    p.add_argument('--lr_head',      type=float, default=0.001)
    p.add_argument('--log_every',    type=int,   default=50)
    p.add_argument('--save_every',   type=int,   default=250)
    p.add_argument('--no_head',      action='store_true',
                   help='Disable radiative head (Gaussians only)')
    p.add_argument('--no_valid_mask', action='store_true',
                   help='Include out-of-volume Gaussians in loss')
    args = p.parse_args()

    train(
        data_path    = args.data_path,
        output_dir   = args.output_dir,
        n_iters      = args.n_iters,
        lr_gaussians = args.lr_gaussians,
        lr_head      = args.lr_head,
        log_every    = args.log_every,
        save_every   = args.save_every,
        use_head     = not args.no_head,
        valid_only   = not args.no_valid_mask,
    )

if __name__ == '__main__':
    main()
