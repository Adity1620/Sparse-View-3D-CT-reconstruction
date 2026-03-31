"""
R-VGGT Phase 5: End-to-End Training  (friend's version — corrected)
=====================================================================
Keeps the exact same logic, classes, and structure as the original,
with three fixes applied:

  FIX 1 — Real VGGT loading
    DummyVGGT replaced with actual VGGT loaded from model.pt,
    using the same hook-based token extraction.

  FIX 2 — Multi-patient data
    Accepts a folder of per-patient phase3_gaussians.pt files
    (one sub-folder per patient, same layout produced by run_all_patients.py).
    Falls back to a single .pt file if that is what is supplied.

  FIX 3 — Corrected CLI
    Arguments match what this script actually needs:
      --data_path   (folder OR single .pt)
      --vggt_model_path
      --output_dir
      --epochs / --lr / --lora_rank / --validate_every / --no_lora

Usage (multi-patient folder):
    python phase5_friend_corrected.py \\
        --data_path /path/to/phase3_output \\
        --vggt_model_path /path/to/model.pt \\
        --output_dir checkpoints/phase5_training \\
        --epochs 200 \\
        --lr 1e-3

Usage (single patient .pt file, original friend style):
    python phase5_friend_corrected.py \\
        --data_path data/phase3_output/phase3_gaussians.pt \\
        --vggt_model_path model.pt \\
        --output_dir checkpoints/phase5_training \\
        --epochs 200 \\
        --lr 1e-3
"""

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter
    HAVE_TB = True
except ImportError:
    HAVE_TB = False
    print("WARNING: tensorboard not available. Install with: pip install tensorboard")

try:
    from peft import LoraConfig, get_peft_model
    HAVE_PEFT = True
except ImportError:
    HAVE_PEFT = False
    print("WARNING: peft not available. Install with: pip install peft")
    print("LoRA will be disabled.")

try:
    from skimage.metrics import peak_signal_noise_ratio as sk_psnr
    from skimage.metrics import structural_similarity as sk_ssim
    HAVE_SKIMAGE = True
except ImportError:
    HAVE_SKIMAGE = False


# ============================================================================
# FIX 1 — REAL VGGT LOADER
# ============================================================================

def load_real_vggt(model_path: str, device: str):
    """
    Load the actual pretrained VGGT model from model.pt.

    model.pt is a state_dict (raw weight tensors, keys like
    'aggregator.patch_embed.blocks.0.attn.qkv.weight', ...).
    We must instantiate the VGGT architecture first, then load weights.

    Requires the vggt package to be importable (it lives in vggt-main/
    next to model.pt — make sure that folder is on PYTHONPATH or run
    from the project root where vggt/ is a sub-package).
    """
    print(f"  Loading VGGT from: {model_path}")

    # ── Import VGGT architecture ──────────────────────────────
    try:
        from vggt.models.vggt import VGGT
    except ImportError:
        # Try adding vggt-main to path automatically
        import sys
        model_dir = Path(model_path).parent
        for candidate in [model_dir, model_dir.parent]:
            sys.path.insert(0, str(candidate))
            try:
                from vggt.models.vggt import VGGT
                print(f"  (Added {candidate} to sys.path)")
                break
            except ImportError:
                continue
        else:
            raise ImportError(
                "Cannot import vggt.models.vggt.VGGT.\n"
                "Make sure the vggt-main folder is on your PYTHONPATH:\n"
                "  export PYTHONPATH=/media/viplab/DATADRIVE1/Sajan_DDS/vggt-main:$PYTHONPATH\n"
                "Then re-run the script."
            )

    # ── Build model and load weights ─────────────────────────
    state_dict = torch.load(model_path, map_location='cpu')

    # state_dict may be wrapped in a dict
    if isinstance(state_dict, dict) and 'model' in state_dict:
        state_dict = state_dict['model']
    elif isinstance(state_dict, nn.Module):
        # Already a full model object — rare but handle it
        vggt_model = state_dict.to(device).eval()
        for param in vggt_model.parameters():
            param.requires_grad = False
        n_params = sum(p.numel() for p in vggt_model.parameters())
        print(f"  ✓ VGGT loaded as full model  ({n_params/1e6:.1f}M params, all frozen)")
        return vggt_model

    # Instantiate with default config (matches the released model.pt)
    vggt_model = VGGT()
    missing, unexpected = vggt_model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  ⚠ Missing keys  ({len(missing)}): {missing[:5]} ...")
    if unexpected:
        print(f"  ⚠ Unexpected keys ({len(unexpected)}): {unexpected[:5]} ...")

    # Keep VGGT on CPU — if tok_grid is pre-computed (Phase 3), VGGT never
    # needs to run on GPU during training, saving ~5GB of VRAM.
    vggt_model = vggt_model.to('cpu')
    vggt_model.eval()

    # Freeze everything — LoRA will selectively unfreeze Q/V later
    for param in vggt_model.parameters():
        param.requires_grad = False

    n_params = sum(p.numel() for p in vggt_model.parameters())
    print(f"  ✓ VGGT loaded from state_dict  ({n_params/1e6:.1f}M params, kept on CPU)")
    return vggt_model


# ============================================================================
# FIX 2 — MULTI-PATIENT DATA LOADING
# ============================================================================

def load_data(data_path: str) -> List[Dict]:
    """
    Load phase3 data.  Accepts either:
      (a) A folder  — scans for phase3_gaussians.pt in every sub-folder
      (b) A single  — loads just that one .pt file

    Returns a list of data dicts (one per patient).
    """
    p = Path(data_path)

    if p.is_dir():
        # Find all per-patient phase3_gaussians.pt files
        pt_files = sorted(p.rglob("phase3_gaussians.pt"))
        if not pt_files:
            raise RuntimeError(
                f"No phase3_gaussians.pt files found under {data_path}.\n"
                f"Expected structure: data_path/<patient_id>/phase3_gaussians.pt"
            )
        print(f"  Found {len(pt_files)} patient(s):")
        patients = []
        for f in pt_files:
            pid = f.parent.name
            print(f"    {pid}  →  {f}")
            d = torch.load(f, map_location='cpu')
            d['patient_id'] = pid
            patients.append(d)
        return patients

    elif p.suffix == '.pt':
        # Single-file mode (original friend style)
        print(f"  Loading single file: {p}")
        d = torch.load(p, map_location='cpu')
        d['patient_id'] = p.parent.name or 'patient'
        return [d]

    else:
        raise RuntimeError(f"data_path must be a folder or a .pt file, got: {data_path}")


# ============================================================================
# RADIATIVE DENSITY HEAD  (unchanged from friend's original)
# ============================================================================

class RadiativeDensityHead(nn.Module):
    """
    Maps VGGT token features to volumetric densities.
    Shared across all patients — this is the component that generalises.

    KEY MEMORY OPTIMIZATION:
    Instead of expanding features to [V, H, W, N, D] (2GB+), we predict
    N density values directly from each pixel feature [D] → [N].
    This keeps peak memory at [V, H, W, D] (~50MB) throughout.
    """

    def __init__(
        self,
        feature_dim: int = 1024,
        hidden_dim: int = 512,
        n_gaussians: int = 8,   # predict all N densities at once per pixel
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_gaussians = n_gaussians

        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_gaussians),  # predict N densities at once
            nn.Softplus(),   # Ensures positive densities
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [V, H, W, D]  one feature vector per pixel
        Returns:
            densities: [V, H, W, N, 1]
        """
        # [V, H, W, D] -> [V, H, W, N]
        out = self.mlp(features)
        # [V, H, W, N] -> [V, H, W, N, 1]
        return out.unsqueeze(-1)


# ============================================================================
# VGGT TOKEN EXTRACTOR  (unchanged from friend's original)
# ============================================================================

class VGGTTokenExtractor(nn.Module):
    """Extract transformer tokens from VGGT backbone via a forward hook."""

    def __init__(self, vggt_model):
        super().__init__()
        self.model = vggt_model
        self.activation = {}

        # Hook on the last AA transformer block (before DPT head)
        last_block = self.model.aggregator.global_blocks[-1]
        last_block.register_forward_hook(self._get_activation('tokens'))

    def _get_activation(self, name):
        def hook(module, input, output):
            self.activation[name] = output
        return hook

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [1, V, 3, H, W]  input X-ray views (batch=1)
        Returns:
            tokens: [V, N_tokens, D]
        """
        # VGGT patch size is 14 — H and W must be multiples of 14
        # 256 is not (256/14=18.28), so resize to nearest: round(256/14)*14 = 252
        B, V, C, H, W = x.shape
        new_H = round(H / 14) * 14
        new_W = round(W / 14) * 14
        if new_H != H or new_W != W:
            x_vggt = F.interpolate(
                x.view(B * V, C, H, W),
                size=(new_H, new_W),
                mode='bilinear',
                align_corners=False,
            ).view(B, V, C, new_H, new_W)
        else:
            x_vggt = x

        with torch.no_grad():    # backbone always frozen
            _ = self.model(x_vggt)

        raw_tokens = self.activation['tokens']

        # Handle concatenated format  [1, V*N, D]  →  [V, N, D]
        if raw_tokens.dim() == 3 and raw_tokens.shape[0] == 1:
            V = x.shape[1]
            tokens_per_view = raw_tokens.shape[1] // V
            tokens = raw_tokens.view(V, tokens_per_view, -1)
        else:
            tokens = raw_tokens

        return tokens


# ============================================================================
# BEER-LAMBERT RENDERER  (unchanged from friend's original)
# ============================================================================

class BeerLambertRenderer(nn.Module):
    """
    Differentiable Beer-Lambert volume renderer for X-rays.
    I(x,y) = I₀ × exp(-Σ σᵢ Δxᵢ)
    """

    def __init__(self, i0: float = 1.0):
        super().__init__()
        self.i0 = i0

    def forward(
        self,
        means: torch.Tensor,        # [V, H, W, N, 3]
        densities: torch.Tensor,    # [V, H, W, N, 1]
        valid_mask: torch.Tensor,   # [V, H, W, N]
        t_values: torch.Tensor,     # [V, H, W, N]
        height: int,
        width: int,
    ) -> torch.Tensor:
        """Returns rendered X-rays [V, H, W]."""

        # Apply valid mask
        densities = densities * valid_mask.unsqueeze(-1).float()

        # Sort by depth (front-to-back)
        t_sorted, indices = torch.sort(t_values, dim=-1)
        densities_sorted = torch.gather(
            densities, dim=3,
            index=indices.unsqueeze(-1).expand_as(densities)
        )

        # Path lengths between consecutive samples
        dt = torch.diff(t_sorted, dim=-1, prepend=t_sorted[..., :1])
        dt = dt.clamp(min=1e-6)

        # Beer-Lambert attenuation
        attenuation = (densities_sorted.squeeze(-1) * dt).sum(dim=-1)  # [V, H, W]
        intensity   = self.i0 * torch.exp(-attenuation)

        return intensity


# ============================================================================
# COMPLETE R-VGGT MODEL  (unchanged from friend's original)
# ============================================================================

class RVGGTModel(nn.Module):
    """
    Complete R-VGGT model:
      - VGGT backbone  (frozen, optionally + LoRA)
      - Radiative density head  (trainable, shared across patients)
      - Beer-Lambert renderer  (differentiable, no parameters)
    """

    def __init__(
        self,
        vggt_model,
        feature_dim: int = 1024,
        use_lora: bool = True,
        lora_rank: int = 8,
        lora_alpha: int = 16,
    ):
        super().__init__()

        if use_lora and HAVE_PEFT:
            print("  Setting up LoRA adapters...")
            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                target_modules=["qkv", "proj"],  # VGGT uses fused qkv, not separate q_proj/v_proj
                lora_dropout=0.1,
                bias="none",
            )
            vggt_model = get_peft_model(vggt_model, lora_config)
            vggt_model.print_trainable_parameters()
        else:
            if use_lora:
                print("  WARNING: LoRA requested but peft not available — backbone stays fully frozen.")
            else:
                print("  Using frozen VGGT backbone (no LoRA).")

        self.vggt_extractor = VGGTTokenExtractor(vggt_model)
        self.radiative_head = RadiativeDensityHead(
            feature_dim=feature_dim,
            hidden_dim=512,
            n_gaussians=8,   # must match Phase 3 N value
        )
        self.renderer = BeerLambertRenderer()

    def forward(
        self,
        xrays: torch.Tensor,       # [1, V, 3, H, W]
        gaussians: Dict,
        camera_params: Dict,
        tok_grid: torch.Tensor = None,  # [V, H, W, D] pre-computed tokens (preferred)
    ) -> Dict[str, torch.Tensor]:
        """Full forward pass for one patient.

        Memory strategy:
          If tok_grid is provided (pre-computed in Phase 3), use it directly
          and skip the 1.2B-param VGGT forward pass entirely.
          This is the recommended path — saves ~18GB of GPU memory.
          Only falls back to live VGGT inference if tok_grid is None.
        """

        # ── 1. Get pixel-level features ───────────────────────
        means      = gaussians['means']       # [V, H, W, N, 3]
        t_values   = gaussians['t_values']    # [V, H, W, N]
        valid_mask = gaussians['valid_mask']  # [V, H, W, N]
        V, H, W, N, _ = means.shape

        if tok_grid is not None:
            # Fast path: use pre-saved tokens from Phase 3
            # tok_grid is [V, H, W, D] — already at pixel resolution
            features_pixel = tok_grid  # [V, H, W, D]
        else:
            # Slow path: run live VGGT inference (high memory cost)
            features = self.vggt_extractor(xrays)   # [V, N_tokens, D]

            N_tokens  = features.shape[1]
            new_H = round(H / 14) * 14
            new_W = round(W / 14) * 14
            n_patch_h = new_H // 14
            n_patch_w = new_W // 14
            n_patch   = n_patch_h * n_patch_w
            n_extra   = N_tokens - n_patch

            patch_tokens  = features[:, n_extra:, :]
            features_grid = patch_tokens.view(V, n_patch_h, n_patch_w, -1)
            features_pixel = F.interpolate(
                features_grid.permute(0, 3, 1, 2),
                size=(H, W),
                mode='bilinear',
                align_corners=False,
            ).permute(0, 2, 3, 1)   # [V, H, W, D]

        # ── 2. Predict per-Gaussian density ───────────────────
        # features_pixel is [V, H, W, D] — feed directly, get [V, H, W, N, 1]
        # Peak memory: [V, H, W, D] = 4*256*256*1024 * 4 bytes = ~1GB
        # Much better than previous [V, H, W, N, D] = ~8GB
        densities = self.radiative_head(features_pixel)   # [V, H, W, N, 1]

        # ── 3. Beer-Lambert render ─────────────────────────────
        rendered = self.renderer(
            means, densities, valid_mask, t_values, H, W
        )   # [V, H, W]

        return {
            'rendered':  rendered,
            'densities': densities,
            'features':  features_pixel,
        }


# ============================================================================
# TRAINING / VALIDATION HELPERS  (unchanged from friend's original)
# ============================================================================

def train_one_patient(
    model: RVGGTModel,
    optimizer: torch.optim.Optimizer,
    patient: Dict,
    device: str,
) -> Dict[str, float]:
    """One gradient step on a single patient."""

    model.train()

    # Only move what's needed right now — keep everything else on CPU
    xrays = patient['xrays'].to(device)           # [1, V, 3, H, W]

    # Move gaussians to GPU one tensor at a time (lazy, not all upfront)
    gaussians = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in patient['gaussians'].items()
    }
    camera_params = patient.get('camera', {})

    # Use pre-computed VGGT tokens if available (saves ~18GB GPU memory)
    tok_grid = patient.get('vggt_pixel_tokens', None)
    if tok_grid is not None:
        tok_grid = tok_grid.to(device)

    outputs  = model(xrays, gaussians, camera_params, tok_grid=tok_grid)
    rendered = outputs['rendered']                 # [V, H, W]
    target   = xrays[0, :, 0, :, :]              # [V, H, W]

    loss_recon    = F.mse_loss(rendered, target)
    loss_sparsity = 0.001 * outputs['densities'].abs().mean()
    loss_total    = loss_recon + loss_sparsity

    optimizer.zero_grad()
    loss_total.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    psnr = -10.0 * math.log10(max(loss_recon.item(), 1e-10))

    return {
        'loss_total':    loss_total.item(),
        'loss_recon':    loss_recon.item(),
        'loss_sparsity': loss_sparsity.item(),
        'psnr':          psnr,
    }


@torch.no_grad()
def validate_patient(
    model: RVGGTModel,
    patient: Dict,
    device: str,
) -> Dict[str, float]:
    """Validation pass on one patient."""

    model.eval()

    xrays = patient['xrays'].to(device)
    gaussians = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in patient['gaussians'].items()
    }
    camera_params = patient.get('camera', {})

    tok_grid = patient.get('vggt_pixel_tokens', None)
    if tok_grid is not None:
        tok_grid = tok_grid.to(device)

    outputs  = model(xrays, gaussians, camera_params, tok_grid=tok_grid)
    rendered = outputs['rendered']
    target   = xrays[0, :, 0, :, :]

    loss = F.mse_loss(rendered, target)
    psnr = -10.0 * math.log10(max(loss.item(), 1e-10))

    # Optional SSIM
    ssim_val = 0.0
    if HAVE_SKIMAGE:
        r_np = rendered[0].cpu().numpy().clip(0, 1)
        t_np = target[0].cpu().numpy().clip(0, 1)
        w    = min(7, min(r_np.shape) - 1)
        if w % 2 == 0:
            w -= 1
        if w >= 3:
            ssim_val = float(sk_ssim(t_np, r_np, data_range=1.0, win_size=w))

    return {
        'val_loss': loss.item(),
        'val_psnr': psnr,
        'val_ssim': ssim_val,
    }


def save_checkpoint(
    model: RVGGTModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict,
    output_dir: Path,
    is_best: bool = False,
):
    """Save training checkpoint."""

    checkpoint = {
        'epoch':                epoch,
        'model_state_dict':     model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics':              metrics,
    }

    torch.save(checkpoint, output_dir / 'checkpoint_latest.pt')

    if epoch % 20 == 0:
        torch.save(checkpoint, output_dir / f'checkpoint_epoch_{epoch:04d}.pt')

    if is_best:
        torch.save(checkpoint, output_dir / 'checkpoint_best.pt')
        print(f"  ★ New best checkpoint saved  (epoch {epoch})")


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def run_training(
    data_path:      str,
    vggt_model_path: str,
    output_dir:     str,
    epochs:         int   = 200,
    lr:             float = 1e-3,
    use_lora:       bool  = True,
    lora_rank:      int   = 8,
    validate_every: int   = 5,
):
    print("=" * 70)
    print("PHASE 5: END-TO-END TRAINING  (friend's version — corrected)")
    print("=" * 70)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    writer = None
    if HAVE_TB:
        writer = SummaryWriter(output_dir / 'logs')

    # ── 1. Load data ──────────────────────────────────────────
    print(f"\n[1/4] Loading data from: {data_path}")
    patients = load_data(data_path)
    print(f"  ✓ {len(patients)} patient(s) loaded")

    # ── 2. Load real VGGT ─────────────────────────────────────
    print(f"\n[2/4] Loading VGGT model from: {vggt_model_path}")
    vggt_model = load_real_vggt(vggt_model_path, device)

    # ── 3. Build R-VGGT ───────────────────────────────────────
    print(f"\n[3/4] Building R-VGGT model...")
    model = RVGGTModel(
        vggt_model=vggt_model,
        feature_dim=1024,
        use_lora=use_lora,
        lora_rank=lora_rank,
    ).to(device)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total     = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters:     {n_total:,}")
    print(f"  Trainable parameters: {n_trainable:,}  ({100*n_trainable/n_total:.2f}%)")

    # ── 4. Optimizer & scheduler ──────────────────────────────
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr / 10
    )

    # ── 5. Training loop ──────────────────────────────────────
    print(f"\n[4/4] Starting training for {epochs} epochs  "
          f"({len(patients)} patient(s) per epoch)...")
    print("=" * 70)

    best_val_loss = float('inf')

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # --- train: cycle through all patients ---
        epoch_metrics: Dict[str, List[float]] = {
            'loss_total': [], 'loss_recon': [], 'loss_sparsity': [], 'psnr': []
        }

        for patient in patients:
            m = train_one_patient(model, optimizer, patient, device)
            for k in epoch_metrics:
                epoch_metrics[k].append(m[k])
            # Free GPU memory between patients
            torch.cuda.empty_cache()

        train_summary = {k: float(np.mean(v)) for k, v in epoch_metrics.items()}

        # --- validate ---
        val_summary: Dict[str, float] = {}
        if epoch % validate_every == 0:
            val_losses, val_psnrs, val_ssims = [], [], []
            for patient in patients:
                vm = validate_patient(model, patient, device)
                val_losses.append(vm['val_loss'])
                val_psnrs.append(vm['val_psnr'])
                val_ssims.append(vm['val_ssim'])
            val_summary = {
                'val_loss': float(np.mean(val_losses)),
                'val_psnr': float(np.mean(val_psnrs)),
                'val_ssim': float(np.mean(val_ssims)),
            }

        scheduler.step()

        epoch_time = time.time() - epoch_start

        # --- log ---
        log_str = (
            f"Epoch {epoch:03d}/{epochs} | "
            f"Loss: {train_summary['loss_recon']:.4f} | "
            f"PSNR: {train_summary['psnr']:.2f} dB | "
            f"Time: {epoch_time:.1f}s"
        )
        if val_summary:
            log_str += (
                f" | Val Loss: {val_summary['val_loss']:.4f} | "
                f"Val PSNR: {val_summary['val_psnr']:.2f} dB"
            )
            if val_summary.get('val_ssim', 0) > 0:
                log_str += f" | Val SSIM: {val_summary['val_ssim']:.4f}"
        print(log_str)

        if writer:
            for k, v in train_summary.items():
                writer.add_scalar(f'train/{k}', v, epoch)
            for k, v in val_summary.items():
                writer.add_scalar(f'val/{k}', v, epoch)
            writer.add_scalar('lr', scheduler.get_last_lr()[0], epoch)

        # --- checkpoint ---
        is_best = bool(val_summary and val_summary['val_loss'] < best_val_loss)
        if is_best:
            best_val_loss = val_summary['val_loss']

        if epoch % 10 == 0 or is_best:
            save_checkpoint(
                model, optimizer, epoch,
                {**train_summary, **val_summary},
                output_dir, is_best,
            )

    # ── Done ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE ✓")
    print("=" * 70)
    print(f"\nCheckpoints saved to: {output_dir}")
    print(f"Best validation loss:  {best_val_loss:.4f}")
    print(f"\nNext step: run Phase 6 validation")

    if writer:
        writer.close()


# ============================================================================
# CLI  (FIX 3 — matches what the script actually needs)
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Phase 5: Train R-VGGT model end-to-end (friend's version — corrected)"
    )
    parser.add_argument(
        "--data_path", type=str, required=True,
        help=(
            "Path to phase3 data. Two accepted formats:\n"
            "  (a) Folder containing per-patient sub-folders "
            "      (each with phase3_gaussians.pt) — multi-patient mode\n"
            "  (b) Single phase3_gaussians.pt file — single-patient mode"
        )
    )
    parser.add_argument(
        "--vggt_model_path", type=str, required=True,
        help="Path to pretrained VGGT model.pt"
    )
    parser.add_argument(
        "--output_dir", type=str, default="checkpoints/phase5_training",
        help="Output directory for checkpoints and logs"
    )
    parser.add_argument(
        "--epochs", type=int, default=200,
        help="Number of training epochs (default: 200)"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="Learning rate (default: 1e-3)"
    )
    parser.add_argument(
        "--no_lora", action="store_true",
        help="Disable LoRA (not recommended)"
    )
    parser.add_argument(
        "--lora_rank", type=int, default=8,
        help="LoRA rank r (default: 8)"
    )
    parser.add_argument(
        "--validate_every", type=int, default=5,
        help="Run validation every N epochs (default: 5)"
    )

    args = parser.parse_args()

    run_training(
        data_path       = args.data_path,
        vggt_model_path = args.vggt_model_path,
        output_dir      = args.output_dir,
        epochs          = args.epochs,
        lr              = args.lr,
        use_lora        = not args.no_lora,
        lora_rank       = args.lora_rank,
        validate_every  = args.validate_every,
    )


if __name__ == "__main__":
    main()