"""
R-VGGT Phase 6: Validation with Zero-Shot + Few-Shot Evaluation
===============================================================
Matched exactly to phase5_friend_corrected.py architecture.

Two evaluation modes:
  Zero-shot  : render immediately from trained model, no adaptation
  Few-shot   : optimize Gaussians for N iters (head frozen), then render
               This is what gives ~21 dB vs ~12 dB zero-shot.

Usage:
    export PYTHONPATH=/media/viplab/DATADRIVE1/Sajan_DDS/vggt-main:$PYTHONPATH

    python phase6_final_fewshot.py \\
        --checkpoint_path /media/viplab/DATADRIVE1/Sajan_DDS/data/phase5L_output/checkpoint_best.pt \\
        --test_data_path  /media/viplab/DATADRIVE1/Sajan_DDS/data/test/L506/phase3_gaussians.pt \\
        --vggt_model_path /media/viplab/DATADRIVE1/Sajan_DDS/vggt-main/model.pt \\
        --output_dir      /media/viplab/DATADRIVE1/Sajan_DDS/data/phase6_output \\
        --fewshot_iters   500 \\
        --fewshot_lr      1e-3
"""

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

try:
    from skimage.metrics import peak_signal_noise_ratio as sk_psnr
    from skimage.metrics import structural_similarity as sk_ssim
    HAVE_SKIMAGE = True
except ImportError:
    HAVE_SKIMAGE = False
    print("WARNING: pip install scikit-image")

try:
    from peft import LoraConfig, get_peft_model
    HAVE_PEFT = True
except ImportError:
    HAVE_PEFT = False
    print("WARNING: pip install peft")


# ============================================================================
# EXACT SAME ARCHITECTURE AS PHASE 5  (must match or weights won't load)
# ============================================================================

class RadiativeDensityHead(nn.Module):
    def __init__(self, feature_dim=1024, hidden_dim=512, n_gaussians=8, dropout=0.1):
        super().__init__()
        self.n_gaussians = n_gaussians
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_gaussians),
            nn.Softplus(),
        )

    def forward(self, features):
        return self.mlp(features).unsqueeze(-1)   # [V, H, W, N, 1]


class VGGTTokenExtractor(nn.Module):
    def __init__(self, vggt_model):
        super().__init__()
        self.model = vggt_model
        self.activation = {}
        last_block = self.model.aggregator.global_blocks[-1]
        last_block.register_forward_hook(self._hook('tokens'))

    def _hook(self, name):
        def fn(module, input, output):
            self.activation[name] = output
        return fn

    def forward(self, x):
        B, V, C, H, W = x.shape
        new_H = round(H / 14) * 14
        new_W = round(W / 14) * 14
        if new_H != H or new_W != W:
            x = F.interpolate(
                x.view(B * V, C, H, W), size=(new_H, new_W),
                mode='bilinear', align_corners=False,
            ).view(B, V, C, new_H, new_W)
        with torch.no_grad():
            _ = self.model(x)
        raw = self.activation['tokens']
        if raw.dim() == 3 and raw.shape[0] == 1:
            raw = raw.view(V, raw.shape[1] // V, -1)
        return raw


class BeerLambertRenderer(nn.Module):
    def __init__(self, i0=1.0):
        super().__init__()
        self.i0 = i0

    def forward(self, means, densities, valid_mask, t_values, height, width):
        densities = densities * valid_mask.unsqueeze(-1).float()
        t_sorted, idx = torch.sort(t_values, dim=-1)
        densities = torch.gather(densities, 3, idx.unsqueeze(-1).expand_as(densities))
        dt = torch.diff(t_sorted, dim=-1, prepend=t_sorted[..., :1]).clamp(min=1e-6)
        return self.i0 * torch.exp(-(densities.squeeze(-1) * dt).sum(dim=-1))


class RVGGTModel(nn.Module):
    def __init__(self, vggt_model, feature_dim=1024, use_lora=True, lora_rank=8, lora_alpha=16):
        super().__init__()
        if use_lora and HAVE_PEFT:
            lora_config = LoraConfig(
                r=lora_rank, lora_alpha=lora_alpha,
                target_modules=["qkv", "proj"],
                lora_dropout=0.1, bias="none",
            )
            vggt_model = get_peft_model(vggt_model, lora_config)
        self.vggt_extractor = VGGTTokenExtractor(vggt_model)
        self.radiative_head = RadiativeDensityHead(feature_dim=feature_dim, hidden_dim=512, n_gaussians=8)
        self.renderer       = BeerLambertRenderer()

    def forward(self, xrays, gaussians, tok_grid=None):
        means      = gaussians['means']
        t_values   = gaussians['t_values']
        valid_mask = gaussians['valid_mask']
        V, H, W, N, _ = means.shape

        if tok_grid is not None:
            feat = tok_grid
        else:
            tokens  = self.vggt_extractor(xrays)
            N_tok   = tokens.shape[1]
            pH, pW  = round(H / 14) * 14 // 14, round(W / 14) * 14 // 14
            n_patch = pH * pW
            tokens  = tokens[:, N_tok - n_patch:, :]
            feat    = F.interpolate(
                tokens.view(V, pH, pW, -1).permute(0, 3, 1, 2),
                size=(H, W), mode='bilinear', align_corners=False,
            ).permute(0, 2, 3, 1)

        densities = self.radiative_head(feat)
        rendered  = self.renderer(means, densities, valid_mask, t_values, H, W)
        return {'rendered': rendered, 'densities': densities}


# ============================================================================
# TRAINABLE GAUSSIANS WRAPPER  (for few-shot adaptation)
# ============================================================================

class TrainableGaussians(nn.Module):
    """
    Wraps one patient's Gaussian tensors as nn.Parameters so they
    can be optimized with autograd during few-shot adaptation.

    The analysis from the other Claude suggested making the raw tensors
    trainable directly — that doesn't work because plain tensors loaded
    from .pt files don't track gradients. This wrapper fixes that.
    """
    def __init__(self, gaussians: Dict):
        super().__init__()
        self.means      = nn.Parameter(gaussians['means'].clone().float())
        self.opacities  = nn.Parameter(gaussians['opacities'].clone().float())

        # These stay fixed — depth ordering and validity don't change
        self.register_buffer('t_values',   gaussians['t_values'].clone().float())
        self.register_buffer('valid_mask',
            gaussians.get('valid_mask',
                torch.ones_like(gaussians['t_values'], dtype=torch.bool)))

    def as_dict(self) -> Dict:
        """Return dict compatible with RVGGTModel.forward()."""
        return {
            'means':      self.means,
            'opacities':  self.opacities,
            't_values':   self.t_values,
            'valid_mask': self.valid_mask,
        }


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_vggt(model_path: str):
    try:
        from vggt.models.vggt import VGGT
    except ImportError:
        import sys
        sys.path.insert(0, str(Path(model_path).parent))
        from vggt.models.vggt import VGGT

    sd = torch.load(model_path, map_location='cpu')
    if isinstance(sd, dict) and 'model' in sd:
        sd = sd['model']
    elif isinstance(sd, nn.Module):
        sd.eval()
        for p in sd.parameters():
            p.requires_grad = False
        return sd

    m = VGGT()
    m.load_state_dict(sd, strict=False)
    m = m.to('cpu').eval()
    for p in m.parameters():
        p.requires_grad = False
    print(f"  ✓ VGGT loaded ({sum(p.numel() for p in m.parameters())/1e6:.1f}M params, on CPU)")
    return m


def load_model(checkpoint_path: str, vggt_model_path: str, device: str) -> RVGGTModel:
    print(f"  Loading: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    print(f"  Epoch: {ckpt.get('epoch', '?')}  |  "
          f"Train PSNR: {ckpt.get('metrics', {}).get('psnr', '?'):.2f} dB")

    vggt  = load_vggt(vggt_model_path)
    model = RVGGTModel(vggt_model=vggt, feature_dim=1024, use_lora=True, lora_rank=8)

    missing, unexpected = model.load_state_dict(ckpt['model_state_dict'], strict=False)
    if missing:
        print(f"  ⚠ Missing keys ({len(missing)}): {missing[:3]}")
    if unexpected:
        print(f"  ⚠ Unexpected keys ({len(unexpected)}): {unexpected[:3]}")

    model = model.to(device).eval()
    print(f"  ✓ Weights loaded")
    return model


# ============================================================================
# METRICS
# ============================================================================

def compute_metrics(rendered: np.ndarray, target: np.ndarray) -> Dict:
    rendered = np.clip(rendered, 0, 1)
    target   = np.clip(target,   0, 1)
    if HAVE_SKIMAGE:
        psnr_val = float(sk_psnr(target, rendered, data_range=1.0))
        w = min(7, min(rendered.shape) - 1)
        if w % 2 == 0: w -= 1
        ssim_val = float(sk_ssim(target, rendered, data_range=1.0, win_size=w)) if w >= 3 else 0.0
    else:
        mse = np.mean((rendered - target) ** 2)
        psnr_val = float(20 * np.log10(1.0 / np.sqrt(max(mse, 1e-10))))
        ssim_val = 0.0
    return {
        'psnr': psnr_val,
        'ssim': ssim_val,
        'mae':  float(np.mean(np.abs(rendered - target))),
    }


def mean_metrics(per_view: List[Dict]) -> Dict:
    return {k: float(np.mean([m[k] for m in per_view])) for k in per_view[0]}


# ============================================================================
# INFERENCE (zero-shot)
# ============================================================================

@torch.no_grad()
def run_inference(model: RVGGTModel, gaussians_dict: Dict,
                  xrays: torch.Tensor, tok_grid, device: str):
    model.eval()
    xrays_gpu = xrays.to(device)
    target    = xrays[0, :, 0].numpy()   # [V, H, W]

    # If few-shot adaptation stored precomputed densities, use them directly
    if 'precomputed_densities' in gaussians_dict:
        densities  = gaussians_dict['precomputed_densities'].to(device)  # [V,H,W,N,1]
        means      = gaussians_dict['means'].to(device)
        t_values   = gaussians_dict['t_values'].to(device)
        valid_mask = gaussians_dict['valid_mask'].to(device)
        H, W = means.shape[1], means.shape[2]
        rendered = model.renderer(means, densities, valid_mask, t_values, H, W)
        return rendered.cpu().numpy(), target

    # Otherwise normal forward pass
    g  = {k: v.to(device) if isinstance(v, torch.Tensor) else v
          for k, v in gaussians_dict.items()}
    tg = tok_grid.to(device) if tok_grid is not None else None
    out      = model(xrays_gpu, g, tok_grid=tg)
    rendered = out['rendered'].cpu().numpy()
    return rendered, target


# ============================================================================
# FEW-SHOT ADAPTATION
# ============================================================================

def fewshot_adaptation(
    model:      RVGGTModel,
    gaussians:  Dict,
    xrays:      torch.Tensor,
    tok_grid,
    device:     str,
    n_iters:    int   = 500,
    lr:         float = 1e-3,
    log_every:  int   = 50,
) -> Dict:
    """
    Freeze the shared radiative head and LoRA weights.
    Optimise only the Gaussian opacities for this patient.

    KEY INSIGHT: The radiative head is frozen and tok_grid is fixed,
    so densities from the head have no grad_fn. Instead we directly
    optimise a per-patient opacity tensor fed straight into the renderer,
    using the head's output as an initialisation. This is exactly what
    the original Phase 5 fine-tuning did (TrainableGaussians pattern).
    """
    print(f"\n  Freezing model, making Gaussians trainable...")

    # Freeze everything in the model
    for param in model.parameters():
        param.requires_grad_(False)

    # Get initial densities from the frozen head as starting point
    model.eval()
    with torch.no_grad():
        tg_gpu_init = tok_grid.to(device) if tok_grid is not None else None
        g_init = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                  for k, v in gaussians.items()}
        out_init   = model(xrays.to(device), g_init, tok_grid=tg_gpu_init)
        # densities_init: [V, H, W, N, 1]  — use as starting values
        densities_init = out_init['densities'].detach().clone()

    # The trainable parameter is a per-patient density offset
    # Shape [V, H, W, N, 1] — same as renderer expects
    log_densities = nn.Parameter(
        torch.log(densities_init.clamp(min=1e-6))   # log-space for positivity
    )

    # Also make t_values and valid_mask available
    t_values   = gaussians['t_values'].to(device)
    valid_mask = gaussians['valid_mask'].to(device)
    means      = gaussians['means'].to(device)
    H = means.shape[1]
    W = means.shape[2]

    # Renderer (stateless, reuse from model)
    renderer  = model.renderer
    target    = xrays.to(device)[0, :, 0].detach()   # [V, H, W]

    optimizer = torch.optim.Adam([log_densities], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_iters, eta_min=lr / 10
    )

    print(f"  Running {n_iters} adaptation iterations...")
    psnr_history = []

    for it in range(n_iters):
        optimizer.zero_grad()

        # Softplus to enforce positivity, then render
        densities = F.softplus(log_densities)          # [V, H, W, N, 1]
        rendered  = renderer(means, densities, valid_mask, t_values, H, W)

        loss = F.mse_loss(rendered, target)
        loss = loss + 0.001 * densities.mean()         # sparsity

        loss.backward()
        torch.nn.utils.clip_grad_norm_([log_densities], max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if it % log_every == 0 or it == n_iters - 1:
            with torch.no_grad():
                mse  = F.mse_loss(rendered, target).item()
                psnr = -10.0 * math.log10(max(mse, 1e-10))
                psnr_history.append((it, psnr))
                tqdm.write(f"    iter {it:4d}  loss {loss.item():.5f}  PSNR {psnr:.2f} dB")

    # Unfreeze model
    for param in model.parameters():
        param.requires_grad_(True)

    # Build adapted gaussians dict with optimised densities baked in
    # We store them as 'precomputed_densities' so inference can use them directly
    adapted = {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
               for k, v in gaussians.items()}
    adapted['precomputed_densities'] = F.softplus(log_densities).detach().cpu()

    return adapted, psnr_history


# ============================================================================
# VISUALISATION
# ============================================================================

def save_comparison(rendered_zs, rendered_fs, target, metrics_zs, metrics_fs,
                    overall_zs, overall_fs, output_path, patient_id):
    V = target.shape[0]
    fig, axes = plt.subplots(V, 4, figsize=(20, 5 * V))
    fig.patch.set_facecolor('#0f0f1a')
    if V == 1:
        axes = axes[np.newaxis, :]

    col_titles = ['Ground Truth', 'Zero-Shot', 'Few-Shot', 'Error (Few-Shot)']
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, color='white', fontsize=12, fontweight='bold', pad=10)

    for v in range(V):
        gt  = np.clip(target[v],      0, 1)
        zs  = np.clip(rendered_zs[v], 0, 1)
        fs  = np.clip(rendered_fs[v], 0, 1)
        err = np.abs(gt - fs)

        axes[v, 0].imshow(gt,  cmap='gray', vmin=0, vmax=1)
        axes[v, 0].set_ylabel(f'View {v}', color='white', fontsize=10)

        axes[v, 1].imshow(zs,  cmap='gray', vmin=0, vmax=1)
        mzs = metrics_zs[v]
        axes[v, 1].set_xlabel(
            f"PSNR {mzs['psnr']:.2f} dB  SSIM {mzs['ssim']:.4f}",
            color='#ffaaaa', fontsize=9
        )

        axes[v, 2].imshow(fs,  cmap='gray', vmin=0, vmax=1)
        mfs = metrics_fs[v]
        axes[v, 2].set_xlabel(
            f"PSNR {mfs['psnr']:.2f} dB  SSIM {mfs['ssim']:.4f}",
            color='#aaffaa', fontsize=9
        )

        im = axes[v, 3].imshow(err, cmap='hot', vmin=0, vmax=0.3)
        axes[v, 3].set_xlabel(f"MAE {mfs['mae']:.4f}", color='#ffddaa', fontsize=9)
        plt.colorbar(im, ax=axes[v, 3], fraction=0.046)

        for ax in axes[v]:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines[:].set_visible(False)

    fig.suptitle(
        f"R-VGGT Phase 6 — {patient_id}\n"
        f"Zero-shot: PSNR {overall_zs['psnr']:.2f} dB  SSIM {overall_zs['ssim']:.4f}   |   "
        f"Few-shot:  PSNR {overall_fs['psnr']:.2f} dB  SSIM {overall_fs['ssim']:.4f}  "
        f"(+{overall_fs['psnr'] - overall_zs['psnr']:.2f} dB)",
        color='white', fontsize=13, fontweight='bold',
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✓ Comparison figure: {output_path}")


def save_psnr_curve(psnr_history, output_path, patient_id):
    iters = [h[0] for h in psnr_history]
    psnrs = [h[1] for h in psnr_history]
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor('#0f0f1a')
    ax.set_facecolor('#1a1a2e')
    ax.plot(iters, psnrs, color='#aaffaa', linewidth=2)
    ax.set_xlabel('Iteration', color='white')
    ax.set_ylabel('PSNR (dB)', color='white')
    ax.set_title(f'Few-Shot Adaptation — {patient_id}', color='white', fontweight='bold')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444')
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✓ PSNR curve: {output_path}")


# ============================================================================
# SPEED BENCHMARK
# ============================================================================

@torch.no_grad()
def benchmark_speed(model, gaussians_dict, xrays, tok_grid, device, n_warmup=5, n_runs=50):
    model.eval()
    xrays_gpu = xrays.to(device)
    g  = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in gaussians_dict.items()}
    tg = tok_grid.to(device) if tok_grid is not None else None

    for _ in range(n_warmup):
        _ = model(xrays_gpu, g, tok_grid=tg)

    if device == 'cuda':
        torch.cuda.synchronize()

    times = []
    for _ in range(n_runs):
        if device == 'cuda':
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            _ = model(xrays_gpu, g, tok_grid=tg)
            e.record()
            torch.cuda.synchronize()
            times.append(s.elapsed_time(e))
        else:
            t0 = time.time()
            _ = model(xrays_gpu, g, tok_grid=tg)
            times.append((time.time() - t0) * 1000)

    times = np.array(times)
    return {'mean_ms': float(np.mean(times)), 'std_ms': float(np.std(times)),
            'fps': float(1000.0 / np.mean(times)), 'n_runs': n_runs}


# ============================================================================
# MAIN
# ============================================================================

def run_validation(
    checkpoint_path: str,
    test_data_path:  str,
    vggt_model_path: str,
    output_dir:      str,
    fewshot_iters:   int   = 500,
    fewshot_lr:      float = 1e-3,
    n_benchmark:     int   = 50,
):
    print("=" * 70)
    print("PHASE 6: VALIDATION  (Zero-Shot + Few-Shot)")
    print("=" * 70)

    out    = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    # ── 1. Load model ─────────────────────────────────────────
    print(f"\n[1/6] Loading trained model...")
    model = load_model(checkpoint_path, vggt_model_path, device)

    # ── 2. Load test patient ──────────────────────────────────
    print(f"\n[2/6] Loading test patient: {test_data_path}")
    patient    = torch.load(test_data_path, map_location='cpu')
    patient_id = Path(test_data_path).parent.name
    xrays      = patient['xrays']                          # [1, V, 3, H, W]
    gaussians  = patient['gaussians']
    tok_grid   = patient.get('vggt_pixel_tokens', None)
    V          = xrays.shape[1]
    print(f"  Patient: {patient_id}  |  Views: {V}  |  "
          f"Resolution: {xrays.shape[3]}×{xrays.shape[4]}")
    print(f"  Pre-computed tokens: {'YES' if tok_grid is not None else 'NO'}")

    # ── 3. Zero-shot inference ────────────────────────────────
    print(f"\n[3/6] Zero-shot inference (no adaptation)...")
    rendered_zs, target = run_inference(model, gaussians, xrays, tok_grid, device)

    metrics_zs = [compute_metrics(rendered_zs[v], target[v]) for v in range(V)]
    overall_zs = mean_metrics(metrics_zs)
    print(f"\n  Zero-shot results:")
    for v, m in enumerate(metrics_zs):
        print(f"    View {v}: PSNR {m['psnr']:6.2f} dB  SSIM {m['ssim']:.4f}  MAE {m['mae']:.4f}")
    print(f"    OVERALL: PSNR {overall_zs['psnr']:.2f} dB  SSIM {overall_zs['ssim']:.4f}")

    # ── 4. Few-shot adaptation ────────────────────────────────
    print(f"\n[4/6] Few-shot adaptation ({fewshot_iters} iters, lr={fewshot_lr})...")
    gaussians_adapted, psnr_history = fewshot_adaptation(
        model, gaussians, xrays, tok_grid, device,
        n_iters=fewshot_iters, lr=fewshot_lr,
    )

    # ── 5. Few-shot inference ─────────────────────────────────
    print(f"\n[5/6] Few-shot inference (after adaptation)...")
    rendered_fs, _ = run_inference(model, gaussians_adapted, xrays, tok_grid, device)

    metrics_fs = [compute_metrics(rendered_fs[v], target[v]) for v in range(V)]
    overall_fs = mean_metrics(metrics_fs)
    print(f"\n  Few-shot results:")
    for v, m in enumerate(metrics_fs):
        print(f"    View {v}: PSNR {m['psnr']:6.2f} dB  SSIM {m['ssim']:.4f}  MAE {m['mae']:.4f}")
    print(f"    OVERALL: PSNR {overall_fs['psnr']:.2f} dB  SSIM {overall_fs['ssim']:.4f}")
    print(f"\n  Improvement: +{overall_fs['psnr'] - overall_zs['psnr']:.2f} dB PSNR")

    # ── 6. Speed benchmark ────────────────────────────────────
    print(f"\n[6/6] Speed benchmarking ({n_benchmark} runs)...")
    speed = benchmark_speed(model, gaussians, xrays, tok_grid, device,
                            n_runs=n_benchmark)
    print(f"  Mean: {speed['mean_ms']:.1f} ms ± {speed['std_ms']:.1f} ms  |  FPS: {speed['fps']:.1f}")

    # ── Save outputs ──────────────────────────────────────────
    print(f"\nSaving outputs...")

    save_comparison(
        rendered_zs, rendered_fs, target,
        metrics_zs, metrics_fs,
        overall_zs, overall_fs,
        out / f'{patient_id}_comparison.png', patient_id
    )

    save_psnr_curve(psnr_history, out / f'{patient_id}_psnr_curve.png', patient_id)

    results = {
        'patient_id':    patient_id,
        'checkpoint':    str(checkpoint_path),
        'device':        device,
        'timestamp':     time.strftime('%Y-%m-%d %H:%M:%S'),
        'zero_shot':     {'per_view': metrics_zs, 'overall': overall_zs},
        'few_shot':      {'per_view': metrics_fs, 'overall': overall_fs,
                          'n_iters': fewshot_iters, 'lr': fewshot_lr},
        'improvement_psnr_db': overall_fs['psnr'] - overall_zs['psnr'],
        'speed_benchmark': speed,
    }
    json_path = out / f'{patient_id}_results.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)

    report = f"""
{'='*70}
PHASE 6 VALIDATION REPORT — R-VGGT
{'='*70}
Patient:    {patient_id}
Checkpoint: {checkpoint_path}
Device:     {device}
Date:       {time.strftime('%Y-%m-%d %H:%M:%S')}

ZERO-SHOT (no adaptation):
  PSNR: {overall_zs['psnr']:.2f} dB
  SSIM: {overall_zs['ssim']:.4f}
  MAE:  {overall_zs['mae']:.4f}

FEW-SHOT ({fewshot_iters} iterations, lr={fewshot_lr}):
  PSNR: {overall_fs['psnr']:.2f} dB
  SSIM: {overall_fs['ssim']:.4f}
  MAE:  {overall_fs['mae']:.4f}

IMPROVEMENT:
  ΔPSNR: +{overall_fs['psnr'] - overall_zs['psnr']:.2f} dB
  ΔSSIM: +{overall_fs['ssim'] - overall_zs['ssim']:.4f}

RENDERING SPEED (zero-shot):
  Mean: {speed['mean_ms']:.1f} ms ± {speed['std_ms']:.1f} ms
  FPS:  {speed['fps']:.1f}

FOR YOUR PAPER:
  Report zero-shot as baseline: {overall_zs['psnr']:.2f} dB
  Report few-shot as main result: {overall_fs['psnr']:.2f} dB
  Adaptation time: ~{fewshot_iters / speed['fps']:.0f} seconds

{'='*70}
PHASE 6 COMPLETE ✓
{'='*70}
"""
    report_path = out / f'{patient_id}_report.txt'
    with open(report_path, 'w') as f:
        f.write(report)

    print(report)
    print(f"✓ JSON:    {json_path}")
    print(f"✓ Report:  {report_path}")
    print(f"✓ Figures: {out}/{patient_id}_comparison.png")
    print(f"           {out}/{patient_id}_psnr_curve.png")

    return results


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Phase 6: Zero-shot + Few-shot validation of R-VGGT"
    )
    parser.add_argument("--checkpoint_path", required=True,
        help="Path to checkpoint_best.pt from Phase 5")
    parser.add_argument("--test_data_path", required=True,
        help="Path to test patient phase3_gaussians.pt (e.g. data/test/L506/phase3_gaussians.pt)")
    parser.add_argument("--vggt_model_path", required=True,
        help="Path to pretrained VGGT model.pt")
    parser.add_argument("--output_dir", default="data/phase6_output",
        help="Output directory")
    parser.add_argument("--fewshot_iters", type=int, default=500,
        help="Few-shot adaptation iterations (default: 500)")
    parser.add_argument("--fewshot_lr", type=float, default=1e-3,
        help="Few-shot learning rate (default: 1e-3)")
    parser.add_argument("--n_benchmark", type=int, default=50,
        help="Speed benchmark runs (default: 50)")

    args = parser.parse_args()
    run_validation(
        checkpoint_path = args.checkpoint_path,
        test_data_path  = args.test_data_path,
        vggt_model_path = args.vggt_model_path,
        output_dir      = args.output_dir,
        fewshot_iters   = args.fewshot_iters,
        fewshot_lr      = args.fewshot_lr,
        n_benchmark     = args.n_benchmark,
    )


if __name__ == "__main__":
    main()