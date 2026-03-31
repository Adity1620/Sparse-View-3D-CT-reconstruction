"""
R-VGGT Phase 6: Validation & Evaluation
========================================
Matched exactly to phase5_friend_corrected.py architecture.

What this does (all REAL, no placeholders):
  1. Loads checkpoint_best.pt from Phase 5
  2. Rebuilds the exact same RVGGTModel architecture
  3. Runs on test patient (L506) — never seen during training
  4. Computes real PSNR / SSIM on all 4 views
  5. Saves side-by-side comparison figures
  6. Benchmarks rendering speed
  7. Saves full validation report

Usage:
    export PYTHONPATH=/media/viplab/DATADRIVE1/Sajan_DDS/vggt-main:$PYTHONPATH

    python phase6_validation_final.py \\
        --checkpoint_path /media/viplab/DATADRIVE1/Sajan_DDS/data/phase5L_output/checkpoint_best.pt \\
        --test_data_path  /media/viplab/DATADRIVE1/Sajan_DDS/data/test/L506/phase3_gaussians.pt \\
        --vggt_model_path /media/viplab/DATADRIVE1/Sajan_DDS/vggt-main/model.pt \\
        --output_dir      /media/viplab/DATADRIVE1/Sajan_DDS/data/phase6_output
"""

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from skimage.metrics import peak_signal_noise_ratio as sk_psnr
    from skimage.metrics import structural_similarity as sk_ssim
    HAVE_SKIMAGE = True
except ImportError:
    HAVE_SKIMAGE = False
    print("WARNING: scikit-image not available. Install: pip install scikit-image")

try:
    import lpips
    HAVE_LPIPS = True
except ImportError:
    HAVE_LPIPS = False

try:
    from peft import LoraConfig, get_peft_model, PeftModel
    HAVE_PEFT = True
except ImportError:
    HAVE_PEFT = False
    print("WARNING: peft not available. Install: pip install peft")


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
        out = self.mlp(features)           # [V, H, W, N]
        return out.unsqueeze(-1)           # [V, H, W, N, 1]


class VGGTTokenExtractor(nn.Module):
    def __init__(self, vggt_model):
        super().__init__()
        self.model = vggt_model
        self.activation = {}
        last_block = self.model.aggregator.global_blocks[-1]
        last_block.register_forward_hook(self._get_activation('tokens'))

    def _get_activation(self, name):
        def hook(module, input, output):
            self.activation[name] = output
        return hook

    def forward(self, x):
        B, V, C, H, W = x.shape
        new_H = round(H / 14) * 14
        new_W = round(W / 14) * 14
        if new_H != H or new_W != W:
            x_vggt = F.interpolate(
                x.view(B * V, C, H, W), size=(new_H, new_W),
                mode='bilinear', align_corners=False,
            ).view(B, V, C, new_H, new_W)
        else:
            x_vggt = x
        with torch.no_grad():
            _ = self.model(x_vggt)
        raw_tokens = self.activation['tokens']
        if raw_tokens.dim() == 3 and raw_tokens.shape[0] == 1:
            tokens_per_view = raw_tokens.shape[1] // V
            tokens = raw_tokens.view(V, tokens_per_view, -1)
        else:
            tokens = raw_tokens
        return tokens


class BeerLambertRenderer(nn.Module):
    def __init__(self, i0=1.0):
        super().__init__()
        self.i0 = i0

    def forward(self, means, densities, valid_mask, t_values, height, width):
        densities = densities * valid_mask.unsqueeze(-1).float()
        t_sorted, indices = torch.sort(t_values, dim=-1)
        densities_sorted = torch.gather(
            densities, dim=3,
            index=indices.unsqueeze(-1).expand_as(densities)
        )
        dt = torch.diff(t_sorted, dim=-1, prepend=t_sorted[..., :1]).clamp(min=1e-6)
        attenuation = (densities_sorted.squeeze(-1) * dt).sum(dim=-1)
        return self.i0 * torch.exp(-attenuation)


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
        self.radiative_head = RadiativeDensityHead(
            feature_dim=feature_dim, hidden_dim=512, n_gaussians=8,
        )
        self.renderer = BeerLambertRenderer()

    def forward(self, xrays, gaussians, camera_params, tok_grid=None):
        means      = gaussians['means']
        t_values   = gaussians['t_values']
        valid_mask = gaussians['valid_mask']
        V, H, W, N, _ = means.shape

        if tok_grid is not None:
            features_pixel = tok_grid
        else:
            features = self.vggt_extractor(xrays)
            N_tokens  = features.shape[1]
            new_H = round(H / 14) * 14
            new_W = round(W / 14) * 14
            n_patch = (new_H // 14) * (new_W // 14)
            n_extra = N_tokens - n_patch
            patch_tokens = features[:, n_extra:, :]
            features_grid = patch_tokens.view(V, new_H // 14, new_W // 14, -1)
            features_pixel = F.interpolate(
                features_grid.permute(0, 3, 1, 2), size=(H, W),
                mode='bilinear', align_corners=False,
            ).permute(0, 2, 3, 1)

        densities = self.radiative_head(features_pixel)
        rendered  = self.renderer(means, densities, valid_mask, t_values, H, W)
        return {'rendered': rendered, 'densities': densities, 'features': features_pixel}


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_vggt(model_path: str):
    """Load VGGT architecture and weights from model.pt (state_dict)."""
    try:
        from vggt.models.vggt import VGGT
    except ImportError:
        import sys
        sys.path.insert(0, str(Path(model_path).parent))
        from vggt.models.vggt import VGGT

    state_dict = torch.load(model_path, map_location='cpu')
    if isinstance(state_dict, dict) and 'model' in state_dict:
        state_dict = state_dict['model']
    elif isinstance(state_dict, nn.Module):
        state_dict.eval()
        for p in state_dict.parameters():
            p.requires_grad = False
        return state_dict

    vggt_model = VGGT()
    vggt_model.load_state_dict(state_dict, strict=False)
    vggt_model = vggt_model.to('cpu').eval()
    for p in vggt_model.parameters():
        p.requires_grad = False
    print(f"  ✓ VGGT loaded ({sum(p.numel() for p in vggt_model.parameters())/1e6:.1f}M params)")
    return vggt_model


def load_model_from_checkpoint(checkpoint_path: str, vggt_model_path: str, device: str):
    """
    Reconstruct the full RVGGTModel and load trained weights from checkpoint.
    """
    print(f"  Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location='cpu')

    print(f"  Checkpoint epoch: {ckpt.get('epoch', 'unknown')}")
    if 'metrics' in ckpt:
        m = ckpt['metrics']
        print(f"  Checkpoint metrics: "
              f"loss={m.get('loss_recon', '?'):.4f}  "
              f"PSNR={m.get('psnr', '?'):.2f} dB")

    # Build architecture
    vggt_model = load_vggt(vggt_model_path)
    model = RVGGTModel(vggt_model=vggt_model, feature_dim=1024,
                       use_lora=True, lora_rank=8)

    # Load trained weights
    missing, unexpected = model.load_state_dict(ckpt['model_state_dict'], strict=False)
    if missing:
        print(f"  ⚠ Missing keys ({len(missing)}): {missing[:3]}")
    if unexpected:
        print(f"  ⚠ Unexpected keys ({len(unexpected)}): {unexpected[:3]}")

    model = model.to(device).eval()
    print(f"  ✓ Model loaded and ready for inference")
    return model


# ============================================================================
# METRICS
# ============================================================================

def compute_metrics(rendered: np.ndarray, target: np.ndarray) -> Dict:
    """Compute PSNR and SSIM between two [H, W] images in [0, 1]."""
    rendered = np.clip(rendered, 0, 1)
    target   = np.clip(target,   0, 1)

    if HAVE_SKIMAGE:
        psnr_val = float(sk_psnr(target, rendered, data_range=1.0))
        w = min(7, min(rendered.shape) - 1)
        if w % 2 == 0:
            w -= 1
        ssim_val = float(sk_ssim(target, rendered, data_range=1.0, win_size=w)) if w >= 3 else 0.0
    else:
        mse = np.mean((rendered - target) ** 2)
        psnr_val = float(20 * np.log10(1.0 / np.sqrt(max(mse, 1e-10))))
        ssim_val = 0.0

    mae_val = float(np.mean(np.abs(rendered - target)))
    return {'psnr': psnr_val, 'ssim': ssim_val, 'mae': mae_val}


# ============================================================================
# INFERENCE ON ONE PATIENT
# ============================================================================

@torch.no_grad()
def run_inference(model: RVGGTModel, patient: Dict, device: str):
    """
    Run model inference on one patient.
    Returns rendered [V, H, W] and target [V, H, W] as numpy arrays.
    """
    model.eval()

    xrays     = patient['xrays'].to(device)       # [1, V, 3, H, W]
    gaussians = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in patient['gaussians'].items()
    }
    tok_grid = patient.get('vggt_pixel_tokens', None)
    if tok_grid is not None:
        tok_grid = tok_grid.to(device)

    outputs  = model(xrays, gaussians, {}, tok_grid=tok_grid)
    rendered = outputs['rendered'].cpu().numpy()   # [V, H, W]
    target   = xrays[0, :, 0].cpu().numpy()       # [V, H, W]

    return rendered, target


# ============================================================================
# VISUALISATION
# ============================================================================

def save_comparison_figure(rendered, target, metrics_per_view, overall, output_path, patient_id):
    """
    Save side-by-side GT vs Rendered vs Error for all 4 views.
    """
    V = rendered.shape[0]
    fig = plt.figure(figsize=(18, 5 * V))
    fig.patch.set_facecolor('#0f0f1a')

    for v in range(V):
        gt  = np.clip(target[v],   0, 1)
        rnd = np.clip(rendered[v], 0, 1)
        err = np.abs(gt - rnd)
        m   = metrics_per_view[v]

        # GT
        ax = fig.add_subplot(V, 3, v * 3 + 1)
        ax.imshow(gt, cmap='gray', vmin=0, vmax=1)
        ax.set_title(f'View {v} — Ground Truth', color='white', fontsize=11)
        ax.axis('off')

        # Rendered
        ax = fig.add_subplot(V, 3, v * 3 + 2)
        ax.imshow(rnd, cmap='gray', vmin=0, vmax=1)
        ax.set_title(
            f'View {v} — Rendered\nPSNR {m["psnr"]:.2f} dB  SSIM {m["ssim"]:.4f}',
            color='#aaffaa', fontsize=11
        )
        ax.axis('off')

        # Error map
        ax = fig.add_subplot(V, 3, v * 3 + 3)
        im = ax.imshow(err, cmap='hot', vmin=0, vmax=0.3)
        ax.set_title(f'View {v} — Abs Error  MAE {m["mae"]:.4f}', color='#ffaaaa', fontsize=11)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle(
        f'R-VGGT Phase 6 — {patient_id}\n'
        f'Overall PSNR: {overall["psnr"]:.2f} dB  |  SSIM: {overall["ssim"]:.4f}  |  MAE: {overall["mae"]:.4f}',
        color='white', fontsize=14, fontweight='bold', y=1.01
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✓ Comparison figure saved: {output_path}")


def save_rendered_grid(rendered, target, output_path, patient_id):
    """Save compact 2-row grid: top=GT, bottom=rendered."""
    V = rendered.shape[0]
    fig, axes = plt.subplots(2, V, figsize=(5 * V, 10))
    fig.patch.set_facecolor('#0f0f1a')

    for v in range(V):
        axes[0, v].imshow(np.clip(target[v],   0, 1), cmap='gray', vmin=0, vmax=1)
        axes[0, v].set_title(f'GT View {v}', color='white', fontsize=10)
        axes[0, v].axis('off')

        axes[1, v].imshow(np.clip(rendered[v], 0, 1), cmap='gray', vmin=0, vmax=1)
        axes[1, v].set_title(f'Rendered View {v}', color='#aaffaa', fontsize=10)
        axes[1, v].axis('off')

    fig.suptitle(f'R-VGGT — {patient_id}  (top: GT, bottom: rendered)',
                 color='white', fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✓ Grid figure saved: {output_path}")


# ============================================================================
# SPEED BENCHMARK
# ============================================================================

@torch.no_grad()
def benchmark_speed(model, patient, device, n_warmup=5, n_runs=50):
    """Measure mean inference time in milliseconds."""
    model.eval()
    xrays     = patient['xrays'].to(device)
    gaussians = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in patient['gaussians'].items()}
    tok_grid  = patient.get('vggt_pixel_tokens', None)
    if tok_grid is not None:
        tok_grid = tok_grid.to(device)

    # Warmup
    for _ in range(n_warmup):
        _ = model(xrays, gaussians, {}, tok_grid=tok_grid)

    torch.cuda.synchronize() if device == 'cuda' else None

    times = []
    for _ in range(n_runs):
        if device == 'cuda':
            start = torch.cuda.Event(enable_timing=True)
            end   = torch.cuda.Event(enable_timing=True)
            start.record()
            _ = model(xrays, gaussians, {}, tok_grid=tok_grid)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))   # ms
        else:
            t0 = time.time()
            _ = model(xrays, gaussians, {}, tok_grid=tok_grid)
            times.append((time.time() - t0) * 1000)

    times = np.array(times)
    return {
        'mean_ms':  float(np.mean(times)),
        'std_ms':   float(np.std(times)),
        'fps':      float(1000.0 / np.mean(times)),
        'n_runs':   n_runs,
    }


# ============================================================================
# MAIN VALIDATION PIPELINE
# ============================================================================

def run_validation(
    checkpoint_path: str,
    test_data_path:  str,
    vggt_model_path: str,
    output_dir:      str,
    n_warmup:        int = 5,
    n_benchmark:     int = 50,
):
    print("=" * 70)
    print("PHASE 6: VALIDATION & EVALUATION")
    print("=" * 70)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    # ── 1. Load model ─────────────────────────────────────────
    print(f"\n[1/5] Loading trained model...")
    model = load_model_from_checkpoint(checkpoint_path, vggt_model_path, device)

    # ── 2. Load test patient data ─────────────────────────────
    print(f"\n[2/5] Loading test patient data: {test_data_path}")
    patient = torch.load(test_data_path, map_location='cpu')
    patient_id = Path(test_data_path).parent.name

    xrays = patient['xrays']   # [1, V, 3, H, W]
    V = xrays.shape[1]
    H = xrays.shape[3]
    W = xrays.shape[4]
    has_tokens = 'vggt_pixel_tokens' in patient
    print(f"  Patient:     {patient_id}")
    print(f"  Views:       {V}")
    print(f"  Resolution:  {H}×{W}")
    print(f"  Precomputed tokens: {'YES (fast path)' if has_tokens else 'NO (live VGGT inference)'}")

    # ── 3. Run inference & compute metrics ────────────────────
    print(f"\n[3/5] Running inference and computing metrics...")
    rendered, target = run_inference(model, patient, device)

    metrics_per_view = []
    for v in range(V):
        m = compute_metrics(rendered[v], target[v])
        metrics_per_view.append(m)
        print(f"  View {v}: PSNR {m['psnr']:6.2f} dB  |  SSIM {m['ssim']:.4f}  |  MAE {m['mae']:.4f}")

    overall = {
        'psnr': float(np.mean([m['psnr'] for m in metrics_per_view])),
        'ssim': float(np.mean([m['ssim'] for m in metrics_per_view])),
        'mae':  float(np.mean([m['mae']  for m in metrics_per_view])),
    }
    print(f"\n  ── OVERALL ──────────────────────────────────────")
    print(f"  PSNR: {overall['psnr']:.2f} dB")
    print(f"  SSIM: {overall['ssim']:.4f}")
    print(f"  MAE:  {overall['mae']:.4f}")

    # ── 4. Speed benchmark ────────────────────────────────────
    print(f"\n[4/5] Speed benchmarking ({n_benchmark} runs)...")
    speed = benchmark_speed(model, patient, device, n_warmup, n_benchmark)
    print(f"  Mean: {speed['mean_ms']:.1f} ms  ±  {speed['std_ms']:.1f} ms")
    print(f"  FPS:  {speed['fps']:.1f}")

    # ── 5. Save outputs ───────────────────────────────────────
    print(f"\n[5/5] Saving outputs to: {out}")

    # Comparison figure (GT vs rendered vs error, all views)
    save_comparison_figure(
        rendered, target, metrics_per_view, overall,
        out / f'{patient_id}_comparison.png', patient_id
    )

    # Compact grid
    save_rendered_grid(
        rendered, target,
        out / f'{patient_id}_grid.png', patient_id
    )

    # JSON results
    results = {
        'patient_id':       patient_id,
        'checkpoint':       str(checkpoint_path),
        'device':           device,
        'timestamp':        time.strftime('%Y-%m-%d %H:%M:%S'),
        'overall_metrics':  overall,
        'per_view_metrics': metrics_per_view,
        'speed_benchmark':  speed,
    }
    json_path = out / f'{patient_id}_results.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Text report
    report_lines = [
        "=" * 70,
        "PHASE 6 VALIDATION REPORT — R-VGGT",
        "=" * 70,
        f"Patient:    {patient_id}",
        f"Checkpoint: {checkpoint_path}",
        f"Device:     {device}",
        f"Date:       {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "RECONSTRUCTION QUALITY (per view):",
    ]
    for v, m in enumerate(metrics_per_view):
        report_lines.append(
            f"  View {v}: PSNR {m['psnr']:6.2f} dB  |  SSIM {m['ssim']:.4f}  |  MAE {m['mae']:.4f}"
        )
    report_lines += [
        "",
        "OVERALL:",
        f"  PSNR: {overall['psnr']:.2f} dB",
        f"  SSIM: {overall['ssim']:.4f}",
        f"  MAE:  {overall['mae']:.4f}",
        "",
        "RENDERING SPEED:",
        f"  Mean: {speed['mean_ms']:.1f} ms  ±  {speed['std_ms']:.1f} ms",
        f"  FPS:  {speed['fps']:.1f}",
        "",
        "=" * 70,
        "PHASE 6 COMPLETE ✓",
        "=" * 70,
    ]
    report = "\n".join(report_lines)
    report_path = out / f'{patient_id}_report.txt'
    with open(report_path, 'w') as f:
        f.write(report)

    print("\n" + report)
    print(f"\n✓ JSON results: {json_path}")
    print(f"✓ Text report:  {report_path}")
    print(f"✓ Figures:      {out}/{patient_id}_comparison.png")

    return results


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Phase 6: Validate trained R-VGGT model (matched to phase5_friend_corrected.py)"
    )
    parser.add_argument(
        "--checkpoint_path", type=str, required=True,
        help="Path to checkpoint_best.pt from Phase 5"
    )
    parser.add_argument(
        "--test_data_path", type=str, required=True,
        help="Path to test patient's phase3_gaussians.pt (e.g. data/test/L506/phase3_gaussians.pt)"
    )
    parser.add_argument(
        "--vggt_model_path", type=str, required=True,
        help="Path to pretrained VGGT model.pt"
    )
    parser.add_argument(
        "--output_dir", type=str, default="data/phase6_output",
        help="Directory to save results, figures, and report"
    )
    parser.add_argument(
        "--n_benchmark", type=int, default=50,
        help="Number of speed benchmark runs (default: 50)"
    )

    args = parser.parse_args()

    run_validation(
        checkpoint_path = args.checkpoint_path,
        test_data_path  = args.test_data_path,
        vggt_model_path = args.vggt_model_path,
        output_dir      = args.output_dir,
        n_benchmark     = args.n_benchmark,
    )


if __name__ == "__main__":
    main()
