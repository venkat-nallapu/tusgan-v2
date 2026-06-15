# -*- coding: utf-8 -*-
"""
TUS-GAN — Evaluation Script (v2)
==================================
Loads a trained Generator from a checkpoint, generates synthetic 9-channel
daily activity diaries, and compares them against the real ITUS 2019 dataset
using statistical metrics and visualisations.

Outputs
-------
  • Per-division frequency comparison (real vs synthetic)
  • Jensen-Shannon Divergence (JSD) between activity distributions
  • Average minutes per major activity division
  • Bar charts, heatmaps, and step-plot diary visualisations
"""

import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon


# ─────────────────────────────────────────────────────────────
# 1. CONSTANTS & ACTIVITY MAPPING
# ─────────────────────────────────────────────────────────────

DIVISION_LABELS = {
    1: "Employment",
    2: "Production",
    3: "Unpaid Domestic",
    4: "Unpaid Caregiving",
    5: "Unpaid Volunteer",
    6: "Learning",
    7: "Socializing & Religious",
    8: "Leisure & Sports",
    9: "Self-care & Maintenance",
}

MINUTES_PER_SLOT = 30  # Each of the 48 time-slots spans 30 minutes


# ─────────────────────────────────────────────────────────────
# 2. HELPER — LOAD GENERATOR FROM CHECKPOINT
# ─────────────────────────────────────────────────────────────

def load_generator(checkpoint_path: str, data_path: str, device: torch.device):
    """
    Reconstruct the Generator from a saved checkpoint and load its weights.

    Uses the real dataset NPZ to determine the correct cond_dim,
    num_districts, and num_states (since the checkpoint config may
    have stale defaults for these values).

    Parameters
    ----------
    checkpoint_path : str
        Path to a `.pt` file saved by ``train.py``.
    data_path : str
        Path to the real-data `.npz` file (used for accurate dimensions).
    device : torch.device

    Returns
    -------
    G : Generator
        The model in eval mode on *device*.
    cfg : dict
        The training config dict stored alongside the checkpoint.
    """
    from generator import Generator

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    # Read accurate dimensions from the real dataset (checkpoint config
    # may have stale defaults if cond_dim was not updated before saving).
    data = np.load(data_path)
    actual_cond_dim = data["cond_vector"].shape[1]
    actual_num_districts = int(data["num_districts"])
    actual_num_states = int(data["num_states"])

    G = Generator(
        noise_dim=cfg["noise_dim"],
        cond_dim=actual_cond_dim,
        num_districts=actual_num_districts,
        num_states=actual_num_states,
        district_embed_dim=cfg["district_embed_dim"],
        state_embed_dim=cfg["state_embed_dim"],
        base_channels=cfg["g_base_channels"],
    ).to(device)

    G.load_state_dict(ckpt["G_state"])
    G.eval()

    # Update config with accurate values for downstream use
    cfg["cond_dim"] = actual_cond_dim
    cfg["num_districts"] = actual_num_districts
    cfg["num_states"] = actual_num_states

    epoch = ckpt.get("epoch", "?")
    print(f"✅ Generator loaded from {checkpoint_path}  (epoch {epoch})")
    print(f"   cond_dim={actual_cond_dim}, districts={actual_num_districts}, states={actual_num_states}")
    return G, cfg


# ─────────────────────────────────────────────────────────────
# 3. HELPER — LOAD REAL DATASET
# ─────────────────────────────────────────────────────────────

def load_real_data(data_path: str):
    """
    Load the pre-encoded ITUS 2019 dataset.

    Returns
    -------
    diary_tensor : np.ndarray  (N, 9, 48, 1)
    cond_vector  : np.ndarray  (N, 83)
    district_ids : np.ndarray  (N,)
    state_ids    : np.ndarray  (N,)
    """
    if not os.path.exists(data_path):
        alt = os.path.join("wgan-gp", data_path)
        if os.path.exists(alt):
            data_path = alt

    data = np.load(data_path)
    print(f"✅ Real data loaded from {data_path}  ({data['diary_tensor'].shape[0]:,} diaries)")
    return (
        data["diary_tensor"],   # (N, 9, 48, 1)
        data["cond_vector"],    # (N, 83)
        data["district_ids"],   # (N,)
        data["state_ids"],      # (N,)
    )


# ─────────────────────────────────────────────────────────────
# 4. HELPER — GENERATE SYNTHETIC DIARIES
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_synthetic(
    G,
    cond_vector: np.ndarray,
    district_ids: np.ndarray,
    state_ids: np.ndarray,
    noise_dim: int,
    n_samples: int,
    device: torch.device,
    batch_size: int = 512,
) -> np.ndarray:
    """
    Generate *n_samples* synthetic diaries using conditioning vectors drawn
    (with replacement) from the real dataset.

    Returns
    -------
    fake_tensor : np.ndarray  (n_samples, 9, 48, 1)
    """
    N = cond_vector.shape[0]
    indices = np.random.choice(N, size=n_samples, replace=True)

    cond_all = torch.from_numpy(cond_vector[indices]).float()
    dist_all = torch.from_numpy(district_ids[indices]).long()
    state_all = torch.from_numpy(state_ids[indices]).long()

    all_fakes = []
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        z = torch.randn(end - start, noise_dim, device=device)
        c = cond_all[start:end].to(device)
        d = dist_all[start:end].to(device)
        s = state_all[start:end].to(device)
        fake = G(z, c, d, s)
        all_fakes.append(fake.cpu().numpy())

    fake_tensor = np.concatenate(all_fakes, axis=0)
    print(f"✅ Generated {fake_tensor.shape[0]:,} synthetic diaries  {fake_tensor.shape}")
    return fake_tensor


# ─────────────────────────────────────────────────────────────
# 5. HELPER — DECODE 9-CHANNEL TENSOR TO ACTIVITY CODES
# ─────────────────────────────────────────────────────────────

def decode_to_codes(tensor_9ch: np.ndarray) -> np.ndarray:
    """
    Convert a (N, 9, 48, 1) one-hot-ish tensor to activity codes (N, 48).
    Channel argmax maps 0-8 → divisions 1-9.
    """
    return np.argmax(tensor_9ch, axis=1).squeeze(-1) + 1  # (N, 48)


# ─────────────────────────────────────────────────────────────
# 6. METRICS — FREQUENCY, JSD, AVERAGE MINUTES
# ─────────────────────────────────────────────────────────────

def compute_metrics(real_codes: np.ndarray, fake_codes: np.ndarray):
    """
    Compute per-division frequency (%), Jensen-Shannon Divergence, and
    average daily minutes for both real and synthetic diaries.

    Parameters
    ----------
    real_codes, fake_codes : np.ndarray  (N, 48), values in 1..9

    Returns
    -------
    metrics : dict
    """
    divisions = np.arange(1, 10)

    # ── Per-division frequency (% of all slots) ───────────────
    real_flat = real_codes.flatten()
    fake_flat = fake_codes.flatten()

    real_freq = np.array([(real_flat == d).sum() for d in divisions], dtype=float)
    fake_freq = np.array([(fake_flat == d).sum() for d in divisions], dtype=float)

    real_pct = real_freq / real_flat.size * 100
    fake_pct = fake_freq / fake_flat.size * 100

    # ── Jensen-Shannon Divergence ─────────────────────────────
    # Normalise to probability distributions
    real_prob = real_freq / real_freq.sum()
    fake_prob = fake_freq / fake_freq.sum()
    jsd = float(jensenshannon(real_prob, fake_prob) ** 2)  # squared = JSD

    # ── Average daily minutes per division ────────────────────
    real_minutes = np.array(
        [np.mean(np.sum(real_codes == d, axis=1)) * MINUTES_PER_SLOT for d in divisions]
    )
    fake_minutes = np.array(
        [np.mean(np.sum(fake_codes == d, axis=1)) * MINUTES_PER_SLOT for d in divisions]
    )

    return dict(
        divisions=divisions,
        real_pct=real_pct,
        fake_pct=fake_pct,
        jsd=jsd,
        real_minutes=real_minutes,
        fake_minutes=fake_minutes,
    )


def print_metrics(m: dict):
    """Pretty-print the evaluation metrics."""
    divs = m["divisions"]
    print("\n" + "=" * 78)
    print("  EVALUATION RESULTS")
    print("=" * 78)

    # Frequency table
    print(f"\n{'Div':>4}  {'Activity':<26}  {'Real %':>8}  {'Synth %':>8}  {'Δ':>7}")
    print("─" * 60)
    for i, d in enumerate(divs):
        delta = m["fake_pct"][i] - m["real_pct"][i]
        print(
            f"  {d:>2}  {DIVISION_LABELS[d]:<26}  "
            f"{m['real_pct'][i]:7.2f}%  {m['fake_pct'][i]:7.2f}%  {delta:+6.2f}%"
        )

    # JSD
    print(f"\n  Jensen-Shannon Divergence (JSD):  {m['jsd']:.6f}")

    # Minutes table
    print(f"\n{'Div':>4}  {'Activity':<26}  {'Real min':>9}  {'Synth min':>10}  {'Δ':>8}")
    print("─" * 65)
    for i, d in enumerate(divs):
        delta = m["fake_minutes"][i] - m["real_minutes"][i]
        print(
            f"  {d:>2}  {DIVISION_LABELS[d]:<26}  "
            f"{m['real_minutes'][i]:8.1f}   {m['fake_minutes'][i]:9.1f}   {delta:+7.1f}"
        )
    print("=" * 78)


# ─────────────────────────────────────────────────────────────
# 7. VISUALISATIONS
# ─────────────────────────────────────────────────────────────

def _bar_label_positions(n_groups):
    """X positions for side-by-side bar charts."""
    x = np.arange(n_groups)
    width = 0.35
    return x, width


def plot_frequency_distribution(m: dict, output_dir: str):
    """Side-by-side bar chart of activity frequency (%)."""
    x, w = _bar_label_positions(len(m["divisions"]))
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - w / 2, m["real_pct"], w, label="Real", alpha=0.8, color="#4C72B0")
    ax.bar(x + w / 2, m["fake_pct"], w, label="Synthetic", alpha=0.8, color="#DD8452")
    ax.set_xlabel("Major Division")
    ax.set_ylabel("Frequency (%)")
    ax.set_title("Activity Code Distribution: Real vs Synthetic")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d}\n{DIVISION_LABELS[d][:12]}" for d in m["divisions"]],
                       fontsize=8)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    path = os.path.join(output_dir, "activity_distribution.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  📊 Saved {path}")


def plot_time_use(m: dict, output_dir: str):
    """Side-by-side bar chart of average daily minutes per division."""
    x, w = _bar_label_positions(len(m["divisions"]))
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - w / 2, m["real_minutes"], w, label="Real", alpha=0.8, color="#4C72B0")
    ax.bar(x + w / 2, m["fake_minutes"], w, label="Synthetic", alpha=0.8, color="#DD8452")
    ax.set_xlabel("Major Division")
    ax.set_ylabel("Average Minutes per Day")
    ax.set_title("Time Use Comparison (Major Divisions): Real vs Synthetic")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d}\n{DIVISION_LABELS[d][:12]}" for d in m["divisions"]],
                       fontsize=8)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    path = os.path.join(output_dir, "time_use_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  📊 Saved {path}")


def plot_heatmap_comparison(
    real_tensor: np.ndarray,
    fake_tensor: np.ndarray,
    output_dir: str,
):
    """
    Heatmap of the *average* diary across the dataset (9 channels × 48 slots).
    Real on top, synthetic on bottom.
    """
    # Average across batch and squeeze width dim → (9, 48)
    real_avg = real_tensor.mean(axis=0).squeeze(-1)
    fake_avg = fake_tensor.mean(axis=0).squeeze(-1)

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    im0 = axes[0].imshow(real_avg, aspect="auto", cmap="viridis", interpolation="nearest")
    axes[0].set_title("Average Real Diary")
    axes[0].set_ylabel("Division (ch 0-8 → div 1-9)")
    axes[0].set_yticks(range(9))
    axes[0].set_yticklabels([DIVISION_LABELS[d] for d in range(1, 10)], fontsize=7)
    fig.colorbar(im0, ax=axes[0], fraction=0.02, pad=0.02)

    im1 = axes[1].imshow(fake_avg, aspect="auto", cmap="viridis", interpolation="nearest")
    axes[1].set_title("Average Synthetic Diary")
    axes[1].set_ylabel("Division (ch 0-8 → div 1-9)")
    axes[1].set_xlabel("Time Slot (30-min intervals starting 04:00 AM)")
    axes[1].set_yticks(range(9))
    axes[1].set_yticklabels([DIVISION_LABELS[d] for d in range(1, 10)], fontsize=7)
    fig.colorbar(im1, ax=axes[1], fraction=0.02, pad=0.02)

    fig.tight_layout()
    path = os.path.join(output_dir, "heatmap_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  📊 Saved {path}")


def plot_sample_diaries(real_codes: np.ndarray, fake_codes: np.ndarray, output_dir: str):
    """Step-plot visualisation: 5 real diaries (top row) vs 5 synthetic (bottom)."""
    fig, axes = plt.subplots(2, 5, figsize=(22, 8), sharex=True, sharey=True)

    for i in range(5):
        axes[0, i].step(range(48), real_codes[i], where="post", color="#4C72B0", linewidth=1.2)
        axes[0, i].set_title(f"Real Diary {i + 1}", fontsize=9)

        axes[1, i].step(range(48), fake_codes[i], where="post", color="#DD8452", linewidth=1.2)
        axes[1, i].set_title(f"Synthetic Diary {i + 1}", fontsize=9)

    for ax in axes.flatten():
        ax.set_ylim(0.5, 9.5)
        ax.set_yticks(range(1, 10))
        ax.set_yticklabels([DIVISION_LABELS[d][:10] for d in range(1, 10)], fontsize=6)
        ax.grid(axis="y", linestyle=":", alpha=0.3)

    fig.supxlabel("Time Slot")
    fig.supylabel("Activity Division")
    fig.tight_layout()
    path = os.path.join(output_dir, "sample_diaries.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  📊 Saved {path}")


# ─────────────────────────────────────────────────────────────
# 8. CLI ARGUMENT PARSER
# ─────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="TUS-GAN v2 — Evaluate a trained Generator checkpoint",
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to the .pt checkpoint file",
    )
    parser.add_argument(
        "--data", type=str, default="2019/img-encode/tusgan_encode.npz",
        help="Path to the real-data .npz file",
    )
    parser.add_argument(
        "--n-samples", type=int, default=10_000,
        help="Number of synthetic diaries to generate (default: 10 000)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="evaluation_results",
        help="Directory for saved plots and metrics",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────
# 9. MAIN
# ─────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Device: {device}")

    # ── Load Generator ────────────────────────────────────────
    G, cfg = load_generator(args.checkpoint, args.data, device)

    # ── Load real data ────────────────────────────────────────
    diary_tensor, cond_vector, district_ids, state_ids = load_real_data(args.data)

    # ── Generate synthetic diaries ────────────────────────────
    fake_tensor = generate_synthetic(
        G, cond_vector, district_ids, state_ids,
        noise_dim=cfg["noise_dim"],
        n_samples=args.n_samples,
        device=device,
    )

    # ── Decode to activity codes ──────────────────────────────
    real_codes = decode_to_codes(diary_tensor)
    fake_codes = decode_to_codes(fake_tensor)

    # ── Compute & print metrics ───────────────────────────────
    metrics = compute_metrics(real_codes, fake_codes)
    print_metrics(metrics)

    # ── Save visualisations ───────────────────────────────────
    print(f"\nSaving visualisations to {args.output_dir}/")
    plot_frequency_distribution(metrics, args.output_dir)
    plot_time_use(metrics, args.output_dir)
    plot_heatmap_comparison(diary_tensor, fake_tensor, args.output_dir)
    plot_sample_diaries(real_codes, fake_codes, args.output_dir)

    print("\n✅ Evaluation complete.")


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
