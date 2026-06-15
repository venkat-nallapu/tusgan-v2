# -*- coding: utf-8 -*-
"""
TUS-GAN — Training Script for 9-Channel (WGAN-GP with STATE)
==============================================================
Trains the conditional Generator and Critic using WGAN-GP
for 9-channel diary tensors with state and district embeddings.

Adapted to match the formatting structure of train (1).py.
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

# Import our models
from generator import Generator
from critic import Critic, compute_gradient_penalty, critic_loss, generator_loss


# ─────────────────────────────────────────────────────────────
# 1. DATASET (with STATE & SUBSET support)
# ─────────────────────────────────────────────────────────────

class TUSDataset(Dataset):
    """
    Loads the pre-encoded 9-channel ITUS data with state and district.

    Each item is a tuple:
        diary_tensor  : (9, 48, 1)   float32  binary one-hot
        cond_vector   : (83,)        float32  one-hot demographics
        district_id   : ()           int64    district index
        state_id      : ()           int64    state index
    """

    def __init__(self, npz_path: str, subset_size: int = None, device=None):
        if not os.path.exists(npz_path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            npz_path = os.path.join(script_dir, os.path.basename(npz_path))

        data = np.load(npz_path)

        # Load arrays
        diary = torch.from_numpy(data["diary_tensor"]).float()   # (N, 9, 48, 1)
        cond = torch.from_numpy(data["cond_vector"]).float()     # (N, 83)
        district_ids = torch.from_numpy(data["district_ids"]).long()  # (N,)
        state_ids = torch.from_numpy(data["state_ids"]).long()        # (N,)

        if subset_size is not None and subset_size > 0:
            diary = diary[:subset_size]
            cond = cond[:subset_size]
            district_ids = district_ids[:subset_size]
            state_ids = state_ids[:subset_size]

        self.diary = diary
        self.cond = cond
        self.district_ids = district_ids
        self.state_ids = state_ids

        # Store metadata
        self.num_districts = int(data["num_districts"])
        self.num_states = int(data["num_states"])
        self.num_channels = self.diary.shape[1]
        self.cond_dim = self.cond.shape[1]

        if device is not None:
            self.diary = self.diary.to(device)
            self.cond = self.cond.to(device)
            self.district_ids = self.district_ids.to(device)
            self.state_ids = self.state_ids.to(device)

        print(f"✅ Dataset loaded: {len(self.diary):,} diaries")
        print(f"   Diary shape: {self.diary.shape} (9-channel representation)")
        print(f"   Cond dim: {self.cond_dim}")
        print(f"   Num districts: {self.num_districts}")
        print(f"   Num states: {self.num_states}")

    def __len__(self):
        return len(self.diary)

    def __getitem__(self, idx):
        return self.diary[idx], self.cond[idx], self.district_ids[idx], self.state_ids[idx]


# ─────────────────────────────────────────────────────────────
# 2. HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────

def get_config():
    return dict(
        # ── Data ────────────────────────────────────────────────
        data_path    = "2019/img-encode/tusgan_encode.npz",
        num_workers  = 2,
        subset       = None,

        # ── Model Architecture ───────────────────────────────────
        noise_dim          = 128,
        cond_dim           = 83,
        district_embed_dim = 16,
        state_embed_dim    = 8,
        g_base_channels    = 256,
        d_base_channels    = 64,
        num_channels       = 9,

        # ── Training ────────────────────────────────────────────
        epochs        = 250,
        batch_size    = 512,
        n_critic      = 5,
        lambda_gp     = 10.0,

        # ── Optimizer ────────────────────────────────────────────
        lr            = 0.0001,
        beta1         = 0.0,
        beta2         = 0.9,

        # ── Logging & Checkpoints ────────────────────────────────
        log_every     = 50,
        save_every    = 10,
        sample_every  = 10,
        n_samples     = 16,

        # ── Paths ────────────────────────────────────────────────
        checkpoint_dir = "checkpoints",
        sample_dir     = "samples",
        log_dir        = "runs/tusgan_9channel",

        # ── Resume ───────────────────────────────────────────────
        resume         = None,
    )


# ─────────────────────────────────────────────────────────────
# 3. CHECKPOINT HELPERS
# ─────────────────────────────────────────────────────────────

def save_checkpoint(path, epoch, G, D, opt_G, opt_D, cfg):
    torch.save({
        "epoch"      : epoch,
        "G_state"    : G.state_dict(),
        "D_state"    : D.state_dict(),
        "opt_G_state": opt_G.state_dict(),
        "opt_D_state": opt_D.state_dict(),
        "config"     : cfg,
    }, path)
    print(f"  Checkpoint saved → {path}")


def load_checkpoint(path, G, D, opt_G, opt_D, device):
    ckpt = torch.load(path, map_location=device)
    G.load_state_dict(ckpt["G_state"])
    D.load_state_dict(ckpt["D_state"])
    opt_G.load_state_dict(ckpt["opt_G_state"])
    opt_D.load_state_dict(ckpt["opt_D_state"])
    start_epoch = ckpt["epoch"] + 1
    print(f"  Resumed from {path} at epoch {ckpt['epoch']}")
    return start_epoch


# ─────────────────────────────────────────────────────────────
# 4. SAMPLE HELPER (with visual heatmap logging)
# ─────────────────────────────────────────────────────────────

def save_samples(G, fixed_z, fixed_cond, fixed_dist, fixed_state, epoch, sample_dir, writer):
    G.eval()
    with torch.no_grad():
        fake = G(fixed_z, fixed_cond, fixed_dist, fixed_state)
    G.train()

    fake_np = fake.cpu().numpy()
    path = os.path.join(sample_dir, f"epoch_{epoch:04d}.npy")
    np.save(path, fake_np)

    # Log activation summary
    channel_counts = fake_np.sum(axis=(0, 2, 3))
    summary_path = os.path.join(sample_dir, f"epoch_{epoch:04d}_channels.txt")
    with open(summary_path, 'w') as f:
        f.write(f"Channel activation counts for epoch {epoch}:\n")
        for c in range(fake_np.shape[1]):
            f.write(f"  Channel {c}: {channel_counts[c]:.0f} activations\n")

    return fake_np


# ─────────────────────────────────────────────────────────────
# 5. TRAINING LOOP
# ─────────────────────────────────────────────────────────────

def train(cfg):
    # ── Device ──────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️ Device: {device}")
    if device.type == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Directories ─────────────────────────────────────────────
    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)
    os.makedirs(cfg["sample_dir"], exist_ok=True)
    os.makedirs(cfg["log_dir"], exist_ok=True)

    # ── Dataset & DataLoader ─────────────────────────────────────
    data_path = cfg["data_path"]
    if not os.path.exists(data_path) and os.path.exists(os.path.join("wgan-gp", data_path)):
        data_path = os.path.join("wgan-gp", data_path)
        print(f"Resolving dataset path to: {data_path}")

    dataset = TUSDataset(data_path, subset_size=cfg.get("subset"))

    # Update config with actual dataset dimensions so checkpoints
    # save the correct values (defaults may differ from real data).
    cfg["cond_dim"] = dataset.cond_dim
    cfg["num_districts"] = dataset.num_districts
    cfg["num_states"] = dataset.num_states
    cfg["num_channels"] = dataset.num_channels

    num_workers = min(4, os.cpu_count() or 2)
    loader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    print(f"Batches per epoch: {len(loader)}")

    # Create iterator for the loader
    data_iter = iter(loader)

    # ── Models (with STATE) ───────────────────────────────────────
    G = Generator(
        noise_dim=cfg["noise_dim"],
        cond_dim=cfg["cond_dim"],
        num_districts=dataset.num_districts,
        num_states=dataset.num_states,
        district_embed_dim=cfg["district_embed_dim"],
        state_embed_dim=cfg["state_embed_dim"],
        base_channels=cfg["g_base_channels"],
    ).to(device)

    D = Critic(
        cond_dim=dataset.cond_dim,
        num_districts=dataset.num_districts,
        num_states=dataset.num_states,
        district_embed_dim=cfg["district_embed_dim"],
        state_embed_dim=cfg["state_embed_dim"],
        base_channels=cfg["d_base_channels"],
    ).to(device)

    g_params = sum(p.numel() for p in G.parameters() if p.requires_grad)
    d_params = sum(p.numel() for p in D.parameters() if p.requires_grad)
    print(f"\n📊 Model Parameters:")
    print(f"   Generator: {g_params:,}")
    print(f"   Critic:    {d_params:,}")
    print(f"   Total:     {g_params + d_params:,}")

    # ── Optimizers ───────────────────────────────────────────────
    opt_G = torch.optim.Adam(G.parameters(), lr=cfg["lr"],
                             betas=(cfg["beta1"], cfg["beta2"]))
    opt_D = torch.optim.Adam(D.parameters(), lr=cfg["lr"],
                             betas=(cfg["beta1"], cfg["beta2"]))

    # Learning rate schedulers to stabilize convergence
    sched_G = torch.optim.lr_scheduler.StepLR(opt_G, step_size=30, gamma=0.5)
    sched_D = torch.optim.lr_scheduler.StepLR(opt_D, step_size=30, gamma=0.5)

    # ── Resume from checkpoint ───────────────────────────────────
    start_epoch = 1
    if cfg["resume"]:
        start_epoch = load_checkpoint(cfg["resume"], G, D, opt_G, opt_D, device)

    # ── TensorBoard writer ───────────────────────────────────────
    writer = SummaryWriter(cfg["log_dir"])

    # ── Fixed samples for consistent visualization ───────────────
    fixed_batch = next(iter(DataLoader(dataset, batch_size=cfg["n_samples"], shuffle=True)))
    fixed_cond = fixed_batch[1].to(device).float()
    fixed_dist = fixed_batch[2].to(device).long()
    fixed_state = fixed_batch[3].to(device).long()
    fixed_z = torch.randn(cfg["n_samples"], cfg["noise_dim"], device=device)

    # ── Global step counter ─────────────────────────────────────
    global_step = (start_epoch - 1) * (len(loader) // cfg["n_critic"])

    # ════════════════════════════════════════════════════════════
    #  MAIN TRAINING LOOP
    # ════════════════════════════════════════════════════════════
    print(f"\n🚀 Starting training from epoch {start_epoch} → {cfg['epochs']}\n")
    print("="*70)

    for epoch in range(start_epoch, cfg["epochs"] + 1):

        # Accumulators
        epoch_loss_D = 0.0
        epoch_loss_G = 0.0
        epoch_w_dist = 0.0
        epoch_gp = 0.0
        n_critic_steps = 0
        n_gen_steps = 0

        n_gen_batches = max(1, len(loader) // cfg["n_critic"])


        for gen_step in range(n_gen_batches):

            # ════════════════════════════════════════════════════
            # STEP A — Train Critic (n_critic times)
            # ════════════════════════════════════════════════════
            for _ in range(cfg["n_critic"]):
                try:
                    real_diaries, cond_vec, dist_ids, state_ids = next(data_iter)
                except StopIteration:
                    data_iter = iter(loader)
                    real_diaries, cond_vec, dist_ids, state_ids = next(data_iter)

                # Move to device
                real_diaries = real_diaries.to(device).float()
                cond_vec = cond_vec.to(device).float()
                dist_ids = dist_ids.to(device).long()
                state_ids = state_ids.to(device).long()
                B = real_diaries.size(0)

                # Sample noise
                z = torch.randn(B, cfg["noise_dim"], device=device)

                # Generate fake diaries
                with torch.no_grad():
                    fake_diaries = G(z, cond_vec, dist_ids, state_ids)

                # Score real and fake
                real_scores = D(real_diaries, cond_vec, dist_ids, state_ids)
                fake_scores = D(fake_diaries, cond_vec, dist_ids, state_ids)

                # Gradient penalty
                gp = compute_gradient_penalty(
                    D, real_diaries, fake_diaries,
                    cond_vec, dist_ids, state_ids, device, cfg["lambda_gp"]
                )

                # Critic loss
                loss_D = critic_loss(real_scores, fake_scores, gp)

                opt_D.zero_grad()
                loss_D.backward()
                opt_D.step()

                # Track metrics
                w_dist = (real_scores.mean() - fake_scores.mean()).item()
                epoch_w_dist += w_dist
                epoch_loss_D += loss_D.item()
                epoch_gp += gp.item()
                n_critic_steps += 1

            # ════════════════════════════════════════════════════
            # STEP B — Train Generator
            # ════════════════════════════════════════════════════
            z_g = torch.randn(B, cfg["noise_dim"], device=device)
            fake_diaries = G(z_g, cond_vec, dist_ids, state_ids)
            fake_scores = D(fake_diaries, cond_vec, dist_ids, state_ids)

            loss_G = generator_loss(fake_scores)

            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

            epoch_loss_G += loss_G.item()
            n_gen_steps += 1
            global_step += 1

            # Batch-level logging
            if global_step % cfg["log_every"] == 0:
                writer.add_scalar("Batch/loss_D", loss_D.item(), global_step)
                writer.add_scalar("Batch/loss_G", loss_G.item(), global_step)
                writer.add_scalar("Batch/w_distance", w_dist, global_step)
                writer.add_scalar("Batch/gradient_penalty", gp.item(), global_step)

        # Decay learning rates
        sched_G.step()
        sched_D.step()

        # ── End of epoch ─────────────────────────────────────────
        avg_loss_D = epoch_loss_D / n_critic_steps if n_critic_steps > 0 else 0
        avg_loss_G = epoch_loss_G / n_gen_steps if n_gen_steps > 0 else 0
        avg_w_dist = epoch_w_dist / n_critic_steps if n_critic_steps > 0 else 0
        avg_gp = epoch_gp / n_critic_steps if n_critic_steps > 0 else 0

        # Epoch-level logging
        writer.add_scalar("Epoch/loss_D", avg_loss_D, epoch)
        writer.add_scalar("Epoch/loss_G", avg_loss_G, epoch)
        writer.add_scalar("Epoch/w_distance", avg_w_dist, epoch)
        writer.add_scalar("Epoch/gradient_penalty", avg_gp, epoch)
        writer.add_scalar("LearningRate/Generator", sched_G.get_last_lr()[0], epoch)
        writer.add_scalar("LearningRate/Critic", sched_D.get_last_lr()[0], epoch)

        # Console output
        print(f"Epoch [{epoch:>4}/{cfg['epochs']}] | "
              f"W-dist: {avg_w_dist:+.4f} | "
              f"Loss_D: {avg_loss_D:.4f} | "
              f"Loss_G: {avg_loss_G:.4f} | "
              f"GP: {avg_gp:.4f}")

        # Save checkpoint
        if epoch % cfg["save_every"] == 0:
            ckpt_path = os.path.join(cfg["checkpoint_dir"], f"epoch_{epoch:04d}.pt")
            save_checkpoint(ckpt_path, epoch, G, D, opt_G, opt_D, cfg)

        # Save samples & log visual heatmap comparison to TensorBoard
        if epoch % cfg["sample_every"] == 0:
            fake_samples = save_samples(G, fixed_z, fixed_cond, fixed_dist, fixed_state,
                                       epoch, cfg["sample_dir"], writer)
            
            # Create TensorBoard visual comparison heatmap
            fig, axes = plt.subplots(2, 1, figsize=(10, 6))
            im0 = axes[0].imshow(fake_samples[0, :, :, 0], aspect='auto', cmap='viridis')
            axes[0].set_title(f"Generated Synthetic Diary (Epoch {epoch})")
            axes[0].set_ylabel("Divisions (1-9)")
            
            im1 = axes[1].imshow(fixed_batch[0][0, :, :, 0].numpy(), aspect='auto', cmap='viridis')
            axes[1].set_title("Real Reference Diary")
            axes[1].set_ylabel("Divisions (1-9)")
            axes[1].set_xlabel("Time Slots (30-min intervals starting 04:00 AM)")
            
            fig.colorbar(im0, ax=axes[0])
            fig.colorbar(im1, ax=axes[1])
            plt.tight_layout()
            
            writer.add_figure("VisualComparison/DiaryHeatmap", fig, epoch)
            plt.close(fig)

    # ── Final checkpoint ─────────────────────────────────────────
    save_checkpoint(
        os.path.join(cfg["checkpoint_dir"], "final.pt"),
        cfg["epochs"], G, D, opt_G, opt_D, cfg
    )
    writer.close()
    print("\n✅ Training complete!")
    print(f"   Final model saved to {cfg['checkpoint_dir']}/final.pt")
    print(f"   Samples saved to {cfg['sample_dir']}")
    print(f"   Logs saved to {cfg['log_dir']}")


# ─────────────────────────────────────────────────────────────
# 6. CLI ARGUMENT PARSER
# ─────────────────────────────────────────────────────────────

def parse_args():
    cfg = get_config()
    parser = argparse.ArgumentParser(description="TUS-GAN 9-Channel WGAN-GP Training with STATE")

    parser.add_argument("--data", type=str, default=cfg["data_path"])
    parser.add_argument("--epochs", type=int, default=cfg["epochs"])
    parser.add_argument("--batch", type=int, default=cfg["batch_size"])
    parser.add_argument("--noise_dim", type=int, default=cfg["noise_dim"])
    parser.add_argument("--n_critic", type=int, default=cfg["n_critic"])
    parser.add_argument("--lambda_gp", type=float, default=cfg["lambda_gp"])
    parser.add_argument("--lr", type=float, default=cfg["lr"])
    parser.add_argument("--save_every", type=int, default=cfg["save_every"])
    parser.add_argument("--sample_every", type=int, default=cfg["sample_every"])
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--subset", type=int, default=None)

    args = parser.parse_args()

    cfg["data_path"] = args.data
    cfg["epochs"] = args.epochs
    cfg["batch_size"] = args.batch
    cfg["noise_dim"] = args.noise_dim
    cfg["n_critic"] = args.n_critic
    cfg["lambda_gp"] = args.lambda_gp
    cfg["lr"] = args.lr
    cfg["save_every"] = args.save_every
    cfg["sample_every"] = args.sample_every
    cfg["resume"] = args.resume
    cfg["subset"] = args.subset

    return cfg


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = parse_args()

    print("\n" + "="*70)
    print("TUS-GAN 9-CHANNEL WGAN-GP TRAINING (with STATE + DISTRICT)")
    print("="*70)
    print("\n📋 Configuration:")
    for k, v in cfg.items():
        print(f"   {k:<22}: {v}")
    print("="*70)

    train(cfg)