
"""
TUS-GAN — Training Script (WGAN-GP)
=====================================
Trains the conditional Generator and Critic using the
Wasserstein GAN with Gradient Penalty objective.

Usage
-----
    python train.py                          # uses all defaults
    python train.py --data path/to/file.npz  # custom data path
    python train.py --epochs 300 --batch 64  # override hyperparams
    python train.py --resume checkpoints/epoch_100.pt  # resume training

Expected files in the same directory
--------------------------------------
    tusgan_encoded.npz   (output of encode.ipynb)
    generator.py
    critic.py

Output
------
    checkpoints/epoch_N.pt   model snapshots
    samples/epoch_N.npy      generated diaries for visual inspection
    runs/tusgan/             TensorBoard logs

Quick TensorBoard view
----------------------
    tensorboard --logdir runs/tusgan
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

# Import our models and loss helpers
from generator import Generator
from critic    import Critic, compute_gradient_penalty, critic_loss, generator_loss


# ─────────────────────────────────────────────────────────────
# 1. DATASET
# ─────────────────────────────────────────────────────────────

class TUSDataset(Dataset):
    """
    Loads the pre-encoded ITUS 2019 data produced by encode.ipynb.

    Each item is a tuple:
        diary_tensor  : (1, 48, 1)  float32  values in [-1, +1]
        cond_vector   : (49,)       float32  one-hot demographics
        district_id   : ()          int64    district index

    Parameters
    ----------
    npz_path : path to tusgan_encoded.npz
    device   : if given, tensors are pre-loaded to that device.
               For large datasets leave as None and let the DataLoader
               move batches in the training loop.
    """

    def __init__(self, npz_path: str, device=None):
        if not os.path.exists(npz_path):
            # Try look in the same directory as this script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            npz_path = os.path.join(script_dir, os.path.basename(npz_path))
            
        data = np.load(npz_path)

        # Load arrays from the .npz file and ensure correct types
        self.diary    = torch.from_numpy(data["diary_tensor"]).float()   # (N, 1, 48, 1)
        self.cond     = torch.from_numpy(data["cond_vector"]).float()    # (N, 49)
        self.dist_ids = torch.from_numpy(data["district_ids"]).long()    # (N,)

        # Store number of districts so we can pass it to the models
        self.num_districts = int(data["num_districts"])

        if device is not None:
            self.diary    = self.diary.to(device)
            self.cond     = self.cond.to(device)
            self.dist_ids = self.dist_ids.to(device)

        print(f"Dataset loaded: {len(self.diary):,} diaries | "
              f"cond_dim={self.cond.shape[1]} | "
              f"num_districts={self.num_districts}")

    def __len__(self):
        return len(self.diary)

    def __getitem__(self, idx):
        return self.diary[idx], self.cond[idx], self.dist_ids[idx]


# ─────────────────────────────────────────────────────────────
# 2. HYPERPARAMETERS (central config dict)
# ─────────────────────────────────────────────────────────────

def get_config():
    """
    All tunable hyperparameters in one place.
    Command-line arguments (parsed below) override these defaults.
    """
    return dict(
        # ── Data ────────────────────────────────────────────────
        data_path    = "tusgan_encoded.npz",
        num_workers  = 2,         # DataLoader workers (0 on Windows)

        # ── Model ───────────────────────────────────────────────
        noise_dim          = 128,
        cond_dim           = 49,
        district_embed_dim = 16,
        g_base_channels    = 256,
        d_base_channels    = 64,

        # ── Training ────────────────────────────────────────────
        epochs        = 200,
        batch_size    = 128,
        n_critic      = 5,        # Critic updates per Generator update
        lambda_gp     = 10.0,     # Gradient penalty weight

        # ── Optimiser (Adam with WGAN-GP recommended betas) ─────
        lr            = 1e-4,
        beta1         = 0.0,      # Do NOT use 0.9; WGAN-GP needs 0.0
        beta2         = 0.9,

        # ── Logging & checkpoints ────────────────────────────────
        log_every     = 50,       # log to TensorBoard every N batches
        save_every    = 10,       # save checkpoint every N epochs
        sample_every  = 10,       # save sample diaries every N epochs
        n_samples     = 16,       # how many fake diaries to save

        # ── Paths ────────────────────────────────────────────────
        checkpoint_dir = "checkpoints",
        sample_dir     = "samples",
        log_dir        = "runs/tusgan",

        # ── Resume ───────────────────────────────────────────────
        resume         = None,    # path to a .pt checkpoint to resume from
    )


# ─────────────────────────────────────────────────────────────
# 3. CHECKPOINT HELPERS
# ─────────────────────────────────────────────────────────────

def save_checkpoint(path, epoch, G, D, opt_G, opt_D, cfg):
    """Save everything needed to resume training exactly."""
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
    """Load a checkpoint. Returns the epoch to resume from."""
    ckpt = torch.load(path, map_location=device)
    G.load_state_dict(ckpt["G_state"])
    D.load_state_dict(ckpt["D_state"])
    opt_G.load_state_dict(ckpt["opt_G_state"])
    opt_D.load_state_dict(ckpt["opt_D_state"])
    start_epoch = ckpt["epoch"] + 1
    print(f"  Resumed from {path} at epoch {ckpt['epoch']}")
    return start_epoch


# ─────────────────────────────────────────────────────────────
# 4. SAMPLE HELPER
# ─────────────────────────────────────────────────────────────

def save_samples(G, fixed_z, fixed_cond, fixed_dist, epoch, sample_dir, device):
    """
    Generate a fixed batch of fake diaries and save as .npy.
    Using a *fixed* noise vector across epochs lets you visually
    track how the same latent code evolves during training.
    """
    G.eval()
    with torch.no_grad():
        fake = G(fixed_z, fixed_cond, fixed_dist)   # (N, 1, 48, 1)
    G.train()

    # Convert back to numpy and save
    fake_np = fake.cpu().numpy()
    path    = os.path.join(sample_dir, f"epoch_{epoch:04d}.npy")
    np.save(path, fake_np)
    return fake_np


# ─────────────────────────────────────────────────────────────
# 5. TRAINING LOOP
# ─────────────────────────────────────────────────────────────

def train(cfg):
    # ── Device ──────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # ── Directories ─────────────────────────────────────────────
    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)
    os.makedirs(cfg["sample_dir"],     exist_ok=True)
    os.makedirs(cfg["log_dir"],        exist_ok=True)

    # ── Dataset & DataLoader ─────────────────────────────────────
    dataset = TUSDataset(cfg["data_path"])

    # num_districts comes from the dataset, not a hard-coded constant,
    # so the model matches whatever is actually in the data file.
    num_districts = dataset.num_districts

    loader = DataLoader(
        dataset,
        batch_size  = cfg["batch_size"],
        shuffle     = True,
        num_workers = cfg["num_workers"],
        pin_memory  = (device.type == "cuda"),   # speeds up CPU→GPU transfer
        drop_last   = True,                      # keeps batch size constant
    )
    print(f"Batches per epoch: {len(loader)}")

    # Create an iterator for the loader so we can fetch fresh batches for every critic step
    data_iter = iter(loader)

    # ── Models ───────────────────────────────────────────────────
    G = Generator(
        noise_dim          = cfg["noise_dim"],
        cond_dim           = cfg["cond_dim"],
        num_districts      = num_districts,
        district_embed_dim = cfg["district_embed_dim"],
        base_channels      = cfg["g_base_channels"],
    ).to(device)

    D = Critic(
        cond_dim           = cfg["cond_dim"],
        num_districts      = num_districts,
        district_embed_dim = cfg["district_embed_dim"],
        base_channels      = cfg["d_base_channels"],
    ).to(device)

    g_params = sum(p.numel() for p in G.parameters() if p.requires_grad)
    d_params = sum(p.numel() for p in D.parameters() if p.requires_grad)
    print(f"Generator params : {g_params:,}")
    print(f"Critic    params : {d_params:,}")

    # ── Optimisers ───────────────────────────────────────────────
    # WGAN-GP paper recommends Adam with beta1=0, beta2=0.9.
    # Using beta1=0.9 (the PyTorch default) causes unstable training.
    opt_G = torch.optim.Adam(G.parameters(), lr=cfg["lr"],
                             betas=(cfg["beta1"], cfg["beta2"]))
    opt_D = torch.optim.Adam(D.parameters(), lr=cfg["lr"],
                             betas=(cfg["beta1"], cfg["beta2"]))

    # ── Optional: resume from checkpoint ────────────────────────
    start_epoch = 1
    if cfg["resume"]:
        start_epoch = load_checkpoint(cfg["resume"], G, D, opt_G, opt_D, device)

    # ── TensorBoard writer ───────────────────────────────────────
    writer = SummaryWriter(cfg["log_dir"])

    # ── Fixed noise for consistent samples across epochs ─────────
    # We pick the first n_samples from the dataset as fixed conditions
    # so generated samples always represent the same demographics.
    fixed_batch   = next(iter(DataLoader(dataset, batch_size=cfg["n_samples"], shuffle=True)))
    fixed_cond    = fixed_batch[1].to(device).float()    # (n_samples, 49)
    fixed_dist    = fixed_batch[2].to(device).long()     # (n_samples,)
    fixed_z       = torch.randn(cfg["n_samples"], cfg["noise_dim"], device=device)

    # ── Global step counter for TensorBoard ─────────────────────
    global_step = (start_epoch - 1) * (len(loader) // cfg["n_critic"])

    # ════════════════════════════════════════════════════════════
    #  MAIN TRAINING LOOP
    # ════════════════════════════════════════════════════════════
    print(f"\nStarting training from epoch {start_epoch} → {cfg['epochs']}\n")

    for epoch in range(start_epoch, cfg["epochs"] + 1):

        # Accumulators for epoch-level averages
        epoch_loss_D  = 0.0
        epoch_loss_G  = 0.0
        epoch_w_dist  = 0.0   # Wasserstein distance estimate
        epoch_gp      = 0.0
        n_critic_steps = 0
        n_gen_steps    = 0

        # Number of Generator steps per epoch
        n_gen_batches = len(loader) // cfg["n_critic"]

        for _ in range(n_gen_batches):

            # ════════════════════════════════════════════════════
            # STEP A — Train Critic (n_critic times per G step)
            # ════════════════════════════════════════════════════
            for _ in range(cfg["n_critic"]):
                try:
                    real_diaries, cond_vec, dist_ids = next(data_iter)
                except StopIteration:
                    data_iter = iter(loader)
                    real_diaries, cond_vec, dist_ids = next(data_iter)

                # Move data to the target device and ensure correct types
                real_diaries = real_diaries.to(device).float()   # (B, 1, 48, 1)
                cond_vec     = cond_vec.to(device).float()        # (B, 49)
                dist_ids     = dist_ids.to(device).long()         # (B,)
                B            = real_diaries.size(0)

                # ── Sample fresh noise for each Critic update ──────
                z = torch.randn(B, cfg["noise_dim"], device=device)

                # Generate fake diaries (no gradient needed for G here)
                with torch.no_grad():
                    fake_diaries = G(z, cond_vec, dist_ids)   # (B, 1, 48, 1)

                # Score real and fake
                real_scores = D(real_diaries, cond_vec, dist_ids)  # (B, 1)
                fake_scores = D(fake_diaries, cond_vec, dist_ids)  # (B, 1)

                # Gradient penalty at interpolated points
                gp = compute_gradient_penalty(
                    D, real_diaries, fake_diaries,
                    cond_vec, dist_ids, device, cfg["lambda_gp"]
                )

                # Critic loss: minimise fake - real + GP
                loss_D = critic_loss(real_scores, fake_scores, gp)

                opt_D.zero_grad()
                loss_D.backward()
                opt_D.step()

                # Track Wasserstein distance: real - fake (before GP term)
                w_dist = (real_scores.mean() - fake_scores.mean()).item()
                epoch_w_dist += w_dist
                epoch_loss_D += loss_D.item()
                epoch_gp     += gp.item()
                n_critic_steps += 1

            # ════════════════════════════════════════════════════
            # STEP B — Train Generator (once per n_critic steps)
            # ════════════════════════════════════════════════════
            # We reuse the conditions and districts from the LAST Critic batch
            # but sample FRESH noise for the Generator update.
            z_g          = torch.randn(B, cfg["noise_dim"], device=device)
            fake_diaries = G(z_g, cond_vec, dist_ids)
            fake_scores  = D(fake_diaries, cond_vec, dist_ids)

            loss_G = generator_loss(fake_scores)

            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

            epoch_loss_G += loss_G.item()
            n_gen_steps  += 1
            global_step  += 1

            # ── Batch-level TensorBoard logging ─────────────────
            if global_step % cfg["log_every"] == 0:
                writer.add_scalar("Batch/loss_D",       loss_D.item(),  global_step)
                writer.add_scalar("Batch/loss_G",       loss_G.item(),  global_step)
                writer.add_scalar("Batch/w_distance",   w_dist,         global_step)
                writer.add_scalar("Batch/gradient_penalty", gp.item(),  global_step)

        # ── End of epoch: compute averages ───────────────────────
        avg_loss_D = epoch_loss_D / n_critic_steps
        avg_loss_G = epoch_loss_G / n_gen_steps
        avg_w_dist = epoch_w_dist / n_critic_steps
        avg_gp     = epoch_gp     / n_critic_steps

        # ── Epoch-level TensorBoard logging ─────────────────────
        writer.add_scalar("Epoch/loss_D",          avg_loss_D, epoch)
        writer.add_scalar("Epoch/loss_G",          avg_loss_G, epoch)
        writer.add_scalar("Epoch/w_distance",      avg_w_dist, epoch)
        writer.add_scalar("Epoch/gradient_penalty",avg_gp,     epoch)

        # ── Console output ───────────────────────────────────────
        print(
            f"Epoch [{epoch:>4}/{cfg['epochs']}] | "
            f"W-dist: {avg_w_dist:+.4f} | "
            f"Loss_D: {avg_loss_D:.4f} | "
            f"Loss_G: {avg_loss_G:.4f} | "
            f"GP: {avg_gp:.4f}"
        )

        # ── Save checkpoint ──────────────────────────────────────
        if epoch % cfg["save_every"] == 0:
            ckpt_path = os.path.join(cfg["checkpoint_dir"], f"epoch_{epoch:04d}.pt")
            save_checkpoint(ckpt_path, epoch, G, D, opt_G, opt_D, cfg)

        # ── Save generated samples ────────────────────────────────
        if epoch % cfg["sample_every"] == 0:
            save_samples(G, fixed_z, fixed_cond, fixed_dist, epoch,
                         cfg["sample_dir"], device)

    # ── Final checkpoint ─────────────────────────────────────────
    save_checkpoint(
        os.path.join(cfg["checkpoint_dir"], "final.pt"),
        cfg["epochs"], G, D, opt_G, opt_D, cfg
    )
    writer.close()
    print("\nTraining complete.")


# ─────────────────────────────────────────────────────────────
# 6. CLI ARGUMENT PARSER
# ─────────────────────────────────────────────────────────────

def parse_args():
    cfg = get_config()
    parser = argparse.ArgumentParser(description="TUS-GAN WGAN-GP Training")

    # Allow every config key to be overridden from the command line
    parser.add_argument("--data",        type=str,   default=cfg["data_path"])
    parser.add_argument("--epochs",      type=int,   default=cfg["epochs"])
    parser.add_argument("--batch",       type=int,   default=cfg["batch_size"])
    parser.add_argument("--noise_dim",   type=int,   default=cfg["noise_dim"])
    parser.add_argument("--n_critic",    type=int,   default=cfg["n_critic"])
    parser.add_argument("--lambda_gp",   type=float, default=cfg["lambda_gp"])
    parser.add_argument("--lr",          type=float, default=cfg["lr"])
    parser.add_argument("--save_every",  type=int,   default=cfg["save_every"])
    parser.add_argument("--sample_every",type=int,   default=cfg["sample_every"])
    parser.add_argument("--resume",      type=str,   default=None,
                        help="Path to checkpoint to resume from")

    args = parser.parse_args()

    # Merge parsed args back into cfg dict
    cfg["data_path"]    = args.data
    cfg["epochs"]       = args.epochs
    cfg["batch_size"]   = args.batch
    cfg["noise_dim"]    = args.noise_dim
    cfg["n_critic"]     = args.n_critic
    cfg["lambda_gp"]    = args.lambda_gp
    cfg["lr"]           = args.lr
    cfg["save_every"]   = args.save_every
    cfg["sample_every"] = args.sample_every
    cfg["resume"]       = args.resume

    return cfg


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = parse_args()

    # Print the full config so every run is reproducible
    print("\n── TUS-GAN Configuration ──────────────────────")
    for k, v in cfg.items():
        print(f"  {k:<22}: {v}")
    print("────────────────────────────────────────────────\n")

    train(cfg)