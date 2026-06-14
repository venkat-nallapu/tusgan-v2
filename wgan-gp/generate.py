
"""
TUS-GAN — Training Script (WGAN-GP v2)
=======================================
Trains the conditional Generator and Critic using the
Wasserstein GAN with Gradient Penalty objective.

v2 Updates:
  - Dynamic cond_dim and num_districts/num_states.
  - Dual embedding support (District + State).
  - 9-Channel Diary (Major Divisions 1-9).
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
    def __init__(self, npz_path: str):
        if not os.path.exists(npz_path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            npz_path = os.path.join(script_dir, os.path.basename(npz_path))

        data = np.load(npz_path)
        self.diary    = torch.from_numpy(data["diary_tensor"]).float()   # (N, 9, 48, 1)
        self.cond     = torch.from_numpy(data["cond_vector"]).float()    # (N, C)
        self.dist_ids = torch.from_numpy(data["district_ids"]).long()    # (N,)
        self.state_ids= torch.from_numpy(data["state_ids"]).long()       # (N,)

        self.num_districts = int(data["num_districts"])
        self.num_states    = int(data["num_states"])
        self.cond_dim      = self.cond.shape[1]

        print(f"Dataset: {len(self.diary):,} samples | OH: {self.cond_dim} | Dists: {self.num_districts} | States: {self.num_states}")

    def __len__(self):
        return len(self.diary)

    def __getitem__(self, idx):
        return self.diary[idx], self.cond[idx], self.dist_ids[idx], self.state_ids[idx]


# ─────────────────────────────────────────────────────────────
# 2. CONFIG & HYPERPARAMS
# ─────────────────────────────────────────────────────────────

def get_config():
    return dict(
        data_path          = "tusgan_encoded.npz",
        noise_dim          = 128,
        district_embed_dim = 16,
        state_embed_dim    = 8,
        g_base_channels    = 256,
        d_base_channels    = 64,
        epochs             = 200,
        batch_size         = 128,
        n_critic           = 5,
        lambda_gp          = 10.0,
        lr                 = 1e-4,
        log_every          = 50,
        save_every         = 10,
        checkpoint_dir     = "checkpoints",
        log_dir            = "runs/tusgan-v2",
    )


# ─────────────────────────────────────────────────────────────
# 3. TRAINING LOOP
# ─────────────────────────────────────────────────────────────

def train(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)

    dataset = TUSDataset(cfg["data_path"])
    cfg["cond_dim"] = dataset.cond_dim
    cfg["num_districts"] = dataset.num_districts
    cfg["num_states"] = dataset.num_states

    loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True, drop_last=True)
    writer = SummaryWriter(cfg["log_dir"])

    G = Generator(
        noise_dim          = cfg["noise_dim"],
        cond_dim           = cfg["cond_dim"],
        num_districts      = cfg["num_districts"],
        num_states         = cfg["num_states"],
        district_embed_dim = cfg["district_embed_dim"],
        state_embed_dim    = cfg["state_embed_dim"],
        base_channels      = cfg["g_base_channels"],
    ).to(device)

    D = Critic(
        cond_dim           = cfg["cond_dim"],
        num_districts      = cfg["num_districts"],
        num_states         = cfg["num_states"],
        district_embed_dim = cfg["district_embed_dim"],
        state_embed_dim    = cfg["state_embed_dim"],
        base_channels      = cfg["d_base_channels"],
    ).to(device)

    opt_G = torch.optim.Adam(G.parameters(), lr=cfg["lr"], betas=(0.0, 0.9))
    opt_D = torch.optim.Adam(D.parameters(), lr=cfg["lr"], betas=(0.0, 0.9))

    # Fixed noise for monitoring
    fixed_batch = next(iter(DataLoader(dataset, batch_size=16, shuffle=True)))
    fz = torch.randn(16, cfg["noise_dim"], device=device)
    fc = fixed_batch[1].to(device)
    fd = fixed_batch[2].to(device)
    fs = fixed_batch[3].to(device)

    print("\nStarting v2 training loop...")
    global_step = 0

    for epoch in range(1, cfg["epochs"] + 1):
        for real_d, cv, di, si in loader:
            real_d, cv, di, si = real_d.to(device), cv.to(device), di.to(device), si.to(device)
            B = real_d.size(0)

            # A. Train Critic
            for _ in range(cfg["n_critic"]):
                z = torch.randn(B, cfg["noise_dim"], device=device)
                with torch.no_grad():
                    fake_d = G(z, cv, di, si)
                
                real_s = D(real_d, cv, di, si)
                fake_s = D(fake_d, cv, di, si)
                gp = compute_gradient_penalty(D, real_d, fake_d, cv, di, si, device, cfg["lambda_gp"])
                loss_D = critic_loss(real_s, fake_s, gp)

                opt_D.zero_grad()
                loss_D.backward()
                opt_D.step()

            # B. Train Generator
            z = torch.randn(B, cfg["noise_dim"], device=device)
            fake_d = G(z, cv, di, si)
            fake_s = D(fake_d, cv, di, si)
            loss_G = generator_loss(fake_s)

            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

            global_step += 1
            if global_step % cfg["log_every"] == 0:
                w_dist = (real_s.mean() - fake_s.mean()).item()
                writer.add_scalar("Loss/Critic", loss_D.item(), global_step)
                writer.add_scalar("Loss/Generator", loss_G.item(), global_step)
                writer.add_scalar("Distance/Wasserstein", w_dist, global_step)

        print(f"Epoch [{epoch}/{cfg['epochs']}] | Loss_D: {loss_D.item():.4f} | Loss_G: {loss_G.item():.4f}")

        if epoch % cfg["save_every"] == 0 or epoch == cfg["epochs"]:
            path = os.path.join(cfg["checkpoint_dir"], f"epoch_{epoch:04d}.pt")
            torch.save({"G_state": G.state_dict(), "config": cfg}, path)
            # Save a copy as 'final.pt' for the dashboard/generate scripts
            torch.save({"G_state": G.state_dict(), "config": cfg}, os.path.join(cfg["checkpoint_dir"], "final.pt"))

    writer.close()
    print("Training Complete.")

if __name__ == "__main__":
    cfg = get_config()
    # Simple CLI override
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=cfg["epochs"])
    parser.add_argument("--batch", type=int, default=cfg["batch_size"])
    args = parser.parse_args()
    cfg["epochs"] = args.epochs
    cfg["batch_size"] = args.batch
    
    train(cfg)
