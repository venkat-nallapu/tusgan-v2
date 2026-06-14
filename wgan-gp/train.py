
"""
TUS-GAN — Training Script (WGAN-GP v2)
=======================================
Trains the conditional Generator and Critic using the
Wasserstein GAN with Gradient Penalty objective.

v2 Updates:
  - Dynamic cond_dim and num_districts/num_states.
  - Dual embedding support (District + State).
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

from generator import Generator
from critic    import Critic, compute_gradient_penalty, critic_loss, generator_loss


class TUSDataset(Dataset):
    def __init__(self, npz_path: str, subset_size: int = None):
        data = np.load(npz_path)
        diary = torch.from_numpy(data["diary_tensor"]).float()
        cond = torch.from_numpy(data["cond_vector"]).float()
        dist_ids = torch.from_numpy(data["district_ids"]).long()
        state_ids = torch.from_numpy(data["state_ids"]).long()

        if subset_size is not None and subset_size > 0:
            # Slice first N samples for fast local prototyping
            diary = diary[:subset_size]
            cond = cond[:subset_size]
            dist_ids = dist_ids[:subset_size]
            state_ids = state_ids[:subset_size]

        self.diary = diary
        self.cond = cond
        self.dist_ids = dist_ids
        self.state_ids = state_ids

        self.num_districts = int(data["num_districts"])
        self.num_states    = int(data["num_states"])
        self.cond_dim      = self.cond.shape[1]

    def __len__(self):
        return len(self.diary)

    def __getitem__(self, idx):
        return self.diary[idx], self.cond[idx], self.dist_ids[idx], self.state_ids[idx]


def train(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Resolve data path
    data_path = cfg["data_path"]
    if not os.path.exists(data_path) and os.path.exists(os.path.join("wgan-gp", data_path)):
        data_path = os.path.join("wgan-gp", data_path)
        print(f"Resolving dataset path to: {data_path}")

    dataset = TUSDataset(data_path, subset_size=cfg.get("subset"))
    cfg["cond_dim"] = dataset.cond_dim

    # Create checkpoints directory if it doesn't exist
    os.makedirs("checkpoints", exist_ok=True)

    # Optimize data loading for GPUs (pin_memory and num_workers)
    num_workers = min(4, os.cpu_count() or 2)
    loader = DataLoader(
        dataset, 
        batch_size=cfg["batch_size"], 
        shuffle=True, 
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )


    G = Generator(
        noise_dim          = cfg["noise_dim"],
        cond_dim           = dataset.cond_dim,
        num_districts      = dataset.num_districts,
        num_states         = dataset.num_states,
    ).to(device)

    D = Critic(
        cond_dim           = dataset.cond_dim,
        num_districts      = dataset.num_districts,
        num_states         = dataset.num_states,
    ).to(device)

    opt_G = torch.optim.Adam(G.parameters(), lr=cfg["lr"], betas=(0.0, 0.9))
    opt_D = torch.optim.Adam(D.parameters(), lr=cfg["lr"], betas=(0.0, 0.9))

    # Add learning rate schedulers to stabilize late training phase
    sched_G = torch.optim.lr_scheduler.StepLR(opt_G, step_size=30, gamma=0.5)
    sched_D = torch.optim.lr_scheduler.StepLR(opt_D, step_size=30, gamma=0.5)

    fixed_batch = next(iter(DataLoader(dataset, batch_size=16, shuffle=True)))
    fz = torch.randn(16, cfg["noise_dim"], device=device)
    fc = fixed_batch[1].to(device)
    fd = fixed_batch[2].to(device)
    fs = fixed_batch[3].to(device)

    writer = SummaryWriter("runs/tusgan-v2")

    for epoch in range(1, cfg["epochs"] + 1):
        loss_D_accum = 0.0
        loss_G_accum = 0.0
        gp_accum = 0.0
        batches = 0

        for real_d, cv, di, si in loader:
            real_d, cv, di, si = real_d.to(device), cv.to(device), di.to(device), si.to(device)
            B = real_d.size(0)

            # Train Critic
            for _ in range(5):
                z = torch.randn(B, cfg["noise_dim"], device=device)
                with torch.no_grad():
                    fake_d = G(z, cv, di, si)
                
                real_s = D(real_d, cv, di, si)
                fake_s = D(fake_d, cv, di, si)
                gp = compute_gradient_penalty(D, real_d, fake_d, cv, di, si, device)
                loss_D = critic_loss(real_s, fake_s, gp)

                opt_D.zero_grad()
                loss_D.backward()
                opt_D.step()

            # Train Generator
            z = torch.randn(B, cfg["noise_dim"], device=device)
            fake_d = G(z, cv, di, si)
            fake_s = D(fake_d, cv, di, si)
            loss_G = generator_loss(fake_s)

            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

            # Accumulate metrics for logging
            loss_D_accum += loss_D.item()
            loss_G_accum += loss_G.item()
            gp_accum += gp.item()
            batches += 1

        sched_G.step()
        sched_D.step()

        # Compute average epoch statistics
        avg_loss_D = loss_D_accum / batches
        avg_loss_G = loss_G_accum / batches
        avg_gp = gp_accum / batches

        print(f"Epoch {epoch:03d}/{cfg['epochs']:03d} | Loss_D: {avg_loss_D:.4f} | Loss_G: {avg_loss_G:.4f} | GP: {avg_gp:.4f}")
        
        # Tensorboard scalar logs
        writer.add_scalar("Loss/Critic", avg_loss_D, epoch)
        writer.add_scalar("Loss/Generator", avg_loss_G, epoch)
        writer.add_scalar("Loss/GradientPenalty", avg_gp, epoch)
        writer.add_scalar("LearningRate/Generator", sched_G.get_last_lr()[0], epoch)
        writer.add_scalar("LearningRate/Critic", sched_D.get_last_lr()[0], epoch)
        
        # Periodic evaluation visual tracking (Heatmap of synthetic routine vs real)
        if epoch % 10 == 0:
            with torch.no_grad():
                fakes = G(fz, fc, fd, fs).cpu().numpy() # Shape: (16, 9, 48, 1)
            
            # Create comparison figure
            fig, axes = plt.subplots(2, 1, figsize=(10, 6))
            # Plot first fake sample diary probabilities/activations
            im0 = axes[0].imshow(fakes[0, :, :, 0], aspect='auto', cmap='viridis')
            axes[0].set_title("Generated Synthetic Diary (Sample 0)")
            axes[0].set_ylabel("Divisions (1-9)")
            
            # Plot real diary from fixed batch for reference
            im1 = axes[1].imshow(fixed_batch[0][0, :, :, 0].numpy(), aspect='auto', cmap='viridis')
            axes[1].set_title("Real Reference Diary (Sample 0)")
            axes[1].set_ylabel("Divisions (1-9)")
            axes[1].set_xlabel("Time Slots (30-min intervals starting 04:00 AM)")
            
            fig.colorbar(im0, ax=axes[0])
            fig.colorbar(im1, ax=axes[1])
            plt.tight_layout()
            
            writer.add_figure("VisualComparison/DiaryHeatmap", fig, epoch)
            plt.close(fig)

            # Save checkoint
            torch.save({"G_state": G.state_dict(), "config": cfg}, f"checkpoints/v2_epoch_{epoch}.pt")

    writer.close()
    print("Training finished. Checkpoints and tensorboard logs saved successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TUS-GAN v2 Training Script")
    parser.add_argument("--data-path", type=str, default="tusgan_encoded.npz", help="Path to encoded dataset NPZ")
    parser.add_argument("--noise-dim", type=int, default=128, help="Latent noise vector dimensionality")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (Adam)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--subset", type=int, default=None, help="If set, only trains on first N samples for fast testing")
    
    args = parser.parse_args()
    
    cfg = {
        "data_path": args.data_path,
        "noise_dim": args.noise_dim,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "epochs": args.epochs,
        "subset": args.subset
    }
    train(cfg)

