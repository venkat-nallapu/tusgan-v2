
import os
import numpy as np
import torch
import pandas as pd
from generator import Generator

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
CHECKPOINT_PATH = "checkpoints/final.pt"
DATA_NPZ_PATH   = "wgan-gp/tusgan_encoded.npz"
OUTPUT_CSV      = "synthetic_diaries.csv"
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Activity code decoding
ACT_MID = 50.0

TIME_COLS = [
    "04:00-04:30","04:30-05:00","05:00-05:30","05:30-06:00",
    "06:00-06:30","06:30-07:00","07:00-07:30","07:30-08:00",
    "08:00-08:30","08:30-09:00","09:00-09:30","09:30-10:00",
    "10:00-10:30","10:30-11:00","11:00-11:30","11:30-12:00",
    "12:00-12:30","12:30-13:00","13:00-13:30","13:30-14:00",
    "14:00-14:30","14:30-15:00","15:00-15:30","15:30-16:00",
    "16:00-16:30","16:30-17:00","17:00-17:30","17:30-18:00",
    "18:00-18:30","18:30-19:00","19:00-19:30","19:30-20:00",
    "20:00-20:30","20:30-21:00","21:00-21:30","21:30-22:00",
    "22:00-22:30","22:30-23:00","23:00-23:30","23:30-00:00",
    "00:00-00:30","00:30-01:00","01:00-01:30","01:30-02:00",
    "02:00-02:30","02:30-03:00","03:00-03:30","03:30-04:00",
]

def main():
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Error: Checkpoint not found at {CHECKPOINT_PATH}")
        return

    # 1. Load data to get demographics and dimensions
    print(f"Loading data from {DATA_NPZ_PATH}...")
    data = np.load(DATA_NPZ_PATH)
    cond_vector = torch.from_numpy(data["cond_vector"]).float()
    district_ids = torch.from_numpy(data["district_ids"]).long()
    num_districts = int(data["num_districts"])
    
    # 2. Instantiate and load Generator
    print(f"Loading Generator from {CHECKPOINT_PATH}...")
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    cfg = ckpt["config"]
    
    G = Generator(
        noise_dim          = cfg["noise_dim"],
        cond_dim           = cfg["cond_dim"],
        num_districts      = num_districts,
        district_embed_dim = cfg["district_embed_dim"],
        base_channels      = cfg["g_base_channels"],
    ).to(DEVICE)
    
    G.load_state_dict(ckpt["G_state"])
    G.eval()
    
    # 3. Generate
    print(f"Generating {len(cond_vector)} synthetic diaries...")
    batch_size = 512
    all_fakes = []
    
    with torch.no_grad():
        for i in range(0, len(cond_vector), batch_size):
            batch_cond = cond_vector[i:i+batch_size].to(DEVICE)
            batch_dist = district_ids[i:i+batch_size].to(DEVICE)
            B = batch_cond.size(0)
            
            z = torch.randn(B, cfg["noise_dim"], device=DEVICE)
            fake = G(z, batch_cond, batch_dist) # (B, 1, 48, 1)
            all_fakes.append(fake.cpu().numpy())
            
    all_fakes = np.concatenate(all_fakes, axis=0)
    all_fakes = all_fakes.squeeze() # (N, 48)
    
    # 4. Decode
    print("Decoding activity codes...")
    # Reverse normalisation: code = (norm + 1.0) * ACT_MID
    decoded_codes = (all_fakes + 1.0) * ACT_MID
    # Round to nearest integer (assuming codes are integers)
    decoded_codes = np.round(decoded_codes).astype(int)
    # Clip to valid range
    decoded_codes = np.clip(decoded_codes, 0, 99)
    
    # 5. Save to CSV
    print(f"Saving to {OUTPUT_CSV}...")
    df_fake = pd.DataFrame(decoded_codes, columns=TIME_COLS)
    df_fake.to_csv(OUTPUT_CSV, index=False)
    print("Done.")

if __name__ == "__main__":
    main()
