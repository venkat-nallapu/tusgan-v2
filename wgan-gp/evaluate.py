
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
REAL_DATA_PATH = "wgan-gp/tusgan_encoded.npz"
FAKE_DATA_PATH = "synthetic_diaries.csv"
OUTPUT_DIR     = "evaluation_results"
ACT_MID        = 50.0

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load Real Data and Decode
    print(f"Loading real data from {REAL_DATA_PATH}...")
    real_data = np.load(REAL_DATA_PATH)
    real_tensors = real_data["diary_tensor"].squeeze() # (N, 48)
    real_codes = (real_tensors + 1.0) * ACT_MID
    real_codes = np.round(real_codes).astype(int)
    
    # 2. Load Fake Data
    print(f"Loading synthetic data from {FAKE_DATA_PATH}...")
    if not os.path.exists(FAKE_DATA_PATH):
        print(f"Error: {FAKE_DATA_PATH} not found. Run generate.py first.")
        return
    df_fake = pd.read_csv(FAKE_DATA_PATH)
    fake_codes = df_fake.values
    
    # 3. Compare Global Activity Frequencies
    print("Comparing activity frequencies...")
    real_flat = real_codes.flatten()
    fake_flat = fake_codes.flatten()
    
    unique_real, counts_real = np.unique(real_flat, return_counts=True)
    unique_fake, counts_fake = np.unique(fake_flat, return_counts=True)
    
    # Normalise to percentages
    counts_real = counts_real / len(real_flat) * 100
    counts_fake = counts_fake / len(fake_flat) * 100
    
    # Plot Histograms
    plt.figure(figsize=(12, 6))
    plt.bar(unique_real - 0.2, counts_real, width=0.4, label="Real", alpha=0.7)
    plt.bar(unique_fake + 0.2, counts_fake, width=0.4, label="Synthetic", alpha=0.7)
    plt.xlabel("Activity Code")
    plt.ylabel("Frequency (%)")
    plt.title("Activity Code Distribution: Real vs Synthetic")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(OUTPUT_DIR, "activity_distribution.png"))
    print(f"Saved distribution plot to {OUTPUT_DIR}/activity_distribution.png")
    
    # 4. Average Time Spent per Activity
    print("Calculating average time spent...")
    # Assume 30 min per slot
    real_time = np.zeros(100)
    fake_time = np.zeros(100)
    
    for c in range(100):
        real_time[c] = np.mean(np.sum(real_codes == c, axis=1)) * 30
        fake_time[c] = np.mean(np.sum(fake_codes == c, axis=1)) * 30
        
    # Plot comparison for present activities
    active_codes = np.unique(np.concatenate([unique_real, unique_fake]))
    plt.figure(figsize=(12, 6))
    plt.bar(active_codes - 0.2, real_time[active_codes], width=0.4, label="Real", alpha=0.7)
    plt.bar(active_codes + 0.2, fake_time[active_codes], width=0.4, label="Synthetic", alpha=0.7)
    plt.xlabel("Activity Code")
    plt.ylabel("Avg Minutes per Day")
    plt.title("Time Use Comparison: Real vs Synthetic")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(OUTPUT_DIR, "time_use_comparison.png"))
    print(f"Saved time use comparison to {OUTPUT_DIR}/time_use_comparison.png")
    
    # 5. Visualise some diaries
    print("Visualising sample diaries...")
    fig, axes = plt.subplots(2, 5, figsize=(20, 8), sharex=True, sharey=True)
    for i in range(5):
        axes[0, i].step(range(48), real_codes[i], where='post', color='blue', alpha=0.7)
        axes[0, i].set_title(f"Real Diary {i+1}")
        axes[1, i].step(range(48), fake_codes[i], where='post', color='red', alpha=0.7)
        axes[1, i].set_title(f"Synthetic Diary {i+1}")
    
    for ax in axes.flatten():
        ax.set_ylim(-0.5, max(active_codes) + 0.5)
        ax.set_yticks(active_codes)
        
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "sample_diaries.png"))
    print(f"Saved sample diaries to {OUTPUT_DIR}/sample_diaries.png")
    
    print("Evaluation complete.")

if __name__ == "__main__":
    main()
