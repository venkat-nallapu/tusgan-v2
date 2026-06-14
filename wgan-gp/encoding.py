import numpy as np
import pandas as pd
import os

def main():
    # CONFIG
    CSV_PATH   = "2019/data/ITUS_2019.csv"
    SAVE_PATH  = "2019/img-encode/tusgan_encode.npz"

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

    # 1. Load Data
    print("Step 1: Loading and Cleaning Headers...")
    if not os.path.exists(CSV_PATH):
        print(f"Error: Could not find dataset at {CSV_PATH}")
        return
        
    df = pd.read_csv(CSV_PATH, low_memory=False)

    # Force clean headers
    df.columns = [str(c).strip().replace('\r', '') for c in df.columns]

    EXP_COL = "usual monthly consumer expenditure E: [A+B+C+(D/12)]"
    COLS_REQUIRED = [
        "Gender", "Marital_Status", "Highest_level_of_education",
        "usual principal activity: status (code)", "age", "Sector", 
        "District", "Household size", "caregiving_dummy", "State", "day of week", EXP_COL
    ]

    # Verify all columns exist
    missing = [c for c in COLS_REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    # 2. Cast to Numeric and Drop Missing
    print("Step 2: Casting to Numeric and Filtering...")
    for col in COLS_REQUIRED + TIME_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows where conditioning features are NaN
    before_len = len(df)
    df = df.dropna(subset=COLS_REQUIRED)
    print(f"Dropped {before_len - len(df):,} rows with missing demographics.")

    # Forward-fill time gaps, then fill any remaining with Sleep (911)
    df[TIME_COLS] = df[TIME_COLS].ffill(axis=1).fillna(911)
    print(f"Final dataset size: {len(df):,}")

    print("Step 3: Encoding 9-Channel Diary...")
    # Division = floor(ActivityCode / 100)
    diary_vals = df[TIME_COLS].values
    diary_divisions = (diary_vals // 100).astype(int)
    diary_divisions = np.clip(diary_divisions, 1, 9)

    N = len(df)
    diary_onehot = np.zeros((N, 9, 48), dtype=np.float32)

    # Vectorized one-hot mapping
    n_idx = np.arange(N).reshape(-1, 1)
    t_idx = np.arange(48)
    diary_onehot[n_idx, diary_divisions - 1, t_idx] = 1.0

    # Scale to [-1, 1] for GAN compatibility
    diary_tensor = (diary_onehot * 2.0) - 1.0
    diary_tensor = diary_tensor.reshape(N, 9, 48, 1)

    print(f"Diary Tensor: {diary_tensor.shape}, Range: [{diary_tensor.min()}, {diary_tensor.max()}]")

    print("Step 4: Encoding Conditioning Features (v2)...")

    def safe_onehot(series, bins=None, prefix=""):
        """Robustly creates a one-hot matrix from a Series."""
        if series is None or len(series) == 0:
            raise ValueError("Empty or None series passed to safe_onehot")
            
        if bins is not None:
            # Use binned indexing
            indices = np.digitize(series.values, bins)
            n_classes = len(bins) + 1
            return np.eye(n_classes, dtype=np.float32)[indices]
        else:
            # Use Pandas get_dummies for categorical encoding
            # This is safer than manual mapping
            oh = pd.get_dummies(series, prefix=prefix).values.astype(np.float32)
            return oh

    try:
        # Standard demographics
        age_oh     = safe_onehot(df["age"], bins=[15, 18, 25, 35, 45, 60])
        gender_oh  = safe_onehot(df["Gender"])
        marital_oh = safe_onehot(df["Marital_Status"])
        edu_oh     = safe_onehot(df["Highest_level_of_education"])
        act_oh     = safe_onehot(df["usual principal activity: status (code)"])
        dow_oh     = safe_onehot(df["day of week"])
        sector_oh  = safe_onehot(df["Sector"])
        care_oh    = safe_onehot(df["caregiving_dummy"])
        hh_size_oh = safe_onehot(df["Household size"])

        # Expenditure deciles
        log_exp = np.log10(df[EXP_COL].values + 1.0)
        exp_bins = np.percentile(log_exp, np.linspace(0, 100, 11)[1:-1])
        exp_indices = np.digitize(log_exp, exp_bins)
        exp_oh = np.eye(10, dtype=np.float32)[exp_indices]

        # Final concat
        cond_vector = np.concatenate([
            age_oh, gender_oh, marital_oh, edu_oh, act_oh, dow_oh, sector_oh, care_oh, 
            hh_size_oh, exp_oh
        ], axis=1)

        # IDs for Embeddings
        def get_clean_ids(series):
            unique_vals = sorted(series.unique())
            mapping = {v: i for i, v in enumerate(unique_vals)}
            return series.map(mapping).values.astype(np.int64), len(unique_vals)

        district_ids, num_districts = get_clean_ids(df["District"])
        state_ids, num_states       = get_clean_ids(df["State"])

        print(f"One-hot Dims: {cond_vector.shape[1]}")
        print(f"Districts: {num_districts}, States: {num_states}")
    except Exception as e:
        print(f"CRITICAL ERROR in Step 4: {e}")
        # Print diagnostics
        for col in COLS_REQUIRED:
            if col in df.columns:
                print(f"  Column '{col}': type={df[col].dtype}, nulls={df[col].isna().sum()}")
        raise e

    print(f"Saving to {SAVE_PATH}...")
    np.savez_compressed(
        SAVE_PATH,
        diary_tensor  = diary_tensor,
        cond_vector   = cond_vector,
        district_ids  = district_ids,
        state_ids     = state_ids,
        num_districts = np.array(num_districts),
        num_states    = np.array(num_states)
    )
    print("Success! ✓✓")

if __name__ == "__main__":
    main()
