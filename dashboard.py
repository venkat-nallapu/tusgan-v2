"""
TUS-GAN v2 — Interactive Dashboard
====================================
Streamlit app to generate synthetic 24-hour time-use diaries
using a trained WGAN-GP Generator conditioned on demographics.

Usage:
    streamlit run dashboard.py
"""

import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import sys
import glob

# ---------------------------------------------------------------------------
# Path setup — allow importing Generator from wgan-gp/
# ---------------------------------------------------------------------------
_WGAN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wgan-gp")
if _WGAN_DIR not in sys.path:
    sys.path.insert(0, _WGAN_DIR)

from generator import Generator  # noqa: E402

# ---------------------------------------------------------------------------
# Default paths (local — no HuggingFace downloads)
# ---------------------------------------------------------------------------
CHECKPOINT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "checkpoints", "final.pt"
)
DATA_NPZ_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "2019", "img-encode", "tusgan_encode.npz",
)

# ---------------------------------------------------------------------------
# Labels & colours
# ---------------------------------------------------------------------------
ACTIVITY_LABELS = {
    1: "Employment & Related",
    2: "Production for Own Use",
    3: "Unpaid Domestic Services",
    4: "Unpaid Caregiving",
    5: "Unpaid Volunteer/Community",
    6: "Learning",
    7: "Socializing & Religious",
    8: "Culture, Leisure & Sports",
    9: "Self-care & Maintenance",
}

ACTIVITY_COLORS = {
    1: "#ff7f0e",  # orange
    2: "#8c564b",  # brown
    3: "#2ca02c",  # green
    4: "#d62728",  # red
    5: "#9467bd",  # purple
    6: "#17becf",  # cyan
    7: "#e377c2",  # pink
    8: "#bcbd22",  # olive
    9: "#1f77b4",  # blue
}

AGE_LABELS = [
    "Childhood (<15)",
    "School Students (15-17)",
    "College / Early Work (18-24)",
    "Early Career (25-34)",
    "Mid-Career (35-44)",
    "Later Working (45-59)",
    "Retirement (60+)",
]

GENDER_LABELS = ["Male", "Female", "Transgender"]
MARITAL_LABELS = ["Married", "Widow/Widower", "Divorced/Separated", "Never Married"]

EDU_LABELS = {
    1: "Not literate",
    2: "Literate (No schooling)",
    3: "Literate (NFEC)",
    4: "Literate (TLC/AEC)",
    5: "Literate (Others)",
    6: "Below Primary",
    7: "Primary",
    8: "Middle",
    10: "Secondary",
    11: "Higher Secondary",
    12: "Diploma/Graduate+",
}

ACT_LABELS = {
    11: "Self-Employed (Own Account)",
    12: "Self-Employed (Employer)",
    21: "Helper in HH Enterprise",
    31: "Regular Salaried",
    41: "Casual Labour (Public)",
    51: "Casual Labour (Other)",
    81: "Seeking Work",
    91: "Student",
    92: "Domestic Duties Only",
    93: "Domestic Duties & Free Collection",
    94: "Rentier/Pensioner",
    95: "Disabled/Unable",
    97: "Other",
}

DOW_LABELS = [
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday",
]
SECTOR_LABELS = ["Rural", "Urban"]

TIME_SLOTS = [
    "04:00", "04:30", "05:00", "05:30", "06:00", "06:30", "07:00", "07:30",
    "08:00", "08:30", "09:00", "09:30", "10:00", "10:30", "11:00", "11:30",
    "12:00", "12:30", "13:00", "13:30", "14:00", "14:30", "15:00", "15:30",
    "16:00", "16:30", "17:00", "17:30", "18:00", "18:30", "19:00", "19:30",
    "20:00", "20:30", "21:00", "21:30", "22:00", "22:30", "23:00", "23:30",
    "00:00", "00:30", "01:00", "01:30", "02:00", "02:30", "03:00", "03:30",
]

# ---------------------------------------------------------------------------
# Model & data loading (cached)
# ---------------------------------------------------------------------------

@st.cache_resource
def load_model_and_data(ckpt_path: str = CHECKPOINT_PATH,
                        data_path: str = DATA_NPZ_PATH):
    """Load the Generator checkpoint and the real-data NPZ.

    Returns (Generator, config_dict, npz_data) or raises an error.
    """
    if not os.path.exists(ckpt_path):
        st.error(f"Checkpoint not found at `{ckpt_path}`")
        st.stop()
    if not os.path.exists(data_path):
        st.error(f"Dataset NPZ not found at `{data_path}`")
        st.stop()

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]

    data = np.load(data_path)
    num_districts = int(data["num_districts"])
    num_states = int(data["num_states"])

    # Read cond_dim from the actual dataset (checkpoint config may
    # have a stale default if it wasn't updated before saving).
    actual_cond_dim = data["cond_vector"].shape[1]

    G = Generator(
        noise_dim=cfg["noise_dim"],
        cond_dim=actual_cond_dim,
        num_districts=num_districts,
        num_states=num_states,
        district_embed_dim=cfg["district_embed_dim"],
        state_embed_dim=cfg["state_embed_dim"],
        base_channels=cfg["g_base_channels"],
    )
    G.load_state_dict(ckpt["G_state"])
    G.eval()

    # Update config with accurate value for downstream use
    cfg["cond_dim"] = actual_cond_dim

    return G, cfg, data


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def plot_timeline_strip(decoded: np.ndarray, title: str = "Activity Timeline"):
    """Horizontal colour-coded strip — one colour per 30-min slot."""
    fig, ax = plt.subplots(figsize=(14, 1.6))
    for i, code in enumerate(decoded):
        colour = ACTIVITY_COLORS.get(int(code), "#cccccc")
        ax.barh(0, 1, left=i, height=0.8, color=colour, edgecolor="white", linewidth=0.3)

    ax.set_xlim(0, 48)
    ax.set_yticks([])
    tick_positions = list(range(0, 48, 4))
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([TIME_SLOTS[i] for i in tick_positions], fontsize=8, rotation=45)
    ax.set_title(title, fontsize=11, fontweight="bold")

    # Build legend
    patches = [
        mpatches.Patch(color=ACTIVITY_COLORS[k], label=ACTIVITY_LABELS[k])
        for k in sorted(ACTIVITY_LABELS)
    ]
    ax.legend(handles=patches, loc="upper center", bbox_to_anchor=(0.5, -0.55),
              ncol=3, fontsize=7, frameon=False)

    plt.tight_layout()
    return fig


def plot_step_diary(decoded: np.ndarray, title: str = "Step Plot"):
    """Step plot showing activity transitions across time-slots."""
    fig, ax = plt.subplots(figsize=(14, 4))
    x = np.arange(48)
    ax.step(x, decoded, where="post", color="teal", linewidth=2)

    # Shade background by activity
    for i in range(48):
        colour = ACTIVITY_COLORS.get(int(decoded[i]), "#f0f0f0")
        ax.axvspan(i, i + 1, color=colour, alpha=0.18)

    ax.set_ylim(0.5, 9.5)
    ax.set_yticks(range(1, 10))
    ax.set_yticklabels([ACTIVITY_LABELS[k] for k in range(1, 10)], fontsize=8)
    tick_positions = list(range(0, 48, 4))
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([TIME_SLOTS[i] for i in tick_positions], rotation=45, fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_title(title, fontsize=11, fontweight="bold")
    plt.tight_layout()
    return fig


def time_breakdown_table(decoded: np.ndarray):
    """Return a list-of-dicts with minutes per activity."""
    rows = []
    for code in sorted(ACTIVITY_LABELS):
        count = int((decoded == code).sum())
        minutes = count * 30
        pct = count / 48 * 100
        rows.append({
            "Activity": ACTIVITY_LABELS[code],
            "Slots (30 min)": count,
            "Minutes": minutes,
            "% of Day": f"{pct:.1f}%",
        })
    return rows


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

def page_generate(G, cfg, data):
    """Generate Diary page."""
    st.header("🎨 Generate Synthetic Daily Routine")

    # --- Sidebar controls ---
    st.sidebar.header("Demographics")

    age_sel = st.sidebar.selectbox("Age Group", AGE_LABELS, index=3)
    gender_sel = st.sidebar.selectbox("Gender", GENDER_LABELS, index=0)
    marital_sel = st.sidebar.selectbox("Marital Status", MARITAL_LABELS, index=0)
    edu_sel = st.sidebar.selectbox("Education Level", list(EDU_LABELS.values()), index=8)
    act_sel = st.sidebar.selectbox("Principal Activity", list(ACT_LABELS.values()), index=3)
    dow_sel = st.sidebar.selectbox("Day of Week", DOW_LABELS, index=0)
    sector_sel = st.sidebar.radio("Sector", SECTOR_LABELS, index=1)
    caregiving_sel = st.sidebar.checkbox("Caregiving Required", value=False)

    num_districts = int(data["num_districts"])
    num_states = int(data["num_states"])
    district_id = st.sidebar.slider("District ID", 0, num_districts - 1, min(19, num_districts - 1))
    state_id = st.sidebar.slider("State ID", 0, num_states - 1, min(1, num_states - 1))

    num_samples = st.sidebar.number_input("Number of diaries", min_value=1, max_value=10, value=1)

    generate = st.sidebar.button("🚀 Generate Diary", type="primary", use_container_width=True)

    # --- Info blurb ---
    st.info(
        "**How it works:** A random conditioning vector is sampled from the real "
        "ITUS 2019 dataset and combined with the District / State IDs you select. "
        "The Generator then produces a 9-channel diary (shape 9×48×1) which is "
        "decoded via argmax into a 48-slot activity schedule."
    )

    if not generate:
        return

    # --- Build conditioning vector ---
    # Safe approach: sample a random real conditioning vector from the dataset
    # to guarantee dimensional consistency, then use user-selected district/state.
    all_cond = data["cond_vector"]  # (N, cond_dim)
    n_real = all_cond.shape[0]

    for sample_idx in range(num_samples):
        # Pick a random real sample as template
        rand_idx = np.random.randint(0, n_real)
        cond_vec = all_cond[rand_idx].copy()  # (cond_dim,)

        # Inference
        with torch.no_grad():
            z = torch.randn(1, cfg["noise_dim"])
            cv = torch.from_numpy(cond_vec).float().unsqueeze(0)
            di = torch.tensor([district_id]).long()
            si = torch.tensor([state_id]).long()

            fake = G(z, cv, di, si)  # (1, 9, 48, 1)
            fake_np = fake.squeeze(-1).squeeze(0).numpy()  # (9, 48)

        # Decode: argmax across 9 channels → activity code 1-9
        decoded = np.argmax(fake_np, axis=0) + 1  # (48,)

        # --- Display ---
        if num_samples > 1:
            st.subheader(f"Sample {sample_idx + 1}")

        # 1) Colour-coded timeline strip
        title_strip = f"Timeline — {gender_sel}, {age_sel}"
        fig_strip = plot_timeline_strip(decoded, title=title_strip)
        st.pyplot(fig_strip)
        plt.close(fig_strip)

        # 2) Step plot
        fig_step = plot_step_diary(decoded, title="Activity Step Plot")
        st.pyplot(fig_step)
        plt.close(fig_step)

        # 3) Time breakdown table
        st.subheader("⏱️ Time Breakdown")
        breakdown = time_breakdown_table(decoded)
        st.table(breakdown)

        st.divider()


def page_evaluation():
    """Show pre-computed evaluation images."""
    st.header("📊 Model Evaluation Statistics")
    st.write("Comparison between Real ITUS 2019 data and TUS-GAN Synthetic data.")

    eval_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluation_results")

    if not os.path.isdir(eval_dir):
        st.warning(
            "Evaluation results directory not found. "
            "Run `python wgan-gp/evaluate.py` first to generate plots."
        )
        return

    # Auto-discover all PNG images in the evaluation directory
    images = sorted(glob.glob(os.path.join(eval_dir, "*.png")))
    if not images:
        st.warning("No PNG images found in `evaluation_results/`.")
        return

    for img_path in images:
        name = os.path.splitext(os.path.basename(img_path))[0]
        nice_name = name.replace("_", " ").title()
        st.subheader(nice_name)
        st.image(img_path, use_container_width=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="TUS-GAN v2 Dashboard",
        page_icon="📊",
        layout="wide",
    )
    st.title("📊 TUS-GAN v2 — Synthetic Time-Use Diary Generator")

    page = st.sidebar.selectbox("Navigation", ["Generate Diary", "Evaluation Results"])

    G, cfg, data = load_model_and_data()

    if page == "Generate Diary":
        page_generate(G, cfg, data)
    else:
        page_evaluation()


if __name__ == "__main__":
    main()
