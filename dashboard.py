import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add wgan-gp to path to import Generator
sys.path.append(os.path.join(os.getcwd(), "wgan-gp"))
from generator import Generator

# CONFIG
CHECKPOINT_PATH = "checkpoints/final.pt"
DATA_NPZ_PATH = "wgan-gp/tusgan_encoded.npz"
ACT_MID = 50.0

# Labels & Mappings
ACTIVITY_LABELS = {
    0: "Sleep & Personal Care",
    1: "Employment & Related",
    2: "Household & Chores",
    3: "Caregiving",
    4: "Socializing & Leisure",
    5: "Learning",
    7: "Travel / Other"
}

ACTIVITY_COLORS = {
    0: "#1f77b4", # blue
    1: "#ff7f0e", # orange
    2: "#2ca02c", # green
    3: "#d62728", # red
    4: "#9467bd", # purple
    5: "#8c564b", # brown
    7: "#7f7f7f"  # gray
}

AGE_LABELS = [
    "Childhood (<15)",
    "School Students (15-17)",
    "College / Early Work (18-24)",
    "Early Career (25-34)",
    "Mid-Career (35-44)",
    "Later Working (45-59)",
    "Retirement (60+)"
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
    12: "Diploma/Graduate+"
}
EDU_CODES = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12]

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
    97: "Other"
}
ACT_CODES = [11, 12, 21, 31, 41, 51, 81, 91, 92, 93, 94, 95, 97]

DOW_LABELS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
SECTOR_LABELS = ["Rural", "Urban"]

TIME_SLOTS = [
    "04:00","04:30","05:00","05:30","06:00","06:30","07:00","07:30",
    "08:00","08:30","09:00","09:30","10:00","10:30","11:00","11:30",
    "12:00","12:30","13:00","13:30","14:00","14:30","15:00","15:30",
    "16:00","16:30","17:00","17:30","18:00","18:30","19:00","19:30",
    "20:00","20:30","21:00","21:30","22:00","22:30","23:00","23:30",
    "00:00","00:30","01:00","01:30","02:00","02:30","03:00","03:30"
]

@st.cache_resource
def load_model():
    if not os.path.exists(CHECKPOINT_PATH):
        return None, None
    
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
    cfg = ckpt["config"]
    
    # Get num_districts from data
    data = np.load(DATA_NPZ_PATH)
    num_districts = int(data["num_districts"])
    
    G = Generator(
        noise_dim = cfg["noise_dim"],
        cond_dim = cfg["cond_dim"],
        num_districts = num_districts,
        district_embed_dim = cfg["district_embed_dim"],
        base_channels = cfg["g_base_channels"]
    )
    G.load_state_dict(ckpt["G_state"])
    G.eval()
    return G, cfg

def main():
    st.set_page_config(page_title="TUS-GAN Dashboard", layout="wide")
    st.title("📊 TUS-GAN: Synthetic Time-Use Diary Generator")
    
    page = st.sidebar.selectbox("Navigation", ["Generate Diary", "Evaluation Results"])
    
    G, cfg = load_model()
    if G is None:
        st.error(f"Checkpoint not found at {CHECKPOINT_PATH}")
        return

    if page == "Generate Diary":
        st.header("🎨 Generate Synthetic Daily Routine")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Demographics")
            age = st.selectbox("Age Group", AGE_LABELS, index=3)
            gender = st.selectbox("Gender", GENDER_LABELS, index=0)
            marital = st.selectbox("Marital Status", MARITAL_LABELS, index=0)
            edu = st.selectbox("Education Level", list(EDU_LABELS.values()), index=8)
            principal_act = st.selectbox("Principal Activity", list(ACT_LABELS.values()), index=3)
            dow = st.selectbox("Day of Week", DOW_LABELS, index=0)
            sector = st.radio("Sector", SECTOR_LABELS, index=1)
            caregiving = st.checkbox("Needing Special Care / Caregiving required", value=False)
            district_id = st.slider("District ID", 0, 70, 19)
            
            if st.button("Generate Diary", type="primary"):
                # Construct cond_vector
                age_oh = np.eye(7)[AGE_LABELS.index(age)]
                gender_oh = np.eye(3)[GENDER_LABELS.index(gender)]
                marital_oh = np.eye(4)[MARITAL_LABELS.index(marital)]
                
                # Edu one-hot
                edu_idx = [k for k, v in EDU_LABELS.items() if v == edu][0]
                edu_oh = np.eye(11)[EDU_CODES.index(edu_idx)]
                
                # Act one-hot
                act_idx = [k for k, v in ACT_LABELS.items() if v == principal_act][0]
                act_oh = np.eye(13)[ACT_CODES.index(act_idx)]
                
                dow_oh = np.eye(7)[DOW_LABELS.index(dow)]
                sector_oh = np.eye(2)[SECTOR_LABELS.index(sector)]
                care_oh = np.eye(2)[1 if caregiving else 0]
                
                cond_vector = np.concatenate([age_oh, gender_oh, marital_oh, edu_oh, act_oh, dow_oh, sector_oh, care_oh])
                
                # Inference
                with torch.no_grad():
                    z = torch.randn(1, cfg["noise_dim"])
                    cv = torch.from_numpy(cond_vector).float().unsqueeze(0)
                    di = torch.tensor([district_id]).long()
                    
                    fake = G(z, cv, di) # (1, 1, 48, 1)
                    fake = fake.squeeze().numpy()
                    
                # Decode
                decoded = (fake + 1.0) * ACT_MID
                decoded = np.round(decoded).astype(int)
                decoded = np.clip(decoded, 0, 99)
                
                # Map to available labels or fallback
                valid_labels = list(ACTIVITY_LABELS.keys())
                decoded_mapped = [c if c in valid_labels else 7 for c in decoded]
                
                with col2:
                    st.subheader("Generated 24-Hour Diary")
                    
                    fig, ax = plt.subplots(figsize=(10, 4))
                    # Use step plot
                    ax.step(range(48), decoded_mapped, where='post', color='teal', linewidth=2)
                    
                    # Formatting
                    ax.set_ylim(-0.5, 8.5)
                    ax.set_yticks(valid_labels)
                    ax.set_yticklabels([ACTIVITY_LABELS[k] for k in valid_labels])
                    
                    ax.set_xticks(range(0, 48, 4))
                    ax.set_xticklabels([TIME_SLOTS[i] for i in range(0, 48, 4)], rotation=45)
                    
                    ax.grid(axis='y', linestyle='--', alpha=0.3)
                    ax.set_title(f"Synthetic Daily Routine for {gender}, {age}")
                    
                    # Color the background based on activity
                    for i in range(48):
                        code = decoded_mapped[i]
                        color = ACTIVITY_COLORS.get(code, "#f0f0f0")
                        ax.axvspan(i, i+1, color=color, alpha=0.2)
                        
                    st.pyplot(fig)
                    
                    # Legend
                    st.write("**Activity Legend:**")
                    cols = st.columns(4)
                    for i, (code, label) in enumerate(ACTIVITY_LABELS.items()):
                        cols[i % 4].markdown(f"<span style='color:{ACTIVITY_COLORS[code]}'>■</span> {label}", unsafe_allow_html=True)

    else:
        st.header("📊 Model Evaluation Statistics")
        st.write("Comparison between Real ITUS 2019 data and TUS-GAN Synthetic data.")
        
        eval_path = "evaluation_results"
        if os.path.exists(eval_path):
            st.subheader("Activity Distribution")
            if os.path.exists(f"{eval_path}/activity_distribution.png"):
                st.image(f"{eval_path}/activity_distribution.png")
            
            st.subheader("Time Use Comparison")
            if os.path.exists(f"{eval_path}/time_use_comparison.png"):
                st.image(f"{eval_path}/time_use_comparison.png")
                
            st.subheader("Sample Diaries")
            if os.path.exists(f"{eval_path}/sample_diaries.png"):
                st.image(f"{eval_path}/sample_diaries.png")
        else:
            st.warning("Evaluation results not found. Run `wgan-gp/evaluate.py` first.")

if __name__ == "__main__":
    main()
