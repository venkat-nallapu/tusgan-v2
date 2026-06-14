# TUS-GAN: Time Use Survey Generative Adversarial Network (v2)

TUS-GAN is a conditional generative model based on the **WGAN-GP (Wasserstein GAN with Gradient Penalty)** architecture, designed to synthesize realistic 24-hour activity diaries for individuals based on their demographic characteristics.

---

## 🚀 Version 2 (v2) Advancements & Architecture Updates

Version 2 introduces major improvements in network stability, sequence coherence, and training flexibility:

### 1. Advanced Architecture Upgrades
* **Self-Attention Mechanism**: Integrated `SelfAttention2d` blocks inside both the Generator and Critic to capture long-range temporal correlations (e.g., how morning wake-up times correlate with bedtime).
* **Spectral Normalization**: Wrapped Critic convolutional layers and final linear output in Spectral Norm to stabilize the Lipschitz constraint in combination with Gradient Penalty.
* **Residual Connections**: Upgraded `UpsampleBlock` (Generator) and `DownsampleBlock` (Critic) with residual skip paths to prevent gradient decay and improve learning speed.
* **Dual Learned Embeddings**: High-cardinality geographical inputs (**State Codes** and **District Codes**) are mapped to low-dimensional continuous embedding layers (8 and 16 dimensions).

### 2. Enhanced Data Orientation
The model processes the dataset structured as:
* **9-Channel Diary Representation**: Transitioned from a single-intensity sequence to a 9-channel one-hot representation, mapping to the **9 Major Activity Divisions**.
* **Dimensions**: Outputs and inputs are shaped as `(Batch, 9, 48, 1)` where $48$ represents 30-minute intervals covering a 24-hour day starting from 04:00 AM.
* **Conditions Matrix**: The demographics vector $c$ includes binned age, gender, marital status, education level, activity status, day of week, sector, household size, consumer expenditure, and caregiving dummy variables.

---

## 📊 Dataset Keys & Schema

The processed v2 dataset is stored in [2019/img-encode/tusgan_encode.npz](file:///home/venkat/projects/tusgan-v2/2019/img-encode/tusgan_encode.npz) and contains:

| Key | Tensor Shape / Type | Description |
| :--- | :--- | :--- |
| `diary_tensor` | `(445268, 9, 48, 1)` | Float32, scaled to `[-1, 1]` for GAN training |
| `cond_vector` | `(445268, 83)` | One-hot encoded demographic features |
| `district_ids` | `(445268,)` | Long integers (0-70) mapping to districts |
| `state_ids` | `(445268,)` | Long integers (0-35) mapping to states |
| `num_districts`| Scalar | Total count of districts (71) |
| `num_states` | Scalar | Total count of states (36) |

---

## 🛠️ Pipeline & Training Adjustments

* **TensorBoard Integration**: Fully tracks average `Loss/Critic`, `Loss/Generator`, and `Loss/GradientPenalty` per epoch, as well as learning rates.
* **Visual Evaluation**: Every 10 epochs, a Matplotlib figure comparing a generated synthetic diary heatmap with a real reference diary heatmap is logged directly to TensorBoard.
* **Step Learning Rate Scheduler**: Schedulers decrease learning rates by half every 30 epochs (`StepLR(step_size=30, gamma=0.5)`) to stabilize late-stage convergence.
* **CLI Arguments & Argument Parsing**: Standardized parameter passing for customizing epochs, learning rates, batch sizes, and dataset paths.
* **Subset Debugging Flag**: Added a `--subset <N>` parameter to train on only the first $N$ samples of the dataset. This allows fast, low-overhead testing on CPU platforms.

---

## 🚀 Setup & Execution

### 1. Install Dependencies
Ensure PyTorch, NumPy, Pandas, Matplotlib, and TensorBoard are installed:
```bash
source .venv-wgan/bin/activate
pip install -r requirements.txt
```

### 2. Fast CPU Prototyping
Verify the pipeline works by running 5 epochs on a subset of 5,000 samples:
```bash
python wgan-gp/train.py --data-path 2019/img-encode/tusgan_encode.npz --epochs 5 --batch-size 256 --subset 5000
```

### 3. Full GPU Training
Train the complete model with optimized batch configurations on a CUDA GPU:
```bash
python wgan-gp/train.py --data-path 2019/img-encode/tusgan_encode.npz --epochs 100 --batch-size 256
```

### 4. Visualizing Training Progress
Launch TensorBoard to view loss curves and synthetic diary heatmaps:
```bash
tensorboard --logdir runs/
```

### 5. Launch the Streamlit Dashboard
```bash
streamlit run dashboard.py
```
