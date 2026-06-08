# TUS-GAN: Synthetic Time-Use Diary Generation

TUS-GAN is a deep learning project that utilizes a **Conditional Wasserstein GAN with Gradient Penalty (WGAN-GP)** to synthesize realistic Time-Use Survey (TUS) diaries. By treating 24-hour activity sequences as "images," the model learns to generate daily routines that are demographically and geographically consistent with the India Time Use Survey (ITUS) 2019.

## 🚀 Key Features

*   **WGAN-GP Architecture:** Employs Wasserstein loss and Gradient Penalty for superior training stability and convergence compared to standard GANs.
*   **Conditional Generation:** Synthesizes diaries conditioned on 49 demographic dimensions (Age, Gender, Marital Status, Education, etc.) and learned District embeddings (71 districts).
*   **Diary-as-Image Encoding:** Encodes 48 half-hour time slots into a structured tensor format `(N, 1, 48, 1)` suitable for convolutional architectures.
*   **Advanced Modality:** Uses **Conditional Batch Normalization (CBN)** in the Generator to modulate internal features based on the respondent's profile.

## 📂 Project Structure

```text
├── 2019/               # Raw and cleaned ITUS 2019 CSV data
├── wgan-gp/            # Core GAN implementation
│   ├── encode.ipynb    # Data preprocessing & tensor encoding pipeline
│   ├── train.py        # Main training script (WGAN-GP)
│   ├── generator.py    # Generator architecture (ConvTranspose2d + CBN)
│   ├── critic.py       # Critic architecture (Conv2d)
│   ├── generate.py     # Inference script for synthetic data production
│   └── evaluate.py     # Evaluation script for distribution comparison
├── checkpoints/        # Saved model snapshots (.pt)
├── samples/            # Visual samples generated during training (.npy)
└── evaluation_results/ # Plots comparing real vs. synthetic distributions
```

## 🛠️ Pipeline

### 1. Data Preparation
Run the `wgan-gp/encode.ipynb` notebook. It cleans the raw ITUS data, handles missing values, and produces `tusgan_encoded.npz`.

### 2. Training
Execute the training script. It logs to TensorBoard and saves checkpoints every 10 epochs.
```bash
python wgan-gp/train.py --epochs 200 --batch 128
```

### 3. Generation
Generate a synthetic CSV dataset from the final trained model:
```bash
python wgan-gp/generate.py
```

### 4. Evaluation
Compare the synthetic data against the real data to validate statistical fidelity:
```bash
python wgan-gp/evaluate.py
```
This produces histograms and time-use comparison plots in the `evaluation_results/` directory.

## 📊 Concepts: Diary-as-Image
The core innovation is mapping a 24-hour day into 48 discrete slots. Each slot is encoded as a normalized activity code in the range `[-1, +1]`. This allows the GAN to use 2D Convolutions to learn the temporal "shapes" of a day—for example, the characteristic "block" of sleep at night or work during the day.

## 📝 Requirements
*   Python 3.12+
*   PyTorch
*   Pandas / Numpy
*   Matplotlib (for evaluation)
*   TensorBoard (for logging)
