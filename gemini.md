# TUS-GAN (v2) Technical Reference & Strategy Guide

This document provides in-depth technical specifications, architectural decisions, and training strategies for the TUS-GAN project.

## 1. Model Architecture

### Wasserstein GAN with Gradient Penalty (WGAN-GP)
The core backbone is WGAN-GP, which replaces the standard JS-divergence with Earth Mover's Distance. This provides stable gradients even when the Generator and Critic are not perfectly balanced.

#### Generator (Conditional)
- **Latent Space**: 128-dimensional noise vector $z \sim \mathcal{N}(0, 1)$.
- **Conditioning Injection**: 
  - **Early Concatenation**: $[z; c]$ is fed into the first linear projection.
  - **Conditional Batch Normalization (CBN)**: The conditioning vector $c$ modulates the feature maps at every upsampling stage.
- **Layers**: 
  - `Linear` $\rightarrow$ `Reshape` $\rightarrow$ `ConvTranspose2d` (upsampling $12 \rightarrow 24 \rightarrow 48$).
  - **Activation**: `ReLU` for hidden layers, `Tanh` for the final output to bound values within $[-1, 1]$.
- **Shape**: Input $(B, 128)$; Output $(B, 9, 48, 1)$.

#### Critic (Conditional)
- ** Lipschitz Constraint**: Enforced via Gradient Penalty (GP) with weight $\lambda=10$.
- **Conditioning Injection**:
  - **Pixel-wise Concatenation**: Demographic vector is broadcasted and concatenated as extra channels to the input diary.
  - **Conditional Instance Normalization (CIN)**: Ensures per-sample normalization, which is compatible with the per-sample gradient penalty requirement.
- **Layers**: 
  - `Conv2d` (downsampling $48 \rightarrow 24 \rightarrow 12 \rightarrow 6$).
  - **Activation**: `LeakyReLU(0.2)` throughout. No activation on the final scalar output (unbounded score).
- **Shape**: Input $(B, 9, 48, 1)$; Output $(B, 1)$.

## 2. Image Encoding (Diary Tensors)

Each respondent's day is represented as a $(9, 48, 1)$ tensor:
- **9 Channels**: One-hot representation of the 9 Major Divisions of activity.
- **48 Slots**: 30-minute intervals covering 24 hours (starting from 04:00 AM).
- **1 Dimension**: Standard spatial width to allow 2D convolution operations.

**Activity Mapping Logic**:
The raw 3-digit ITUS codes are mapped using the first digit:
`Division = floor(ActivityCode / 100)`

| Div | Major Division | 3-Digit Code Range | Mapping Detail |
|:---:|---|:---:|---|
| **1** | Employment | 111 - 180 | Includes all paid work and related search/travel. |
| **2** | Production | 211 - 250 | Agriculture, hunting, and construction for own use. |
| **3** | Unpaid Domestic | 311 - 390 | Household management, cleaning, and meal prep. |
| **4** | Unpaid Caregiving | 411 - 490 | Childcare and adult care for household members. |
| **5** | Unpaid Volunteer | 511 - 590 | Service to other households and the community. |
| **6** | Learning | 611 - 691 | Education, training, and self-study. |
| **7** | Socializing & Religious | 711 - 790 | Socializing, community and religious activities. |
| **8** | Leisure & Sports | 811 - 890 | Media, hobbies, games, and sports. |
| **9** | Self-care | 911 - 990 | Sleep, eating, and personal hygiene. |

## 3. Conditioning Vector Code Mappings (v2)

The conditioning vector is a concatenation of multiple one-hot encoded features. Below are the specific mappings used during encoding.

### Demographic Features

| Feature | Code | Representation |
|:---:|:---:|---|
| **Gender** | 1 | Male |
| | 2 | Female |
| | 3 | Transgender |
| **Marital Status** | 1 | Married |
| | 2 | Widow / Widower |
| | 3 | Divorced / Separated |
| | 4 | Never Married |
| **Sector** | 1 | Rural |
| | 2 | Urban |
| **Caregiving Dummy** | 0 | No special care needed in household |
| | 1 | Household has member(s) needing special care |

### Education Level (ITUS Codes)

| Code | Label |
|:---:|---|
| 01 | Not literate |
| 02 | Literate (without formal schooling) |
| 03 | Literate (through NFEC) |
| 04 | Literate (through TLC/AEC) |
| 05 | Literate (Others) |
| 06 | Below Primary |
| 07 | Primary |
| 08 | Middle |
| 10 | Secondary |
| 11 | Higher Secondary |
| 12 | Diploma / Graduate & Above |

### Principal Activity Status

| Code | Label |
|:---:|---|
| 11 | Self-Employed (Own Account Worker) |
| 12 | Self-Employed (Employer) |
| 21 | Helper in HH Enterprise (unpaid) |
| 31 | Regular Salaried / Wage Employee |
| 41 | Casual Labour (Public Works) |
| 51 | Casual Labour (Other than Public Works) |
| 81 | Unemployed (Seeking / Available for work) |
| 91 | Student |
| 92 | Domestic Duties Only |
| 93 | Domestic Duties & Free Collection of Goods |
| 94 | Rentier, Pensioner, etc. |
| 95 | Disabled / Unable to work |
| 97 | Others (Infants, etc.) |

### Temporal & Continuous Features

- **Day of Week**: 1 (Monday) to 7 (Sunday).
- **Age Groups**: Binned into 7 categories: `<15, 15-17, 18-24, 25-34, 35-44, 45-59, 60+`.
- **Household Size**: Discrete one-hot from 1 to 23.
- **Expenditure Bins**: 10 bins representing deciles of the log-normalized monthly consumer expenditure.

## 4. Training Strategy & Best Practices

To achieve high-quality synthetic diaries, follow these approaches:

- **Critic Superiority**: Always update the Critic multiple times (`n_critic=5`) for every Generator update. This ensures the Wasserstein distance estimate is accurate.
- **Optimizer Config**: Use `Adam` with `beta1=0.0` and `beta2=0.9`. Using the default `beta1=0.9` leads to momentum-induced oscillations that break the Lipschitz constraint.
- **Batch Size**: Prefer larger batches (128-256) to ensure the Gradient Penalty is estimated over a diverse set of interpolations.
- **Learning Rate**: Keep LR low ($10^{-4}$) to prevent sudden divergence.
- **Evaluation**: Use the `Wasserstein Distance` reported in the logs as a primary metric for convergence. A decreasing and stabilizing W-distance indicates the model is learning the distribution.

## 5. Frameworks & Tools

- **Core**: PyTorch (Deep Learning), NumPy (Numerical processing).
- **Data**: Pandas (CSV handling), Scikit-learn (One-hot encoding).
- **Visualization**: Matplotlib (Stat plots), TensorBoard (Training curves).
- **Deployment**: Streamlit (Interactive dashboard).

## 6. Dashboard Integration

The `dashboard.py` utility allows for:
- **Interactive Generation**: Select demographics (Age, Gender, Education, Sector) and see the generated routine in real-time.
- **Comparison View**: Toggle between real distributions and synthetic results to verify model fidelity.
- **HuggingFace Hub Support**: Automatically downloads the latest checkpoints and encoded data for seamless deployment.

## 7. Future Roadmap (v2+)
- **Refined Encoding**: Moving from 9 divisions to 20+ sub-divisions for more granular activity synthesis.
- **Sequential Context**: Incorporating RNN/Transformer layers to better capture long-range temporal dependencies within the 24-hour cycle.
- **Location Update**: Enhanced district and state-level conditioning (v2 focus).
