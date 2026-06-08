"""
TUS-GAN — Critic (with Gradient Penalty)
==========================================
Conditional Critic for WGAN-GP that scores diary sequences for
realism given the respondent's demographic conditioning vector.

Architecture Overview
---------------------
Input:
  • diary         : (B, 1, 48, 1)   real or generated diary in [-1, +1]
  • cond_vector   : (B, 49)          one-hot conditioning features
  • district_ids  : (B,)             integer district id (0 … NUM_DISTRICTS-1)

Output:
  • score         : (B,)            unbounded real scalar (no sigmoid!)
                    higher  → more "real"
                    lower   → more "fake"

Why no Sigmoid or BN in the Critic?
  WGAN-GP requires the Critic to be a K-Lipschitz function.
  BatchNorm creates correlations between samples inside a batch,
  which breaks the Gradient Penalty's per-sample gradient norm
  assumption.  We use Instance Normalisation (or no normalisation)
  instead, and never apply Sigmoid at the output.

Conditioning mechanism
  At each downsampling block we use Conditional Instance Normalisation
  (CIN), which is the WGAN-GP-safe analogue of CBN in the Generator.
  CIN predicts per-channel scale/shift from the condition vector,
  just like CBN but using InstanceNorm instead of BatchNorm as the
  base normaliser.

Gradient Penalty
  compute_gradient_penalty() implements the two-sided GP from
  "Improved Training of Wasserstein GANs" (Gulrajani et al., 2018):

      GP = E_x̂[ (‖∇_x̂ D(x̂)‖₂ − 1)² ]

  where x̂ = ε·x_real + (1−ε)·x_fake  (random interpolation)
  and ε ~ Uniform(0,1).

Pipeline inside the Critic
--------------------------
  Step 1  District ids → learnable embedding  (B, DISTRICT_EMBED_DIM)
  Step 2  Concatenate condition vector + district embedding → full_cond
  Step 3  Merge diary with condition via pixel-wise injection
          (broadcast cond to time dim, concat channel-wise)
  Step 4  Three downsampling Conv blocks: 48 → 24 → 12 → 6
  Step 5  Flatten → Linear → scalar score
"""

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────
# Helper: Conditional Instance Normalisation (CIN)
# ─────────────────────────────────────────────────────────────

class ConditionalInstanceNorm2d(nn.Module):
    """
    Like ConditionalBatchNorm2d in the Generator, but using InstanceNorm.

    InstanceNorm normalises each sample independently (no cross-sample
    statistics), which is required for the Gradient Penalty to be
    computed correctly in WGAN-GP.

    Parameters
    ----------
    num_features : number of channels C in the feature map
    cond_dim     : length of the conditioning vector
    """

    def __init__(self, num_features: int, cond_dim: int):
        super().__init__()

        # InstanceNorm without learnable affine (we supply gamma/beta ourselves)
        self.norm = nn.InstanceNorm2d(num_features, affine=False)

        # Linear layer: condition → (gamma, beta) concatenated
        self.affine = nn.Linear(cond_dim, 2 * num_features)

        nn.init.ones_(self.affine.weight[:num_features])
        nn.init.zeros_(self.affine.weight[num_features:])
        nn.init.zeros_(self.affine.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        x : (B, C, H, W)
        c : (B, cond_dim)
        """
        params = self.affine(c)
        gamma, beta = params.chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)   # (B, C, 1, 1)
        beta  = beta.unsqueeze(-1).unsqueeze(-1)    # (B, C, 1, 1)
        return gamma * self.norm(x) + beta


# ─────────────────────────────────────────────────────────────
# Helper: Downsampling Block
# ─────────────────────────────────────────────────────────────

class DownsampleBlock(nn.Module):
    """
    A single downsampling step in the Critic backbone.

    Sequence:
        Conv2d (stride 2 along time axis)
        ConditionalInstanceNorm2d
        LeakyReLU

    The convolution halves the time dimension:
        (B, in_ch, T, 1) → (B, out_ch, T//2, 1)

    We use LeakyReLU (slope=0.2) rather than ReLU because it avoids
    dead neurons and has been shown to work better for discriminators /
    critics in GAN literature.

    Parameters
    ----------
    in_channels  : input channel count
    out_channels : output channel count
    cond_dim     : length of conditioning vector (passed to CIN)
    """

    def __init__(self, in_channels: int, out_channels: int, cond_dim: int):
        super().__init__()

        # kernel=(4,1), stride=(2,1), padding=(1,0) halves the time dim.
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=(4, 1),
            stride=(2, 1),
            padding=(1, 0),
            bias=False,
        )
        self.cin = ConditionalInstanceNorm2d(out_channels, cond_dim)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.cin(x, c)
        x = self.act(x)
        return x


# ─────────────────────────────────────────────────────────────
# Main Critic
# ─────────────────────────────────────────────────────────────

class Critic(nn.Module):
    """
    Conditional Critic for TUS-GAN (WGAN-GP).

    Parameters
    ----------
    cond_dim           : length of the one-hot conditioning vector (49)
    num_districts      : number of unique district ids for the embedding
    district_embed_dim : embedding size for district ids (default 16)
    base_channels      : channel width at the widest (first conv) point
    """

    def __init__(
        self,
        cond_dim: int           = 49,
        num_districts: int      = 71,
        district_embed_dim: int = 16,
        base_channels: int      = 64,
    ):
        super().__init__()

        self.cond_dim           = cond_dim
        self.district_embed_dim = district_embed_dim

        full_cond_dim = cond_dim + district_embed_dim

        # ── Step 1: District Embedding ──────────────────────────────
        self.district_embed = nn.Embedding(num_districts, district_embed_dim)

        # ── Step 3: Condition injection ─────────────────────────────
        # We concatenate the condition (broadcast to time dimension) with
        # the diary channel, so the first Conv sees both diary activity
        # codes AND the respondent's demographics.
        # Input channels = diary channels (1) + full_cond_dim
        in_ch = 1 + full_cond_dim

        # ── Step 4: Downsampling backbone ───────────────────────────
        # 48 → 24 → 12 → 6 (three stride-2 convolutions)
        self.down1 = DownsampleBlock(in_ch,                base_channels,     full_cond_dim)  # (B, 64, 24, 1)
        self.down2 = DownsampleBlock(base_channels,        base_channels * 2, full_cond_dim)  # (B,128, 12, 1)
        self.down3 = DownsampleBlock(base_channels * 2,    base_channels * 4, full_cond_dim)  # (B,256,  6, 1)

        # ── Step 5: Output head ──────────────────────────────────────
        # Flatten: 256 channels × 6 time slots × 1 spatial = 1536 features
        flat_dim = base_channels * 4 * 6 * 1
        self.output = nn.Linear(flat_dim, 1)   # single unbounded score

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(
        self,
        diary: torch.Tensor,
        cond_vector: torch.Tensor,
        district_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        diary        : (B, 1, 48, 1)  real or fake diary in [-1, +1]
        cond_vector  : (B, 49)        one-hot demographic features
        district_ids : (B,)           integer district ids

        Returns
        -------
        score : (B, 1)  unbounded scalar — no activation applied
        """
        # ── District embedding ──────────────────────────────────────
        d_emb = self.district_embed(district_ids)              # (B, district_embed_dim)

        # ── Full conditioning vector ────────────────────────────────
        c = torch.cat([cond_vector, d_emb], dim=1)            # (B, full_cond_dim)

        # ── Inject condition into the spatial domain ─────────────────
        # We expand the 1-D condition to match the diary's time and
        # spatial dimensions, then concatenate along the channel axis.
        # Shape: (B, full_cond_dim) → (B, full_cond_dim, 1, 1)
        #                           → (B, full_cond_dim, 48, 1)
        c_spatial = c.unsqueeze(-1).unsqueeze(-1)              # (B, C, 1, 1)
        c_spatial = c_spatial.expand(-1, -1, diary.size(2), diary.size(3))
        # Concatenate diary + condition channels
        x = torch.cat([diary, c_spatial], dim=1)              # (B, 1+C, 48, 1)

        # ── Downsample ──────────────────────────────────────────────
        x = self.down1(x, c)    # (B,  64, 24, 1)
        x = self.down2(x, c)    # (B, 128, 12, 1)
        x = self.down3(x, c)    # (B, 256,  6, 1)

        # ── Score ───────────────────────────────────────────────────
        x = x.view(x.size(0), -1)    # (B, 256*6*1)
        score = self.output(x)       # (B, 1)

        return score


# ─────────────────────────────────────────────────────────────
# Gradient Penalty (WGAN-GP)
# ─────────────────────────────────────────────────────────────

def compute_gradient_penalty(
    critic: Critic,
    real_diaries: torch.Tensor,
    fake_diaries: torch.Tensor,
    cond_vector: torch.Tensor,
    district_ids: torch.Tensor,
    device: torch.device,
    lambda_gp: float = 10.0,
) -> torch.Tensor:
    """
    Compute the WGAN-GP gradient penalty.

    The penalty enforces the 1-Lipschitz constraint on the Critic by
    penalising gradients that deviate from unit norm.  It is evaluated
    at random interpolations between real and fake samples:

        x̂ = ε · x_real + (1 − ε) · x_fake,   ε ~ Uniform(0,1)

    Then:
        GP = λ · E_x̂[ (‖∇_x̂ D(x̂)‖₂ − 1)² ]

    Parameters
    ----------
    critic        : the Critic model
    real_diaries  : (B, 1, 48, 1)  batch of real diary tensors
    fake_diaries  : (B, 1, 48, 1)  batch of generated diary tensors
    cond_vector   : (B, 49)        conditioning features for this batch
    district_ids  : (B,)           district ids for this batch
    device        : torch.device to create epsilon on
    lambda_gp     : GP weight (default 10, from the original paper)

    Returns
    -------
    gp : scalar tensor  (gradient penalty loss term)
    """
    B = real_diaries.size(0)

    # ── Sample interpolation coefficient ε ─────────────────────────
    # Shape (B, 1, 1, 1) broadcasts over (B, 1, 48, 1)
    eps = torch.rand(B, 1, 1, 1, device=device)

    # ── Interpolated samples ────────────────────────────────────────
    x_hat = eps * real_diaries.detach() + (1 - eps) * fake_diaries.detach()

    # We need gradients w.r.t. x_hat, so enable grad tracking
    x_hat.requires_grad_(True)

    # ── Critic score at interpolated points ─────────────────────────
    score_hat = critic(x_hat, cond_vector, district_ids)    # (B, 1)

    # ── Compute ∂score_hat / ∂x_hat ─────────────────────────────────
    # torch.autograd.grad returns a tuple; we take the first element.
    # create_graph=True is required so that the gradient itself can
    # be differentiated (needed to back-prop through the GP loss).
    gradients = torch.autograd.grad(
        outputs=score_hat,
        inputs=x_hat,
        grad_outputs=torch.ones_like(score_hat),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]                                                     # (B, 1, 48, 1)

    # ── Compute gradient norm ────────────────────────────────────────
    # Flatten to (B, 48) then take L2 norm along the feature dimension.
    grad_norm = gradients.view(B, -1).norm(2, dim=1)        # (B,)

    # ── Two-sided penalty: (‖∇‖₂ − 1)² ─────────────────────────────
    gp = lambda_gp * ((grad_norm - 1.0) ** 2).mean()

    return gp


# ─────────────────────────────────────────────────────────────
# Critic loss helpers (call these inside your training loop)
# ─────────────────────────────────────────────────────────────

def critic_loss(
    real_scores: torch.Tensor,
    fake_scores: torch.Tensor,
    gradient_penalty: torch.Tensor,
) -> torch.Tensor:
    """
    Full Critic (Discriminator) loss for WGAN-GP.

    L_D = E[D(fake)] − E[D(real)] + GP

    We maximise E[D(real)] − E[D(fake)], which is equivalent to
    minimising E[D(fake)] − E[D(real)].  The GP term is added to
    regularise the Lipschitz constraint.

    Parameters
    ----------
    real_scores       : (B, 1)  Critic scores for real diaries
    fake_scores       : (B, 1)  Critic scores for fake diaries
    gradient_penalty  : scalar  GP loss from compute_gradient_penalty()

    Returns
    -------
    loss : scalar tensor
    """
    return fake_scores.mean() - real_scores.mean() + gradient_penalty


def generator_loss(fake_scores: torch.Tensor) -> torch.Tensor:
    """
    Generator loss for WGAN-GP.

    L_G = −E[D(fake)]

    The Generator wants the Critic to score its outputs highly, so we
    minimise the negated mean score on fake samples.

    Parameters
    ----------
    fake_scores : (B, 1)  Critic scores for generated diaries

    Returns
    -------
    loss : scalar tensor
    """
    return -fake_scores.mean()


# ─────────────────────────────────────────────────────────────
# Quick smoke-test (run: python critic.py)
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import torch

    BATCH      = 4
    COND_DIM   = 49
    NUM_DIST   = 71
    DIST_EMBED = 16
    LAMBDA_GP  = 10.0
    device     = torch.device("cpu")

    D = Critic(
        cond_dim           = COND_DIM,
        num_districts      = NUM_DIST,
        district_embed_dim = DIST_EMBED,
    ).to(device)

    total_params = sum(p.numel() for p in D.parameters() if p.requires_grad)
    print(f"Critic parameters: {total_params:,}")

    # Dummy tensors
    real_d  = torch.rand(BATCH, 1, 48, 1) * 2 - 1   # uniform in [-1, +1]
    fake_d  = torch.rand(BATCH, 1, 48, 1) * 2 - 1
    cond    = torch.zeros(BATCH, COND_DIM)
    dist_id = torch.randint(0, NUM_DIST, (BATCH,))

    real_scores = D(real_d, cond, dist_id)
    fake_scores = D(fake_d, cond, dist_id)

    print(f"Real scores shape : {real_scores.shape}")   # (4, 1)
    print(f"Fake scores shape : {fake_scores.shape}")   # (4, 1)

    # Gradient penalty
    gp = compute_gradient_penalty(D, real_d, fake_d, cond, dist_id, device, LAMBDA_GP)
    print(f"Gradient penalty  : {gp.item():.6f}")

    # Losses
    c_loss = critic_loss(real_scores, fake_scores, gp)
    g_loss = generator_loss(fake_scores)
    print(f"Critic loss       : {c_loss.item():.6f}")
    print(f"Generator loss    : {g_loss.item():.6f}")

    print("Smoke test passed ✓")
