"""
TUS-GAN — Generator
====================
Conditional Generator for WGAN-GP that synthesises realistic
respondent diary sequences from the ITUS 2019 dataset.

Architecture Overview
---------------------
Input:
  • z              : (B, NOISE_DIM)          latent noise vector (N(0,1))
  • cond_vector    : (B, 49)                 one-hot conditioning features
  • district_ids   : (B,)                    integer district id (0 … NUM_DISTRICTS-1)

Output:
  • diary_fake     : (B, 1, 48, 1)           generated diary in [-1, +1]
                     matching the encoded tensor shape from encode.ipynb

Pipeline inside the Generator
------------------------------
  Step 1  District ids → learnable embedding  (B, DISTRICT_EMBED_DIM)
  Step 2  Concatenate [z | cond_vector | district_embedding] → input vector
  Step 3  Fully-connected projection → (B, 128 * 12 * 1) and reshape
  Step 4  Three transposed-convolution blocks upsample along the time axis
          12 → 24 → 48   (stride-2 ConvTranspose in the time dimension)
  Step 5  Final Conv2d + Tanh squashes output to [-1, +1]

Why Conv2d / ConvTranspose2d?
  The diary tensor is (B, 1, 48, 1) — a 2-D "image" that is 48 wide
  and 1 tall (singleton spatial dimension).  Using standard 2-D ops keeps
  the shape convention consistent with the Critic and lets you later
  extend the spatial dimension if needed (e.g. weekly diaries).

Conditioning mechanism
  Instead of concatenating the condition only at the input we also inject
  it at each upsampling block via Conditional Batch Normalisation (CBN).
  CBN learns per-sample affine scale and bias from the condition vector,
  which gives the network fine-grained control over activity patterns
  for each demographic group.
"""

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────
# Helper: Conditional Batch Normalisation
# ─────────────────────────────────────────────────────────────

class ConditionalBatchNorm2d(nn.Module):
    """
    Replaces the learned affine parameters of BatchNorm with values
    that are *predicted* from the conditioning vector.

    For each sample in the batch:
        y = gamma(c) * BN(x) + beta(c)
    where gamma and beta are produced by small linear projections of c.

    Parameters
    ----------
    num_features   : number of channels (C) in the feature map
    cond_dim       : length of the conditioning vector fed in
    """

    def __init__(self, num_features: int, cond_dim: int):
        super().__init__()

        # Standard BN without its own affine parameters
        # (we will supply gamma and beta ourselves)
        self.bn = nn.BatchNorm2d(num_features, affine=False)

        # Linear layers that predict per-channel gamma and beta from c
        # Output size 2 * num_features: first half = gamma, second = beta
        self.affine = nn.Linear(cond_dim, 2 * num_features)

        # Initialise close to identity so training starts stable
        nn.init.ones_(self.affine.weight[:num_features])   # gamma → 1
        nn.init.zeros_(self.affine.weight[num_features:])  # beta  → 0
        nn.init.zeros_(self.affine.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        x : (B, C, H, W)   feature map
        c : (B, cond_dim)  conditioning vector
        """
        # Predict gamma and beta from condition
        params = self.affine(c)                              # (B, 2C)
        gamma, beta = params.chunk(2, dim=1)                 # each (B, C)

        # Reshape to broadcast over H and W
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)            # (B, C, 1, 1)
        beta  = beta.unsqueeze(-1).unsqueeze(-1)             # (B, C, 1, 1)

        # Apply standard BN (normalise across the batch) then rescale
        return gamma * self.bn(x) + beta


# ─────────────────────────────────────────────────────────────
# Helper: One Upsampling Block
# ─────────────────────────────────────────────────────────────

class UpsampleBlock(nn.Module):
    """
    A single upsampling step used in the Generator backbone.

    Sequence:
        ConvTranspose2d  (stride 2 along time axis only)
        ConditionalBatchNorm2d
        ReLU

    The transposed convolution doubles the time dimension:
        (B, in_ch, T, 1) → (B, out_ch, 2T, 1)

    Parameters
    ----------
    in_channels  : input channel count
    out_channels : output channel count
    cond_dim     : length of conditioning vector (passed to CBN)
    """

    def __init__(self, in_channels: int, out_channels: int, cond_dim: int):
        super().__init__()

        # kernel=(4,1), stride=(2,1), padding=(1,0) is the standard recipe
        # for doubling the time dimension while keeping the spatial dim = 1.
        self.conv_t = nn.ConvTranspose2d(
            in_channels, out_channels,
            kernel_size=(4, 1),
            stride=(2, 1),
            padding=(1, 0),
            bias=False,                      # CBN provides its own bias
        )
        self.cbn   = ConditionalBatchNorm2d(out_channels, cond_dim)
        self.act   = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        x = self.conv_t(x)
        x = self.cbn(x, c)
        x = self.act(x)
        return x


# ─────────────────────────────────────────────────────────────
# Main Generator
# ─────────────────────────────────────────────────────────────

class Generator(nn.Module):
    """
    Conditional Generator for TUS-GAN.

    Parameters
    ----------
    noise_dim          : dimension of the input noise vector z (default 128)
    cond_dim           : dimension of the one-hot conditioning vector (49)
    num_districts      : number of unique district ids for the embedding
    district_embed_dim : embedding size for district ids (default 16)
    base_channels      : channel width at the narrowest (bottleneck) point
    """

    def __init__(
        self,
        noise_dim: int          = 128,
        cond_dim: int           = 49,
        num_districts: int      = 71,
        district_embed_dim: int = 16,
        base_channels: int      = 256,
    ):
        super().__init__()

        self.noise_dim          = noise_dim
        self.cond_dim           = cond_dim
        self.district_embed_dim = district_embed_dim
        self.base_channels      = base_channels

        # The full conditioning dimension seen by CBN layers and the
        # initial projection includes the district embedding.
        full_cond_dim = cond_dim + district_embed_dim

        # ── Step 1: District Embedding ──────────────────────────────
        # Maps integer district id → dense vector of size district_embed_dim.
        # This is more parameter-efficient than a 71-class one-hot (71 dims
        # vs 16 dims) and lets the model learn similarity between districts.
        self.district_embed = nn.Embedding(num_districts, district_embed_dim)

        # ── Step 2 + 3: Linear projection from noise + condition ────
        # The generator begins with a fully-connected layer that projects
        # the concatenated [z | cond | district_emb] into a small feature
        # map of shape (B, base_channels, 12, 1).
        #
        # 12 time slots × 4 upsample steps would give 12→24→48.
        # We do 2 upsample steps: 12 → 24 → 48.
        self.fc = nn.Sequential(
            nn.Linear(noise_dim + full_cond_dim, base_channels * 12 * 1),
            nn.ReLU(inplace=True),
        )

        # After fc we reshape to (B, base_channels, 12, 1).
        self.start_time = 12
        self.start_ch   = base_channels

        # ── Step 4: Upsampling blocks ────────────────────────────────
        # Each block doubles the time dimension.
        # 12 → 24 → 48.  Two blocks, halving channels at each step.
        self.up1 = UpsampleBlock(base_channels,       base_channels // 2, full_cond_dim)  # → (B, 128, 24, 1)
        self.up2 = UpsampleBlock(base_channels // 2,  base_channels // 4, full_cond_dim)  # → (B,  64, 48, 1)

        # ── Step 5: Output head ──────────────────────────────────────
        # A final convolution refines channel 64 → 1 and Tanh maps to [-1, +1].
        self.out_conv = nn.Sequential(
            nn.Conv2d(
                base_channels // 4, 1,
                kernel_size=(3, 1),
                stride=1,
                padding=(1, 0),
            ),
            nn.Tanh(),   # output lives in [-1, +1], same range as encoded data
        )

        # Weight initialisation: normal distribution with small std
        self._init_weights()

    def _init_weights(self):
        """Apply sensible default initialisation to Conv and Linear layers."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
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
        z: torch.Tensor,
        cond_vector: torch.Tensor,
        district_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        z            : (B, noise_dim)   sampled from N(0, I)
        cond_vector  : (B, 49)          one-hot demographic features
        district_ids : (B,)             integer district ids

        Returns
        -------
        diary_fake : (B, 1, 48, 1)  values in [-1, +1]
        """
        # ── District embedding ──────────────────────────────────────
        d_emb = self.district_embed(district_ids)              # (B, district_embed_dim)

        # ── Build full conditioning vector ──────────────────────────
        # This vector will be passed to CBN at every block so the network
        # can modulate its internal representations per demographic group.
        c = torch.cat([cond_vector, d_emb], dim=1)            # (B, cond_dim + embed_dim)

        # ── Project noise + condition to initial feature map ────────
        h = self.fc(torch.cat([z, c], dim=1))                  # (B, base_ch * 12)
        h = h.view(-1, self.start_ch, self.start_time, 1)      # (B, 256, 12, 1)

        # ── Upsample with conditional batch norm ────────────────────
        h = self.up1(h, c)    # (B, 128, 24, 1)
        h = self.up2(h, c)    # (B,  64, 48, 1)

        # ── Output head ─────────────────────────────────────────────
        diary_fake = self.out_conv(h)                          # (B, 1, 48, 1)

        return diary_fake


# ─────────────────────────────────────────────────────────────
# Quick smoke-test (run this file directly: python generator.py)
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import torch

    BATCH        = 4
    NOISE_DIM    = 128
    COND_DIM     = 49       # from encode.ipynb Step 7
    NUM_DIST     = 71       # number of unique districts in ITUS 2019
    DIST_EMBED   = 16

    G = Generator(
        noise_dim          = NOISE_DIM,
        cond_dim           = COND_DIM,
        num_districts      = NUM_DIST,
        district_embed_dim = DIST_EMBED,
    )

    # Count trainable parameters
    total_params = sum(p.numel() for p in G.parameters() if p.requires_grad)
    print(f"Generator parameters: {total_params:,}")

    # Dummy inputs
    z            = torch.randn(BATCH, NOISE_DIM)
    cond_vector  = torch.zeros(BATCH, COND_DIM)   # replace with real one-hots
    district_ids = torch.randint(0, NUM_DIST, (BATCH,))

    fake_diary = G(z, cond_vector, district_ids)

    print(f"Output shape : {fake_diary.shape}")    # expected (4, 1, 48, 1)
    print(f"Output range : [{fake_diary.min():.4f}, {fake_diary.max():.4f}]")
    assert fake_diary.shape == (BATCH, 1, 48, 1), "Shape mismatch!"
    assert fake_diary.min() >= -1.0 and fake_diary.max() <= 1.0, "Range error!"
    print("Smoke test passed ✓")
