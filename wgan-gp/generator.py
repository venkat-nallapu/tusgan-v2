"""
TUS-GAN — Generator (v2)
=========================
Conditional Generator for WGAN-GP that synthesises realistic
respondent diary sequences from the ITUS 2019 dataset.

v2 Updates:
  - 9-Channel output (Major Divisions 1-9).
  - Dual Learned Embeddings: District and State.
"""

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────
# Helper: Conditional Batch Normalisation
# ─────────────────────────────────────────────────────────────

class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features: int, cond_dim: int):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.affine = nn.Linear(cond_dim, 2 * num_features)
        nn.init.ones_(self.affine.weight[:num_features])
        nn.init.zeros_(self.affine.weight[num_features:])
        nn.init.zeros_(self.affine.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        params = self.affine(c)
        gamma, beta = params.chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta  = beta.unsqueeze(-1).unsqueeze(-1)
        return gamma * self.bn(x) + beta


# ─────────────────────────────────────────────────────────────
# Helper: Self-Attention for 2D/1D Feature Maps
# ─────────────────────────────────────────────────────────────

class SelfAttention2d(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key   = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.size()
        # Project and reshape to (B, H*W, C//8)
        q = self.query(x).view(B, -1, H * W).permute(0, 2, 1)
        k = self.key(x).view(B, -1, H * W)
        attn = torch.bmm(q, k)
        attn = torch.softmax(attn, dim=-1)

        v = self.value(x).view(B, -1, H * W)
        out = torch.bmm(v, attn.permute(0, 2, 1))
        out = out.view(B, C, H, W)

        return x + self.gamma * out


# ─────────────────────────────────────────────────────────────
# Helper: One Upsampling Block (Residual + CBN)
# ─────────────────────────────────────────────────────────────

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, cond_dim: int):
        super().__init__()
        self.conv_t = nn.ConvTranspose2d(
            in_channels, out_channels,
            kernel_size=(4, 1), stride=(2, 1), padding=(1, 0),
            bias=False,
        )
        self.cbn   = ConditionalBatchNorm2d(out_channels, cond_dim)
        self.act   = nn.LeakyReLU(0.2, inplace=True)
        
        # Shortcut link for residual learning
        self.shortcut = nn.Sequential(
            nn.Upsample(scale_factor=(2, 1), mode='nearest'),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            ConditionalBatchNorm2d(out_channels, cond_dim)
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # Residual path
        res = self.shortcut[0](x)
        res = self.shortcut[1](res)
        res = self.shortcut[2](res, c)
        
        # Main path
        out = self.conv_t(x)
        out = self.cbn(out, c)
        out = self.act(out)
        
        return out + res


# ─────────────────────────────────────────────────────────────
# Main Generator
# ─────────────────────────────────────────────────────────────

class Generator(nn.Module):
    def __init__(
        self,
        noise_dim: int          = 128,
        cond_dim: int           = 83,   # v2 OH dims
        num_districts: int      = 71,
        num_states: int         = 36,
        district_embed_dim: int = 16,
        state_embed_dim: int    = 8,
        base_channels: int      = 256,
    ):
        super().__init__()

        self.noise_dim          = noise_dim
        self.cond_dim           = cond_dim
        self.base_channels      = base_channels

        # Learned Embeddings
        self.district_embed = nn.Embedding(num_districts, district_embed_dim)
        self.state_embed    = nn.Embedding(num_states, state_embed_dim)

        # Full conditioning vector includes: OH vector + District + State
        full_cond_dim = cond_dim + district_embed_dim + state_embed_dim
        self.full_cond_dim = full_cond_dim

        # Backbone
        self.fc = nn.Sequential(
            nn.Linear(noise_dim + full_cond_dim, base_channels * 12 * 1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.start_time = 12
        self.start_ch   = base_channels

        self.up1 = UpsampleBlock(base_channels,       base_channels // 2, full_cond_dim)
        self.attn1 = SelfAttention2d(base_channels // 2)
        self.up2 = UpsampleBlock(base_channels // 2,  base_channels // 4, full_cond_dim)

        self.out_conv = nn.Sequential(
            nn.Conv2d(base_channels // 4, 9, kernel_size=(3, 1), stride=1, padding=(1, 0)),
            nn.Tanh(),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear, nn.Embedding, nn.BatchNorm2d)):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z, cond_vector, district_ids, state_ids):
        # Embeddings
        d_emb = self.district_embed(district_ids)
        s_emb = self.state_embed(state_ids)

        # Full condition
        c = torch.cat([cond_vector, d_emb, s_emb], dim=1)

        # Initial map
        h = self.fc(torch.cat([z, c], dim=1))
        h = h.view(-1, self.start_ch, self.start_time, 1)

        # Upsample
        h = self.up1(h, c)
        h = self.attn1(h)
        h = self.up2(h, c)

        # Output
        return self.out_conv(h)


if __name__ == "__main__":
    BATCH = 4
    G = Generator()
    z = torch.randn(BATCH, 128)
    cv = torch.zeros(BATCH, 83)
    di = torch.randint(0, 71, (BATCH,))
    si = torch.randint(0, 36, (BATCH,))
    fake = G(z, cv, di, si)
    print(f"Output shape: {fake.shape}")
    assert fake.shape == (BATCH, 9, 48, 1)
    print("Smoke test passed ✓")
