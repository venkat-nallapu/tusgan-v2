"""
TUS-GAN — Critic (v2)
======================
Conditional Critic for WGAN-GP that scores diary sequences for
realism given the respondent's demographic conditioning vector.

v2 Updates:
  - 9-Channel input (Major Divisions 1-9).
  - Dual Learned Embeddings: District and State.
"""

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────
# Helper: Conditional Instance Normalisation (CIN)
# ─────────────────────────────────────────────────────────────

class ConditionalInstanceNorm2d(nn.Module):
    def __init__(self, num_features: int, cond_dim: int):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.affine = nn.Linear(cond_dim, 2 * num_features)
        nn.init.ones_(self.affine.weight[:num_features])
        nn.init.zeros_(self.affine.weight[num_features:])
        nn.init.zeros_(self.affine.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        params = self.affine(c)
        gamma, beta = params.chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta  = beta.unsqueeze(-1).unsqueeze(-1)
        return gamma * self.norm(x) + beta


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
        q = self.query(x).view(B, -1, H * W).permute(0, 2, 1)
        k = self.key(x).view(B, -1, H * W)
        attn = torch.bmm(q, k)
        attn = torch.softmax(attn, dim=-1)

        v = self.value(x).view(B, -1, H * W)
        out = torch.bmm(v, attn.permute(0, 2, 1))
        out = out.view(B, C, H, W)

        return x + self.gamma * out


# ─────────────────────────────────────────────────────────────
# Helper: Downsampling Block (Residual + CIN + Spectral Norm)
# ─────────────────────────────────────────────────────────────

class DownsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, cond_dim: int):
        super().__init__()
        self.conv = nn.utils.spectral_norm(nn.Conv2d(
            in_channels, out_channels,
            kernel_size=(4, 1), stride=(2, 1), padding=(1, 0),
            bias=False,
        ))
        self.cin = ConditionalInstanceNorm2d(out_channels, cond_dim)
        self.act = nn.LeakyReLU(0.2, inplace=True)

        # Shortcut path for downsampling
        self.shortcut = nn.Sequential(
            nn.AvgPool2d(kernel_size=(2, 1), stride=(2, 1), padding=(0, 0)),
            nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)),
            ConditionalInstanceNorm2d(out_channels, cond_dim)
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # Residual path
        res = self.shortcut[0](x)
        res = self.shortcut[1](res)
        res = self.shortcut[2](res, c)

        # Main path
        out = self.conv(x)
        out = self.cin(out, c)
        out = self.act(out)

        return out + res


# ─────────────────────────────────────────────────────────────
# Main Critic
# ─────────────────────────────────────────────────────────────

class Critic(nn.Module):
    def __init__(
        self,
        cond_dim: int           = 83,
        num_districts: int      = 71,
        num_states: int         = 36,
        district_embed_dim: int = 16,
        state_embed_dim: int    = 8,
        base_channels: int      = 64,
    ):
        super().__init__()

        self.district_embed = nn.Embedding(num_districts, district_embed_dim)
        self.state_embed    = nn.Embedding(num_states, state_embed_dim)

        full_cond_dim = cond_dim + district_embed_dim + state_embed_dim
        in_ch = 9 + full_cond_dim

        self.down1 = DownsampleBlock(in_ch,                base_channels,     full_cond_dim)
        self.down2 = DownsampleBlock(base_channels,        base_channels * 2, full_cond_dim)
        self.attn1 = SelfAttention2d(base_channels * 2)
        self.down3 = DownsampleBlock(base_channels * 2,    base_channels * 4, full_cond_dim)

        flat_dim = base_channels * 4 * 6 * 1
        self.output = nn.utils.spectral_norm(nn.Linear(flat_dim, 1))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear, nn.Embedding)):
                if hasattr(m, 'weight') and m.weight is not None:
                    # Let spectral norm handle Conv2d & Linear weight init wrappers if applicable
                    # but standard init is still fine.
                    try:
                        nn.init.normal_(m.weight, mean=0.0, std=0.02)
                    except Exception:
                        pass
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, diary, cond_vector, district_ids, state_ids):
        d_emb = self.district_embed(district_ids)
        s_emb = self.state_embed(state_ids)
        c = torch.cat([cond_vector, d_emb, s_emb], dim=1)

        c_spatial = c.unsqueeze(-1).unsqueeze(-1)
        c_spatial = c_spatial.expand(-1, -1, diary.size(2), diary.size(3))
        x = torch.cat([diary, c_spatial], dim=1)

        x = self.down1(x, c)
        x = self.down2(x, c)
        x = self.attn1(x)
        x = self.down3(x, c)

        x = x.view(x.size(0), -1)
        return self.output(x)


def compute_gradient_penalty(critic, real, fake, cond, dist, state, device, lambda_gp=10.0):
    B = real.size(0)
    eps = torch.rand(B, 1, 1, 1, device=device)
    x_hat = eps * real.detach() + (1 - eps) * fake.detach()
    x_hat.requires_grad_(True)

    score_hat = critic(x_hat, cond, dist, state)

    gradients = torch.autograd.grad(
        outputs=score_hat,
        inputs=x_hat,
        grad_outputs=torch.ones_like(score_hat),
        create_graph=True, retain_graph=True, only_inputs=True,
    )[0]

    grad_norm = gradients.view(B, -1).norm(2, dim=1)
    return lambda_gp * ((grad_norm - 1.0) ** 2).mean()


def critic_loss(real_scores, fake_scores, gp):
    return fake_scores.mean() - real_scores.mean() + gp


def generator_loss(fake_scores):
    return -fake_scores.mean()


if __name__ == "__main__":
    BATCH = 4
    D = Critic()
    real = torch.randn(BATCH, 9, 48, 1)
    fake = torch.randn(BATCH, 9, 48, 1)
    cv = torch.zeros(BATCH, 83)
    di = torch.randint(0, 71, (BATCH,))
    si = torch.randint(0, 36, (BATCH,))
    scores = D(real, cv, di, si)
    print(f"Scores shape: {scores.shape}")
    gp = compute_gradient_penalty(D, real, fake, cv, di, si, torch.device("cpu"))
    print(f"GP: {gp.item():.4f}")
    print("Smoke test passed ✓")

