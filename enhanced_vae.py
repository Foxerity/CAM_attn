import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


# ------------------------------
# Flow modules implementation
# ------------------------------
# ------------------ Utility: ActNorm（保持不变） ------------------ #
class ActNorm(nn.Module):
    def __init__(self, num_features, eps=1e-6):
        super().__init__()
        self.initialized = False
        self.eps = eps
        self.bias      = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.log_scale = nn.Parameter(torch.zeros(1, num_features, 1, 1))

    @torch.no_grad()
    def _data_init(self, x):
        mean = x.mean([0,2,3], keepdim=True)
        std  = x.std ([0,2,3], keepdim=True) + self.eps
        self.bias.data      = -mean
        self.log_scale.data = torch.log(1.0 / std)
        self.initialized = True

    def forward(self, x, reverse=False):
        if not self.initialized: self._data_init(x)
        if reverse:
            x = (x - self.bias) * torch.exp(-self.log_scale)
            sign = -1.
        else:
            x = x * torch.exp(self.log_scale) + self.bias
            sign = +1.
        B, _, H, W = x.shape
        logdet = sign * self.log_scale.view(-1).sum()
        return x, logdet.expand(B)            # (B,)



# ------------------ Slim Affine Coupling ------------------------- #
class AffineCouplingLite(nn.Module):
    """
    Depthwise-Separable + 1×1 轻量耦合
    """
    def __init__(self, in_ch, cond_ch, hidden=None, clamp=2.):
        super().__init__()
        self.clamp = clamp
        hidden = hidden or max(64, in_ch // 4)

        def dw_pw(cin, cout, k=3, g=1):
            return nn.Sequential(
                nn.Conv2d(cin, cin, k, padding=k//2, groups=cin),  # depth-wise
                nn.Conv2d(cin, cout, 1, groups=g))                 # point-wise

        self.net = nn.Sequential(
            dw_pw(in_ch//2 + cond_ch, hidden),
            nn.ELU(inplace=True),
            dw_pw(hidden, hidden),
            nn.ELU(inplace=True),
            nn.Conv2d(hidden, in_ch//2 * 2, 1)           # [log_s, t]
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z, ftr, reverse=False):
        z_a, z_b = torch.chunk(z, 2, 1)
        h = torch.cat([z_a, ftr], 1)
        log_s, t = torch.chunk(self.net(h), 2, 1)
        log_s = torch.tanh(log_s) * self.clamp

        if reverse:
            z_b = (z_b - t) * torch.exp(-log_s)
            logdet = -log_s
        else:
            z_b = z_b * torch.exp(log_s) + t
            logdet =  log_s

        z_out = torch.cat([z_a, z_b], 1)
        return z_out, logdet.sum([1,2,3])     # (B,C,H,W) , (B,)


# ------------------ Slim FlowStep ------------------------------- #
class FlowStep(nn.Module):
    """
    ActNorm ➔ AffineCouplingLite
    非零 log-det，参数量≈原版 25 %。
    """
    def __init__(self, z_channels, ftr_channels,
                 hidden_channels=None, clamp=3.5):
        super().__init__()
        self.actnorm = nn.Identity()
        self.coupling = AffineCouplingLite(z_channels, ftr_channels,
                                           hidden=hidden_channels, clamp=clamp)

    def forward(self, z, ftr, reverse=False):
        logdet_total = torch.zeros(z.size(0), device=z.device)  # (B,)

        # z, ld = self.actnorm(z)
        # logdet_total += ld

        z, ld = self.coupling(z, ftr, reverse)
        logdet_total += ld

        return z, logdet_total        # (B,C,H,W) , (B,)

# ------------------------------
# NVAE-style Bottleneck
# ------------------------------
class NVAEBottleneck(nn.Module):
    """
    NVAE-style VAE bottleneck: initial Gaussian posterior + normalizing flows.
    """
    def __init__(self, in_channels, latent_channels, num_flows=2):
        super().__init__()
        self.latent_channels = latent_channels
        self.mu_head = nn.Sequential(
            nn.Conv2d(in_channels, latent_channels, 3, padding=1),
            nn.ELU(inplace=True)
        )
        self.s_head = nn.Sequential(
            nn.Conv2d(in_channels, latent_channels, 3, padding=1),
            nn.ELU(inplace=True)
        )
        self.eps = 1e-5
        # create flow steps
        self.flows = nn.ModuleList([
            FlowStep(in_channels, latent_channels)
            for _ in range(num_flows)
        ])
    def forward(self, x):
        # 1) Initial posterior q0
        mu = self.mu_head(x)  # μ
        s = self.s_head(x)  # unconstrained logits
        sigma = F.softplus(s) + self.eps  # σ > 0
        log_sigma = torch.log(sigma).clamp(-3., 4.)  # log σ
        logvar = 2.0 * log_sigma  # log σ²
        q0 = Normal(mu, sigma)
        # z0 = loc + eps * scale
        z0 = q0.rsample()
        # log_q0 = q0.log_prob(z0)
        log_q0 = 0.
        # 2) Flows: z0 -> zK
        z = z0
        total_log_det = torch.zeros(z.size(0), device=z.device)  # (B,)
        for flow in self.flows:
            z, log_det = flow(z, x)
            total_log_det = total_log_det + log_det
        return z, mu, logvar, log_q0, total_log_det


class EntropyKLLoss(nn.Module):
    """
    KL + 负熵正则，缩放方式：
      · 对 (H,W) 求和     → [B, C]
      · 通道维取平均      → [B]
      · batch 维取平均    → 标量

    并在通道维上施加 free‑bits（最小 KL 阈值）。
    """

    def __init__(
            self,
            free_nats: float = 5.,  # τ：每通道最小 KL（单位 nat）
            z_prior_weight: float = 0.5,  # z先验正则化权重
            kl_weight: float = 0.5
    ):
        super().__init__()
        self.kl_weight = float(kl_weight)
        self.free_nats = float(free_nats)
        self.z_prior_weight = float(z_prior_weight)
        print(f"[EnhancedEntropyKLLoss] "
              f"free_nats = {self.free_nats} | "
              f"z_prior_weight = {self.z_prior_weight} | "
              f"kl_weight = {self.kl_weight}")

    def forward(
            self,
            z: torch.Tensor,                                                # [B, C, H, W]
            mu: torch.Tensor,                                               # [B, C, H, W]
            logvar: torch.Tensor,                                           # [B, C, H, W]
            total_log_det: torch.Tensor                                     # [B]
    ):
        # # ---------- KL_0 ----------                                      # [B, C]
        var = torch.exp(logvar)
        kl0 = 0.5 * (mu.pow(2) + var - 1.0 - logvar)  # (B,C,H,W)
        kl0_hw = kl0.sum([1, 2, 3])  # (B,)
        # kl0_hw = torch.clamp(kl0_hw, min=self.free_nats)
        kl = kl0_hw - total_log_det

        # ---------- z的先验正则化（直接约束） ----------
        # 鼓励z接近标准正态分布的中心
        z_prior_loss = z.pow(2).sum(dim=[1, 2, 3])  # [B, C]
        # z_prior_loss = z_prior_loss_hw.mean(dim=1)                          # [B]
        # ---------- 总损失 ----------
        loss_per_img = (
                self.kl_weight * kl +
                self.z_prior_weight * z_prior_loss
        )                                                                   # [B]
        output = {"loss": loss_per_img.mean(),
                  "kl": kl.mean().item(),
                  "z":z_prior_loss.mean().item(),
                  "log_det": total_log_det.mean().item()}
        return output                                          # scalar

