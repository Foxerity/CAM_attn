import torch
import torch.nn as nn

from model import ConvBlock, AttentionModule, ResidualBlock
from torch.distributions import Normal


class EnhancedVAEBottleneck(nn.Module):
    """增强版VAE瓶颈层
    
    基于VAE架构的改进瓶颈层，具有更强的表征能力和正则化效果。
    添加了残差连接、注意力机制和多层感知，增强了隐空间的表达能力。
    """
    def __init__(self, in_channels, out_channels, beta=0.01, attention_type='cbam'):
        super(EnhancedVAEBottleneck, self).__init__()
        self.beta = beta  # KL散度权重参数
        
        # 编码器部分
        self.encoder = nn.Sequential(
            ConvBlock(in_channels, in_channels),
            ResidualBlock(in_channels),
            AttentionModule(in_channels, attention_type)
        )
        
        # 均值和对数方差预测
        self.mu_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )
        
        self.logvar_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )
        
        # 解码器部分（用于重建隐空间特征）
        self.decoder = nn.Sequential(
            ConvBlock(out_channels, out_channels),
            ResidualBlock(out_channels),
            AttentionModule(out_channels, attention_type)
        )
    
    def reparameterize(self, mu, logvar):
        """重参数化技巧，使得反向传播可行"""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            # 测试时直接返回均值，减少随机性
            return mu
    
    def forward(self, x, condition=None):
        """前向传播
        
        Args:
            x: 输入特征
            condition: 可选的条件特征，用于条件调制
            
        Returns:
            z: 采样得到的隐变量
            mu: 均值
            logvar: 对数方差
            features: 中间特征列表
        """
        
        # 编码
        encoded = self.encoder(x)
        
        # 预测均值和对数方差
        mu = self.mu_conv(encoded)
        logvar = self.logvar_conv(encoded)
        
        # 采样隐变量
        z = self.reparameterize(mu, logvar)
        
        # 解码增强
        enhanced_z = self.decoder(z)
        
        return enhanced_z, mu, logvar


# ------------------------------
# Flow modules implementation
# ------------------------------
class ARInvertedResidual(nn.Module):
    """
    Simplified invertible residual block for normalizing flow.
    """
    def __init__(self, z_channels, ftr_channels, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or ftr_channels
        self.conv_z = nn.Conv2d(z_channels, hidden_dim, kernel_size=3, padding=1)
        self.conv_ftr = nn.Conv2d(ftr_channels, hidden_dim, kernel_size=1)
        self.act = nn.Hardswish(inplace=True)
    def forward(self, z, ftr):
        return self.act(self.conv_z(z) + self.conv_ftr(ftr))

class MixLogCDFParam(nn.Module):
    """
    Simplified mixture logistic CDF parameters.
    """
    def __init__(self, z_channels, num_mix, hidden_dim):
        super().__init__()
        self.num_mix = num_mix
        self.z_channels = z_channels
        self.conv = nn.Conv2d(hidden_dim, num_mix * z_channels * 5, kernel_size=1)
    def forward(self, feat):
        B, _, H, W = feat.shape
        out = self.conv(feat).view(B, self.num_mix, 5, self.z_channels, H, W)
        return out[:, :, 0], out[:, :, 1], out[:, :, 2], out[:, :, 3], out[:, :, 4]

def mix_log_cdf_flow(z, logit_pi, mu, log_s, log_a, b):
    """Identity flow stub. Returns z unchanged and zero log-det."""
    return z, torch.zeros_like(z)

class ARELUConv(nn.Module):
    """ELU-activated 1×1 conv for affine flow head."""
    def __init__(self, in_ch, out_ch, weight_init_coeff=0.1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        nn.init.kaiming_normal_(self.conv.weight, a=0)
        self.conv.weight.data.mul_(weight_init_coeff)
        self.conv.bias.data.zero_()
        self.act = nn.Hardswish(inplace=True)
    def forward(self, x):
        return self.act(self.conv(x))

class FlowStep(nn.Module):
    """
    A single flow step: ar_conv + optional mix-log-cdf or affine transform.
    """
    def __init__(self, z_channels, ftr_channels, use_mix_log_cdf=False, num_mix=2):
        super().__init__()
        self.ar_conv = ARInvertedResidual(z_channels, ftr_channels)
        self.use_mix = use_mix_log_cdf
        if self.use_mix:
            self.param = MixLogCDFParam(z_channels, num_mix, hidden_dim=self.ar_conv.conv_z.out_channels)
        else:
            self.param = ARELUConv(self.ar_conv.conv_z.out_channels, z_channels)
    def forward(self, z, ftr):
        feat = self.ar_conv(z, ftr)
        if self.use_mix:
            logit_pi, mu_f, log_s, log_a, b = self.param(feat)
            return mix_log_cdf_flow(z, logit_pi, mu_f, log_s, log_a, b)
        else:
            mu_f = self.param(feat)
            z_new = z - mu_f
            return z_new, torch.zeros_like(z)

# ------------------------------
# NVAE-style Bottleneck
# ------------------------------
class NVAEBottleneck(nn.Module):
    """
    NVAE-style VAE bottleneck: initial Gaussian posterior + normalizing flows.
    """
    def __init__(self, in_channels, latent_channels, num_flows=1, use_mix_log_cdf=False):
        super().__init__()
        self.latent_channels = latent_channels
        self.sampler = nn.Conv2d(in_channels, 2 * latent_channels, kernel_size=3, padding=1)
        # create flow steps
        self.flows = nn.ModuleList([
            FlowStep(latent_channels, in_channels, use_mix_log_cdf)
            for _ in range(num_flows)
        ])
    def forward(self, x):
        # 1) Initial posterior q0
        params = self.sampler(x)
        mu, logvar = torch.chunk(params, 2, dim=1)
        sigma = torch.exp(0.5 * logvar).clamp(min=1e-6)
        q0 = Normal(mu, sigma)
        # z0 = loc + eps * scale
        z0 = q0.rsample()
        log_q0 = q0.log_prob(z0)
        # 2) Flows: z0 -> zK
        z = z0
        total_log_det = torch.zeros_like(z)
        for flow in self.flows:
            z, log_det = flow(z, x)
            total_log_det = total_log_det + log_det
        return z, mu, logvar, log_q0, total_log_det


# ------------------------------
# KL Loss Module (Analytic + Flow)
# ------------------------------
class FlowKLLoss(nn.Module):
    """
    Combined KL divergence:
      - Gaussian posterior analytic KL (q0 || N(0,1))
      - subtract flow log-determinants
    """
    def __init__(self):
        super().__init__()

    def forward(self, mu: torch.Tensor, logvar: torch.Tensor, total_log_det: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mu:       [B, Z, H, W], posterior mean
            logvar:   [B, Z, H, W], posterior log-variance
            total_log_det: [B, Z, H, W], sum of flow log-determinants
        Returns:
            scalar KL loss (batch mean)
        """
        # 1) analytic KL0 between N(mu,σ²) and N(0,1)
        #    0.5 * ∑ (μ² + σ² − 1 − log σ²)
        logvar_clamped = logvar
        var = torch.exp(logvar_clamped)
        kl0 = 0.5 * (mu.pow(2) + var - 1.0 - logvar_clamped)
        # sum over latent dims and spatial dims → [B]
        kl0 = kl0.sum(dim=[1, 2, 3])

        # 2) sum of all flow log‐det per sample → [B]
        flow_ld = total_log_det.sum(dim=[1, 2, 3])

        # 3) combined KL per sample
        kl = kl0 - flow_ld

        # 4) batch mean and scale
        return kl.mean()


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
            entropy_weight: float = 0.5,  # α：负熵项权重
            free_nats: float = 0.1,  # τ：每通道最小 KL（单位 nat）
            z_prior_weight: float = 0.5,  # z先验正则化权重
            kl_weight: float = 0.5
    ):
        super().__init__()
        self.kl_weight = float(kl_weight)
        self.entropy_weight = float(entropy_weight)
        self.free_nats = float(free_nats)
        self.free_nats = self.free_nats * 16.
        self.z_prior_weight = float(z_prior_weight)
        print(f"[EnhancedEntropyKLLoss] "
              f"entropy_weight = {self.entropy_weight} | "
              f"free_nats = {self.free_nats} | "
              f"z_prior_weight = {self.z_prior_weight} | "
              f"kl_weight = {self.kl_weight}")

    def forward(
            self,
            z: torch.Tensor,                                                # [B, C, H, W]
            mu: torch.Tensor,                                               # [B, C, H, W]
            logvar: torch.Tensor,                                           # [B, C, H, W]
            log_q0: torch.Tensor,                                           # [B, C, H, W]  ——  q0.log_prob(z0)
            total_log_det: torch.Tensor                                     # [B, C, H, W]
    ):
        # ---------- KL_0 ----------
        var = torch.exp(logvar)
        kl0 = 0.5 * (mu.pow(2) + var - 1.0 - logvar)                        # [B,C,H,W]

        # ---------- HW 维度求和 ----------
        kl0_hw = kl0.sum(dim=[1, 2, 3])                                        # [B, C]
        ld_hw = total_log_det.sum(dim=[1, 2, 3])                               # [B, C]
        kl_hw = kl0_hw - ld_hw                                              # [B, C]

        # ---------- Free‑bits ----------
        # kl_clamped = torch.clamp(kl_hw, min=self.free_nats)                 # [B, C]
        # ---------- 负熵项 ----------
        # neg_entropy_hw = (-log_q0).sum(dim=[2, 3])
        # neg_entropy = neg_entropy_hw.mean(dim=1)
        q0 = torch.exp(log_q0)
        neg_entropy_hw = 0.5 * (q0 - log_q0 - 0.5).sum(dim=[1, 2, 3])          # [B, C]
        # neg_entropy = neg_entropy_hw.mean(dim=1)                            # [B,]
        neg_entropy = neg_entropy_hw

        # ---------- z的先验正则化（直接约束） ----------
        # 鼓励z接近标准正态分布的中心
        z_prior_loss_hw = 0.5 * (z.pow(2) - 0.5).sum(dim=[1, 2, 3])            # [B, C]
        # z_prior_loss = z_prior_loss_hw.mean(dim=1)                          # [B]
        z_prior_loss = z_prior_loss_hw
        # ---------- 总损失 ----------
        loss_per_img = (
                self.kl_weight * kl_hw +
                self.entropy_weight * neg_entropy +
                self.z_prior_weight * z_prior_loss
        )                                                                   # [B]
        output = {"loss": loss_per_img.mean(),
                  "kl": kl_hw.mean().item(),
                  "q0": neg_entropy.mean().item(),
                  "z":z_prior_loss.mean().item()}
        return output                                          # scalar


class FreeBitsKLLoss(nn.Module):
    def __init__(self, free_nats=0.1, group_by='channel'):
        super().__init__()
        self.free_nats = free_nats          # τ，单位 nat
        self.group_by = group_by            # 'channel' 或 'all'
        print(f"Using {self.free_nats}/nat FreeBits Control...")

    def forward(self, mu, logvar, total_log_det):
        var  = torch.exp(logvar)
        kl0  = 0.5 * (mu.pow(2) + var - 1.0 - logvar)     # [B,C,H,W]
        kl   = kl0 - total_log_det                        # 与 flow 结合

        kl_group = kl.sum(dim=[2, 3])
        # ② “free bits” 下限 = τ × H × W
        H, W = kl.size(2), kl.size(3)
        free_nats_per_chan = self.free_nats * H * W

        # ③ clamp 然后再 sum → batch 均值
        kl_group = torch.clamp(kl_group, min=free_nats_per_chan)
        kl_final = kl_group.sum(dim=1).mean()  # 标量

        return kl_final

