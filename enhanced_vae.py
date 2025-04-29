import torch
import torch.nn as nn
import torch.nn.functional as F
from model import ConvBlock, AttentionModule, ResidualBlock


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
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )
        
        self.logvar_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )
        
        # 解码器部分（用于重建隐空间特征）
        self.decoder = nn.Sequential(
            ConvBlock(out_channels, out_channels),
            ResidualBlock(out_channels),
            AttentionModule(out_channels, attention_type)
        )
        
        # 特征投影层，用于对比学习
        self.projection = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1)
        )
        
        # 条件调制层，用于融合条件信息
        self.condition_modulation = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 条件特征适配层，确保条件特征通道数匹配
        self.condition_adapter = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
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
        
        # 应用条件调制（如果提供了条件）
        if condition is not None:
            # 确保条件特征与z具有相同的通道数和空间尺寸
            condition = self.condition_adapter(condition)
            
            # 确保条件特征与z具有相同的空间尺寸
            if condition.shape[2:] != z.shape[2:]:
                condition = F.interpolate(condition, size=z.shape[2:], mode='bilinear', align_corners=False)
            
            # 拼接并生成调制参数
            combined = torch.cat([z, condition], dim=1)
            modulation = self.condition_modulation(combined)
            
            # 应用调制
            z = z * modulation
        
        # 解码增强
        enhanced_z = self.decoder(z)
        
        # 生成投影特征
        proj_features = self.projection(enhanced_z)
        
        return enhanced_z, mu, logvar
    
    def kl_divergence_loss(self, mu, logvar):
        """计算KL散度损失"""
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return self.beta * kl_loss


class VAEFeatureFusion(nn.Module):
    """基于VAE的特征融合模块
    
    将多个条件的特征融合到共享的隐空间中，同时保持各自的独特性。
    使用VAE的隐空间表示进行特征融合和对齐。
    """
    def __init__(self, channels, num_conditions, beta=0.01, attention_type='cbam'):
        super(VAEFeatureFusion, self).__init__()
        self.num_conditions = num_conditions
        
        # 为每个条件创建独立的VAE编码器
        self.encoders = nn.ModuleList()
        for _ in range(num_conditions):
            self.encoders.append(EnhancedVAEBottleneck(channels, channels, beta, attention_type))
        
        # 融合层
        self.fusion = nn.Sequential(
            ConvBlock(channels * num_conditions, channels),
            ResidualBlock(channels),
            AttentionModule(channels, attention_type)
        )
    
    def forward(self, features_list):
        """前向传播
        
        Args:
            features_list: 不同条件的特征列表
            
        Returns:
            融合后的特征、均值列表、对数方差列表和中间特征列表
        """
        assert len(features_list) == self.num_conditions, "特征列表长度必须等于条件数量"
        
        # 编码每个条件的特征
        encoded_features = []
        mus = []
        logvars = []
        all_features = []
        
        for i, (feature, encoder) in enumerate(zip(features_list, self.encoders)):
            # 使用其他条件的平均特征作为条件输入
            other_features = [f for j, f in enumerate(features_list) if j != i]
            condition = torch.mean(torch.stack(other_features), dim=0) if other_features else None
            
            # 编码
            z, mu, logvar, features = encoder(feature, condition)
            encoded_features.append(z)
            mus.append(mu)
            logvars.append(logvar)
            all_features.append(features)
        
        # 拼接并融合
        concat_features = torch.cat(encoded_features, dim=1)
        fused = self.fusion(concat_features)
        
        return fused, mus, logvars, all_features