import torch
import torch.nn as nn
import torch.nn.functional as F
from model import ConvBlock, AttentionModule


class FeaturePyramidNetwork(nn.Module):
    """特征金字塔网络 (Feature Pyramid Network)
    
    用于增强编码器的多尺度特征提取能力，通过自顶向下的路径和横向连接
    融合不同尺度的特征，生成具有强语义信息的特征金字塔。
    参考论文: "Feature Pyramid Networks for Object Detection"
    """
    def __init__(self, in_channels_list, out_channels, attention_type='self'):
        super(FeaturePyramidNetwork, self).__init__()
        
        # 横向连接层 (lateral connections)
        self.lateral_convs = nn.ModuleList()
        for in_channels in in_channels_list:
            self.lateral_convs.append(ConvBlock(in_channels, out_channels, kernel_size=1, padding=0))
        
        # 特征融合后的3x3卷积，用于消除上采样的混叠效应
        self.smooth_convs = nn.ModuleList()
        for _ in range(len(in_channels_list)):
            self.smooth_convs.append(
                ConvBlock(out_channels, out_channels, kernel_size=3, padding=1)
            )
        
        # 注意力模块，用于增强特征
        self.attention_modules = nn.ModuleList()
        for _ in range(len(in_channels_list)):
            self.attention_modules.append(AttentionModule(out_channels, attention_type))
    
    def forward(self, features):
        """
        Args:
            features: 编码器各层特征的列表，从浅层到深层排列
            
        Returns:
            增强后的特征金字塔列表
        """
        # 应用横向连接，将所有特征映射到相同的通道数
        laterals = [conv(feature) for feature, conv in zip(features, self.lateral_convs)]
        
        # 自顶向下的路径和特征融合
        fpn_features = [laterals[-1]]  # 从最深层开始
        for i in range(len(laterals)-2, -1, -1):  # 从倒数第二层到第一层
            # 上采样深层特征
            upsampled = F.interpolate(fpn_features[0], size=laterals[i].shape[2:], mode='bilinear', align_corners=False)
            # 融合当前层的特征
            merged = laterals[i] + upsampled
            # 应用平滑卷积
            smoothed = self.smooth_convs[i](merged)
            # 应用注意力
            enhanced = self.attention_modules[i](smoothed)
            # 将当前层的增强特征添加到结果列表的开头
            fpn_features.insert(0, enhanced)
        
        return fpn_features


class CrossScaleFeatureFusion(nn.Module):
    """跨尺度特征融合模块
    
    融合特征金字塔中不同尺度的特征，生成更丰富的表示。
    使用自适应权重机制动态调整不同尺度特征的重要性。
    """
    def __init__(self, channels, num_scales, attention_type='cbam', mode='none'):
        super(CrossScaleFeatureFusion, self).__init__()

        self.mode = mode
        if mode != 'none':
            # 特征转换层
            self.transforms = nn.ModuleList()
            for _ in range(num_scales):
                self.transforms.append(ConvBlock(channels, channels))

            # 自适应权重生成
            self.weight_generator = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels * num_scales, num_scales, kernel_size=1),
                nn.Softmax(dim=1)
            )

            # 融合后的特征增强
            self.fusion_enhance = nn.Sequential(
                ConvBlock(channels, channels),
                AttentionModule(channels, attention_type)
            )
        else:
            self.weight_generator = ConvBlock(channels * num_scales, channels)

    def forward(self, features):
        """
        Args:
            features: 不同尺度的特征列表
            
        Returns:
            融合后的特征
        """
        # 确保所有特征具有相同的空间尺寸（调整到第一个特征的尺寸）
        # target_size = features[0].shape[2:]
        target_size = features[-1].shape[2:]
        aligned_features = []
        
        for i, feature in enumerate(features):
            # 应用特征转换
            if self.mode != 'none':
                transformed = self.transforms[i](feature)
            else:
                transformed = feature
            # 调整空间尺寸
            if transformed.shape[2:] != target_size:
                transformed = F.interpolate(transformed, size=target_size, mode='bilinear', align_corners=False)
            aligned_features.append(transformed)
        
        # 计算自适应权重
        if self.mode != 'none':
            concat_features = torch.cat(aligned_features, dim=1)
            weights = self.weight_generator(concat_features)

            # 应用权重并融合
            weighted_features = []
            for i, feature in enumerate(aligned_features):
                # 提取对应的权重并扩展维度以匹配特征图
                weight = weights[:, i:i+1].expand_as(feature)
                weighted_features.append(feature * weight)
        
            # 求和融合
            fused = sum(weighted_features)
            # 增强融合特征
            enhanced = self.fusion_enhance(fused)
        else:
            fused = sum(aligned_features) / len(aligned_features)
            concat_features = torch.cat(aligned_features, dim=1)
            weight = self.weight_generator(concat_features)
            enhanced = fused + weight
        return enhanced

        
