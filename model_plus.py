import torch
import torch.nn as nn
from model import ConvBlock, UNetBlock, AttentionModule
# 导入新模块
from enhanced_vae import NVAEBottleneck
from fpn_modules import FeaturePyramidNetwork, CrossScaleFeatureFusion


class ConditionSpecificEncoder(nn.Module):
    """条件特定的编码器前端
    
    处理不同视觉条件的特定特征，作为两阶段编码器的第一阶段
    简化版本，只包含基本的下采样和注意力机制
    """

    def __init__(self, input_channels, base_channels, depth=2, attention_type='cbam'):
        super(ConditionSpecificEncoder, self).__init__()

        # 初始卷积层
        self.inc = ConvBlock(input_channels, base_channels)

        # 下采样路径（只包含前depth层）
        self.down_blocks = nn.ModuleList()
        in_channels = base_channels
        self.channels_list = [base_channels]  # 记录每层的通道数

        for i in range(depth):
            out_channels = in_channels * 2
            self.down_blocks.append(UNetBlock(in_channels, out_channels, attention_type, down=True))
            in_channels = out_channels
            self.channels_list.append(out_channels)

        # 条件特定的特征增强层 - 简化版，只使用基本注意力
        self.feature_enhance = nn.Sequential(
            ConvBlock(in_channels, in_channels),
            AttentionModule(in_channels, attention_type)
        )

        # 输出通道数
        self.out_channels = in_channels

    def forward(self, x):
        """前向传播
        
        Args:
            x: 输入图像
            
        Returns:
            编码器特征、跳跃连接和中间特征
        """
        # 初始特征
        x = self.inc(x)

        # 下采样路径，保存跳跃连接和所有特征
        skips = [x]
        features = [x]

        for i, block in enumerate(self.down_blocks):
            x = block(x)
            features.append(x)
            skips.append(x)

        # 特征增强
        x = self.feature_enhance(x)

        return x, skips, features


class SharedEncoder(nn.Module):
    """共享编码器后端
    
    处理来自不同条件特定编码器的特征，提取共享表示
    作为两阶段编码器的第二阶段，应用高级特征工程技术
    """

    def __init__(self, input_channels, base_channels, depth=2, total_depth=4, attention_type='cbam'):
        super(SharedEncoder, self).__init__()

        # 下采样路径（包含后depth层）
        self.down_blocks = nn.ModuleList()
        in_channels = input_channels
        self.channels_list = [input_channels]  # 记录每层的通道数

        for i in range(depth):
            out_channels = in_channels * 2
            self.down_blocks.append(UNetBlock(in_channels, out_channels, attention_type, down=True))
            in_channels = out_channels
            self.channels_list.append(out_channels)

        # 特征金字塔网络
        # 计算所有层级的通道数（包括条件特定编码器的层级）
        fpn_channels = []
        current_channels = base_channels
        for i in range(total_depth + 1):
            fpn_channels.append(current_channels)
            current_channels *= 2

        self.fpn = FeaturePyramidNetwork(
            in_channels_list=fpn_channels,
            out_channels=base_channels * 4,  # 中间层通道数
            attention_type=attention_type
        )

        # 跨尺度特征融合模块
        self.cross_fusion = CrossScaleFeatureFusion(
            channels=base_channels * 4,
            num_scales=len(fpn_channels),
            attention_type=attention_type
        )

        # 最终特征调整
        self.final_adjust = nn.Conv2d(base_channels * 4, in_channels, kernel_size=1, padding=0)

        # 输出通道数
        self.out_channels = in_channels

    def forward(self, x, all_features=None):
        """前向传播
        
        Args:
            x: 条件特定编码器的输出特征
            all_features: 条件特定编码器的所有中间特征
            
        Returns:
            编码器特征和跳跃连接
        """
        # 下采样路径，保存跳跃连接
        skips = []

        # 确保特征列表的一致性，即使对于target特征也能正确处理
        features = all_features.copy()


        for i, block in enumerate(self.down_blocks):
            x = block(x)
            features.append(x)
            skips.append(x)

        # 应用特征金字塔网络增强多尺度特征
        fpn_features = self.fpn(features)
        fused_features = self.cross_fusion(fpn_features)
        enhanced_features = self.final_adjust(fused_features)

        x = x + enhanced_features  # 残差连接

        return x, skips


class Decoder(nn.Module):
    """解码器模块
    
    将编码器特征解码为输出图像
    """

    def __init__(self, in_channels, base_channels, depth=4, attention_type='cbam', output_channels=1):
        super(Decoder, self).__init__()
        # 上采样路径
        self.up_blocks = nn.ModuleList()
        current_channels = in_channels

        for i in range(depth):
            out_channels = current_channels // 2
            self.up_blocks.append(
                UNetBlock(
                    current_channels, out_channels, attention_type,
                    down=False)
                                  )
            current_channels = out_channels

        # 输出层
        self.outc = nn.Sequential(
            nn.Conv2d(base_channels, output_channels, kernel_size=3, padding=1),
            nn.Tanh()  # 输出范围为[-1, 1]
        )

    def forward(self, x, skips=None):
        """前向传播
        
        Args:
            x: 编码器特征
            skips: 跳跃连接列表
            
        Returns:
            解码器输出和中间特征
        """
        # 上采样路径
        for i, block in enumerate(self.up_blocks):
            # 获取对应的跳跃连接
            skip = skips[-(i + 1)] if skips and i < len(skips) else None
            # 应用上采样块
            x = block(x, skip)

        # 输出层
        output = self.outc(x)

        return output


class CAMPlus(nn.Module):
    """增强版条件对齐模块 (CAM+)
    
    使用两阶段编码器架构，前两层为条件特定的，后两层为共享的
    每个条件有自己的专属编码器，但共享一个共享编码器和解码器
    """

    def __init__(self, config):
        super(CAMPlus, self).__init__()
        self.config = config

        # 获取配置参数
        base_channels = config.get('base_channels', 64)
        depth = config.get('depth', 4)  # UNet深度
        attention_type = config.get('attention_type', 'cbam')
        self.source_conditions = config.get('source_conditions', ['canny', 'sketch', 'color'])
        self.bs = config.get("batch_size")

        # 分割深度，前半部分为条件特定，后半部分为共享
        specific_depth = 2
        shared_depth = depth - specific_depth

        # 为每种条件创建专属编码器
        self.specific_encoders = nn.ModuleDict()
        for condition in self.source_conditions:
            # 颜色条件使用3通道，其他条件（sketch、canny、depth）使用1通道
            input_channels = 3 if condition == 'color' else 1
            self.specific_encoders[condition] = ConditionSpecificEncoder(
                input_channels, base_channels, specific_depth, attention_type
            )
            print(f"为条件 {condition} 创建专属编码器，输入通道数: {input_channels}")

        # 创建单个共享编码器实例（所有条件共享）
        specific_out_channels = base_channels * (2 ** specific_depth)
        self.shared_encoder = SharedEncoder(
            specific_out_channels, base_channels, shared_depth, depth, attention_type
        )
        print(f"创建共享编码器，输入通道数: {specific_out_channels}")

        # 为每个条件创建独立的增强版VAE瓶颈层
        self.bottlenecks = nn.ModuleDict()
        bottleneck_channels = base_channels * (2 ** depth)
        for condition in self.source_conditions:
            self.bottlenecks[condition] = NVAEBottleneck(
                bottleneck_channels, bottleneck_channels
            )
        # self.bottlenecks = NVAEBottleneck(
        #         bottleneck_channels, bottleneck_channels
        #     )

        # 共享解码器
        self.decoder = Decoder(
            bottleneck_channels, base_channels, depth, attention_type, output_channels=1
        )

        # 特征提取器，用于从目标图像中提取特征进行匹配
        # 目标条件（通常是depth）使用单通道输入
        target_input_channels = 1  # 目标条件通常是单通道（如深度图）
        self.target_specific_encoder = ConditionSpecificEncoder(
            target_input_channels, base_channels, specific_depth, attention_type
        )
        # 确保目标编码器的输出通道数与其他条件编码器一致
        print(
            f"为目标条件创建专属编码器，输入通道数: {target_input_channels}，输出通道数: {self.target_specific_encoder.out_channels}")

    def forward(self, source_images, target_img=None):
        """前向传播
        
        Args:
            source_images: 字典，键为条件名，值为对应的图像张量
            target_img: 可选，目标图像，用于提取目标特征
            
        Returns:
            包含每个条件生成结果的字典
        """
        # 第一阶段：使用每个条件的专属编码器处理对应的输入图像
        specific_features = {}
        all_skips = {}
        all_features = {}

        for condition, encoder in self.specific_encoders.items():
            if condition in source_images:
                # 使用专属编码器处理输入
                features, skips, mid_features = encoder(source_images[condition])
                specific_features[condition] = features
                all_skips[condition] = skips
                all_features[condition] = mid_features

        # 第二阶段：使用共享编码器处理每个条件的特征
        shared_outputs = {}
        shared_skips = {}

        for condition, features in specific_features.items():
            # 使用共享编码器处理特征，传入所有条件的特征用于跨条件融合
            shared_output, shared_skip = self.shared_encoder(
                features,
                all_features[condition],
            )
            shared_outputs[condition] = shared_output
            shared_skips[condition] = all_skips[condition] + shared_skip

        # 瓶颈阶段：每个编码器输出通过对应的增强版VAE瓶颈层
        all_results = {}
        all_mus = {}
        all_logvars = {}
        all_log_qk = {}
        all_total_log_det = {}
        all_z = {}


        for condition in shared_outputs.keys():
            z, mu, logvar, log_qk, total_log_det = self.bottlenecks[condition](shared_outputs[condition])
            # z, mu, logvar, log_qk, total_log_det = self.bottlenecks(shared_outputs[condition])

            all_mus[condition] = mu
            all_logvars[condition] = logvar
            all_log_qk[condition] = log_qk
            all_total_log_det[condition] = total_log_det
            all_z[condition] = z

            # 使用共享解码器生成输出
            output = self.decoder(z, shared_skips[condition][: -1])
            all_results[condition] = output

        # 返回所有结果
        return {
            'outputs': all_results,  # 每个条件的输出
            'mus': all_mus,
            'logvars': all_logvars,
            'encoder_features': shared_outputs,  # 编码器输出
            'total_log_det': all_total_log_det,
            'all_log_qk': all_log_qk,
            'z': all_z
        }

