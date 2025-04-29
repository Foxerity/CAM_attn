import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class ConvBlock(nn.Module):
    """基本卷积块，包含卷积、批归一化和激活函数"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, num_groups=4):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels, affine=True)
        self.relu = nn.LeakyReLU(inplace=True)
        
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    """残差块，用于保持特征信息的同时进行转换"""
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvBlock(channels, channels)
        self.conv2 = ConvBlock(channels, channels)
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + residual  # 残差连接
        return out


class ChannelAttention(nn.Module):
    """通道注意力模块，基于SE-Net设计"""
    def __init__(self, channels, reduction_ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction_ratio, kernel_size=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(channels // reduction_ratio, channels, kernel_size=1)
        )
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return torch.sigmoid(out)


class SpatialAttention(nn.Module):
    """空间注意力模块，关注图像的空间区域"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding)
        
    def forward(self, x):
        # 生成空间注意力图
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention_map = torch.cat([avg_out, max_out], dim=1)
        attention_map = self.conv(attention_map)
        
        return torch.sigmoid(attention_map)


class CBAM(nn.Module):
    """卷积块注意力模块 (Convolutional Block Attention Module)
    
    结合通道注意力和空间注意力，提供更全面的特征增强
    参考论文: "CBAM: Convolutional Block Attention Module"
    """
    def __init__(self, channels, reduction_ratio=8, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)
        
    def forward(self, x):
        # 应用通道注意力
        channel_att = self.channel_attention(x)
        x = x * channel_att
        
        # 应用空间注意力
        spatial_att = self.spatial_attention(x)
        x = x * spatial_att
        
        # 保存注意力图用于可视化
        self.channel_att_map = channel_att
        self.spatial_att_map = spatial_att
        
        return x


class SelfAttention(nn.Module):
    """自注意力模块，基于Transformer设计
    
    允许模型关注输入特征的不同部分，捕获长距离依赖关系
    """
    def __init__(self, channels, reduction_ratio=8):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(channels, channels // reduction_ratio, kernel_size=1)
        self.key = nn.Conv2d(channels, channels // reduction_ratio, kernel_size=1)
        self.value = nn.Conv2d(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        # 生成查询、键和值
        query = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        key = self.key(x).view(batch_size, -1, height * width)
        value = self.value(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        
        # 计算注意力分数
        attention = torch.bmm(query, key)
        attention = F.softmax(attention, dim=-1)
        
        # 应用注意力权重
        out = torch.bmm(attention, value)
        out = out.permute(0, 2, 1).contiguous().view(batch_size, channels, height, width)
        
        # 残差连接
        out = self.gamma * out + x
        
        # 保存注意力图用于可视化
        self.attention_map = attention.detach()
        
        return out


class AttentionModule(nn.Module):
    """综合注意力模块，结合多种注意力机制
    
    基于信息瓶颈理论，通过注意力机制筛选重要信息，减少冗余
    """
    def __init__(self, channels, attention_type='cbam', reduction_ratio=8, kernel_size=7):
        super(AttentionModule, self).__init__()
        self.attention_type = attention_type
        
        if attention_type == 'cbam':
            self.attention = CBAM(channels, reduction_ratio, kernel_size)
        elif attention_type == 'self':
            self.attention = SelfAttention(channels, reduction_ratio)
        elif attention_type == 'channel':
            self.attention = ChannelAttention(channels, reduction_ratio)
        elif attention_type == 'spatial':
            self.attention = SpatialAttention(kernel_size)
        else:
            raise ValueError(f"不支持的注意力类型: {attention_type}")
        
    def forward(self, x):
        return self.attention(x)
    
    def get_attention_maps(self):
        """获取注意力图用于可视化"""
        if self.attention_type == 'cbam':
            return {
                'channel': self.attention.channel_att_map.detach(),
                'spatial': self.attention.spatial_att_map.detach()
            }
        elif self.attention_type == 'self':
            return {'self': self.attention.attention_map.detach()}
        elif self.attention_type == 'channel':
            return {'channel': self.attention.detach()}
        elif self.attention_type == 'spatial':
            return {'spatial': self.attention.detach()}
        else:
            return {}


class InformationBottleneckLayer(nn.Module):
    """基于信息瓶颈理论的特征提取层
    
    通过控制信息流动，保留与目标相关的信息，丢弃无关信息
    """
    def __init__(self, in_channels, out_channels, beta=0.1, attention_type='cbam', reduction_ratio=8):
        super(InformationBottleneckLayer, self).__init__()
        self.beta = beta  # 信息瓶颈中的权衡参数
        
        # 编码器部分
        self.conv = ConvBlock(in_channels, out_channels)
        self.res = ResidualBlock(out_channels)
        self.attention = AttentionModule(out_channels, attention_type, reduction_ratio)
        
        # 均值和方差预测
        self.mu_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.logvar_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
    def reparameterize(self, mu, logvar):
        """重参数化技巧，使得反向传播可行"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        # 编码特征
        features = self.conv(x)
        features = self.res(features)
        features = self.attention(features)
        
        # 预测均值和方差
        mu = self.mu_conv(features)
        logvar = self.logvar_conv(features)
        
        # 应用重参数化
        z = self.reparameterize(mu, logvar)
        
        # 保存注意力图用于可视化
        self.attention_maps = self.attention.get_attention_maps()
        
        return z, mu, logvar
    
    def get_attention_maps(self):
        """获取注意力图用于可视化"""
        return self.attention_maps


class UNetBlock(nn.Module):
    """UNet基本块，包含卷积、注意力和下采样/上采样操作"""
    def __init__(self, in_channels, out_channels, attention_type='cbam', down=True):
        super(UNetBlock, self).__init__()
        self.down = down
        
        if down:
            self.pool = nn.MaxPool2d(2)
            self.conv = nn.Sequential(
                ConvBlock(in_channels, out_channels),
                ConvBlock(out_channels, out_channels)
            )
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = nn.Sequential(
                ConvBlock(in_channels, out_channels),
                ConvBlock(out_channels, out_channels)
            )
            
        self.attention = AttentionModule(out_channels, attention_type)
        
    def forward(self, x, skip=None):
        if self.down:
            if x is not None:  # 第一层没有池化
                x = self.pool(x)
            x = self.conv(x)
            x = self.attention(x)
            return x
        else:
            x = self.up(x)
            x = torch.cat([x, skip], dim=1)
            x = self.conv(x)
            x = self.attention(x)
            return x
    
    def get_attention_maps(self):
        return self.attention.get_attention_maps()


class CAM(nn.Module):
    """条件对齐模块 (Condition Alignment Module)
    
    基于信息瓶颈理论设计，将不同视觉条件对齐到目标条件
    使用UNet架构和注意力机制增强特征提取和生成能力
    """
    def __init__(self, config):
        super(CAM, self).__init__()
        self.config = config
        
        # 获取配置参数
        input_channels = config.get('input_channels', 3)
        output_channels = config.get('output_channels', 3)
        base_channels = config.get('base_channels', 64)
        depth = config.get('depth', 4)  # UNet深度
        attention_type = config.get('attention_type', 'cbam')
        beta = config.get('beta', 0.01)
        
        # 初始卷积层
        self.inc = ConvBlock(input_channels, base_channels)
        
        # 下采样路径
        self.down_blocks = nn.ModuleList()
        in_channels = base_channels
        for i in range(depth):
            out_channels = in_channels * 2
            self.down_blocks.append(UNetBlock(in_channels, out_channels, attention_type, down=True))
            in_channels = out_channels
        
        # 信息瓶颈层
        self.bottleneck = InformationBottleneckLayer(in_channels, in_channels, beta=beta, attention_type=attention_type)
        
        # 上采样路径
        self.up_blocks = nn.ModuleList()
        for i in range(depth):
            out_channels = in_channels // 2
            self.up_blocks.append(UNetBlock(in_channels, out_channels, attention_type, down=False))
            in_channels = out_channels
        
        # 输出层
        self.outc = nn.Sequential(
            nn.Conv2d(base_channels, output_channels, kernel_size=3, padding=1),
            nn.Tanh()  # 输出范围为[-1, 1]
        )
        
    def forward(self, source_img):
        # 初始特征
        x = self.inc(source_img)
        
        # 下采样路径，保存跳跃连接
        skips = [x]
        for i, block in enumerate(self.down_blocks):
            x = block(x)
            if i < len(self.down_blocks) - 1:  # 最后一层不作为跳跃连接
                skips.append(x)
        
        # 信息瓶颈
        z, mu, logvar = self.bottleneck(x)
        x = z
        
        # 上采样路径，使用跳跃连接
        for i, block in enumerate(self.up_blocks):
            skip = skips[-(i+1)] if i < len(skips) else None
            x = block(x, skip)
        
        # 输出层
        output = self.outc(x)
        
        return output, mu, logvar
    
    def kl_divergence_loss(self, mu, logvar):
        """计算KL散度损失，用于信息瓶颈约束"""
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return kl_loss
    
    def get_attention_maps(self):
        """获取所有注意力模块的注意力图"""
        attention_maps = {
            'bottleneck': self.bottleneck.get_attention_maps()
        }
        
        # 获取下采样路径的注意力图
        for i, block in enumerate(self.down_blocks):
            attention_maps[f'down_{i}'] = block.get_attention_maps()
        
        # 获取上采样路径的注意力图
        for i, block in enumerate(self.up_blocks):
            attention_maps[f'up_{i}'] = block.get_attention_maps()
        
        return attention_maps


def visualize_attention_maps(attention_maps, epoch, config):
    """可视化注意力图
    
    Args:
        attention_maps (dict): 注意力图字典，包含各层的注意力图
        epoch (int): 当前epoch
        config (dict): 配置参数
    """
    # 创建保存目录
    save_dir = os.path.join(config['output_dir'], 'attention_maps', f'epoch_{epoch+1}')
    os.makedirs(save_dir, exist_ok=True)
    
    # 遍历所有注意力图
    for layer_name, maps in attention_maps.items():
        for att_type, att_map in maps.items():
            # 处理不同类型的注意力图
            if att_type == 'channel':
                # 通道注意力图通常是 [B, C, 1, 1]
                # 将其转换为热力图
                att = att_map.squeeze().cpu().numpy()
                # 处理多维数组情况
                if att.ndim > 1:
                    # 如果是多维数组，取第一个维度
                    att = att[0] if att.shape[0] <= 32 else att.mean(axis=0)
                plt.figure(figsize=(10, 4))
                plt.bar(range(len(att)), att)
                plt.title(f'Channel Attention - {layer_name}')
                plt.xlabel('Channel')
                plt.ylabel('Attention Weight')
                plt.savefig(os.path.join(save_dir, f'{layer_name}_channel.png'))
                plt.close()
            
            elif att_type == 'spatial':
                # 空间注意力图通常是 [B, 1, H, W]
                att = att_map.squeeze().cpu().numpy()
                # 处理多维数组情况
                if att.ndim > 2:
                    # 如果是多维数组，取第一个样本
                    att = att[0]
                plt.figure(figsize=(6, 6))
                plt.imshow(att, cmap='jet')
                plt.colorbar()
                plt.title(f'Spatial Attention - {layer_name}')
                plt.axis('off')
                plt.savefig(os.path.join(save_dir, f'{layer_name}_spatial.png'))
                plt.close()
            
            elif att_type == 'self':
                # 自注意力图通常是 [B, HW, HW]
                # 可视化为热力图
                att = att_map.cpu().numpy()  # 转换为numpy数组
                # 处理多维数组情况
                if att.ndim > 2:
                    # 取第一个样本
                    att = att[0]
                # 如果维度太大，可以降采样或只显示一部分
                if att.shape[0] > 100 and att.shape[1] > 100:
                    # 降采样到合理大小
                    sample_size = min(100, min(att.shape[0], att.shape[1]))
                    indices = np.linspace(0, min(att.shape[0], att.shape[1])-1, sample_size).astype(int)
                    att = att[indices][:, indices]
                plt.figure(figsize=(10, 10))
                plt.imshow(att, cmap='viridis')
                plt.colorbar()
                plt.title(f'Self Attention - {layer_name}')
                plt.savefig(os.path.join(save_dir, f'{layer_name}_self.png'))
                plt.close()


def visualize_samples(samples, epoch, config):
    """可视化生成的样本
    
    Args:
        samples (list): 样本列表，每个样本包含源图像、目标图像和生成图像
        epoch (int): 当前epoch
        config (dict): 配置参数
    """
    n_samples = len(samples)
    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4 * n_samples))
    
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i, sample in enumerate(samples):
        # 转换张量为numpy数组并调整范围到[0, 1]
        source = sample['source'].permute(1, 2, 0).numpy() * 0.5 + 0.5
        target = sample['target'].permute(1, 2, 0).numpy() * 0.5 + 0.5
        output = sample['output'].permute(1, 2, 0).numpy() * 0.5 + 0.5
        
        # 裁剪到有效范围
        source = np.clip(source, 0, 1)
        target = np.clip(target, 0, 1)
        output = np.clip(output, 0, 1)
        
        # 显示图像
        axes[i, 0].imshow(source)
        axes[i, 0].set_title('Source')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(target)
        axes[i, 1].set_title('Target')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(output)
        axes[i, 2].set_title('Generated')
        axes[i, 2].axis('off')
        
        # 如果有注意力图，也进行可视化
        if 'attention_maps' in sample:
            visualize_attention_maps(sample['attention_maps'], epoch, config)
    
    plt.tight_layout()
    plt.savefig(os.path.join(config['output_dir'], 'samples', f'epoch_{epoch+1}.png'))
    plt.close()
    
    


def inference(model, image_path, config):
    """使用训练好的模型进行推理
    
    Args:
        model: 训练好的CAM模型
        image_path: 输入图像路径
        config: 配置参数
        
    Returns:
        生成的目标条件图像
    """
    from PIL import Image
    from torchvision import transforms
    
    # 加载和预处理图像
    transform = transforms.Compose([
        transforms.Resize((config['img_size'], config['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(config['device'])
    
    # 推理
    model.eval()
    with torch.no_grad():
        output, _, _ = model(image)
    
    # 后处理
    output = output.squeeze(0).cpu()
    output = output.permute(1, 2, 0).numpy() * 0.5 + 0.5
    output = np.clip(output * 255, 0, 255).astype(np.uint8)
    
    return Image.fromarray(output)


def load_model(checkpoint_path, device):
    """加载预训练模型
    
    Args:
        checkpoint_path: 检查点路径
        device: 设备
        
    Returns:
        加载的模型
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    model = CAM(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, config


def visualize_attention(model, image_path, output_path, config):
    """可视化模型的注意力图
    
    Args:
        model: 训练好的CAM模型
        image_path: 输入图像路径
        output_path: 输出目录路径
        config: 配置参数
    """
    from PIL import Image
    from torchvision import transforms
    
    # 加载和预处理图像
    transform = transforms.Compose([
        transforms.Resize((config['img_size'], config['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(config['device'])
    
    # 推理
    model.eval()
    with torch.no_grad():
        output, _, _ = model(image)
        attention_maps = model.get_attention_maps()
    
    # 创建输出目录
    os.makedirs(output_path, exist_ok=True)
    
    # 保存原始图像和生成图像
    original_img = image.squeeze(0).cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5
    original_img = np.clip(original_img * 255, 0, 255).astype(np.uint8)
    Image.fromarray(original_img).save(os.path.join(output_path, 'original.png'))
    
    generated_img = output.squeeze(0).cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5
    generated_img = np.clip(generated_img * 255, 0, 255).astype(np.uint8)
    Image.fromarray(generated_img).save(os.path.join(output_path, 'generated.png'))
    
    # 可视化注意力图
    visualize_attention_maps(attention_maps, 0, {'output_dir': output_path})
    
    return output


if __name__ == "__main__":
    # 配置参数
    config = {
        'dataset_path': r'B:\datasets\test',  # 数据集路径
        'target_condition': 'depth',  # 目标条件
        'source_conditions': ['canny', 'sketch'],  # 源条件
        'img_size': 256,  # 图像大小
        'batch_size': 32,  # 批量大小
        'num_workers': 4,  # 数据加载线程数
        'epochs': 100,  # 训练轮数
        'lr': 2e-4,  # 学习率
        'lr_step': 20,  # 学习率衰减步长
        'beta': 0.01,  # 信息瓶颈权衡参数
        'seed': 42,  # 随机种子
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),  # 设备
        'output_dir': './output',  # 输出目录
        'save_interval': 10,  # 保存间隔
        
        # 新增配置参数
        'input_channels': 3,  # 输入通道数
        'output_channels': 3,  # 输出通道数
        'base_channels': 64,  # 基础通道数
        'depth': 4,  # UNet深度
        'attention_type': 'cbam',  # 注意力类型: 'cbam', 'self', 'channel', 'spatial'
    }
    
    # 训练模型
    train_model(config)