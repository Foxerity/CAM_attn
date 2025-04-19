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

from data_loader import get_data_loaders, seed_everything


class ConvBlock(nn.Module):
    """基本卷积块，包含卷积、批归一化和激活函数"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        
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


class AttentionModule(nn.Module):
    """注意力模块，用于关注重要特征"""
    def __init__(self, channels):
        super(AttentionModule, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.conv2 = nn.Conv2d(channels // 8, channels, kernel_size=1)
        
    def forward(self, x):
        # 空间注意力
        attention = F.avg_pool2d(x, x.size()[2:])  # 全局平均池化
        attention = self.conv1(attention)
        attention = F.relu(attention)
        attention = self.conv2(attention)
        attention = torch.sigmoid(attention)  # 归一化到0-1
        
        return x * attention  # 应用注意力权重


class InformationBottleneckLayer(nn.Module):
    """基于信息瓶颈理论的特征提取层
    
    通过控制信息流动，保留与目标相关的信息，丢弃无关信息
    """
    def __init__(self, in_channels, out_channels, beta=0.1):
        super(InformationBottleneckLayer, self).__init__()
        self.beta = beta  # 信息瓶颈中的权衡参数
        
        # 编码器部分
        self.encoder = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            ResidualBlock(out_channels),
            AttentionModule(out_channels)
        )
        
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
        features = self.encoder(x)
        
        # 预测均值和方差
        mu = self.mu_conv(features)
        logvar = self.logvar_conv(features)
        
        # 应用重参数化
        z = self.reparameterize(mu, logvar)
        
        return z, mu, logvar


class CAM(nn.Module):
    """条件对齐模块 (Condition Alignment Module)
    
    基于信息瓶颈理论设计，将不同视觉条件对齐到目标条件
    """
    def __init__(self, config):
        super(CAM, self).__init__()
        self.config = config
        
        # 编码器 - 将源条件图像编码为潜在表示
        self.encoder = nn.Sequential(
            ConvBlock(3, 64),
            ConvBlock(64, 128, stride=2),  # 下采样
            ResidualBlock(128),
            ConvBlock(128, 256, stride=2),  # 下采样
            ResidualBlock(256),
        )
        
        # 信息瓶颈层 - 提取与目标相关的信息
        self.bottleneck = InformationBottleneckLayer(256, 256, beta=config['beta'])
        
        # 解码器 - 将潜在表示解码为目标条件图像
        self.decoder = nn.Sequential(
            ResidualBlock(256),
            nn.Upsample(scale_factor=2),  # 上采样
            ConvBlock(256, 128),
            ResidualBlock(128),
            nn.Upsample(scale_factor=2),  # 上采样
            ConvBlock(128, 64),
            ResidualBlock(64),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Tanh()  # 输出范围为[-1, 1]
        )
        
    def forward(self, source_img):
        # 编码
        features = self.encoder(source_img)
        
        # 信息瓶颈
        z, mu, logvar = self.bottleneck(features)
        
        # 解码
        output = self.decoder(z)
        
        return output, mu, logvar
    
    def kl_divergence_loss(self, mu, logvar):
        """计算KL散度损失，用于信息瓶颈约束"""
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return kl_loss


def train_model(config):
    """训练CAM模型
    
    Args:
        config (dict): 配置参数
    """
    # 设置随机种子
    seed_everything(config['seed'])
    
    # 创建输出目录
    os.makedirs(config['output_dir'], exist_ok=True)
    os.makedirs(os.path.join(config['output_dir'], 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(config['output_dir'], 'samples'), exist_ok=True)
    
    # 初始化TensorBoard
    writer = SummaryWriter(log_dir=os.path.join(config['output_dir'], 'logs'))
    
    # 获取数据加载器
    train_loader, val_loader = get_data_loaders(config)
    
    # 初始化模型
    model = CAM(config).to(config['device'])
    
    # 初始化优化器
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], betas=(0.5, 0.999))
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config['lr_step'], gamma=0.5)
    
    # 损失函数
    l1_loss = nn.L1Loss()
    
    # 训练循环
    best_val_loss = float('inf')
    for epoch in range(config['epochs']):
        model.train()
        epoch_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_kl_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}")
        for batch in pbar:
            source_img = batch['source_img'].to(config['device'])
            target_img = batch['target_img'].to(config['device'])
            
            # 前向传播
            output, mu, logvar = model(source_img)
            
            # 计算损失
            recon_loss = l1_loss(output, target_img)
            kl_loss = model.kl_divergence_loss(mu, logvar) / (output.size(0) * output.size(2) * output.size(3))
            
            # 总损失 = 重建损失 + beta * KL散度损失
            loss = recon_loss + config['beta'] * kl_loss
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 更新损失统计
            epoch_loss += loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_kl_loss += kl_loss.item()
            
            # 更新进度条
            pbar.set_postfix({
                'loss': loss.item(),
                'recon_loss': recon_loss.item(),
                'kl_loss': kl_loss.item()
            })
        
        # 更新学习率
        scheduler.step()
        
        # 计算平均损失
        avg_loss = epoch_loss / len(train_loader)
        avg_recon_loss = epoch_recon_loss / len(train_loader)
        avg_kl_loss = epoch_kl_loss / len(train_loader)
        
        # 记录训练损失
        writer.add_scalar('Loss/train', avg_loss, epoch)
        writer.add_scalar('Loss/recon_train', avg_recon_loss, epoch)
        writer.add_scalar('Loss/kl_train', avg_kl_loss, epoch)
        
        # 验证
        val_loss = validate_model(model, val_loader, l1_loss, config, epoch, writer)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
                'config': config
            }, os.path.join(config['output_dir'], 'checkpoints', 'best_model.pth'))
            print(f"Saved best model with validation loss: {best_val_loss:.4f}")
        
        # 每N个epoch保存一次检查点
        if (epoch + 1) % config['save_interval'] == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'config': config
            }, os.path.join(config['output_dir'], 'checkpoints', f'model_epoch_{epoch+1}.pth'))
    
    writer.close()
    print("Training completed!")


def validate_model(model, val_loader, criterion, config, epoch, writer):
    """验证模型性能
    
    Args:
        model: CAM模型
        val_loader: 验证数据加载器
        criterion: 损失函数
        config: 配置参数
        epoch: 当前epoch
        writer: TensorBoard写入器
        
    Returns:
        float: 验证损失
    """
    model.eval()
    val_loss = 0.0
    val_recon_loss = 0.0
    val_kl_loss = 0.0
    
    # 用于可视化的样本
    vis_samples = []
    
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            source_img = batch['source_img'].to(config['device'])
            target_img = batch['target_img'].to(config['device'])
            
            # 前向传播
            output, mu, logvar = model(source_img)
            
            # 计算损失
            recon_loss = criterion(output, target_img)
            kl_loss = model.kl_divergence_loss(mu, logvar) / (output.size(0) * output.size(2) * output.size(3))
            
            # 总损失
            loss = recon_loss + config['beta'] * kl_loss
            
            # 更新损失统计
            val_loss += loss.item()
            val_recon_loss += recon_loss.item()
            val_kl_loss += kl_loss.item()
            
            # 保存前几个批次的样本用于可视化
            if i < 2:  # 只保存前两个批次
                for j in range(min(4, source_img.size(0))):  # 每个批次最多4个样本
                    vis_samples.append({
                        'source': source_img[j].cpu(),
                        'target': target_img[j].cpu(),
                        'output': output[j].cpu()
                    })
    
    # 计算平均损失
    avg_val_loss = val_loss / len(val_loader)
    avg_val_recon_loss = val_recon_loss / len(val_loader)
    avg_val_kl_loss = val_kl_loss / len(val_loader)
    
    # 记录验证损失
    writer.add_scalar('Loss/val', avg_val_loss, epoch)
    writer.add_scalar('Loss/recon_val', avg_val_recon_loss, epoch)
    writer.add_scalar('Loss/kl_val', avg_val_kl_loss, epoch)
    
    # 可视化样本
    visualize_samples(vis_samples, epoch, config)
    
    print(f"Validation Loss: {avg_val_loss:.4f}, Recon Loss: {avg_val_recon_loss:.4f}, KL Loss: {avg_val_kl_loss:.4f}")
    
    return avg_val_loss


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


if __name__ == "__main__":
    # 配置参数
    config = {
        'dataset_path': '/data/ymx/dataset/imagenet-100',  # 数据集路径
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
    }
    
    # 训练模型
    train_model(config)