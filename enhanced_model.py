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
from model import ConvBlock, ResidualBlock, AttentionModule, UNetBlock
from losses import EncoderSupervisionLoss


class EnhancedInformationBottleneckLayer(nn.Module):
    """增强版信息瓶颈层，添加了特征提取和保存功能
    
    在原有信息瓶颈层的基础上，增加了特征提取和保存的功能，
    便于后续对编码器进行更直接的监督。
    """
    def __init__(self, in_channels, out_channels, beta=0.1, attention_type='cbam', reduction_ratio=8):
        super(EnhancedInformationBottleneckLayer, self).__init__()
        self.beta = beta  # 信息瓶颈中的权衡参数
        
        # 编码器部分
        self.conv = ConvBlock(in_channels, out_channels)
        self.res = ResidualBlock(out_channels)
        self.attention = AttentionModule(out_channels, attention_type, reduction_ratio)
        
        # 均值和方差预测
        self.mu_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.logvar_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        # 特征投影层，用于对比学习
        self.projection = nn.Sequential(
            nn.Conv2d(out_channels, out_channels // 2, kernel_size=1),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 2, out_channels, kernel_size=1)
        )
        
    def reparameterize(self, mu, logvar):
        """重参数化技巧，使得反向传播可行"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        # 保存中间特征用于监督
        features = []
        
        # 编码特征
        x = self.conv(x)
        features.append(x)  # 保存第一层特征
        
        x = self.res(x)
        features.append(x)  # 保存残差块后的特征
        
        x = self.attention(x)
        features.append(x)  # 保存注意力后的特征
        
        # 预测均值和方差
        mu = self.mu_conv(x)
        logvar = self.logvar_conv(x)
        
        # 应用重参数化
        z = self.reparameterize(mu, logvar)
        
        # 生成投影特征用于对比学习
        proj_features = self.projection(z)
        features.append(proj_features)  # 保存投影后的特征
        
        # 保存注意力图用于可视化
        self.attention_maps = self.attention.get_attention_maps()
        
        return z, mu, logvar, features
    
    def get_attention_maps(self):
        """获取注意力图用于可视化"""
        return self.attention_maps


class EnhancedCAM(nn.Module):
    """增强版条件对齐模块 (Enhanced Condition Alignment Module)
    
    在原有CAM的基础上，增加了对编码器的直接监督机制，
    包括对比学习和特征匹配损失，提高编码器的特征提取能力。
    """
    def __init__(self, config):
        super(EnhancedCAM, self).__init__()
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
        
        # 增强版信息瓶颈层
        self.bottleneck = EnhancedInformationBottleneckLayer(in_channels, in_channels, beta=beta, attention_type=attention_type)
        
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
        
        # 特征提取器，用于从目标图像中提取特征进行匹配
        # 这是一个简化版的编码器，用于提取目标图像的特征
        self.target_encoder = nn.ModuleList()
        in_channels = input_channels
        channels = [base_channels]
        for i in range(depth):
            out_channels = in_channels * 2 if i > 0 else base_channels
            self.target_encoder.append(nn.Sequential(
                ConvBlock(in_channels, out_channels),
                nn.AvgPool2d(kernel_size=2, stride=2)
            ))
            in_channels = out_channels
            channels.append(out_channels)
        
        # 最终的特征提取层
        self.target_encoder.append(nn.Sequential(
            ConvBlock(in_channels, in_channels),
            AttentionModule(in_channels, attention_type)
        ))
        
    def extract_target_features(self, target_img):
        """从目标图像中提取特征，用于特征匹配损失
        
        Args:
            target_img: 目标图像
            
        Returns:
            目标图像的特征列表
        """
        features = []
        x = target_img
        
        for encoder in self.target_encoder:
            x = encoder(x)
            features.append(x)
        
        return features
    
    def forward(self, source_img, target_img=None):
        """前向传播
        
        Args:
            source_img: 源图像
            target_img: 可选，目标图像，用于提取目标特征
            
        Returns:
            输出图像、mu、logvar和特征字典
        """
        # 初始特征
        x = self.inc(source_img)
        
        # 下采样路径，保存跳跃连接和编码器特征
        skips = [x]
        encoder_features = []
        
        for i, block in enumerate(self.down_blocks):
            x = block(x)
            encoder_features.append(x)  # 保存编码器特征
            if i < len(self.down_blocks) - 1:  # 最后一层不作为跳跃连接
                skips.append(x)
        
        # 信息瓶颈
        z, mu, logvar, bottleneck_features = self.bottleneck(x)
        x = z
        
        # 将瓶颈层特征添加到编码器特征列表
        encoder_features.extend(bottleneck_features)
        
        # 上采样路径，使用跳跃连接
        decoder_features = []
        for i, block in enumerate(self.up_blocks):
            skip = skips[-(i+1)] if i < len(skips) else None
            x = block(x, skip)
            decoder_features.append(x)  # 保存解码器特征
        
        # 输出层
        output = self.outc(x)
        
        # 提取目标特征（如果提供了目标图像）
        target_features = None
        if target_img is not None:
            target_features = self.extract_target_features(target_img)
        
        # 返回输出和所有中间特征
        return {
            'output': output,
            'mu': mu,
            'logvar': logvar,
            'encoder_features': encoder_features,
            'decoder_features': decoder_features,
            'target_features': target_features
        }
    
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


def train_enhanced_model(config):
    """训练增强版CAM模型
    
    Args:
        config (dict): 配置参数
    """
    # 设置随机种子
    seed_everything(config['seed'])
    
    # 创建输出目录
    os.makedirs(config['output_dir'], exist_ok=True)
    os.makedirs(os.path.join(config['output_dir'], 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(config['output_dir'], 'samples'), exist_ok=True)
    os.makedirs(os.path.join(config['output_dir'], 'attention_maps'), exist_ok=True)
    
    # 初始化TensorBoard
    writer = SummaryWriter(log_dir=os.path.join(config['output_dir'], 'logs'))
    
    # 获取数据加载器
    train_loader, val_loader = get_data_loaders(config)
    
    # 初始化模型
    model = EnhancedCAM(config).to(config['device'])
    
    # 初始化优化器
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], betas=(0.5, 0.999))
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config['lr_step'], gamma=0.5)
    
    # 初始化损失函数
    loss_fn = EncoderSupervisionLoss(config)
    
    # 训练循环
    best_val_loss = float('inf')
    for epoch in range(config['epochs']):
        model.train()
        epoch_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_kl_loss = 0.0
        epoch_feature_matching_loss = 0.0
        epoch_perceptual_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}")
        for batch in pbar:
            source_img = batch['source_img'].to(config['device'])
            target_img = batch['target_img'].to(config['device'])
            
            # 前向传播
            outputs = model(source_img, target_img)
            
            # 计算损失
            losses = loss_fn(outputs, batch)
            
            # 反向传播和优化
            optimizer.zero_grad()
            losses['total'].backward()
            optimizer.step()
            
            # 更新损失统计
            epoch_loss += losses['total'].item()
            epoch_recon_loss += losses['recon'].item()
            epoch_kl_loss += losses['kl'].item()
            epoch_feature_matching_loss += losses['feature_matching'].item()
            epoch_perceptual_loss = epoch_perceptual_loss + losses['perceptual'].item() if 'perceptual' in losses else epoch_perceptual_loss
            
            # 更新进度条
            pbar.set_postfix({
                'loss': losses['total'].item(),
                'recon_loss': losses['recon'].item(),
                'kl_loss': losses['kl'].item(),
                'feature_matching_loss': losses['feature_matching'].item(),
                'perceptual_loss': losses['perceptual'].item() if 'perceptual' in losses else 0.0
            })
        
        # 更新学习率
        scheduler.step()
        
        # 计算平均损失
        avg_loss = epoch_loss / len(train_loader)
        avg_recon_loss = epoch_recon_loss / len(train_loader)
        avg_kl_loss = epoch_kl_loss / len(train_loader)
        avg_feature_matching_loss = epoch_feature_matching_loss / len(train_loader)
        avg_perceptual_loss = epoch_perceptual_loss / len(train_loader)
        
        # 记录训练损失
        writer.add_scalar('Loss/train', avg_loss, epoch)
        writer.add_scalar('Loss/recon_train', avg_recon_loss, epoch)
        writer.add_scalar('Loss/kl_train', avg_kl_loss, epoch)
        writer.add_scalar('Loss/feature_matching_train', avg_feature_matching_loss, epoch)
        writer.add_scalar('Loss/perceptual_train', avg_perceptual_loss, epoch)
        
        # 验证
        val_loss = validate_enhanced_model(model, val_loader, loss_fn, config, epoch, writer)
        
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


def validate_enhanced_model(model, val_loader, loss_fn, config, epoch, writer):
    """验证增强版模型性能
    
    Args:
        model: 增强版CAM模型
        val_loader: 验证数据加载器
        loss_fn: 损失函数
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
    val_feature_matching_loss = 0.0
    val_perceptual_loss = 0.0
    
    # 用于可视化的样本
    vis_samples = []
    
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            source_img = batch['source_img'].to(config['device'])
            target_img = batch['target_img'].to(config['device'])
            
            # 前向传播
            outputs = model(source_img, target_img)
            
            # 计算损失
            losses = loss_fn(outputs, batch)
            
            # 更新损失统计
            val_loss += losses['total'].item()
            val_recon_loss += losses['recon'].item()
            val_kl_loss += losses['kl'].item()
            val_feature_matching_loss += losses['feature_matching'].item()
            val_perceptual_loss = val_perceptual_loss + losses['perceptual'].item() if 'perceptual' in losses else val_perceptual_loss
            
            # 保存前几个批次的样本用于可视化
            if i < 2:  # 只保存前两个批次
                for j in range(min(4, source_img.size(0))):  # 每个批次最多4个样本
                    sample_dict = {
                        'source': source_img[j].cpu(),
                        'target': target_img[j].cpu(),
                        'output': outputs['output'][j].cpu()
                    }
                    
                    # 如果是第一个样本，获取注意力图用于可视化
                    if i == 0 and j == 0:
                        # 获取模型的注意力图
                        attention_maps = model.get_attention_maps()
                        sample_dict['attention_maps'] = attention_maps
                    
                    vis_samples.append(sample_dict)
    
    # 计算平均损失
    avg_val_loss = val_loss / len(val_loader)
    avg_val_recon_loss = val_recon_loss / len(val_loader)
    avg_val_kl_loss = val_kl_loss / len(val_loader)
    avg_val_feature_matching_loss = val_feature_matching_loss / len(val_loader)
    avg_val_perceptual_loss = val_perceptual_loss / len(val_loader)
    
    # 记录验证损失
    writer.add_scalar('Loss/val', avg_val_loss, epoch)
    writer.add_scalar('Loss/recon_val', avg_val_recon_loss, epoch)
    writer.add_scalar('Loss/kl_val', avg_val_kl_loss, epoch)
    writer.add_scalar('Loss/feature_matching_val', avg_val_feature_matching_loss, epoch)
    writer.add_scalar('Loss/perceptual_val', avg_val_perceptual_loss, epoch)
    
    # 可视化样本
    visualize_samples(vis_samples, epoch, config)
    
    print(f"Validation Loss: {avg_val_loss:.4f}, Recon Loss: {avg_val_recon_loss:.4f}, KL Loss: {avg_val_kl_loss:.4f}, "
          f"Feature Matching Loss: {avg_val_feature_matching_loss:.4f}, "
          f"Perceptual Loss: {avg_val_perceptual_loss:.4f}")
    
    return avg_val_loss


def visualize_samples(samples, epoch, config):
    """可视化生成的样本
    
    Args:
        samples (list): 样本列表，每个样本是一个字典，包含源图像、目标图像和输出图像
        epoch (int): 当前epoch
        config (dict): 配置参数
    """
    # 创建保存目录
    save_dir = os.path.join(config['output_dir'], 'samples', f'epoch_{epoch+1}')
    os.makedirs(save_dir, exist_ok=True)
    
    # 可视化样本
    for i, sample in enumerate(samples):
        # 将张量转换为numpy数组，并调整范围为[0, 1]
        source = sample['source'].permute(1, 2, 0).numpy() * 0.5 + 0.5
        target = sample['target'].permute(1, 2, 0).numpy() * 0.5 + 0.5
        output = sample['output'].permute(1, 2, 0).numpy() * 0.5 + 0.5
        
        # 创建图像网格
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 显示源图像
        axes[0].imshow(source)
        axes[0].set_title('Source')
        axes[0].axis('off')
        
        # 显示目标图像
        axes[1].imshow(target)
        axes[1].set_title('Target')
        axes[1].axis('off')
        
        # 显示输出图像
        axes[2].imshow(output)
        axes[2].set_title('Output')
        axes[2].axis('off')
        
        # 保存图像
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'sample_{i+1}.png'))
        plt.close()
        
        # 如果有注意力图，也进行可视化
        if 'attention_maps' in sample:
            visualize_attention_maps(sample['attention_maps'], epoch, config)


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
            # 将张量转换为numpy数组
            att_map = att_map.cpu().numpy()
            
            # 处理不同类型的注意力图
            if att_type == 'channel':
                # 通道注意力图处理
                # 如果是批次数据，取第一个样本或计算平均值
                if att_map.ndim > 1:
                    if att_map.shape[0] <= 32:
                        att_map = att_map[0]  # 取第一个样本
                    else:
                        att_map = att_map.mean(axis=0)  # 计算平均值
                
                # 确保是1D或2D数据
                att_map = att_map.squeeze()
                plt.figure(figsize=(10, 2))
                if att_map.ndim == 1:
                    plt.imshow(att_map.reshape(1, -1), cmap='viridis', aspect='auto')
                else:
                    plt.imshow(att_map, cmap='viridis', aspect='auto')
                plt.colorbar()
                plt.title(f'{layer_name} - Channel Attention')
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f'{layer_name}_channel_attention.png'))
                plt.close()
            
            elif att_type == 'spatial':
                # 空间注意力图处理
                # 如果是批次数据，取第一个样本或计算平均值
                if att_map.ndim > 2:
                    if att_map.shape[0] <= 32:
                        att_map = att_map[0]  # 取第一个样本
                    else:
                        att_map = att_map.mean(axis=0)  # 计算平均值
                
                # 确保是2D数据
                att_map = att_map.squeeze()
                plt.figure(figsize=(5, 5))
                plt.imshow(att_map, cmap='viridis')
                plt.colorbar()
                plt.title(f'{layer_name} - Spatial Attention')
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f'{layer_name}_spatial_attention.png'))
                plt.close()
            
            elif att_type == 'self':
                # 自注意力图处理
                # 如果是批次数据，取第一个样本
                if att_map.ndim > 2:
                    att_map = att_map[0]  # 取第一个样本
                
                # 如果维度太大，可以降采样或只显示一部分
                if att_map.shape[0] > 100 and att_map.shape[1] > 100:
                    # 降采样到合理大小
                    sample_size = min(100, min(att_map.shape[0], att_map.shape[1]))
                    indices = np.linspace(0, min(att_map.shape[0], att_map.shape[1])-1, sample_size).astype(int)
                    att_map = att_map[indices][:, indices]
                
                plt.figure(figsize=(10, 10))
                plt.imshow(att_map, cmap='viridis')
                plt.colorbar()
                plt.title(f'{layer_name} - Self Attention')
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f'{layer_name}_self_attention.png'))
                plt.close()


def load_enhanced_model(checkpoint_path, config=None, device='cuda'):
    """加载增强版CAM模型
    
    Args:
        checkpoint_path: 检查点路径
        config: 可选，配置字典
        device: 设备，'cuda'或'cpu'
        
    Returns:
        加载的模型
    """
    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 如果没有提供配置，则使用检查点中的配置
    if config is None:
        config = checkpoint['config']
    
    # 确保设备正确
    config['device'] = device
    
    # 创建模型
    model = EnhancedCAM(config).to(device)
    
    # 加载模型参数
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 设置为评估模式
    model.eval()
    
    return model


def enhanced_inference(model, source_img):
    """使用增强版CAM模型进行推理
    
    Args:
        model: 增强版CAM模型
        source_img: 源图像，可以是PIL图像或张量
        
    Returns:
        生成的图像
    """
    # 确保模型处于评估模式
    model.eval()
    
    # 如果输入是PIL图像，则转换为张量
    if not isinstance(source_img, torch.Tensor):
        transform = transforms.Compose([
            transforms.Resize((model.config['img_size'], model.config['img_size'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        source_img = transform(source_img).unsqueeze(0).to(model.config['device'])
    
    # 进行推理
    with torch.no_grad():
        outputs = model(source_img)
        output_img = outputs['output']
    
    return output_img