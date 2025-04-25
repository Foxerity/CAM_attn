import torch
import torch.nn as nn
import torch.nn.functional as F
from model import ConvBlock, UNetBlock, AttentionModule
from enhanced_model import EnhancedInformationBottleneckLayer


class ColorEncoder(nn.Module):
    """专门用于处理颜色条件的编码器
    
    颜色条件与其他条件（如边缘、草图等）有显著差异，需要特殊处理
    """
    def __init__(self, input_channels, base_channels, depth, attention_type='cbam'):
        super(ColorEncoder, self).__init__()
        
        # 初始卷积层
        self.inc = ConvBlock(input_channels, base_channels)
        
        # 下采样路径
        self.down_blocks = nn.ModuleList()
        in_channels = base_channels
        for i in range(depth):
            out_channels = in_channels * 2
            self.down_blocks.append(UNetBlock(in_channels, out_channels, attention_type, down=True))
            in_channels = out_channels
        
        # 颜色特征提取层
        self.color_feature_layer = nn.Sequential(
            ConvBlock(in_channels, in_channels),
            AttentionModule(in_channels, attention_type)
        )
    
    def forward(self, x):
        """前向传播
        
        Args:
            x: 输入图像
            
        Returns:
            编码器特征和跳跃连接
        """
        # 初始特征
        x = self.inc(x)
        
        # 下采样路径，保存跳跃连接
        skips = [x]
        for i, block in enumerate(self.down_blocks):
            x = block(x)
            if i < len(self.down_blocks) - 1:  # 最后一层不作为跳跃连接
                skips.append(x)
        
        # 颜色特征提取
        x = self.color_feature_layer(x)
        
        return x, skips


class StandardEncoder(nn.Module):
    """标准编码器，用于处理边缘、草图等条件
    """
    def __init__(self, input_channels, base_channels, depth, attention_type='cbam'):
        super(StandardEncoder, self).__init__()
        
        # 初始卷积层
        self.inc = ConvBlock(input_channels, base_channels)
        
        # 下采样路径
        self.down_blocks = nn.ModuleList()
        in_channels = base_channels
        for i in range(depth):
            out_channels = in_channels * 2
            self.down_blocks.append(UNetBlock(in_channels, out_channels, attention_type, down=True))
            in_channels = out_channels
    
    def forward(self, x):
        """前向传播
        
        Args:
            x: 输入图像
            
        Returns:
            编码器特征和跳跃连接
        """
        # 初始特征
        x = self.inc(x)
        
        # 下采样路径，保存跳跃连接
        skips = [x]
        for i, block in enumerate(self.down_blocks):
            x = block(x)
            if i < len(self.down_blocks) - 1:  # 最后一层不作为跳跃连接
                skips.append(x)
        
        return x, skips


class SkipConnectionFusion(nn.Module):
    """跳跃连接融合模块
    
    将不同编码器的跳跃连接特征融合为单一特征，用于解码器
    """
    def __init__(self, channels, num_encoders):
        super(SkipConnectionFusion, self).__init__()
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * num_encoders, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, skip_connections):
        # 输入是一个列表，包含来自不同编码器的同一层级的跳跃连接
        # 将它们在通道维度上拼接
        if not skip_connections:
            return None
        
        # 过滤掉None值
        valid_skips = [skip for skip in skip_connections if skip is not None]
        if not valid_skips:
            return None
        
        # 拼接并融合
        concat_skips = torch.cat(valid_skips, dim=1)
        fused_skip = self.fusion(concat_skips)
        return fused_skip


class StandardEncoder(nn.Module):
    """标准编码器，用于处理所有类型的条件
    """
    def __init__(self, input_channels, base_channels, depth, attention_type='cbam'):
        super(StandardEncoder, self).__init__()
        
        # 初始卷积层
        self.inc = ConvBlock(input_channels, base_channels)
        
        # 下采样路径
        self.down_blocks = nn.ModuleList()
        in_channels = base_channels
        for i in range(depth):
            out_channels = in_channels * 2
            self.down_blocks.append(UNetBlock(in_channels, out_channels, attention_type, down=True))
            in_channels = out_channels
        
        # 特征提取层（对所有条件通用）
        self.feature_layer = nn.Sequential(
            ConvBlock(in_channels, in_channels),
            AttentionModule(in_channels, attention_type)
        )
    
    def forward(self, x):
        """前向传播
        
        Args:
            x: 输入图像
            
        Returns:
            编码器特征和跳跃连接
        """
        # 初始特征
        x = self.inc(x)
        
        # 下采样路径，保存跳跃连接
        skips = [x]
        for i, block in enumerate(self.down_blocks):
            x = block(x)
            if i < len(self.down_blocks) - 1:  # 最后一层不作为跳跃连接
                skips.append(x)
        
        # 特征提取
        x = self.feature_layer(x)
        
        return x, skips


class CAMPlus(nn.Module):
    """增强版条件对齐模块 (CAM+)
    
    多编码器单解码器架构，每种条件使用专用编码器，共享一个解码器
    每个条件独立生成结果，并分别计算损失
    """
    def __init__(self, config):
        super(CAMPlus, self).__init__()
        self.config = config
        
        # 获取配置参数
        base_channels = config.get('base_channels', 64)
        depth = config.get('depth', 4)  # UNet深度
        attention_type = config.get('attention_type', 'cbam')
        beta = config.get('beta', 0.01)
        self.source_conditions = config.get('source_conditions', ['canny', 'sketch', 'color'])
        
        # 为每种条件创建专用编码器，根据条件类型设置不同的输入通道数
        self.encoders = nn.ModuleDict()
        for condition in self.source_conditions:
            # 颜色条件使用3通道，其他条件（sketch、canny、depth）使用1通道
            input_channels = 3 if condition == 'color' else 1
            self.encoders[condition] = StandardEncoder(
                input_channels, base_channels, depth, attention_type
            )
            print(f"为条件 {condition} 创建编码器，输入通道数: {input_channels}")
        
        # 为每个编码器创建独立的信息瓶颈层
        self.bottlenecks = nn.ModuleDict()
        bottleneck_channels = base_channels * (2 ** depth)
        for condition in self.source_conditions:
            self.bottlenecks[condition] = EnhancedInformationBottleneckLayer(
                bottleneck_channels, bottleneck_channels, beta=beta, attention_type=attention_type
            )
        
        # 跳跃连接融合模块（每个层级一个）
        self.skip_fusions = nn.ModuleList()
        skip_channels = [base_channels]
        for i in range(depth-1):
            skip_channels.append(base_channels * (2 ** (i+1)))
        
        for channels in skip_channels:
            self.skip_fusions.append(SkipConnectionFusion(channels, len(self.source_conditions)))
        
        # 共享解码器
        self.up_blocks = nn.ModuleList()
        in_channels = bottleneck_channels
        for i in range(depth):
            out_channels = in_channels // 2
            self.up_blocks.append(UNetBlock(in_channels, out_channels, attention_type, down=False))
            in_channels = out_channels
        
        # 输出层 - 修改为输出单通道深度图
        output_channels = 1  # 固定为单通道输出
        self.outc = nn.Sequential(
            nn.Conv2d(base_channels, output_channels, kernel_size=3, padding=1),
            nn.Tanh()  # 输出范围为[-1, 1]
        )
        
        # 特征提取器，用于从目标图像中提取特征进行匹配
        # 目标条件（通常是depth）使用单通道输入
        target_input_channels = 1  # 目标条件通常是单通道（如深度图）
        self.target_encoder = StandardEncoder(target_input_channels, base_channels, depth, attention_type)
        print(f"为目标条件创建编码器，输入通道数: {target_input_channels}")
    
    def forward(self, source_images, target_img=None):
        """前向传播
        
        Args:
            source_images: 字典，键为条件名，值为对应的图像张量
            target_img: 可选，目标图像，用于提取目标特征
            
        Returns:
            包含每个条件生成结果的字典
        """
        # 使用每个条件的专用编码器处理对应的输入图像
        encoder_outputs = {}
        all_skips = {}
        all_results = {}
        all_mus = {}
        all_logvars = {}
        all_bottleneck_features = {}
        
        # 1. 编码阶段：每个条件通过对应的编码器
        for condition, encoder in self.encoders.items():
            if condition in source_images:
                encoder_outputs[condition], all_skips[condition] = encoder(source_images[condition])
        
        # 2. 瓶颈阶段：每个编码器输出通过对应的信息瓶颈层
        for condition in encoder_outputs.keys():
            z, mu, logvar, bottleneck_features = self.bottlenecks[condition](encoder_outputs[condition])
            encoder_outputs[condition] = z  # 更新为瓶颈层输出
            all_mus[condition] = mu
            all_logvars[condition] = logvar
            all_bottleneck_features[condition] = bottleneck_features
        
        # 3. 解码阶段：每个条件独立解码
        decoder_features = {condition: [] for condition in encoder_outputs.keys()}
        
        # 为每个条件单独解码生成结果
        for condition, z in encoder_outputs.items():
            x = z
            # 使用当前条件的跳跃连接
            current_skips = all_skips[condition]
            
            # 上采样路径，使用当前条件的跳跃连接
            for i, block in enumerate(self.up_blocks):
                skip = current_skips[-(i+1)] if i < len(current_skips) else None
                x = block(x, skip)
                decoder_features[condition].append(x)
            
            # 输出层
            output = self.outc(x)
            all_results[condition] = output
        
        # 提取目标特征（如果提供了目标图像）
        target_features = None
        if target_img is not None:
            target_features, _ = self.target_encoder(target_img)
        
        # 返回所有结果
        return {
            'outputs': all_results,  # 每个条件的输出
            'mus': all_mus,
            'logvars': all_logvars,
            'encoder_features': encoder_outputs,  # 编码器输出
            'decoder_features': decoder_features,  # 解码器特征
            'target_features': target_features,  # 目标特征
            'bottleneck_features': all_bottleneck_features  # 瓶颈层特征
        }
    
    def kl_divergence_loss(self, mu, logvar):
        """计算KL散度损失，用于信息瓶颈约束"""
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return kl_loss


def train_model_plus(config):
    """训练CAM+模型
    
    Args:
        config: 配置参数
    """
    # 确保配置中包含设备信息
    if 'device' not in config:
        config['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"使用设备: {config['device']}")
    
    # 设置CUDA性能优化
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True  # 加速卷积操作
        torch.backends.cudnn.deterministic = False  # 允许非确定性优化
    from data_loader_plus import get_multi_condition_loaders
    from losses import ReconstructionLoss, FeatureMatchingLoss, ContrastiveLoss
    import os
    import time
    import torch.optim as optim
    from torch.optim.lr_scheduler import StepLR
    from tqdm import tqdm
    import numpy as np
    from utils import save_image_grid, compute_psnr, compute_ssim
    
    # 创建输出目录
    output_dir = config.get('output_dir', './output_plus')
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取数据加载器
    train_loader, val_loader = get_multi_condition_loaders(config)
    
    # 创建模型
    model = CAMPlus(config).to(config['device'])
    
    # 定义损失函数
    recon_loss_fn = ReconstructionLoss(loss_type=config.get('recon_loss_type', 'l1'))
    feature_matching_loss_fn = FeatureMatchingLoss(loss_type=config.get('feature_matching_loss_type', 'l1'))
    contrastive_loss_fn = ContrastiveLoss(temperature=config.get('temperature', 0.5))
    
    # 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    
    # 学习率调度器
    scheduler = StepLR(optimizer, step_size=config['lr_step'], gamma=0.5)
    
    # 训练循环
    best_val_loss = float('inf')
    start_time = time.time()
    
    for epoch in range(config['epochs']):
        # 训练阶段
        model.train()
        train_losses = []
        train_condition_losses = {condition: [] for condition in config['source_conditions']}
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']} [Train]"):
            # 获取数据
            source_images = batch['source_images']  # 字典，键为条件名，值为对应的图像张量
            target_img = batch['target_img'].to(config['device'])
            
            # 将所有源图像移动到设备
            for condition in source_images:
                source_images[condition] = source_images[condition].to(config['device'])
            
            # 前向传播
            outputs = model(source_images, target_img)
            output_imgs = outputs['outputs']  # 字典，键为条件名，值为对应的生成图像
            mus = outputs['mus']
            logvars = outputs['logvars']
            encoder_features = outputs['encoder_features']
            target_features = outputs['target_features']
            
            # 计算每个条件的损失
            total_recon_loss = 0
            total_kl_loss = 0
            condition_losses = {}
            
            for condition, output_img in output_imgs.items():
                # 1. 重建损失
                recon_loss = recon_loss_fn(output_img, target_img)
                
                # 2. KL散度损失
                kl_loss = model.kl_divergence_loss(mus[condition], logvars[condition])
                
                # 3. 特征匹配损失
                feature_matching_loss = 0
                if target_features is not None:
                    feature_matching_loss = feature_matching_loss_fn(encoder_features[condition], target_features)
                
                # 条件权重（可以为不同条件设置不同权重）
                condition_weight = config.get(f'{condition}_weight', 1.0)
                feature_matching_weight = config.get('feature_matching_weight', 0.1)
                kl_weight = config.get('beta', 0.01)
                
                # 计算当前条件的总损失
                condition_loss = (
                    recon_loss + 
                    kl_weight * kl_loss + 
                    feature_matching_weight * feature_matching_loss
                ) * condition_weight
                
                condition_losses[condition] = condition_loss
                total_recon_loss += recon_loss * condition_weight
                total_kl_loss += kl_loss * condition_weight
                
                # 记录每个条件的损失
                train_condition_losses[condition].append(condition_loss.item())
            
            # 4. 对比学习损失（在所有编码器特征之间）
            contrastive_loss = 0
            if len(encoder_features) > 1:
                contrastive_loss = contrastive_loss_fn(list(encoder_features.values()))
            
            # 总损失：所有条件损失之和 + 对比损失
            contrastive_weight = config.get('contrastive_weight', 0.1)
            total_loss = sum(condition_losses.values()) + contrastive_weight * contrastive_loss
            
            # 反向传播和优化
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # 记录总损失
            train_losses.append(total_loss.item())
        
        # 更新学习率
        scheduler.step()
        
        # 计算平均训练损失
        avg_train_loss = np.mean(train_losses)
        avg_condition_losses = {cond: np.mean(losses) for cond, losses in train_condition_losses.items()}
        
        # 验证阶段
        model.eval()
        val_losses = []
        val_condition_metrics = {condition: {'psnr': [], 'ssim': []} for condition in config['source_conditions']}
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{config['epochs']} [Val]"):
                # 获取数据
                source_images = batch['source_images']
                target_img = batch['target_img'].to(config['device'])
                
                # 将所有源图像移动到设备
                for condition in source_images:
                    source_images[condition] = source_images[condition].to(config['device'])
                
                # 前向传播
                outputs = model(source_images)
                output_imgs = outputs['outputs']
                mus = outputs['mus']
                logvars = outputs['logvars']
                
                # 计算每个条件的损失和评估指标
                batch_loss = 0
                
                for condition, output_img in output_imgs.items():
                    # 计算损失
                    recon_loss = recon_loss_fn(output_img, target_img)
                    kl_loss = model.kl_divergence_loss(mus[condition], logvars[condition])
                    
                    condition_weight = config.get(f'{condition}_weight', 1.0)
                    kl_weight = config.get('beta', 0.01)
                    
                    condition_loss = (recon_loss + kl_weight * kl_loss) * condition_weight
                    batch_loss += condition_loss
                    
                    # 计算评估指标
                    val_condition_metrics[condition]['psnr'].append(compute_psnr(output_img, target_img))
                    val_condition_metrics[condition]['ssim'].append(compute_ssim(output_img, target_img))
                
                val_losses.append(batch_loss.item())
        
        # 计算平均验证损失和评估指标
        avg_val_loss = np.mean(val_losses)
        avg_condition_metrics = {}
        for condition, metrics in val_condition_metrics.items():
            avg_condition_metrics[condition] = {
                'psnr': np.mean(metrics['psnr']),
                'ssim': np.mean(metrics['ssim'])
            }
        
        # 打印训练信息
        print(f"Epoch {epoch+1}/{config['epochs']} - "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}")
        
        # 打印每个条件的评估指标
        for condition, metrics in avg_condition_metrics.items():
            print(f"  {condition} - PSNR: {metrics['psnr']:.2f}, SSIM: {metrics['ssim']:.4f}")
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'config': config
            }
            torch.save(checkpoint, os.path.join(output_dir, 'best_model.pth'))
            print(f"保存最佳模型，验证损失: {avg_val_loss:.4f}")
        
        # 定期保存模型
        if (epoch + 1) % config.get('save_interval', 10) == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'config': config
            }
            torch.save(checkpoint, os.path.join(output_dir, f'model_epoch_{epoch+1}.pth'))
        
        # 保存生成的图像样本
        if (epoch + 1) % config.get('sample_interval', 5) == 0:
            # 选择一个批次的验证数据
            val_batch = next(iter(val_loader))
            source_images = val_batch['source_images']
            target_img = val_batch['target_img'].to(config['device'])
            
            # 将所有源图像移动到设备
            for condition in source_images:
                source_images[condition] = source_images[condition].to(config['device'])
            
            # 生成图像
            with torch.no_grad():
                outputs = model(source_images)
                output_imgs = outputs['outputs']
            
            # 保存图像网格
            # 创建一个包含源图像、每个条件生成的图像和目标图像的网格
            images = []
            
            # 添加目标图像
            images.append(("Target", target_img))
            
            # 添加每个条件的源图像和生成图像
            for condition in source_images:
                images.append((f"Source ({condition})", source_images[condition]))
                if condition in output_imgs:
                    images.append((f"Generated ({condition})", output_imgs[condition]))
            
            save_image_grid(
                images,
                os.path.join(output_dir, f'samples_epoch_{epoch+1}.png'),
                nrow=3  # 每行显示3张图像
            )
    
    # 训练完成
    total_time = time.time() - start_time
    print(f"训练完成，总用时: {total_time/60:.2f}分钟")
    print(f"最佳验证损失: {best_val_loss:.4f}")
    
    # 打印每个条件的最终评估指标
    print("最终评估指标:")
    for condition, metrics in avg_condition_metrics.items():
        print(f"  {condition} - PSNR: {metrics['psnr']:.2f}, SSIM: {metrics['ssim']:.4f}")


def load_model_plus(checkpoint_path, config=None, device='cuda'):
    """加载CAM+模型
    
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
    model = CAMPlus(config).to(device)
    
    # 加载模型参数
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 设置为评估模式
    model.eval()
    
    return model


def inference_plus(model, source_images, config):
    """使用CAM+模型进行推理
    
    Args:
        model: CAM+模型
        source_images: 字典，键为条件名，值为对应的图像张量或PIL图像
        config: 配置参数
        
    Returns:
        生成的图像张量
    """
    from torchvision import transforms
    from PIL import Image
    
    # 为RGB和灰度图像定义不同的转换
    transform_rgb = transforms.Compose([
        transforms.Resize((config['img_size'], config['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    transform_gray = transforms.Compose([
        transforms.Resize((config['img_size'], config['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # 确保模型处于评估模式
    model.eval()
    
    # 处理输入图像
    processed_images = {}
    for condition, img in source_images.items():
        if isinstance(img, Image.Image):
            # 根据条件类型选择正确的转换
            if condition == 'color':
                # 颜色图像使用RGB转换
                img_tensor = transform_rgb(img).unsqueeze(0).to(config['device'])
            else:
                # 其他条件（sketch、canny、depth）使用灰度转换
                # 确保图像是单通道
                img_gray = img.convert('L') if img.mode != 'L' else img
                img_tensor = transform_gray(img_gray).unsqueeze(0).to(config['device'])
        else:
            # 如果已经是张量，确保形状正确并移动到正确的设备
            img_tensor = img.to(config['device'])
            if img_tensor.dim() == 3:
                img_tensor = img_tensor.unsqueeze(0)
        
        processed_images[condition] = img_tensor
    
    # 推理
    with torch.no_grad():
        outputs = model(processed_images)
        # 获取第一个条件的输出作为结果
        # 注意：outputs['outputs']是一个字典，包含每个条件的输出
        first_condition = list(outputs['outputs'].keys())[0]
        output_img = outputs['outputs'][first_condition]
    
    return output_img