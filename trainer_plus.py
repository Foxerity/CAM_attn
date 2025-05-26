import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import time
import numpy as np
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms
from PIL import Image

from model_plus import CAMPlus
from data_loader_plus import get_multi_condition_loaders
from losses import ReconstructionLoss, ContrastiveLoss
from utils import save_image_grid, compute_psnr, compute_ssim


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
    
    # 创建输出目录
    output_dir = config.get('output_dir', './output_plus')
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取数据加载器
    train_loader, val_loader = get_multi_condition_loaders(config)
    
    # 创建模型
    model = CAMPlus(config).to(config['device'])
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # 定义损失函数
    recon_loss_fn = ReconstructionLoss(loss_type=config.get('recon_loss_type', 'l1'))
    feature_matching_loss_fn = None
    contrastive_loss_fn = ContrastiveLoss(temperature=config.get('temperature', 0.5))
    
    # 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    scaler = torch.amp.GradScaler("cuda")
    
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
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']} [Train]", leave=True, position=0)
        for batch in train_pbar:
            # 获取数据
            source_images = batch['source_images']  # 字典，键为条件名，值为对应的图像张量
            target_img = batch['target_img'].to(config['device'])
            
            # 将所有源图像移动到设备
            for condition in source_images:
                source_images[condition] = source_images[condition].to(config['device'])
            
            # 前向传播
            with torch.amp.autocast('cuda'):
                outputs = model(source_images, target_img)

            output_imgs = outputs['outputs']  # 字典，键为条件名，值为对应的生成图像
            mus = outputs['mus']
            logvars = outputs['logvars']
            encoder_features = outputs['encoder_features']
            target_features = outputs['target_features']
            
            # 计算每个条件的损失
            total_recon_loss = 0
            total_kl_loss = 0
            total_feature_matching_loss = 0
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
                    total_feature_matching_loss += feature_matching_loss * config.get(f'{condition}_weight', 1.0)
                
                # 条件权重（可以为不同条件设置不同权重）
                # condition_weight = config.get(f'{condition}_weight', 1.0)
                feature_matching_weight = config.get('feature_matching_weight', 0.1)
                kl_weight = config.get('beta', 0.01)
                
                # 计算当前条件的总损失
                condition_loss = (
                    recon_loss + 
                    kl_weight * kl_loss + 
                    feature_matching_weight * feature_matching_loss
                )
                
                condition_losses[condition] = condition_loss
                total_recon_loss += recon_loss
                total_kl_loss += kl_loss
                
                # 记录每个条件的损失
                train_condition_losses[condition].append(condition_loss.item())
            
            total_loss = sum(condition_losses.values())
            
            # 反向传播和优化
            optimizer.zero_grad()
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # 记录总损失
            train_losses.append(total_loss.item())
            
            # 更新进度条显示的损失值
            train_pbar.set_postfix({
                'Loss': f'{total_loss.item():.4f}',
                'Recon': f'{total_recon_loss.item():.4f}',
                'KL': f'{total_kl_loss.item():.4f}',
                # 'Contrast': f'{contrastive_loss.item():.4f}',
                'FeatMatch': f'{total_feature_matching_loss.item():.4f}'
            })
        
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
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config['epochs']} [Val]", leave=True, position=0)
            for batch in val_pbar:
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
                total_recon_loss = 0
                total_kl_loss = 0
                batch_psnr = 0
                batch_ssim = 0
                
                for condition, output_img in output_imgs.items():
                    # 计算损失
                    recon_loss = recon_loss_fn(output_img, target_img)
                    kl_loss = model.kl_divergence_loss(mus[condition], logvars[condition])
                    
                    condition_weight = config.get(f'{condition}_weight', 1.0)
                    kl_weight = config.get('beta', 0.01)
                    
                    condition_loss = (recon_loss + kl_weight * kl_loss) * condition_weight
                    batch_loss += condition_loss
                    
                    # 累计重建损失和KL损失
                    total_recon_loss += recon_loss * condition_weight
                    total_kl_loss += kl_loss * condition_weight
                    
                    # 计算评估指标
                    psnr = compute_psnr(output_img, target_img)
                    ssim = compute_ssim(output_img, target_img)
                    val_condition_metrics[condition]['psnr'].append(psnr)
                    val_condition_metrics[condition]['ssim'].append(ssim)
                    
                    # 累计PSNR和SSIM
                    batch_psnr += psnr / len(output_imgs)
                    batch_ssim += ssim / len(output_imgs)
                
                val_losses.append(batch_loss.item())
                
                # 更新验证进度条显示的损失值和评估指标
                val_pbar.set_postfix({
                    'Loss': f'{batch_loss.item():.4f}',
                    'Recon': f'{total_recon_loss.item():.4f}',
                    'KL': f'{total_kl_loss.item():.4f}',
                    'PSNR': f'{batch_psnr:.2f}',
                    'SSIM': f'{batch_ssim:.4f}'
                })
        
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