import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
from tqdm import tqdm
import json
import random
from model import load_model


def seed_everything(seed):
    """设置随机种子以确保实验可重复性"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def tensor_to_image(tensor):
    """将张量转换为PIL图像
    
    Args:
        tensor: 形状为[C, H, W]的张量，范围为[-1, 1]
        
    Returns:
        PIL.Image: 转换后的图像
    """
    # 转换为numpy数组并调整范围到[0, 1]
    img = tensor.permute(1, 2, 0).numpy() * 0.5 + 0.5
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(img)


def visualize_alignment(model, source_paths, target_path, output_dir, config):
    """可视化不同源条件到目标条件的对齐效果
    
    Args:
        model: CAM模型
        source_paths: 源条件图像路径字典，格式为{条件名: 路径}
        target_path: 目标条件图像路径
        output_dir: 输出目录
        config: 配置参数
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 图像转换
    transform = transforms.Compose([
        transforms.Resize((config['img_size'], config['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # 加载目标图像
    target_img = Image.open(target_path).convert('RGB')
    target_tensor = transform(target_img).unsqueeze(0).to(config['device'])
    
    # 创建图像网格
    n_sources = len(source_paths)
    fig, axes = plt.subplots(n_sources + 1, 3, figsize=(12, 4 * (n_sources + 1)))
    
    # 显示目标图像
    axes[0, 0].imshow(np.array(target_img))
    axes[0, 0].set_title('Target Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(np.array(target_img))
    axes[0, 1].set_title('Target Condition')
    axes[0, 1].axis('off')
    
    axes[0, 2].axis('off')
    
    # 处理每个源条件
    model.eval()
    with torch.no_grad():
        for i, (condition_name, source_path) in enumerate(source_paths.items(), 1):
            # 加载源图像
            source_img = Image.open(source_path).convert('RGB')
            source_tensor = transform(source_img).unsqueeze(0).to(config['device'])
            
            # 生成对齐图像
            output, mu, logvar = model(source_tensor)
            
            # 计算PSNR和SSIM
            psnr = compute_psnr(output, target_tensor)
            ssim = compute_ssim(output, target_tensor)
            
            # 转换为PIL图像
            output_img = tensor_to_image(output.squeeze(0).cpu())
            
            # 显示图像
            axes[i, 0].imshow(np.array(source_img))
            axes[i, 0].set_title(f'Source: {condition_name}')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(np.array(output_img))
            axes[i, 1].set_title(f'Generated (PSNR: {psnr:.2f}, SSIM: {ssim:.4f})')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(np.array(target_img))
            axes[i, 2].set_title('Target')
            axes[i, 2].axis('off')
            
            # 保存单独的对齐结果
            output_img.save(os.path.join(output_dir, f'{condition_name}_aligned.png'))
    
    # 保存完整比较图
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'alignment_comparison.png'))
    plt.close()


def compute_psnr(img1, img2):
    """计算峰值信噪比 (PSNR)
    
    Args:
        img1, img2: 形状为[B, C, H, W]的张量
        
    Returns:
        float: PSNR值
    """
    mse = F.mse_loss(img1, img2).item()
    if mse == 0:
        return float('inf')
    max_pixel = 2.0  # 范围为[-1, 1]
    psnr = 20 * np.log10(max_pixel) - 10 * np.log10(mse)
    return psnr


def compute_ssim(img1, img2):
    """计算结构相似性 (SSIM)
    
    简化版SSIM计算，仅用于评估目的
    
    Args:
        img1, img2: 形状为[B, C, H, W]的张量
        
    Returns:
        float: SSIM值
    """
    # 转换为范围[0, 1]
    img1 = (img1 + 1) / 2
    img2 = (img2 + 1) / 2
    
    # 常数
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    # 计算均值
    mu1 = F.avg_pool2d(img1, kernel_size=11, stride=1, padding=5)
    mu2 = F.avg_pool2d(img2, kernel_size=11, stride=1, padding=5)
    
    # 计算方差和协方差
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.avg_pool2d(img1 * img1, kernel_size=11, stride=1, padding=5) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2 * img2, kernel_size=11, stride=1, padding=5) - mu2_sq
    sigma12 = F.avg_pool2d(img1 * img2, kernel_size=11, stride=1, padding=5) - mu1_mu2
    
    # 计算SSIM
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean().item()


def save_image_grid(images, output_path, nrow=4, padding=2, normalize=True, value_range=(-1, 1)):
    """将多张图像保存为网格布局
    
    Args:
        images: 形状为[B, C, H, W]的张量，或者包含(标签,张量)元组的列表
        output_path: 输出文件路径
        nrow: 每行图像数量
        padding: 图像间的填充像素数
        normalize: 是否归一化图像
        value_range: 输入图像的值范围
    """
    import torch
    from torchvision.utils import save_image
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 检查输入类型，处理包含元组的列表
    if isinstance(images, list) and len(images) > 0 and isinstance(images[0], tuple):
        # 创建一个新的图像列表，每个批次中只取第一个样本
        # 这样可以确保所有图像具有相同的维度 [C, H, W]
        tensor_images = []
        sample_idx = 0  # 选择批次中的第一个样本
        
        for label, img_tensor in images:
            try:
                # 如果是批量张量 [B, C, H, W]，只取第一个样本
                if img_tensor.dim() == 4:
                    img = img_tensor[sample_idx].cpu()
                else:  # 如果已经是 [C, H, W]
                    img = img_tensor.cpu()
                
                # 统一通道数：将单通道图像转换为三通道图像
                if img.dim() == 3 and img.size(0) == 1:  # 单通道图像 [1, H, W]
                    # 复制单通道到三个通道
                    img = img.repeat(3, 1, 1)  # 变为 [3, H, W]
                
                tensor_images.append(img)
            except Exception as e:
                print(f"处理图像 '{label}' 时出错: {e}")
                print(f"图像形状: {img_tensor.shape}")
        
        # 保存图像网格
        try:
            save_image(tensor_images, output_path, nrow=nrow, padding=padding, normalize=normalize, value_range=value_range)
            print(f"成功保存图像网格到: {output_path}")
        except RuntimeError as e:
            print(f"保存图像网格时出错: {e}")
            print("尝试单独保存每个图像...")
            
            # 创建输出目录
            base_dir = os.path.dirname(output_path)
            base_name = os.path.basename(output_path).split('.')[0]
            single_img_dir = os.path.join(base_dir, f"{base_name}_single")
            os.makedirs(single_img_dir, exist_ok=True)
            
            # 单独保存每个图像
            for i, (label, img_tensor) in enumerate(images):
                try:
                    if img_tensor.dim() == 4:  # [B, C, H, W]
                        # 只保存第一个样本
                        img = img_tensor[sample_idx].cpu()
                    else:  # [C, H, W]
                        img = img_tensor.cpu()
                    
                    # 保存图像，单通道图像会自动转换为灰度图
                    save_image(img, 
                              os.path.join(single_img_dir, f"{label}.png"), 
                              normalize=normalize, 
                              value_range=value_range)
                    print(f"已保存单独图像: {label}")
                except Exception as e:
                    print(f"保存图像 '{label}' 时出错: {e}")
    else:
        # 直接保存图像网格
        try:
            # 如果是批量张量列表，确保所有张量都是 [C, H, W] 格式
            if isinstance(images, list):
                processed_images = []
                for i, img in enumerate(images):
                    if img.dim() == 4:  # [B, C, H, W]
                        # 只取第一个样本
                        img = img[0].cpu()
                    else:  # [C, H, W]
                        img = img.cpu()
                    
                    # 统一通道数：将单通道图像转换为三通道图像
                    if img.dim() == 3 and img.size(0) == 1:  # 单通道图像 [1, H, W]
                        # 复制单通道到三个通道
                        img = img.repeat(3, 1, 1)  # 变为 [3, H, W]
                    
                    processed_images.append(img)
                save_image(processed_images, output_path, nrow=nrow, padding=padding, normalize=normalize, value_range=value_range)
            else:
                # 如果是单个批量张量，也只取第一个样本
                if images.dim() == 4 and images.size(0) > 1:
                    save_image(images[0].cpu(), output_path, normalize=normalize, value_range=value_range)
                else:
                    save_image(images.cpu(), output_path, nrow=nrow, padding=padding, normalize=normalize, value_range=value_range)
            print(f"成功保存图像到: {output_path}")
        except RuntimeError as e:
            print(f"保存图像时出错: {e}")
            print("请检查所有图像的形状是否一致")
            # 打印图像形状以便调试
            if isinstance(images, list):
                for i, img in enumerate(images):
                    print(f"图像 {i} 形状: {img.shape}")
            else:
                print(f"图像形状: {images.shape}")



def batch_process(model, source_dir, target_dir, output_dir, config):
    """批量处理图像
    
    Args:
        model: CAM模型
        source_dir: 源条件图像目录
        target_dir: 目标条件图像目录
        output_dir: 输出目录
        config: 配置参数
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 图像转换
    transform = transforms.Compose([
        transforms.Resize((config['img_size'], config['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # 获取所有图像文件
    image_files = [f for f in os.listdir(source_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    # 评估指标
    metrics = {'psnr': [], 'ssim': []}
    
    model.eval()
    with torch.no_grad():
        for img_file in tqdm(image_files, desc="Processing images"):
            # 加载源图像和目标图像
            source_path = os.path.join(source_dir, img_file)
            target_path = os.path.join(target_dir, img_file)
            
            if not os.path.exists(target_path):
                print(f"Warning: Target image {target_path} not found, skipping...")
                continue
            
            source_img = Image.open(source_path).convert('RGB')
            target_img = Image.open(target_path).convert('RGB')
            
            source_tensor = transform(source_img).unsqueeze(0).to(config['device'])
            target_tensor = transform(target_img).unsqueeze(0).to(config['device'])
            
            # 生成对齐图像
            output, _, _ = model(source_tensor)
            
            # 计算评估指标
            psnr = compute_psnr(output, target_tensor)
            ssim = compute_ssim(output, target_tensor)
            
            metrics['psnr'].append(psnr)
            metrics['ssim'].append(ssim)
            
            # 保存生成的图像
            output_img = tensor_to_image(output.squeeze(0).cpu())
            output_img.save(os.path.join(output_dir, img_file))
    
    # 计算平均指标
    avg_metrics = {
        'avg_psnr': np.mean(metrics['psnr']),
        'avg_ssim': np.mean(metrics['ssim']),
        'std_psnr': np.std(metrics['psnr']),
        'std_ssim': np.std(metrics['ssim']),
    }
    
    # 保存评估指标
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(avg_metrics, f, indent=4)
    
    print(f"Average PSNR: {avg_metrics['avg_psnr']:.2f} ± {avg_metrics['std_psnr']:.2f}")
    print(f"Average SSIM: {avg_metrics['avg_ssim']:.4f} ± {avg_metrics['std_ssim']:.4f}")
    
    return avg_metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='CAM Utilities')
    parser.add_argument('--mode', type=str, choices=['visualize', 'batch'], required=True,
                        help='运行模式: visualize或batch')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='模型检查点路径')
    parser.add_argument('--config', type=str, default='config.json',
                        help='配置文件路径')
    parser.add_argument('--source', type=str, required=True,
                        help='源条件图像路径或目录')
    parser.add_argument('--target', type=str, required=True,
                        help='目标条件图像路径或目录')
    parser.add_argument('--output', type=str, default='./results',
                        help='输出目录')
    
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    config['device'] = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    
    # 加载模型
    model, _ = load_model(args.checkpoint, config['device'])
    
    if args.mode == 'visualize':
        # 解析源条件路径
        source_paths = {}
        for item in args.source.split(','):
            condition, path = item.split('=')
            source_paths[condition] = path
        
        visualize_alignment(model, source_paths, args.target, args.output, config)
        print(f"可视化结果已保存到: {args.output}")
    
    elif args.mode == 'batch':
        metrics = batch_process(model, args.source, args.target, args.output, config)
        print(f"批处理结果已保存到: {args.output}")