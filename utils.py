import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
from tqdm import tqdm
import json

from model import load_model


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