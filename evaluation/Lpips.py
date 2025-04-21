import os
import torch
import numpy as np
import json
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import glob
import argparse
import sys

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import load_model
from enhanced_model import load_enhanced_model
from data_loader import seed_everything


class LPIPS:
    """LPIPS (Learned Perceptual Image Patch Similarity) 计算器
    
    LPIPS是一种基于深度学习的感知相似度度量，它比传统的PSNR和SSIM更符合人类感知。
    它使用预训练的网络（如VGG、AlexNet或ResNet）来提取特征，然后计算特征之间的距离。
    """
    def __init__(self, net='alex', device='cuda'):
        """初始化LPIPS模型
        
        Args:
            net: 使用的网络，可以是'alex'、'vgg'或'squeeze'
            device: 计算设备
        """
        self.device = device
        self.net = net
        
        # 导入lpips库（如果没有安装，请使用pip install lpips）
        try:
            import lpips
            self.model = lpips.LPIPS(net=net).to(device)
        except ImportError:
            print("请安装lpips库: pip install lpips")
            sys.exit(1)
    
    def calculate(self, img1, img2):
        """计算两个图像之间的LPIPS距离
        
        Args:
            img1, img2: 形状为[B, C, H, W]的张量，范围为[-1, 1]
            
        Returns:
            float: LPIPS距离
        """
        with torch.no_grad():
            dist = self.model(img1, img2)
        return dist


class ImagePairDataset(Dataset):
    """图像对数据集，用于批量计算LPIPS"""
    def __init__(self, generated_dir, target_dir, transform=None):
        self.generated_dir = generated_dir
        self.target_dir = target_dir
        self.transform = transform
        
        # 获取生成图像路径
        self.generated_paths = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            self.generated_paths.extend(glob.glob(os.path.join(generated_dir, '**', ext), recursive=True))
        
        self.generated_paths.sort()
        
        # 构建目标图像路径（保持相同的相对路径结构）
        self.target_paths = []
        self.class_names = []
        
        for gen_path in self.generated_paths:
            rel_path = os.path.relpath(gen_path, generated_dir)
            target_path = os.path.join(target_dir, rel_path)
            
            if os.path.exists(target_path):
                self.target_paths.append(target_path)
                # 提取类别信息（假设图像路径格式为：/path/to/class_name/image.jpg）
                class_name = os.path.basename(os.path.dirname(gen_path))
                self.class_names.append(class_name)
            else:
                print(f"警告: 目标图像不存在: {target_path}")
        
        # 确保数据集非空
        if not self.target_paths:
            raise ValueError("没有找到匹配的图像对")
    
    def __len__(self):
        return len(self.target_paths)
    
    def __getitem__(self, idx):
        gen_path = self.generated_paths[idx]
        target_path = self.target_paths[idx]
        class_name = self.class_names[idx]
        
        gen_img = Image.open(gen_path).convert('RGB')
        target_img = Image.open(target_path).convert('RGB')
        
        if self.transform:
            gen_img = self.transform(gen_img)
            target_img = self.transform(target_img)
        
        return {
            'generated': gen_img,
            'target': target_img,
            'class_name': class_name,
            'gen_path': gen_path,
            'target_path': target_path
        }


def generate_images(model, source_dir, output_dir, config, model_type='standard'):
    """使用模型生成图像
    
    Args:
        model: CAM模型或增强版CAM模型
        source_dir: 源条件图像目录
        output_dir: 输出目录
        config: 配置参数
        model_type: 模型类型，'standard'或'enhanced'
    
    Returns:
        生成的图像路径列表
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 图像转换
    transform = transforms.Compose([
        transforms.Resize((config['img_size'], config['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # 获取所有图像文件
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        image_files.extend(glob.glob(os.path.join(source_dir, '**', ext), recursive=True))
    
    image_files.sort()  # 确保顺序一致
    
    # 生成图像
    model.eval()
    with torch.no_grad():
        for img_path in tqdm(image_files, desc="生成图像"):
            # 获取相对路径，用于保持目录结构
            rel_path = os.path.relpath(img_path, source_dir)
            output_path = os.path.join(output_dir, rel_path)
            
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 加载源图像
            source_img = Image.open(img_path).convert('RGB')
            source_tensor = transform(source_img).unsqueeze(0).to(config['device'])
            
            # 生成图像
            if model_type == 'enhanced':
                outputs = model(source_tensor)
                output = outputs['output']
            else:
                output, _, _ = model(source_tensor)
            
            # 保存生成的图像
            output_img = tensor_to_image(output.squeeze(0).cpu())
            output_img.save(output_path)
    
    return output_dir


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


def copy_target_images(target_dir, output_dir):
    """复制目标图像到输出目录，保持相同的目录结构
    
    Args:
        target_dir: 目标图像目录
        output_dir: 输出目录
    
    Returns:
        复制的目标图像目录
    """
    target_output_dir = os.path.join(output_dir, 'target')
    os.makedirs(target_output_dir, exist_ok=True)
    
    # 获取所有图像文件
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        image_files.extend(glob.glob(os.path.join(target_dir, '**', ext), recursive=True))
    
    # 复制图像
    for img_path in tqdm(image_files, desc="复制目标图像"):
        # 获取相对路径，用于保持目录结构
        rel_path = os.path.relpath(img_path, target_dir)
        output_path = os.path.join(target_output_dir, rel_path)
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 复制图像
        Image.open(img_path).save(output_path)
    
    return target_output_dir


def evaluate_lpips(generated_dir, target_dir, batch_size=32, device='cuda'):
    """评估LPIPS分数
    
    Args:
        generated_dir: 生成图像目录
        target_dir: 目标图像目录
        batch_size: 批量大小
        device: 设备
    
    Returns:
        LPIPS评估结果
    """
    # 设置转换
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        # LPIPS期望输入范围为[-1, 1]
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # 创建数据集和数据加载器
    dataset = ImagePairDataset(generated_dir, target_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # 初始化LPIPS模型
    lpips_model = LPIPS(device=device)
    
    # 计算LPIPS
    lpips_values = []
    class_lpips = {}
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="计算LPIPS"):
            gen_imgs = batch['generated'].to(device)
            target_imgs = batch['target'].to(device)
            class_names = batch['class_name']
            
            # 计算LPIPS
            batch_lpips = lpips_model.calculate(gen_imgs, target_imgs)
            
            # 保存结果
            for i, (lpips_val, class_name) in enumerate(zip(batch_lpips, class_names)):
                lpips_values.append(lpips_val.item())
                
                if class_name not in class_lpips:
                    class_lpips[class_name] = []
                class_lpips[class_name].append(lpips_val.item())
    
    # 计算平均LPIPS
    avg_lpips = np.mean(lpips_values)
    
    # 计算每个类别的平均LPIPS
    class_avg_lpips = {}
    for class_name, values in class_lpips.items():
        class_avg_lpips[class_name] = np.mean(values)
    
    # 计算几何平均LPIPS (gLPIPS)
    if class_avg_lpips:
        glpips = np.exp(np.mean(np.log([val for val in class_avg_lpips.values() if val > 0])))
    else:
        glpips = None
    
    return {
        'lpips': avg_lpips,
        'class_lpips': class_avg_lpips,
        'glpips': glpips
    }


def save_results(results, output_dir):
    """保存评估结果
    
    Args:
        results: 评估结果
        output_dir: 输出目录
    """
    # 保存为JSON
    with open(os.path.join(output_dir, 'lpips_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    # 打印结果
    print(f"总体LPIPS: {results['lpips']:.4f}")
    if results['glpips'] is not None:
        print(f"几何平均LPIPS (gLPIPS): {results['glpips']:.4f}")
    
    print("\n各类别LPIPS:")
    for class_name, lpips in sorted(results['class_lpips'].items()):
        print(f"{class_name}: {lpips:.4f}")
    
    # 绘制条形图
    plt.figure(figsize=(12, 6))
    
    # 按LPIPS值排序
    sorted_classes = sorted(results['class_lpips'].items(), key=lambda x: x[1])
    class_names = [c[0] for c in sorted_classes]
    lpips_values = [c[1] for c in sorted_classes]
    
    plt.bar(class_names, lpips_values)
    plt.axhline(y=results['lpips'], color='r', linestyle='-', label=f'total LPIPS: {results["lpips"]:.4f}')
    if results['glpips'] is not None:
        plt.axhline(y=results['glpips'], color='g', linestyle='--', label=f'gLPIPS: {results["glpips"]:.4f}')
    
    plt.xlabel('Classes')
    plt.ylabel('LPIPS Scores')
    plt.title('Each Class LPIPS')
    plt.xticks(rotation=90)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'lpips_by_class.png'))
    plt.close()


def compute_psnr_ssim(generated_dir, target_dir, batch_size=32, device='cuda'):
    """计算PSNR和SSIM
    
    Args:
        generated_dir: 生成图像目录
        target_dir: 目标图像目录
        batch_size: 批量大小
        device: 设备
    
    Returns:
        PSNR和SSIM评估结果
    """
    # 设置转换
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    # 创建数据集和数据加载器
    dataset = ImagePairDataset(generated_dir, target_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # 计算PSNR和SSIM
    psnr_values = []
    ssim_values = []
    class_psnr = {}
    class_ssim = {}
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="计算PSNR和SSIM"):
            gen_imgs = batch['generated'].to(device)
            target_imgs = batch['target'].to(device)
            class_names = batch['class_name']
            
            # 计算PSNR
            mse = F.mse_loss(gen_imgs, target_imgs, reduction='none').mean(dim=[1, 2, 3])
            batch_psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
            
            # 计算SSIM
            batch_ssim = []
            for i in range(gen_imgs.size(0)):
                ssim_val = compute_ssim(gen_imgs[i].unsqueeze(0), target_imgs[i].unsqueeze(0))
                batch_ssim.append(ssim_val)
            batch_ssim = torch.tensor(batch_ssim, device=device)
            
            # 保存结果
            for i, (psnr_val, ssim_val, class_name) in enumerate(zip(batch_psnr, batch_ssim, class_names)):
                psnr_values.append(psnr_val.item())
                ssim_values.append(ssim_val.item())
                
                if class_name not in class_psnr:
                    class_psnr[class_name] = []
                    class_ssim[class_name] = []
                class_psnr[class_name].append(psnr_val.item())
                class_ssim[class_name].append(ssim_val.item())
    
    # 计算平均值
    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)
    
    # 计算每个类别的平均值
    class_avg_psnr = {}
    class_avg_ssim = {}
    for class_name in class_psnr.keys():
        class_avg_psnr[class_name] = np.mean(class_psnr[class_name])
        class_avg_ssim[class_name] = np.mean(class_ssim[class_name])
    
    return {
        'psnr': avg_psnr,
        'ssim': avg_ssim,
        'class_psnr': class_avg_psnr,
        'class_ssim': class_avg_ssim
    }


def compute_ssim(img1, img2):
    """计算结构相似性 (SSIM)
    
    Args:
        img1, img2: 形状为[B, C, H, W]的张量，范围为[0, 1]
        
    Returns:
        float: SSIM值
    """
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


def main():
    parser = argparse.ArgumentParser(description='LPIPS评估工具')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='模型检查点路径')
    parser.add_argument('--config', type=str, default='config.json',
                        help='配置文件路径')
    parser.add_argument('--source_dir', type=str, required=True,
                        help='源条件图像目录')
    parser.add_argument('--target_dir', type=str, required=True,
                        help='目标条件图像目录')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                        help='输出目录')
    parser.add_argument('--image_dir', type=str, default='./evaluation_results/img_results',
                        help='输出目录')
    parser.add_argument('--enhanced', action='store_true',
                        help='是否使用增强版模型')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批量大小')
    parser.add_argument('--skip_generation', action='store_true',
                        help='跳过图像生成步骤（如果已经生成）')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载配置
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config['device'] = device
    
    # 设置随机种子
    seed_everything(config['seed'])
    
    generated_dir = os.path.join(args.image_dir, 'generated')
    target_output_dir = os.path.join(args.image_dir, 'target')
    
    if not args.skip_generation:
        # 加载模型
        print(f"加载{'增强版' if args.enhanced else '标准'}模型...")
        if args.enhanced:
            model = load_enhanced_model(args.checkpoint, config, device)
            model_type = 'enhanced'
        else:
            model, _ = load_model(args.checkpoint, device)
            model_type = 'standard'
        
        # 生成图像
        print("生成图像...")
        generate_images(model, args.source_dir, generated_dir, config, model_type)
        
        # 复制目标图像
        print("复制目标图像...")
        copy_target_images(args.target_dir, args.image_dir)
    else:
        print("跳过图像生成步骤...")
        if not os.path.exists(generated_dir) or not os.path.exists(target_output_dir):
            print("错误: 生成的图像目录或目标图像目录不存在，请先生成图像")
            return
    
    # 评估LPIPS
    print("计算LPIPS...")
    lpips_results = evaluate_lpips(generated_dir, target_output_dir, args.batch_size, device)
    
    # 计算PSNR和SSIM
    print("计算PSNR和SSIM...")
    psnr_ssim_results = compute_psnr_ssim(generated_dir, target_output_dir, args.batch_size, device)
    
    # 合并结果
    results = {**lpips_results, **psnr_ssim_results}
    
    # 保存结果
    save_results(results, args.output_dir)
    
    print(f"评估完成，结果已保存到: {args.output_dir}")


if __name__ == "__main__":
    main()