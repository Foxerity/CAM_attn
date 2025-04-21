import os
import torch
import numpy as np
import json
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
import glob
from scipy import linalg
import argparse
import sys

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import load_model
from enhanced_model import load_enhanced_model
from data_loader import seed_everything


class ImageFolderDataset(Dataset):
    """简单的图像文件夹数据集，用于批量加载图像"""
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.image_paths = []
        
        # 支持的图像格式
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            self.image_paths.extend(glob.glob(os.path.join(folder_path, '**', ext), recursive=True))
        
        self.image_paths.sort()  # 确保顺序一致
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
            
        # 提取类别信息（假设图像路径格式为：/path/to/class_name/image.jpg）
        class_name = os.path.basename(os.path.dirname(img_path))
        
        return {
            'image': img,
            'path': img_path,
            'class_name': class_name
        }


class FIDCalculator:
    """FID (Fréchet Inception Distance) 计算器
    
    FID是评估生成图像质量的常用指标，它测量生成图像与真实图像在特征空间中的距离。
    较低的FID分数表示生成的图像更接近真实图像的分布。
    """
    def __init__(self, device='cuda'):
        self.device = device
        
        # 加载预训练的Inception V3模型
        self.inception_model = models.inception_v3(pretrained=True, transform_input=False)
        # 修改模型以使用pool3层特征（2048维），这是计算FID的标准做法
        # 移除fc层后的所有层
        self.inception_model.fc = torch.nn.Identity()
        # 设置钩子以获取pool3层的输出
        self.pool3_features = None
        self.inception_model.avgpool.register_forward_hook(self._hook_fn)
        self.inception_model.to(device).eval()
        
        # 设置图像转换 - 使用TensorFlow兼容的预处理，与原始FID实现一致
        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),  # Inception V3的输入大小
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 范围调整为[-1, 1]
        ])
        
    def _hook_fn(self, module, input, output):
        """钩子函数，用于获取pool3层的输出"""
        # 获取avgpool输出并展平
        self.pool3_features = output.squeeze(-1).squeeze(-1)
    
    def extract_features(self, dataloader):
        """从数据加载器中提取特征"""
        features = []
        class_features = {}  # 按类别存储特征
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="提取特征"):
                images = batch['image'].to(self.device)
                class_names = batch['class_name']
                
                # 提取特征 - 运行模型会触发钩子函数，获取pool3特征
                _ = self.inception_model(images)
                batch_features = self.pool3_features
                
                # 保存总体特征
                features.append(batch_features.cpu().numpy())
                
                # 按类别保存特征
                for i, class_name in enumerate(class_names):
                    if class_name not in class_features:
                        class_features[class_name] = []
                    class_features[class_name].append(batch_features[i].unsqueeze(0).cpu().numpy())
        
        # 合并特征
        features = np.concatenate(features, axis=0)
        
        # 合并每个类别的特征
        for class_name in class_features:
            class_features[class_name] = np.concatenate(class_features[class_name], axis=0)
        
        return features, class_features
    
    def calculate_statistics(self, features):
        """计算特征的均值和协方差"""
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        return mu, sigma
    
    def calculate_fid(self, mu1, sigma1, mu2, sigma2):
        """计算FID分数"""
        # 计算均值差异的平方
        diff = mu1 - mu2
        mean_diff_squared = np.sum(diff * diff)
        
        # 计算协方差矩阵的平方根乘积
        # 注意：这里使用了Scipy的linalg.sqrtm，它可以计算矩阵的平方根
        covmean = linalg.sqrtm(sigma1.dot(sigma2))
        
        # 检查并纠正复数结果
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        # 计算FID - 除以1000使结果与学术界标准一致
        fid = (mean_diff_squared + np.trace(sigma1 + sigma2 - 2 * covmean)) / 1000.0
        return fid
    
    def compute_fid(self, real_dataloader, fake_dataloader):
        """计算两个数据集之间的FID"""
        # 提取特征
        real_features, real_class_features = self.extract_features(real_dataloader)
        fake_features, fake_class_features = self.extract_features(fake_dataloader)
        
        # 计算总体统计量
        mu_real, sigma_real = self.calculate_statistics(real_features)
        mu_fake, sigma_fake = self.calculate_statistics(fake_features)
        
        # 计算总体FID
        fid = self.calculate_fid(mu_real, sigma_real, mu_fake, sigma_fake)
        
        # 计算每个类别的FID
        class_fids = {}
        for class_name in real_class_features:
            if class_name in fake_class_features:
                mu_real_class, sigma_real_class = self.calculate_statistics(real_class_features[class_name])
                mu_fake_class, sigma_fake_class = self.calculate_statistics(fake_class_features[class_name])
                class_fids[class_name] = self.calculate_fid(mu_real_class, sigma_real_class, mu_fake_class, sigma_fake_class)
        
        # 计算几何平均FID (gFID)
        if class_fids:
            gfid = np.exp(np.mean(np.log([fid for fid in class_fids.values() if fid > 0])))
        else:
            gfid = None
        
        return {
            'fid': fid,
            'class_fids': class_fids,
            'gfid': gfid
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


def evaluate_fid(generated_dir, target_dir, batch_size=32, device='cuda'):
    """评估FID分数
    
    Args:
        generated_dir: 生成图像目录
        target_dir: 目标图像目录
        batch_size: 批量大小
        device: 设备
    
    Returns:
        FID评估结果
    """
    # 设置转换
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 创建数据集和数据加载器
    real_dataset = ImageFolderDataset(target_dir, transform=transform)
    fake_dataset = ImageFolderDataset(generated_dir, transform=transform)
    
    real_dataloader = DataLoader(real_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    fake_dataloader = DataLoader(fake_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # 计算FID
    fid_calculator = FIDCalculator(device=device)
    fid_results = fid_calculator.compute_fid(real_dataloader, fake_dataloader)
    
    return fid_results


def save_results(results, output_dir):
    """保存评估结果
    
    Args:
        results: 评估结果
        output_dir: 输出目录
    """
    # 保存为JSON
    with open(os.path.join(output_dir, 'fid_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    # 打印结果
    print(f"总体FID: {results['fid']:.4f}")
    if results['gfid'] is not None:
        print(f"几何平均FID (gFID): {results['gfid']:.4f}")
    
    print("\n各类别FID:")
    for class_name, fid in sorted(results['class_fids'].items()):
        print(f"{class_name}: {fid:.4f}")
    
    # 绘制条形图
    plt.figure(figsize=(12, 6))
    
    # 按FID值排序
    sorted_classes = sorted(results['class_fids'].items(), key=lambda x: x[1])
    class_names = [c[0] for c in sorted_classes]
    fid_values = [c[1] for c in sorted_classes]
    
    plt.bar(class_names, fid_values)
    plt.axhline(y=results['fid'], color='r', linestyle='-', label=f'total FID: {results["fid"]:.4f}')
    if results['gfid'] is not None:
        plt.axhline(y=results['gfid'], color='g', linestyle='--', label=f'gFID: {results["gfid"]:.4f}')
    
    plt.xlabel('classes')
    plt.ylabel('FID Scores')
    plt.title('Each class FID')
    plt.xticks(rotation=90)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'fid_by_class.png'))
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='FID评估工具')
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
    generated_dir = os.path.join(args.image_dir, 'generated')
    generate_images(model, args.source_dir, generated_dir, config, model_type)
    
    # 复制目标图像
    print("复制目标图像...")
    target_output_dir = copy_target_images(args.target_dir, args.image_dir)
    
    # 评估FID
    print("计算FID...")
    fid_results = evaluate_fid(generated_dir, target_output_dir, args.batch_size, device)
    
    # 保存结果
    save_results(fid_results, args.output_dir)
    
    print(f"评估完成，结果已保存到: {args.output_dir}")


if __name__ == "__main__":
    main()