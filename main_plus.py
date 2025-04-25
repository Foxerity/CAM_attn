import os
import argparse
import torch
import json
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from model_plus import CAMPlus, train_model_plus, load_model_plus, inference_plus
from data_loader_plus import get_multi_condition_loaders
from utils import seed_everything, tensor_to_image


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='CAM+: 多条件对齐模块')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'],
                        help='运行模式: train或test')
    parser.add_argument('--config', type=str, default='config_plus.json',
                        help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='测试模式下的检查点路径')
    parser.add_argument('--input_dir', type=str, default=None,
                        help='测试模式下的输入图像目录，包含不同条件的子目录')
    parser.add_argument('--output', type=str, default='output.png',
                        help='测试模式下的输出图像路径')
    
    return parser.parse_args()


def load_config(config_path):
    """加载配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置字典
    """
    if not os.path.exists(config_path):
        # 默认配置
        config = {
            'dataset_path': r"B:\datasets\test",  # 数据集路径
            'target_condition': 'depth',  # 目标条件
            'source_conditions': ['canny', 'sketch', 'color'],  # 源条件
            'img_size': 256,  # 图像大小
            'batch_size': 16,  # 批量大小
            'num_workers': 4,  # 数据加载线程数
            'epochs': 100,  # 训练轮数
            'lr': 2e-4,  # 学习率
            'lr_step': 20,  # 学习率衰减步长
            'beta': 0.01,  # 信息瓶颈权衡参数
            'seed': 42,  # 随机种子
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',  # 设备
            'output_dir': './output_plus',  # 输出目录
            'save_interval': 10,  # 保存间隔
            'sample_interval': 5,  # 样本保存间隔
            'contrastive_weight': 0.1,  # 对比学习损失权重
            'feature_matching_weight': 0.1,  # 特征匹配损失权重
            'temperature': 0.5,  # 对比学习温度参数
            'feature_matching_loss_type': 'l1',  # 特征匹配损失类型
            'recon_loss_type': 'l1',  # 重建损失类型
            'input_channels': 3,  # 输入通道数
            'output_channels': 3,  # 输出通道数
            'base_channels': 64,  # 基础通道数
            'depth': 4,  # UNet深度
            'attention_type': 'cbam'  # 注意力类型
        }
        
        # 保存默认配置
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        
        print(f"创建默认配置文件: {config_path}")
    else:
        # 加载配置
        with open(config_path, 'r') as f:
            config = json.load(f)
    
    # 确保设备正确设置
    config['device'] = torch.device(config['device'])
    
    return config


def test_model(checkpoint_path, input_dir, output_path, config):
    """测试模型
    
    Args:
        checkpoint_path: 检查点路径
        input_dir: 输入图像目录，包含不同条件的子目录
        output_path: 输出图像路径
        config: 配置参数
    """
    # 加载模型
    model = load_model_plus(checkpoint_path, config, config['device'])
    
    # 加载输入图像
    source_images = {}
    for condition in config['source_conditions']:
        condition_dir = os.path.join(input_dir, f"img_{condition}")
        if not os.path.exists(condition_dir):
            print(f"警告: 条件 {condition} 的目录不存在: {condition_dir}")
            continue
        
        # 获取目录中的第一个图像
        image_files = [f for f in os.listdir(condition_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        if not image_files:
            print(f"警告: 条件 {condition} 的目录中没有图像: {condition_dir}")
            continue
        
        # 加载图像 - 根据条件类型选择正确的转换模式
        image_path = os.path.join(condition_dir, image_files[0])
        if condition == 'color':
            # 颜色条件使用RGB模式（3通道）
            source_images[condition] = Image.open(image_path).convert('RGB')
            print(f"加载 {condition} 条件图像为RGB模式（3通道）")
        else:
            # 其他条件（sketch、canny、depth）使用L模式（单通道）
            source_images[condition] = Image.open(image_path).convert('L')
            print(f"加载 {condition} 条件图像为L模式（单通道）")
    
    if not source_images:
        print("错误: 没有找到任何输入图像")
        return
    
    # 推理
    output_tensor = inference_plus(model, source_images, config)
    
    # 转换为PIL图像
    output_img = tensor_to_image(output_tensor.squeeze(0).cpu())
    
    # 保存结果
    output_img.save(output_path)
    print(f"生成的图像已保存到: {output_path}")
    
    # 可视化
    fig, axes = plt.subplots(1, len(source_images) + 1, figsize=(5 * (len(source_images) + 1), 5))
    
    # 显示输入图像
    for i, (condition, img) in enumerate(source_images.items()):
        img = img.resize((config['img_size'], config['img_size']))
        axes[i].imshow(np.array(img))
        axes[i].set_title(f'Input ({condition})')
        axes[i].axis('off')
    
    # 显示输出图像
    axes[-1].imshow(np.array(output_img))
    axes[-1].set_title('Generated Image')
    axes[-1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.splitext(output_path)[0] + '_comparison.png')
    plt.show()


def batch_process(checkpoint_path, source_dir, target_dir, output_dir, config):
    """批量处理图像
    
    Args:
        checkpoint_path: 检查点路径
        source_dir: 源条件图像目录，包含不同条件的子目录
        target_dir: 目标条件图像目录
        output_dir: 输出目录
        config: 配置参数
    """
    from tqdm import tqdm
    from utils import compute_psnr, compute_ssim
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载模型
    model = load_model_plus(checkpoint_path, config, config['device'])
    
    # 图像转换 - 为RGB和灰度图像定义不同的转换
    from torchvision import transforms
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
    
    # 获取目标目录中的所有图像文件
    target_files = [f for f in os.listdir(target_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    # 评估指标
    metrics = {'psnr': [], 'ssim': []}
    
    model.eval()
    with torch.no_grad():
        for img_file in tqdm(target_files, desc="处理图像"):
            # 检查所有条件下是否都有这个图像
            source_images = {}
            all_conditions_exist = True
            
            for condition in config['source_conditions']:
                condition_path = os.path.join(source_dir, f"img_{condition}", img_file)
                if not os.path.exists(condition_path):
                    all_conditions_exist = False
                    break
                
                # 加载源条件图像 - 根据条件类型选择正确的转换
                if condition == 'color':
                    # 颜色条件使用RGB模式（3通道）
                    source_img = Image.open(condition_path).convert('RGB')
                    source_tensor = transform_rgb(source_img).unsqueeze(0).to(config['device'])
                else:
                    # 其他条件（sketch、canny、depth）使用L模式（单通道）
                    source_img = Image.open(condition_path).convert('L')
                    source_tensor = transform_gray(source_img).unsqueeze(0).to(config['device'])
                source_images[condition] = source_tensor
            
            if not all_conditions_exist:
                print(f"警告: 图像 {img_file} 在某些条件下不存在，跳过...")
                continue
            
            # 加载目标图像（深度图为单通道）
            target_path = os.path.join(target_dir, img_file)
            target_img = Image.open(target_path).convert('L')  # 转换为单通道
            target_tensor = transform_gray(target_img).unsqueeze(0).to(config['device'])
            
            # 生成对齐图像
            outputs = model(source_images)
            output_tensor = outputs['output']
            
            # 计算评估指标
            psnr = compute_psnr(output_tensor, target_tensor)
            ssim = compute_ssim(output_tensor, target_tensor)
            
            metrics['psnr'].append(psnr)
            metrics['ssim'].append(ssim)
            
            # 保存生成的图像
            output_img = tensor_to_image(output_tensor.squeeze(0).cpu())
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
    
    print(f"平均PSNR: {avg_metrics['avg_psnr']:.2f} ± {avg_metrics['std_psnr']:.2f}")
    print(f"平均SSIM: {avg_metrics['avg_ssim']:.4f} ± {avg_metrics['std_ssim']:.4f}")
    
    return avg_metrics


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 设置随机种子
    seed_everything(config['seed'])
    
    if args.mode == 'train':
        # 训练模式
        print("开始训练CAM+模型...")
        train_model_plus(config)
    else:
        # 测试模式
        if args.checkpoint is None or args.input_dir is None:
            print("错误: 测试模式需要指定检查点路径和输入图像目录")
            return
        
        print("开始测试CAM+模型...")
        test_model(args.checkpoint, args.input_dir, args.output, config)


if __name__ == "__main__":
    main()