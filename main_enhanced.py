import os
import argparse
import torch
import json
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from model import CAM, train_model, load_model, inference
from enhanced_model import EnhancedCAM, train_enhanced_model, load_enhanced_model, enhanced_inference
from data_loader import seed_everything


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='CAM: Condition Alignment Module')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'],
                        help='运行模式: train或test')
    parser.add_argument('--config', type=str, default='config_enhanced.json',
                        help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, default=r"output_enhanced\checkpoints\best_model.pth",
                        help='测试模式下的检查点路径')
    parser.add_argument('--input', type=str, default=None,
                        help='测试模式下的输入图像路径')
    parser.add_argument('--output', type=str, default='output.png',
                        help='测试模式下的输出图像路径')
    parser.add_argument('--enhanced', action='store_true',
                        help='是否使用增强版模型')
    
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
            'output_dir': './output',  # 输出目录
            'save_interval': 10,  # 保存间隔
            'use_enhanced_model': False,  # 是否使用增强版模型
            'contrastive_weight': 0.1,  # 对比学习损失权重
            'feature_matching_weight': 0.1,  # 特征匹配损失权重
            'temperature': 0.5,  # 对比学习温度参数
            'feature_matching_loss_type': 'l1'  # 特征匹配损失类型
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


def test_model(checkpoint_path, input_path, output_path, config, use_enhanced_model=False):
    """测试模型
    
    Args:
        checkpoint_path: 检查点路径
        input_path: 输入图像路径
        output_path: 输出图像路径
        config: 配置参数
        use_enhanced_model: 是否使用增强版模型
    """
    # 加载模型
    if use_enhanced_model:
        model = load_enhanced_model(checkpoint_path, config, config['device'])
        # 加载输入图像
        input_img = Image.open(input_path).convert('RGB')
        # 推理
        output_tensor = enhanced_inference(model, input_img)
        # 转换为PIL图像
        output_img = tensor_to_pil(output_tensor)
    else:
        model, _ = load_model(checkpoint_path, config['device'])
        # 推理
        output_img = inference(model, input_path, config)
    
    # 保存结果
    output_img.save(output_path)
    print(f"生成的图像已保存到: {output_path}")
    
    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # 显示输入图像
    input_img = Image.open(input_path).convert('RGB')
    input_img = input_img.resize((config['img_size'], config['img_size']))
    axes[0].imshow(np.array(input_img))
    axes[0].set_title('Input Image')
    axes[0].axis('off')
    
    # 显示输出图像
    axes[1].imshow(np.array(output_img))
    axes[1].set_title('Generated Image')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.splitext(output_path)[0] + '_comparison.png')
    plt.show()


def tensor_to_pil(tensor):
    """将张量转换为PIL图像
    
    Args:
        tensor: 输入张量，形状为[1, C, H, W]，范围为[-1, 1]
        
    Returns:
        PIL图像
    """
    # 确保张量在CPU上
    tensor = tensor.cpu().squeeze(0)
    
    # 将范围从[-1, 1]调整为[0, 1]
    tensor = (tensor + 1) / 2
    
    # 将范围从[0, 1]调整为[0, 255]
    tensor = tensor.clamp(0, 1) * 255
    
    # 转换为PIL图像
    tensor = tensor.detach().numpy().astype(np.uint8)
    tensor = np.transpose(tensor, (1, 2, 0))
    
    return Image.fromarray(tensor)


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 检查是否使用增强版模型
    use_enhanced_model = args.enhanced or config.get('use_enhanced_model', False)
    config['use_enhanced_model'] = use_enhanced_model
    
    # 设置随机种子
    seed_everything(config['seed'])
    
    if args.mode == 'train':
        # 训练模式
        print(f"开始训练{'增强版' if use_enhanced_model else ''}模型...")
        if use_enhanced_model:
            train_enhanced_model(config)
        else:
            train_model(config)
    else:
        # 测试模式
        if args.checkpoint is None or args.input is None:
            print("错误: 测试模式需要指定检查点路径和输入图像路径")
            return
        
        print(f"开始测试{'增强版' if use_enhanced_model else ''}模型...")
        test_model(args.checkpoint, args.input, args.output, config, use_enhanced_model)


if __name__ == "__main__":
    main()