import os
import argparse
import torch
from convert_to_grayscale import process_directory

def main():
    parser = argparse.ArgumentParser(description='图像处理和模型训练工具')
    parser.add_argument('--data_dir', type=str, default=r"/data/ymx/dataset/imagenet-part", help='数据集根目录')
    parser.add_argument('--convert_only', type=bool, default=True, help='仅执行图像转换，不训练模型')
    parser.add_argument('--conditions', nargs='+', default=["illusion"],
                        help='需要转换的条件类型，默认为sketch、canny和depth')
    args = parser.parse_args()
    
    # 执行图像转换
    print("开始将三通道图像转换为单通道...")
    for condition in args.conditions:
        condition_dir = os.path.join(args.data_dir, f"img_{condition}")
        if os.path.exists(condition_dir):
            print(f"\n处理 {condition} 条件图像...")
            
            # 分别处理训练集和验证集
            for split in ['train', 'val']:
                split_dir = os.path.join(condition_dir, split)
                if os.path.exists(split_dir):
                    print(f"处理 {split} 集...")
                    total, successful = process_directory(split_dir)
                    print(f"{split} 集处理完成: 共 {total} 张图像，成功转换 {successful} 张")
                else:
                    print(f"警告: {split} 目录不存在")
        else:
            print(f"警告: {condition_dir} 目录不存在")
    
    print("\n所有图像转换完成!")
    
    # 如果不是仅转换模式，则训练模型
    if not args.convert_only:
        print("\n准备训练模型...")
        print("请在训练脚本中导入并调用train_model_plus函数，并提供适当的配置")
        print("示例配置:")
        print("""
        config = {
            'dataset_path': '数据集路径',
            'target_condition': 'depth',
            'source_conditions': ['canny', 'sketch', 'color'],
            'img_size': 256,
            'batch_size': 16,
            'num_workers': 4,
            'lr': 0.0002,
            'beta': 0.01,
            'epochs': 100,
            'lr_step': 30,
            'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            'output_dir': './output_plus'
        }
        from model_plus import train_model_plus
        train_model_plus(config)
        """)

if __name__ == "__main__":
    main()