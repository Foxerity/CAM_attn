import os
import argparse
import json
import torch
import sys
import subprocess
from datetime import datetime

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import load_model
from enhanced_model import load_enhanced_model
from data_loader import seed_everything


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='CAM模型评估工具')
    parser.add_argument('--checkpoint', type=str, default='output\checkpoints\model_epoch_200.pth',
                        help='模型检查点路径')
    parser.add_argument('--config', type=str, default='config.json',
                        help='配置文件路径')
    parser.add_argument('--source', type=str, default='canny',
                        help='源条件类型，例如：canny, sketch等')
    parser.add_argument('--target_condition', type=str, default=None,
                        help='目标条件类型，如果不指定则使用配置文件中的target_condition')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                        help='输出目录')
    parser.add_argument('--enhanced', action='store_true',
                        help='是否使用增强版模型')
    parser.add_argument('--metrics', type=str, default='all',
                        help='要计算的指标，可选：fid, lpips, all')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批量大小')
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 加载配置
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config['device'] = device
    
    # 设置随机种子
    seed_everything(config['seed'])
    
    # 确定目标条件
    target_condition = args.target_condition if args.target_condition else config['target_condition']
    
    # 构建数据集路径
    source_dir = os.path.join(config['dataset_path'], f"img_{args.source}", 'val')
    target_dir = os.path.join(config['dataset_path'], f"img_{target_condition}", 'val')
    
    # 检查路径是否存在
    if not os.path.exists(source_dir):
        print(f"错误: 源条件目录不存在: {source_dir}")
        return
    
    if not os.path.exists(target_dir):
        print(f"错误: 目标条件目录不存在: {target_dir}")
        return
    
    model_name = "enhanced" if args.enhanced else "standard"
    output_dir = os.path.join(args.output_dir, f"{model_name}_{args.source}_to_{target_condition}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存评估配置
    eval_config = {
        'checkpoint': args.checkpoint,
        'config_file': args.config,
        'source': args.source,
        'target_condition': target_condition,
        'enhanced': args.enhanced,
        'metrics': args.metrics,
    }
    
    with open(os.path.join(output_dir, 'eval_config.json'), 'w') as f:
        json.dump(eval_config, f, indent=4)
    
    # 运行评估
    metrics_to_run = []
    if args.metrics == 'all':
        metrics_to_run = ['Fid', 'Lpips']
    else:
        metrics_to_run = args.metrics.split(',')
    
    results = {}
    
    # 生成图像（只需要生成一次）
    skip_generation = False
    
    # 运行每个指标
    for metric in metrics_to_run:
        print(f"\n{'='*50}")
        print(f"运行 {metric.upper()} 评估...")
        print(f"{'='*50}\n")
        
        # 构建命令
        cmd = [
            sys.executable,
            os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{metric}.py"),
            "--checkpoint", args.checkpoint,
            "--config", args.config,
            "--source_dir", source_dir,
            "--target_dir", target_dir,
            "--output_dir", output_dir,
            "--image_dir", os.path.join(output_dir, "img_results"),
            "--batch_size", str(args.batch_size)
        ]
        
        if args.enhanced:
            cmd.append("--enhanced")
        
        if skip_generation:
            cmd.append("--skip_generation")
        
        # 运行命令
        try:
            subprocess.run(cmd, check=True)
            skip_generation = True  # 第一次运行后，后续指标可以跳过图像生成
            
            # 加载结果
            result_file = os.path.join(output_dir, metric, f"{metric}_results.json")
            if os.path.exists(result_file):
                with open(result_file, 'r') as f:
                    metric_results = json.load(f)
                results[metric] = metric_results
        except subprocess.CalledProcessError as e:
            print(f"运行 {metric} 评估时出错: {e}")
    
    # 汇总所有结果
    with open(os.path.join(output_dir, 'all_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    print("\n评估完成！汇总结果:")
    print(f"{'='*50}")
    
    # 打印汇总结果
    if 'fid' in results:
        print(f"FID: {results['fid']['fid']:.4f}")
        if results['fid']['gfid'] is not None:
            print(f"几何平均FID (gFID): {results['fid']['gfid']:.4f}")
    
    if 'lpips' in results:
        print(f"LPIPS: {results['lpips']['lpips']:.4f}")
        if 'psnr' in results['lpips']:
            print(f"PSNR: {results['lpips']['psnr']:.4f}")
        if 'ssim' in results['lpips']:
            print(f"SSIM: {results['lpips']['ssim']:.4f}")
    
    print(f"\n详细结果已保存到: {output_dir}")


if __name__ == "__main__":
    main()