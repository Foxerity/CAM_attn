import os
import argparse
import json
import torch
import subprocess
import platform
import sys


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='CAM+ 分布式训练启动器')
    parser.add_argument('--config', type=str, default='config_plus.json',
                        help='配置文件路径')
    parser.add_argument('--num_nodes', type=int, default=1,
                        help='节点数量')
    parser.add_argument('--gpus_per_node', type=int, default=torch.cuda.device_count(),
                        help='每个节点的GPU数量')
    parser.add_argument('--device_ids', type=str, default=None,
                        help='指定要使用的GPU设备ID列表，例如"4,5,6,7"表示使用ID为4,5,6,7的GPU')
    parser.add_argument('--master_addr', type=str, default='localhost',
                        help='主节点地址')
    parser.add_argument('--master_port', type=str, default='12355',
                        help='主节点端口')
    parser.add_argument('--use_torchrun', action='store_true',
                        help='是否使用torchrun启动（Linux环境推荐）')
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 处理device_ids参数
    device_ids = None
    if args.device_ids:
        # 解析设备ID列表
        device_ids = [int(x.strip()) for x in args.device_ids.split(',')]
        # 更新配置
        config['device_ids'] = device_ids
        # 保存更新后的配置
        with open(args.config, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        # 如果指定了device_ids，则更新gpus_per_node
        if args.gpus_per_node == torch.cuda.device_count():
            args.gpus_per_node = len(device_ids)
    
    # 检测操作系统
    is_windows = platform.system() == 'Windows'
    
    # 在Windows上默认不使用torchrun，除非明确指定
    use_torchrun = args.use_torchrun and not is_windows
    
    # 如果是Windows且没有明确要求使用torchrun，则使用直接调用方式
    if is_windows and not args.use_torchrun:
        if device_ids:
            print(f"在Windows环境下使用直接调用方式启动训练，使用GPU: {args.device_ids}")
        else:
            print(f"在Windows环境下使用直接调用方式启动训练，使用 {args.gpus_per_node} 个GPU...")
        
        # 构建命令
        cmd = [
            sys.executable,
            'lightning_plus.py',
            f'--config={args.config}'
        ]
        
        # 如果没有指定device_ids，则使用gpus参数
        if not device_ids:
            cmd.append(f'--gpus={args.gpus_per_node}')
        
        print(f"命令: {' '.join(cmd)}")
        
        # 执行命令
        process = subprocess.Popen(cmd)
        process.wait()
    else:
        # 计算总进程数
        world_size = args.num_nodes * args.gpus_per_node
        
        # 构建环境变量
        env = os.environ.copy()
        env['MASTER_ADDR'] = args.master_addr
        env['MASTER_PORT'] = args.master_port
        
        # 如果指定了device_ids，设置CUDA_VISIBLE_DEVICES环境变量
        if device_ids:
            env['CUDA_VISIBLE_DEVICES'] = args.device_ids
        
        # 构建命令
        cmd = [
            'torchrun',
            f'--nnodes={args.num_nodes}',
            f'--nproc_per_node={args.gpus_per_node}',
            f'--master_addr={args.master_addr}',
            f'--master_port={args.master_port}',
            'lightning_plus.py',
            f'--config={args.config}'
        ]
        
        if device_ids:
            print(f"启动分布式训练，使用GPU: {args.device_ids}")
        else:
            print(f"启动分布式训练，使用 {world_size} 个GPU...")
        
        print(f"命令: {' '.join(cmd)}")
        
        # 执行命令
        process = subprocess.Popen(cmd, env=env)
        process.wait()
    
    if process.returncode != 0:
        print(f"训练失败，返回码: {process.returncode}")
    else:
        print("训练成功完成！")


if __name__ == "__main__":
    main()