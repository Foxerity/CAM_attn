import os
import argparse
from PIL import Image
import numpy as np
from tqdm import tqdm

def convert_to_grayscale(image_path):
    """将三通道图像转换为单通道灰度图像
    
    Args:
        image_path: 图像路径
        
    Returns:
        成功转换返回True，否则返回False
    """
    try:
        # 打开图像
        img = Image.open(image_path)
        
        # 检查图像是否为RGB模式
        if img.mode == 'RGB':
            # 将图像转换为numpy数组
            img_array = np.array(img)
            
            # 检查三个通道是否相同
            r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
            if np.array_equal(r, g) and np.array_equal(g, b):
                # 直接取第一个通道作为灰度图
                gray_img = Image.fromarray(r)
            else:
                # 计算三个通道的平均值
                gray_array = np.mean(img_array, axis=2).astype(np.uint8)
                gray_img = Image.fromarray(gray_array)
            
            # 保存为单通道图像并覆盖原图
            gray_img.save(image_path)
            return True
        elif img.mode == 'L':
            # 已经是单通道图像，不需要转换
            return True
        else:
            print(f"警告: {image_path} 不是RGB或L模式，而是 {img.mode} 模式")
            return False
    except Exception as e:
        print(f"处理 {image_path} 时出错: {e}")
        return False

def process_directory(directory):
    """处理目录中的所有图像
    
    Args:
        directory: 目录路径
        
    Returns:
        处理的图像数量和成功转换的图像数量
    """
    total_images = 0
    successful_conversions = 0
    
    # 遍历目录中的所有文件和子目录
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                total_images += 1
                image_path = os.path.join(root, file)
                if convert_to_grayscale(image_path):
                    successful_conversions += 1
    
    return total_images, successful_conversions

def main():
    parser = argparse.ArgumentParser(description='将三通道图像转换为单通道灰度图像')
    parser.add_argument('--data_dir', type=str, required=True, help='数据集根目录')
    parser.add_argument('--conditions', nargs='+', default=['sketch', 'canny', 'depth'], 
                        help='需要转换的条件类型，默认为sketch、canny和depth')
    args = parser.parse_args()
    
    # 处理每种条件
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
    
    print("\n所有图像处理完成!")

if __name__ == "__main__":
    main()