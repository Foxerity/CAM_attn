import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random
import numpy as np
# from utils import seed_everything


def seed_everything(seed):
    """设置随机种子以确保实验可重复性"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MultiConditionDataset(Dataset):
    """用于多条件对齐的数据集加载器
    
    同时加载多种视觉条件下的图像，用于训练多编码器单解码器架构的条件对齐模块(CAM+)。
    每个样本包含所有指定条件下的同一图像。
    """
    def __init__(self, root_dir, target_condition, source_conditions, split='train', transform=None):
        """
        Args:
            root_dir (str): 数据集根目录
            target_condition (str): 目标条件类型，如'depth'
            source_conditions (list): 源条件类型列表，如['canny', 'sketch', 'color']
            split (str): 'train'或'val'
            transform (callable, optional): 应用于图像的转换
        """
        self.root_dir = root_dir
        self.target_condition = target_condition
        self.source_conditions = source_conditions
        self.split = split
        self.transform = transform
        
        # 构建目标条件路径
        self.target_path = os.path.join(root_dir, f"img_{target_condition}", split)
        
        # 获取所有类别
        self.classes = [d for d in os.listdir(self.target_path) 
                       if os.path.isdir(os.path.join(self.target_path, d))]
        
        # 构建图像路径列表
        self.image_paths = []
        for class_name in self.classes:
            class_path = os.path.join(self.target_path, class_name)
            for img_name in os.listdir(class_path):
                if img_name.endswith('.png') or img_name.endswith('.jpg'):
                    # 检查所有条件下是否都有这个图像
                    all_conditions_exist = True
                    for condition in source_conditions:
                        condition_path = os.path.join(
                            root_dir, 
                            f"img_{condition}", 
                            split, 
                            class_name, 
                            img_name
                        )
                        if not os.path.exists(condition_path):
                            all_conditions_exist = False
                            break
                    
                    if all_conditions_exist:
                        self.image_paths.append({
                            'class': class_name,
                            'name': img_name,
                            'target_path': os.path.join(self.target_path, class_name, img_name)
                        })
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_info = self.image_paths[idx]
        class_name = img_info['class']
        img_name = img_info['name']
        
        # 加载目标条件图像 (深度图为单通道)
        target_img = Image.open(img_info['target_path']).convert('L')  # 转换为单通道
        
        # 加载所有源条件图像
        source_images = {}
        for condition in self.source_conditions:
            source_path = os.path.join(
                self.root_dir, 
                f"img_{condition}", 
                self.split, 
                class_name, 
                img_name
            )
            
            # 根据条件类型选择不同的加载模式
            if condition == 'color':
                # 颜色图像保持三通道
                source_img = Image.open(source_path).convert('RGB')
                # 应用RGB转换
                if hasattr(self, 'transform_rgb'):
                    source_img = self.transform_rgb(source_img)
                elif self.transform:
                    source_img = self.transform(source_img)
            else:
                # sketch、canny、depth转为单通道
                source_img = Image.open(source_path).convert('L')
                # 应用灰度转换
                if hasattr(self, 'transform_gray'):
                    source_img = self.transform_gray(source_img)
                elif self.transform:
                    source_img = self.transform(source_img)
            
            source_images[condition] = source_img
        
        # 应用转换到目标图像 (单通道)
        if hasattr(self, 'transform_gray'):
            target_img = self.transform_gray(target_img)
        elif self.transform:
            target_img = self.transform(target_img)
        
        return {
            'source_images': source_images,  # 字典，键为条件名，值为对应的图像张量
            'target_img': target_img,
            'target_condition': self.target_condition,
            'class_name': class_name,
            'img_name': img_name
        }


def get_multi_condition_loaders(config):
    """创建多条件数据加载器
    
    Args:
        config (dict): 配置字典，包含数据集路径、批量大小等参数
        
    Returns:
        tuple: (train_loader, val_loader)
    """
    # 定义图像转换 - 为不同通道数的图像定义不同的归一化参数
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
    
    # 创建训练集
    train_dataset = MultiConditionDataset(
        root_dir=config['dataset_path'],
        target_condition=config['target_condition'],
        source_conditions=config['source_conditions'],
        split='train',
        transform=None  # 不使用统一的transform
    )
    # 设置不同类型的转换
    train_dataset.transform_rgb = transform_rgb
    train_dataset.transform_gray = transform_gray
    
    # 创建验证集
    val_dataset = MultiConditionDataset(
        root_dir=config['dataset_path'],
        target_condition=config['target_condition'],
        source_conditions=config['source_conditions'],
        split='val',
        transform=None  # 不使用统一的transform
    )
    # 设置不同类型的转换
    val_dataset.transform_rgb = transform_rgb
    val_dataset.transform_gray = transform_gray
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    return train_loader, val_loader


def get_data_loaders(config):
    """创建数据加载器
    
    Args:
        config (dict): 配置字典，包含数据集路径、批量大小等参数
        
    Returns:
        tuple: (train_loader, val_loader)
    """
    # 定义图像转换
    transform = transforms.Compose([
        transforms.Resize((config['img_size'], config['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # 创建训练集
    train_dataset = ConditionAlignmentDataset(
        root_dir=config['dataset_path'],
        target_condition=config['target_condition'],
        source_conditions=config['source_conditions'],
        split='train',
        transform=transform
    )
    
    # 创建验证集
    val_dataset = ConditionAlignmentDataset(
        root_dir=config['dataset_path'],
        target_condition=config['target_condition'],
        source_conditions=config['source_conditions'],
        split='val',
        transform=transform
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    return train_loader, val_loader