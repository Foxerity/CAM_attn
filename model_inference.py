import json
import os

import torch
from PIL import Image
from datasets import tqdm

from data_loader_plus import get_multi_condition_loaders
from model_plus import CAMPlus
from torchvision import transforms

from utils import save_image_grid


def load_config(config_path):
    """加载配置文件

    Args:
        config_path: 配置文件路径

    Returns:
        配置字典
    """
    # 加载配置
    with open(config_path, 'r') as f:
        config = json.load(f)

    return config

# 加载配置
config = load_config('config_plus.json')
config["batch_size"] = 1
train_loader, val_loader = get_multi_condition_loaders(config)

model = CAMPlus(config).to(config['device'])

ckpt = torch.load(r"output_plus/new3_flow1_0.5_0.5/572-0.05.ckpt", map_location="cpu")

# 如果权重嵌套在 'state_dict' 或 'model' 字段中，先取出来
state_dict = ckpt.get('state_dict', ckpt.get('model', ckpt))

# 去除 'model.' 前缀
new_state_dict = {}
for k, v in state_dict.items():
    if k.startswith("model."):
        new_key = k[len("model."):]  # 去掉前缀
    else:
        new_key = k
    new_state_dict[new_key] = v
model.load_state_dict(new_state_dict)
model.eval()

# 为RGB和灰度图像定义不同的转换
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

with torch.no_grad():
    for idx, batch in enumerate(val_loader):
        # 获取数据
        source_images = batch['source_images']
        target_img = batch['target_img'].to(config['device'])

        # 将所有源图像移动到设备
        for condition in source_images:
            source_images[condition] = source_images[condition].to(config['device'])

        # 处理输入图像
        processed_images = {}
        for condition, img in source_images.items():
            if isinstance(img, Image.Image):
                # 根据条件类型选择正确的转换
                if condition == 'color':
                    # 颜色图像使用RGB转换
                    img_tensor = transform_rgb(img).unsqueeze(0).to(config['device'])
                else:
                    # 其他条件（sketch、canny、depth）使用灰度转换
                    # 确保图像是单通道
                    img_gray = img.convert('L') if img.mode != 'L' else img
                    img_tensor = transform_gray(img_gray).unsqueeze(0).to(config['device'])
            else:
                # 如果已经是张量，确保形状正确并移动到正确的设备
                img_tensor = img.to(config['device'])
                if img_tensor.dim() == 3:
                    img_tensor = img_tensor.unsqueeze(0)

            processed_images[condition] = img_tensor

        # 推理
        with torch.no_grad():
            outputs = model(processed_images)
            # 获取第一个条件的输出作为结果
            # 注意：outputs['outputs']是一个字典，包含每个条件的输出

        # 保存图像网格
        # 创建一个包含源图像、每个条件生成的图像和目标图像的网格
        images = [("Target", target_img)]

        # 添加目标图像
        outputs = outputs["outputs"]
        # 添加每个条件的源图像和生成图像
        for condition in source_images:
            images.append((f"Source ({condition})", source_images[condition]))
            if condition in outputs:
                images.append((f"Generated ({condition})", outputs[condition]))

        save_image_grid(
            images,
            os.path.join("./infer_results", f'samples_{idx}.png'),
            nrow=3  # 每行显示3张图像
        )