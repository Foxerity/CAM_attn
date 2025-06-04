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


class CAMInfer:
    def __init__(self, config_path, model_path):
        self.config_path = config_path
        self.config = load_config(self.config_path)
        self.model_path = r"output_plus/new3_flow1_0.5_0.5/572-0.05.ckpt"
        self.train_loader, self.val_loader = get_multi_condition_loaders(self.config)

        self.model = None

    def load_model(self):
        self.config["batch_size"] = 1

        self.model = CAMPlus(self.config).to(self.config['device'])

        ckpt = torch.load(self.model_path, map_location="cpu")

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
        self.model.load_state_dict(new_state_dict)
        self.model.eval()


    def inference(self):

        with torch.no_grad():
            for idx, batch in enumerate(self.val_loader):
                # 获取数据
                source_images = batch['source_images']
                target_img = batch['target_img'].to(self.config['device'])

                # 将所有源图像移动到设备
                for condition in source_images:
                    source_images[condition] = source_images[condition].to(self.config['device'])



                # 推理
                with torch.no_grad():
                    outputs = self.model(batch)
                    outputs = outputs["outputs"]
                    # 获取第一个条件的输出作为结果
                    # 注意：outputs['outputs']是一个字典，包含每个条件的输出

                images = [("Target", target_img)]
                
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