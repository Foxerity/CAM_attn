import json
import os

import torch

from data_loader_plus import get_multi_condition_loaders
from model_plus import CAMPlus

from utils import save_image_grid, compute_psnr, compute_ssim


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
    def __init__(self, config_path = "config_infer.json"):
        self.config_path = config_path
        self.config = load_config(self.config_path)
        self.model_path = r"output_plus/new3_flow1_shareB/new3_flow1_shareB/574-0.05.ckpt"
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
                source_images:dict = batch['source_images']
                target_img:torch.Tensor = batch['target_img'].to(self.config['device']) # (B, 1, 128, 128)
                class_name:list = batch['class_name']

                # 将所有源图像移动到设备
                for condition in source_images:
                    source_images[condition] = source_images[condition].to(self.config['device'])



                # 推理
                with torch.no_grad():
                    outputs = self.model(source_images)
                    outputs = outputs["outputs"]
                    # 获取第一个条件的输出作为结果
                    # 注意：outputs['outputs']是一个字典，包含每个条件的输出

                images = [("Target", target_img)]
                for k, v in outputs.items():
                    psnr_attr = f"{k}_psnr_list"
                    ssim_attr = f"{k}_ssim_list"
                    if not hasattr(self, psnr_attr):
                        setattr(self, psnr_attr, [])
                        setattr(self, ssim_attr, [])
                    psnr_list = getattr(self, psnr_attr)
                    ssim_list = getattr(self, ssim_attr)
                    psnr_list.append(compute_psnr(target_img, v))
                    ssim_list.append(compute_ssim(target_img, v))

                
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

            # 2. 打印每条分支的平均值
            total_psnr_sum = 0.0
            total_count = 0
            for branch_name in outputs.keys():
                psnr_list = getattr(self, f"{branch_name}_psnr_list")
                avg_psnr = sum(psnr_list) / len(psnr_list)
                print(f"{branch_name} PSNR average: {avg_psnr:.4f}")

                total_psnr_sum += sum(psnr_list)
                total_count += len(psnr_list)

            # 3. 打印所有分支 PSNR 的总体平均
            overall_avg_psnr = total_psnr_sum / total_count
            print(f"Overall PSNR average: {overall_avg_psnr:.4f}")

            # （同理，如果需要，也可以打印 SSIM 的平均值）
            total_ssim_sum = 0.0
            total_ssim_count = 0
            for branch_name in outputs.keys():
                ssim_list = getattr(self, f"{branch_name}_ssim_list")
                avg_ssim = sum(ssim_list) / len(ssim_list)
                print(f"{branch_name} SSIM average: {avg_ssim:.4f}")

                total_ssim_sum += sum(ssim_list)
                total_ssim_count += len(ssim_list)

            overall_avg_ssim = total_ssim_sum / total_ssim_count
            print(f"Overall SSIM average: {overall_avg_ssim:.4f}")


if __name__ == '__main__':
    CAM = CAMInfer()
    CAM.load_model()
    CAM.inference()