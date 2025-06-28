import json
import math
import os

import torch
from PIL import Image
from pytorch_fid import fid_score
from piq import LPIPS
from torchvision.transforms.functional import to_tensor

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


def compute_gfid(fid_dict: dict[str, float]) -> float:
    """
    计算几何平均 FID（gFID）。

    Args:
        fid_dict: 类别到 FID 值的映射
    Returns:
        gfid: float
    """
    # 只保留 FID>0 的值
    values = [v for v in fid_dict.values() if v > 0]
    if not values:
        raise ValueError("没有合法的 FID 值")
    # 求对数、平均、再指数化
    log_sum = sum(math.log(v) for v in values)
    mean_log = log_sum / len(values)
    return math.exp(mean_log)


def save_inference_tensors(target: torch.Tensor,
                           source: dict[str, torch.Tensor],
                           class_names: list[str],
                           root: str) -> None:
    """
    将一批 target 和 source tensors 保存为灰度图，根据 class_name 分类存放。

    Args:
        target: torch.Tensor, 形状 (B, 1, 128, 128)，值域 [-1, 1]
        source: dict[str, torch.Tensor]，每个 value 形状同 target
        class_names: 长度为 B 的字符串列表，对应每个样本的类别
        root: 保存根目录，会在其下创建 `target/` 和每个 source key 的子目录
    """
    # 把所有 tensor 转到 CPU
    target = target.detach().cpu()
    for key in source:
        source[key] = source[key].detach().cpu()

    B, C, H, W = target.shape
    assert C == 1 and H == 128 and W == 128, "target 尺寸需为 (B,1,128,128)"
    for key, tensor in source.items():
        assert tensor.shape == (B, 1, 128, 128), f"source['{key}'] 尺寸需为 (B,1,128,128)"
    assert len(class_names) == B, "class_names 长度必须等于 batch 大小"

    def _save_one(img_tensor: torch.Tensor, save_dir: str, idx: int):
        # 归一化到 [0,255]
        arr = ((img_tensor + 1.0) / 2.0 * 255.0) \
            .clamp(0, 255) \
            .to(torch.uint8) \
            .squeeze(0) \
            .numpy()  # (128,128)
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"{idx}.png")
        Image.fromarray(arr).save(path)

    for i in range(B):
        cls = class_names[i]
        # 保存 target
        tgt_dir = os.path.join(root, "target", cls)
        _save_one(target[i], tgt_dir, i)

        # 保存每个 source
        for key, tensor in source.items():
            src_dir = os.path.join(root, key, cls)
            _save_one(tensor[i], src_dir, i)



class CAMInfer:
    def __init__(self, config_path = "config_infer.json"):
        self.config_path = config_path
        self.config = load_config(self.config_path)
        self.model_path = r"output_plus/new3_flow2_shareB/new3_flow2_shareB/568-0.05.ckpt"
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

                save_inference_tensors(target_img, outputs, class_name, "./infer_results")
                # save_batch_tensors_as_images(target_img, class_name, "./infer_results")

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

            lpips_model = LPIPS()
            root = "./infer_results"
            types = os.listdir(root)
            real_images_folder = 'infer_results/target'
            cls_fid = {}
            cls_lpips = {}
            for typ in types:
                if typ != 'target':
                    generated_images_folder = os.path.join(root, typ)
                    for cls in os.listdir(generated_images_folder):
                        gen_cls_folder = os.path.join(generated_images_folder, cls)
                        real_cls_folder = os.path.join(real_images_folder, cls)

                        # —— 1) FID ——
                        fid_value = fid_score.calculate_fid_given_paths(
                            [gen_cls_folder, real_cls_folder],
                            batch_size=50, device=self.config['device'], dims=64
                        )
                        cls_fid[cls] = fid_value

                        # —— 2) LPIPS ——
                        # 2.1 加载所有生成/真实图像
                        gen_images = sorted(os.listdir(gen_cls_folder))
                        real_images = sorted(os.listdir(real_cls_folder))

                        fake_tensors = torch.stack([
                            to_tensor(Image.open(os.path.join(gen_cls_folder, fn)).convert('RGB'))
                            for fn in gen_images
                        ]).to(self.config['device'])
                        real_tensors = torch.stack([
                            to_tensor(Image.open(os.path.join(real_cls_folder, fn)).convert('RGB'))
                            for fn in real_images
                        ]).to(self.config['device'])

                        # 2.2 分批计算 LPIPS
                        batch_size = 50
                        with torch.no_grad():
                            for i in range(0, fake_tensors.size(0), batch_size):
                                f_batch = fake_tensors[i:i + batch_size]
                                r_batch = real_tensors[i:i + batch_size]
                                # LPIPS 返回形如 (B,) 或 (B,1,1)
                                lpips_vals = lpips_model(f_batch, r_batch)

                        # 2.3 求平均 LPIPS
                        cls_lpips[cls] = lpips_vals

                        # —— 3) 汇总并打印 ——
                        # FID
                    gfid = compute_gfid(cls_fid)
                    avgfid = sum(cls_fid.values()) / len(cls_fid)

                    # LPIPS
                    gplips = compute_gfid(cls_lpips)  # gLPIPS 用相同的几何平均函数
                    avglpips = sum(cls_lpips.values()) / len(cls_lpips)

                    print(f"{typ}  gFID:    {gfid:.5f}, avgFID:    {avgfid:.5f}")
                    print(f"{typ}  gLPIPS:  {gplips:.5f}, avgLPIPS:  {avglpips:.5f}")

                    cls_fid = {}
                    cls_lpips = {}



if __name__ == '__main__':
    CAM = CAMInfer()
    CAM.load_model()
    CAM.inference()