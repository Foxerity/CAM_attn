import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms
from PIL import Image
import numpy as np

from model_plus import CAMPlus
from data_loader_plus import get_multi_condition_loaders, MultiConditionDataset
from losses import ReconstructionLoss, FeatureMatchingLoss, ContrastiveLoss
from utils import save_image_grid, compute_psnr, compute_ssim


class CAMPlusLightningModule(pl.LightningModule):
    """CAM+模型的PyTorch Lightning实现
    
    封装CAM+模型，支持多GPU训练和验证
    """
    def __init__(self, config):
        super(CAMPlusLightningModule, self).__init__()
        self.save_hyperparameters(config)
        self.config = config
        
        # 创建模型
        self.model = CAMPlus(config)
        
        # 定义损失函数
        self.recon_loss_fn = ReconstructionLoss(loss_type=config.get('recon_loss_type', 'l1'))
        self.feature_matching_loss_fn = FeatureMatchingLoss(loss_type=config.get('feature_matching_loss_type', 'l1'))
        self.contrastive_loss_fn = ContrastiveLoss(temperature=config.get('temperature', 0.5))
        
        # 设置自动优化
        self.automatic_optimization = True
        
        # 最佳验证损失
        self.best_val_loss = float('inf')
        
    def forward(self, source_images, target_img=None):
        """前向传播
        
        Args:
            source_images: 字典，键为条件名，值为对应的图像张量
            target_img: 可选，目标图像
            
        Returns:
            模型输出
        """
        return self.model(source_images, target_img)
    
    def training_step(self, batch, batch_idx):
        """训练步骤
        
        Args:
            batch: 批次数据
            batch_idx: 批次索引
            
        Returns:
            损失值
        """
        # 获取数据
        source_images = batch['source_images']  # 字典，键为条件名，值为对应的图像张量
        target_img = batch['target_img']
        
        # 前向传播
        outputs = self(source_images, target_img)
        
        output_imgs = outputs['outputs']  # 字典，键为条件名，值为对应的生成图像
        mus = outputs['mus']
        logvars = outputs['logvars']
        encoder_features = outputs['encoder_features']
        target_features = outputs['target_features']
        
        # 计算每个条件的损失
        total_recon_loss = 0
        total_kl_loss = 0
        total_feature_matching_loss = 0
        condition_losses = {}
        
        for condition, output_img in output_imgs.items():
            # 1. 重建损失
            recon_loss = self.recon_loss_fn(output_img, target_img)
            
            # 2. KL散度损失
            kl_loss = self.model.kl_divergence_loss(mus[condition], logvars[condition])
            
            # 3. 特征匹配损失
            feature_matching_loss = 0
            if target_features is not None:
                feature_matching_loss = self.feature_matching_loss_fn(encoder_features[condition], target_features)
                total_feature_matching_loss += feature_matching_loss * self.config.get(f'{condition}_weight', 1.0)
            
            # 条件权重
            feature_matching_weight = self.config.get('feature_matching_weight', 0.1)
            kl_weight = self.config.get('beta', 0.01)
            
            # 计算当前条件的总损失
            condition_loss = (
                recon_loss + 
                kl_weight * kl_loss + 
                feature_matching_weight * feature_matching_loss
            )
            
            condition_losses[condition] = condition_loss
            total_recon_loss += recon_loss
            total_kl_loss += kl_loss
            
            # 记录每个条件的损失
            self.log(f'train/{condition}_loss', condition_loss, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        
        # 计算总损失
        total_loss = sum(condition_losses.values())
        
        # 记录损失
        self.log('train/loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train/recon_loss', total_recon_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train/kl_loss', total_kl_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train/feature_matching_loss', total_feature_matching_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        """验证步骤
        
        Args:
            batch: 批次数据
            batch_idx: 批次索引
            
        Returns:
            损失值
        """
        # 获取数据
        source_images = batch['source_images']
        target_img = batch['target_img']
        
        # 前向传播
        outputs = self(source_images)
        output_imgs = outputs['outputs']
        mus = outputs['mus']
        logvars = outputs['logvars']
        
        # 计算每个条件的损失和评估指标
        batch_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        batch_psnr = 0
        batch_ssim = 0
        
        for condition, output_img in output_imgs.items():
            # 计算损失
            recon_loss = self.recon_loss_fn(output_img, target_img)
            kl_loss = self.model.kl_divergence_loss(mus[condition], logvars[condition])
            
            condition_weight = self.config.get(f'{condition}_weight', 1.0)
            kl_weight = self.config.get('beta', 0.01)
            
            condition_loss = (recon_loss + kl_weight * kl_loss) * condition_weight
            batch_loss += condition_loss
            
            # 累计重建损失和KL损失
            total_recon_loss += recon_loss * condition_weight
            total_kl_loss += kl_loss * condition_weight
            
            # 计算评估指标
            psnr = compute_psnr(output_img, target_img)
            ssim = compute_ssim(output_img, target_img)
            
            # 记录每个条件的评估指标
            self.log(f'val/{condition}_psnr', psnr, on_epoch=True, prog_bar=False, sync_dist=True)
            self.log(f'val/{condition}_ssim', ssim, on_epoch=True, prog_bar=False, sync_dist=True)
            
            # 累计PSNR和SSIM
            batch_psnr += psnr / len(output_imgs)
            batch_ssim += ssim / len(output_imgs)
        
        # 记录损失和评估指标
        self.log('val/loss', batch_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val/recon_loss', total_recon_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val/kl_loss', total_kl_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val/psnr', batch_psnr, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val/ssim', batch_ssim, on_epoch=True, prog_bar=True, sync_dist=True)
        
        # 保存生成的图像样本（仅在第一个进程上执行）
        if batch_idx == 0 and self.global_rank == 0:
            self._log_images(source_images, target_img, output_imgs)
        
        return batch_loss
    
    def _log_images(self, source_images, target_img, output_imgs):
        """记录图像样本
        
        Args:
            source_images: 源图像字典
            target_img: 目标图像
            output_imgs: 生成的图像字典
        """
        # 创建一个包含源图像、每个条件生成的图像和目标图像的网格
        images = []
        
        # 添加目标图像
        images.append(("Target", target_img))
        
        # 添加每个条件的源图像和生成图像
        for condition in source_images:
            images.append((f"Source ({condition})", source_images[condition]))
            if condition in output_imgs:
                images.append((f"Generated ({condition})", output_imgs[condition]))
        
        # 保存图像网格
        output_dir = self.config.get('output_dir', './output_plus')
        os.makedirs(output_dir, exist_ok=True)
        
        save_image_grid(
            images,
            os.path.join(output_dir, f'samples_epoch_{self.current_epoch}.png'),
            nrow=3  # 每行显示3张图像
        )
    
    def configure_optimizers(self):
        """配置优化器和学习率调度器
        
        Returns:
            优化器和学习率调度器
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config['lr'])
        scheduler = StepLR(optimizer, step_size=self.config['lr_step'], gamma=0.5)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        }


class CAMPlusDataModule(pl.LightningDataModule):
    """CAM+数据模块
    
    封装数据加载和处理逻辑
    """
    def __init__(self, config):
        super(CAMPlusDataModule, self).__init__()
        self.config = config
        self.batch_size = config['batch_size']
        self.num_workers = config.get('num_workers', 4)
        self.dataset_path = config['dataset_path']
        self.target_condition = config['target_condition']
        self.source_conditions = config['source_conditions']
        self.img_size = config['img_size']
        
    def setup(self, stage=None):
        """准备数据集
        
        Args:
            stage: 'fit'或'test'
        """
        # 为RGB和灰度图像定义不同的转换
        self.transform_rgb = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        self.transform_gray = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        # 创建训练和验证数据集
        if stage == 'fit' or stage is None:
            self.train_dataset = MultiConditionDataset(
                root_dir=self.dataset_path,
                target_condition=self.target_condition,
                source_conditions=self.source_conditions,
                split='train'
            )
            
            self.val_dataset = MultiConditionDataset(
                root_dir=self.dataset_path,
                target_condition=self.target_condition,
                source_conditions=self.source_conditions,
                split='val'
            )
    
    def train_dataloader(self):
        """获取训练数据加载器
        
        Returns:
            训练数据加载器
        """
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self._collate_fn
        )
    
    def val_dataloader(self):
        """获取验证数据加载器
        
        Returns:
            验证数据加载器
        """
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self._collate_fn
        )
    
    def _collate_fn(self, batch):
        """自定义批次收集函数
        
        处理不同条件的图像，并将它们组织成字典形式
        
        Args:
            batch: 批次数据列表
            
        Returns:
            处理后的批次数据
        """
        source_images = {condition: [] for condition in self.source_conditions}
        target_imgs = []
        
        for item in batch:
            # 收集所有源条件图像
            for condition in self.source_conditions:
                source_images[condition].append(item[f'{condition}_img'])
            
            # 收集目标图像
            target_imgs.append(item['target_img'])
        
        # 将列表转换为张量
        result = {
            'source_images': {},
            'target_img': torch.stack(target_imgs)
        }
        
        for condition in self.source_conditions:
            result['source_images'][condition] = torch.stack(source_images[condition])
        
        return result


def train_with_lightning(config):
    """使用PyTorch Lightning训练CAM+模型
    
    Args:
        config: 配置参数
    """
    # 设置随机种子
    pl.seed_everything(config['seed'])
    
    # 创建输出目录
    output_dir = config.get('output_dir', './output_plus')
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建模型和数据模块
    model = CAMPlusLightningModule(config)
    data_module = CAMPlusDataModule(config)
    
    # 创建回调
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,
        filename='cam_plus-{epoch:02d}-{val/loss:.4f}',
        save_top_k=3,
        verbose=True,
        monitor='val/loss',
        mode='min'
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    # 创建日志记录器
    logger = TensorBoardLogger(save_dir=output_dir, name='lightning_logs')
    
    # 确定使用的GPU数量
    if 'gpus' in config:
        devices = config['gpus']
    else:
        devices = -1  # 使用所有可用的GPU
    
    # 确定使用的策略
    import platform
    is_windows = platform.system() == 'Windows'
    
    # Windows环境下默认使用单GPU模式
    strategy = config.get('strategy', 'ddp')  # 默认使用DDP策略
    
    # 创建训练器
    trainer = pl.Trainer(
        max_epochs=config['epochs'],
        accelerator='gpu',  # 使用GPU加速
        devices=devices,
        strategy=strategy,
        precision=config.get('precision', 16),  # 使用混合精度训练
        callbacks=[checkpoint_callback, lr_monitor],
        logger=logger,
        log_every_n_steps=50,
        deterministic=False,  # 允许非确定性优化以提高性能
        accumulate_grad_batches=config.get('accumulate_grad_batches', 1),
        gradient_clip_val=config.get('gradient_clip_val', None),
        sync_batchnorm=config.get('sync_batchnorm', False)
    )
    
    # 训练模型
    trainer.fit(model, data_module)
    
    # 返回最佳模型路径
    return checkpoint_callback.best_model_path


class LightningCLIApp(LightningCLI):
    """Lightning CLI应用
    
    提供命令行接口，用于训练和测试CAM+模型
    """
    def add_arguments_to_parser(self, parser):
        parser.add_argument('--config', type=str, default='config_plus.json',
                            help='配置文件路径')


def main():
    """主函数"""
    import json
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='CAM+ Lightning: 多条件对齐模块（多GPU版本）')
    parser.add_argument('--config', type=str, default='config_plus.json',
                        help='配置文件路径')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'],
                        help='运行模式: train或test')
    parser.add_argument('--gpus', type=int, default=None,
                        help='使用的GPU数量，仅在直接调用模式下有效')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='本地进程排名，由torchrun自动设置')
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 确保设备正确设置
    config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 如果指定了GPU数量，更新配置
    if args.gpus is not None:
        config['gpus'] = args.gpus
    
    if args.mode == 'train':
        # 训练模式
        print("开始使用PyTorch Lightning训练CAM+模型...")
        best_model_path = train_with_lightning(config)
        print(f"训练完成，最佳模型保存在: {best_model_path}")
    else:
        # 测试模式 - 可以根据需要实现
        print("测试模式尚未实现")


if __name__ == "__main__":
    main()