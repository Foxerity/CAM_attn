import os
import torch

import pytorch_lightning as pl

from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim.lr_scheduler import OneCycleLR
from torchvision import transforms

from enhanced_vae import EntropyKLLoss
from model_plus import CAMPlus
from data_loader_plus import MultiConditionDataset
from losses import ReconstructionLoss, FeatureMatchingLoss
from utils import save_image_grid, compute_psnr, compute_ssim


class CAMPlusLightningModule(pl.LightningModule):
    """CAM+模型的PyTorch Lightning实现
    
    封装CAM+模型，支持多GPU训练和验证
    """
    def __init__(self, config):
        super(CAMPlusLightningModule, self).__init__()
        self._last_val_batch = None
        self.save_hyperparameters(config)
        self.config = config

        self.conditions = list(config['source_conditions'])

        self.KLLoss = EntropyKLLoss()

        self.tag = config.get("tag")

        self.kl_min = 1e-5
        self.kl_max = config.get('beta', 0.1)                               # config
        self.kl_warmup_epochs = config.get('beta_warmup_epochs', 20)        # config
        self.kl_increase_epochs = config.get('kl_increase_epochs', 10)
        
        # 创建模型
        self.model = CAMPlus(config)
        
        # 定义损失函数
        self.recon_loss_fn = ReconstructionLoss(loss_type=config.get('recon_loss_type', 'l1'))
        self.feature_matching_loss_fn = FeatureMatchingLoss(loss_type=config.get('feature_matching_loss_type', 'l1'))
        
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

        epoch = float(self.current_epoch)
        
        # 前向传播
        outputs = self(source_images, target_img)
        
        output_imgs = outputs['outputs']  # 字典，键为条件名，值为对应的生成图像
        mus = outputs['mus']
        logvars = outputs['logvars']
        z = outputs['z']
        total_log_det = outputs['total_log_det']
        log_qk = outputs['all_log_qk']
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
            # kl_loss = self.model.kl_divergence_loss(mus[condition], logvars[condition])

            # 3. 特征匹配损失
            feature_matching_loss = 0
            # if target_features is not None:
                # feature_matching_loss = self.feature_matching_loss_fn(encoder_features[condition], target_features)
                # total_feature_matching_loss += feature_matching_loss * self.config.get(f'{condition}_weight', 1.0)
            
            # 条件权重
            # feature_matching_weight = self.config.get('feature_matching_weight', 0.1)
            # kl_weight = self.config.get('beta', 0.01)


            if epoch < self.kl_warmup_epochs:
                kl_weight = self.kl_min + (self.kl_max - self.kl_min) * (epoch
                                                                         / self.kl_warmup_epochs)
            else:
                # 达到 n epoch 后就保持最大值
                kl_weight = self.kl_max
            self.log('kl_weight', kl_weight, prog_bar=True)

            kl_loss_outputs = self.KLLoss(z[condition],
                                  mus[condition],
                                  logvars[condition],
                                  log_qk[condition],
                                  total_log_det[condition]
                                  )
            kl_loss = kl_loss_outputs["loss"]

            # kl_loss = self.KLLoss(total_log_det[condition])
            # 计算当前条件的总损失
            # condition_loss = (
            #     recon_loss +
            #     kl_weight * kl_loss +
            #     feature_matching_weight * feature_matching_loss
            # )
            condition_loss = (
                    recon_loss +
                    kl_loss * kl_weight
            )
            
            condition_losses[condition] = condition_loss
            total_recon_loss += recon_loss
            total_kl_loss += kl_loss
            
            # 记录每个条件的损失
            self.log(f'train/{condition}_loss', condition_loss, on_epoch=True, prog_bar=False, sync_dist=True)
            self.log(f'train/{condition}_kl', kl_loss_outputs["kl"], on_epoch=True, prog_bar=False, sync_dist=True)
            self.log(f'train/{condition}_q0', kl_loss_outputs["q0"], on_epoch=True, prog_bar=False, sync_dist=True)
            self.log(f'train/{condition}_z', kl_loss_outputs["z"], on_epoch=True, prog_bar=False, sync_dist=True)

        
        # 计算总损失
        total_loss = sum(condition_losses.values())

        # 记录损失
        self.log('train/loss', total_loss, on_step=True, prog_bar=True, sync_dist=True)
        self.log('recon', total_recon_loss, on_step=True, prog_bar=True, sync_dist=True)
        self.log('kl', total_kl_loss, on_step=True, prog_bar=True, sync_dist=False)
        # self.log('feature', total_feature_matching_loss, on_step=True, prog_bar=True, sync_dist=True)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        """
        验证一步：计算各 condition 的 loss、PSNR、SSIM，并在 step 层面 log，
        设置 on_epoch=True + sync_dist=True，让 Lightning 汇总到 epoch 结束后。
        """
        source_images = batch['source_images']
        target_img    = batch['target_img']

        # 前向
        outs   = self(source_images)
        outputs = outs['outputs']    # dict: condition -> generated image
        mus     = outs['mus']
        logvars = outs['logvars']

        z = outs['z']
        log_qk = outs['all_log_qk']
        total_log_det = outs['total_log_det']

        batch_loss      = 0.0
        total_recon_loss = 0.0
        total_kl_loss    = 0.0

        psnr_vals = []
        ssim_vals = []


        # 对每个 condition 分别计算
        for cond, out_img in outputs.items():
            # 重建 + KL loss
            recon = self.recon_loss_fn(out_img, target_img)
            # kl    = self.model.kl_divergence_loss(mus[cond], logvars[cond])
            w_cond = self.config.get(f'{cond}_weight', 1.0)
            w_kl   = self.kl_max

            # kl = FlowKLLoss(beta=w_kl)(mus[cond], logvars[cond], total_log_det[cond])
            kl_outputs = self.KLLoss(z[cond],
                             mus[cond],
                             logvars[cond],
                             log_qk[cond],
                             total_log_det[cond]
                             )
            # kl = self.KLLoss(total_log_det[cond])
            # cond_loss = (recon + w_kl * kl) * w_cond

            kl = kl_outputs["loss"]

            cond_loss = (recon + kl * w_kl) * w_cond

            batch_loss      += cond_loss
            total_recon_loss += recon * w_cond
            total_kl_loss    += kl    * w_cond

            # 评估指标
            psnr = compute_psnr(out_img, target_img)
            ssim = compute_ssim(out_img, target_img)

            psnr_vals.append(psnr)
            ssim_vals.append(ssim)

            # Per‐condition log，on_epoch + sync_dist
            self.log(f'val/{cond}_psnr', psnr,
                     on_epoch=True, prog_bar=False, sync_dist=True)
            self.log(f'val/{cond}_ssim', ssim,
                     on_epoch=True, prog_bar=False, sync_dist=True)
            self.log(f'val/{cond}_kl', kl_outputs["kl"],
                     on_epoch=True, prog_bar=False, sync_dist=True)
            self.log(f'val/{cond}_q0', kl_outputs["q0"],
                     on_epoch=True, prog_bar=False, sync_dist=True)
            self.log(f'val/{cond}_z', kl_outputs["z"],
                     on_epoch=True, prog_bar=False, sync_dist=True)

        # 全局 loss 和两项子项
        self.log('v/loss', batch_loss,
                 on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('v/recon', total_recon_loss,
                 on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('v/kl',    total_kl_loss,
                 on_epoch=True, prog_bar=True, sync_dist=True)

        # 计算所有 condition 平均 PSNR/SSIM 并 log
        batch_psnr = sum(psnr_vals) / len(psnr_vals)
        batch_ssim = sum(ssim_vals) / len(ssim_vals)
        self.log('psnr', batch_psnr,
                 on_epoch=True, sync_dist=True)
        self.log('ssim', batch_ssim,
                 on_epoch=True, sync_dist=True)

        if (batch_idx == 0 and self.trainer.is_global_zero) and not self.trainer.sanity_checking:
            # 存起来，给 epoch_end 用
            self._last_val_batch = (batch['source_images'], batch['target_img'], outputs)

    def on_validation_epoch_end(self) -> None:
        # 只在 global zero （rank0） 打印，避免多卡重复输出
        if not self.trainer.is_global_zero or self.trainer.sanity_checking:
            return

        metrics = self.trainer.callback_metrics
        # 整体指标
        loss = metrics.get('v/loss_epoch') or metrics.get('v/loss')
        recon = metrics.get('v/recon_epoch') or metrics.get('v/recon')
        kl = metrics.get('v/kl_epoch') or metrics.get('v/kl')
        psnr = metrics.get('psnr_epoch') or metrics.get('psnr')
        ssim = metrics.get('ssim_epoch') or metrics.get('ssim')

        print(f"\n=== Epoch {self.current_epoch} Summary ===")
        print(f"  loss:  {loss:.2f}, recon: {recon:.2f}, kl: {kl:.2f}")
        print(f"  psnr:  {psnr:.2f}, ssim: {ssim:.2f}")

        # 各 condition 的 psnr/ssim
        for cond in self.conditions:
            p = metrics.get(f'val/{cond}_psnr_epoch') or metrics.get(f'val/{cond}_psnr')
            s = metrics.get(f'val/{cond}_ssim_epoch') or metrics.get(f'val/{cond}_ssim')
            print(f"  [{cond}]  PSNR: {p:.2f}, SSIM: {s:.2f}")

        n = self.config.get('sample_interval', 5)
        if self.current_epoch % n == 0:
            source_images, target_img, outputs = self._last_val_batch
            self._log_images(source_images, target_img, outputs)

    
    def _log_images(self, source_images, target_img, output_imgs):
        """记录图像样本
        
        Args:
            source_images: 源图像字典
            target_img: 目标图像
            output_imgs: 生成的图像字典
        """
        # 创建一个包含源图像、每个条件生成的图像和目标图像的网格
        images = [("Target", target_img)]
        
        # 添加目标图像

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
            os.path.join(output_dir, f'{self.tag}/ep_{self.current_epoch}.png'),
            nrow=3  # 每行显示3张图像
        )
    
    def configure_optimizers(self):
        """配置优化器和学习率调度器
        
        Returns:
            优化器和学习率调度器
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config['lr'])
        total_steps = self.trainer.estimated_stepping_batches
        scheduler = OneCycleLR(optimizer,
                               max_lr=self.config["lr"] * 10,
                               total_steps=total_steps,
                               anneal_strategy=self.config.get('anneal_strategy', 'cos'),
                               div_factor=self.config.get('div_factor', 20),
                               final_div_factor=1e3,
                               last_epoch=-1)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
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
        transform_rgb = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        transform_gray = transforms.Compose([
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
            # 设置不同类型的转换
            self.train_dataset.transform_rgb = transform_rgb
            self.train_dataset.transform_gray = transform_gray
            
            self.val_dataset = MultiConditionDataset(
                root_dir=self.dataset_path,
                target_condition=self.target_condition,
                source_conditions=self.source_conditions,
                split='val'
            )

            self.val_dataset.transform_rgb = transform_rgb
            self.val_dataset.transform_gray = transform_gray
    
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
                # 直接从source_images字典中获取对应条件的图像
                source_images[condition].append(item['source_images'][condition])

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
    tag = config.get('tag')
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建模型和数据模块
    model = CAMPlusLightningModule(config)
    data_module = CAMPlusDataModule(config)
    
    # 创建回调
    checkpoint_callback = ModelCheckpoint(
        dirpath=f'{tag}/{output_dir}',
        filename=f'{tag}'+'/{epoch:02d}-{v/recon:.2f}',
        save_top_k=1,
        verbose=False,
        monitor='v/recon',
        mode='min',
        auto_insert_metric_name=False,
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    # 创建日志记录器
    logger = TensorBoardLogger(save_dir=output_dir, name='lightning_logs')
    
    # 确定使用的GPU设备
    if 'device_ids' in config:
        # 使用指定的GPU设备ID列表
        devices = config['device_ids']
    elif 'gpus' in config:
        # 使用指定数量的GPU（从0开始）
        devices = config['gpus']
    else:
        # 使用所有可用的GPU
        devices = -1

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
        sync_batchnorm=config.get('sync_batchnorm', True),
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
    parser.add_argument('--batch_size', type=int, default=None,
                        help='batch_size')
    parser.add_argument('--device_ids', type=int, nargs='+',default=[4, 5, 6, 7],
                        help='标识')
    parser.add_argument('--tag', type=str, default=None,
                        help='标识')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='本地进程排名，由torchrun自动设置')
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 确保设备正确设置
    config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.device_ids is not None:
        config['device_ids'] = args.device_ids

    if args.tag is not None:
        config['tag'] = args.tag

    # 如果指定了GPU数量，更新配置
    if args.gpus is not None:
        config['gpus'] = args.gpus

    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    
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