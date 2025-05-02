# CAM+ 多GPU训练指南

本文档提供了使用PyTorch Lightning框架对CAM+模型进行多GPU训练的指南。

## 文件说明

- `lightning_plus.py`: 主要的Lightning模块实现，包含模型、数据加载和训练逻辑
- `run_distributed.py`: 分布式训练启动脚本
- `config_lightning.json`: 多GPU训练的配置文件

## 环境要求

```
pip install pytorch-lightning tensorboard
```

## 单机多GPU训练

使用以下命令启动单机多GPU训练：

```bash
python run_distributed.py --config config_lightning.json --gpus_per_node 4
```

其中，`--gpus_per_node`参数指定每个节点使用的GPU数量，默认使用所有可用的GPU。

## 多节点训练

对于多节点训练，需要在每个节点上运行以下命令：

```bash
# 在主节点上
python run_distributed.py --config config_lightning.json --num_nodes 2 --master_addr <主节点IP> --master_port 12355 --node_rank 0

# 在从节点上
python run_distributed.py --config config_lightning.json --num_nodes 2 --master_addr <主节点IP> --master_port 12355 --node_rank 1
```

## 直接使用Lightning模块

也可以直接使用Lightning模块进行训练：

```bash
python lightning_plus.py --config config_lightning.json
```

## 配置文件说明

`config_lightning.json`文件包含了训练所需的所有配置参数，可以根据需要进行修改：

- `batch_size`: 每个GPU上的批量大小
- `num_workers`: 每个GPU上的数据加载线程数
- `precision`: 训练精度，可选16（混合精度）或32（全精度）
- `strategy`: 分布式策略，默认为'ddp'
- `sync_batchnorm`: 是否同步批量归一化层，多GPU训练时建议设为true
- `accumulate_grad_batches`: 梯度累积步数，可用于增大有效批量大小
- `gradient_clip_val`: 梯度裁剪值，防止梯度爆炸

## 训练监控

训练过程中的损失和评估指标会记录到TensorBoard中，可以使用以下命令查看：

```bash
tensorboard --logdir ./output_lightning/lightning_logs
```

## 模型保存

训练过程中，最佳模型会自动保存到`output_lightning`目录下，文件名格式为`cam_plus-{epoch:02d}-{val/loss:.4f}.ckpt`。

## 使用保存的模型

可以使用以下代码加载保存的模型：

```python
from lightning_plus import CAMPlusLightningModule

# 加载模型
model = CAMPlusLightningModule.load_from_checkpoint('path/to/checkpoint.ckpt')
model.eval()

# 使用模型进行推理
with torch.no_grad():
    outputs = model(source_images)
```

## 注意事项

1. 多GPU训练时，实际的批量大小为`batch_size * num_gpus`，请根据GPU内存调整批量大小
2. 使用混合精度训练可以显著减少GPU内存使用并提高训练速度
3. 如果遇到OOM（内存不足）错误，可以尝试减小批量大小或使用梯度累积
4. 分布式训练需要确保所有节点能够相互通信，并且主节点的IP地址和端口对所有节点可访问