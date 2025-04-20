# CAM: 视觉条件对齐模块

## 项目概述

本项目实现了一个轻量级的视觉条件对齐模块（Condition Alignment Module, CAM），用于将不同的视觉条件（如深度图、草图、边缘图等）对齐到目标条件上。这种对齐方式可以使下游生成模型无需为每种视觉条件单独微调，从而实现条件切换与下游模型的解耦。

### 核心特点

- **基于信息瓶颈理论**：利用信息瓶颈理论提取与目标条件相关的关键信息，丢弃无关信息
- **轻量级设计**：模型结构轻量，训练成本低
- **模块化架构**：可以轻松集成到现有的生成模型管道中
- **条件解耦**：实现视觉条件与下游模型的解耦，无需为每种条件重新微调模型

## 理论基础

本项目基于信息瓶颈理论（Information Bottleneck Theory）设计。信息瓶颈理论提供了一个框架，用于在保留与目标相关信息的同时，压缩输入数据中的冗余信息。在视觉条件对齐任务中，我们希望：

1. 从源条件（如草图）中提取与目标条件（如深度图）相关的信息
2. 丢弃源条件中与目标条件无关的信息
3. 生成与目标条件在分布上一致的新表示

通过变分信息瓶颈（Variational Information Bottleneck）的实现，我们的模型可以学习到一个压缩但信息丰富的潜在表示，从而实现不同视觉条件之间的有效对齐。

## 安装指南

### 环境要求

- Python 3.8+
- PyTorch 1.8+
- CUDA（推荐用于加速训练）

### 安装步骤

1. 克隆仓库：

```bash
git clone https://github.com/yourusername/CAM_attn.git
cd CAM_attn
```

2. 安装依赖：

```bash
pip install -r requirements.txt
```

## 使用方法

### 数据准备

数据集应按以下结构组织：

```
/data/ymx/dataset/imagenet-100/
├── img_depth/
│   ├── train/
│   │   ├── class1/
│   │   │   ├── img1.png
│   │   │   ├── img2.png
│   │   │   └── ...
│   │   ├── class2/
│   │   └── ...
│   └── val/
│       └── ...
├── img_canny/
│   ├── train/
│   └── val/
├── img_sketch/
│   ├── train/
│   └── val/
└── ...
```

注意：不同视觉条件下，对应图片的命名应完全相同，只需更换路径即可实现对应。

### 配置

在`config.json`中设置训练参数：

```json
{
    "dataset_path": "/data/ymx/dataset/imagenet-100",
    "target_condition": "depth",
    "source_conditions": ["canny", "sketch"],
    "img_size": 256,
    "batch_size": 32,
    "num_workers": 4,
    "epochs": 100,
    "lr": 2e-4,
    "lr_step": 20,
    "beta": 0.01,
    "seed": 42,
    "device": "cuda",
    "output_dir": "./output",
    "save_interval": 10
}
```

### 训练模型

```bash
python main.py --mode train --config config.json
```

训练过程中，模型会定期保存检查点到`output/checkpoints/`目录，并在`output/samples/`目录生成可视化结果。

### 测试模型

```bash
python main.py --mode test --checkpoint output/checkpoints/best_model.pth --input path/to/input/image.png --output path/to/output/image.png
```

## 模型架构

CAM模型采用了UNet架构，结合先进的注意力机制和信息瓶颈理论，由以下组件组成：

1. **UNet编解码器**：
   - 采用跳跃连接的UNet结构，可配置深度
   - 下采样路径：逐步减小特征图尺寸，增加通道数
   - 上采样路径：逐步恢复特征图尺寸，结合跳跃连接保留细节

2. **信息瓶颈层**：基于变分信息瓶颈理论，提取与目标条件相关的信息
   - 通过均值和方差编码潜在表示
   - 使用KL散度损失实现信息瓶颈约束

3. **多种注意力机制**：
   - CBAM (Convolutional Block Attention Module)：结合通道和空间注意力
   - 自注意力 (Self-Attention)：捕获长距离依赖关系
   - 通道注意力 (Channel Attention)：基于SE-Net设计
   - 空间注意力 (Spatial Attention)：关注图像的空间区域

4. **注意力可视化**：提供热力图形式的注意力可视化功能

模型通过注意力机制筛选重要信息，减少冗余，符合信息瓶颈理论的核心思想。

## 引用

如果您在研究中使用了本项目，请引用以下论文：

```
@article{tishby2000information,
  title={The information bottleneck method},
  author={Tishby, Naftali and Pereira, Fernando C and Bialek, William},
  journal={arXiv preprint physics/0004057},
  year={2000}
}

@article{alemi2016deep,
  title={Deep variational information bottleneck},
  author={Alemi, Alexander A and Fischer, Ian and Dillon, Joshua V and Murphy, Kevin},
  journal={arXiv preprint arXiv:1612.00410},
  year={2016}
}
```

## 许可证

MIT