# CAM+ 模型增强版

本项目实现了一个增强版的条件对齐模块(CAM+)，用于多条件图像生成任务。最近的更新包括将三通道图像转换为单通道，并修改模型以适应不同通道数的输入。

## 最新更新

1. **图像转换工具**：新增了将三通道图像(sketch、canny、depth)转换为单通道的工具，保持图像信息的同时减少存储空间和计算量。
2. **模型适配**：修改了CAM+模型以适应不同通道数的输入（color为3通道，sketch/canny/depth为1通道）。
3. **输出优化**：将模型输出统一为单通道深度图。
4. **GPU优化**：优化了模型训练过程，确保所有计算都在GPU上进行，减少CPU计算量。

## 文件说明

- `convert_to_grayscale.py`: 将三通道图像转换为单通道的核心脚本
- `convert_images.py`: 图像转换的主脚本，提供命令行接口
- `model_plus.py`: 修改后的CAM+模型，支持不同通道数的输入
- `data_loader_plus.py`: 修改后的数据加载器，适应新的通道配置

## 使用方法

### 1. 图像转换

将三通道图像转换为单通道：

```bash
python convert_images.py --data_dir 数据集路径 --conditions sketch canny depth --convert_only
```

参数说明：
- `--data_dir`: 数据集根目录
- `--conditions`: 需要转换的条件类型，默认为sketch、canny和depth
- `--convert_only`: 仅执行图像转换，不训练模型

### 2. 模型训练

训练修改后的CAM+模型：

```python
config = {
    'dataset_path': '数据集路径',
    'target_condition': 'depth',
    'source_conditions': ['canny', 'sketch', 'color'],
    'img_size': 256,
    'batch_size': 16,
    'num_workers': 4,
    'lr': 0.0002,
    'beta': 0.01,
    'epochs': 100,
    'lr_step': 30,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'output_dir': './output_plus'
}

from model_plus import train_model_plus
train_model_plus(config)
```

## 数据集结构

数据集应按以下结构组织：

```
数据集根目录/
├── img_canny/
│   ├── train/
│   │   ├── 类别1/
│   │   │   ├── 图像1.png
│   │   │   └── ...
│   │   └── ...
│   └── val/
│       └── ...
├── img_sketch/
│   └── ...
├── img_color/
│   └── ...
└── img_depth/
    └── ...
```

## 技术说明

1. **通道处理**：
   - color条件：保持3通道RGB格式
   - sketch/canny/depth条件：转换为1通道灰度格式
   - 输出：统一为1通道深度图

2. **GPU优化**：
   - 启用cudnn.benchmark加速卷积操作
   - 使用pin_memory和适当的num_workers优化数据加载
   - 确保所有张量操作都在GPU上进行

3. **数据归一化**：
   - RGB图像：使用[0.5, 0.5, 0.5]均值和标准差
   - 灰度图像：使用[0.5]均值和标准差