# CAM+ 多条件对齐模块

## 简介

CAM+（多条件对齐模块）是CAM模型的升级版本，支持多种视觉条件的同时处理，特别是新增了对颜色条件（color）的支持。CAM+采用多编码器单解码器架构，为每种条件配备专用编码器，同时共享一个解码器，从而能够更有效地处理不同类型的视觉条件。

## 架构设计

### 多编码器单解码器架构

CAM+的核心设计理念是为每种视觉条件提供专用的编码器，同时共享一个解码器。这种设计有以下优势：

1. **专业化处理**：每种条件的编码器可以专注于提取该条件特有的特征
2. **灵活性**：可以轻松添加新的条件类型，只需增加相应的编码器
3. **特征融合**：通过特征融合层，将不同条件的特征有效地结合起来
4. **参数共享**：共享解码器减少了模型参数，提高了训练效率

### 颜色条件的特殊处理

CAM+对颜色条件（color）进行了特殊处理，因为颜色条件与其他条件（如边缘、草图等）有显著差异：

- 设计了专用的`ColorEncoder`，包含额外的颜色特征提取层
- 颜色特征提取层使用注意力机制，更好地捕捉颜色分布和关系
- 其他条件（如边缘、草图）使用标准的`StandardEncoder`

### 数据加载器改进

新的`MultiConditionDataset`数据加载器确保每次返回所有条件对应同一图像的样本：

- 同时加载多种视觉条件下的图像
- 每个样本包含所有指定条件下的同一图像
- 自动检查所有条件下是否都有对应的图像

## 使用方法

### 训练模型

```bash
python main_plus.py --mode train --config config_plus.json
```

### 测试模型

```bash
python main_plus.py --mode test --checkpoint ./output_plus/best_model.pth --input_dir ./test_images --output ./output.png
```

### 批量处理

可以通过修改`main_plus.py`调用`batch_process`函数进行批量处理：

```python
batch_process(
    checkpoint_path='./output_plus/best_model.pth',
    source_dir='./source_images',
    target_dir='./target_images/img_depth',
    output_dir='./batch_output',
    config=config
)
```

## 配置参数

主要配置参数（`config_plus.json`）：

- `source_conditions`: 源条件列表，如`["canny", "sketch", "color"]`
- `target_condition`: 目标条件，如`"depth"`
- `contrastive_weight`: 对比学习损失权重
- `feature_matching_weight`: 特征匹配损失权重

## 文件结构

- `main_plus.py`: 主程序文件
- `model_plus.py`: 模型定义文件
- `data_loader_plus.py`: 数据加载器文件
- `config_plus.json`: 配置文件

## 注意事项

1. 确保数据集目录结构正确，每种条件应有对应的子目录，如：
   - `img_canny/`
   - `img_sketch/`
   - `img_color/`
   - `img_depth/`

2. 训练时会自动检查所有条件下是否都有对应的图像，只有在所有条件下都存在的图像才会被用于训练

3. 对于颜色条件，建议使用RGB图像，以便充分利用颜色编码器的特性