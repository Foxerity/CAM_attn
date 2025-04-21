# CAM模型评估工具

本目录包含用于评估CAM模型（条件对齐模块）图像生成质量的各种指标计算工具。

## 功能特点

- 支持FID（Fréchet Inception Distance）评估
- 支持LPIPS（Learned Perceptual Image Patch Similarity）评估
- 支持PSNR（峰值信噪比）和SSIM（结构相似性）评估
- 自动批量处理所有类别的图像
- 详细的类别级别指标统计
- 支持标准CAM模型和增强版CAM模型
- 一键运行评估流程
- 自动保存生成图像和目标图像，便于直观比较

## 使用方法

### 一键评估（推荐）

使用`evaluate.py`脚本可以一键运行所有评估指标：

```bash
python evaluation/evaluate.py \
    --checkpoint path/to/model_checkpoint.pth \
    --config config.json \
    --source_condition canny \
    --output_dir ./evaluation_results \
    --metrics all
```

### 参数说明

- `--checkpoint`: 模型检查点路径（必需）
- `--config`: 配置文件路径（默认：config.json）
- `--source_condition`: 源条件类型，如canny、sketch等（必需）
- `--target_condition`: 目标条件类型，如不指定则使用配置文件中的target_condition
- `--output_dir`: 输出目录（默认：./evaluation_results）
- `--enhanced`: 是否使用增强版模型（默认：False）
- `--metrics`: 要计算的指标，可选：fid, lpips, all（默认：all）
- `--batch_size`: 批量大小（默认：32）

### 单独运行FID评估

```bash
python evaluation/fid.py \
    --checkpoint path/to/model_checkpoint.pth \
    --config config.json \
    --source_dir path/to/source_images \
    --target_dir path/to/target_images \
    --output_dir ./evaluation_results/fid
```

### 单独运行LPIPS评估

```bash
python evaluation/lpips.py \
    --checkpoint path/to/model_checkpoint.pth \
    --config config.json \
    --source_dir path/to/source_images \
    --target_dir path/to/target_images \
    --output_dir ./evaluation_results/lpips
```

## 输出结果

评估工具会生成以下输出：

1. 生成的图像（保存在`output_dir/generated/`目录下）
2. 目标图像（保存在`output_dir/target/`目录下）
3. 评估指标结果（JSON格式）
4. 可视化图表（各类别指标对比）

## 指标说明

- **FID**: 衡量生成图像与真实图像分布之间的距离，值越小越好
- **gFID**: 几何平均FID，对各类别FID取几何平均值，减少极端值影响
- **LPIPS**: 感知相似度度量，值越小表示感知上越相似
- **PSNR**: 峰值信噪比，值越大表示图像质量越好
- **SSIM**: 结构相似性，值越接近1表示结构越相似

## 依赖库

- torch
- torchvision
- numpy
- scipy
- PIL
- matplotlib
- tqdm
- lpips (用于LPIPS评估，可通过`pip install lpips`安装)

## 注意事项

- 评估过程可能需要较长时间，特别是对于大型数据集
- FID计算需要足够数量的图像（建议至少100张）才能得到可靠结果
- 评估结果会保存在指定的输出目录中，可以随时查看
- 生成的图像和评估结果可能占用较大磁盘空间，已在.gitignore中排除