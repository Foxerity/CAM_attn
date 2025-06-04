import os
import shutil
import random
from pathlib import Path
from typing import List, Dict

def create_per_class_val_sets(dataset_roots: List[Path],
                              num_samples_per_class: int = 50,
                              seed: int = None,
                              valid_exts: List[str] = None) -> None:
    """
    为每个类别从 train 中抽取 num_samples_per_class 张图片到 val，
    并确保这些图片在所有条件下都存在（允许扩展名不同）。
    """
    if seed is not None:
        random.seed(seed)

    if valid_exts is None:
        valid_exts = ['.png', '.jpg', '.jpeg']

    # 用于检查某 root/train/<cls>/<stem>.* 是否存在
    def find_image(root: Path, cls: str, stem: str) -> Path:
        base = root / "train" / cls
        for ext in valid_exts:
            p = base / (stem + ext)
            if p.exists():
                return p
        return None

    first_root = dataset_roots[0]
    train0 = first_root / "train"

    # 1. 对于每个类别，找出公共可用的 stem 列表
    stems_per_class: Dict[str, List[str]] = {}
    for cls_dir in sorted(train0.iterdir()):
        if not cls_dir.is_dir():
            continue
        cls_name = cls_dir.name

        # 所有文件的 stem
        candidate_stems = [p.stem for p in cls_dir.iterdir() if p.is_file()]
        # 过滤：只保留在所有 root 下都存在的 stem
        common_stems = []
        for stem in candidate_stems:
            if all(find_image(r, cls_name, stem) for r in dataset_roots):
                common_stems.append(stem)

        if len(common_stems) < num_samples_per_class:
            raise ValueError(
                f"类别 `{cls_name}` 可用图片只有 {len(common_stems)} 张，"
                f"少于 {num_samples_per_class} 张。"
            )
        stems_per_class[cls_name] = common_stems

    # 2. 每个类别随机抽样
    sampled_per_class: Dict[str, List[str]] = {}
    for cls_name, stems in stems_per_class.items():
        sampled = random.sample(stems, num_samples_per_class)
        sampled_per_class[cls_name] = sampled
        print(f"类别 `{cls_name}`：抽取 {len(sampled)} 张")

    # 3. 清空并复制到 val
    for root in dataset_roots:
        val_dir = root / "val"
        # 清空旧目录
        if val_dir.exists():
            shutil.rmtree(val_dir)
        val_dir.mkdir(parents=True, exist_ok=True)

        # 复制每个类别的样本
        for cls_name, stems in sampled_per_class.items():
            for stem in stems:
                src = find_image(root, cls_name, stem)
                dst = val_dir / cls_name / src.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
        print(f"→ `{root}`: 已复制 {num_samples_per_class}×{len(sampled_per_class)} 张到 `{val_dir}`")

if __name__ == "__main__":
    # 示例根目录列表
    dataset_paths = [
        Path("/data/ymx/dataset/imagenet-part/img_depth"),
        Path("/data/ymx/dataset/imagenet-part/img_canny"),
        Path("/data/ymx/dataset/imagenet-part/img_sketch"),
        Path("/data/ymx/dataset/imagenet-part/img_hed"),
        Path("/data/ymx/dataset/imagenet-part/img_illusion"),
        Path("/data/ymx/dataset/imagenet-part/img_lineart"),
        Path("/data/ymx/dataset/imagenet-part/img_color"),
        # …如有更多条件，再添加
    ]

    create_per_class_val_sets(
        dataset_paths,
        num_samples_per_class=50,
        seed=42,
        valid_exts=['.png', '.jpg']
    )