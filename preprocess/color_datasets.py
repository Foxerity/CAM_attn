import os
from PIL import Image


def generate_color_condition_dataset(src_root, dst_root, low_res=(16, 16), final_res=(256, 256)):
    """
    Generate a pixelated 'color' condition dataset from an existing ImageFolder-structured dataset.

    Args:
        src_root (str): Path to the source dataset root, e.g., 'B:/datasets/test/img'
        dst_root (str): Path to the destination dataset root, e.g., 'B:/datasets/test/img_color'
        low_res (tuple): Resolution to downscale to for pixelation, e.g., (16,16)
        final_res (tuple): Final resolution to upscale back to, e.g., (256,256)
    """
    splits = ['train', 'val']
    src_split = os.path.join(src_root, 'train')
    for split in splits:
        dst_split = os.path.join(dst_root, split)
        if not os.path.isdir(src_split):
            print(f"Warning: Source split directory not found: {src_split}")
            continue

        for class_name in os.listdir(dst_split):
            src_class_dir = os.path.join(src_split, class_name)
            dst_class_dir = os.path.join(dst_split, class_name)
            if not os.path.isdir(src_class_dir):
                continue
            os.makedirs(dst_class_dir, exist_ok=True)

            for fname in os.listdir(src_class_dir):
                if not fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                    continue

                src_path = os.path.join(src_class_dir, fname)
                dst_path = os.path.join(dst_class_dir, fname)

                img = Image.open(src_path)

                # 1. 中心裁剪为正方形
                w, h = img.size
                min_dim = min(w, h)
                left = (w - min_dim) // 2
                top = (h - min_dim) // 2
                img_cropped = img.crop((left, top, left + min_dim, top + min_dim))

                # 2. 缩小到低分辨率（像素化）
                img_small = img_cropped.resize(low_res, Image.Resampling.NEAREST)

                # 3. 放大到目标尺寸，使用最近邻保持块感
                img_pixelated = img_small.resize(final_res, Image.Resampling.NEAREST)

                img_pixelated.save(dst_path)
        print(f"Finished processing split: {split}")


if __name__ == "__main__":
    src_root = r"/data/ymx/dataset/imagenet-part/img"
    dst_root = r"/data/ymx/dataset/imagenet-part/img_color"
    # 你可以根据需要调整 low_res 和 final_res
    generate_color_condition_dataset(src_root, dst_root, low_res=(24, 24), final_res=(256, 256))
    print("Dataset generation complete!")

