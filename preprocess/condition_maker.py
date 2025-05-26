#!/usr/bin/env python3
# coding: utf-8
"""
visual_condition_dataset_generator.py

Generate visual condition datasets (canny, depth, blur, etc.) from a source dataset
structured as ImageFolder. Outputs maintain the same structure under parallel directories
(e.g., img_canny, img_depth).
Supports multi-threaded acceleration and rich progress reporting.
Usage:
    python visual_condition_dataset_generator.py \
        --input_root /data/ymx/dataset/imagenet-100/img \
        --workers 8 \
        --methods canny depth blur
"""
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image

# Import your preprocessor implementations here
from preprocess.utils.annotator import CannyDetector, HEDdetector, LineartDetector, IllusionConverter
from preprocess.utils.annotator.util import HWC3

# Map method names to (class, init_kwargs)
preprocessor_classes = {
    # "canny": (CannyDetector, dict(low_threshold=50, high_threshold=200)),
    # "hed": (HEDdetector, {}),
    # "lineart": (LineartDetector, dict(coarse=False)),
    "illusion": (IllusionConverter, {})
}


def read_image(path: Path) -> np.ndarray:
    """Read image with PIL, convert to RGB. Handles ICC profiles without libpng warnings."""
    with Image.open(path) as im:
        im_converted = im.convert("RGB")
        return np.array(im_converted)


def process_image(img_path: Path, output_path: Path, processor, init_kwargs: dict):
    """Process a single image and save the result."""
    # Read and convert to RGB
    img_bgr = read_image(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = HWC3(img_rgb)

    # Inference
    with torch.no_grad():
        out = processor(img_rgb, **init_kwargs)
        out = HWC3(out)

    # Save result via PIL
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out).save(output_path)


def generate_for_method(input_root: Path, output_root: Path, method_name: str, workers: int):
    """Generate dataset for a single visual condition method."""
    cls, init_kwargs = preprocessor_classes[method_name]
    processor = cls()
    tasks = []
    # Traverse train/val splits
    for subset in ["train", "val"]:
        subset_dir = input_root / subset
        if not subset_dir.exists():
            print("Skipping subset {}".format(subset_dir))
            continue
        for class_dir in subset_dir.iterdir():
            for img_path in class_dir.iterdir():
                if not img_path.is_file():
                    continue
                if img_path.suffix.lower() not in [".png", ".jpg"]:
                    continue
                rel_path = img_path.relative_to(input_root)
                rel_png = rel_path.with_suffix('.png')
                output_path = output_root / rel_png
                tasks.append((img_path, output_path))

    print(f"[{method_name}] Found {len(tasks)} images under {input_root}")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(process_image, img_path, out_path, processor, init_kwargs): (img_path, out_path)
            for img_path, out_path in tasks
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {method_name}"):
            img_path, out_path = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"[{method_name}] Error processing {img_path}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Generate visual condition datasets.")
    parser.add_argument(
        "--input_root", type=str, default=r"/data/ymx/dataset/imagenet-part/img",
        help="Path to input dataset root (e.g., /data/.../imagenet-100/img)"
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Number of parallel workers (default: CPU core count)"
    )
    parser.add_argument(
        "--methods", nargs="+", default=list(preprocessor_classes.keys()),
        help="List of methods to generate (default: all)"
    )
    args = parser.parse_args()

    normalized_root = args.input_root.replace('\\', '/')
    input_root = Path(normalized_root)
    parent = input_root.parent
    base_name = input_root.name  # e.g., 'img'

    for method_name in args.methods:
        if method_name not in preprocessor_classes:
            print(f"Unknown method: {method_name}. Skipping.")
            continue
        output_root = parent / f"{base_name}_{method_name}"
        output_root.mkdir(parents=True, exist_ok=True)
        generate_for_method(input_root, output_root, method_name, args.workers)


if __name__ == "__main__":
    main()
