import os
import glob
from PIL import Image
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def visualize_tsne(folder_dict,
                   max_images_per_folder=None,
                   image_size=(64, 64),
                   tsne_perplexity=30,
                   tsne_learning_rate=200,
                   random_state=42):
    """
    对多个文件夹里的图片做 t-SNE 可视化。

    参数：
    - folder_dict: Dict[str, str]，键为标签名，值为对应的文件夹路径。
    - max_images_per_folder: int or None，最多读取每个文件夹的图片数量（None 则读取全部）。
    - image_size: (w, h)，统一缩放后再拉平。
    - tsne_perplexity, tsne_learning_rate, random_state: 传给 TSNE 的参数。
    """
    X, y = [], []
    labels = list(folder_dict.keys())  # 保持键的顺序

    for idx, label in enumerate(labels):
        folder = folder_dict[label]
        # 收集该文件夹下所有常见图片文件
        exts = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif')
        all_files = []
        for ext in exts:
            all_files.extend(glob.glob(os.path.join(folder, ext)))
        if max_images_per_folder is not None:
            all_files = all_files[:max_images_per_folder]

        for fpath in all_files:
            img = Image.open(fpath).convert('RGB')  # 强制三通道
            img = img.resize(image_size, Image.BILINEAR)
            arr = np.asarray(img, dtype=np.float32) / 255.0  # 归一化
            X.append(arr.ravel())  # 拉平
            y.append(idx)

    X = np.stack(X, axis=0)
    y = np.array(y)

    # 运行 t-SNE
    tsne = TSNE(n_components=2,
                perplexity=tsne_perplexity,
                learning_rate=tsne_learning_rate,
                random_state=random_state)
    X2 = tsne.fit_transform(X)

    # 绘图
    plt.figure(figsize=(8, 6))
    for idx, label in enumerate(labels):
        mask = (y == idx)
        plt.scatter(
            X2[mask, 0], X2[mask, 1],
            label=label,
            alpha=0.7,
            s=10
        )
    plt.legend(loc='best', fontsize='small')
    plt.title('t-SNE visualization of image folders')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.tight_layout()
    plt.savefig("test.png")


if __name__ == '__main__':
    # 使用字典，键为你想要显示的标签名称
    folders = {
        'canny':  '/data/ymx/dataset/imagenet-part/img_canny/val/1',
        'depth':  '/data/ymx/dataset/imagenet-part/img_depth/val/1',
        'color':  '/data/ymx/dataset/imagenet-part/img_color/val/1',
        'sketch': '/data/ymx/dataset/imagenet-part/img_sketch/val/1',
        'hed':    '/data/ymx/dataset/imagenet-part/img_hed/val/1',
        'illusion': '/data/ymx/dataset/imagenet-part/img_illusion/val/1',
        'lineart':   '/data/ymx/dataset/imagenet-part/img_lineart/val/1'
    }

    visualize_tsne(
        folder_dict=folders,
        max_images_per_folder=100,  # 最多每类读 100 张
        image_size=(128, 128),        # 缩放到 64×64
        tsne_perplexity=30,
        tsne_learning_rate=60,
        random_state=123
    )
