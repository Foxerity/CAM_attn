"""
Adapted from https://gitlab.com/-/snippets/3611640
Reference: https://civitai.com/models/137638/controlnet-mysee-light-and-dark-squint-illusions-hidden-symbols-subliminal-text-qr-codes
"""

import cv2
import numpy as np

class IllusionConverter:
    def __init__(self):
        """No configuration here—全在 __call__ 里传入。"""
        pass

    def __call__(
        self,
        img: np.ndarray,
        segments: int = 7,
        large_kernel_size: tuple[int, int] = (7, 7),
        small_kernel_size: tuple[int, int] = (3, 3),
    ) -> np.ndarray:
        """
        :param img:               输入 RGB 图，dtype uint8，shape (H, W, 3)
        :param segments:          灰度分段数，>=2
        :param large_kernel_size: 闭运算(填洞)核大小
        :param small_kernel_size: 开运算(去噪)核大小
        :return:                  输出 RGB 图，shape (H, W, 3)，dtype uint8
        """
        # 1) 提取 V 通道做灰度
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        gray = hsv[:, :, 2]

        # 2) 生成 levels（线性等距灰度值）
        levels = np.linspace(0, 255, segments, dtype=np.uint8)

        # 3) 将 0–255 灰度映射到 [0, segments-1]
        idx = (gray.astype(np.int32) * segments) // 256
        remap = levels[idx]

        # 4) 形态学核
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, large_kernel_size)
        open_kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, small_kernel_size)

        # 5) 先闭后开
        out = cv2.morphologyEx(remap, cv2.MORPH_CLOSE, close_kernel)
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN,  open_kernel)

        kernel_sharp = np.array([[0, -1, 0],
                                 [-1, 5, -1],
                                 [0, -1, 0]])
        out = cv2.filter2D(out, -1, kernel_sharp)

        # 6) 单通道→三通道
        # return cv2.cvtColor(out[:, :, None], cv2.COLOR_GRAY2RGB)
        return out

