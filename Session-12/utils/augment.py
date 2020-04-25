import numpy as np
from PIL import Image
import random


class Fliplr:
    """
    Flip PIL Image from Left to Right
    """

    def __call__(self, pil):
        if not isinstance(pil, Image.Image):
            raise TypeError(f"Argument must be {Image.Image} type")
        tmp = np.asarray(pil)
        tmp = np.fliplr(tmp)
        return Image.fromarray(tmp)


class Cutout:
    """
    Add Cutout to PIL.Image
    """

    def __init__(self, max_hw, fill_value=0):
        """
        :param max_hw: tuple(max_height, max_width) of cutout area
        :param fill_value: default 0
        """
        self.max_h_size, self.max_w_size = max_hw
        self.fill_value = fill_value

    def __call__(self, pil):
        if not isinstance(pil, Image.Image):
            raise TypeError(f"Argument must be {Image.Image} type")

        tmp = np.asarray(pil).copy()
        height, width = tmp.shape[:2]
        y = random.randint(0, height)
        x = random.randint(0, width)

        y1 = np.clip(y - self.max_h_size // 2, 0, height)
        y2 = np.clip(y1 + self.max_h_size, 0, height)
        x1 = np.clip(x - self.max_w_size // 2, 0, width)
        x2 = np.clip(x1 + self.max_w_size, 0, width)
        tmp[y1:y2, x1:x2] = self.fill_value
        return Image.fromarray(tmp)
