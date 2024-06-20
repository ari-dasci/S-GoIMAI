
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont
import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import random
import numpy as np
import cv2
import inspect


class Compose:
    """
    Apply a list of transformation techniques to an image.

    - aug_list: list that contains the transformations

    Returns an image.
    """

    def __init__(self, aug_list):

        self.aug_list = aug_list


    def __call__(self, img):

        nimg = img

        for aug in self.aug_list:
            nimg = aug(nimg)

        return nimg

    def __str__(self):

        return "\n".join(str(aug) for aug in self.aug_list)



class Base:

    def __init__(self, p: float = 1.0):
        """
        @param p: the probability of the transformation been applied
        """
        self.p = p

    def __call__(self, image: Image.Image):
        """
        @param image: PIL image to which the transformation is applied.
        """
        
        # image must be an instance of PIL
        assert isinstance(image, Image.Image)

        if random.random() > self.p:
            return image

        return self.apply(image)

    def apply(self, image: Image.Image):

        raise NotImplementedError()

    def __str__(self):

        class_name = self.__class__.__name__

        # Get the names and values of parameters in __init__
        init_params = inspect.signature(self.__init__).parameters
        params_str = ', '.join(f'{param}={getattr(self, param)}' for param in init_params if param != 'self')

        return f"{class_name}({params_str})"


class RandomBrightness(Base):

    def __init__(self, min_factor, max_factor, p: float = 1.0):

        self.min_factor = min_factor
        self.max_factor = max_factor

        super().__init__(p)

    def apply(self, image):

        factor = random.uniform(self.min_factor, self.max_factor)

        return ImageEnhance.Brightness(image).enhance(factor)

class Resize(Base):

    def __init__(self, w, h):

        self.w = w
        self.h = h

        super().__init__(1.0)

    def apply(self, image):

        return image.resize((self.w, self.h))

class RandomCrop(Base):

    def __init__(self, w, h, p: float = 1.0):
        self.w = w
        self.h = h
        super().__init__(p)

    def apply(
            self,
            image: Image.Image,
    ):
        w, h = image.size
        x = random.randint(0, w - self.w)
        y = random.randint(0, h - self.h)
        return image.crop((x, y, x + self.w, y + self.h))

class RandomFlip(Base):

    def __init__(self, p: float = 1.0):

        super().__init__(p)

    def apply(self, image):

        aug_image = image
        if random.random() > 0.5:
           aug_image = aug_image.transpose(Image.FLIP_LEFT_RIGHT)
        
        if random.random() > 0.5:
           aug_image = aug_image.transpose(Image.FLIP_TOP_BOTTOM)

        return aug_image

class RandomZoom(Base):

    def __init__(self, height, p: float = 1.0):

        self.height = height
        super().__init__(p)

    def apply(self, image):

        factor = random.uniform(self.height[0], self.height[1])
        
        w, h = image.size
        new_w = int(w * factor)
        new_h = int(h * factor)

        return image.resize((new_w, new_h))

class RandomRotation(Base):

    def __init__(
        self, min_degrees: float = 0.0, max_degrees: float = 180.0, p: float = 1.0
    ):
        """
        @param min_degrees: the lower value on the range of degree values to choose from

        @param max_degrees: the upper value on the range of degree values to choose from

        @param p: the probability of the transform being applied; default value is 1.0
        """
        self.min_degrees = min_degrees
        self.max_degrees = max_degrees
        self.p = p
        super().__init__(p)

    def apply(self, image: Image.Image):

        self.choosen_degree = random.uniform(self.min_degrees, self.max_degrees)

        rotated_image = image.rotate(self.choosen_degree, expand=True)

        center_x, center_y = rotated_image.width / 2, rotated_image.height / 2
        wr, hr = self.rotated_rect_with_max_area(image.width, image.height, self.choosen_degree)
        aug_image = rotated_image.crop(
            (
                int(center_x - wr / 2),
                int(center_y - hr / 2),
                int(center_x + wr / 2),
                int(center_y + hr / 2),
            )
        )

        return aug_image


    def rotated_rect_with_max_area(self, w: int, h: int, angle: float):

        """
         Computes the width and height of the largest possible axis-aligned
         rectangle (maximal area) within the rotated rectangle

         source:
         https://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders # noqa: B950
        """
        width_is_longer = w >= h
        side_long, side_short = (w, h) if width_is_longer else (h, w)

        sin_a = abs(math.sin(math.radians(angle)))
        cos_a = abs(math.cos(math.radians(angle)))
        if side_short <= 2.0 * sin_a * cos_a * side_long or abs(sin_a - cos_a) < 1e-10:
            x = 0.5 * side_short
            wr, hr = (x / sin_a, x / cos_a) if width_is_longer else (x / cos_a, x / sin_a)
        else:
            cos_2a = cos_a * cos_a - sin_a * sin_a
            wr = (w * cos_a - h * sin_a) / cos_2a
            hr = (h * cos_a - w * sin_a) / cos_2a
        return wr, hr
