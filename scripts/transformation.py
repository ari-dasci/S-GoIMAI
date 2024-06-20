
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont
import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import random
import numpy as np
import cv2
import inspect


class Compose:
    """
    Sequential container for applying multiple image transformations in order.
    
    Args:
        aug_list (list[Base]): Ordered list of transformation objects to apply
    
    Raises:
        TypeError: If aug_list contains non-transformation objects
        TypeError: If input image is not a PIL.Image
    
    Example:
        >>> transforms = Compose([RandomFlip(), RandomRotation(-30, 30)])
        >>> augmented_img = transforms(original_img)
    """

    def __init__(self, aug_list):
        if not isinstance(aug_list, list):
            raise TypeError("aug_list must be a list of transformation objects")
        self.aug_list = aug_list

    def __call__(self, img: Image.Image) -> Image.Image:
        """Apply transformations sequentially to input image"""
        if not isinstance(img, Image.Image):
            raise TypeError("Input must be a PIL.Image object")
            
        transformed_img = img
        for transform in self.aug_list:
            transformed_img = transform(transformed_img)
        return transformed_img

    def __str__(self):
        return "\n".join(str(aug) for aug in self.aug_list)



class Base:
    """Abstract base class for all image transformation operations.
    
    Provides probability control and common interface for transformations.
    
    Args:
        p (float): Probability [0.0-1.0] of applying the transformation.
                  Default=1.0 (always apply)
    
    Raises:
        AssertionError: If input is not a PIL.Image
    """

    def __init__(self, p: float = 1.0):
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
    """
    Randomly adjusts image brightness within specified range.
    
    Args:
        min_factor (float): Minimum brightness multiplier (>=0.0, <1.0 darkens)
        max_factor (float): Maximum brightness multiplier (>1.0 brightens)
        p (float): Application probability
    
    Example:
        >>> RandomBrightness(0.8, 1.2)  # 80-120% brightness adjustment
    """

    def __init__(self, min_factor, max_factor, p: float = 1.0):

        self.min_factor = min_factor
        self.max_factor = max_factor

        super().__init__(p)

    def apply(self, image):

        factor = random.uniform(self.min_factor, self.max_factor)

        return ImageEnhance.Brightness(image).enhance(factor)

class Resize(Base):
    """
    Resizes image to exact dimensions with high-quality filtering.
    
    Args:
        w (int): Target width in pixels (>0)
        h (int): Target height in pixels (>0)
    
    Notes:
        - Always applies (probability locked to 1.0)
        - Uses Lanczos resampling for high-quality results
    """

    def __init__(self, w, h):

        self.w = w
        self.h = h

        super().__init__(1.0)

    def apply(self, image):

        return image.resize((self.w, self.h))

class RandomCrop(Base):
    """
    Randomly crops image to specified dimensions.
    
    Args:
        w (int): Crop width in pixels (<= original width)
        h (int): Crop height in pixels (<= original height)
        p (float): Application probability
    
    Raises:
        ValueError: If crop size exceeds original dimensions
    """

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
    """
    Applies random horizontal and/or vertical flips.
    
    Features:
        - 50% chance for horizontal flip
        - 50% chance for vertical flip
        - Flips can combine independently
    
    Args:
        p (float): Application probability
    """

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
    """
    Applies random scaling (zoom) within specified range.
    
    Args:
        height (tuple[float, float]): Min/max scale factors (e.g. (0.8, 1.2))
        p (float): Application probability
    
    Notes:
        - Values <1.0 zoom out (shrink image)
        - Values >1.0 zoom in (enlarge image)
        - Maintains aspect ratio during scaling
    """

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
    """
    Applies random rotation with automatic border cropping.
    
    Args:
        min_degrees (float): Minimum rotation angle (-180 to 180)
        max_degrees (float): Maximum rotation angle (-180 to 180)
        p (float): Application probability
    
    Features:
        - Rotates image without black borders
        - Maintains original image center
        - Automatically crops to largest possible area
    
    Example:
        >>> RandomRotation(-45, 45)  # Rotate between -45° and +45°
    """

    def __init__(
        self, min_degrees: float = 0.0, max_degrees: float = 180.0, p: float = 1.0
    ):
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
        Calculates maximum axis-aligned rectangle after rotation.
        
        Args:
            w (int): Original width
            h (int): Original height
            angle (float): Rotation angle in degrees
            
        Returns:
            tuple[float, float]: (width, height) of largest possible rectangle
        
        Note: Adapted from proven mathematical solution to remove black borders
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
