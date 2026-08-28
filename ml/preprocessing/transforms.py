"""
Histopathology image preprocessing and training-time data augmentation pipeline.
"""

import random
from typing import Tuple, List, Optional, Union
import numpy as np
from PIL import Image, ImageEnhance


class Transform:
    """Base class for all image transformations."""
    def __call__(self, img: Union[Image.Image, np.ndarray]) -> Union[Image.Image, np.ndarray]:
        raise NotImplementedError


class Compose(Transform):
    """Composes several transforms together in sequence."""

    def __init__(self, transforms: List[Transform]):
        self.transforms = transforms

    def __call__(self, img: Union[Image.Image, np.ndarray]) -> Union[Image.Image, np.ndarray]:
        for t in self.transforms:
            img = t(img)
        return img


class ColorModeTransform(Transform):
    """Ensure image is in specified color mode (e.g. 'RGB')."""

    def __init__(self, mode: str = "RGB"):
        self.mode = mode

    def __call__(self, img: Image.Image) -> Image.Image:
        if isinstance(img, Image.Image):
            if img.mode != self.mode:
                return img.convert(self.mode)
            return img
        return img


class ResizeTransform(Transform):
    """Resize image to target (width, height)."""

    def __init__(self, size: Tuple[int, int] = (96, 96), resample=Image.Resampling.BILINEAR):
        self.size = size
        self.resample = resample

    def __call__(self, img: Image.Image) -> Image.Image:
        if isinstance(img, Image.Image):
            if img.size != self.size:
                return img.resize(self.size, resample=self.resample)
            return img
        elif isinstance(img, np.ndarray):
            # Convert ndarray to PIL, resize, and return
            pil_img = Image.fromarray(img)
            return np.array(pil_img.resize(self.size, resample=self.resample))
        return img


class RandomHorizontalFlip(Transform):
    """Randomly flip image horizontally with probability p."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            if isinstance(img, Image.Image):
                return img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            elif isinstance(img, np.ndarray):
                return np.fliplr(img)
        return img


class RandomVerticalFlip(Transform):
    """Randomly flip image vertically with probability p."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            if isinstance(img, Image.Image):
                return img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            elif isinstance(img, np.ndarray):
                return np.flipud(img)
        return img


class RandomRotation90(Transform):
    """Randomly rotate image by 0, 90, 180, or 270 degrees (H&E pathology orientation invariance)."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            k = random.choice([1, 2, 3])  # 90, 180, or 270 degrees
            rotations = {
                1: Image.Transpose.ROTATE_90,
                2: Image.Transpose.ROTATE_180,
                3: Image.Transpose.ROTATE_270,
            }
            if isinstance(img, Image.Image):
                return img.transpose(rotations[k])
            elif isinstance(img, np.ndarray):
                return np.rot90(img, k)
        return img


class ColorJitter(Transform):
    """Subtle brightness and contrast jitter without compromising tissue morphology."""

    def __init__(self, brightness: float = 0.1, contrast: float = 0.1):
        self.brightness = brightness
        self.contrast = contrast

    def __call__(self, img: Image.Image) -> Image.Image:
        if not isinstance(img, Image.Image):
            return img

        if self.brightness > 0:
            factor = 1.0 + random.uniform(-self.brightness, self.brightness)
            img = ImageEnhance.Brightness(img).enhance(factor)

        if self.contrast > 0:
            factor = 1.0 + random.uniform(-self.contrast, self.contrast)
            img = ImageEnhance.Contrast(img).enhance(factor)

        return img


class ToTensorTransform(Transform):
    """
    Converts PIL Image to float32 NumPy array with shape (Channels, Height, Width)
    and pixel intensities normalized to [0.0, 1.0].
    """

    def __call__(self, img: Image.Image) -> np.ndarray:
        if isinstance(img, Image.Image):
            arr = np.array(img, dtype=np.float32) / 255.0
            if arr.ndim == 2:
                arr = np.expand_dims(arr, axis=-1)
            # Transpose (H, W, C) -> (C, H, W)
            return np.transpose(arr, (2, 0, 1))
        elif isinstance(img, np.ndarray):
            arr = img.astype(np.float32)
            if arr.max() > 1.0:
                arr = arr / 255.0
            if arr.ndim == 3 and arr.shape[2] in (1, 3, 4):
                return np.transpose(arr, (2, 0, 1))
            return arr
        return img


class NormalizeTransform(Transform):
    """
    Normalizes a (C, H, W) float array with given mean and std per channel.
    Standard ImageNet defaults: mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225).
    """

    def __init__(
        self,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        self.mean = np.array(mean, dtype=np.float32).reshape(-1, 1, 1)
        self.std = np.array(std, dtype=np.float32).reshape(-1, 1, 1)

    def __call__(self, tensor: np.ndarray) -> np.ndarray:
        if isinstance(tensor, np.ndarray) and tensor.ndim == 3:
            return (tensor - self.mean) / self.std
        return tensor


def get_train_transforms(
    target_size: Tuple[int, int] = (96, 96),
    color_jitter: bool = True,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> Compose:
    """Factory function for training augmentation pipeline."""
    transforms: List[Transform] = [
        ColorModeTransform("RGB"),
        ResizeTransform(target_size),
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.5),
        RandomRotation90(p=0.5),
    ]
    if color_jitter:
        transforms.append(ColorJitter(brightness=0.1, contrast=0.1))

    transforms.extend([
        ToTensorTransform(),
        NormalizeTransform(mean=mean, std=std),
    ])
    return Compose(transforms)


def get_eval_transforms(
    target_size: Tuple[int, int] = (96, 96),
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> Compose:
    """Factory function for deterministic validation/test preprocessing pipeline."""
    return Compose([
        ColorModeTransform("RGB"),
        ResizeTransform(target_size),
        ToTensorTransform(),
        NormalizeTransform(mean=mean, std=std),
    ])

