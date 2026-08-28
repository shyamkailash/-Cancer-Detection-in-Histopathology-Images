"""
Image Quality and Artifact Assessment for Histopathology Images.
"""

from typing import Dict, Any, Union
import numpy as np
from PIL import Image


class ImageQualityChecker:
    """
    Assesses histopathology image quality metrics including whitespace/background ratio,
    blurriness (Laplacian variance), and contrast.
    """

    def calculate_whitespace_ratio(self, img: Image.Image, threshold: int = 220) -> float:
        """
        Calculate fraction of pixels that represent empty slide glass (bright background).
        In H&E staining, glass background appears nearly white (high RGB values).
        """
        rgb_img = img.convert("RGB")
        arr = np.array(rgb_img, dtype=np.uint8)
        # Pixel is white background if all R, G, B channels exceed threshold
        white_pixels = np.all(arr > threshold, axis=2)
        total_pixels = arr.shape[0] * arr.shape[1]
        return float(np.sum(white_pixels) / total_pixels) if total_pixels > 0 else 0.0

    def calculate_blurriness(self, img: Image.Image) -> float:
        """
        Calculate blurriness metric using the variance of the Laplacian operator on grayscale image.
        Higher variance indicates sharper edges/focus; low variance suggests blurriness or empty fields.
        """
        gray = np.array(img.convert("L"), dtype=np.float32)

        # 3x3 Discrete Laplacian Kernel:
        # [ 0,  1,  0]
        # [ 1, -4,  1]
        # [ 0,  1,  0]
        padded = np.pad(gray, pad_width=1, mode="edge")
        laplacian = (
            padded[0:-2, 1:-1] +
            padded[2:, 1:-1] +
            padded[1:-1, 0:-2] +
            padded[1:-1, 2:] -
            4.0 * gray
        )
        return float(np.var(laplacian))

    def calculate_contrast(self, img: Image.Image) -> float:
        """Calculate standard deviation of pixel intensities as a proxy for contrast."""
        gray = np.array(img.convert("L"), dtype=np.float32)
        return float(np.std(gray))

    def assess_quality(self, img: Image.Image) -> Dict[str, Any]:
        """Run complete quality assessment on an image."""
        ws_ratio = self.calculate_whitespace_ratio(img)
        blur = self.calculate_blurriness(img)
        contrast = self.calculate_contrast(img)

        # Heuristic quality flags
        is_mostly_background = ws_ratio > 0.85
        is_low_contrast = contrast < 10.0

        return {
            "whitespace_ratio": round(ws_ratio, 4),
            "blurriness_laplacian_var": round(blur, 2),
            "contrast_std": round(contrast, 2),
            "is_mostly_background": is_mostly_background,
            "is_low_contrast": is_low_contrast,
        }
