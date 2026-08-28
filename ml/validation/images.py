"""
Image integrity and format validation for histopathology image files.
"""

import hashlib
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
from PIL import Image, UnidentifiedImageError


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


class ImageValidator:
    """
    Validates file existence, non-emptiness, format integrity, and PIL loadability for image files.
    """

    def __init__(self, supported_extensions: Optional[set] = None):
        self.supported_extensions = supported_extensions or SUPPORTED_EXTENSIONS

    def compute_sha256(self, filepath: Path, chunk_size: int = 65536) -> str:
        """Compute SHA-256 checksum of a file."""
        hasher = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def validate_file(self, filepath: Path) -> Tuple[bool, List[str]]:
        """Validate file-level properties."""
        errors: List[str] = []
        path = Path(filepath)

        if not path.exists():
            return False, [f"File does not exist: {path}"]

        if not path.is_file():
            return False, [f"Path is not a regular file: {path}"]

        if path.suffix.lower() not in self.supported_extensions:
            errors.append(f"Unsupported file extension '{path.suffix.lower()}'. Expected one of {sorted(self.supported_extensions)}")

        try:
            size = path.stat().st_size
            if size == 0:
                errors.append("File is empty (0 bytes).")
        except OSError as e:
            errors.append(f"Unable to read file stat: {e}")

        return len(errors) == 0, errors

    def validate_image(self, filepath: Path) -> Tuple[bool, List[str], Dict[str, Any]]:
        """
        Validate complete image integrity, dimensions, channels, and extract metadata.
        Returns: (is_valid, errors, metadata_dict)
        """
        path = Path(filepath)
        is_file_valid, file_errors = self.validate_file(path)
        if not is_file_valid:
            return False, file_errors, {}

        errors = list(file_errors)
        metadata: Dict[str, Any] = {}

        # 1. Compute file hash and size
        try:
            metadata["file_size_bytes"] = path.stat().st_size
            metadata["file_hash"] = self.compute_sha256(path)
        except Exception as e:
            errors.append(f"Failed to read file for hashing: {e}")
            return False, errors, metadata

        # 2. Open and verify image with PIL
        try:
            with Image.open(path) as img:
                img_format = img.format
                width, height = img.size
                mode = img.mode

                if width <= 0 or height <= 0:
                    errors.append(f"Invalid image dimensions: {width}x{height}")

                # Verify image can be decoded and converted to RGB
                img_rgb = img.convert("RGB")
                img_rgb.load()

                metadata["image_width"] = width
                metadata["image_height"] = height
                metadata["image_format"] = img_format or path.suffix.lstrip(".").upper()
                metadata["image_mode"] = mode
                metadata["channels"] = len(img_rgb.getbands())

        except UnidentifiedImageError:
            errors.append("PIL cannot identify image file format (corrupted or non-image content).")
        except (IOError, SyntaxError, ValueError) as e:
            errors.append(f"Image decode error / corruption: {e}")
        except Exception as e:
            errors.append(f"Unexpected image processing error: {e}")

        is_valid = len(errors) == 0
        return is_valid, errors, metadata

