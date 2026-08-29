"""
Grad-CAM (Gradient-weighted Class Activation Mapping) and Heatmap Overlay for Histopathology Images.
Provides visual explainability for metastasis predictions without external C dependencies.
"""

import base64
import io
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from PIL import Image


def apply_jet_colormap(normalized_map: np.ndarray) -> np.ndarray:
    """
    Apply standard JET colormap to a 2D float array in range [0, 1].
    Returns uint8 RGB array of shape (H, W, 3).
    """
    x = np.clip(normalized_map, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255.0).astype(np.uint8)


def generate_heatmap_overlay(
    original_image: Image.Image,
    cam_map: np.ndarray,
    alpha: float = 0.5,
) -> Tuple[Image.Image, Image.Image]:
    """
    Generate standalone heatmap image and blended overlay with the original histopathology image.
    Args:
        original_image: PIL RGB image
        cam_map: 2D float numpy array in range [0, 1]
        alpha: Blending factor (0.0 = only original, 1.0 = only heatmap)
    Returns:
        (heatmap_image, overlay_image) as PIL RGB Images
    """
    orig_rgb = original_image.convert("RGB")
    width, height = orig_rgb.size

    # Resize CAM map to match original image dimensions
    cam_img = Image.fromarray((cam_map * 255.0).astype(np.uint8), mode="L")
    cam_resized = cam_img.resize((width, height), resample=Image.BILINEAR)
    cam_arr = np.array(cam_resized, dtype=np.float32) / 255.0

    # Colorize
    heatmap_arr = apply_jet_colormap(cam_arr)
    heatmap_pil = Image.fromarray(heatmap_arr, mode="RGB")

    # Blend with original histopathology patch
    orig_arr = np.array(orig_rgb, dtype=np.float32)
    overlay_arr = (1.0 - alpha) * orig_arr + alpha * heatmap_arr.astype(np.float32)
    overlay_arr = np.clip(overlay_arr, 0.0, 255.0).astype(np.uint8)
    overlay_pil = Image.fromarray(overlay_arr, mode="RGB")

    return heatmap_pil, overlay_pil


def pil_to_base64(image: Image.Image, image_format: str = "PNG") -> str:
    """Convert PIL image to base64 data URI string."""
    buffer = io.BytesIO()
    image.save(buffer, format=image_format)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/{image_format.lower()};base64,{encoded}"


def base64_to_pil(base64_str: str) -> Image.Image:
    """Convert base64 data URI string or raw base64 to PIL Image."""
    if "," in base64_str:
        base64_str = base64_str.split(",", 1)[1]
    decoded = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(decoded)).convert("RGB")


class GradCAM:
    """
    Grad-CAM implementation for PyTorch convolutional models (e.g. ResNet-18).
    Hooks into target layer activations and backpropagates class gradients to compute
    activation heatmaps highlighting predictive tissue regions.
    """

    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        self.model = model
        self.model.eval()

        if target_layer is None:
            # Default to layer4 (last convolutional block in ResNet)
            if hasattr(model, "layer4"):
                self.target_layer = model.layer4
            else:
                # Find last Conv2d layer
                conv_layers = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
                if not conv_layers:
                    raise ValueError("No convolutional layers found in model.")
                self.target_layer = conv_layers[-1]
        else:
            self.target_layer = target_layer

        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._handlers = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self._handlers.append(self.target_layer.register_forward_hook(forward_hook))
        self._handlers.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        for h in self._handlers:
            h.remove()
        self._handlers.clear()

    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> Tuple[np.ndarray, int, float]:
        """
        Compute Grad-CAM activation map for input_tensor.
        Args:
            input_tensor: shape (1, C, H, W) on model's device
            target_class: Target class index. If None, uses model's predicted class.
        Returns:
            (cam_map, predicted_class, class_probability)
            where cam_map is 2D numpy array [0, 1].
        """
        self.model.zero_grad()

        # Forward pass
        outputs = self.model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)

        pred_class = outputs.argmax(dim=1).item()
        if target_class is None:
            target_class = pred_class

        target_score = outputs[0, target_class]
        target_prob = probabilities[0, target_class].item()

        # Backward pass for target class score
        target_score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Gradients or activations were not captured by hooks.")

        # Pool gradients over spatial dimensions (global average pooling)
        # gradients: (1, C, H, W) -> weights: (C,)
        weights = torch.mean(self.gradients[0], dim=(1, 2))

        # Weighted combination of activation maps
        cam = torch.zeros(self.activations.shape[2:], dtype=torch.float32, device=self.activations.device)
        for i, w in enumerate(weights):
            cam += w * self.activations[0, i]

        # Apply ReLU to retain only positive influences
        cam = torch.relu(cam)

        cam_np = cam.detach().cpu().numpy()

        # Normalize between 0 and 1
        cam_min = np.min(cam_np)
        cam_max = np.max(cam_np)

        if cam_max - cam_min > 1e-8:
            cam_normalized = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_normalized = np.zeros_like(cam_np)

        return cam_normalized, pred_class, target_prob

    def __del__(self):
        self.remove_hooks()
