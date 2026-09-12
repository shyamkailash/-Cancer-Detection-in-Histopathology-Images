"""
Inference and Explainability Pipeline for Histopathology Patch Classification.
Combines image preprocessing, multi-model execution, and Grad-CAM visual heatmaps.
"""

import io
import time
from pathlib import Path
from typing import Dict, Any, Optional, Union
import numpy as np
import torch
from PIL import Image

from ml.preprocessing.transforms import get_eval_transforms
from .model_manager import ModelManager, get_model_manager
from .gradcam import GradCAM, generate_heatmap_overlay, pil_to_base64, base64_to_pil


CLASS_NAMES = {
    0: "normal",
    1: "metastasis",
}

CLASS_DESCRIPTIONS = {
    0: "Normal Lymph Node Tissue (No Tumor Detected)",
    1: "Metastatic Tumor Tissue (Malignant Cells Detected)",
}


class InferencePipeline:
    """
    End-to-end inference pipeline for histopathology cancer detection.
    """

    def __init__(self, model_manager: Optional[ModelManager] = None):
        self.model_manager = model_manager or get_model_manager()
        self.transform = get_eval_transforms(target_size=(96, 96))

    def _load_image(self, image_input: Union[Image.Image, bytes, str, Path]) -> Image.Image:
        """Standardize various input formats into a PIL RGB Image."""
        if isinstance(image_input, Image.Image):
            return image_input.convert("RGB")
        elif isinstance(image_input, bytes):
            return Image.open(io.BytesIO(image_input)).convert("RGB")
        elif isinstance(image_input, (str, Path)):
            path_or_str = str(image_input)
            if path_or_str.startswith("data:image") or len(path_or_str) > 500:
                # Likely base64 data URI
                return base64_to_pil(path_or_str)
            p = Path(path_or_str)
            if p.exists():
                return Image.open(p).convert("RGB")
            else:
                raise FileNotFoundError(f"Image file not found: {p.resolve()}")
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

    def predict(
        self,
        image_input: Union[Image.Image, bytes, str, Path],
        model_name: str = "centralized",
        include_gradcam: bool = True,
        target_class: Optional[int] = None,
        heatmap_alpha: float = 0.5,
        checkpoint_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """
        Run cancer detection prediction and optional Grad-CAM explainability.

        Args:
            image_input: PIL Image, raw bytes, filepath, or base64 string
            model_name: "centralized", "fedavg", or "dp_fedavg"
            include_gradcam: Whether to generate Grad-CAM heatmaps
            target_class: Specific class to explain (0 or 1). If None, explains predicted class.
            heatmap_alpha: Overlay blending opacity factor (0.0 to 1.0)

        Returns:
            Structured dictionary with prediction, probabilities, model metadata, and base64 visuals.
        """
        start_time = time.time()

        # 1. Load and validate image
        pil_image = self._load_image(image_input)
        orig_width, orig_height = pil_image.size

        # 2. Apply preprocessing transform (Resize to 96x96, Normalize)
        transformed_np = self.transform(pil_image)
        if isinstance(transformed_np, np.ndarray):
            input_tensor = torch.from_numpy(transformed_np).unsqueeze(0).float()
        elif isinstance(transformed_np, torch.Tensor):
            input_tensor = transformed_np.unsqueeze(0).float()
        else:
            raise RuntimeError(f"Unexpected transform output type: {type(transformed_np)}")

        # 3. Retrieve model and execute forward pass
        checkpoint = self.model_manager.resolve_checkpoint_path(model_name, checkpoint_path)
        selected_checkpoint = str(checkpoint) if checkpoint.exists() else checkpoint_path
        model = self.model_manager.get_model(model_name, checkpoint_path=selected_checkpoint)
        device = next(model.parameters()).device
        input_tensor = input_tensor.to(device)

        model_info = self.model_manager.get_model_info(model_name)

        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            pred_class_id = int(np.argmax(probs))

        prob_normal = float(probs[0])
        prob_metastasis = float(probs[1])
        confidence = float(np.max(probs))

        explain_class = target_class if target_class is not None else pred_class_id

        # 4. Generate Grad-CAM Explainability Heatmap
        explainability = None
        if include_gradcam:
            try:
                gradcam = GradCAM(model)
                cam_map, _, _ = gradcam.generate_cam(input_tensor, target_class=explain_class)
                gradcam.remove_hooks()

                heatmap_pil, overlay_pil = generate_heatmap_overlay(
                    original_image=pil_image,
                    cam_map=cam_map,
                    alpha=heatmap_alpha,
                )

                explainability = {
                    "method": "Grad-CAM",
                    "target_layer": "layer4",
                    "explained_class_id": explain_class,
                    "explained_class_name": CLASS_NAMES.get(explain_class, "unknown"),
                    "heatmap_base64": pil_to_base64(heatmap_pil, "PNG"),
                    "overlay_base64": pil_to_base64(overlay_pil, "PNG"),
                    "original_base64": pil_to_base64(pil_image, "PNG"),
                }
            except Exception as exc:
                explainability = {
                    "method": "Grad-CAM",
                    "error": f"Grad-CAM generation failed: {str(exc)}",
                }

        elapsed_ms = (time.time() - start_time) * 1000.0

        return {
            "status": "success",
            "prediction": {
                "class_id": pred_class_id,
                "class_name": CLASS_NAMES[pred_class_id],
                "description": CLASS_DESCRIPTIONS[pred_class_id],
                "confidence": round(confidence, 4),
                "probabilities": {
                    "normal": round(prob_normal, 4),
                    "metastasis": round(prob_metastasis, 4),
                },
            },
            "model": {
                "id": model_info["canonical_id"],
                "display_name": model_info["display_name"],
                "paradigm": model_info["paradigm"],
                "device": str(device),
                "checkpoint_path": str(checkpoint.resolve()),
            },
            "image_metadata": {
                "original_width": orig_width,
                "original_height": orig_height,
                "input_channels": 3,
                "processed_size": [96, 96],
            },
            "explainability": explainability,
            "inference_time_ms": round(elapsed_ms, 2),
            "disclaimer": "Academic & Research Prototype. Not approved for medical diagnosis or clinical use.",
        }


# Global singleton pipeline
_GLOBAL_INFERENCE_PIPELINE: Optional[InferencePipeline] = None


def get_inference_pipeline() -> InferencePipeline:
    global _GLOBAL_INFERENCE_PIPELINE
    if _GLOBAL_INFERENCE_PIPELINE is None:
        _GLOBAL_INFERENCE_PIPELINE = InferencePipeline()
    return _GLOBAL_INFERENCE_PIPELINE

