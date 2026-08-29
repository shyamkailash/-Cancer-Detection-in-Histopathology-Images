"""
Inference and explainability package for histopathology cancer detection.
"""

from .model_manager import ModelManager, get_model_manager
from .gradcam import GradCAM, generate_heatmap_overlay, pil_to_base64, base64_to_pil
from .pipeline import InferencePipeline, get_inference_pipeline

__all__ = [
    "ModelManager",
    "get_model_manager",
    "GradCAM",
    "generate_heatmap_overlay",
    "pil_to_base64",
    "base64_to_pil",
    "InferencePipeline",
    "get_inference_pipeline",
]

