"""
Model Manager for loading, caching, and serving trained Cancer Detection models.
Supports Centralized, FedAvg, FedProx, FedBN, and DP-FedAvg ResNet-18 variants.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
from typing import Dict, Any, List, Optional, Union
import torch
import torch.nn as nn

from ml.models.resnet import create_resnet18

PROJECT_ROOT = Path(__file__).resolve().parents[2]


MODEL_REGISTRY_CONFIG = {
    "centralized": {
        "name": "centralized",
        "display_name": "Centralized ResNet-18 Baseline",
        "description": "Trained on pooled multi-site data in a centralized manner (ImageNet Pretrained).",
        "paradigm": "Centralized",
        "default_checkpoint": "artifacts/checkpoints/pcam_resnet18_best.pt",
        "fallback_checkpoint": "artifacts/experiments/centralized/pcam_resnet18_best.pt",
        "benchmark_accuracy": 94.13,
        "benchmark_sensitivity": 89.68,
        "benchmark_roc_auc": 0.9822,
    },
    "centralized_finetuned": {
        "name": "centralized_finetuned",
        "display_name": "Centralized ResNet-18 Fine-Tuned",
        "description": "2-stage transfer learning fine-tuned ResNet-18 with frozen BatchNorm and discriminative learning rates.",
        "paradigm": "Centralized Transfer Learning",
        "default_checkpoint": "artifacts/finetuning/centralized/best_model.pt",
        "fallback_checkpoint": "artifacts/finetuning/centralized_finetune_best.pt",
        "benchmark_accuracy": 95.35,
        "benchmark_sensitivity": 94.18,
        "benchmark_roc_auc": 0.9882,
    },
    "fedavg": {
        "name": "fedavg",
        "display_name": "Federated ResNet-18 (FedAvg)",
        "description": "Trained across 5 simulated healthcare sites using Federated Averaging with Non-IID Dirichlet skew (alpha=0.5).",
        "paradigm": "Federated Learning",
        "default_checkpoint": "artifacts/federated/fedavg/best_global_model.pt",
        "fallback_checkpoint": "artifacts/federated/fedavg/best_model.pt",
        "benchmark_accuracy": 95.09,
        "benchmark_sensitivity": 92.03,
        "benchmark_roc_auc": 0.9872,
    },
    "fedprox": {
        "name": "fedprox",
        "display_name": "Federated ResNet-18 (FedProx)",
        "description": "Trained across simulated healthcare sites with proximal regularization (mu=0.01) to mitigate Non-IID client drift.",
        "paradigm": "Federated Learning (Proximal Regularization)",
        "default_checkpoint": "artifacts/federated/fedprox/best_global_model.pt",
        "fallback_checkpoint": "artifacts/federated/fedprox/best_model.pt",
        "benchmark_accuracy": 95.05,
        "benchmark_sensitivity": 91.91,
        "benchmark_roc_auc": 0.9870,
    },
    "fedbn": {
        "name": "fedbn",
        "display_name": "Federated ResNet-18 (FedBN)",
        "description": "Trained with local BatchNorm layers preserved on each client site to adapt to multi-site staining shifts.",
        "paradigm": "Federated Learning (Local BatchNorm)",
        "default_checkpoint": "artifacts/federated/fedbn/best_global_model.pt",
        "fallback_checkpoint": "artifacts/federated/fedbn/best_model.pt",
        "benchmark_accuracy": 91.77,
        "benchmark_sensitivity": 82.18,
        "benchmark_roc_auc": 0.9796,
    },
    "dp_fedavg": {
        "name": "dp_fedavg",
        "display_name": "Privacy-Preserving Federated ResNet-18 (DP-FedAvg)",
        "description": "Trained with client-side Differential Privacy (gradient norm clipping C=1.0 + Gaussian noise sigma=1.0).",
        "paradigm": "Privacy-Preserving Federated Learning",
        "default_checkpoint": "artifacts/federated/dp_fedavg/best_global_model.pt",
        "fallback_checkpoint": "artifacts/federated/dp_fedavg/best_model.pt",
        "benchmark_accuracy": 94.33,
        "benchmark_sensitivity": 90.44,
        "benchmark_roc_auc": 0.9831,
    },
}

# Alias mappings
MODEL_ALIASES = {
    "central": "centralized",
    "baseline": "centralized",
    "pcam_resnet18": "centralized",
    "finetuned": "centralized_finetuned",
    "centralized_ft": "centralized_finetuned",
    "ft": "centralized_finetuned",
    "central_ft": "centralized_finetuned",
    "federated": "fedavg",
    "federated_fedavg": "fedavg",
    "fl": "fedavg",
    "prox": "fedprox",
    "proximal": "fedprox",
    "federated_prox": "fedprox",
    "bn": "fedbn",
    "batchnorm": "fedbn",
    "federated_bn": "fedbn",
    "dp": "dp_fedavg",
    "dp_fl": "dp_fedavg",
    "dpfedavg": "dp_fedavg",
    "federated_dp": "dp_fedavg",
    "privacy_federated": "dp_fedavg",
}


class ModelManager:
    """Manages loading, caching, and serving inference models."""

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._models: Dict[str, nn.Module] = {}

    def _resolve_path(self, path_like: Optional[Union[str, Path]]) -> Path:
        """Resolve a path to an absolute path relative to project root if relative."""
        if not path_like:
            return Path("")
        p = Path(path_like)
        if p.is_absolute():
            return p
        return (PROJECT_ROOT / p).resolve()

    def canonicalize_model_name(self, model_name: str) -> str:
        """Resolve model alias to canonical identifier."""
        name_clean = str(model_name).lower().strip()
        if name_clean in MODEL_REGISTRY_CONFIG:
            return name_clean
        if name_clean in MODEL_ALIASES:
            return MODEL_ALIASES[name_clean]
        raise ValueError(
            f"Unknown model '{model_name}'. Available models: {list(MODEL_REGISTRY_CONFIG.keys())}"
        )

    def list_models(self) -> List[Dict[str, Any]]:
        """Return catalog of available models and their status."""
        catalog = []
        for model_id, cfg in MODEL_REGISTRY_CONFIG.items():
            ckpt_path = Path(cfg["default_checkpoint"])
            fallback_path = Path(cfg.get("fallback_checkpoint", ""))
            has_ckpt = ckpt_path.exists() or (fallback_path.exists() if str(fallback_path) else False)
            default_str = cfg.get("default_checkpoint", "")
            fallback_str = cfg.get("fallback_checkpoint", "")

            chosen_path = str(ckpt_path) if ckpt_path.exists() else (str(fallback_path) if fallback_path.exists() else "")
            default_abs = self._resolve_path(default_str)
            fallback_abs = self._resolve_path(fallback_str)

            if default_str and default_abs.exists():
                has_ckpt = True
                chosen_path = default_str
            elif fallback_str and fallback_abs.exists():
                has_ckpt = True
                chosen_path = fallback_str
            else:
                has_ckpt = False
                chosen_path = ""

            catalog.append({
                "id": model_id,
                "display_name": cfg["display_name"],
                "description": cfg["description"],
                "paradigm": cfg["paradigm"],
                "checkpoint_exists": has_ckpt,
                "checkpoint_path": chosen_path,
                "is_cached": model_id in self._models,
                "is_cached": any(k.startswith(f"{model_id}:") for k in self._models),
                "benchmark_accuracy": cfg.get("benchmark_accuracy"),
                "benchmark_sensitivity": cfg.get("benchmark_sensitivity"),
                "benchmark_roc_auc": cfg.get("benchmark_roc_auc"),
            })
        return catalog

    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Return metadata for a specific model."""
        canonical = self.canonicalize_model_name(model_name)
        cfg = dict(MODEL_REGISTRY_CONFIG[canonical])
        cfg["canonical_id"] = canonical
        cfg["device"] = str(self.device)
        return cfg

    def get_model(
        self,
        model_name: str = "centralized",
        checkpoint_path: Optional[str] = None,
    ) -> nn.Module:
        """
        Get or load the requested model. Caches loaded models in memory.
        """
        canonical = self.canonicalize_model_name(model_name)

        cfg = MODEL_REGISTRY_CONFIG[canonical]
        ckpt_path = Path(checkpoint_path) if checkpoint_path else Path(cfg["default_checkpoint"])

        if checkpoint_path and not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path.resolve()}")
        target_path_str = checkpoint_path or cfg.get("default_checkpoint", "")
        abs_path = self._resolve_path(target_path_str)

        if not checkpoint_path and not ckpt_path.exists() and "fallback_checkpoint" in cfg:
            fallback = Path(cfg["fallback_checkpoint"])
            if fallback.exists():
                ckpt_path = fallback
        if checkpoint_path and not abs_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {abs_path}")

        cache_key = f"{canonical}:{ckpt_path.resolve()}"
        if not checkpoint_path and not abs_path.exists() and "fallback_checkpoint" in cfg:
            fallback_abs = self._resolve_path(cfg["fallback_checkpoint"])
            if fallback_abs.exists():
                abs_path = fallback_abs

        cache_key = f"{canonical}:{abs_path}"
        if cache_key in self._models:
            return self._models[cache_key]

        # Initialize ResNet-18
        model = create_resnet18(num_classes=2, pretrained=True).to(self.device)

        if ckpt_path.exists():
            checkpoint = torch.load(str(ckpt_path), map_location=self.device, weights_only=False)
        if abs_path.exists() and abs_path.is_file():
            checkpoint = torch.load(str(abs_path), map_location=self.device, weights_only=False)
            state_dict = checkpoint.get("model_state_dict") if isinstance(checkpoint, dict) else None
            if state_dict is None and isinstance(checkpoint, dict):
                state_dict = checkpoint
            if not isinstance(state_dict, dict):
                raise ValueError(
                    f"Checkpoint '{ckpt_path}' does not contain a valid model_state_dict"
                    f"Checkpoint '{abs_path}' does not contain a valid model_state_dict"
                )
            try:
                model.load_state_dict(state_dict, strict=True)
            except RuntimeError as exc:
                raise ValueError(
                    f"Checkpoint '{ckpt_path}' is incompatible with the expected ResNet-18 architecture: {exc}"
                    f"Checkpoint '{abs_path}' is incompatible with the expected ResNet-18 architecture: {exc}"
                ) from exc
        else:
            # Fallback to pretrained weights if checkpoint file is not yet generated
            pass

        model.eval()
        self._models[cache_key] = model
        return model

    def resolve_checkpoint_path(
        self,
        model_name: str = "centralized",
        checkpoint_path: Optional[str] = None,
    ) -> Path:
        """Resolve the checkpoint selected for a model without loading it."""
        canonical = self.canonicalize_model_name(model_name)
        if checkpoint_path:
            path = Path(checkpoint_path)
            if not path.exists():
                raise FileNotFoundError(f"Checkpoint file not found: {path.resolve()}")
            abs_path = self._resolve_path(path)
            if not abs_path.exists():
                raise FileNotFoundError(f"Checkpoint file not found: {abs_path}")
            return path

        config = MODEL_REGISTRY_CONFIG[canonical]
        path = Path(config["default_checkpoint"])
        if not path.exists() and config.get("fallback_checkpoint"):
            fallback = Path(config["fallback_checkpoint"])
            if fallback.exists():
                return fallback
        return path
        default_path = Path(config["default_checkpoint"])
        if self._resolve_path(default_path).exists():
            return default_path
        if config.get("fallback_checkpoint"):
            fallback_path = Path(config["fallback_checkpoint"])
            if self._resolve_path(fallback_path).exists():
                return fallback_path
        return default_path

    def preload_all(self):
        """Preload all configured models into memory."""
        for model_id in MODEL_REGISTRY_CONFIG:
            self.get_model(model_id)


# Global singleton instance
_GLOBAL_MODEL_MANAGER: Optional[ModelManager] = None


def get_model_manager() -> ModelManager:
    global _GLOBAL_MODEL_MANAGER
    if _GLOBAL_MODEL_MANAGER is None:
        _GLOBAL_MODEL_MANAGER = ModelManager()
    return _GLOBAL_MODEL_MANAGER
