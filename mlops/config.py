"""
Centralized Configuration Loader for Phase 6 MLOps.
Supports YAML file loading, environment overrides, and strict type casting.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, Union
import os
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "mlops.yaml"


@dataclass
class MLflowConfig:
    enabled: bool = False
    tracking_uri: str = "artifacts/mlflow"
    experiment_name: str = "pcam_cancer_detection"


@dataclass
class DriftConfig:
    method: str = "psi"
    threshold: float = 0.10
    warning_threshold: float = 0.10
    critical_threshold: float = 0.25
    min_samples: int = 50


@dataclass
class RegistryConfig:
    min_accuracy: float = 0.90
    min_roc_auc: float = 0.90
    min_sensitivity: float = 0.80
    require_checkpoint_exists: bool = True


@dataclass
class PrivacySettings:
    enabled: bool = False
    strategy: str = "none"
    clipping_norm: float = 1.0
    noise_multiplier: float = 1.0
    delta: float = 1e-5
    accountant: str = "rdp"


@dataclass
class RetrainingSettings:
    enabled: bool = True
    dry_run: bool = True
    min_samples_since_retrain: int = 500
    cooldown_seconds: int = 3600


@dataclass
class OrchestratorSettings:
    dry_run: bool = True
    max_retries: int = 3
    timeout_seconds: int = 300


@dataclass
class MLOpsGlobalConfig:
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)
    drift: DriftConfig = field(default_factory=DriftConfig)
    registry: RegistryConfig = field(default_factory=RegistryConfig)
    privacy: PrivacySettings = field(default_factory=PrivacySettings)
    retraining: RetrainingSettings = field(default_factory=RetrainingSettings)
    orchestrator: OrchestratorSettings = field(default_factory=OrchestratorSettings)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mlflow": vars(self.mlflow),
            "drift": vars(self.drift),
            "registry": vars(self.registry),
            "privacy": vars(self.privacy),
            "retraining": vars(self.retraining),
            "orchestrator": vars(self.orchestrator),
        }


def load_mlops_config(config_path: Optional[Union[str, Path]] = None) -> MLOpsGlobalConfig:
    """Load and merge MLOps configuration from YAML and environment variables."""
    cfg = MLOpsGlobalConfig()
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH

    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

            if "mlflow" in data and isinstance(data["mlflow"], dict):
                cfg.mlflow = MLflowConfig(**data["mlflow"])
            if "drift" in data and isinstance(data["drift"], dict):
                cfg.drift = DriftConfig(**data["drift"])
            if "registry" in data and isinstance(data["registry"], dict):
                cfg.registry = RegistryConfig(**data["registry"])
            if "privacy" in data and isinstance(data["privacy"], dict):
                cfg.privacy = PrivacySettings(**data["privacy"])
            if "retraining" in data and isinstance(data["retraining"], dict):
                cfg.retraining = RetrainingSettings(**data["retraining"])
            if "orchestrator" in data and isinstance(data["orchestrator"], dict):
                cfg.orchestrator = OrchestratorSettings(**data["orchestrator"])
        except Exception:
            pass

    # Environment variable overrides
    if "MLFLOW_ENABLED" in os.environ:
        cfg.mlflow.enabled = os.environ["MLFLOW_ENABLED"].lower() in ("1", "true", "yes")
    if "MLFLOW_TRACKING_URI" in os.environ:
        cfg.mlflow.tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
    if "MLOPS_DRY_RUN" in os.environ:
        cfg.orchestrator.dry_run = os.environ["MLOPS_DRY_RUN"].lower() in ("1", "true", "yes")

    return cfg
