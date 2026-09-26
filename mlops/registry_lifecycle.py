"""
Model Registry Lifecycle and Quality Gating Architecture.
Implements candidate staging, validation gating, explicit approval, and archiving.
"""

from enum import Enum
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import time
import json

from mlops.contracts import QualityGateConfig
from ml.inference.model_manager import MODEL_REGISTRY_CONFIG, PROJECT_ROOT


class ModelLifecycleState(str, Enum):
    CANDIDATE = "CANDIDATE"
    VALIDATED = "VALIDATED"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    ARCHIVED = "ARCHIVED"


class ModelRegistryLifecycle:
    """
    Manages the multi-stage lifecycle of cancer detection model candidates.
    Enforces quality gating and requires explicit approval before promotion.
    """

    def __init__(self, storage_file: Optional[Path] = None):
        self.storage_file = storage_file or (PROJECT_ROOT / "artifacts" / "registry" / "lifecycle_registry.json")
        self._models: Dict[str, Dict[str, Any]] = {}
        self._load_seed_registry()

    def _load_seed_registry(self) -> None:
        """Seed registry with existing verified benchmark models in APPROVED state."""
        for model_id, cfg in MODEL_REGISTRY_CONFIG.items():
            ckpt_path = str((PROJECT_ROOT / cfg["default_checkpoint"]).resolve())
            self._models[model_id] = {
                "model_id": model_id,
                "display_name": cfg["display_name"],
                "paradigm": cfg["paradigm"],
                "state": ModelLifecycleState.APPROVED.value,
                "checkpoint_path": ckpt_path,
                "metrics": {
                    "accuracy": cfg.get("benchmark_accuracy", 0.0) / 100.0 if cfg.get("benchmark_accuracy") else 0.0,
                    "sensitivity": cfg.get("benchmark_sensitivity", 0.0) / 100.0 if cfg.get("benchmark_sensitivity") else 0.0,
                    "roc_auc": cfg.get("benchmark_roc_auc"),
                },
                "registered_at": time.time(),
                "updated_at": time.time(),
                "approver": "system_baseline_promotion",
                "validation_notes": "Existing verified benchmark model.",
                "history": [
                    {"state": ModelLifecycleState.CANDIDATE.value, "timestamp": time.time(), "note": "Initial baseline"},
                    {"state": ModelLifecycleState.VALIDATED.value, "timestamp": time.time(), "note": "Quality gate passed"},
                    {"state": ModelLifecycleState.APPROVED.value, "timestamp": time.time(), "note": "Promoted to production baseline"},
                ],
            }

    def register_candidate(
        self,
        model_id: str,
        checkpoint_path: str,
        metrics: Dict[str, Any],
        display_name: Optional[str] = None,
        paradigm: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Stage a new trained checkpoint as a CANDIDATE."""
        now = time.time()
        record = {
            "model_id": model_id,
            "display_name": display_name or model_id,
            "paradigm": paradigm or "Custom Training",
            "state": ModelLifecycleState.CANDIDATE.value,
            "checkpoint_path": checkpoint_path,
            "metrics": metrics,
            "registered_at": now,
            "updated_at": now,
            "metadata": metadata or {},
            "history": [
                {"state": ModelLifecycleState.CANDIDATE.value, "timestamp": now, "note": "Staged candidate model"},
            ],
        }
        self._models[model_id] = record
        return record

    def validate_candidate(
        self,
        model_id: str,
        quality_gate: Optional[QualityGateConfig] = None,
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Evaluate candidate against quality gates. If passed, transition to VALIDATED; else REJECTED.
        """
        if model_id not in self._models:
            raise KeyError(f"Model '{model_id}' not found in lifecycle registry.")

        record = self._models[model_id]
        qg = quality_gate or QualityGateConfig()
        metrics = record.get("metrics", {})
        ckpt_path = Path(record.get("checkpoint_path", ""))

        passed = True
        reasons = []

        if qg.require_checkpoint_exists:
            if not ckpt_path.is_absolute():
                ckpt_path = (PROJECT_ROOT / ckpt_path).resolve()
            if not ckpt_path.exists():
                passed = False
                reasons.append(f"Checkpoint file does not exist at '{ckpt_path}'")

        if not metrics:
            passed = False
            reasons.append("Missing evaluation metrics")
        else:
            raw_acc = metrics.get("accuracy") or metrics.get("benchmark_accuracy")
            if raw_acc is not None:
                acc = float(raw_acc) / 100.0 if float(raw_acc) > 1.0 else float(raw_acc)
                if acc < qg.min_accuracy:
                    passed = False
                    reasons.append(f"Accuracy {acc:.4f} < {qg.min_accuracy:.4f}")
            else:
                passed = False
                reasons.append("Accuracy metric missing")

            if qg.min_roc_auc is not None:
                raw_roc = metrics.get("roc_auc") or metrics.get("benchmark_roc_auc")
                if raw_roc is not None and float(raw_roc) < qg.min_roc_auc:
                    passed = False
                    reasons.append(f"ROC-AUC {float(raw_roc):.4f} < {qg.min_roc_auc:.4f}")

            if qg.min_sensitivity is not None:
                raw_sens = metrics.get("sensitivity") or metrics.get("benchmark_sensitivity")
                if raw_sens is not None:
                    sens = float(raw_sens) / 100.0 if float(raw_sens) > 1.0 else float(raw_sens)
                    if sens < qg.min_sensitivity:
                        passed = False
                        reasons.append(f"Sensitivity {sens:.4f} < {qg.min_sensitivity:.4f}")

        now = time.time()
        record["updated_at"] = now
        if passed:
            record["state"] = ModelLifecycleState.VALIDATED.value
            note = "Quality gate PASSED: " + ", ".join([f"Acc>={qg.min_accuracy}"])
            record["history"].append({"state": ModelLifecycleState.VALIDATED.value, "timestamp": now, "note": note})
            return True, note, record
        else:
            record["state"] = ModelLifecycleState.REJECTED.value
            note = "Quality gate FAILED: " + "; ".join(reasons)
            record["history"].append({"state": ModelLifecycleState.REJECTED.value, "timestamp": now, "note": note})
            return False, note, record

    def approve_model(self, model_id: str, approver_id: str = "admin") -> Dict[str, Any]:
        """
        Explicit approval transitioning a VALIDATED candidate to APPROVED state.
        """
        if model_id not in self._models:
            raise KeyError(f"Model '{model_id}' not found.")

        record = self._models[model_id]
        if record["state"] not in (ModelLifecycleState.VALIDATED.value, ModelLifecycleState.APPROVED.value):
            raise ValueError(f"Cannot approve model in state '{record['state']}'. Must be VALIDATED first.")

        now = time.time()
        record["state"] = ModelLifecycleState.APPROVED.value
        record["updated_at"] = now
        record["approver"] = approver_id
        record["history"].append({
            "state": ModelLifecycleState.APPROVED.value,
            "timestamp": now,
            "note": f"Explicitly approved by '{approver_id}'",
        })
        return record

    def archive_model(self, model_id: str, reason: str = "") -> Dict[str, Any]:
        """Archive a model."""
        if model_id not in self._models:
            raise KeyError(f"Model '{model_id}' not found.")
        record = self._models[model_id]
        now = time.time()
        record["state"] = ModelLifecycleState.ARCHIVED.value
        record["updated_at"] = now
        record["history"].append({"state": ModelLifecycleState.ARCHIVED.value, "timestamp": now, "note": reason or "Archived"})
        return record

    def get_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        return self._models.get(model_id)

    def list_models(
        self,
        state: Optional[Any] = None,
        state_filter: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        target_state = state or state_filter
        if target_state:
            st_val = target_state.value if isinstance(target_state, ModelLifecycleState) else str(target_state)
            return [m for m in self._models.values() if m["state"] == st_val]
        return list(self._models.values())


_GLOBAL_LIFECYCLE_REGISTRY: Optional[ModelRegistryLifecycle] = None


def get_lifecycle_registry() -> ModelRegistryLifecycle:
    global _GLOBAL_LIFECYCLE_REGISTRY
    if _GLOBAL_LIFECYCLE_REGISTRY is None:
        _GLOBAL_LIFECYCLE_REGISTRY = ModelRegistryLifecycle()
    return _GLOBAL_LIFECYCLE_REGISTRY
