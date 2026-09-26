"""
Experiment Tracking Abstraction and MLflow Integration for Phase 6 MLOps.
Supports graceful degradation when MLflow is unavailable without breaking core ML or inference.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, List
import time
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class BaseExperimentTracker(ABC):
    """Abstract interface for MLOps experiment tracking and metadata logging."""

    @abstractmethod
    def is_available(self) -> bool:
        """Check if backend tracking engine is functional and available."""
        pass

    @abstractmethod
    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict[str, Any]] = None) -> Optional[str]:
        pass

    @abstractmethod
    def log_params(self, params: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        pass

    @abstractmethod
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        pass

    @abstractmethod
    def set_tags(self, tags: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    def end_run(self, status: str = "FINISHED") -> None:
        pass

    @abstractmethod
    def register_model(self, model_uri: str, name: str, tags: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        pass


class MLflowTracker(BaseExperimentTracker):
    """
    MLflow implementation of BaseExperimentTracker.
    Degrades gracefully to safe in-memory logging if MLflow is not installed or offline.
    """

    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        experiment_name: str = "pcam_cancer_detection",
        enabled: bool = True,
    ):
        self.enabled = enabled
        self.tracking_uri = tracking_uri or str(PROJECT_ROOT / "artifacts" / "mlflow")
        self.experiment_name = experiment_name
        self.active_run_id: Optional[str] = None
        self._mlflow_lib = None
        self._is_available = False
        self._fallback_runs: Dict[str, Dict[str, Any]] = {}

        if self.enabled:
            self._init_mlflow()

    def _init_mlflow(self) -> None:
        try:
            import mlflow
            self._mlflow_lib = mlflow
            if not self.tracking_uri.startswith("http"):
                Path(self.tracking_uri).mkdir(parents=True, exist_ok=True)
                mlflow.set_tracking_uri(f"file://{Path(self.tracking_uri).resolve()}")
            else:
                mlflow.set_tracking_uri(self.tracking_uri)

            mlflow.set_experiment(self.experiment_name)
            self._is_available = True
        except ImportError:
            self._is_available = False
        except Exception:
            self._is_available = False

    def is_available(self) -> bool:
        return self._is_available

    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict[str, Any]] = None) -> str:
        run_id = f"run_{int(time.time()*1000)}"
        clean_tags = tags or {}

        if self.is_available() and self._mlflow_lib:
            try:
                run = self._mlflow_lib.start_run(run_name=run_name, tags=clean_tags)
                self.active_run_id = run.info.run_id
                return self.active_run_id
            except Exception:
                pass

        # Fallback in-memory tracking
        self.active_run_id = run_id
        self._fallback_runs[run_id] = {
            "run_id": run_id,
            "run_name": run_name or run_id,
            "status": "RUNNING",
            "start_time": time.time(),
            "params": {},
            "metrics": {},
            "tags": clean_tags,
            "artifacts": [],
        }
        return run_id

    def log_params(self, params: Dict[str, Any]) -> None:
        if self.is_available() and self._mlflow_lib and self.active_run_id:
            try:
                self._mlflow_lib.log_params(params)
                return
            except Exception:
                pass

        if self.active_run_id and self.active_run_id in self._fallback_runs:
            self._fallback_runs[self.active_run_id]["params"].update(params)

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        if self.is_available() and self._mlflow_lib and self.active_run_id:
            try:
                self._mlflow_lib.log_metrics(metrics, step=step)
                return
            except Exception:
                pass

        if self.active_run_id and self.active_run_id in self._fallback_runs:
            self._fallback_runs[self.active_run_id]["metrics"].update(metrics)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        if self.is_available() and self._mlflow_lib and self.active_run_id:
            try:
                self._mlflow_lib.log_artifact(local_path, artifact_path=artifact_path)
                return
            except Exception:
                pass

        if self.active_run_id and self.active_run_id in self._fallback_runs:
            self._fallback_runs[self.active_run_id]["artifacts"].append({
                "local_path": local_path,
                "artifact_path": artifact_path,
            })

    def set_tags(self, tags: Dict[str, Any]) -> None:
        if self.is_available() and self._mlflow_lib and self.active_run_id:
            try:
                self._mlflow_lib.set_tags(tags)
                return
            except Exception:
                pass

        if self.active_run_id and self.active_run_id in self._fallback_runs:
            self._fallback_runs[self.active_run_id]["tags"].update(tags)

    def end_run(self, status: str = "FINISHED") -> None:
        if self.is_available() and self._mlflow_lib and self.active_run_id:
            try:
                self._mlflow_lib.end_run(status=status)
            except Exception:
                pass

        if self.active_run_id and self.active_run_id in self._fallback_runs:
            self._fallback_runs[self.active_run_id]["status"] = status
            self._fallback_runs[self.active_run_id]["end_time"] = time.time()

        self.active_run_id = None

    def register_model(self, model_uri: str, name: str, tags: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        if self.is_available() and self._mlflow_lib:
            try:
                model_ver = self._mlflow_lib.register_model(model_uri, name, tags=tags)
                return {"name": model_ver.name, "version": model_ver.version, "status": "REGISTERED"}
            except Exception:
                pass

        return {
            "name": name,
            "model_uri": model_uri,
            "version": "1.0",
            "status": "REGISTERED_LOCAL",
            "tags": tags or {},
        }
