"""
Persistent Audit Logging and Event Storage for Phase 6 Agentic MLOps.
Stores runs, drift telemetry, and retraining plans with atomic JSON writes and in-memory caches.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import time
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STORAGE_DIR = PROJECT_ROOT / "artifacts" / "mlops"


@dataclass
class MLOpsRunRecord:
    run_id: str
    created_at: float
    status: str
    stages: List[Dict[str, Any]] = field(default_factory=list)
    artifacts: Dict[str, str] = field(default_factory=dict)
    summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MLOpsStorage:
    """
    Storage manager for MLOps runs, drift assessments, and retraining audit trails.
    """

    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = Path(base_dir or DEFAULT_STORAGE_DIR)
        self.runs_dir = self.base_dir / "runs"
        self.drift_dir = self.base_dir / "drift"
        self.retraining_dir = self.base_dir / "retraining"
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.drift_dir.mkdir(parents=True, exist_ok=True)
        self.retraining_dir.mkdir(parents=True, exist_ok=True)

    def save_run(self, run_id: str, data: Dict[str, Any]) -> str:
        """Atomically persist an MLOps run record."""
        self._ensure_dirs()
        file_path = self.runs_dir / f"{run_id}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return str(file_path)

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific run by ID."""
        file_path = self.runs_dir / f"{run_id}.json"
        if not file_path.exists():
            return None
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def list_runs(self, limit: int = 50) -> List[Dict[str, Any]]:
        """List recent orchestration runs sorted newest first."""
        self._ensure_dirs()
        runs = []
        for file in sorted(self.runs_dir.glob("*.json"), key=os.path.getmtime, reverse=True)[:limit]:
            try:
                with open(file, "r", encoding="utf-8") as f:
                    runs.append(json.load(f))
            except Exception:
                continue
        return runs

    def save_drift_event(self, event_data: Dict[str, Any]) -> str:
        """Persist a drift monitoring evaluation event."""
        self._ensure_dirs()
        event_id = f"drift_{int(time.time() * 1000)}"
        event_data["event_id"] = event_id
        event_data["timestamp"] = event_data.get("timestamp", time.time())
        file_path = self.drift_dir / f"{event_id}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(event_data, f, indent=2)
        return event_id

    def list_drift_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve recent drift evaluation events."""
        self._ensure_dirs()
        events = []
        for file in sorted(self.drift_dir.glob("*.json"), key=os.path.getmtime, reverse=True)[:limit]:
            try:
                with open(file, "r", encoding="utf-8") as f:
                    events.append(json.load(f))
            except Exception:
                continue
        return events

    def save_retraining_event(self, event_data: Dict[str, Any]) -> str:
        """Persist a retraining plan / decision event."""
        self._ensure_dirs()
        event_id = event_data.get("plan_id") or f"retrain_{int(time.time() * 1000)}"
        event_data["event_id"] = event_id
        event_data["timestamp"] = event_data.get("created_at", time.time())
        file_path = self.retraining_dir / f"{event_id}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(event_data, f, indent=2)
        return event_id

    def list_retraining_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve recent retraining plans."""
        self._ensure_dirs()
        events = []
        for file in sorted(self.retraining_dir.glob("*.json"), key=os.path.getmtime, reverse=True)[:limit]:
            try:
                with open(file, "r", encoding="utf-8") as f:
                    events.append(json.load(f))
            except Exception:
                continue
        return events
