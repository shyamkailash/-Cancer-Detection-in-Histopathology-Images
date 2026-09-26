"""
Standard Contracts, Data Transfer Objects (DTOs), and Enums for Phase 6 Agentic MLOps.
Provides shared types across agents, monitors, registry lifecycle, and REST APIs.
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, Any, List, Optional


class RetrainingAction(str, Enum):
    NO_ACTION = "NO_ACTION"
    MONITOR = "MONITOR"
    TRIGGER_RETRAINING = "TRIGGER_RETRAINING"
    EMERGENCY_STOP = "EMERGENCY_STOP"


class OrchestrationStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    PAUSED = "PAUSED"
    DRY_RUN_COMPLETED = "DRY_RUN_COMPLETED"


class ModelLifecycleStage(str, Enum):
    CANDIDATE = "CANDIDATE"
    VALIDATED = "VALIDATED"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    ARCHIVED = "ARCHIVED"


@dataclass
class QualityGateConfig:
    """Quality gating criteria for model validation and promotion."""
    min_accuracy: float = 0.90
    min_roc_auc: Optional[float] = 0.90
    min_sensitivity: Optional[float] = 0.80
    max_drift_psi: Optional[float] = 0.25
    require_checkpoint_exists: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RetrainingResult:
    """Result of a retraining evaluation or agent decision."""
    action: RetrainingAction
    reason: str
    selected_strategy: Optional[str] = "fedavg"
    metadata: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action.value if isinstance(self.action, RetrainingAction) else str(self.action),
            "reason": self.reason,
            "selected_strategy": self.selected_strategy,
            "metadata": self.metadata,
            "confidence": self.confidence,
        }


@dataclass
class AgentResult:
    """Standardized return structure for all autonomous bounded agents."""
    agent_name: str
    status: str  # "SUCCESS", "SKIPPED", "WARNING", "FAILED"
    decision: str
    details: Dict[str, Any] = field(default_factory=dict)
    execution_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DriftEvaluationResult:
    """Structured drift assessment report."""
    status: str
    metric: str
    drift_score: float
    warning_threshold: float
    critical_threshold: float
    num_samples: int
    baseline_samples: int
    drift_detected: bool
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PrivacyEvaluationResult:
    """Formal privacy assessment report."""
    enabled: bool
    accountant: str
    epsilon: Optional[float]
    delta: Optional[float]
    clipping_norm: Optional[float]
    noise_multiplier: Optional[float]
    steps: int
    sample_rate: float
    guarantee_satisfied: bool
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
