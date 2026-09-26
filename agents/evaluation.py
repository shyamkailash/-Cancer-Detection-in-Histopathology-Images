"""
Evaluation Agent for Quality Gating and Candidate Comparison.
Evaluates model candidate performance metrics against baseline and defined quality thresholds.
"""

from typing import Dict, Any, List, Optional
from pathlib import Path

from agents.base import BaseAgent
from mlops.contracts import AgentResult, QualityGateConfig
from ml.inference.model_manager import PROJECT_ROOT


class EvaluationAgent(BaseAgent):
    """
    Sub-agent responsible for validating model metrics against strict quality gates.
    """

    def __init__(self, default_quality_gate: Optional[QualityGateConfig] = None):
        super().__init__(name="EvaluationAgent", description="Evaluates model performance metrics against quality gates")
        self.default_quality_gate = default_quality_gate or QualityGateConfig()

    def execute(self, context: Dict[str, Any]) -> AgentResult:
        metrics = context.get("metrics")
        model_id = context.get("model_id", "candidate_model")
        checkpoint_path = context.get("checkpoint_path")
        quality_gate = context.get("quality_gate") or self.default_quality_gate

        if isinstance(quality_gate, dict):
            quality_gate = QualityGateConfig(**quality_gate)

        if not metrics:
            return AgentResult(
                agent_name=self.name,
                status="FAILED",
                decision="MISSING_METRICS",
                details={"error": "No evaluation metrics provided in context for evaluation."},
            )

        if quality_gate.require_checkpoint_exists and checkpoint_path:
            ckpt_p = Path(checkpoint_path)
            if not ckpt_p.is_absolute():
                ckpt_p = PROJECT_ROOT / ckpt_p
            if not ckpt_p.exists():
                return AgentResult(
                    agent_name=self.name,
                    status="FAILED",
                    decision="CHECKPOINT_MISSING",
                    details={"error": f"Checkpoint does not exist at {ckpt_p}"},
                )

        passed = True
        failures = []
        checks = {}

        raw_acc = metrics.get("accuracy") or metrics.get("benchmark_accuracy")
        if raw_acc is not None:
            acc = float(raw_acc) / 100.0 if float(raw_acc) > 1.0 else float(raw_acc)
            checks["accuracy"] = {"value": acc, "required": quality_gate.min_accuracy, "passed": acc >= quality_gate.min_accuracy}
            if acc < quality_gate.min_accuracy:
                passed = False
                failures.append(f"Accuracy ({acc:.4f}) below minimum ({quality_gate.min_accuracy:.4f})")
        else:
            passed = False
            failures.append("Accuracy metric missing")

        if quality_gate.min_roc_auc is not None:
            raw_roc = metrics.get("roc_auc") or metrics.get("benchmark_roc_auc")
            if raw_roc is not None:
                roc = float(raw_roc)
                checks["roc_auc"] = {"value": roc, "required": quality_gate.min_roc_auc, "passed": roc >= quality_gate.min_roc_auc}
                if roc < quality_gate.min_roc_auc:
                    passed = False
                    failures.append(f"ROC-AUC ({roc:.4f}) below minimum ({quality_gate.min_roc_auc:.4f})")

        if quality_gate.min_sensitivity is not None:
            raw_sens = metrics.get("sensitivity") or metrics.get("benchmark_sensitivity")
            if raw_sens is not None:
                sens = float(raw_sens) / 100.0 if float(raw_sens) > 1.0 else float(raw_sens)
                checks["sensitivity"] = {"value": sens, "required": quality_gate.min_sensitivity, "passed": sens >= quality_gate.min_sensitivity}
                if sens < quality_gate.min_sensitivity:
                    passed = False
                    failures.append(f"Sensitivity ({sens:.4f}) below minimum ({quality_gate.min_sensitivity:.4f})")

        decision = "QUALITY_GATE_PASSED" if passed else "QUALITY_GATE_FAILED"
        status = "SUCCESS" if passed else "WARNING"

        return AgentResult(
            agent_name=self.name,
            status=status,
            decision=decision,
            details={
                "model_id": model_id,
                "passed": passed,
                "failures": failures,
                "checks": checks,
                "quality_gate": quality_gate.to_dict(),
            },
        )
