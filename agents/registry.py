"""
Registry Lifecycle Agent for Model Staging and Promotion.
Controls state transitions across CANDIDATE, VALIDATED, APPROVED, and ARCHIVED.
"""

from typing import Dict, Any, Optional

from agents.base import BaseAgent
from mlops.contracts import AgentResult, QualityGateConfig
from mlops.registry_lifecycle import ModelRegistryLifecycle, ModelLifecycleState


class RegistryAgent(BaseAgent):
    """
    Sub-agent responsible for governing model promotion and registry lifecycle transitions.
    """

    def __init__(self, lifecycle: Optional[ModelRegistryLifecycle] = None):
        super().__init__(name="RegistryAgent", description="Manages model lifecycle staging, quality validation, and promotion")
        self.lifecycle = lifecycle or ModelRegistryLifecycle()

    def execute(self, context: Dict[str, Any]) -> AgentResult:
        action = context.get("action", "list")
        model_id = context.get("model_id")

        if action == "list":
            models = self.lifecycle.list_models()
            return AgentResult(
                agent_name=self.name,
                status="SUCCESS",
                decision="MODELS_LISTED",
                details={"models": models, "count": len(models)},
            )

        if not model_id:
            return AgentResult(
                agent_name=self.name,
                status="FAILED",
                decision="MISSING_MODEL_ID",
                details={"error": "model_id is required for registry actions other than list"},
            )

        if action == "register":
            checkpoint_path = context.get("checkpoint_path", "")
            metrics = context.get("metrics", {})
            display_name = context.get("display_name")
            record = self.lifecycle.register_candidate(
                model_id=model_id,
                checkpoint_path=checkpoint_path,
                metrics=metrics,
                display_name=display_name,
                paradigm=context.get("paradigm"),
                metadata=context.get("metadata"),
            )
            return AgentResult(
                agent_name=self.name,
                status="SUCCESS",
                decision="CANDIDATE_STAGED",
                details={"model": record},
            )

        elif action == "validate":
            qg_data = context.get("quality_gate")
            quality_gate = QualityGateConfig(**qg_data) if isinstance(qg_data, dict) else qg_data
            try:
                passed, note, record = self.lifecycle.validate_candidate(model_id=model_id, quality_gate=quality_gate)
                decision = "VALIDATION_PASSED" if passed else "VALIDATION_REJECTED"
                status = "SUCCESS" if passed else "WARNING"
                return AgentResult(
                    agent_name=self.name,
                    status=status,
                    decision=decision,
                    details={"passed": passed, "note": note, "model": record},
                )
            except KeyError as e:
                return AgentResult(
                    agent_name=self.name,
                    status="FAILED",
                    decision="MODEL_NOT_FOUND",
                    details={"error": str(e)},
                )

        elif action == "approve":
            approver_id = context.get("approver_id", "admin")
            try:
                record = self.lifecycle.approve_model(model_id=model_id, approver_id=approver_id)
                return AgentResult(
                    agent_name=self.name,
                    status="SUCCESS",
                    decision="MODEL_APPROVED",
                    details={"model": record},
                )
            except (KeyError, ValueError) as e:
                return AgentResult(
                    agent_name=self.name,
                    status="FAILED",
                    decision="APPROVAL_FAILED",
                    details={"error": str(e)},
                )

        elif action == "archive":
            try:
                record = self.lifecycle.archive_model(model_id=model_id)
                return AgentResult(
                    agent_name=self.name,
                    status="SUCCESS",
                    decision="MODEL_ARCHIVED",
                    details={"model": record},
                )
            except KeyError as e:
                return AgentResult(
                    agent_name=self.name,
                    status="FAILED",
                    decision="ARCHIVE_FAILED",
                    details={"error": str(e)},
                )

        return AgentResult(
            agent_name=self.name,
            status="FAILED",
            decision="UNKNOWN_ACTION",
            details={"error": f"Unsupported registry action: {action}"},
        )
