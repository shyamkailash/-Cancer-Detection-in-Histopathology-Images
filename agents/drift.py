"""
Drift Detection Agent for Continuous Histopathology Telemetry.
Monitors prediction probability and score distributions using PSI and TVD.
"""

from typing import Dict, Any, List, Optional
import numpy as np

from agents.base import BaseAgent
from mlops.contracts import AgentResult
from mlops.drift_monitor import DriftMonitor, DriftStatus


class DriftAgent(BaseAgent):
    """
    Sub-agent responsible for detecting statistical distribution shifts in inference telemetry.
    """

    def __init__(self, monitor: Optional[DriftMonitor] = None):
        super().__init__(name="DriftAgent", description="Evaluates statistical distribution drift in prediction scores")
        self.monitor = monitor or DriftMonitor()

    def execute(self, context: Dict[str, Any]) -> AgentResult:
        baseline = context.get("baseline")
        current = context.get("current")
        method = context.get("method", "psi")

        report = self.monitor.check_drift(
            baseline_distribution=baseline,
            current_distribution=current,
            method=method,
        )

        status_val = report.get("status", DriftStatus.NO_DRIFT.value)
        drift_detected = report.get("drift_detected", False)

        decision = "DRIFT_DETECTED" if drift_detected else "NO_DRIFT"
        if status_val == DriftStatus.WARNING.value:
            decision = "DRIFT_WARNING"
        elif status_val == DriftStatus.INSUFFICIENT_DATA.value:
            decision = "INSUFFICIENT_DATA"

        return AgentResult(
            agent_name=self.name,
            status="SUCCESS",
            decision=decision,
            details=report,
        )
