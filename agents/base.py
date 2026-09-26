"""
Base Autonomous Bounded Agent Class for Phase 6 MLOps.
Provides standard execution structure, timing diagnostics, and result wrapping.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import time
import logging

from mlops.contracts import AgentResult

logger = logging.getLogger("agents.base")


class BaseAgent(ABC):
    """
    Abstract base class for all bounded MLOps sub-agents.
    Ensures deterministic, inspectable, and sandboxed decision-making.
    """

    def __init__(self, name: str, description: Optional[str] = None):
        self.name = name
        self.description = description or f"Agent {name}"
        self.logger = logging.getLogger(f"agents.{name}")

    @abstractmethod
    def execute(self, context: Dict[str, Any]) -> AgentResult:
        """Core execution logic to be implemented by specialized sub-agents."""
        pass

    def run(self, context: Optional[Dict[str, Any]] = None) -> AgentResult:
        ctx = context or {}
        start_time = time.time()
        try:
            result = self.execute(ctx)
            result.execution_time_ms = round((time.time() - start_time) * 1000, 2)
            return result
        except Exception as e:
            self.logger.exception(f"Error in agent {self.name}: {e}")
            return AgentResult(
                agent_name=self.name,
                status="FAILED",
                decision="ERROR",
                details={"error": str(e), "error_type": type(e).__name__},
                execution_time_ms=round((time.time() - start_time) * 1000, 2),
            )
