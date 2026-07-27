"""Fault tolerance evaluation: measuring recovery from tool failures."""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
import numpy as np
import json


FAULT_TYPES = ["timeout", "rate_limit", "error", "schema_drift"]


@dataclass
class FaultToleranceMetrics:
    """Metrics for agent fault tolerance."""
    model_name: str
    task_id: str
    baseline_success: bool
    fault_results: Dict[str, bool]
    fault_scores: Dict[str, float]
    recovery_rate: float  # How often agent recovers from faults
    fault_tolerance_score: float  # Aggregate score

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model_name,
            "task": self.task_id,
            "baseline_success": self.baseline_success,
            "fault_results": self.fault_results,
            "fault_scores": self.fault_scores,
            "recovery_rate": self.recovery_rate,
            "fault_tolerance_score": self.fault_tolerance_score,
        }


class FaultToleranceEvaluator:
    """
    Evaluates how well an agent handles tool failures.
    
    Injects different types of faults (timeout, rate_limit, error, schema_drift)
    into tool executions and measures if the agent can recover.
    """

    def __init__(self, tool_registry):
        self.tool_registry = tool_registry

    def evaluate(
        self,
        model_name: str,
        agent_runner_fn: Callable,
        task_description: str,
        task_id: str,
        verification_fn: Callable[[str], bool],
        target_tool: Optional[str] = None,
    ) -> FaultToleranceMetrics:
        """Evaluate fault tolerance by injecting tool failures."""
        # Baseline (no faults)
        baseline_result = agent_runner_fn(task_description)
        baseline_success = verification_fn(baseline_result.final_answer) if baseline_result.success else False

        # Fault injection runs
        fault_results = {}
        fault_scores = {}

        for fault_type in FAULT_TYPES:
            # Set the fault on the first tool that will be called
            if target_tool and target_tool in self.tool_registry:
                self.tool_registry.get(target_tool).set_failure_mode(fault_type)

            result = agent_runner_fn(task_description)
            success = verification_fn(result.final_answer) if result.success else False
            fault_results[fault_type] = success
            fault_scores[fault_type] = 1.0 if success else 0.0

            # Reset the fault
            if target_tool and target_tool in self.tool_registry:
                self.tool_registry.get(target_tool).set_failure_mode(None)

        # Compute metrics
        scores = list(fault_scores.values())
        fault_tolerance_score = float(np.mean(scores)) if scores else 0.0
        recovery_rate = fault_tolerance_score  # Same in this implementation

        return FaultToleranceMetrics(
            model_name=model_name,
            task_id=task_id,
            baseline_success=baseline_success,
            fault_results=fault_results,
            fault_scores=fault_scores,
            recovery_rate=recovery_rate,
            fault_tolerance_score=fault_tolerance_score,
        )
