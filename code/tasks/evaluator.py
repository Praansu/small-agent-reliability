"""Task evaluation and scoring."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .task_suite import Task


@dataclass
class EvaluationResult:
    """Result of evaluating an agent's task completion."""
    task_id: str
    task_description: str
    agent_success: bool  # Did the agent produce a FINAL_ANSWER?
    correctness: bool  # Was the answer correct?
    steps_taken: int
    duration_ms: float
    error: Optional[str] = None
    agent_trajectory: List[str] = field(default_factory=list)
    score: float = 0.0  # 0.0 to 1.0 composite score


class TaskEvaluator:
    """Evaluates agent task completion with scoring."""

    def __init__(self, partial_credit: bool = True):
        self.partial_credit = partial_credit

    def evaluate(self, task: Task, agent_final_answer: str, agent_success: bool,
                 steps_taken: int, duration_ms: float, error: Optional[str] = None,
                 trajectory: Optional[List[str]] = None) -> EvaluationResult:
        """
        Evaluate an agent's performance on a task.
        Returns an EvaluationResult with scoring.
        """
        correctness = False
        score = 0.0

        if agent_success and agent_final_answer:
            # Check correctness using the task's verification function
            correctness = task.verification_fn(agent_final_answer)

            # Compute score
            if correctness:
                score = 1.0
            elif self.partial_credit:
                # Partial credit for attempting the right tools
                score = self._compute_partial_credit(task, trajectory)
        elif agent_success and not agent_final_answer:
            # Agent said it completed but gave no answer
            score = 0.3 if self.partial_credit else 0.0
        elif error:
            # Agent failed with error
            score = 0.0
        else:
            # Agent didn't complete
            score = 0.0

        return EvaluationResult(
            task_id=task.id,
            task_description=task.description,
            agent_success=agent_success,
            correctness=correctness,
            steps_taken=steps_taken,
            duration_ms=duration_ms,
            error=error,
            agent_trajectory=trajectory or [],
            score=score,
        )

    def _compute_partial_credit(self, task: Task, trajectory: Optional[List[str]]) -> float:
        """Compute partial credit based on tool usage."""
        if not trajectory:
            return 0.0

        required_tools = set(task.tools_required)
        used_tools = set()
        for step in trajectory:
            for tool in required_tools:
                if tool in step.lower():
                    used_tools.add(tool)

        if not required_tools:
            return 0.5  # Some effort made

        tool_coverage = len(used_tools) / len(required_tools)
        return 0.3 + (tool_coverage * 0.4)  # 0.3 to 0.7 partial credit

    def aggregate_scores(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """Aggregate scores across multiple evaluations."""
        if not results:
            return {"error": "no results"}

        n = len(results)
        correct = sum(1 for r in results if r.correctness)
        succeeded = sum(1 for r in results if r.agent_success)
        total_score = sum(r.score for r in results)

        return {
            "num_tasks": n,
            "accuracy": correct / n,
            "success_rate": succeeded / n,
            "average_score": total_score / n,
            "std_score": (sum((r.score - total_score / n) ** 2 for r in results) / n) ** 0.5 if n > 1 else 0,
            "total_duration_ms": sum(r.duration_ms for r in results),
            "average_steps": sum(r.steps_taken for r in results) / n,
            "errors": [r.error for r in results if r.error],
            "error_rate": sum(1 for r in results if r.error) / n,
        }
