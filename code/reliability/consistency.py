"""Consistency evaluation: measuring run-to-run variance."""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
import numpy as np
import json
import time


@dataclass
class ConsistencyMetrics:
    """Metrics for agent consistency."""
    model_name: str
    task_id: str
    num_runs: int
    success_rates: List[bool]
    success_rate: float
    pass_at_k: Dict[int, float]  # k=1,2,3,5
    trajectory_similarity: float  # How similar are the tool sequences across runs?
    answer_consistency: float  # How consistent are the final answers?
    variance_score: float  # Overall measure of variance (lower = more consistent)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model_name,
            "task": self.task_id,
            "num_runs": self.num_runs,
            "success_rate": self.success_rate,
            "pass_at_k": self.pass_at_k,
            "trajectory_similarity": self.trajectory_similarity,
            "answer_consistency": self.answer_consistency,
            "variance_score": self.variance_score,
        }


class ConsistencyEvaluator:
    """
    Evaluates how consistently an agent performs the same task across multiple runs.
    
    Key metrics:
    - pass@k: Probability that at least one of k runs succeeds
    - Trajectory similarity: Jaccard similarity of tool sequences
    - Answer consistency: Agreement between final answers
    """

    def __init__(self, num_runs: int = 5):
        self.num_runs = num_runs

    def evaluate(
        self,
        model_name: str,
        agent_runner: Callable,
        task_description: str,
        task_id: str,
        verification_fn: Callable[[str], bool],
    ) -> ConsistencyMetrics:
        """Run the agent multiple times on the same task and measure consistency."""
        results = []
        trajectories = []

        for run_idx in range(self.num_runs):
            # Use different seeds for each run
            result = agent_runner(task_description)
            success = verification_fn(result.final_answer) if result.success else False
            results.append(success)

            # Extract tool sequence from trajectory
            tool_seq = self._extract_tool_sequence(result)
            trajectories.append(tool_seq)

        # Compute metrics
        success_rate = sum(results) / self.num_runs

        # pass@k
        pass_at_k = {}
        for k in [1, 2, 3, 5]:
            if k <= self.num_runs:
                pass_at_k[f"pass@{k}"] = 1.0 - (1.0 - success_rate) ** k

        # Trajectory similarity (pairwise Jaccard)
        traj_sim = self._compute_trajectory_similarity(trajectories) if len(trajectories) > 1 else 1.0

        # Answer consistency
        ans_consistency = self._compute_answer_consistency(results)

        # Variance score (composite)
        variance_score = 1.0 - (success_rate * 0.4 + traj_sim * 0.3 + ans_consistency * 0.3)

        return ConsistencyMetrics(
            model_name=model_name,
            task_id=task_id,
            num_runs=self.num_runs,
            success_rates=results,
            success_rate=success_rate,
            pass_at_k=pass_at_k,
            trajectory_similarity=traj_sim,
            answer_consistency=ans_consistency,
            variance_score=variance_score,
        )

    def _extract_tool_sequence(self, agent_result) -> List[str]:
        """Extract the sequence of tool calls from an agent result."""
        tool_seq = []
        for step in agent_result.steps:
            if step.tool_call:
                tool_seq.append(step.tool_call[0])
        return tool_seq

    def _compute_trajectory_similarity(self, trajectories: List[List[str]]) -> float:
        """Compute average pairwise Jaccard similarity between trajectories."""
        similarities = []
        for i in range(len(trajectories)):
            for j in range(i + 1, len(trajectories)):
                set_i = set(trajectories[i])
                set_j = set(trajectories[j])
                if not set_i and not set_j:
                    sim = 1.0
                elif not set_i or not set_j:
                    sim = 0.0
                else:
                    intersection = set_i & set_j
                    union = set_i | set_j
                    sim = len(intersection) / len(union)
                similarities.append(sim)
        return float(np.mean(similarities)) if similarities else 1.0

    def _compute_answer_consistency(self, results: List[bool]) -> float:
        """Compute consistency of outcomes. Higher = more consistent."""
        if len(results) <= 1:
            return 1.0
        # Measure how clustered the results are (all same = 1.0, exactly split = 0.0)
        unique_outcomes = len(set(results))
        if unique_outcomes == 1:
            return 1.0
        # Compute entropy-based consistency
        p_success = sum(results) / len(results)
        # Maximum inconsistency at p=0.5
        inconsistency = 1.0 - 2.0 * abs(p_success - 0.5)
        return 1.0 - inconsistency
