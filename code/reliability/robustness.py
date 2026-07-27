"""Robustness evaluation: measuring stability under input perturbations."""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
import numpy as np
import json


# Perturbation templates for task instructions
PERTURBATIONS = {
    "paraphrase": {
        "description": "Task instruction rephrased with different wording",
        "transform": lambda x: f"Can you please handle this request: {x}",
    },
    "verbose": {
        "description": "Task instruction with extra context added",
        "transform": lambda x: f"I need your help with something. Here's the situation: {x}. Let me know when you've completed this.",
    },
    "concise": {
        "description": "Task instruction shortened",
        "transform": lambda x: x.split(".")[0] + ".",
    },
    "typo": {
        "description": "Task instruction with minor typos",
        "transform": lambda x: x.replace("the", "teh").replace("and", "adn").replace("use", "uuse"),
    },
    "reordered": {
        "description": "Task steps presented in different order",
        "transform": lambda x: self._reorder_sentences(x),
    },
}


@dataclass
class RobustnessMetrics:
    """Metrics for agent robustness."""
    model_name: str
    task_id: str
    baseline_success: bool
    perturbation_results: Dict[str, bool]
    perturbation_scores: Dict[str, float]
    robustness_score: float  # Average performance across perturbations
    degradation: float  # How much performance dropped from baseline

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model_name,
            "task": self.task_id,
            "baseline_success": self.baseline_success,
            "perturbation_results": self.perturbation_results,
            "perturbation_scores": self.perturbation_scores,
            "robustness_score": self.robustness_score,
            "degradation": self.degradation,
        }


class RobustnessEvaluator:
    """
    Evaluates how robust an agent is to input perturbations.
    
    Applies multiple perturbation types to task instructions and
    measures whether the agent still succeeds.
    """

    def __init__(self, perturbation_types: Optional[List[str]] = None):
        self.perturbation_types = perturbation_types or list(PERTURBATIONS.keys())

    def evaluate(
        self,
        model_name: str,
        agent_runner: Callable,
        task_description: str,
        task_id: str,
        verification_fn: Callable[[str], bool],
    ) -> RobustnessMetrics:
        """Evaluate robustness by running with perturbed instructions."""
        # Baseline (unperturbed)
        baseline_result = agent_runner(task_description)
        baseline_success = verification_fn(baseline_result.final_answer) if baseline_result.success else False

        # Perturbed runs
        perturbation_results = {}
        perturbation_scores = {}

        for pert_type in self.perturbation_types:
            if pert_type not in PERTURBATIONS:
                continue

            transform_fn = PERTURBATIONS[pert_type]["transform"]
            # For methods that need instance reference, handle specially
            if pert_type == "reordered":
                perturbed = self._reorder_sentences(task_description)
            else:
                perturbed = PERTURBATIONS[pert_type]["transform"](task_description)

            result = agent_runner(perturbed)
            success = verification_fn(result.final_answer) if result.success else False
            perturbation_results[pert_type] = success
            perturbation_scores[pert_type] = 1.0 if success else 0.0

        # Compute aggregate metrics
        pert_scores = list(perturbation_scores.values())
        robustness_score = float(np.mean(pert_scores)) if pert_scores else 0.0
        degradation = (1.0 if baseline_success else 0.0) - robustness_score

        return RobustnessMetrics(
            model_name=model_name,
            task_id=task_id,
            baseline_success=baseline_success,
            perturbation_results=perturbation_results,
            perturbation_scores=perturbation_scores,
            robustness_score=robustness_score,
            degradation=max(0.0, degradation),
        )

    def _reorder_sentences(self, text: str) -> str:
        """Reorder sentences in a task description."""
        import random
        sentences = text.replace("!", ".").replace("?", ".").split(".")
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        if len(sentences) > 1:
            random.shuffle(sentences)
            return ". ".join(sentences) + "."
        return text
