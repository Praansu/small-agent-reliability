#!/usr/bin/env python3
"""
Main experiment runner for the small language model agent reliability evaluation.

This script orchestrates the complete evaluation pipeline:
1. Load models via Ollama
2. Run task suite with each model
3. Evaluate across 4 reliability dimensions
4. Save results for analysis

Usage:
    python run_experiments.py --models qwen2.5:7b,mistral:7b --tasks all
    python run_experiments.py --quick  # Quick test with 1 model, subset of tasks
    python run_experiments.py --full   # Full evaluation
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from harness.agent import ReActAgent, AgentConfig, AgentResult
from harness.tools.base import ToolRegistry
from harness.tools.search import SearchTool, KnowledgeBaseTool
from harness.tools.calendar import CalendarTool, ScheduleManager
from harness.tools.data import DataQueryTool, CSVProcessor
from harness.tools.code import CodeInterpreterTool, FileOperator
from harness.tools.email import EmailTool, NotificationTool
from harness.utils import ModelConfig, load_model, set_seed

from tasks.task_suite import TaskSuite, TaskCategory
from tasks.evaluator import TaskEvaluator

from reliability.consistency import ConsistencyEvaluator
from reliability.robustness import RobustnessEvaluator
from reliability.fault_tolerance import FaultToleranceEvaluator
from reliability.safety import SafetyEvaluator

from models.model_registry import ModelRegistry, get_recommended_models, AVAILABLE_MODELS


def build_tool_registry() -> ToolRegistry:
    """Create and populate the tool registry."""
    registry = ToolRegistry()
    registry.register(SearchTool())
    registry.register(KnowledgeBaseTool())
    registry.register(CalendarTool())
    registry.register(ScheduleManager())
    registry.register(DataQueryTool())
    registry.register(CSVProcessor())
    registry.register(CodeInterpreterTool())
    registry.register(FileOperator())
    registry.register(EmailTool())
    registry.register(NotificationTool())
    return registry


def create_agent_runner(model_call_fn, tool_registry, config: AgentConfig):
    """Create a closure that runs a task and returns the result."""
    def run_task(task_description: str) -> AgentResult:
        agent = ReActAgent(model_call_fn, tool_registry, config)
        return agent.run(task_description)
    return run_task


def run_capability_evaluation(
    model_name: str,
    agent_runner,
    task_suite: TaskSuite,
    evaluator: TaskEvaluator,
) -> Dict[str, Any]:
    """Run the basic capability evaluation (task completion accuracy)."""
    print(f"\n{'='*60}")
    print(f"  RUNNING CAPABILITY EVALUATION: {model_name}")
    print(f"{'='*60}")

    tasks = task_suite.get_all_tasks()
    results = []

    for i, task in enumerate(tasks):
        print(f"  [{i+1}/{len(tasks)}] Task {task.id}: {task.description[:60]}...", end=" ", flush=True)

        try:
            agent_result = agent_runner(task.instruction)
            eval_result = evaluator.evaluate(
                task=task,
                agent_final_answer=agent_result.final_answer,
                agent_success=agent_result.success,
                steps_taken=len(agent_result.steps),
                duration_ms=agent_result.total_duration_ms,
                error=agent_result.error,
                trajectory=[s.response for s in agent_result.steps if s.response],
            )
            results.append(eval_result)

            status = "PASS" if eval_result.correctness else ("PART" if eval_result.score > 0 else "FAIL")
            print(f"{status} (score: {eval_result.score:.2f})")

        except Exception as e:
            print(f"ERROR: {e}")
            results.append(evaluator.evaluate(
                task=task, agent_final_answer="", agent_success=False,
                steps_taken=0, duration_ms=0, error=str(e),
            ))

    # Aggregate
    aggregated = evaluator.aggregate_scores(results)

    print(f"\n  Results for {model_name}:")
    print(f"    Accuracy: {aggregated['accuracy']*100:.1f}%")
    print(f"    Success Rate: {aggregated['success_rate']*100:.1f}%")
    print(f"    Average Score: {aggregated['average_score']:.3f}")
    print(f"    Error Rate: {aggregated['error_rate']*100:.1f}%")

    return {
        "model": model_name,
        "capability": aggregated,
        "per_task": [
            {
                "task_id": r.task_id,
                "description": r.task_description,
                "correctness": r.correctness,
                "score": r.score,
                "steps": r.steps_taken,
                "duration_ms": r.duration_ms,
                "error": r.error,
            }
            for r in results
        ],
    }


def run_consistency_evaluation(
    model_name: str,
    agent_runner,
    task_suite: TaskSuite,
    num_runs: int = 5,
) -> Dict[str, Any]:
    """Evaluate consistency across multiple runs."""
    print(f"\n{'='*60}")
    print(f"  RUNNING CONSISTENCY EVALUATION: {model_name}")
    print(f"{'='*60}")

    evaluator = ConsistencyEvaluator(num_runs=num_runs)
    # Use a subset of tasks for consistency (it's expensive)
    tasks = task_suite.get_tasks_by_category(TaskCategory.INFORMATION_RETRIEVAL)[:2]
    tasks += task_suite.get_tasks_by_category(TaskCategory.SCHEDULING)[:1]

    results = []
    for task in tasks:
        print(f"  Task {task.id}: {task.description[:50]}... ({num_runs} runs)", flush=True)
        metrics = evaluator.evaluate(
            model_name=model_name,
            agent_runner=agent_runner,
            task_description=task.instruction,
            task_id=task.id,
            verification_fn=task.verification_fn,
        )
        results.append(metrics.to_dict())
        print(f"    Success rate: {metrics.success_rate*100:.0f}%, "
              f"Trajectory similarity: {metrics.trajectory_similarity:.3f}, "
              f"Variance: {metrics.variance_score:.3f}")

    # Aggregate
    avg_success = sum(r["success_rate"] for r in results) / len(results)
    avg_similarity = sum(r["trajectory_similarity"] for r in results) / len(results)
    avg_variance = sum(r["variance_score"] for r in results) / len(results)

    return {
        "model": model_name,
        "num_runs": num_runs,
        "consistency_score": 1.0 - avg_variance,
        "avg_success_rate": avg_success,
        "avg_trajectory_similarity": avg_similarity,
        "avg_variance": avg_variance,
        "per_task": results,
    }


def run_robustness_evaluation(
    model_name: str,
    agent_runner,
    task_suite: TaskSuite,
) -> Dict[str, Any]:
    """Evaluate robustness under perturbations."""
    print(f"\n{'='*60}")
    print(f"  RUNNING ROBUSTNESS EVALUATION: {model_name}")
    print(f"{'='*60}")

    evaluator = RobustnessEvaluator()
    tasks = task_suite.get_tasks_by_category(TaskCategory.INFORMATION_RETRIEVAL)[:2]

    results = []
    for task in tasks:
        metrics = evaluator.evaluate(
            model_name=model_name,
            agent_runner=agent_runner,
            task_description=task.instruction,
            task_id=task.id,
            verification_fn=task.verification_fn,
        )
        results.append(metrics.to_dict())

        print(f"  Task {task.id}:")
        print(f"    Baseline: {'OK' if metrics.baseline_success else 'FAIL'}")
        for pert, success in metrics.perturbation_results.items():
            print(f"    {pert}: {'OK' if success else 'FAIL'}")
        print(f"    Robustness: {metrics.robustness_score:.3f}, Degradation: {metrics.degradation:.3f}")

    avg_robustness = sum(r["robustness_score"] for r in results) / len(results)
    avg_degradation = sum(r["degradation"] for r in results) / len(results)

    return {
        "model": model_name,
        "robustness_score": avg_robustness,
        "avg_degradation": avg_degradation,
        "per_task": results,
    }


def run_fault_tolerance_evaluation(
    model_name: str,
    agent_runner,
    task_suite: TaskSuite,
    tool_registry: ToolRegistry,
) -> Dict[str, Any]:
    """Evaluate fault tolerance with tool failures."""
    print(f"\n{'='*60}")
    print(f"  RUNNING FAULT TOLERANCE EVALUATION: {model_name}")
    print(f"{'='*60}")

    evaluator = FaultToleranceEvaluator(tool_registry)
    tasks = task_suite.get_tasks_by_category(TaskCategory.INFORMATION_RETRIEVAL)[:1]
    tasks += task_suite.get_tasks_by_category(TaskCategory.SCHEDULING)[:1]

    results = []
    for task in tasks:
        metrics = evaluator.evaluate(
            model_name=model_name,
            agent_runner_fn=agent_runner,
            task_description=task.instruction,
            task_id=task.id,
            verification_fn=task.verification_fn,
        )
        results.append(metrics.to_dict())

        print(f"  Task {task.id}:")
        print(f"    Baseline: {'OK' if metrics.baseline_success else 'FAIL'}")
        for fault, success in metrics.fault_results.items():
            print(f"    {fault}: {'OK' if success else 'FAIL'}")
        print(f"    Fault Tolerance: {metrics.fault_tolerance_score:.3f}")

    avg_ft = sum(r["fault_tolerance_score"] for r in results) / len(results)

    return {
        "model": model_name,
        "fault_tolerance_score": avg_ft,
        "per_task": results,
    }


def run_safety_evaluation(
    model_name: str,
    agent_runner,
) -> Dict[str, Any]:
    """Evaluate safety and refusal behavior."""
    print(f"\n{'='*60}")
    print(f"  RUNNING SAFETY EVALUATION: {model_name}")
    print(f"{'='*60}")

    evaluator = SafetyEvaluator()
    metrics = evaluator.evaluate(model_name=model_name, agent_runner=agent_runner)

    print(f"  Overall Safety Score: {metrics.overall_safety_score:.3f}")
    print(f"  Refusal Rate: {metrics.refusal_rate:.3f}")
    print(f"  Scope Adherence: {metrics.scope_adherence:.3f}")

    return {
        "model": model_name,
        "overall_safety_score": metrics.overall_safety_score,
        "refusal_rate": metrics.refusal_rate,
        "scope_adherence": metrics.scope_adherence,
        "per_test": [
            {"test_id": r["test_id"], "category": r["category"], "safe": r["safe"]}
            for r in metrics.per_test_results
        ],
    }


def compile_model_report(
    model_name: str,
    capability: Dict,
    consistency: Dict,
    robustness: Dict,
    fault_tolerance: Dict,
    safety: Dict,
) -> Dict[str, Any]:
    """Compile a comprehensive reliability report for a single model."""
    return {
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
        "capability": capability,
        "consistency": consistency,
        "robustness": robustness,
        "fault_tolerance": fault_tolerance,
        "safety": safety,
        "summary": {
            "accuracy": capability.get("capability", {}).get("accuracy", 0),
            "consistency_score": consistency.get("consistency_score", 0),
            "robustness_score": robustness.get("robustness_score", 0),
            "fault_tolerance_score": fault_tolerance.get("fault_tolerance_score", 0),
            "safety_score": safety.get("overall_safety_score", 0),
            "composite_reliability": (
                consistency.get("consistency_score", 0) * 0.25
                + robustness.get("robustness_score", 0) * 0.25
                + fault_tolerance.get("fault_tolerance_score", 0) * 0.25
                + safety.get("overall_safety_score", 0) * 0.25
            ),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Run small model agent reliability evaluation")
    parser.add_argument("--models", type=str, default="",
                        help="Comma-separated list of Ollama model IDs to evaluate")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test: 1 model, minimal tasks")
    parser.add_argument("--full", action="store_true",
                        help="Full evaluation: all models, all dimensions")
    parser.add_argument("--output", type=str, default="",
                        help="Output directory for results")
    parser.add_argument("--runs", type=int, default=3,
                        help="Number of runs for consistency evaluation")
    args = parser.parse_args()

    # Determine models
    if args.models:
        model_ids = [m.strip() for m in args.models.split(",")]
    elif args.quick:
        model_ids = ["llama3.2:3b"]
    else:
        # Default: recommended models
        model_ids = [m.ollama_id for m in get_recommended_models()]

    # Setup
    set_seed(42)
    tool_registry = build_tool_registry()
    task_suite = TaskSuite()
    task_evaluator = TaskEvaluator()

    print(f"\n{'#'*60}")
    print(f"  SMALL LANGUAGE MODEL AGENT RELIABILITY EVALUATION")
    print(f"  Models: {', '.join(model_ids)}")
    print(f"  Tasks: {len(task_suite.get_all_tasks())}")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'#'*60}\n")

    all_reports = []

    for model_id in model_ids:
        print(f"\n{'#'*60}")
        print(f"  EVALUATING MODEL: {model_id}")
        print(f"{'#'*60}")

        # Create model config and load
        model_config = ModelConfig(
            name=model_id,
            display_name=model_id,
            provider="ollama",
            model_id=model_id,
        )

        try:
            model_fn = load_model(model_config)
        except Exception as e:
            print(f"  FAILED to load model {model_id}: {e}")
            print(f"  Make sure the model is pulled: ollama pull {model_id}")
            continue

        agent_runner = create_agent_runner(
            model_fn, tool_registry,
            AgentConfig(max_steps=12, verbose=False, seed=42)
        )

        # Run evaluations
        capability = run_capability_evaluation(model_id, agent_runner, task_suite, task_evaluator)
        consistency = run_consistency_evaluation(model_id, agent_runner, task_suite, args.runs)
        robustness = run_robustness_evaluation(model_id, agent_runner, task_suite)
        fault_tolerance = run_fault_tolerance_evaluation(model_id, agent_runner, task_suite, tool_registry)
        safety = run_safety_evaluation(model_id, agent_runner)

        # Compile report
        report = compile_model_report(
            model_id, capability, consistency, robustness, fault_tolerance, safety
        )
        all_reports.append(report)

        # Save individual report
        output_dir = args.output or os.path.join(os.path.dirname(__file__), "..", "data", "raw")
        os.makedirs(output_dir, exist_ok=True)
        safe_name = model_id.replace(":", "_").replace("/", "_")
        report_path = os.path.join(output_dir, f"report_{safe_name}.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n  Report saved to: {report_path}")

    # Save aggregate report
    if all_reports:
        output_dir = args.output or os.path.join(os.path.dirname(__file__), "..", "data", "raw")
        aggregate_path = os.path.join(output_dir, "aggregate_report.json")
        with open(aggregate_path, "w") as f:
            json.dump({
                "experiment_date": datetime.now().isoformat(),
                "num_models": len(all_reports),
                "models": [r["model"] for r in all_reports],
                "reports": all_reports,
                "summary_comparison": {
                    r["model"]: r["summary"] for r in all_reports
                },
            }, f, indent=2)
        print(f"\n  Aggregate report saved to: {aggregate_path}")

        # Print final comparison table
        print(f"\n{'='*70}")
        print(f"  FINAL COMPARISON")
        print(f"{'='*70}")
        print(f"  {'Model':<20} {'Accuracy':>8} {'Consist.':>8} {'Robust.':>8} {'FaultTol':>8} {'Safety':>8} {'Reliability':>10}")
        print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")
        for r in all_reports:
            s = r["summary"]
            print(f"  {r['model']:<20} {s['accuracy']*100:>7.1f}% "
                  f"{s['consistency_score']*100:>7.1f}% "
                  f"{s['robustness_score']*100:>7.1f}% "
                  f"{s['fault_tolerance_score']*100:>7.1f}% "
                  f"{s['safety_score']*100:>7.1f}% "
                  f"{s['composite_reliability']*100:>8.1f}%")
        print(f"{'='*70}")

    print(f"\n*** Evaluation complete! ***\n")


if __name__ == "__main__":
    main()
