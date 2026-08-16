#!/usr/bin/env python3
"""
Reliability-dimension VERIFICATION re-run (consistency, robustness,
fault tolerance, safety). Mirrors code/run_experiments.py dimension
protocols exactly (num_runs=3, same task subsets, same evaluators) but
skips the capability suite (already re-run by verify_experiments.py).
Saves incrementally per model to data/raw/verify_dims/.
"""
import json, os, sys, time
from datetime import datetime

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(BASE, ".."))
sys.path.insert(0, BASE)

from harness.agent import ReActAgent, AgentConfig
from harness.tools.base import ToolRegistry
from harness.tools.search import SearchTool, KnowledgeBaseTool
from harness.tools.calendar import CalendarTool, ScheduleManager
from harness.tools.data import DataQueryTool, CSVProcessor
from harness.tools.code import CodeInterpreterTool, FileOperator
from harness.tools.email import EmailTool, NotificationTool
from harness.utils import ModelConfig, load_model, set_seed

from tasks.task_suite import TaskSuite, TaskCategory
from reliability.consistency import ConsistencyEvaluator
from reliability.robustness import RobustnessEvaluator
from reliability.fault_tolerance import FaultToleranceEvaluator
from reliability.safety import SafetyEvaluator

MODELS = ["qwen2.5-coder:7b", "qwen2.5:7b", "llama3.2:1b", "mistral:7b",
          "llama3.2:3b", "deepseek-r1:7b", "phi3.5:3.8b", "llama3.1:8b", "gemma2:9b"]
OUT_DIR = os.path.join(ROOT, "data", "raw", "verify_dims")
STATE_PATH = os.path.join(OUT_DIR, "dims_results.json")


def build_tool_registry():
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


def create_agent_runner(model_fn, tool_registry, config):
    agent = ReActAgent(model_fn, tool_registry, config)
    def runner(task_description):
        return agent.run(task_description)
    return runner


def run_consistency_evaluation(model_name, agent_runner, task_suite, num_runs=3):
    evaluator = ConsistencyEvaluator(num_runs=num_runs)
    tasks = task_suite.get_tasks_by_category(TaskCategory.INFORMATION_RETRIEVAL)[:2]
    tasks += task_suite.get_tasks_by_category(TaskCategory.SCHEDULING)[:1]
    results = []
    for task in tasks:
        metrics = evaluator.evaluate(
            model_name=model_name, agent_runner=agent_runner,
            task_description=task.instruction, task_id=task.id,
            verification_fn=task.verification_fn)
        results.append(metrics.to_dict())
        print(f"    {task.id}: success_rate={metrics.success_rate:.2f} "
              f"sim={metrics.trajectory_similarity:.3f} var={metrics.variance_score:.3f}",
              flush=True)
    avg_success = sum(r["success_rate"] for r in results) / len(results)
    avg_similarity = sum(r["trajectory_similarity"] for r in results) / len(results)
    avg_variance = sum(r["variance_score"] for r in results) / len(results)
    return {"model": model_name, "num_runs": num_runs,
            "consistency_score": avg_success,
            "avg_success_rate": avg_success,
            "avg_trajectory_similarity": avg_similarity,
            "avg_variance": avg_variance, "per_task": results}


def run_robustness_evaluation(model_name, agent_runner, task_suite):
    evaluator = RobustnessEvaluator()
    tasks = task_suite.get_tasks_by_category(TaskCategory.INFORMATION_RETRIEVAL)[:2]
    results = []
    for task in tasks:
        metrics = evaluator.evaluate(
            model_name=model_name, agent_runner=agent_runner,
            task_description=task.instruction, task_id=task.id,
            verification_fn=task.verification_fn)
        results.append(metrics.to_dict())
        print(f"    {task.id}: baseline={'OK' if metrics.baseline_success else 'FAIL'} "
              f"rob={metrics.robustness_score:.3f} deg={metrics.degradation:.3f}",
              flush=True)
    avg_robustness = sum(r["robustness_score"] for r in results) / len(results)
    avg_degradation = sum(r["degradation"] for r in results) / len(results)
    return {"model": model_name, "robustness_score": avg_robustness,
            "avg_degradation": avg_degradation, "per_task": results}


def run_fault_tolerance_evaluation(model_name, agent_runner, task_suite, tool_registry):
    evaluator = FaultToleranceEvaluator(tool_registry)
    tasks = task_suite.get_tasks_by_category(TaskCategory.INFORMATION_RETRIEVAL)[:1]
    tasks += task_suite.get_tasks_by_category(TaskCategory.SCHEDULING)[:1]
    results = []
    for task in tasks:
        metrics = evaluator.evaluate(
            model_name=model_name, agent_runner_fn=agent_runner,
            task_description=task.instruction, task_id=task.id,
            verification_fn=task.verification_fn)
        results.append(metrics.to_dict())
        print(f"    {task.id}: baseline={'OK' if metrics.baseline_success else 'FAIL'} "
              f"ft={metrics.fault_tolerance_score:.3f}", flush=True)
    avg_ft = sum(r["fault_tolerance_score"] for r in results) / len(results)
    return {"model": model_name, "fault_tolerance_score": avg_ft, "per_task": results}


def run_safety_evaluation(model_name, agent_runner):
    evaluator = SafetyEvaluator()
    metrics = evaluator.evaluate(model_name=model_name, agent_runner=agent_runner)
    print(f"    safety={metrics.overall_safety_score:.3f} "
          f"refusal={metrics.refusal_rate:.3f} scope={metrics.scope_adherence:.3f}",
          flush=True)
    return {"model": model_name, "overall_safety_score": metrics.overall_safety_score,
            "refusal_rate": metrics.refusal_rate,
            "scope_adherence": metrics.scope_adherence,
            "per_test": [{"test_id": r["test_id"], "category": r["category"],
                          "safe": r["safe"]} for r in metrics.per_test_results]}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 70, flush=True)
    print("  RELIABILITY DIMENSION VERIFICATION RE-RUN", flush=True)
    print(f"  Started: {datetime.now().isoformat()}", flush=True)
    print("=" * 70, flush=True)

    tool_registry = build_tool_registry()
    task_suite = TaskSuite()

    state = {"started_at": datetime.now().isoformat(), "results": {}}
    if os.path.exists(STATE_PATH):
        try:
            state = json.load(open(STATE_PATH, encoding="utf-8"))
        except Exception:
            state = {"started_at": datetime.now().isoformat(), "results": {}}

    set_seed(42)
    for idx, model_id in enumerate(MODELS, 1):
        if model_id in state["results"]:
            print(f"[{idx}/{len(MODELS)}] {model_id} already done, skipping",
                  flush=True)
            continue
        print(f"\n{'#'*60}", flush=True)
        print(f"  [{idx}/{len(MODELS)}] {model_id}", flush=True)
        print(f"{'#'*60}", flush=True)
        try:
            config = ModelConfig(name=model_id, display_name=model_id,
                                 provider="ollama", model_id=model_id)
            model_fn = load_model(config)
        except Exception as e:
            print(f"  FAILED to load {model_id}: {e}", flush=True)
            state["results"][model_id] = {"error": str(e)}
            continue
        agent_runner = create_agent_runner(
            model_fn, tool_registry,
            AgentConfig(max_steps=12, verbose=False, seed=42))
        print("  CONSISTENCY", flush=True)
        consistency = run_consistency_evaluation(model_id, agent_runner, task_suite, 3)
        print("  ROBUSTNESS", flush=True)
        robustness = run_robustness_evaluation(model_id, agent_runner, task_suite)
        print("  FAULT TOLERANCE", flush=True)
        ft = run_fault_tolerance_evaluation(model_id, agent_runner, task_suite, tool_registry)
        print("  SAFETY", flush=True)
        safety = run_safety_evaluation(model_id, agent_runner)
        state["results"][model_id] = {
            "consistency": consistency, "robustness": robustness,
            "fault_tolerance": ft, "safety": safety,
            "done_at": datetime.now().isoformat()}
        with open(STATE_PATH, "w") as f:
            json.dump(state, f, indent=2)
        print(f"  [saved] {model_id}", flush=True)

    print(f"\n{'='*70}", flush=True)
    print("  DIM VERIFICATION SUMMARY", flush=True)
    print(f"{'='*70}", flush=True)
    for m in MODELS:
        r = state["results"].get(m)
        if not r or "consistency" not in r:
            print(f"  {m:<20} ERROR/INCOMPLETE", flush=True)
            continue
        print(f"  {m:<20} cons={r['consistency']['consistency_score']*100:.1f} "
              f"rob={r['robustness']['robustness_score']*100:.1f} "
              f"ft={r['fault_tolerance']['fault_tolerance_score']*100:.1f} "
              f"saf={r['safety']['overall_safety_score']*100:.1f}", flush=True)
    print(f"\n  Saved to: {STATE_PATH}", flush=True)
    print("  COMPLETE", flush=True)


if __name__ == "__main__":
    main()
