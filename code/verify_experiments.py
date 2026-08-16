#!/usr/bin/env python3
"""
VERIFICATION RE-RUN: re-executes the V2 capability protocol (9 models x 31 tasks,
t=0.0, seed 42, max_steps 12) and saves results to data/raw/verify/ WITHOUT
touching the original data. Run in background, poll the log.

Usage: python code/verify_experiments.py
"""
import json, os, sys, time, io
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from harness.agent import ReActAgent, AgentConfig
from harness.tools.base import ToolRegistry
from harness.tools.search import SearchTool, KnowledgeBaseTool
from harness.tools.calendar import CalendarTool, ScheduleManager
from harness.tools.data import DataQueryTool, CSVProcessor
from harness.tools.code import CodeInterpreterTool, FileOperator
from harness.tools.email import EmailTool, NotificationTool
from harness.utils import ModelConfig, load_model, set_seed
from tasks.task_suite import TaskSuite
from tasks.evaluator import TaskEvaluator

ALL_MODELS = [
    "qwen2.5-coder:7b", "qwen2.5:7b", "llama3.2:1b", "mistral:7b",
    "llama3.2:3b", "deepseek-r1:7b", "phi3.5:3.8b", "llama3.1:8b", "gemma2:9b",
]

BASE = os.path.dirname(os.path.abspath(__file__))
VERIFY_DIR = os.path.join(BASE, "..", "data", "raw", "verify")
os.makedirs(VERIFY_DIR, exist_ok=True)
LOG = os.path.join(VERIFY_DIR, "progress.json")


def log_state(state):
    with open(LOG, "w") as f:
        json.dump(state, f, indent=2)


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


def run_capability(model_id, temperature, task_suite, evaluator, tool_registry):
    config = ModelConfig(
        name=model_id, display_name=model_id, provider="ollama",
        model_id=model_id, temperature=temperature,
    )
    set_seed(42)
    try:
        model_fn = load_model(config)
    except Exception as e:
        return {"model": model_id, "temperature": temperature, "error": str(e), "accuracy": 0}

    agent = ReActAgent(model_fn, tool_registry, AgentConfig(max_steps=12, verbose=False, seed=42))
    tasks = task_suite.get_all_tasks()
    results = []
    total_duration = 0.0

    for task in tasks:
        start = time.time()
        error_str = None
        try:
            result = agent.run(task.instruction, task_id=task.id)
        except Exception as exc:
            result = None
            error_str = str(exc)

        if result is None:
            eval_result = evaluator.evaluate(
                task=task, agent_final_answer="", agent_success=False,
                steps_taken=0, duration_ms=0, error=error_str,
            )
        else:
            eval_result = evaluator.evaluate(
                task=task,
                agent_final_answer=result.final_answer,
                agent_success=result.success,
                steps_taken=len(result.steps),
                duration_ms=result.total_duration_ms,
                error=result.error,
            )
        results.append(eval_result)
        elapsed = time.time() - start
        total_duration += elapsed

        status = "PASS" if eval_result.correctness else ("PART" if eval_result.score > 0 else "FAIL")
        print(f"    {task.id} {status} score={eval_result.score:.2f} ({elapsed:.1f}s)", flush=True)

    aggregated = evaluator.aggregate_scores(results)
    print(f"  -> Accuracy: {aggregated['accuracy']*100:.1f}%  Avg time: {total_duration/len(tasks):.1f}s", flush=True)

    return {
        "model": model_id,
        "temperature": temperature,
        "num_tasks": len(tasks),
        "accuracy": aggregated["accuracy"],
        "success_rate": aggregated["success_rate"],
        "average_score": aggregated["average_score"],
        "error_rate": aggregated["error_rate"],
        "avg_duration_s": total_duration / len(tasks),
        "total_duration_s": total_duration,
        "per_task": [
            {
                "task_id": r.task_id,
                "description": r.task_description[:50],
                "correctness": r.correctness,
                "score": r.score,
                "steps": r.steps_taken,
                "duration_ms": r.duration_ms,
            }
            for r in results
        ],
    }


def main():
    state = {"started_at": datetime.now().isoformat(), "completed": [], "current": None}
    log_state(state)
    print("=" * 70, flush=True)
    print("  VERIFICATION RE-RUN: 9 models x 31 tasks (t=0.0, seed 42)", flush=True)
    print("  Saving to data/raw/verify/ (originals untouched)", flush=True)
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M')}", flush=True)
    print("=" * 70, flush=True)

    tool_registry = build_tool_registry()
    task_suite = TaskSuite()
    evaluator = TaskEvaluator()
    tasks = task_suite.get_all_tasks()
    print(f"  Task suite: {len(tasks)} tasks", flush=True)

    capability_results = {}
    for i, model_id in enumerate(ALL_MODELS):
        print(f"\n  [{i+1}/{len(ALL_MODELS)}] {model_id}", flush=True)
        state["current"] = model_id
        log_state(state)
        try:
            result = run_capability(model_id, 0.0, task_suite, evaluator, tool_registry)
        except Exception as e:
            print(f"  FAILED {model_id}: {e}", flush=True)
            result = {"model": model_id, "temperature": 0.0, "error": str(e), "accuracy": 0}
        capability_results[model_id] = result

        safe = model_id.replace(":", "_")
        with open(os.path.join(VERIFY_DIR, f"capability_{safe}.json"), "w") as f:
            json.dump(result, f, indent=2)
        state["completed"].append(model_id)
        state["current"] = None
        log_state(state)
        print(f"  Saved verify capability for {model_id}", flush=True)

    # aggregate
    with open(os.path.join(VERIFY_DIR, "aggregate_verify.json"), "w") as f:
        json.dump({
            "experiment_date": datetime.now().isoformat(),
            "num_models": len(capability_results),
            "num_tasks": len(tasks),
            "protocol": {"temperature": 0.0, "seed": 42, "max_steps": 12},
            "results": capability_results,
        }, f, indent=2)

    print("\n" + "=" * 70, flush=True)
    print("  VERIFICATION SUMMARY", flush=True)
    print("=" * 70, flush=True)
    for model_id in ALL_MODELS:
        r = capability_results.get(model_id, {})
        acc = r.get("accuracy", 0) * 100
        err = r.get("error", "")
        print(f"  {model_id:<22} {acc:>6.1f}% {err}", flush=True)
    print(f"\n*** Verification complete: {datetime.now().strftime('%H:%M')} ***", flush=True)


if __name__ == "__main__":
    main()
