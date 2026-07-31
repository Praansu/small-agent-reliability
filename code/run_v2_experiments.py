#!/usr/bin/env python3
"""
V2 experiment orchestrator: re-runs capability evaluation on ALL 9 models
with the expanded 31-task suite, then runs temperature sweep on top 3.

Saves results to data/raw/v2/ and data/processed/temperature_sweep.json
"""
import json, os, sys, time
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
TEMP_MODELS = ["qwen2.5-coder:7b", "qwen2.5:7b", "mistral:7b"]
TEMPERATURES = [0.3, 0.7, 1.0]

BASE = os.path.dirname(os.path.abspath(__file__))
V2_DIR = os.path.join(BASE, "..", "data", "raw", "v2")
os.makedirs(V2_DIR, exist_ok=True)


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
    """Run capability evaluation at a specific temperature."""
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
    print("=" * 70, flush=True)
    print("  V2 EXPERIMENTS: 9-model capability on 31 tasks + temperature sweep", flush=True)
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M')}", flush=True)
    print("=" * 70, flush=True)

    tool_registry = build_tool_registry()
    task_suite = TaskSuite()
    evaluator = TaskEvaluator()
    tasks = task_suite.get_all_tasks()
    print(f"  Task suite: {len(tasks)} tasks across "
          f"{len(task_suite.get_summary()['categories'])} categories", flush=True)

    # ---- PHASE 1: capability on all 9 models (t=0.0) ----
    print("\n" + "#" * 70, flush=True)
    print("  PHASE 1: CAPABILITY (t=0.0) - ALL 9 MODELS ON 31 TASKS", flush=True)
    print("#" * 70, flush=True)

    capability_results = {}
    for i, model_id in enumerate(ALL_MODELS):
        print(f"\n  [{i+1}/9] {model_id}", flush=True)
        try:
            result = run_capability(model_id, 0.0, task_suite, evaluator, tool_registry)
        except Exception as e:
            print(f"  FAILED {model_id}: {e}", flush=True)
            result = {"model": model_id, "temperature": 0.0, "error": str(e), "accuracy": 0}
        capability_results[model_id] = result

        # Save incrementally
        safe = model_id.replace(":", "_")
        with open(os.path.join(V2_DIR, f"capability_{safe}.json"), "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved v2 capability for {model_id}", flush=True)

    # ---- PHASE 2: temperature sweep on top 3 ----
    print("\n" + "#" * 70, flush=True)
    print("  PHASE 2: TEMPERATURE SWEEP (t=0.3/0.7/1.0)", flush=True)
    print("#" * 70, flush=True)

    temp_results = []
    total_runs = len(TEMP_MODELS) * len(TEMPERATURES)
    run_count = 0
    for model_id in TEMP_MODELS:
        for temp in TEMPERATURES:
            run_count += 1
            print(f"\n  [{run_count}/{total_runs}] {model_id} @ t={temp}", flush=True)
            try:
                result = run_capability(model_id, temp, task_suite, evaluator, tool_registry)
            except Exception as e:
                print(f"  FAILED: {e}", flush=True)
                result = {"model": model_id, "temperature": temp, "error": str(e), "accuracy": 0}
            temp_results.append(result)

    # Save temperature sweep
    out_dir = os.path.join(BASE, "..", "data", "processed")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "temperature_sweep.json"), "w") as f:
        json.dump({
            "experiment_date": datetime.now().isoformat(),
            "models": TEMP_MODELS,
            "temperatures": TEMPERATURES,
            "num_tasks": len(tasks),
            "results": temp_results,
        }, f, indent=2)

    # Save capability v2 aggregate
    with open(os.path.join(V2_DIR, "aggregate_v2.json"), "w") as f:
        json.dump({
            "experiment_date": datetime.now().isoformat(),
            "num_models": len(capability_results),
            "num_tasks": len(tasks),
            "results": capability_results,
        }, f, indent=2)

    # Summary
    print("\n" + "=" * 70, flush=True)
    print("  V2 SUMMARY (31-task capability)", flush=True)
    print("=" * 70, flush=True)
    for model_id in ALL_MODELS:
        r = capability_results.get(model_id, {})
        acc = r.get("accuracy", 0) * 100
        err = r.get("error", "")
        print(f"  {model_id:<22} {acc:>6.1f}% {err}", flush=True)

    print("\n  TEMPERATURE SWEEP:", flush=True)
    for r in temp_results:
        print(f"  {r['model']:<22} t={r['temperature']:<4} acc={r['accuracy']*100:>6.1f}%", flush=True)

    print(f"\n*** V2 experiments complete: {datetime.now().strftime('%H:%M')} ***", flush=True)


if __name__ == "__main__":
    main()
