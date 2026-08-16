#!/usr/bin/env python3
"""
Temperature sweep VERIFICATION re-run.
Mirrors code/temperature_sweep.py protocol exactly (same models, temps, tools,
agent config, seed) but writes to data/raw/verify_temp/ with incremental
per-pair saving so results survive crashes. Output never touches original data.
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
from tasks.task_suite import TaskSuite
from tasks.evaluator import TaskEvaluator

MODELS = ["qwen2.5-coder:7b", "qwen2.5:7b", "mistral:7b"]
TEMPERATURES = [0.3, 0.7, 1.0]
OUT_DIR = os.path.join(ROOT, "data", "raw", "verify_temp")
STATE_PATH = os.path.join(OUT_DIR, "sweep_results.json")


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
        return {"model": model_id, "temperature": temperature,
                "error": str(e), "accuracy": 0, "per_task": []}

    agent = ReActAgent(model_fn, tool_registry,
                       AgentConfig(max_steps=12, verbose=False, seed=42))
    tasks = task_suite.get_all_tasks()
    results = []
    total_duration = 0
    for task in tasks:
        start = time.time()
        try:
            result = agent.run(task.instruction, task_id=task.id)
        except Exception as exc:
            result = None
            error_str = str(exc)
        else:
            error_str = None
        if result is None:
            eval_result = evaluator.evaluate(
                task=task, agent_final_answer="", agent_success=False,
                steps_taken=0, duration_ms=0, error=error_str)
        else:
            eval_result = evaluator.evaluate(
                task=task, agent_final_answer=result.final_answer,
                agent_success=result.success,
                steps_taken=len(result.steps),
                duration_ms=result.total_duration_ms,
                error=result.error)
        results.append(eval_result)
        elapsed = time.time() - start
        total_duration += elapsed
        status = ("PASS" if eval_result.correctness
                  else ("PART" if eval_result.score > 0 else "FAIL"))
        print(f"  [{datetime.now().strftime('%H:%M:%S')}] {task.id} {status} "
              f"score={eval_result.score:.2f} ({elapsed:.1f}s)", flush=True)

    aggregated = evaluator.aggregate_scores(results)
    print(f"  -> Accuracy: {aggregated['accuracy']*100:.1f}%, "
          f"Avg time: {total_duration/len(tasks):.1f}s", flush=True)
    return {
        "model": model_id, "temperature": temperature,
        "accuracy": aggregated["accuracy"],
        "success_rate": aggregated["success_rate"],
        "average_score": aggregated["average_score"],
        "error_rate": aggregated["error_rate"],
        "avg_duration_s": total_duration / len(tasks),
        "total_duration_s": total_duration,
        "per_task": [{"task_id": r.task_id,
                      "correctness": r.correctness,
                      "score": r.score,
                      "steps": r.steps_taken,
                      "duration_ms": r.duration_ms}
                     for r in results],
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 70, flush=True)
    print("  TEMPERATURE SWEEP VERIFICATION RE-RUN", flush=True)
    print(f"  Started: {datetime.now().isoformat()}", flush=True)
    print(f"  Models: {MODELS}  Temps: {TEMPERATURES}", flush=True)
    print("=" * 70, flush=True)

    tool_registry = build_tool_registry()
    task_suite = TaskSuite()
    evaluator = TaskEvaluator()

    # resume from state if it exists
    state = {"started_at": datetime.now().isoformat(), "results": []}
    if os.path.exists(STATE_PATH):
        try:
            state = json.load(open(STATE_PATH, encoding="utf-8"))
        except Exception:
            state = {"started_at": datetime.now().isoformat(), "results": []}
    done = {(r["model"], r["temperature"]) for r in state["results"]}

    total_runs = len(MODELS) * len(TEMPERATURES)
    run_count = len(done)
    for model_id in MODELS:
        for temp in TEMPERATURES:
            if (model_id, temp) in done:
                continue
            run_count += 1
            print(f"\n{'#'*60}", flush=True)
            print(f"  [{run_count}/{total_runs}] {model_id} @ t={temp}",
                  flush=True)
            print(f"{'#'*60}", flush=True)
            result = run_capability(model_id, temp, task_suite, evaluator,
                                    tool_registry)
            state["results"].append(result)
            with open(STATE_PATH, "w") as f:
                json.dump(state, f, indent=2)
            print(f"  [saved] {model_id}@t={temp} -> "
                  f"{result['accuracy']*100:.1f}%", flush=True)

    print(f"\n{'='*70}", flush=True)
    print("  TEMP SWEEP VERIFICATION SUMMARY", flush=True)
    print(f"{'='*70}", flush=True)
    for r in state["results"]:
        print(f"  {r['model']:<22} t={r['temperature']}  "
              f"acc={r['accuracy']*100:.1f}%", flush=True)
    print(f"\n  Saved to: {STATE_PATH}", flush=True)
    print("  COMPLETE", flush=True)


if __name__ == "__main__":
    main()
