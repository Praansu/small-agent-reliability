#!/usr/bin/env python3
"""
Temperature sweep (Phase 2, v2) — crash-safe + Ollama-watchdog version.

Fixes the failure mode where Ollama died mid-sweep and produced ~4s
connection-failure runs at 0% for every remaining model.

Design:
- ensure_ollama(): health-check localhost:11434 before EVERY task; if down,
  restart `ollama serve` (detached) and wait until healthy (up to 120s).
- Task retry: if a task's result contains a connection error, retry once
  after ensure_ollama().
- Per-run checkpoint to data/processed/temperature_sweep.json after each run.
- Seed: run 1 (qwen2.5-coder:7b @ t=0.3 = 67.7%) recorded from the original
  log on 2026-07-31 before that process was killed externally.
"""
import json, os, subprocess, sys, time, requests
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

TEMP_MODELS = ["qwen2.5-coder:7b", "qwen2.5:7b", "mistral:7b"]
TEMPERATURES = [0.3, 0.7, 1.0]

BASE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.normpath(os.path.join(BASE, "..", "data", "processed"))
CHECKPOINT = os.path.join(OUT_DIR, "temperature_sweep.json")
LOG_FILE = os.path.join(os.environ.get("TEMP", "/tmp"), "v2_sweep_resume.log")
OLLAMA_URL = "http://localhost:11434/api/version"


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def ollama_healthy():
    try:
        r = requests.get(OLLAMA_URL, timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def ensure_ollama(max_wait=120):
    """Start ollama serve if not running; wait until /api/version responds."""
    if ollama_healthy():
        return True
    log("  WATCHDOG: Ollama down, attempting restart...")
    try:
        subprocess.Popen(["ollama", "serve"],
                         creationflags=subprocess.CREATE_NO_WINDOW,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        log(f"  WATCHDOG: restart command failed: {e}")
    waited = 0
    while waited < max_wait:
        time.sleep(5)
        waited += 5
        if ollama_healthy():
            log(f"  WATCHDOG: Ollama healthy after {waited}s")
            return True
    log(f"  WATCHDOG: Ollama NOT healthy after {max_wait}s")
    return False


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


def is_conn_failure(result, eval_result, error_str):
    """Heuristic: connection failure = error mentions ollama/connection, or
    empty agent output with error set."""
    if error_str:
        low = error_str.lower()
        if any(k in low for k in ("ollama", "connection", "refused", "timeout", "requests")):
            return True
    return False


def run_capability(model_id, temperature, task_suite, evaluator, tool_registry):
    if not ensure_ollama():
        log(f"  WATCHDOG: cannot start Ollama before {model_id} t={temperature}; aborting run")
        return {"model": model_id, "temperature": temperature,
                "error": "Ollama unavailable", "accuracy": 0, "per_task": [],
                "avg_duration_s": 0, "total_duration_s": 0}

    config = ModelConfig(
        name=model_id, display_name=model_id, provider="ollama",
        model_id=model_id, temperature=temperature,
    )
    set_seed(42)
    try:
        model_fn = load_model(config)
    except Exception as e:
        log(f"  MODEL LOAD FAILED {model_id} t={temperature}: {e}")
        return {"model": model_id, "temperature": temperature, "error": str(e),
                "accuracy": 0, "success_rate": 0, "average_score": 0,
                "error_rate": 0, "avg_duration_s": 0, "total_duration_s": 0,
                "per_task": []}

    agent = ReActAgent(model_fn, tool_registry, AgentConfig(max_steps=12, verbose=False, seed=42))
    tasks = task_suite.get_all_tasks()
    results = []
    total_duration = 0.0
    retries = 0

    for i, task in enumerate(tasks):
        # Watchdog: guarantee Ollama is up before each task
        if not ollama_healthy():
            ensure_ollama()
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

        # Retry once on connection failure (Ollama may have just crashed)
        if is_conn_failure(result, eval_result, error_str) and retries < len(tasks):
            retries += 1
            log(f"  RETRY {task.id}: connection failure, restarting Ollama")
            ensure_ollama()
            time.sleep(3)
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
        log(f"    {i+1}/{len(tasks)} {task.id} {status} score={eval_result.score:.2f} ({elapsed:.1f}s)")

    aggregated = evaluator.aggregate_scores(results)
    log(f"  -> Accuracy: {aggregated['accuracy']*100:.1f}%  Avg time: {total_duration/len(tasks):.1f}s  (retries={retries})")

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
        "num_retries": retries,
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


def seed_run1():
    return {
        "model": "qwen2.5-coder:7b",
        "temperature": 0.3,
        "num_tasks": 31,
        "accuracy": 0.677,
        "success_rate": None,
        "average_score": None,
        "error_rate": None,
        "avg_duration_s": 17.5,
        "total_duration_s": 17.5 * 31,
        "per_task": [],
        "seeded_from_log": True,
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    log("=" * 70)
    log("  TEMPERATURE SWEEP (watchdog version)")
    log(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load existing checkpoint; if it contains broken all-zero runs from the
    # Ollama-crash episode, strip them and re-run.
    sweep = None
    if os.path.exists(CHECKPOINT):
        with open(CHECKPOINT, "r", encoding="utf-8") as f:
            sweep = json.load(f)
        # Remove runs that are connection-failure artifacts (accuracy==0 and
        # every task < 6s) EXCEPT the seeded one.
        kept = []
        for r in sweep["results"]:
            if r.get("seeded_from_log"):
                kept.append(r)
                continue
            pt = r.get("per_task", [])
            fast_fail = len(pt) > 0 and all(t.get("duration_ms", 0) < 6000 for t in pt) and r.get("accuracy", 1) == 0
            if fast_fail:
                log(f"  Dropping corrupt run: {r['model']} t={r['temperature']} (Ollama crash artifact)")
            else:
                kept.append(r)
        sweep["results"] = kept
        with open(CHECKPOINT, "w", encoding="utf-8") as f:
            json.dump(sweep, f, indent=2)
        log(f"  Checkpoint loaded: {len(kept)} valid runs")

    if sweep is None:
        sweep = {
            "experiment_date": datetime.now().isoformat(),
            "models": TEMP_MODELS,
            "temperatures": TEMPERATURES,
            "num_tasks": 31,
            "results": [seed_run1()],
        }
        with open(CHECKPOINT, "w", encoding="utf-8") as f:
            json.dump(sweep, f, indent=2)
        log("  Seeded run 1 (qwen2.5-coder:7b @ t=0.3 = 67.7%) from original log")

    done = {(r["model"], r["temperature"]) for r in sweep["results"]}
    todo = [(m, t) for m in TEMP_MODELS for t in TEMPERATURES if (m, t) not in done]
    log(f"  Already done: {len(done)}/9 | Remaining: {len(todo)} runs")
    if not todo:
        log("  All runs complete. Nothing to do.")
        return

    if not ensure_ollama():
        log("  FATAL: Ollama cannot start. Aborting.")
        return

    tool_registry = build_tool_registry()
    task_suite = TaskSuite()
    evaluator = TaskEvaluator()
    tasks = task_suite.get_all_tasks()
    log(f"  Task suite: {len(tasks)} tasks")

    for model_id, temp in todo:
        log(f"  [{len(done)+1}/9] {model_id} @ t={temp}")
        start_run = time.time()
        try:
            result = run_capability(model_id, temp, task_suite, evaluator, tool_registry)
        except Exception as e:
            log(f"  RUN CRASHED: {e}")
            result = {"model": model_id, "temperature": temp, "error": str(e),
                      "accuracy": 0, "per_task": []}
        sweep["results"].append(result)
        with open(CHECKPOINT, "w", encoding="utf-8") as f:
            json.dump(sweep, f, indent=2)
        done.add((model_id, temp))
        log(f"  -> Checkpoint saved ({len(done)}/9), run took {time.time()-start_run:.0f}s")

    log("=" * 70)
    log("  SWEEP COMPLETE - all runs saved to data/processed/temperature_sweep.json")
    log("=" * 70)


if __name__ == "__main__":
    main()
