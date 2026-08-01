#!/usr/bin/env python3
"""
One-off fix: re-run qwen2.5-coder:7b @ t=0.7 and replace the contaminated
checkpoint entry (it was partially written when Ollama crashed — only 6 of 31
tasks were real, the rest were 4s connection failures).

Uses the same watchdog (ensure_ollama per task + retry on connection error)
as resume_temp_sweep.py.
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

BASE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.normpath(os.path.join(BASE, "..", "data", "processed"))
CHECKPOINT = os.path.join(OUT_DIR, "temperature_sweep.json")
LOG_FILE = os.path.join(os.environ.get("TEMP", "/tmp"), "v2_sweep_fix.log")
OLLAMA_URL = "http://localhost:11434/api/version"
MODEL = "qwen2.5-coder:7b"
TEMP = 0.7


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
    if ollama_healthy():
        return True
    log("  WATCHDOG: Ollama down, restarting...")
    try:
        subprocess.Popen(["ollama", "serve"],
                         creationflags=subprocess.CREATE_NO_WINDOW,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        log(f"  restart failed: {e}")
    waited = 0
    while waited < max_wait:
        time.sleep(5)
        waited += 5
        if ollama_healthy():
            log(f"  Ollama healthy after {waited}s")
            return True
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


def run_one():
    ensure_ollama()
    config = ModelConfig(name=MODEL, display_name=MODEL, provider="ollama",
                         model_id=MODEL, temperature=TEMP)
    set_seed(42)
    model_fn = load_model(config)
    agent = ReActAgent(model_fn, build_tool_registry(),
                       AgentConfig(max_steps=12, verbose=False, seed=42))
    suite = TaskSuite()
    evaluator = TaskEvaluator()
    tasks = suite.get_all_tasks()
    results = []
    total = 0.0
    retries = 0
    for i, task in enumerate(tasks):
        if not ollama_healthy():
            ensure_ollama()
        start = time.time()
        err = None
        try:
            result = agent.run(task.instruction, task_id=task.id)
        except Exception as e:
            result = None
            err = str(e)
        if result is None:
            ev = evaluator.evaluate(task=task, agent_final_answer="", agent_success=False,
                                    steps_taken=0, duration_ms=0, error=err)
        else:
            ev = evaluator.evaluate(task=task, agent_final_answer=result.final_answer,
                                    agent_success=result.success, steps_taken=len(result.steps),
                                    duration_ms=result.total_duration_ms, error=result.error)
        # retry once on connection failure
        if err and any(k in err.lower() for k in ("ollama", "connection", "refused", "timeout")):
            retries += 1
            log(f"  RETRY {task.id}")
            ensure_ollama()
            time.sleep(3)
            try:
                result = agent.run(task.instruction, task_id=task.id)
                ev = evaluator.evaluate(task=task, agent_final_answer=result.final_answer,
                                        agent_success=result.success, steps_taken=len(result.steps),
                                        duration_ms=result.total_duration_ms, error=result.error)
            except Exception as e:
                ev = evaluator.evaluate(task=task, agent_final_answer="", agent_success=False,
                                        steps_taken=0, duration_ms=0, error=str(e))
        results.append(ev)
        elapsed = time.time() - start
        total += elapsed
        st = "PASS" if ev.correctness else ("PART" if ev.score > 0 else "FAIL")
        log(f"    {i+1}/{len(tasks)} {task.id} {st} score={ev.score:.2f} ({elapsed:.1f}s)")
    agg = evaluator.aggregate_scores(results)
    log(f"  -> Accuracy: {agg['accuracy']*100:.1f}%  Avg: {total/len(tasks):.1f}s  retries={retries}")
    return {
        "model": MODEL, "temperature": TEMP, "num_tasks": len(tasks),
        "accuracy": agg["accuracy"], "success_rate": agg["success_rate"],
        "average_score": agg["average_score"], "error_rate": agg["error_rate"],
        "avg_duration_s": total / len(tasks), "total_duration_s": total,
        "num_retries": retries,
        "per_task": [{"task_id": r.task_id, "description": r.task_description[:50],
                      "correctness": r.correctness, "score": r.score,
                      "steps": r.steps_taken, "duration_ms": r.duration_ms}
                     for r in results],
    }


def main():
    log("FIX RUN: " + MODEL + " @ t=" + str(TEMP))
    new_result = run_one()
    with open(CHECKPOINT, "r", encoding="utf-8") as f:
        sweep = json.load(f)
    sweep["results"] = [r for r in sweep["results"]
                        if not (r["model"] == MODEL and r.get("temperature") == TEMP
                                and not r.get("seeded_from_log"))]
    sweep["results"].append(new_result)
    sweep["results"].sort(key=lambda r: (r["model"], r.get("temperature", 0)))
    with open(CHECKPOINT, "w", encoding="utf-8") as f:
        json.dump(sweep, f, indent=2)
    log("Checkpoint updated and sorted. FIX COMPLETE.")


if __name__ == "__main__":
    main()
