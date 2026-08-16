#!/usr/bin/env python3
"""
STABILITY TEST: re-run each task that flipped between original and verify runs,
3 fresh attempts each, to distinguish stochastic noise from systematic change.

Usage: python code/verify_stability.py
Saves: data/raw/verify/stability.json (incremental) + data/raw/verify/stability.log
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

# (model, task) pairs that flipped in the verify run
FLIPPED = [
    ("qwen2.5-coder:7b", "DM-3"), ("qwen2.5-coder:7b", "SAF-1"),
    ("qwen2.5:7b", "DM-2"), ("qwen2.5:7b", "DM-3"), ("qwen2.5:7b", "IR-1"),
    ("qwen2.5:7b", "MSR-3"), ("qwen2.5:7b", "SAF-1"),
    ("llama3.2:1b", "COD-3"), ("llama3.2:1b", "DA-3"), ("llama3.2:1b", "SCH-4"),
    ("mistral:7b", "SAF-4"), ("mistral:7b", "SCH-4"),
    ("deepseek-r1:7b", "COD-3"), ("deepseek-r1:7b", "DM-1"), ("deepseek-r1:7b", "MSR-3"),
    ("phi3.5:3.8b", "DM-3"),
    ("llama3.1:8b", "DA-3"),
    ("gemma2:9b", "DA-3"), ("gemma2:9b", "IR-5"), ("gemma2:9b", "SCH-2"),
]
REPS = 3

BASE = os.path.dirname(os.path.abspath(__file__))
VERIFY_DIR = os.path.join(BASE, "..", "data", "raw", "verify")
os.makedirs(VERIFY_DIR, exist_ok=True)
STATE = os.path.join(VERIFY_DIR, "stability.json")


def load_state():
    if os.path.exists(STATE):
        return json.load(open(STATE, encoding="utf-8"))
    return {"started_at": datetime.now().isoformat(), "results": {}}


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


def main():
    state = load_state()
    print(f"STABILITY TEST: {len(FLIPPED)} flipped pairs x {REPS} reps", flush=True)
    registry = build_tool_registry()
    suite = TaskSuite()
    evaluator = TaskEvaluator()
    tasks_by_id = {t.id: t for t in suite.get_all_tasks()}

    agents = {}
    for model_id in sorted(set(m for m, _ in FLIPPED)):
        config = ModelConfig(name=model_id, display_name=model_id, provider="ollama",
                             model_id=model_id, temperature=0.0)
        set_seed(42)
        model_fn = load_model(config)
        agents[model_id] = ReActAgent(model_fn, registry, AgentConfig(max_steps=12, verbose=False, seed=42))
        print(f"  loaded {model_id}", flush=True)

    for (model_id, task_id) in FLIPPED:
        key = f"{model_id}|{task_id}"
        if key in state["results"]:
            print(f"  skip {key} (done)", flush=True)
            continue
        task = tasks_by_id[task_id]
        outcomes = []
        for rep in range(REPS):
            start = time.time()
            try:
                res = agents[model_id].run(task.instruction, task_id=task.id)
                er = evaluator.evaluate(task=task, agent_final_answer=res.final_answer,
                                        agent_success=res.success, steps_taken=len(res.steps),
                                        duration_ms=res.total_duration_ms, error=res.error)
            except Exception as e:
                er = evaluator.evaluate(task=task, agent_final_answer="", agent_success=False,
                                        steps_taken=0, duration_ms=0, error=str(e))
            outcomes.append({"rep": rep + 1, "correct": er.correctness, "score": er.score,
                             "time_s": round(time.time() - start, 1)})
            print(f"  {key} rep{rep+1}: {'PASS' if er.correctness else 'FAIL'} "
                  f"score={er.score:.2f} ({outcomes[-1]['time_s']}s)", flush=True)
        state["results"][key] = {"model": model_id, "task": task_id, "outcomes": outcomes}
        with open(STATE, "w") as f:
            json.dump(state, f, indent=2)

    # summary
    print("\n=== STABILITY SUMMARY (3 reps each) ===", flush=True)
    for (model_id, task_id) in FLIPPED:
        r = state["results"].get(f"{model_id}|{task_id}")
        if not r:
            continue
        o = [x["correct"] for x in r["outcomes"]]
        print(f"  {model_id:<20} {task_id:<6} {'/'.join('P' if c else 'F' for c in o)}", flush=True)
    print(f"\n*** Stability complete: {datetime.now().strftime('%H:%M')} ***", flush=True)


if __name__ == "__main__":
    main()
