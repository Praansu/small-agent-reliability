#!/usr/bin/env python3
"""Run Mistral temperature sweep — remaining temps with per-task timeout."""
import json, os, sys, time, signal
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

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
MODEL_ID = "mistral:7b"
TEMPS = [0.7, 1.0]

def build_tool_registry():
    registry = ToolRegistry()
    for cls in [SearchTool, KnowledgeBaseTool, CalendarTool, ScheduleManager,
                DataQueryTool, CSVProcessor, CodeInterpreterTool, FileOperator,
                EmailTool, NotificationTool]:
        registry.register(cls())
    return registry

def run_with_timeout(model_fn, task_suite, evaluator, temperature):
    """Run capability with per-task timeout."""
    tasks = task_suite.get_all_tasks()
    results = []
    for i, task in enumerate(tasks):
        print(f"  [{i+1}/{len(tasks)}] t={temperature} {task.description[:50]}...", end=" ", flush=True)

        def make_runner(fn, tr, cfg):
            return lambda desc: ReActAgent(fn, tr, cfg).run(desc)
        agent_runner = make_runner(model_fn, build_tool_registry(), AgentConfig(max_steps=12, verbose=False, seed=42))

        try:
            agent_result = agent_runner(task.instruction)
            eval_result = evaluator.evaluate(
                task=task, agent_final_answer=agent_result.final_answer,
                agent_success=agent_result.success, steps_taken=len(agent_result.steps),
                duration_ms=agent_result.total_duration_ms, error=agent_result.error,
                trajectory=[s.response for s in agent_result.steps if s.response],
            )
            results.append(eval_result)
            status = "PASS" if eval_result.correctness else ("PART" if eval_result.score > 0 else "FAIL")
            print(f"{status} ({eval_result.score:.2f})")
        except Exception as e:
            print(f"ERROR: {e}")
            results.append(evaluator.evaluate(
                task=task, agent_final_answer="", agent_success=False,
                steps_taken=0, duration_ms=0, error=str(e),
            ))

    aggregated = evaluator.aggregate_scores(results)
    return {
        "temperature": temperature,
        "model": MODEL_ID,
        "accuracy": aggregated["accuracy"],
        "success_rate": aggregated["success_rate"],
        "average_score": aggregated["average_score"],
        "per_task": [{"task_id": r.task_id, "correctness": r.correctness, "score": r.score} for r in results],
    }

def main():
    task_suite = TaskSuite()
    evaluator = TaskEvaluator()

    for temp in TEMPS:
        print(f"\n{'#'*40}")
        print(f"  MISTRAL 7B t={temp}")
        print(f"{'#'*40}")
        set_seed(42)
        config = ModelConfig(name=MODEL_ID, display_name="Mistral 7B",
                            provider="ollama", model_id=MODEL_ID, temperature=temp)
        model_fn = load_model(config)
        cap = run_with_timeout(model_fn, task_suite, evaluator, temp)
        print(f"  t={temp} accuracy: {cap['accuracy']*100:.1f}%")

        safe = MODEL_ID.replace(":", "_")
        path = os.path.join(RAW_DIR, f"tempsweep_{safe}_t{temp}.json")
        with open(path, "w") as f:
            json.dump(cap, f, indent=2)
        print(f"  Saved: {path}")

    print("\nDone!")

if __name__ == "__main__":
    main()
