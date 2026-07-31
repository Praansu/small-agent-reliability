#!/usr/bin/env python3
"""
Temperature sensitivity analysis for top models.
Runs capability evaluation at t=0.3, 0.7, 1.0 on top 3 models.
Also collects latency/token data for cost-reliability tradeoff curves.
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

# Top 3 models + temperatures (t=0.0 baseline comes from v2 capability run)
MODELS = ["qwen2.5-coder:7b", "qwen2.5:7b", "mistral:7b"]
TEMPERATURES = [0.3, 0.7, 1.0]

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
        name=model_id,
        display_name=model_id,
        provider="ollama",
        model_id=model_id,
        temperature=temperature,
    )
    set_seed(42)
    
    try:
        model_fn = load_model(config)
    except Exception as e:
        return {"model": model_id, "temperature": temperature, "error": str(e), "accuracy": 0}
    
    agent = ReActAgent(model_fn, tool_registry, AgentConfig(max_steps=12, verbose=False, seed=42))
    
    tasks = task_suite.get_all_tasks()
    results = []
    total_tokens = 0
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
        
        # Track tokens from metadata
        steps = getattr(result, 'steps', [])
        for s in steps:
            if hasattr(s, 'response') and s.response:
                pass  # Ollama metadata not easily accessible per-step
        
        elapsed = time.time() - start
        total_duration += elapsed
        
        status = "PASS" if eval_result.correctness else ("PART" if eval_result.score > 0 else "FAIL")
        emoji = "✓" if eval_result.correctness else ("~" if eval_result.score > 0 else "✗")
        print(f"  {emoji} {task.id} {status} score={eval_result.score:.2f} ({elapsed:.1f}s)")
    
    # Aggregate
    aggregated = evaluator.aggregate_scores(results)
    
    print(f"  → Accuracy: {aggregated['accuracy']*100:.1f}%, Avg time: {total_duration/len(tasks):.1f}s")
    
    return {
        "model": model_id,
        "temperature": temperature,
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
    print("=" * 70)
    print("  TEMPERATURE SENSITIVITY ANALYSIS")
    print(f"  Models: {', '.join(MODELS)}")
    print(f"  Temperatures: {TEMPERATURES}")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)
    
    tool_registry = build_tool_registry()
    task_suite = TaskSuite()
    evaluator = TaskEvaluator()
    
    all_results = []
    total_runs = len(MODELS) * len(TEMPERATURES)
    run_count = 0
    
    for model_id in MODELS:
        for temp in TEMPERATURES:
            run_count += 1
            print(f"\n{'#'*60}")
            print(f"  [{run_count}/{total_runs}] {model_id} @ t={temp}")
            print(f"{'#'*60}")
            
            result = run_capability(model_id, temp, task_suite, evaluator, tool_registry)
            all_results.append(result)
    
    # Save results
    out_dir = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "temperature_sweep.json")
    
    output = {
        "experiment_date": datetime.now().isoformat(),
        "models": MODELS,
        "temperatures": TEMPERATURES,
        "results": all_results,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    
    # Print summary table
    print(f"\n{'='*70}")
    print(f"  TEMPERATURE SENSITIVITY SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Model':<22} {'t=0.0':>8} {'t=0.3':>8} {'t=0.7':>8} {'t=1.0':>8} {'Deg.@1.0':>8}")
    print(f"  {'-'*22} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    
    for model_id in MODELS:
        model_results = [r for r in all_results if r["model"] == model_id]
        baseline = next((r for r in model_results if r["temperature"] == 0.0), None)
        accs = {}
        for r in model_results:
            accs[r["temperature"]] = r["accuracy"] * 100
        
        baseline_acc = accs.get(0.0, 0)
        deg_10 = baseline_acc - accs.get(1.0, baseline_acc)
        print(f"  {model_id:<22} {accs.get(0.0, 0):>7.1f}% {accs.get(0.3, 0):>7.1f}% "
              f"{accs.get(0.7, 0):>7.1f}% {accs.get(1.0, 0):>7.1f}% {deg_10:>7.1f}%")
    
    print(f"\n  Saved to: {out_path}")
    print(f"\n{'='*70}")
    print("  TEMPERATURE ANALYSIS COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()
