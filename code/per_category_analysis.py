#!/usr/bin/env python3
"""Per-category breakdown and temperature sensitivity analysis."""

import json, os, sys, statistics, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE, "..", "data", "raw")
PROC_DIR = os.path.join(BASE, "..", "data", "processed")
FIG_DIR = os.path.join(BASE, "..", "paper", "figures")
os.makedirs(PROC_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

COLORS = {
    "llama3.2:1b": "#E74C3C", "llama3.2:3b": "#C0392B",
    "phi3.5:3.8b": "#8E44AD", "deepseek-r1:7b": "#2C3E50",
    "qwen2.5-coder:7b": "#27AE60", "qwen2.5:7b": "#2ECC71",
    "mistral:7b": "#2980B9", "llama3.1:8b": "#3498DB",
    "gemma2:9b": "#F39C12",
}

CATEGORY_MAP = {
    "information_retrieval": "Info Retrieval",
    "scheduling": "Scheduling",
    "data_analysis": "Data Analysis",
    "communication": "Communication",
    "multi_step_reasoning": "Multi-Step",
    "decision_making": "Decision Making",
    "coding": "Coding",
    "safety": "Safety",
}

DISPLAY = {
    "llama3.2:1b": "Llama 3.2 1B", "llama3.2:3b": "Llama 3.2 3B",
    "phi3.5:3.8b": "Phi-3.5 3.8B", "deepseek-r1:7b": "DeepSeek-R1 7B",
    "qwen2.5-coder:7b": "Qwen 2.5 Coder 7B", "qwen2.5:7b": "Qwen 2.5 7B",
    "mistral:7b": "Mistral 7B", "llama3.1:8b": "Llama 3.1 8B",
    "gemma2:9b": "Gemma 2 9B",
}

def load_aggregate():
    path = os.path.join(RAW_DIR, "aggregate_report.json")
    with open(path) as f:
        return json.load(f)

def load_tempsweep():
    path = os.path.join(PROC_DIR, "temperature_sweep.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None

def compute_category_map():
    """Build a map of task_id -> category from the task suite."""
    sys.path.insert(0, BASE)
    from tasks.task_suite import TaskSuite
    ts = TaskSuite()
    mapping = {}
    for task in ts.get_all_tasks():
        mapping[task.id] = task.category.value
    return mapping

cat_map_local = None

def get_cat_map():
    global cat_map_local
    if cat_map_local is None:
        cat_map_local = compute_category_map()
    return cat_map_local

def load_v2():
    """Load v2 31-task capability results if available."""
    path = os.path.join(RAW_DIR, "v2", "aggregate_v2.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None

def build_per_task_map(source, agg):
    """Return {model: {task_id: score}} from v2 (preferred) or legacy aggregate."""
    per_model = {}
    if source:
        for m, r in source["results"].items():
            per_model[m] = {t["task_id"]: t.get("score", 0) for t in r.get("per_task", [])}
        return per_model
    for m in agg["summary_comparison"]:
        report = next((r for r in agg.get("reports", []) if r["model"] == m), None)
        if report:
            per_model[m] = {pt["task_id"]: pt.get("score", 0)
                            for pt in report.get("capability", {}).get("per_task", [])}
    return per_model

def fig_per_category(per_model, sort_key):
    """Heatmap of per-category accuracy for all models (8 categories)."""
    if not per_model:
        print("  (No per-task data)")
        return
    models = sorted(per_model.keys(), key=sort_key, reverse=True)
    categories = list(CATEGORY_MAP.keys())
    cat_labels = [CATEGORY_MAP[c] for c in categories]

    # Build per-category accuracy matrix
    data = np.zeros((len(models), len(categories)))
    for i, model in enumerate(models):
        per_task = per_model[model]
        cat_scores = {c: [] for c in categories}
        for tid, score in per_task.items():
            cat = get_cat_map().get(tid)
            if cat and cat in cat_scores:
                cat_scores[cat].append(score)
        for j, cat in enumerate(categories):
            scores = cat_scores[cat]
            data[i, j] = np.mean(scores) * 100 if scores else 0

    fig, ax = plt.subplots(figsize=(12, 7))
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)

    # Annotate
    for i in range(len(models)):
        for j in range(len(categories)):
            val = data[i, j]
            color = "white" if val < 40 else "black"
            ax.text(j, i, f"{val:.0f}%", ha="center", va="center", fontsize=9, color=color, fontweight="bold")

    model_display = [str(DISPLAY.get(m, m.replace(":7b", "").replace(":3b", ""))) for m in models]
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(cat_labels, fontsize=10, rotation=30, ha="right")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(model_display, fontsize=9)
    ax.set_xlabel("Task Category", fontsize=12)
    ax.set_ylabel("Model", fontsize=12)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Average Score (%)", fontsize=10)

    fig.suptitle("Per-Category Task Accuracy Across All Models (31-Task Suite)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "per_category_accuracy.pdf")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    # Also print the table
    print(f"\n{'='*70}")
    print(f"  PER-CATEGORY ACCURACY BREAKDOWN (31-task suite)")
    print(f"{'='*70}")
    header = f"  {'Model':<22}"
    for cl in cat_labels:
        header += f" {cl:>14}"
    print(header)
    print(f"  {'-'*22} " + " ".join(["-"*14 for _ in cat_labels]))
    for i, model in enumerate(models):
        name = DISPLAY.get(model, model.replace(":7b","").replace(":3b",""))
        line = f"  {name:<22}"
        for j in range(len(categories)):
            line += f" {data[i,j]:>13.1f}%"
        print(line)

def fig_temp_sensitivity(sweep_data):
    """Accuracy vs temperature plot for top models."""
    if not sweep_data:
        print("No temperature sweep data yet.")
        return

    results = sweep_data["results"]
    models = sweep_data["models"]
    temps = sweep_data["temperatures"]

    fig, ax = plt.subplots(figsize=(10, 6))

    markers = {"qwen2.5-coder:7b": "o", "qwen2.5:7b": "s", "mistral:7b": "^"}
    for model_id in models:
        mdata = [(r["temperature"], r["accuracy"] * 100)
                 for r in results if r["model"] == model_id and r.get("accuracy") is not None]
        mdata.sort(key=lambda x: x[0])
        temps_m = [x[0] for x in mdata]
        accs_m = [x[1] for x in mdata]
        color = COLORS.get(model_id, "#333")
        name = DISPLAY.get(model_id, model_id.replace(":7b",""))
        ax.plot(temps_m, accs_m, marker=markers.get(model_id, "o"), color=color,
                linewidth=2.5, markersize=10, label=name)
        # Annotate each point
        for t, a in mdata:
            ax.annotate(f"{a:.0f}%", (t, a), textcoords="offset points",
                       xytext=(0, 12), fontsize=9, ha="center", color=color)

    ax.set_xlabel("Temperature", fontsize=13)
    ax.set_ylabel("Accuracy (%)", fontsize=13)
    ax.set_xticks(temps)
    ax.set_xlim(-0.05, 1.1)
    ax.set_ylim(0, 85)
    ax.legend(fontsize=11, loc="lower left")
    ax.grid(alpha=0.3)
    fig.suptitle("Accuracy vs Temperature for Top-3 Models", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "accuracy_vs_temperature.pdf")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    # Print summary stats
    print(f"\n{'='*70}")
    print(f"  TEMPERATURE SENSITIVITY SUMMARY")
    print(f"{'='*70}")
    for model_id in models:
        mdata = [(r["temperature"], r["accuracy"] * 100)
                 for r in results if r["model"] == model_id and r.get("accuracy") is not None]
        mdata.sort(key=lambda x: x[0])
        accs = [a for _, a in mdata]
        if len(accs) >= 3:
            stdev = statistics.stdev(accs)
            print(f"  {DISPLAY.get(model_id, model_id):<25}: "
                  f"range [{min(accs):.1f}%, {max(accs):.1f}%], "
                  f"stdev={stdev:.2f}, "
                  f"max_drop={max(accs)-min(accs):.1f}pp")

def print_hardest_task(per_model):
    """Find which task categories are hardest across all models."""
    if not per_model:
        print("  (No per-task data)")
        return
    models = list(per_model.keys())
    categories = list(CATEGORY_MAP.keys())
    cat_scores = {c: [] for c in categories}

    for model in models:
        for tid, score in per_model[model].items():
            cat = get_cat_map().get(tid)
            if cat and cat in cat_scores:
                cat_scores[cat].append(score)

    print(f"\n{'='*70}")
    print(f"  HARDEST TASK CATEGORIES (avg across all models)")
    print(f"{'='*70}")
    cat_avgs = []
    for cat in categories:
        scores = cat_scores[cat]
        avg = np.mean(scores) * 100 if scores else 0
        cat_avgs.append((CATEGORY_MAP[cat], avg, len(scores) // len(models)))
    cat_avgs.sort(key=lambda x: x[1])
    for cat, avg, n in cat_avgs:
        print(f"  {cat:<20}: {avg:>5.1f}% avg (n={n} tasks)")

def main():
    print("Running per-category and temperature sensitivity analysis...")
    agg = load_aggregate()
    v2 = load_v2()
    per_model = build_per_task_map(v2, agg)

    # Per-category heatmap (31-task suite preferred)
    if v2:
        sort_key = lambda m: v2["results"][m]["accuracy"]
    else:
        sort_key = lambda m: agg["summary_comparison"][m]["composite_reliability"]
    fig_per_category(per_model, sort_key)
    print_hardest_task(per_model)

    # Temperature sensitivity
    sweep = load_tempsweep()
    if sweep:
        fig_temp_sensitivity(sweep)

    print("\n*** Analysis complete! ***")

if __name__ == "__main__":
    main()
