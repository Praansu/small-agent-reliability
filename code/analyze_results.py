#!/usr/bin/env python3
"""Analyze results and generate publication-quality figures."""

import json, os, math, sys
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

# Colors for models
COLORS = {
    "llama3.2:1b": "#E74C3C", "llama3.2:3b": "#C0392B",
    "phi3.5:3.8b": "#8E44AD", "deepseek-r1:7b": "#2C3E50",
    "qwen2.5-coder:7b": "#27AE60", "qwen2.5:7b": "#2ECC71",
    "mistral:7b": "#2980B9", "llama3.1:8b": "#3498DB",
    "gemma2:9b": "#F39C12",
}
DIMENSIONS = ["accuracy", "consistency_score", "robustness_score",
              "fault_tolerance_score", "safety_score"]

def load_aggregate():
    path = os.path.join(RAW_DIR, "aggregate_report.json")
    with open(path) as f:
        return json.load(f)

def compute_stats(scores):
    n = len(scores)
    mean = np.mean(scores)
    se = np.std(scores, ddof=1) / math.sqrt(n) if n > 1 else 0
    ci = 1.96 * se
    return mean, ci

def fig_radar(agg):
    models = sorted(agg["summary_comparison"].keys(),
                    key=lambda m: agg["summary_comparison"][m]["composite_reliability"], reverse=True)
    n = len(models)
    angles = np.linspace(0, 2 * np.pi, len(DIMENSIONS), endpoint=False).tolist()
    angles += angles[:1]

    fig, axes = plt.subplots(3, 3, figsize=(16, 16), subplot_kw=dict(polar=True))
    axes_flat = axes.flatten()
    for idx, model in enumerate(models):
        ax = axes_flat[idx]
        s = agg["summary_comparison"][model]
        vals = [s[d] * 100 for d in DIMENSIONS]
        vals += vals[:1]
        color = COLORS.get(model, "#333333")
        ax.fill(angles, vals, alpha=0.15, color=color)
        ax.plot(angles, vals, "o-", linewidth=2, color=color, markersize=6)
        ax.set_ylim(0, 100)
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_yticklabels(["20%", "40%", "60%", "80%", "100%"], fontsize=7)
        ax.set_xticks(angles[:-1])
        labels = ["Accuracy", "Consistency", "Robustness", "Fault Tol.", "Safety"]
        ax.set_xticklabels(labels, fontsize=9)
        comp = s["composite_reliability"] * 100
        params = {"llama3.2:1b": 1.0, "llama3.2:3b": 3.0, "phi3.5:3.8b": 3.8,
                  "deepseek-r1:7b": 7.0, "qwen2.5-coder:7b": 7.0, "qwen2.5:7b": 7.0,
                  "mistral:7b": 7.0, "llama3.1:8b": 8.0, "gemma2:9b": 9.0}
        p = params.get(model, 0)
        ax.set_title(f"{model.replace(':', ' ')} ({p}B)\nComposite: {comp:.1f}%",
                     fontsize=11, fontweight="bold", pad=20)

    for j in range(n, 9):
        axes_flat[j].set_visible(False)
    fig.suptitle("Small LM Reliability by Model (9 Models, 5 Dimensions)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "reliability_radar.pdf")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

def fig_bars(agg):
    models = sorted(agg["summary_comparison"].keys(),
                    key=lambda m: agg["summary_comparison"][m]["composite_reliability"], reverse=True)
    x = np.arange(len(models))
    width = 0.15
    fig, ax = plt.subplots(figsize=(16, 7))
    colors_dim = ["#E74C3C", "#3498DB", "#2ECC71", "#F39C12", "#9B59B6"]
    labels = ["Accuracy", "Consistency", "Robustness", "Fault Tol.", "Safety"]
    for i, dim in enumerate(DIMENSIONS):
        vals = [agg["summary_comparison"][m][dim] * 100 for m in models]
        bars = ax.bar(x + i * width, vals, width, label=labels[i],
                      color=colors_dim[i], edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f"{v:.0f}", ha="center", va="bottom", fontsize=6, rotation=90)

    ax.set_xticks(x + width * 2)
    labels_short = [m.replace(":7b", "").replace(":3b", "").replace(":8b", "")
                     .replace(":9b", "").replace(":1b", "").replace(":3.8b", "")
                    for m in models]
    ax.set_xticklabels(labels_short, fontsize=9, rotation=30, ha="right")
    ax.set_ylabel("Score (%)", fontsize=12)
    ax.set_ylim(0, 115)
    ax.legend(fontsize=10, loc="upper right", ncol=5)
    ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Per-Dimension Reliability Scores (9 Models)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "reliability_bars.pdf")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

def fig_composite(agg):
    models_sorted = sorted(agg["summary_comparison"].items(),
                           key=lambda x: x[1]["composite_reliability"], reverse=True)
    names = [m[0].replace(":7b", "").replace(":3b", "").replace(":8b", "")
             .replace(":9b", "").replace(":1b", "").replace(":3.8b", "") for m in models_sorted]
    comps = [m[1]["composite_reliability"] * 100 for m in models_sorted]
    params = {"llama3.2:1b": 1.0, "llama3.2:3b": 3.0, "phi3.5:3.8b": 3.8,
              "deepseek-r1:7b": 7.0, "qwen2.5-coder:7b": 7.0, "qwen2.5:7b": 7.0,
              "mistral:7b": 7.0, "llama3.1:8b": 8.0, "gemma2:9b": 9.0}
    p_sizes = [params.get(m[0], 0) for m in models_sorted]
    colors = [COLORS.get(m[0], "#333") for m in models_sorted]
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(range(len(names)), comps, color=colors, edgecolor="white", height=0.6)
    for i, (bar, comp, param) in enumerate(zip(bars, comps, p_sizes)):
        ax.text(comp + 0.5, i, f"{comp:.1f}%  ({param}B)", va="center", fontsize=10)

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel("Composite Reliability (%)", fontsize=12)
    ax.set_xlim(0, 105)
    ax.grid(axis="x", alpha=0.3)
    fig.suptitle("Composite Reliability Ranking (9 Models)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "composite_reliability.pdf")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

def fig_accuracy_vs_params(agg):
    """Scatter plot: model params vs accuracy."""
    params_map = {"llama3.2:1b": 1.0, "llama3.2:3b": 3.0, "phi3.5:3.8b": 3.8,
                  "deepseek-r1:7b": 7.0, "qwen2.5-coder:7b": 7.0, "qwen2.5:7b": 7.0,
                  "mistral:7b": 7.0, "llama3.1:8b": 8.0, "gemma2:9b": 9.0}
    fig, ax = plt.subplots(figsize=(10, 6))
    models_data = []
    for model, s in agg["summary_comparison"].items():
        p = params_map.get(model, 0)
        models_data.append((p, s["accuracy"] * 100, s["composite_reliability"] * 100, model))
    models_data.sort(key=lambda x: x[0])
    ps = [m[0] for m in models_data]
    accs = [m[1] for m in models_data]
    comps = [m[2] for m in models_data]
    labels_short = [m[3].replace(":7b", "").replace(":3b", "").replace(":8b", "")
                    .replace(":9b", "").replace(":1b", "").replace(":3.8b", "") for m in models_data]
    colors = [COLORS.get(m[3], "#333") for m in models_data]
    for i in range(len(models_data)):
        ax.scatter(ps[i], accs[i], s=150, color=colors[i], zorder=5, edgecolors="black", linewidth=0.5)
        offset = 5 if i % 2 == 0 else -8
        ax.annotate(labels_short[i], (ps[i], accs[i]), (ps[i] + 0.15, accs[i] + offset),
                    fontsize=8, ha="left" if i % 2 == 0 else "left")
    # Trend line
    z = np.polyfit(ps, accs, 1)
    p = np.poly1d(z)
    x_line = np.linspace(0, 10, 100)
    ax.plot(x_line, p(x_line), "--", color="#888", alpha=0.5, label=f"Trend (r={np.corrcoef(ps, accs)[0,1]:.3f})")
    ax.set_xlabel("Model Size (Billions of Parameters)", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_xlim(0, 10.5)
    ax.set_ylim(-5, 85)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    fig.suptitle("Model Size vs Task Accuracy (9 Models)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "accuracy_vs_params.pdf")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

def gen_latex_table(agg):
    """Generate LaTeX results table."""
    models_sorted = sorted(agg["summary_comparison"].items(),
                           key=lambda x: x[1]["composite_reliability"], reverse=True)
    DISPLAY = {
        "llama3.2:1b": "Llama 3.2 1B", "llama3.2:3b": "Llama 3.2 3B",
        "phi3.5:3.8b": "Phi-3.5 3.8B", "deepseek-r1:7b": "DeepSeek-R1 7B",
        "qwen2.5-coder:7b": "Qwen 2.5 Coder 7B", "qwen2.5:7b": "Qwen 2.5 7B",
        "mistral:7b": "Mistral 7B", "llama3.1:8b": "Llama 3.1 8B",
        "gemma2:9b": "Gemma 2 9B",
    }
    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Comprehensive reliability evaluation across 9 small language "
                 "models. Scores are percentages. Composite reliability is the unweighted "
                 "average of all five dimensions.}")
    lines.append("\\label{tab:main_results}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{lcccccc}")
    lines.append("\\toprule")
    lines.append("\\textbf{Model} & \\textbf{Acc.} & \\textbf{Cons.} & \\textbf{Rob.} "
                 "& \\textbf{F.T.} & \\textbf{Saf.} & \\textbf{Comp.} \\\\")
    lines.append("\\midrule")
    for model, s in models_sorted:
        display = DISPLAY.get(model, model)
        lines.append(f"        {display} & {s['accuracy']*100:.1f}\\% & "
                     f"{s['consistency_score']*100:.1f}\\% & "
                     f"{s['robustness_score']*100:.1f}\\% & "
                     f"{s['fault_tolerance_score']*100:.1f}\\% & "
                     f"{s['safety_score']*100:.1f}\\% & "
                     f"{s['composite_reliability']*100:.1f}\\% \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    content = "\n".join(lines)
    path = os.path.join(PROC_DIR, "results_table.tex")
    with open(path, "w") as f:
        f.write(content)
    print(f"Saved: {path}")

def print_stats(agg):
    """Print key statistics for the paper."""
    models_sorted = sorted(agg["summary_comparison"].items(),
                           key=lambda x: x[1]["composite_reliability"], reverse=True)
    accs = [s["accuracy"] * 100 for _, s in models_sorted]
    comps = [s["composite_reliability"] * 100 for _, s in models_sorted]
    safes = [s["safety_score"] * 100 for _, s in models_sorted]
    params = {"llama3.2:1b": 1.0, "llama3.2:3b": 3.0, "phi3.5:3.8b": 3.8,
              "deepseek-r1:7b": 7.0, "qwen2.5-coder:7b": 7.0, "qwen2.5:7b": 7.0,
              "mistral:7b": 7.0, "llama3.1:8b": 8.0, "gemma2:9b": 9.0}
    p_list = [params[m] for m, _ in models_sorted]
    print("\n=== KEY STATISTICS ===")
    print(f"Models evaluated: {len(models_sorted)}")
    print(f"Best model: {models_sorted[0][0]} ({comps[0]:.1f}%)")
    print(f"Worst model: {models_sorted[-1][0]} ({comps[-1]:.1f}%)")
    print(f"Mean accuracy: {np.mean(accs):.1f}%")
    print(f"Mean composite: {np.mean(comps):.1f}%")
    print(f"Mean safety: {np.mean(safes):.1f}%")
    corr = np.corrcoef(p_list, comps)[0, 1]
    print(f"Params vs composite: r={corr:.3f}")
    best_acc = max(accs)
    print(f"Best accuracy: {best_acc:.1f}%")
    print(f"Safety range: {min(safes):.1f}% - {max(safes):.1f}%")

def main():
    print("Generating visualizations...")
    agg = load_aggregate()
    fig_radar(agg)
    fig_bars(agg)
    fig_composite(agg)
    fig_accuracy_vs_params(agg)
    gen_latex_table(agg)
    print_stats(agg)

    summary = {
        "num_models": agg["num_models"],
        "experiment_date": agg["experiment_date"],
        "model_summaries": agg["summary_comparison"],
    }
    with open(os.path.join(PROC_DIR, "analysis_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved analysis to: {os.path.join(PROC_DIR, 'analysis_summary.json')}")
    print("\n*** Analysis complete! All figures and tables generated. ***")

if __name__ == "__main__":
    main()
