#!/usr/bin/env python3
"""
Analysis and visualization script for agent reliability evaluation results.
Generates figures and tables for the research paper.
"""

import json
import os
import sys
from typing import Any, Dict, List
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "paper", "figures")


def load_results() -> Dict[str, Any]:
    """Load aggregate results from the experiment."""
    path = os.path.join(DATA_DIR, "aggregate_report.json")
    if not os.path.exists(path):
        print(f"No results found at {path}")
        print("Run experiments first: python run_experiments.py")
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


def create_reliability_radar_chart(results: Dict[str, Any]):
    """Create a radar chart comparing models across reliability dimensions."""
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    dimensions = ["Accuracy", "Consistency", "Robustness", "Fault\nTolerance", "Safety"]
    num_dims = len(dimensions)
    angles = np.linspace(0, 2 * np.pi, num_dims, endpoint=False).tolist()
    angles += angles[:1]

    colors = sns.color_palette("muted", len(results["reports"]))

    for idx, report in enumerate(results["reports"]):
        s = report["summary"]
        values = [
            s["accuracy"],
            s["consistency_score"],
            s["robustness_score"],
            s["fault_tolerance_score"],
            s["safety_score"],
        ]
        values += values[:1]
        model_label = report["model"].replace(":", " ")
        ax.plot(angles, values, "o-", linewidth=2, label=model_label, color=colors[idx])
        ax.fill(angles, values, alpha=0.1, color=colors[idx])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dimensions, fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["20%", "40%", "60%", "80%", "100%"], fontsize=10)
    ax.set_title("Reliability Profile by Model", fontsize=16, fontweight="bold", pad=30)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=11)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "reliability_radar.pdf")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()


def create_reliability_bar_chart(results: Dict[str, Any]):
    """Create a grouped bar chart comparing models across dimensions."""
    fig, ax = plt.subplots(figsize=(12, 6))

    reports = results["reports"]
    models = [r["model"].replace(":", " ") for r in reports]
    dimensions = ["Accuracy", "Consistency", "Robustness", "Fault Tol.", "Safety"]
    
    x = np.arange(len(models))
    width = 0.15
    colors = sns.color_palette("husl", len(dimensions))

    for i, dim in enumerate(dimensions):
        keys = ["accuracy", "consistency_score", "robustness_score", 
                "fault_tolerance_score", "safety_score"]
        values = [r["summary"][keys[i]] * 100 for r in reports]
        bars = ax.bar(x + i * width, values, width, label=dim, color=colors[i])

    ax.set_xlabel("Model", fontsize=14)
    ax.set_ylabel("Score (%)", fontsize=14)
    ax.set_title("Agent Reliability Scores by Model and Dimension", fontsize=15, fontweight="bold")
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(models, fontsize=11)
    ax.legend(fontsize=11, loc="upper right")
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "reliability_bars.pdf")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()


def create_composite_reliability_chart(results: Dict[str, Any]):
    """Create a horizontal bar chart of composite reliability scores."""
    fig, ax = plt.subplots(figsize=(10, 5))

    reports = sorted(results["reports"], key=lambda r: r["summary"]["composite_reliability"])
    models = [r["model"].replace(":", " ").replace("_", " ") for r in reports]
    scores = [r["summary"]["composite_reliability"] * 100 for r in reports]

    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(models)))

    bars = ax.barh(models, scores, color=colors)
    ax.set_xlabel("Composite Reliability Score (%)", fontsize=14)
    ax.set_title("Overall Agent Reliability (Composite Score)", fontsize=15, fontweight="bold")
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_xlim(0, 105)

    for bar, score in zip(bars, scores):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{score:.1f}%", va="center", fontsize=11, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "composite_reliability.pdf")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()


def create_perturbation_heatmap(results: Dict[str, Any]):
    """Create a heatmap showing robustness across perturbation types."""
    reports = results["reports"]
    
    # Extract perturbation data
    pert_types = []
    pert_data = []
    model_labels = []

    for report in reports:
        model = report["model"].replace(":", " ")
        model_labels.append(model)
        rob = report.get("robustness", {})
        tasks = rob.get("per_task", [])
        if tasks:
            pert_results = tasks[0].get("perturbation_results", {})
            if not pert_types:
                pert_types = list(pert_results.keys())
            row = [1.0 if pert_results.get(pt) else 0.0 for pt in pert_types]
            pert_data.append(row)

    if not pert_data:
        print("No perturbation data available")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(pert_data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(pert_types)))
    ax.set_xticklabels(pert_types, fontsize=12, rotation=30, ha="right")
    ax.set_yticks(range(len(model_labels)))
    ax.set_yticklabels(model_labels, fontsize=11)
    ax.set_title("Robustness: Performance Under Perturbation Type", fontsize=14, fontweight="bold")

    for i in range(len(model_labels)):
        for j in range(len(pert_types)):
            val = pert_data[i][j]
            color = "white" if val > 0.5 else "black"
            ax.text(j, i, "✓" if val else "✗", ha="center", va="center",
                    fontsize=14, color=color, fontweight="bold")

    plt.colorbar(im, ax=ax, label="Success", ticks=[0, 1])
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "perturbation_heatmap.pdf")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()


def generate_latex_table(results: Dict[str, Any]) -> str:
    """Generate LaTeX table of results."""
    reports = results["reports"]
    
    latex = r"""
\begin{table}[t]
\centering
\caption{Comprehensive reliability evaluation results across all models. 
         Scores are percentages. Composite reliability is the unweighted 
         average of all five dimensions.}
\label{tab:main_results}
\small
\begin{tabular}{lcccccc}
\toprule
\textbf{Model} & \textbf{Acc.} & \textbf{Cons.} & \textbf{Rob.} & \textbf{F.T.} & \textbf{Saf.} & \textbf{Comp.} \\
\midrule
"""

    for report in reports:
        s = report["summary"]
        model = report["model"].replace(":", " ")
        row = (
            f"{model} & "
            f"{s['accuracy']*100:.1f}\\% & "
            f"{s['consistency_score']*100:.1f}\\% & "
            f"{s['robustness_score']*100:.1f}\\% & "
            f"{s['fault_tolerance_score']*100:.1f}\\% & "
            f"{s['safety_score']*100:.1f}\\% & "
            f"{s['composite_reliability']*100:.1f}\\% \\\\\n"
        )
        latex += f"        {row}"

    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    return latex


def generate_all():
    """Run all analysis and visualization."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results = load_results()

    print("Generating visualizations...")
    create_reliability_radar_chart(results)
    create_reliability_bar_chart(results)
    create_composite_reliability_chart(results)
    create_perturbation_heatmap(results)

    print("\nGenerating LaTeX table...")
    latex_table = generate_latex_table(results)
    table_path = os.path.join(OUTPUT_DIR, "results_table.tex")
    with open(table_path, "w") as f:
        f.write(latex_table)
    print(f"Saved: {table_path}")

    # Save processed data
    summary = {
        "num_models": results.get("num_models", 0),
        "experiment_date": results.get("experiment_date", ""),
        "model_summaries": {
            r["model"]: r["summary"] for r in results["reports"]
        },
    }
    summary_path = os.path.join(OUTPUT_DIR, "analysis_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {summary_path}")

    print("\n*** Analysis complete! All figures and tables generated. ***")


if __name__ == "__main__":
    generate_all()
