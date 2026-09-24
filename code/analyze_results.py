#!/usr/bin/env python3
"""Makes the plots and tables. Run after run_experiments, reads data/raw, writes paper/figures."""

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

def fig_accuracy_vs_params(agg, v2=None):
    params_map = {"llama3.2:1b": 1.0, "llama3.2:3b": 3.0, "phi3.5:3.8b": 3.8,
                  "deepseek-r1:7b": 7.0, "qwen2.5-coder:7b": 7.0, "qwen2.5:7b": 7.0,
                  "mistral:7b": 7.0, "llama3.1:8b": 8.0, "gemma2:9b": 9.0}
    fig, ax = plt.subplots(figsize=(10, 6))
    models_data = []
    for model, s in agg["summary_comparison"].items():
        p = params_map.get(model, 0)
        if v2 and model in v2["results"]:
            acc = v2["results"][model]["accuracy"] * 100
        else:
            acc = s["accuracy"] * 100
        models_data.append((p, acc, s["composite_reliability"] * 100, model))
    models_data.sort(key=lambda x: x[0])
    ps = [m[0] for m in models_data]
    accs = [m[1] for m in models_data]
    labels_short = [m[3].replace(":7b", "").replace(":3b", "").replace(":8b", "")
                    .replace(":9b", "").replace(":1b", "").replace(":3.8b", "") for m in models_data]
    colors = [COLORS.get(m[3], "#333") for m in models_data]
    for i in range(len(models_data)):
        ax.scatter(ps[i], accs[i], s=150, color=colors[i], zorder=5, edgecolors="black", linewidth=0.5)
        ax.annotate(labels_short[i], (ps[i], accs[i]), (ps[i] + 0.15, accs[i] + 4),
                    fontsize=8, ha="left")
    z = np.polyfit(ps, accs, 1)
    p = np.poly1d(z)
    x_line = np.linspace(0, 10, 100)
    ax.plot(x_line, p(x_line), "--", color="#888", alpha=0.5, label=f"Trend (r={np.corrcoef(ps, accs)[0,1]:.3f})")
    ax.set_xlabel("Model Size (Billions of Parameters)", fontsize=12)
    ax.set_ylabel("Accuracy on 31 Tasks (%)" if v2 else "Accuracy (%)", fontsize=12)
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    fig.suptitle("Model Size vs Task Accuracy (9 Models, 31-Task Suite)" if v2
                 else "Model Size vs Task Accuracy (9 Models)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "accuracy_vs_params.pdf")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

def gen_latex_table(agg):
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
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Comprehensive reliability evaluation across 9 small language "
                 "models. Scores are percentages.}")
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
    lines.append("\\end{table*}")
    content = "\n".join(lines)
    path = os.path.join(PROC_DIR, "results_table.tex")
    with open(path, "w") as f:
        f.write(content)
    print(f"Saved: {path}")

def print_stats(agg):
    models_sorted = sorted(agg["summary_comparison"].items(),
                           key=lambda x: x[1]["composite_reliability"], reverse=True)
    accs = [s["accuracy"] * 100 for _, s in models_sorted]
    comps = [s["composite_reliability"] * 100 for _, s in models_sorted]
    safes = [s["safety_score"] * 100 for _, s in models_sorted]
    robs = [s["robustness_score"] * 100 for _, s in models_sorted]
    fts = [s["fault_tolerance_score"] * 100 for _, s in models_sorted]
    cons = [s["consistency_score"] * 100 for _, s in models_sorted]
    params = {"llama3.2:1b": 1.0, "llama3.2:3b": 3.0, "phi3.5:3.8b": 3.8,
              "deepseek-r1:7b": 7.0, "qwen2.5-coder:7b": 7.0, "qwen2.5:7b": 7.0,
              "mistral:7b": 7.0, "llama3.1:8b": 8.0, "gemma2:9b": 9.0}
    p_list = [params[m] for m, _ in models_sorted]
    print("\n=== KEY STATISTICS ===")
    print(f"Models: {len(models_sorted)}, best: {models_sorted[0][0]} ({comps[0]:.1f}%), worst: {models_sorted[-1][0]} ({comps[-1]:.1f}%)")
    print(f"Mean acc: {np.mean(accs):.1f}%, mean composite: {np.mean(comps):.1f}%")
    print(f"Mean safety: {np.mean(safes):.1f}%, robustness: {np.mean(robs):.1f}%")

def _wilson_ci(k, n, z=1.96):
    phat = k / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)

def _pearson_p(r, n):
    import scipy.stats as st
    if n <= 2:
        return 1.0
    t = r * math.sqrt((n - 2) / (1 - r * r)) if abs(r) < 1 else float("inf")
    return 2 * (1 - st.t.cdf(abs(t), df=n - 2))

def _cohens_h(p1, p2):
    return 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))

def v2_stats(v2):
    if not v2:
        return
    params_map = {"llama3.2:1b": 1.0, "llama3.2:3b": 3.0, "phi3.5:3.8b": 3.8,
                  "deepseek-r1:7b": 7.0, "qwen2.5-coder:7b": 7.0, "qwen2.5:7b": 7.0,
                  "mistral:7b": 7.0, "llama3.1:8b": 8.0, "gemma2:9b": 9.0}
    n = v2["num_tasks"]
    stats = {"num_tasks": n}
    accs = {}
    for m, r in v2["results"].items():
        k = round(r["accuracy"] * n)
        lo, hi = _wilson_ci(k, n)
        accs[m] = {"accuracy": r["accuracy"], "accuracy_pct": r["accuracy"] * 100,
                   "success_rate": r["success_rate"], "avg_duration_s": r["avg_duration_s"],
                   "wilson_ci": [round(lo * 100, 1), round(hi * 100, 1)]}
    stats["models"] = accs
    vals = [a["accuracy_pct"] for a in accs.values()]
    stats["mean_accuracy"] = round(float(np.mean(vals)), 1)
    stats["min_accuracy"] = min(vals)
    stats["max_accuracy"] = max(vals)
    ps = [params_map[m] for m in accs]
    vs = [accs[m]["accuracy_pct"] for m in accs]
    r_acc = float(np.corrcoef(ps, vs)[0, 1])
    stats["params_vs_acc_r"] = round(r_acc, 3)
    with open(os.path.join(PROC_DIR, "v2_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved v2 stats, mean acc {stats['mean_accuracy']}%")

def load_v2_capability():
    path = os.path.join(RAW_DIR, "v2", "aggregate_v2.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

def load_temperature_sweep():
    path = os.path.join(PROC_DIR, "temperature_sweep.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

def fig_temperature(ts):
    if not ts:
        print("  (No temperature sweep data)")
        return
    models = ts["models"]
    temps = [0.0] + ts["temperatures"]
    results = ts["results"]
    acc_by_model = {m: {0.0: None} for m in models}
    v2 = load_v2_capability()
    if v2:
        for m in models:
            if m in v2["results"]:
                acc_by_model[m][0.0] = v2["results"][m].get("accuracy", 0) * 100
    for r in results:
        m = r["model"]
        if m in acc_by_model:
            acc_by_model[m][r["temperature"]] = r.get("accuracy", 0) * 100
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["#27AE60", "#2ECC71", "#2980B9"]
    for i, m in enumerate(models):
        xs, ys = [], []
        for t in temps:
            if acc_by_model[m].get(t) is not None:
                xs.append(t)
                ys.append(acc_by_model[m][t])
        ax.plot(xs, ys, "o-", linewidth=2, markersize=7, color=colors[i],
                label=m.replace(":7b", "").replace(":3b", "").replace(":8b", ""))
    ax.set_xlabel("Temperature", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    fig.suptitle("Accuracy vs Temperature (31-Task Suite)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "temperature_sensitivity.pdf")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

def fig_cost_reliability(v2):
    if not v2:
        return
    vram_map = {"llama3.2:1b": 1.0, "llama3.2:3b": 2.5, "phi3.5:3.8b": 2.8,
                "deepseek-r1:7b": 4.5, "qwen2.5-coder:7b": 4.5, "qwen2.5:7b": 4.5,
                "mistral:7b": 4.5, "llama3.1:8b": 5.5, "gemma2:9b": 5.5}
    fig, ax = plt.subplots(figsize=(10, 6))
    for model, r in v2["results"].items():
        acc = r.get("accuracy", 0) * 100
        vram = vram_map.get(model, 3.0)
        color = COLORS.get(model, "#333")
        label = model.replace(":7b", "").replace(":3b", "").replace(":8b", "")
        ax.scatter(vram, acc, s=180, color=color, zorder=5, edgecolors="black", linewidth=0.5)
        ax.annotate(label, (vram, acc), (vram + 0.1, acc + 3), fontsize=8)
    ax.set_xlabel("VRAM Usage (GB, Q4_K_M)", fontsize=12)
    ax.set_ylabel("Accuracy on 31 Tasks (%)", fontsize=12)
    ax.grid(alpha=0.3)
    fig.suptitle("Cost (VRAM) vs Reliability (31-Task Accuracy)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "cost_reliability.pdf")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

def gen_v2_latex_table(v2):
    if not v2:
        return
    DISPLAY = {
        "llama3.2:1b": "Llama 3.2 1B", "llama3.2:3b": "Llama 3.2 3B",
        "phi3.5:3.8b": "Phi-3.5 3.8B", "deepseek-r1:7b": "DeepSeek-R1 7B",
        "qwen2.5-coder:7b": "Qwen 2.5 Coder 7B", "qwen2.5:7b": "Qwen 2.5 7B",
        "mistral:7b": "Mistral 7B", "llama3.1:8b": "Llama 3.1 8B",
        "gemma2:9b": "Gemma 2 9B",
    }
    models_sorted = sorted(v2["results"].items(), key=lambda x: x[1].get("accuracy", 0), reverse=True)
    lines = ["\\begin{table}[t]", "\\centering",
             "\\caption{Capability accuracy on the 31-task suite.}",
             "\\label{tab:capability_v2}", "\\footnotesize",
             "\\begin{tabular}{lcccc}", "\\toprule",
             "\\textbf{Model} & \\textbf{Accuracy} & \\textbf{Success} & \\textbf{Err.} & \\textbf{Avg Time} \\\\",
             "\\midrule"]
    for model, r in models_sorted:
        display = DISPLAY.get(model, model)
        lines.append(f"        {display} & {r['accuracy']*100:.1f}\\% & {r['success_rate']*100:.1f}\\% & {r['error_rate']*100:.1f}\\% & {r['avg_duration_s']:.1f}s \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    with open(os.path.join(PROC_DIR, "results_table_v2.tex"), "w") as f:
        f.write("\n".join(lines))

def gen_appendix_detailed(v2):
    if not v2:
        return
    SHORT = {"llama3.2:1b": "L3.2-1B", "llama3.2:3b": "L3.2-3B", "phi3.5:3.8b": "Phi-3.5",
             "deepseek-r1:7b": "DS-R1-7B", "qwen2.5-coder:7b": "Qw-Coder", "qwen2.5:7b": "Qwen-7B",
             "mistral:7b": "Mistral", "llama3.1:8b": "L3.1-8B", "gemma2:9b": "Gemma-9B"}
    models_sorted = sorted(v2["results"].items(), key=lambda x: x[1].get("accuracy", 0), reverse=True)
    first = v2["results"][models_sorted[0][0]]
    task_ids = [t["task_id"] for t in first.get("per_task", [])]
    lines = ["\\begin{table*}[h]", "\\centering",
             "\\caption{Per-task scores (31-task suite, t=0).}",
             "\\label{tab:detailed_results}", "\\scriptsize",
             "\\setlength{\\tabcolsep}{3.5pt}"]
    cols = "l" + "c" * len(models_sorted)
    lines.append(f"\\begin{{tabular}}{{{cols}}}")
    lines.append("\\toprule")
    lines.append("\\textbf{Task} & " + " & ".join([f"\\textbf{{{SHORT.get(m, m)}}}" for m, _ in models_sorted]) + " \\\\")
    lines.append("\\midrule")
    for tid in task_ids:
        row = []
        for model, r in models_sorted:
            per = {t["task_id"]: t for t in r.get("per_task", [])}
            t = per.get(tid)
            row.append(f"{t['score']:.2f}" if t else "--")
        lines.append(f"{tid} & " + " & ".join(row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    with open(os.path.join(PROC_DIR, "appendix_detailed.tex"), "w") as f:
        f.write("\n".join(lines))

def main():
    print("Generating visualizations...")
    agg = load_aggregate()
    v2 = load_v2_capability()
    ts = load_temperature_sweep()
    fig_radar(agg)
    fig_bars(agg)
    fig_composite(agg)
    fig_accuracy_vs_params(agg, v2)
    gen_latex_table(agg)
    print_stats(agg)
    fig_temperature(ts)
    fig_cost_reliability(v2)
    gen_v2_latex_table(v2)
    gen_appendix_detailed(v2)
    v2_stats(v2)
    print("\n*** Done ***")

if __name__ == "__main__":
    main()
