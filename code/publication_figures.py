#!/usr/bin/env python3
"""Publication-grade figures for the small-agent reliability paper.

Generates consistent, high-resolution figures from the raw experiment data:
  1. reliability_dims.pdf        - grouped bars of the 4 reliability dims per model
  2. composite_vs_params.pdf     - composite reliability vs parameter count (neg. corr.)
  3. capability_vs_reliability.pdf - 31-task accuracy vs composite (the r=0.435 disconnect)
  4. perturbation_heatmap.pdf    - model x perturbation type degradation heatmap
  5. fault_heatmap.pdf           - model x fault type recovery heatmap
  6. safety_breakdown.pdf        - per-category safety pass rates + refusal vs scope
  7. accuracy_vs_params.pdf      - restyled: 31-task accuracy vs params (existing finding)
  8. temperature_sensitivity.pdf - restyled: accuracy vs temperature (existing finding)
  9. cost_reliability.pdf        - restyled: accuracy vs VRAM (existing finding)
 10. per_category_accuracy.pdf   - restyled: model x category heatmap (existing finding)

All figures share a consistent publication style: no chartjunk, muted palette,
consistent fonts, sized for a two-column paper (single-col ~3.25in, full-width ~6.7in).
"""

import json, os, sys, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

def _rankdata_avg(a):
    """Average ranks for ties (scipy.stats.rankdata(method='average') equivalent,
    pure numpy). Required: naive argsort ranks mishandle ties (e.g. four tied
    7B-parameter models), which shifts Spearman values."""
    a = np.asarray(a, dtype=float)
    sorter = np.argsort(a, kind="mergesort")
    a_sorted = a[sorter]
    inv = np.empty_like(sorter, dtype=int)
    inv[sorter] = np.arange(a.size)
    obs = np.r_[True, a_sorted[1:] != a_sorted[:-1]]     # group starts
    dense = np.cumsum(obs)[inv]                          # group id per element
    bounds = np.nonzero(obs)[0]
    ends = np.r_[bounds[1:], a.size]
    avg_rank = (bounds + ends + 1) / 2.0                 # per-group average rank
    return avg_rank[dense - 1]

def _spearman(x, y):
    """Spearman rank correlation with average-rank tie handling (matches
    scipy.stats.spearmanr and the paper's reported values)."""
    rx = _rankdata_avg(x)
    ry = _rankdata_avg(y)
    mx, my = rx.mean(), ry.mean()
    num = ((rx - mx) * (ry - my)).sum()
    den = np.sqrt(((rx - mx) ** 2).sum() * ((ry - my) ** 2).sum())
    return num / den if den > 0 else 0.0

BASE = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE, "..", "data", "raw")
PROC_DIR = os.path.join(BASE, "..", "data", "processed")
FIG_DIR = os.path.join(BASE, "..", "paper", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# --------------------------------------------------------------------------
# Shared style
# --------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.4,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.prop_cycle": plt.cycler(color=["#4C72B0", "#DD8452", "#55A868", "#C44E52",
                                          "#8172B3", "#937860", "#DA8BC3", "#8C8C8C",
                                          "#CCB974", "#64B5CD"]),
})

MODELS_ORDER = ["qwen2.5-coder:7b", "qwen2.5:7b", "gemma2:9b", "phi3.5:3.8b",
                "mistral:7b", "llama3.2:3b", "llama3.2:1b", "llama3.1:8b",
                "deepseek-r1:7b"]

DISPLAY = {
    "llama3.2:1b": "Llama 3.2 1B", "llama3.2:3b": "Llama 3.2 3B",
    "phi3.5:3.8b": "Phi-3.5 3.8B", "deepseek-r1:7b": "DeepSeek-R1 7B",
    "qwen2.5-coder:7b": "Qwen 2.5 Coder 7B", "qwen2.5:7b": "Qwen 2.5 7B",
    "mistral:7b": "Mistral 7B", "llama3.1:8b": "Llama 3.1 8B",
    "gemma2:9b": "Gemma 2 9B",
}
SHORT = {k: v.replace(" 7B", "").replace(" 1B", "").replace(" 3B", "")
               .replace(" 8B", "").replace(" 9B", "").replace(" 3.8B", "")
         for k, v in DISPLAY.items()}

PARAMS = {"llama3.2:1b": 1.0, "llama3.2:3b": 3.0, "phi3.5:3.8b": 3.8,
          "deepseek-r1:7b": 7.0, "qwen2.5-coder:7b": 7.0, "qwen2.5:7b": 7.0,
          "mistral:7b": 7.0, "llama3.1:8b": 8.0, "gemma2:9b": 9.0}
VRAM = {"llama3.2:1b": 1.0, "llama3.2:3b": 2.5, "phi3.5:3.8b": 2.8,
        "deepseek-r1:7b": 4.5, "qwen2.5-coder:7b": 4.5, "qwen2.5:7b": 4.5,
        "mistral:7b": 4.5, "llama3.1:8b": 5.5, "gemma2:9b": 5.5}

DIMS = ["consistency_score", "robustness_score", "fault_tolerance_score", "safety_score"]
DIM_LABELS = ["Consistency", "Robustness", "Fault Tol.", "Safety"]
DIM_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

# --------------------------------------------------------------------------
# Data loaders
# --------------------------------------------------------------------------
def load_aggregate():
    with open(os.path.join(RAW_DIR, "aggregate_report.json")) as f:
        return json.load(f)

def load_v2():
    p = os.path.join(RAW_DIR, "v2", "aggregate_v2.json")
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return None

def load_tempsweep():
    p = os.path.join(PROC_DIR, "temperature_sweep.json")
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return None

def load_reports():
    reports = {}
    for fname in sorted(os.listdir(RAW_DIR)):
        if not fname.startswith("report_") or not fname.endswith(".json"):
            continue
        with open(os.path.join(RAW_DIR, fname)) as f:
            rep = json.load(f)
        reports[rep["model"]] = rep
    return reports


def _save(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"Saved: {path}")


# --------------------------------------------------------------------------
# Fig 1: Reliability dimensions grouped bars (single column)
# --------------------------------------------------------------------------
def fig_reliability_dims(agg):
    models = sorted(agg["summary_comparison"].keys(),
                    key=lambda m: agg["summary_comparison"][m]["composite_reliability"],
                    reverse=True)
    x = np.arange(len(models))
    w = 0.2
    fig, ax = plt.subplots(figsize=(3.45, 2.4))
    for i, dim in enumerate(DIMS):
        vals = [agg["summary_comparison"][m][dim] * 100 for m in models]
        ax.bar(x + (i - 1.5) * w, vals, w, label=DIM_LABELS[i], color=DIM_COLORS[i],
               edgecolor="white", linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[m] for m in models], rotation=35, ha="right", fontsize=6.5)
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 110)
    ax.legend(ncol=2, fontsize=6.5, loc="upper right", frameon=False)
    ax.grid(axis="y")
    _save(fig, "reliability_dims.pdf")


# --------------------------------------------------------------------------
# Fig 2: Composite reliability vs params (single column)
# --------------------------------------------------------------------------
def fig_composite_vs_params(agg):
    comps = {m: agg["summary_comparison"][m]["composite_reliability"] * 100 for m in MODELS_ORDER}
    ps = [PARAMS[m] for m in MODELS_ORDER]
    vs = [comps[m] for m in MODELS_ORDER]
    fig, ax = plt.subplots(figsize=(3.45, 2.4))
    for m in MODELS_ORDER:
        ax.scatter(PARAMS[m], comps[m], s=32, zorder=5, edgecolors="black",
                   linewidth=0.4, color="#4C72B0")
        ax.annotate(SHORT[m], (PARAMS[m], comps[m]),
                    (PARAMS[m] + 0.12, comps[m] + 3.2), fontsize=6, ha="left")
    z = np.polyfit(ps, vs, 1)
    xl = np.linspace(0, 10, 100)
    ax.plot(xl, np.poly1d(z)(xl), "--", color="#888", linewidth=0.9,
            label=f"r = {np.corrcoef(ps, vs)[0,1]:.2f}, rho = {_spearman(ps, vs):.2f}")
    ax.set_xlabel("Parameters (billions)")
    ax.set_ylabel("Composite reliability (%)")
    ax.set_xlim(0.4, 10)
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=6.5)
    ax.grid(alpha=0.3)
    _save(fig, "composite_vs_params.pdf")


# --------------------------------------------------------------------------
# Fig 3: Capability vs reliability scatter (single column)
# --------------------------------------------------------------------------
def fig_capability_vs_reliability(agg, v2):
    comps = {m: agg["summary_comparison"][m]["composite_reliability"] * 100 for m in MODELS_ORDER}
    accs = {m: v2["results"][m]["accuracy"] * 100 for m in MODELS_ORDER}
    fig, ax = plt.subplots(figsize=(3.45, 2.4))
    for m in MODELS_ORDER:
        ax.scatter(accs[m], comps[m], s=32, zorder=5, edgecolors="black",
                   linewidth=0.4, color="#55A868")
        ax.annotate(SHORT[m], (accs[m], comps[m]),
                    (accs[m] + 1.5, comps[m] + 2.5), fontsize=6, ha="left")
    ax.axvline(50, color="#999", linestyle=":", linewidth=0.7)
    ax.axhline(50, color="#999", linestyle=":", linewidth=0.7)
    ps = [accs[m] for m in MODELS_ORDER]
    vs = [comps[m] for m in MODELS_ORDER]
    ax.text(3, 97, f"r = {np.corrcoef(ps, vs)[0,1]:.2f}, rho = {_spearman(ps, vs):.2f}",
            fontsize=7, color="#333")
    ax.set_xlabel("31-task accuracy (%)")
    ax.set_ylabel("Composite reliability (%)")
    ax.set_xlim(15, 80)
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)
    _save(fig, "capability_vs_reliability.pdf")


# --------------------------------------------------------------------------
# Fig 4: Perturbation-type heatmap (full width)
# --------------------------------------------------------------------------
def fig_perturbation_heatmap(reports):
    ptypes = ["paraphrase", "verbose", "concise", "typo", "reordered"]
    plabels = ["Paraphrase", "Verbose", "Concise", "Typo", "Reordered"]
    models = [m for m in MODELS_ORDER if "per_task" in reports.get(m, {}).get("robustness", {})]
    data = np.full((len(models), len(ptypes)), np.nan)
    for i, m in enumerate(models):
        per_task = reports[m]["robustness"].get("per_task", [])
        for pj, pt in enumerate(ptypes):
            ok = [t["perturbation_results"][pt] for t in per_task
                  if pt in t.get("perturbation_results", {})]
            if ok:
                data[i, pj] = 100 * sum(ok) / len(ok)
    fig, ax = plt.subplots(figsize=(6.7, 2.5))
    cmap = plt.cm.RdYlGn
    cmap.set_bad("#dddddd")
    im = ax.imshow(data, cmap=cmap, vmin=0, vmax=100, aspect="auto")
    for i in range(len(models)):
        for j in range(len(ptypes)):
            v = data[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7,
                        color="white" if v < 45 else "black", fontweight="bold")
            else:
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=6, color="#777")
    ax.set_xticks(range(len(ptypes)))
    ax.set_xticklabels(plabels, fontsize=7.5)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([DISPLAY[m] for m in models], fontsize=7)
    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("Success under perturbation (%)", fontsize=7)
    cbar.ax.tick_params(labelsize=6.5)
    _save(fig, "perturbation_heatmap.pdf")


# --------------------------------------------------------------------------
# Fig 5: Fault-type heatmap (full width)
# --------------------------------------------------------------------------
def fig_fault_heatmap(reports):
    ftypes = ["timeout", "rate_limit", "error", "schema_drift"]
    flabels = ["Timeout", "Rate limit", "Error", "Schema drift"]
    models = [m for m in MODELS_ORDER if "per_task" in reports.get(m, {}).get("fault_tolerance", {})]
    data = np.full((len(models), len(ftypes)), np.nan)
    for i, m in enumerate(models):
        per_task = reports[m]["fault_tolerance"].get("per_task", [])
        for fj, ft in enumerate(ftypes):
            ok = [t["fault_results"][ft] for t in per_task
                  if ft in t.get("fault_results", {})]
            if ok:
                data[i, fj] = 100 * sum(ok) / len(ok)
    fig, ax = plt.subplots(figsize=(6.7, 2.5))
    cmap = plt.cm.RdYlGn
    cmap.set_bad("#dddddd")
    im = ax.imshow(data, cmap=cmap, vmin=0, vmax=100, aspect="auto")
    for i in range(len(models)):
        for j in range(len(ftypes)):
            v = data[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7,
                        color="white" if v < 45 else "black", fontweight="bold")
            else:
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=6, color="#777")
    ax.set_xticks(range(len(ftypes)))
    ax.set_xticklabels(flabels, fontsize=7.5)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([DISPLAY[m] for m in models], fontsize=7)
    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("Recovery under fault (%)", fontsize=7)
    cbar.ax.tick_params(labelsize=6.5)
    _save(fig, "fault_heatmap.pdf")


# --------------------------------------------------------------------------
# Fig 6: Safety breakdown (single column)
# --------------------------------------------------------------------------
def fig_safety_breakdown(reports):
    models = sorted(MODELS_ORDER, key=lambda m: reports[m]["safety"]["overall_safety_score"],
                    reverse=True)
    cats = ["harmful_requests", "scope_preservation", "bias_awareness", "confidentiality"]
    clabels = ["Harmful", "Scope", "Bias", "Confid."]
    n = len(models)
    x = np.arange(n)
    fig, ax = plt.subplots(figsize=(3.45, 2.4))
    w = 0.2
    colors = ["#C44E52", "#4C72B0", "#DD8452", "#55A868"]
    for ci, c in enumerate(cats):
        vals = []
        for m in models:
            tests = [t for t in reports[m]["safety"]["per_test"] if t["category"] == c]
            vals.append(100 * sum(t["safe"] for t in tests) / len(tests) if tests else 0)
        ax.bar(x + (ci - 1.5) * w, vals, w, label=clabels[ci], color=colors[ci],
               edgecolor="white", linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[m] for m in models], rotation=35, ha="right", fontsize=6)
    ax.set_ylabel("Pass rate (%)")
    ax.set_ylim(0, 110)
    ax.legend(ncol=4, fontsize=6, loc="upper right", frameon=False)
    ax.grid(axis="y")
    _save(fig, "safety_breakdown.pdf")


# --------------------------------------------------------------------------
# Fig 7: Accuracy vs params (restyled)
# --------------------------------------------------------------------------
def fig_accuracy_vs_params(agg, v2):
    accs = {m: v2["results"][m]["accuracy"] * 100 for m in MODELS_ORDER}
    ps = [PARAMS[m] for m in MODELS_ORDER]
    vs = [accs[m] for m in MODELS_ORDER]
    fig, ax = plt.subplots(figsize=(3.45, 2.4))
    for m in MODELS_ORDER:
        ax.scatter(PARAMS[m], accs[m], s=32, zorder=5, edgecolors="black",
                   linewidth=0.4, color="#DD8452")
        ax.annotate(SHORT[m], (PARAMS[m], accs[m]),
                    (PARAMS[m] + 0.12, accs[m] + 3.2), fontsize=6, ha="left")
    z = np.polyfit(ps, vs, 1)
    xl = np.linspace(0, 10, 100)
    ax.plot(xl, np.poly1d(z)(xl), "--", color="#888", linewidth=0.9,
            label=f"r = {np.corrcoef(ps, vs)[0,1]:.2f}, rho = {_spearman(ps, vs):.2f}")
    ax.set_xlabel("Parameters (billions)")
    ax.set_ylabel("31-task accuracy (%)")
    ax.set_xlim(0.4, 10)
    ax.set_ylim(0, 80)
    ax.legend(frameon=False, fontsize=6.5)
    ax.grid(alpha=0.3)
    _save(fig, "accuracy_vs_params.pdf")


# --------------------------------------------------------------------------
# Fig 8: Temperature sensitivity (restyled)
# --------------------------------------------------------------------------
def fig_temperature(ts, v2):
    temps = ts["temperatures"]
    results = ts["results"]
    fig, ax = plt.subplots(figsize=(3.45, 2.4))
    markers = {"qwen2.5-coder:7b": "o", "qwen2.5:7b": "s", "mistral:7b": "^"}
    colors = {"qwen2.5-coder:7b": "#27AE60", "qwen2.5:7b": "#2980B9", "mistral:7b": "#8E44AD"}
    names = {"qwen2.5-coder:7b": "Qwen 2.5 Coder 7B", "qwen2.5:7b": "Qwen 2.5 7B",
             "mistral:7b": "Mistral 7B"}
    for m in ts["models"]:
        base = v2["results"][m]["accuracy"] * 100
        pts = sorted([(r["temperature"], r["accuracy"] * 100) for r in results
                      if r["model"] == m], key=lambda x: x[0])
        tx = [0.0] + [p[0] for p in pts]
        ty = [base] + [p[1] for p in pts]
        ax.plot(tx, ty, marker=markers[m], color=colors[m], linewidth=1.6,
                markersize=5, label=names[m])
    ax.set_xlabel("Temperature")
    ax.set_ylabel("31-task accuracy (%)")
    ax.set_xticks([0.0] + temps)
    ax.set_xlim(-0.05, 1.15)
    ax.set_ylim(30, 75)
    ax.legend(frameon=False, fontsize=6.5, loc="lower left")
    ax.grid(alpha=0.3)
    _save(fig, "temperature_sensitivity.pdf")


# --------------------------------------------------------------------------
# Fig 9: Cost reliability (restyled)
# --------------------------------------------------------------------------
def fig_cost_reliability(v2):
    accs = {m: v2["results"][m]["accuracy"] * 100 for m in MODELS_ORDER}
    fig, ax = plt.subplots(figsize=(3.45, 2.4))
    for m in MODELS_ORDER:
        ax.scatter(VRAM[m], accs[m], s=32, zorder=5, edgecolors="black",
                   linewidth=0.4, color="#8172B3")
        ax.annotate(SHORT[m], (VRAM[m], accs[m]),
                    (VRAM[m] + 0.1, accs[m] + 2.5), fontsize=6, ha="left")
    ax.set_xlabel("VRAM footprint (GB, 4-bit)")
    ax.set_ylabel("31-task accuracy (%)")
    ax.set_xlim(0.5, 6.2)
    ax.set_ylim(0, 80)
    ax.grid(alpha=0.3)
    _save(fig, "cost_reliability.pdf")


# --------------------------------------------------------------------------
# Fig 10: Per-category accuracy (restyled heatmap, full width)
# --------------------------------------------------------------------------
def fig_per_category(v2):
    cat_order = ["information_retrieval", "scheduling", "data_analysis", "communication",
                 "multi_step_reasoning", "decision_making", "coding", "safety"]
    cat_labels = ["Info Retrieval", "Scheduling", "Data Analysis", "Communication",
                  "Multi-Step", "Decision Making", "Coding", "Safety"]
    # task -> category map from the v2 per-task list (use first model's task ids)
    cat_of = {}
    first = v2["results"][MODELS_ORDER[0]]["per_task"]
    # Build from known task ids (mirror appendix table)
    known = {"IR": "information_retrieval", "SCH": "scheduling", "DA": "data_analysis",
             "COM": "communication", "MSR": "multi_step_reasoning", "DM": "decision_making",
             "COD": "coding", "SAF": "safety"}
    for t in first:
        pref = t["task_id"].split("-")[0]
        cat_of[t["task_id"]] = known.get(pref, "information_retrieval")
    data = np.zeros((len(MODELS_ORDER), len(cat_order)))
    for i, m in enumerate(MODELS_ORDER):
        per = {t["task_id"]: t.get("score", 0) for t in v2["results"][m]["per_task"]}
        for tid, cat in cat_of.items():
            j = cat_order.index(cat)
            data[i, j] = per.get(tid, 0) * 100
    fig, ax = plt.subplots(figsize=(6.7, 2.9))
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)
    for i in range(len(MODELS_ORDER)):
        for j in range(len(cat_order)):
            v = data[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=6.5,
                    color="white" if v < 45 else "black", fontweight="bold")
    ax.set_xticks(range(len(cat_order)))
    ax.set_xticklabels(cat_labels, fontsize=7, rotation=25, ha="right")
    ax.set_yticks(range(len(MODELS_ORDER)))
    ax.set_yticklabels([DISPLAY[m] for m in MODELS_ORDER], fontsize=7)
    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("Accuracy (%)", fontsize=7)
    cbar.ax.tick_params(labelsize=6.5)
    _save(fig, "per_category_accuracy.pdf")


# --------------------------------------------------------------------------
def main():
    agg = load_aggregate()
    v2 = load_v2()
    ts = load_tempsweep()
    reports = load_reports()

    print("Generating publication figures...")
    fig_reliability_dims(agg)
    fig_composite_vs_params(agg)
    fig_capability_vs_reliability(agg, v2)
    fig_perturbation_heatmap(reports)
    fig_fault_heatmap(reports)
    fig_safety_breakdown(reports)
    fig_accuracy_vs_params(agg, v2)
    fig_temperature(ts, v2)
    fig_cost_reliability(v2)
    fig_per_category(v2)
    print("All figures generated.")


if __name__ == "__main__":
    main()
