"""Category-level analysis of the 31-task capability suite.

Computes per-category accuracy for all nine models from the raw v2 capability
runs, identifies the hardest / easiest tasks, and emits a LaTeX table for the
paper (data/processed/category_results.tex).

Usage:  python code/category_analysis.py
"""

import json
import os
from collections import OrderedDict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw", "v2")
OUT = os.path.join(ROOT, "data", "processed", "category_results.tex")

MODELS = [
    "qwen2.5-coder:7b",
    "qwen2.5:7b",
    "gemma2:9b",
    "phi3.5:3.8b",
    "mistral:7b",
    "llama3.2:3b",
    "llama3.2:1b",
    "llama3.1:8b",
    "deepseek-r1:7b",
]

DISPLAY = {
    "qwen2.5-coder:7b": "Qwen 2.5 Coder 7B",
    "qwen2.5:7b": "Qwen 2.5 7B",
    "gemma2:9b": "Gemma 2 9B",
    "phi3.5:3.8b": "Phi-3.5 3.8B",
    "mistral:7b": "Mistral 7B",
    "llama3.2:3b": "Llama 3.2 3B",
    "llama3.2:1b": "Llama 3.2 1B",
    "llama3.1:8b": "Llama 3.1 8B",
    "deepseek-r1:7b": "DeepSeek-R1 7B",
}

CATEGORY = OrderedDict([
    ("IR", "Information Retrieval"),
    ("SCH", "Scheduling"),
    ("DA", "Data Analysis"),
    ("COM", "Communication"),
    ("MSR", "Multi-step Reasoning"),
    ("DM", "Decision Making"),
    ("COD", "Coding"),
    ("SAF", "Safety"),
])

# Composite reliability (verified Table 1 values) used to order table rows.
COMPOSITE = {
    "qwen2.5-coder:7b": 85.0,
    "qwen2.5:7b": 60.0,
    "llama3.2:1b": 56.1,
    "mistral:7b": 55.8,
    "llama3.2:3b": 40.0,
    "deepseek-r1:7b": 37.8,
    "phi3.5:3.8b": 30.3,
    "llama3.1:8b": 24.2,
    "gemma2:9b": 15.0,
}


def main():
    per_model = {}
    for m in MODELS:
        path = os.path.join(RAW, f"capability_{m.replace(':', '_')}.json")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        per_model[m] = {p["task_id"]: bool(p["correctness"]) for p in data["per_task"]}

    # --- per-category accuracy per model ------------------------------------
    cat_accuracy = {}
    for m in MODELS:
        cat_accuracy[m] = {}
        for prefix in CATEGORY:
            tasks = [t for t in per_model[m] if t.startswith(prefix)]
            assert tasks, (m, prefix)
            cat_accuracy[m][prefix] = sum(per_model[m][t] for t in tasks) / len(tasks)

    # --- task difficulty across models --------------------------------------
    task_passes = {}
    for m in MODELS:
        for t, ok in per_model[m].items():
            task_passes.setdefault(t, 0)
            task_passes[t] += int(ok)
    hard = sorted([t for t, c in task_passes.items() if c == 0])
    easy = sorted([t for t, c in task_passes.items() if c == len(MODELS)])
    print("Tasks failed by ALL models (%d): %s" % (len(hard), ", ".join(hard)))
    print("Tasks passed by ALL models (%d): %s" % (len(easy), ", ".join(easy)))

    # category difficulty (mean over models)
    cat_difficulty = {
        p: round(100.0 * sum(cat_accuracy[m][p] for m in MODELS) / len(MODELS), 1)
        for p in CATEGORY
    }
    for p, v in sorted(cat_difficulty.items(), key=lambda x: -x[1]):
        print("  %-4s %-24s %6.1f%%" % (p, CATEGORY[p], v))

    # --- emit LaTeX table ----------------------------------------------------
    n_per_cat = {p: len([t for t in per_model[MODELS[0]] if t.startswith(p)]) for p in CATEGORY}
    order = sorted(MODELS, key=lambda m: -COMPOSITE[m])
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Per-category accuracy (\\%) on the 31-task capability suite. Categories are "
        "defined in Table~\\ref{tab:tasks}; per-task counts appear in parentheses. The final "
        "column restates overall 31-task accuracy for reference. Rows are ordered by composite "
        "reliability (Table~\\ref{tab:main_results}).}",
        "\\label{tab:category_results}",
        "\\small",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{l" + "c" * len(CATEGORY) + "c}",
        "\\toprule",
        "\\textbf{Model} & " + " & ".join(
            "%s\\ (%d)" % (CATEGORY[p].replace(" ", "~"), n_per_cat[p])
            for p in CATEGORY
        ) + " & \\textbf{Overall} \\\\",
        "\\midrule",
    ]
    for m in order:
        cells = ["%.1f" % (100.0 * cat_accuracy[m][p]) for p in CATEGORY]
        overall = 100.0 * sum(per_model[m].values()) / len(per_model[m])
        cells.append("%.1f" % overall)
        lines.append("%s & %s \\\\" % (DISPLAY[m], " \\% & ".join(cells) + " \\%"))
    mean_overall = sum(
        100.0 * sum(per_model[m].values()) / len(per_model[m]) for m in MODELS
    ) / len(MODELS)
    mean_cells = ["%.1f" % cat_difficulty[p] for p in CATEGORY] + ["%.1f" % mean_overall]
    lines += [
        "\\midrule",
        "\\textbf{Mean} & " + " \\% & ".join(mean_cells) + " \\% \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table*}",
        "",
    ]
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
