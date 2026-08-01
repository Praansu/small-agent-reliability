#!/usr/bin/env python3
"""Deep analysis of v2 31-task capability results for paper rewrite."""
import json, os, math

BASE = os.path.dirname(os.path.abspath(__file__))
V2_DIR = os.path.join(BASE, "..", "data", "raw", "v2")
with open(os.path.join(V2_DIR, "aggregate_v2.json")) as f:
    agg = json.load(f)

results = agg["results"]
params = {"llama3.2:1b": 1.0, "llama3.2:3b": 3.0, "phi3.5:3.8b": 3.8,
          "deepseek-r1:7b": 7.0, "qwen2.5-coder:7b": 7.0, "qwen2.5:7b": 7.0,
          "mistral:7b": 7.0, "llama3.1:8b": 8.0, "gemma2:9b": 9.0}

# Category map from task ids
cat_of = {}
for tid in ["IR-1","IR-2","IR-3","IR-4","IR-5"]: cat_of[tid]="IR"
for tid in ["SCH-1","SCH-2","SCH-3","SCH-4"]: cat_of[tid]="SCH"
for tid in ["DA-1","DA-2","DA-3","DA-4"]: cat_of[tid]="DA"
for tid in ["COM-1","COM-2","COM-3","COM-4"]: cat_of[tid]="COM"
for tid in ["MSR-1","MSR-2","MSR-3","MSR-4"]: cat_of[tid]="MSR"
for tid in ["DM-1","DM-2","DM-3"]: cat_of[tid]="DM"
for tid in ["COD-1","COD-2","COD-3"]: cat_of[tid]="COD"
for tid in ["SAF-1","SAF-2","SAF-3","SAF-4"]: cat_of[tid]="SAF"

print("=" * 78)
print("  PER-TASK SCORES (1.0 = correct)  -- all 9 models")
print("=" * 78)
models_sorted = sorted(results.items(), key=lambda x: x[1]["accuracy"], reverse=True)
first = results[models_sorted[0][0]]
task_ids = [t["task_id"] for t in first["per_task"]]
hdr = f"  {'Task':<7}" + "".join(f"{m.replace(':7b','').replace(':3b','').replace(':8b','')[:10]:>11}" for m, _ in models_sorted)
print(hdr)
for tid in task_ids:
    row = f"  {tid:<7}"
    for m, r in models_sorted:
        per = {t["task_id"]: t for t in r.get("per_task", [])}
        t = per.get(tid)
        if t is None:
            val = "--"
        elif t["score"] >= 0.99:
            val = "1.0"
        else:
            val = f"{t['score']:.1f}"
        row += f"{val:>11}"
    print(row)

print("\n" + "=" * 78)
print("  PER-CATEGORY ACCURACY (v2)")
print("=" * 78)
cats = ["IR","SCH","DA","COM","MSR","DM","COD","SAF"]
print(f"  {'Model':<20}" + "".join(f"{c:>7}" for c in cats) + f"{'OVERALL':>10}")
for m, r in sorted(results.items(), key=lambda x: x[1]["accuracy"], reverse=True):
    per = {t["task_id"]: t for t in r.get("per_task", [])}
    row = f"  {m:<20}"
    cat_scores = {c: [] for c in cats}
    for tid, t in per.items():
        cat = cat_of.get(tid)
        if cat:
            cat_scores[cat].append(1.0 if t["score"] >= 0.99 else 0.0)
    for c in cats:
        s = cat_scores[c]
        row += f"{sum(s)/len(s)*100 if s else 0:>7.0f}"
    row += f"{r['accuracy']*100:>10.1f}"
    print(row)

print("\n" + "=" * 78)
print("  STATISTICS")
print("=" * 78)
# Pearson r: params vs 31-task accuracy
xs, ys = [], []
for m, r in results.items():
    xs.append(params[m]); ys.append(r["accuracy"] * 100)
n = len(xs)
mx, my = sum(xs)/n, sum(ys)/n
cov = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
vx = sum((x-mx)**2 for x in xs); vy = sum((y-my)**2 for y in ys)
r = cov / math.sqrt(vx * vy)
print(f"  Pearson r (params vs 31-task acc): {r:.3f}   R^2 = {r*r:.3f}")
print(f"  Mean accuracy: {sum(ys)/n:.1f}%   Min: {min(ys):.1f}%   Max: {max(ys):.1f}%")

# Best/worst
print(f"  Best: {max(results.items(), key=lambda x: x[1]['accuracy'])[0]} "
      f"{max(results.items(), key=lambda x: x[1]['accuracy'])[1]['accuracy']*100:.1f}%")
print(f"  Worst: {min(results.items(), key=lambda x: x[1]['accuracy'])[0]} "
      f"{min(results.items(), key=lambda x: x[1]['accuracy'])[1]['accuracy']*100:.1f}%")

# Coder vs qwen per task comparison
print("\n  CODER vs QWEN2.5 (per task, differences)")
c = {t["task_id"]: t for t in results["qwen2.5-coder:7b"]["per_task"]}
q = {t["task_id"]: t for t in results["qwen2.5:7b"]["per_task"]}
diff = []
for tid in task_ids:
    cs = c[tid]["score"]; qs = q[tid]["score"]
    if abs(cs - qs) > 0.01:
        diff.append((tid, cs, qs))
for tid, cs, qs in diff:
    print(f"    {tid:<7} coder={cs:.1f} qwen={qs:.1f} {'CODER+' if cs>qs else 'QWEN+'}")

# DeepSeek passes
print("\n  DEEPSEEK-R1 passing tasks (score>=0.99):")
d = {t["task_id"]: t for t in results["deepseek-r1:7b"]["per_task"]}
for tid in task_ids:
    if d[tid]["score"] >= 0.99:
        print(f"    {tid} ({cat_of.get(tid)})")

# gemma2 passes
print("\n  GEMMA2 passing tasks (score>=0.99):")
g = {t["task_id"]: t for t in results["gemma2:9b"]["per_task"]}
for tid in task_ids:
    if g[tid]["score"] >= 0.99:
        print(f"    {tid} ({cat_of.get(tid)})")
