#!/usr/bin/env python3
"""
COMPARE: original V2 results vs verification re-run.
Usage: python code/verify_compare.py
Reads:
  - original: data/raw/v2/aggregate_v2.json (or per-model capability_*.json)
  - verify:   data/raw/verify/capability_*.json
Prints per-model accuracy diff + per-task discrepancies. Writes compare_report.json.
"""
import json, os, sys

BASE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(BASE, "..", "data", "raw")
V2 = os.path.join(RAW, "v2")
VER = os.path.join(RAW, "verify")

def load_original(model_id):
    safe = model_id.replace(":", "_")
    # try aggregate first
    agg = os.path.join(V2, "aggregate_v2.json")
    if os.path.exists(agg):
        try:
            d = json.load(open(agg, encoding="utf-8"))
            r = d.get("results", {}).get(model_id)
            if r and r.get("per_task"):
                return {pt["task_id"]: pt["correctness"] for pt in r["per_task"]}
        except Exception:
            pass
    cap = os.path.join(V2, f"capability_{safe}.json")
    if os.path.exists(cap):
        try:
            r = json.load(open(cap, encoding="utf-8"))
            if r.get("per_task"):
                return {pt["task_id"]: pt["correctness"] for pt in r["per_task"]}
        except Exception:
            pass
    return None

def load_verify(model_id):
    safe = model_id.replace(":", "_")
    p = os.path.join(VER, f"capability_{safe}.json")
    if not os.path.exists(p):
        return None, None
    r = json.load(open(p, encoding="utf-8"))
    if r.get("error"):
        return r, None
    return r, {pt["task_id"]: pt["correctness"] for pt in r.get("per_task", [])}

MODELS = [
    "qwen2.5-coder:7b", "qwen2.5:7b", "llama3.2:1b", "mistral:7b",
    "llama3.2:3b", "deepseek-r1:7b", "phi3.5:3.8b", "llama3.1:8b", "gemma2:9b",
]

report = {"compared_at": None, "models": {}}
print(f"{'Model':<20} {'Orig%':>6s} {'Ver%':>6s} {'Diffs':>5s}  Task-level discrepancies")
print("-" * 80)
for m in MODELS:
    orig = load_original(m)
    vr, vmap = load_verify(m)
    if orig is None:
        print(f"{m:<20} ORIGINAL DATA NOT FOUND")
        continue
    if vmap is None:
        print(f"{m:<20} VERIFY NOT DONE YET (error={vr.get('error','?') if vr else '?'})")
        continue
    o_acc = sum(orig.values()) / len(orig) * 100
    v_acc = sum(vmap.values()) / len(vmap) * 100
    diffs = []
    for tid in sorted(set(orig) | set(vmap)):
        o, v = orig.get(tid), vmap.get(tid)
        if o != v:
            diffs.append(f"{tid}:{'T' if o else 'F'}->{'T' if v else 'F'}")
    status = "OK" if not diffs else "DIFF"
    print(f"{m:<20} {o_acc:6.1f} {v_acc:6.1f} {len(diffs):5d}  {status} {', '.join(diffs[:8])}")
    report["models"][m] = {
        "orig_acc": o_acc, "verify_acc": v_acc,
        "num_diffs": len(diffs), "diffs": diffs,
    }

with open(os.path.join(VER, "compare_report.json"), "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)
print("\ncompare_report.json written to data/raw/verify/")
