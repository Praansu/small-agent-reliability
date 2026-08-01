#!/usr/bin/env python3
"""Build aggregate_v2.json from the incremental capability_*.json files (Phase 1 done)."""
import json, glob, os

BASE = os.path.dirname(os.path.abspath(__file__))
V2_DIR = os.path.join(BASE, "..", "data", "raw", "v2")

results = {}
for f in glob.glob(os.path.join(V2_DIR, "capability_*.json")):
    with open(f) as fh:
        r = json.load(fh)
    if "model" in r and "accuracy" in r:
        results[r["model"]] = r

agg = {
    "experiment_date": "2026-07-31T00:00:00",
    "num_models": len(results),
    "num_tasks": next(iter(results.values())).get("num_tasks", 31),
    "results": results,
}
with open(os.path.join(V2_DIR, "aggregate_v2.json"), "w") as fh:
    json.dump(agg, fh, indent=2)

print(f"aggregate_v2.json built with {len(results)} models")
for m, r in sorted(results.items(), key=lambda x: x[1]["accuracy"], reverse=True):
    print(f"  {m:<22} acc={r['accuracy']*100:6.1f}%  succ={r.get('success_rate',0)*100:5.1f}%  time={r.get('avg_duration_s',0):5.1f}s")
