"""Check temperature sweep progress."""
import json, os, glob, statistics

files = glob.glob("data/raw/tempsweep_*.json")
results = {}
for f in sorted(files):
    with open(f) as fp:
        d = json.load(fp)
    model = d["model"].replace(":7b","").replace(":3b","")
    t = d["temperature"]
    results.setdefault(model, {})[t] = d["accuracy"] * 100

for model_id in ["qwen2.5-coder:7b", "qwen2.5:7b", "mistral:7b"]:
    safe = model_id.replace(":", "_")
    path = f"data/raw/report_{safe}.json"
    if os.path.exists(path):
        with open(path) as fp:
            r = json.load(fp)
        model = model_id.replace(":7b","").replace(":3b","")
        results.setdefault(model, {})[0.0] = r["summary"]["accuracy"] * 100

print(f"{'Model':<20} {'t=0.0':>8} {'t=0.3':>8} {'t=0.7':>8} {'t=1.0':>8}  {'Stdev':>8}")
print("-" * 60)
for model in sorted(results.keys()):
    r = results[model]
    accs = [r.get(t, -1) for t in [0.0, 0.3, 0.7, 1.0]]
    vals = [a for a in accs if a >= 0]
    stdev = statistics.stdev(vals) if len(vals) >= 3 else 0
    parts = [f"{model:<20}"]
    for a in accs:
        parts.append(f" {a:>7.1f}%" if a >= 0 else "     N/A")
    parts.append(f"  {stdev:>7.2f}")
    print("".join(parts))

print(f"\nFiles completed: {len(files)}/9")
print(f"Models with all 4 temps: {sum(1 for v in results.values() if len(v) == 4)}/3")
