"""Merge all individual model reports into a single aggregate report."""
import json, os, glob

raw_dir = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
reports = []
for f in sorted(glob.glob(os.path.join(raw_dir, "report_*.json"))):
    if "aggregate" in f:
        continue
    with open(f) as fp:
        reports.append(json.load(fp))

aggregate = {
    "experiment_date": reports[-1].get("timestamp", ""),
    "num_models": len(reports),
    "models": [r["model"] for r in reports],
    "reports": reports,
    "summary_comparison": {},
}

for r in reports:
    model = r["model"]
    s = r.get("summary", {})
    if not s or not s.get("composite_reliability"):
        cap = r.get("capability", {})
        if isinstance(cap, dict) and "capability" in cap:
            acc = cap["capability"].get("accuracy", 0)
        else:
            acc = cap.get("accuracy", 0)
        s = {
            "accuracy": acc,
            "consistency_score": r.get("consistency", {}).get("consistency_score", 0),
            "robustness_score": r.get("robustness", {}).get("robustness_score", 0),
            "fault_tolerance_score": r.get("fault_tolerance", {}).get("fault_tolerance_score", 0),
            "safety_score": r.get("safety", {}).get("overall_safety_score", 0),
        }
        s["composite_reliability"] = (
            s["consistency_score"] * 0.25 + s["robustness_score"] * 0.25 +
            s["fault_tolerance_score"] * 0.25 + s["safety_score"] * 0.25
        )
    aggregate["summary_comparison"][model] = s

with open(os.path.join(raw_dir, "aggregate_report.json"), "w") as f:
    json.dump(aggregate, f, indent=2)

print(f"Merged {len(reports)} reports into aggregate_report.json")
