import json

d = json.load(open(r'C:\Users\ASUS\Desktop\Opencode\small-agent-reliability\data\raw\aggregate_report.json'))
s = d['summary_comparison']
print(f"{'model':<22} {'acc':>6} {'cons':>6} {'rob':>6} {'ft':>6} {'saf':>6} {'comp':>6}")
for m, v in s.items():
    ft = v.get('fault_tolerance_score', v.get('fault_tolerance', 0))
    comp = v.get('composite_score', v.get('composite', 0))
    print(f"{m:<22} {v['accuracy']*100:6.1f} {v['consistency_score']*100:6.1f} "
          f"{v['robustness_score']*100:6.1f} {ft*100:6.1f} {v['safety_score']*100:6.1f} {comp*100:6.1f}")
