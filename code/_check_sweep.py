import json
import os

CH = r'C:\Users\ASUS\Desktop\Opencode\small-agent-reliability\data\processed\temperature_sweep.json'
d = json.load(open(CH))
print(f"Checkpoint mtime: {os.path.getmtime(CH)}")
print(f"Runs in checkpoint: {len(d['results'])}")
print(f"{'model':<20} {'t':>5} {'acc':>6} {'avg_s':>7} {'tasks':>6} {'slow>6s':>7}")
for r in d['results']:
    pt = r.get('per_task', [])
    slow = sum(1 for t in pt if t.get('duration_ms', 0) > 6000)
    print(f"{r['model']:<20} {r['temperature']:>5.1f} {r.get('accuracy',0)*100:>6.1f} "
          f"{r.get('avg_duration_s',0):>7.1f} {len(pt):>6} {slow:>7}")
