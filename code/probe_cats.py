import json
from collections import defaultdict
v2 = json.load(open(r'C:\Users\ASUS\Desktop\Opencode\small-agent-reliability\data\raw\v2\aggregate_v2.json', encoding='utf-8'))['results']

def cat(tid):
    return tid.split('-')[0]

cats = defaultdict(lambda: defaultdict(list))
for m, r in v2.items():
    for pt in r['per_task']:
        cats[cat(pt['task_id'])][m].append(pt['correctness'])

print('category means (original data):')
for c in sorted(cats):
    vals = [sum(x)/len(x)*100 for x in cats[c].values()]
    mean = sum(vals)/len(vals)
    models = [int(v) for v in vals]
    print(f'  {c:<6} mean={mean:.1f}%  per-model={models}')
