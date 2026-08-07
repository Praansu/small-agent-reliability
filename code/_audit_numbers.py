import json
MODELS = ['qwen2.5-coder:7b','qwen2.5:7b','gemma2:9b','phi3.5:3.8b','mistral:7b','llama3.2:3b','llama3.2:1b','llama3.1:8b','deepseek-r1:7b']
v2 = json.load(open(r'data/raw/v2/aggregate_v2.json', encoding='utf-8'))

print('=== CLAIM: "run success 58.1% (vs 90-100% for other models)" ===')
srs = [(m, v2['results'][m]['success_rate']*100) for m in MODELS]
for m, s in srs: print('  {:20s} {:6.1f}%'.format(m, s))
others = [s for m, s in srs if m != 'deepseek-r1:7b']
print('  others range: {:.1f}-{:.1f}% -> claim "90-100%" is', 'OK' if min(others) >= 90 else 'WRONG (min={:.1f})'.format(min(others)))

print()
print('=== CLAIM: Qwen 2.5 7B 100% on DM vs Coder 66.7% ===')
cats = {}
for m in MODELS:
    per = v2['results'][m]['per_task']
    d = {}
    for t in per:
        tid = t['task_id']
        c = tid.split('-')[0]
        d.setdefault(c, [0, 0])
        d[c][1] += 1
        if t['correctness']: d[c][0] += 1
    cats[m] = {c: round(100*ok/n, 1) for c, (ok, n) in d.items()}
for m in ['qwen2.5-coder:7b', 'qwen2.5:7b']:
    print('  {:20s} DM={}  all={}'.format(m, cats[m].get('DM'), cats[m]))

print()
print('=== mean 31-task acc claim 47.0% ===')
accs = [v2['results'][m]['accuracy']*100 for m in MODELS]
print('  mean = {:.1f}%'.format(sum(accs)/len(accs)))

print()
print('=== "2.6-9.9x slower" check ===')
dur = {m: v2['results'][m]['avg_duration_s'] for m in MODELS}
r1 = dur['deepseek-r1:7b']
others_d = {m: d for m, d in dur.items() if m != 'deepseek-r1:7b'}
print('  DS-R1={:.1f}s, min peer={:.1f}s -> {:.1f}x; max peer={:.1f}s -> {:.1f}x'.format(
    r1, min(others_d.values()), r1/min(others_d.values()), max(others_d.values()), r1/max(others_d.values())))
