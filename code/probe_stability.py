import json, os
p = r'C:\Users\ASUS\Desktop\Opencode\small-agent-reliability\data\raw\verify\stability.json'
if not os.path.exists(p):
    print('NOT STARTED YET')
else:
    s = json.load(open(p, encoding='utf-8'))
    print('done pairs:', len(s['results']), '/ 20')
    for k, v in s['results'].items():
        o = [x['correct'] for x in v['outcomes']]
        marks = '/'.join('P' if c else 'F' for c in o)
        print(f"{v['model']:<20} {v['task']:<6} {marks}")
