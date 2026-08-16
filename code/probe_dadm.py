import json
from collections import defaultdict

ORDER = ["qwen2.5-coder:7b", "qwen2.5:7b", "llama3.2:1b", "mistral:7b",
         "llama3.2:3b", "deepseek-r1:7b", "phi3.5:3.8b", "llama3.1:8b", "gemma2:9b"]

v2 = json.load(open(r'C:\Users\ASUS\Desktop\Opencode\small-agent-reliability\data\raw\v2\aggregate_v2.json', encoding='utf-8'))['results']

# DA tasks and DM tasks per model
print('Per-model DA tasks (31-task suite):')
for m in ORDER:
    r = v2[m]
    da = [(pt['task_id'], pt['correctness']) for pt in r['per_task'] if pt['task_id'].startswith('DA')]
    dm = [(pt['task_id'], pt['correctness']) for pt in r['per_task'] if pt['task_id'].startswith('DM')]
    print(f"  {m:<20} DA={da} DM={dm}")
