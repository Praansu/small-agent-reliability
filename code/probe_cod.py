import json

ORDER = ["qwen2.5-coder:7b", "qwen2.5:7b", "llama3.2:1b", "mistral:7b",
         "llama3.2:3b", "deepseek-r1:7b", "phi3.5:3.8b", "llama3.1:8b", "gemma2:9b"]

v2 = json.load(open(r'C:\Users\ASUS\Desktop\Opencode\small-agent-reliability\data\raw\v2\aggregate_v2.json', encoding='utf-8'))['results']

# Per-model COD and IR correctness (correct order)
for m in ORDER:
    r = v2[m]
    cod = [(pt['task_id'], pt['correctness']) for pt in r['per_task'] if pt['task_id'].startswith('COD')]
    ir = [(pt['task_id'], pt['correctness']) for pt in r['per_task'] if pt['task_id'].startswith('IR')]
    sch = [(pt['task_id'], pt['correctness']) for pt in r['per_task'] if pt['task_id'].startswith('SCH')]
    com = [(pt['task_id'], pt['correctness']) for pt in r['per_task'] if pt['task_id'].startswith('COM')]
    print(f"{m:<20} COD={sum(c for _, c in cod)}/{len(cod)} IR={sum(c for _, c in ir)}/{len(ir)} "
          f"SCH={sum(c for _, c in sch)}/{len(sch)} COM={sum(c for _, c in com)}/{len(com)}")
