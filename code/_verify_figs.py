#!/usr/bin/env python3
"""Verify figure inputs match the paper tables."""
import json
import numpy as np

agg = json.load(open('data/raw/aggregate_report.json'))
v2 = json.load(open('data/raw/v2/aggregate_v2.json'))

models = ['qwen2.5-coder:7b', 'qwen2.5:7b', 'gemma2:9b', 'phi3.5:3.8b', 'mistral:7b',
          'llama3.2:3b', 'llama3.2:1b', 'llama3.1:8b', 'deepseek-r1:7b']

print('=== Composite (should match results_table.tex) ===')
for m in models:
    print(f'  {m}: {agg["summary_comparison"][m]["composite_reliability"]*100:.1f}%')

print('=== 31-task acc (should match results_table_v2.tex) ===')
for m in models:
    print(f'  {m}: {v2["results"][m]["accuracy"]*100:.1f}%')

print('=== Correlations ===')
params_map = {"llama3.2:1b": 1.0, "llama3.2:3b": 3.0, "phi3.5:3.8b": 3.8,
              "deepseek-r1:7b": 7.0, "qwen2.5-coder:7b": 7.0, "qwen2.5:7b": 7.0,
              "mistral:7b": 7.0, "llama3.1:8b": 8.0, "gemma2:9b": 9.0}
ps = [params_map[m] for m in models]
comps = [agg['summary_comparison'][m]['composite_reliability']*100 for m in models]
accs = [v2['results'][m]['accuracy']*100 for m in models]
print(f'  params vs composite: r={np.corrcoef(ps, comps)[0,1]:.3f} (paper: -0.179)')
print(f'  params vs acc: r={np.corrcoef(ps, accs)[0,1]:.3f} (paper: 0.289)')
print(f'  acc vs composite: r={np.corrcoef(accs, comps)[0,1]:.3f} (paper: 0.435)')

# Temperature check
ts = json.load(open('data/processed/temperature_sweep.json'))
print('=== Temperature (should match 05-results) ===')
for r in ts['results']:
    print(f'  {r["model"]} t={r["temperature"]}: acc={r["accuracy"]*100:.1f}%')

# Perturbation / fault matrix for the 8 models with per_task
import os
print('=== Perturbation types per model (count of tasks) ===')
for f in sorted(os.listdir('data/raw')):
    if f.startswith('report_') and f.endswith('.json'):
        rep = json.load(open(os.path.join('data/raw', f)))
        pt = rep['robustness'].get('per_task')
        ft = rep['fault_tolerance'].get('per_task')
        print(f'  {rep["model"]}: rob_tasks={len(pt) if pt else "n/a"}, ft_tasks={len(ft) if ft else "n/a"}')
