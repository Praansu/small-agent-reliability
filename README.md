# Small Models, Big Failures?

**A Comprehensive Reliability Evaluation of Small Language Models as Autonomous Agents**

[![Paper](https://img.shields.io/badge/PDF-Latest-blue)](paper/main.pdf)
[![DOI](https://img.shields.io/badge/arXiv-xxxx.xxxxx-red)](https://arxiv.org/abs/xxxx.xxxxx)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

This repository presents a comprehensive, multi-dimensional reliability evaluation of open-weight small language models (SLMs) as tool-using autonomous agents. We evaluate **9 models** (1B–9B parameters) across **31 capability tasks** and a **14-task reliability suite** measuring **4 reliability dimensions**, plus a **temperature sensitivity study** (t ∈ {0.3, 0.7, 1.0}) and **cost-reliability analysis**.

| Dimension | What It Measures | How |
|-----------|-----------------|-----|
| **Accuracy** | Task completion rate | 31-task capability suite, 8 categories |
| **Consistency** | Run-to-run variance | 3 repeated trials per task |
| **Robustness** | Stability under input perturbations | 5 perturbation types |
| **Fault Tolerance** | Recovery from tool failures | 4 failure modes |
| **Safety** | Appropriate refusal behavior | 6 safety tests |

## Key Findings

- **Reliability does NOT scale with model size**: the 7B Qwen models dominate, while Gemma 2 9B is the least reliable (15.0%) despite being the largest
- **Qwen 2.5 Coder 7B** is the reliability leader (85.0% composite) — code-specialization beats raw scale
- **Capability leaders**: Qwen 2.5 Coder 7B and Qwen 2.5 7B tie at 67.7% on the 31-task suite; Mistral 7B trails at 45.2%
- **Safety is critically weak**: no model exceeds 50% safety (average 25.9%)
- **Robustness** is the primary bottleneck: average 47.8%
- **Fault tolerance** shows a divide: 100% for the Qwen models vs. near-zero for most others
- **Temperature is a bounded, second-order factor**: accuracy degrades 6.5–9.6 points at high sampling temperatures, but the model ranking is perfectly stable across t
- **Size does not predict reliability**: Pearson r = -0.18 between parameters and composite reliability

## Models Evaluated

| Model | Params | Quantization | VRAM | Context |
|-------|--------|-------------|------|---------|
| Llama 3.2 1B | 1.0 B | Q4_K_M | 1.0 GB | 8K |
| Llama 3.2 3B | 3.0 B | Q4_K_M | 2.5 GB | 8K |
| Phi-3.5-mini | 3.8 B | Q4_K_M | 2.8 GB | 4K |
| DeepSeek-R1 7B | 7.0 B | Q4_K_M | 4.5 GB | 16K |
| Qwen 2.5 Coder 7B | 7.0 B | Q4_K_M | 4.5 GB | 32K |
| Qwen 2.5 7B | 7.0 B | Q4_K_M | 4.5 GB | 32K |
| Mistral 7B | 7.0 B | Q4_K_M | 4.5 GB | 32K |
| Llama 3.1 8B | 8.0 B | Q4_K_M | 5.5 GB | 128K |
| Gemma 2 9B | 9.0 B | Q4_K_M | 5.5 GB | 8K |

## Project Structure

```
├── paper/                  # LaTeX source for the paper
│   ├── main.tex            # Main file
│   ├── sections/           # Individual sections
│   ├── figures/            # Generated visualizations
│   └── references.bib      # Bibliography
├── code/                   # Evaluation framework
│   ├── tasks/              # Task definitions (31 capability + 14 reliability)
│   ├── harness/            # Agent harness + tools
│   ├── models/             # Model registry
│   ├── reliability/        # Metric implementations
│   ├── run_experiments.py  # Experiment runner
│   ├── analyze_results.py  # Analysis + figures
│   └── resume_temp_sweep.py# Temperature sweep orchestrator (checkpointed)
├── data/                   # Experiment data
│   ├── raw/                # Raw reports (JSON)
│   └── processed/          # Analysis output
├── generate_docx.py        # Word doc generation
└── opencode.jsonc          # OpenCode project config
```

## Reproducibility

All models are publicly available via Ollama:
```bash
ollama pull llama3.2:1b
ollama pull llama3.2:3b
ollama pull phi3.5:3.8b
ollama pull mistral:7b
ollama pull qwen2.5:7b
ollama pull qwen2.5-coder:7b
ollama pull llama3.1:8b
ollama pull deepseek-r1:7b
ollama pull gemma2:9b
```

Run the full evaluation:
```bash
python code/run_experiments.py
```

Compile the paper:
```bash
cd paper
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

## Citation

```bibtex
@article{karmacharya2026small,
  title={Small Models, Big Failures? A Comprehensive Reliability Evaluation of Small Language Models as Autonomous Agents},
  author={Karmacharya, Praansu},
  journal={arXiv preprint},
  year={2026}
}
```

## License

MIT
