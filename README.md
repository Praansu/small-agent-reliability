# Small Models, Big Failures?

**A Comprehensive Reliability Evaluation of Small Language Models as Autonomous Agents**

[![Paper](https://img.shields.io/badge/PDF-Latest-blue)](paper/main.pdf)
[![DOI](https://img.shields.io/badge/arXiv-xxxx.xxxxx-red)](https://arxiv.org/abs/xxxx.xxxxx)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

This repository presents the first comprehensive, multi-dimensional reliability evaluation of open-weight small language models (SLMs) as tool-using autonomous agents. We evaluate **5 models** (3B–9B parameters) across **14 tasks** measuring **4 reliability dimensions**:

| Dimension | What It Measures | How |
|-----------|-----------------|-----|
| **Consistency** | Run-to-run variance | 3 repeated trials per task |
| **Robustness** | Stability under input perturbations | 5 perturbation types |
| **Fault Tolerance** | Recovery from tool failures | 4 failure modes |
| **Safety** | Appropriate refusal behavior | 6 safety tests |

## Key Findings

- **Reliability does NOT scale with model size**: the 3B model outperforms the 9B model on every dimension
- **Qwen 2.5 7B** is the reliability leader (60.0% composite)
- **Gemma 2 9B** is the least reliable (15.0%) despite being the largest
- **Safety is critically weak**: no model exceeds 33.3% (average 20.0%)
- **Robustness** is the primary bottleneck: average 52.0% degradation under perturbation
- **Fault tolerance** shows a binary divide: only 7B models can recover from tool failures

## Models Evaluated

| Model | Params | Quantization | VRAM | Context |
|-------|--------|-------------|------|---------|
| Llama 3.2 3B | 3.0 B | Q4_K_M | 2.5 GB | 8K |
| Phi-3.5-mini | 3.8 B | Q4_K_M | 2.8 GB | 4K |
| Qwen 2.5 7B | 7.0 B | Q4_K_M | 4.5 GB | 32K |
| Mistral 7B | 7.0 B | Q4_K_M | 4.5 GB | 32K |
| Gemma 2 9B | 9.0 B | Q4_K_M | 5.5 GB | 8K |

## Project Structure

```
├── paper/                  # LaTeX source for the paper
│   ├── main.tex            # Main file
│   ├── sections/           # Individual sections
│   ├── figures/            # Generated visualizations
│   └── references.bib      # Bibliography
├── code/                   # Evaluation framework
│   ├── tasks/              # Task definitions (14 tasks)
│   ├── harness/            # Agent harness + tools
│   ├── models/             # Model registry
│   ├── reliability/        # Metric implementations
│   ├── run_experiments.py  # Experiment runner
│   └── analyze_results.py  # Analysis + figures
├── data/                   # Experiment data
│   ├── raw/                # Raw reports (JSON)
│   └── processed/          # Analysis output
├── generate_docx.py        # Word doc generation
└── opencode.jsonc          # OpenCode project config
```

## Reproducibility

All models are publicly available via Ollama:
```bash
ollama pull llama3.2:3b
ollama pull phi3.5:3.8b
ollama pull qwen2.5:7b
ollama pull mistral:7b
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
@article{paudyal2026small,
  title={Small Models, Big Failures? A Comprehensive Reliability Evaluation of Small Language Models as Autonomous Agents},
  author={Paudyal, Praansu},
  journal={arXiv preprint},
  year={2026}
}
```

## License

MIT
