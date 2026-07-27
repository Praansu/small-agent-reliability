# Small Agent Reliability — Multi-Phase Research Plan

**Vision**: Build the definitive benchmark and understanding of small language model reliability as autonomous agents. Publish findings at top venues, release open-source tools, and drive industry best practices.

---

## Phase 1: Foundation ✅ DONE
- [x] Define 4-dimension reliability framework (consistency, robustness, fault tolerance, safety)
- [x] Build 14-task evaluation suite across 7 categories
- [x] Implement ReAct agent harness with 6 tools
- [x] Evaluate 5 models (Llama 3.2 3B, Phi-3.5 3.8B, Qwen 2.5 7B, Mistral 7B, Gemma 2 9B)
- [x] Generate 4 figures (radar, bars, composite, perturbation heatmap)
- [x] Write 12-page paper with statistical analysis (Wilson CI, Cohen's h, Pearson r)
- [x] Compile PDF + DOCX deliverables
- [x] All 22 citations verified against real arXiv IDs
- [x] GitHub repo: https://github.com/Praansu/small-agent-reliability
- [x] GitHub Actions CI/CD for paper compilation
- [x] Automation script: `python automate.py publish`
- [x] Custom skill: `small-agent-research`

---

## Phase 2: Expansion 🔄 NEXT — 2-4 weeks

### 2A: Models (add 3+ more)
- [ ] **Llama 3.1 8B** — larger Llama, compare vs 3.2 3B
- [ ] **Gemma 3 12B** — latest Google SLM
- [ ] **DeepSeek-R1-Distill-Qwen-7B** — reasoning-distilled
- [ ] **SmolLM2-1.7B** — extreme edge case
- [ ] **Qwen 2.5 Coder 7B** — code-specialized variant

### 2B: Tasks (add 15+ more)
- [ ] Web navigation tasks (click, scroll, form fill)
- [ ] Multi-turn conversation tasks
- [ ] Code generation + execution tasks
- [ ] JSON/structured data extraction
- [ ] Image analysis with vision models

### 2C: Analysis
- [ ] Cross-perturbation analysis: which perturbation types hurt which models most
- [ ] Cost-reliability tradeoff curves (accuracy vs latency vs VRAM)
- [ ] Per-category breakdown (which task categories are hardest)
- [ ] Temperature sensitivity analysis (t=0, 0.3, 0.7, 1.0)

---

## Phase 3: Depth — 4-8 weeks

### 3A: Real-World Fault Injection
- [ ] Real API rate limits and timeouts
- [ ] Network failures and partial responses
- [ ] Malformed tool outputs
- [ ] Adversarial input attacks

### 3B: Multi-Architecture Comparison
- [ ] ReAct (current) vs Reflexion vs Plan-and-Solve
- [ ] Tool-augmented vs pure prompting
- [ ] Single-agent vs multi-agent systems
- [ ] Different tool-calling formats (JSON, XML, function calling)

### 3C: Safety Deep Dive
- [ ] Adversarial jailbreak attempts
- [ ] Multi-step harmful requests
- [ ] Data exfiltration scenarios
- [ ] Comparison with frontier model safety guardrails

---

## Phase 4: Publication — 8-12 weeks

### 4A: Venue Targeting
- [ ] **arXiv preprint** (immediate — can do now)
- [ ] **ICML Workshop** on AI Safety / Reliable ML
- [ ] **NeurIPS Datasets & Benchmarks** — if we build a proper benchmark release
- [ ] **ACL** — if we add multilingual dimension
- [ ] **CCS / S&P** — if we emphasize safety

### 4B: Paper Enhancements
- [ ] Add real failure trajectory examples
- [ ] Inter-rater reliability for task scoring
- [ ] Ablation study: which framework components matter
- [ ] Comparison with frontier models on same benchmark
- [ ] Literature review expansion (50+ citations)

---

## Phase 5: Platform — 12+ weeks

### 5A: Open-Source Benchmark
- [ ] Standalone Python package (`pip install agent-reliability`)
- [ ] CLI tool for running evaluations
- [ ] Leaderboard website (GitHub Pages)
- [ ] Community contribution guide

### 5B: Multilingual Extension
- [ ] Translate tasks to 5+ languages
- [ ] Evaluate cross-lingual reliability
- [ ] Study language-specific failure modes

### 5C: Multimodal Extension
- [ ] Vision-language models (e.g., LLaVA, Qwen-VL)
- [ ] Audio processing tasks
- [ ] Multi-modal tool use

---

## Phase 6: Industry Impact — 16+ weeks

### 6A: Best Practices Guide
- [ ] "Deploying SLM Agents Safely" — practitioner guide
- [ ] Checklist for reliability testing before deployment
- [ ] Case studies from real deployments

### 6B: Model Development Feedback
- [ ] Collaborate with model developers on reliability improvements
- [ ] Propose reliability-specific fine-tuning methods
- [ ] Develop "reliability score" for model cards

### 6C: Policy Recommendations
- [ ] Safety standards for SLM agent deployment
- [ ] Transparency requirements for reliability reporting
- [ ] Regulatory framework input

---

## Current Metrics
| Metric | Value | Target (Phase 4) |
|--------|-------|-------------------|
| Models evaluated | 5 | 15+ |
| Tasks | 14 | 50+ |
| Reliability dimensions | 4 | 6+ |
| Citations | 22 | 50+ |
| Paper pages | 12 | 20+ |
| Figures | 4 | 12+ |
| Experiments run | 5 | 50+ |

## How to Contribute
1. Pick an item from any phase
2. Create a GitHub issue
3. Run `python automate.py run --model <model>` for experiments
4. Run `python automate.py publish` to regenerate paper
5. Submit a PR with results
