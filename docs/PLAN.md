# Where this is going

Writing this down mostly so I don't lose track. What's done, what's next, what I'd like to get to eventually.

## Done so far

**Phase 1 — the basics:**
- [x] 4-dimension reliability framework (consistency, robustness, fault tolerance, safety)
- [x] 14-task eval suite across 7 categories
- [x] ReAct harness with 6 tools
- [x] First 5 models: Llama 3.2 3B, Phi-3.5, Qwen 2.5 7B, Mistral 7B, Gemma 2 9B
- [x] 4 figures, 12-page paper (Wilson CIs, Cohen's h, Pearson r)
- [x] All 22 citations checked against real arXiv IDs — verify every one, always
- [x] `python automate.py publish` regenerates the whole paper

**Phase 2 — expansion, also done:**
- [x] 4 more models (9 total). Two surprises: Llama 1B at 56% beat Llama 8B at 24%. DeepSeek-R1 7B got 0% — it just fights the ReAct format. Qwen Coder 7B best overall at 85%.
- [x] 31-task suite across 8 categories. Coding easiest (~89%), data analysis and safety hardest (~25%).
- [x] Cost-reliability tradeoff (accuracy vs latency vs VRAM), temperature sweep, per-category breakdown
- [x] One task every model passes (COM-4), three every model fails (DA-4, MSR-2, SAF-3)

Still missing: web nav tasks, multi-turn conversations, JSON extraction, anything with vision models.

## Next up

Real fault injection — rate limits, timeouts, malformed tool outputs, adversarial inputs. Then comparing ReAct against Reflexion and Plan-and-Solve on the same tasks. And a proper safety deep dive (jailbreaks, multi-step harmful requests, exfiltration scenarios).

## Publishing

arXiv preprint first, can do that anytime. After that maybe an ICML workshop, or NeurIPS Datasets & Benchmarks if I package this as a real benchmark release. Paper still needs failure trajectory examples, an ablation study, frontier models on the same benchmark, and a bigger lit review (aiming 50+ citations).

## Later / maybe

- pip package (`pip install agent-reliability`) plus a leaderboard on GitHub Pages
- Multilingual tasks, multimodal tasks (LLaVA, Qwen-VL)
- A practitioner guide — "deploying SLM agents safely" with a pre-deployment checklist
- Reliability scores for model cards, policy input — ambitious, we'll see

## By the numbers

9 models, 31 + 14 tasks, 25 citations, 12 pages, 5 figures. Targets: 15+ models, 50+ tasks, 50+ citations, 20+ pages. Main findings so far: code training transfers to reliability; reasoning-distilled models break ReAct; neither capability nor reliability scales cleanly (r=0.289 and r=-0.179, both non-significant).

## Contributing

Pick something from the list, open an issue first so we don't double up, run experiments with `python automate.py run --model <model>`, regenerate with `python automate.py publish`, and PR the results.
