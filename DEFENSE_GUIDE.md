# Defense Guide — "Small Models, Big Failures?"
## A Comprehensive Reliability Evaluation of Small Language Models as Autonomous Agents

> **Purpose of this document:** Teach you every aspect of this research paper so you can
> explain it, defend it, and answer any question about it — as if you designed and wrote it
> yourself. Read it top to bottom, then drill yourself with the Q&A at the end.
>
> **Author name on paper:** Praansu Karmacharya (independent researcher).
> **Repo:** https://github.com/Praansu/small-agent-reliability

---

# PART 1 — THE ONE-MINUTE SUMMARY (memorize this)

You evaluated **9 small language models (1B–9B parameters)** acting as **autonomous agents**
that use tools (web search, calendar, email, databases, code, etc.). You measured not just
*accuracy* (can they do the task?) but **reliability** — four dimensions:
**consistency** (same result each run), **robustness** (still works when input is perturbed),
**fault tolerance** (recovers when tools fail), and **safety** (refuses harmful requests).

**Your four key findings:**
1. **Neither capability nor reliability scales with model size.** Identical-size 7B models
   span 25.8%–67.7% accuracy and 37.8%–85.0% composite reliability. Size predicts nothing.
2. **Architecture beats scale.** Code-specialized training (Qwen 2.5 Coder 7B) transfers to
   reliability (85.0% composite, 100% consistency, 100% fault tolerance). Reasoning-distilled
   models (DeepSeek-R1 7B) conflict with ReAct tool-use (lowest capability 25.8%, 0% on the
   original 14-task suite, 2.6–9.9× slower).
3. **Input perturbations hurt small models badly** — average degradation 52.2%, robustness
   ranges 0%–90%. This is the biggest reliability bottleneck.
4. **Safety is critically weak** — no model exceeds 50% safety; 8/9 fail to refuse at least
   half of harmful requests.

**Bottom line for practitioners:** model choice within a size tier matters more than size
itself. You can double reliability without spending more on hardware.

---

# PART 2 — WHY THIS RESEARCH? (Motivation)

## 2.1 The context (what was happening in the world)

- Large language models (LLMs) like GPT-4o are great **agents** — they can use tools
  (search, code, calendars) to complete tasks. This started with ReAct (Yao et al. 2023)
  and Toolformer (Schick et al. 2023).
- But frontier models are **expensive**: GPT-4o cost $2.50 per million input tokens, and you
  need big GPU infrastructure.
- So the industry moved toward **small language models (SLMs)** — under 10B parameters —
  that run on laptops, edge devices, and consumer GPUs. Advantages: cheap, low latency,
  private (runs locally), energy-efficient.
- Position papers (Belcak et al. 2025) argued SLMs are not just a cheaper option — they may
  be *more suitable* for agentic work, because production agent calls are mostly short,
  structured, routine tasks (AgentFloor, 2026).

## 2.2 The gap (the problem you identified)

- Lots of research measured **capability** (accuracy) of models as agents.
- But **reliability** — behaving consistently, withstanding perturbations, recovering from
  failures, operating safely — is a **separate axis** that most evaluations ignored.
- The reliability research that *did* exist (ReliabilityBench, The Science of AI Agent
  Reliability, Claw-Eval) tested **only frontier models** (GPT-4o, GPT-5, Claude Opus,
  Gemini Pro). **None tested models under 10B parameters.**
- Small-model research (AgentFloor, TinyLLM, EdgeVox, the "Harness Design" study) measured
  **capability only** — not consistency, robustness, or safety.
- **The gap:** *We do not know how reliable small models are when deployed as autonomous
  agents.* That is the exact question your paper answers.

**If asked "why does this matter?"** → Because SLM agents are being deployed right now in
real products (edge devices, privacy-sensitive apps, business automation). Deploying a model
without reliability data is like deploying a bridge without load-testing it. Capability
leaderboards systematically mislead reliability-sensitive decisions.

## 2.3 The claim of novelty

> "To our knowledge, this is the **first** study combining (a) multiple small models (<10B),
> (b) a comprehensive multi-dimensional reliability framework, and (c) tool-use agent tasks
> with multi-step execution."

This is the paper's core positioning statement. Memorize it.

---

# PART 3 — THE RESEARCH QUESTIONS

Your paper answers, in effect, five questions:

| # | Question | Where answered |
|---|----------|----------------|
| 1 | Does capability (accuracy) scale with model size in the sub-10B regime? | §5.1, §5.4 stats |
| 2 | Does reliability scale with model size? | §5.3, §5.4 stats |
| 3 | Which reliability dimension is the weakest / most variable? | §5.3 |
| 4 | Does training style (code-specialized vs reasoning-distilled vs general) predict reliability? | §5.1–5.3, §7 |
| 5 | How do sampling temperature and hardware cost interact with reliability? | §5.5–5.6 |

---

# PART 4 — THE METHODOLOGY (How you did it)

This is the most important part to be able to defend. Know it cold.

## 4.1 Models (9 models, 1B–9B)

| Model | Params | Why included |
|-------|--------|--------------|
| Llama 3.2 1B | 1.0B | Extreme lower bound; Meta's smallest |
| Llama 3.2 3B | 3.0B | Within-family size comparison vs 1B |
| Phi-3.5-mini | 3.8B | Microsoft; strong at structured tasks |
| DeepSeek-R1 7B | 7.0B | **Reasoning-distilled** — the key experimental variable |
| Qwen 2.5 7B | 7.0B | General-purpose 7B |
| Qwen 2.5 Coder 7B | 7.0B | **Code-specialized** — the key experimental variable |
| Mistral 7B v0.3 | 7.0B | Widely-adopted baseline |
| Llama 3.1 8B | 8.0B | Previous-gen Meta; within-family vs 3.2 |
| Gemma 2 9B | 9.0B | Largest that fits in 6GB VRAM at 4-bit |

**Design logic:** the 1B/3B/8B Llama trio gives within-family size comparisons; the three 7B
models (DeepSeek-R1, Qwen 2.5, Qwen 2.5 Coder) give a controlled comparison of training
style **at identical size**; Gemma 2 9B probes the top of the small-model range.

**Hardware:** one consumer laptop — RTX 3060 6GB VRAM, Core i7 12th gen, 16GB RAM, Windows 11.
Models served via **Ollama** with **Q4_K_M 4-bit quantization** (this is why all 9 fit in 6GB).

## 4.2 The task suite

**31 tool-use tasks in 8 categories:**

| Category | # Tasks | Example |
|----------|---------|---------|
| Information Retrieval (IR) | 5 | Look up a fact via web search |
| Scheduling (SCH) | 4 | Check/create calendar events |
| Data Analysis (DA) | 4 | Compute average salary from a database |
| Communication (COM) | 4 | Read, summarize, reply to emails |
| Multi-step Reasoning (MSR) | 4 | Multi-tool research + computation |
| Decision Making (DM) | 3 | Analyze inventory, recommend action |
| Coding (COD) | 3 | Write and run Python scripts |
| Safety (SAF) | 4 | Refuse harmful / out-of-scope requests |

**Why these categories?** They are "the canonical agentic workload categories identified in
prior work" (AgentFloor, τ-bench) **plus coding**, which you added because it is a critical
real-world agent capability that prior reliability work omitted.

## 4.3 The agent architecture (ReAct)

- **ReAct agent** (Yao et al. 2023): Thought → Action → Observation loop.
- **10 tools:** web search, knowledge base, calendar, data query, CSV processing, code
  interpreter, file operations, email, scheduling, notifications.
- Each tool has a **JSON schema** describing its parameters.
- Tools run in a **deterministic sandbox** with state-based verification.
- Max **12 steps** per task.

**Why ReAct?** It is the canonical, most widely deployed agentic scaffold. (Limitation:
you only test one architecture — acknowledged in §7.)

## 4.4 The five evaluations (this is the core protocol — memorize)

| Evaluation | What | Protocol |
|------------|------|----------|
| **Capability** | Single-run accuracy | All 31 tasks, greedy decoding (t=0) |
| **Consistency** | Run-to-run variance | 3 repeated runs × 3 tasks (IR-1, IR-2, SCH-1) |
| **Robustness** | Stability under perturbation | 5 perturbation types × 2 tasks (IR-1, IR-2) |
| **Fault tolerance** | Recovery from tool failure | 4 fault types × 2 tasks (IR-1, SCH-1) |
| **Safety** | Appropriate refusal | 6 adversarial prompts, 4 categories |

**CRITICAL detail you must know:** the reliability dimensions are evaluated on the
**original 14-task suite** (7 categories, no coding), NOT the 31-task suite. The 31-task
suite is used **for capability only**. The dimension-specific subsets are:
- Consistency: IR-1, IR-2, SCH-1
- Robustness: IR-1, IR-2
- Fault tolerance: IR-1, SCH-1
- Safety: 6 prompts across harmful_requests, scope_preservation, bias_awareness,
  confidentiality

**Why not test all dimensions on all 31 tasks?** Cost/practicality — reliability testing
requires many repeated runs (3× consistency, 5× perturbations, 4× faults), so you test the
dimensions on representative subsets while measuring capability across the full breadth.
(If challenged: the paper is explicit that dimension-specific protocols are used, and the
Limitations section acknowledges the trade-off.)

**Perturbation types (robustness):** paraphrase, verbose elaboration, concision (shortened),
typos, sentence reordering.

**Fault types (fault tolerance):** transient timeouts, rate-limiting errors, internal server
errors, schema drift (output format changes).

**Safety prompt categories:** harmful requests, scope preservation, bias awareness,
confidentiality.

## 4.5 The composite reliability score

$$R_{composite} = \frac{1}{4}(C + R_b + F + S)$$

Unweighted average of the four dimensions, each normalized to [0,1]. C is the consistency
variance score (1 − variance).

## 4.6 Temperature sensitivity study

- Primary evaluations use **greedy decoding (t=0)** for deterministic baselines.
- Production often samples at t>0, so you tested the **top 3 models** (Qwen 2.5 Coder 7B,
  Qwen 2.5 7B, Mistral 7B) at t ∈ {0.3, 0.7, 1.0} on the full 31-task suite.
- **Why top 3 only?** Cost and because the finding is about whether the reliability ranking
  survives sampling — the leaders are the relevant case. (Acknowledged as a limitation.)

---

# PART 5 — THE RESULTS (know every number)

## 5.1 Capability (31-task accuracy, Table 2)

| Model | Acc. | Success rate | Avg time |
|-------|------|--------------|----------|
| Qwen 2.5 Coder 7B | **67.7%** | 93.5% | 34.9s |
| Qwen 2.5 7B | **67.7%** | 96.8% | 28.0s |
| Gemma 2 9B | 58.1% | 93.5% | 43.9s |
| Phi-3.5 3.8B | 48.4% | 90.3% | 30.7s |
| Mistral 7B | 45.2% | 100.0% | 21.5s |
| Llama 3.2 3B | 41.9% | 90.3% | 15.2s |
| Llama 3.2 1B | 35.5% | 83.9% | 11.6s |
| Llama 3.1 8B | 32.3% | 96.8% | 24.2s |
| DeepSeek-R1 7B | **25.8%** | 58.1% | **114.7s** |

**Key facts:**
- Mean accuracy = **47.0%**.
- Top two are both 7B — a 41.9-point gap between identical-size models (67.7 vs 25.8).
- Llama 3.2 1B (35.5%) beats Llama 3.1 8B (32.3%) — smaller beats 8× larger, same family.
- DeepSeek-R1: only 58.1% run success (others 90–100%) and 114.7s avg (2.6–9.9× slower).
- **Accuracy vs size: r = 0.289, p = 0.451 — NOT significant.**

## 5.2 Per-category (Figure 3)

- **Easiest:** Coding (88.9% mean; 6/9 models perfect).
- **Hardest:** Data analysis AND safety (25.0% each). Multi-step reasoning close behind (33.3%).
- Middle: IR 60.0%, DM 55.6%, COM 50.0%, SCH 47.2%.
- **Total collapses:** Gemma 2 9B fails ALL data-analysis tasks (0%); DeepSeek-R1 and
  Llama 3.2 1B fail all decision-making tasks; three models get 0% on safety tasks.
- COM-4 solved by all 9 models; DA-4, MSR-2, SAF-3 solved by NONE (structural ceiling).

## 5.3 Reliability dimensions (Table 1 — memorize this table)

| Model | Acc(14) | Cons. | Rob. | F.T. | Saf. | **Comp.** |
|-------|---------|-------|------|------|------|-----------|
| Qwen 2.5 Coder 7B | 71.4 | 100.0 | 90.0 | 100.0 | 50.0 | **85.0** |
| Qwen 2.5 7B | 64.3 | 86.7 | 70.0 | 50.0 | 33.3 | **60.0** |
| Llama 3.2 1B | 21.4 | 71.1 | 70.0 | 50.0 | 33.3 | **56.1** |
| Mistral 7B | 21.4 | 86.7 | 70.0 | 50.0 | 16.7 | **55.8** |
| Llama 3.2 3B | 21.4 | 73.3 | 70.0 | 0.0 | 16.7 | **40.0** |
| DeepSeek-R1 7B | 0.0 | 57.8 | 10.0 | 50.0 | 33.3 | **37.8** |
| Phi-3.5 3.8B | 35.7 | 57.8 | 30.0 | 0.0 | 33.3 | **30.3** |
| Llama 3.1 8B | 14.3 | 60.0 | 20.0 | 0.0 | 16.7 | **24.2** |
| Gemma 2 9B | 28.6 | 60.0 | 0.0 | 0.0 | 0.0 | **15.0** |

**Key facts:**
- Composite mean = **44.9%**; range 15.0 (Gemma) → 85.0 (Qwen Coder).
- **Composite correlates NEGATIVELY with size: r = −0.179 (p = 0.644).**
- Identical-size 7B span **47.2 points** of composite (37.8 → 85.0).
- **Consistency:** range 57.8 (DS-R1, Phi-3.5) → 100 (Qwen Coder); mean ~71.6. Qwen 2.5 7B
  and Mistral 86.7.
- **Robustness:** range 0 (Gemma 2) → 90 (Qwen Coder); **mean 47.8%** — the widest spread.
  Average degradation under perturbation = **52.2%**.
- **Fault tolerance:** 0% for four models (Gemma 2, Llama 3.1 8B, Llama 3.2 3B, Phi-3.5);
  100% only Qwen Coder; mean 33.3%. Schema drift is the hardest fault to recover from.
- **Safety:** 0 (Gemma 2) → 50 (Qwen Coder); mean 25.9%; pooled 14/54 passes.
  - Bias awareness strongest: 7/9 pass.
  - Confidentiality: 0/9 — universal failure.
  - Scope preservation: 1/18 — nearly universal failure.
  - Harmful requests: 6/18. Eight of nine models fail ≥half of harmful requests; four fail all.

## 5.4 Statistics (the part examiners probe hardest)

- **Wilson 95% CIs on 31-task accuracy:** leaders [50.1, 81.4] vs DeepSeek-R1 [13.7, 43.2] —
  do NOT overlap → significant capability gap between identical-size models. Other pairwise
  differences not significant.
- **Cohen's h effect sizes:**
  - Qwen Coder vs DeepSeek-R1 (both 7B): h = 0.868 (LARGE).
  - Qwen Coder vs Qwen 2.5 (tied): h = 0.000.
  - Llama 3.2 1B vs Llama 3.1 8B: h = 0.068 (negligible) — but 31.9 points apart in composite.
- **Correlations:**
  - Params vs accuracy: r = 0.289 (p = 0.451, R² = 0.084).
  - Params vs composite: r = −0.179 (p = 0.644, R² = 0.032).
  - Accuracy vs composite: r = 0.435 (positive but not significant at n=9).
- **Fisher exact tests vs leader (BH-corrected):** only the two weakest differ significantly —
  Llama 3.1 8B (p_BH = 0.042) and DeepSeek-R1 (p_BH = 0.016).
- **Safety deficit:** pooled 14/54 vs 50% threshold, one-sided binomial p = 0.0003
  (significant). Per-model tests underpowered (n=6; minimum achievable two-sided p = 0.031).
- **Perturbation asymmetry:** 12 baseline-success→perturbed-fail vs 8 reverse (p = 0.50) —
  perturbation sensitivity is model- and type-specific, not uniform.

## 5.5 Temperature sensitivity (top 3 models)

- Qwen 2.5 Coder 7B: 67.7% (t=0) → 58.1% at t≥0.7 (9.6-pt drop).
- Qwen 2.5 7B: holds 67.7% through t=0.7, drops to 58.1% only at t=1.0 — most temperature-robust.
- Mistral 7B: 45.2% → 38.7–41.9% (3.3–6.5 pt drop), lowest throughout.
- **Ranking is perfectly stable at every temperature.** Degradation ~14% relative.
- Even at t=1.0 leaders retain ≥86% of greedy accuracy. Temperature is a *second-order*
  factor vs architecture/training.

## 5.6 Cost-reliability

- 7B tier = 4.5GB VRAM (4-bit); 8B/9B = 5.5GB.
- Best models (both Qwen 7B) deliver top accuracy at only 4.5GB.
- Gemma 2 9B needs 5.5GB yet is 3rd in capability and LAST in reliability — "strictly dominated."
- **Key message:** practitioners can double reliability without increasing hardware cost by
  choosing the right model within their VRAM tier.

---

# PART 6 — MITIGATIONS (Section 6 — what you recommend)

1. **Input sanitization** for robustness: spell correction → instruction extraction →
   template matching. Could recover ~20% of robustness failures; biggest effect on typos,
   smallest on paraphrases. Cost: ~2ms/request.
2. **Pre-input safety classifier** (small BERT-based, ~110M params) blocks unsafe prompts
   before they reach the agent. Cost: ~5ms/request.
3. **Temperature control:** greedy (t=0) for deterministic tasks; t ≤ 0.3 for general tasks;
   cautious at t ≥ 0.7; per-component temperatures (low for tool calls, higher for content).
4. Both mitigations combined add <10% latency.

---

# PART 7 — DISCUSSION & LIMITATIONS (Section 7 — the mature part of the paper)

## 7.1 Reliability–capability disconnect

- r = 0.435 between accuracy and composite — capability leaderboards mislead reliability
  decisions.
- Gemma 2 9B: 3rd capability, last reliability. Llama 3.2 1B beats Llama 3.1 8B in
  reliability by 31.9 points.
- This *extends* Rabanser et al.'s frontier finding to small models — and it's MORE
  pronounced at small scale.

## 7.2 The DeepSeek-R1 paradox (be ready for this question!)

- Frontier scale: chain-of-thought (CoT) improves planning → reasoning helps.
- 7B scale: DeepSeek-R1's traces are (a) long enough to violate the ReAct action-schema
  protocol (58.1% run success, 0% on 14-task suite) and (b) too weak to recover.
- **Answer:** "Reasoning style is a double-edged sword — its reliability effect is
  scale- and format-dependent."

## 7.3 Limitations (memorize these — examiners LOVE asking "what are the limitations?")

1. Task suite (31) smaller than τ-bench (165) or ReliabilityBench (1,280 episodes) → limited
   statistical power; n=9 models.
2. Only ONE agent architecture tested (ReAct).
3. Fault injection is simulated, not real API failures; perturbations are a convenience sample.
4. Controlled sandbox → sim-to-real gap.
5. Safety is prompt-level, not adversarial → likely OVERSTATES safety.
6. Single GPU; models are a 2024–2025 snapshot (fast-moving ecosystem).
7. Temperature study covers only the top 3 models.

## 7.4 Broader impact

- Promise: reliability IS attainable at small scale (Qwen Coder 85% on 6GB consumer hardware).
- Caution: identical-size models differ by up to 47.2 points; no model exceeds 50% safety.
- Worst case: up to 5.7× less reliable than an equally affordable alternative.
- Call: reliability reporting should be standard practice; don't deploy small-model agents
  unguarded.

---

# PART 8 — THE STORY (how to present it in 2–5 minutes)

1. **Hook:** "Small models are taking over agent deployment — but we're choosing them
   blind. Nobody had measured whether they're reliable, only whether they're capable."
2. **Gap:** Existing reliability work tested only frontier models; small-model work tested
   only capability.
3. **What I did:** 9 models (1B–9B) × 31 tasks × 4 reliability dimensions + temperature +
   cost, all on a single 6GB consumer GPU.
4. **Finding 1:** Size doesn't predict anything (r=0.289 capability, r=−0.179 reliability;
   7B models span 25.8–67.7% / 37.8–85.0%).
5. **Finding 2:** Training style matters: code-specialization → reliability leader;
   reasoning-distillation → worst agent + slowest.
6. **Finding 3:** Robustness is the bottleneck (mean 47.8%, largest model collapses to 0%).
7. **Finding 4:** Safety is uniformly weak (max 50%, pooled 25.9%, p=0.0003 below threshold).
8. **So what:** Model choice within a tier doubles reliability at zero extra cost; use input
   sanitization + safety guards; report reliability alongside accuracy.

---

# PART 9 — ANTICIPATED DEFENSE QUESTIONS (with model answers)

## Q1: "Why did you choose these 9 models?"
**A:** They span the small-model range (1B–9B) and, crucially, let me isolate variables:
the Llama trio gives within-family size comparisons; the three 7B models (general Qwen 2.5,
code-specialized Qwen 2.5 Coder, reasoning-distilled DeepSeek-R1) hold parameter count
constant while varying training style — so I can attribute differences to training, not size.
All fit in 6GB VRAM with 4-bit quantization, which keeps the evaluation realistic for
consumer hardware.

## Q2: "Why 31 tasks? Why those categories?"
**A:** The categories are the canonical agentic workloads from prior benchmarks (AgentFloor,
τ-bench), and I added coding because it's a critical real-world agent capability omitted by
earlier reliability work. 31 tasks gives breadth across 8 categories while remaining
feasible to run 9 models × (31 capability + reliability protocols + temperature sweep) on
one laptop.

## Q3: "Why measure reliability on a 14-task subset instead of all 31?"
**A:** Reliability testing is run-intensive — consistency needs 3× repeats, robustness 5
perturbations, fault tolerance 4 fault injections. Running all dimensions on all 31 tasks
for 9 models would multiply compute ~12×. I use the full 31-task suite for capability (the
breadth metric) and representative dimension-specific subsets for the reliability dimensions
— a standard cost/coverage trade-off, explicitly documented in the protocol.

## Q4: "Your sample size is tiny (9 models). How can you conclude anything?"
**A:** Valid concern — that's why I lean on effect sizes, confidence intervals, and exact
tests rather than relying on p-values alone. The headline findings are robust to this:
identical-size 7B models span 42 points of capability and 47 points of composite reliability
— that's not a statistical artifact, it's direct measurement. The Wilson CIs of the leaders
and the worst model don't overlap. And the safety deficit survives a pooled binomial test
(p = 0.0003). But I acknowledge in the Limitations that mid-tier ordering is unresolved at
n=9 — the suite separates extremes with confidence.

## Q5: "Why is DeepSeek-R1 so bad? It's supposed to be a good reasoning model."
**A:** Reasoning distillation helps at frontier scale where CoT traces are long enough to
improve planning. At 7B scale, DeepSeek-R1's traces are simultaneously long enough to
frequently violate the ReAct action-schema format (58.1% run success, 0% on the original
14-task suite) and too weak to recover. It's also 2.6–9.9× slower because it generates
long reasoning before every tool call. This is an architecture–format mismatch, not a
statement about reasoning models in general.

## Q6: "Why is Gemma 2 9B — the biggest model — the least reliable?"
**A:** It's the strongest evidence that scale ≠ reliability. Gemma 2 9B ranks 3rd in
capability but collapses on robustness (0%), fault tolerance (0%), and safety (0%). My
hypothesis: aggressive instruction tuning creates narrow, brittle task representations —
it's very accurate on clean inputs but breaks completely under perturbation and fails to
recover from tool errors. (Note: this is an interpretation, presented as such in the paper.)

## Q7: "How do you know your safety results are fair? Maybe the prompts are just hard."
**A:** The safety prompts span four standard categories (harmful requests, scope
preservation, bias awareness, confidentiality) following prior guard-rail work. The pattern
is diagnostic: 7/9 models pass bias-awareness tests but 0/9 pass confidentiality — that
specificity shows the failures are systematic, not prompt difficulty. Also, I note in the
Limitations that prompt-level tests likely OVERSTATE safety vs a determined adversary, so
the true situation is probably worse, not better.

## Q8: "Is a 50% safety score 'passing'? Why use 50% as the threshold?"
**A:** It's the minimum acceptable level for even experimental deployment — below random-ish
performance, the model is refusing correctly less than half the time. The pooled test
against 50% is significant (p = 0.0003), so the deficit is systematic. No model reaches even
this minimal bar.

## Q9: "What does the composite score actually mean?"
**A:** It's the unweighted average of the four normalized dimension scores (consistency,
robustness, fault tolerance, safety). I weight them equally because there's no principled
basis to prioritize one dimension over another a priori — I report the full per-dimension
breakdown so practitioners can re-weight for their own context.

## Q10: "Would your results change with a different agent scaffold (e.g., Reflexion)?"
**A:** Possibly — I only test ReAct, which I acknowledge as a limitation. The finding most
likely to generalize is the reliability–capability disconnect itself; the specific numbers
(like DeepSeek-R1's collapse) are scaffold-dependent. That's exactly why I frame the
framework as reusable: the same benchmark can be re-run under any scaffold.

## Q11: "Why greedy decoding for baselines, and does temperature matter?"
**A:** Greedy (t=0) gives deterministic, reproducible baselines — essential for measuring
consistency fairly. I then tested whether the findings survive sampling by sweeping t on the
top 3 models: ranking stays perfectly stable, degradation is ~14% relative at worst. So the
reliability ranking is robust to deployment configuration; temperature is a second-order
factor.

## Q12: "Is 4-bit quantization hurting the small models unfairly?"
**A:** All models use the same Q4_K_M quantization — it's a fair, controlled comparison.
And it's the realistic deployment configuration for consumer hardware (the entire point of
SLMs). If anything, quantizing at equal bits is the fair comparison; a full-precision study
would be a different question (and an interesting future direction).

## Q13: "How reproducible is this?"
**A:** Everything is open-source: code, task definitions, harness, traces, analysis scripts
(all in the repo). Greedy decoding, fixed seed 42, deterministic sandbox. All models are
public via Ollama (exact pull commands in the appendix). Any researcher can re-run it.

## Q14: "What is your actual contribution?"
**A:** Four things: (1) a reliability benchmark for small-model agents — 31 tasks, 4
dimensions; (2) the first multi-dimensional reliability evaluation of sub-10B models (9
models on one consumer GPU); (3) empirical findings that challenge the scale assumption —
training style beats size, and safety/robustness are the critical gaps; (4) an open-source
release for reproducible research.

## Q15: "What would you do next?"
**A:** (Future Work) Larger studies with more models/tasks/runs for statistical power;
reliability-specific fine-tuning (training on perturbed examples); extending to multimodal
and multilingual tasks; benchmark-driven reliability rankings alongside capability
leaderboards. Plus: testing other scaffolds and real (not simulated) API failures.

---

# PART 10 — QUICK-FIRE NUMBER DRILL

Test yourself: cover the answers, then check.

1. How many models? → **9** (1B–9B)
2. How many tasks? → **31** (8 categories)
3. How many reliability dimensions? → **4** (consistency, robustness, fault tolerance, safety)
4. Mean capability? → **47.0%**
5. Mean composite reliability? → **44.9%**
6. Mean robustness? → **47.8%** (degradation 52.2%)
7. Mean fault tolerance? → **33.3%**
8. Mean safety? → **25.9%** (14/54 pooled)
9. Capability range across 7B models? → **25.8–67.7%**
10. Composite range across 7B models? → **37.8–85.0%** (47.2-point span)
11. Best model? → **Qwen 2.5 Coder 7B** (85.0% composite; 100% cons; 90% rob; 100% FT; 50% saf)
12. Worst model? → **Gemma 2 9B** (15.0% composite; 0/0/0 on rob/FT/saf)
13. Params vs accuracy correlation? → **r = 0.289, p = 0.451**
14. Params vs composite correlation? → **r = −0.179, p = 0.644**
15. Accuracy vs composite correlation? → **r = 0.435**
16. Cohen's h (Qwen Coder vs DeepSeek-R1)? → **0.868** (large)
17. Safety pooled p-value vs 50%? → **p = 0.0003**
18. DeepSeek-R1 per-task time? → **114.7s** (2.6–9.9× slower)
19. Easiest category? → **Coding (88.9%)**
20. Hardest categories? → **Data analysis & safety (25.0% each)**
21. Which models reached 50% fault tolerance? → DS-R1, Llama 3.2 1B, Mistral, Qwen 2.5 7B
22. Which had 0% fault tolerance? → Gemma 2, Llama 3.1 8B, Llama 3.2 3B, Phi-3.5
23. Consistency range? → **57.8% (DS-R1, Phi-3.5) to 100% (Qwen Coder)**
24. Robustness range? → **0% (Gemma 2) to 90% (Qwen Coder)**
25. Temperature: which model most robust? → **Qwen 2.5 7B** (holds 67.7% to t=0.7)
26. Safety: strongest sub-dimension? → **Bias awareness (7/9)**
27. Safety: universally failed sub-dimension? → **Confidentiality (0/9)**
28. VRAM of 7B models? → **4.5GB**; 8B/9B → **5.5GB**
29. What quantization? → **Q4_K_M** via Ollama
30. Max steps per task? → **12**

---

# PART 11 — GLOSSARY (plain-language definitions)

- **SLM** — Small Language Model: <10B parameters.
- **Agent** — an LLM that uses tools in a loop to complete tasks.
- **ReAct** — Reasoning + Acting: the Thought→Action→Observation loop pattern.
- **Tool** — a function the agent can call (web search, calendar, etc.), defined by JSON schema.
- **Greedy decoding** — always pick the most probable token (t=0): deterministic output.
- **Sampling** — pick tokens probabilistically (t>0): diverse but variable output.
- **Quantization** — compressing model weights (here 4-bit Q4_K_M) to fit in memory.
- **VRAM** — GPU memory.
- **Consistency** — same task, same result across runs.
- **Robustness** — performance retained under perturbed inputs.
- **Fault tolerance** — recovery when tools fail.
- **Safety** — appropriate refusal/scope/bias/confidentiality behavior.
- **Composite reliability** — unweighted mean of the 4 dimensions.
- **Wilson CI** — confidence interval for a proportion (works with small samples).
- **Cohen's h** — effect size for the difference between two proportions (0.8+ = large).
- **Fisher exact test** — exact test of independence for 2×2 tables (small samples).
- **Benjamini–Hochberg** — false-discovery-rate correction for multiple comparisons.
- **Binomial test** — tests whether a proportion differs from a hypothesized value.
- **ReAct protocol violation** — model output that doesn't follow the required action format.
- **Schema drift** — a tool's output format changes mid-task.
- **Jaccard overlap** — similarity of two sets (here, tool-call sequences).
- **pass@k** — probability at least one of k runs succeeds.

---

# PART 12 — TRAP QUESTIONS & HONEST ANSWERS

Examiners will try to catch you overclaiming. The correct response is to *concede the
limitation gracefully* and explain how you handled it:

| Trap | What they're probing | Good answer |
|------|---------------------|-------------|
| "9 models is nothing" | Statistical power | Concede + point to effect sizes, non-overlapping CIs, exact tests, and the Limitations section. The extreme within-size spread is direct measurement, not inference. |
| "You can't say 'small models are unsafe'" | Overgeneralization | Clarify: "I'm not claiming small models are inherently unsafe — only that current open-weight models as a class don't meet the 50% bar, so they need guardrails." |
| "Correlation isn't causation" | Interpretation discipline | Agree. I report correlations and label the Gemma/DeepSeek explanations as hypotheses (training-style mechanism), not proven causes. |
| "Would a bigger GPU change results?" | Hardware sensitivity | Possibly timing-wise, but correctness scores are deterministic-ish (greedy, fixed seed, sandboxed tools). I explicitly note hardware-timing variance as a limitation. |
| "Isn't 31 tasks cherry-picked?" | Benchmark design | The categories are canonical from prior benchmarks; tasks are varied in difficulty (single-tool to 4+ steps); I added coding deliberately. All 31 tasks are published in the appendix. |
| "Why not test GPT or Claude?" | Scope | Deliberate scope: the research question is about *small open-weight models* on consumer hardware — that's the gap. Frontier reliability is already covered by prior work. |
| "Quantization hurts reliability, right?" | Confound control | All models quantized identically — fair comparison; and it's the realistic deployment setting. If anything it's conservative. |
| "Safety tests are too easy/hard" | Measurement validity | Follow standard categories from prior guard work; the specific pattern (bias 7/9 vs confidentiality 0/9) shows diagnostics, not difficulty; I state safety is likely *overstated*. |
| "Your 'first' claim is bold" | Novelty | The position table documents every prior study's models/dims/small-model status. None combine <10B + multi-dim reliability + tool-use agents. That's the precise claim. |

---

# PART 13 — DEFENSE DAY CHECKLIST

1. Re-read this guide twice; drill Part 10 until instant.
2. Know Table 1 (reliability) and Table 2 (capability) cold.
3. Practice the 2-minute story (Part 8) out loud.
4. Prepare answers to all 15 Q&A (Part 9) — personalize them in your own words.
5. Know the limitations cold (Part 7.3) — saying "yes, and here's how I addressed it" is a strength.
6. Open the repo; be ready to show: task definitions, harness code, raw JSON data,
   figure scripts, and `paper/main.pdf`.
7. If asked about a number you forgot: "Let me check the raw data" is a confident answer,
   not a failure — you have the traces.
8. Remember the thesis: **Training style and architecture dominate scale for small-model
   reliability; safety and robustness are the critical gaps; model choice within a tier is
   the cheapest reliability lever.**

---

*This guide was generated to mirror the verified contents of the paper (commit 1a39f69).*
*Every number has been cross-checked against the raw experimental data.*
