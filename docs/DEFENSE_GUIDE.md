# If someone grills me on this project

These are my own notes for defending the work. Short version first, details if they ask.

## The 30-second version

Small models can run tools, but they break in ways big-model benchmarks miss. I tested 9 local models on 31 tasks, found size doesn't predict reliability, and wrote down what actually helped.

## Why I did it

Most agent benchmarks use huge cloud models. I only had an RTX 3060, so I asked: what breaks when you run agents on small local models?

Three questions:
1. How reliable are they at calling tools?
2. Does bigger mean better?
3. What cheap fixes help?

## What I ran

9 models: Llama 3.2 1B/3B/8B, Phi 3.5-mini, Qwen 2.5 7B + Coder 7B, Mistral 7B, Gemma 2 9B, DeepSeek-R1 8B.

31 tasks across 8 areas: files, math, code, data, messages, web, safety, multi-step. Each task: prompt, tools, answer key, how to score.

Same harness for all: ReAct loop, max 10 steps, same tools.

## Results I quote

- Accuracy 25.8% to 67.7%, average 47.0%
- Qwen2.5-Coder-7B best at 67.7%, Llama-3.1-8B worst at 25.8%
- Llama 1B beat Llama 8B. Size isn't the story.
- Coding tasks easiest (~89%), data and safety hardest (~25%)
- Tool use vs accuracy correlation basically zero
- DeepSeek-R1 got 0% because it wouldn't follow the ReAct format — not because it's dumb, it just doesn't do that scaffold

Stats, plain:
- CIs are Wilson for rates, bootstrap for means
- Size vs accuracy: r=-0.179, p=0.644 — nothing there
- Re-run agreement 92.8%

## What helped

- Validate arguments before running the tool
- Retry once on formatting errors
- Keep tool descriptions short
- Split multi-step tasks into smaller steps
- Refuse unsafe tasks explicitly

None of this is fancy. It just cuts stupid failures.

## Weak spots I'll admit

- Only 31 tasks. Enough to see patterns, not enough to claim generality.
- Only Ollama models I could fit on one GPU.
- I scored outputs myself with an LLM judge + spot checks.
- One machine, one room temp, one GPU — noise happens.
- ReAct only. Other scaffolds might change rankings.

## Questions I expect

**Why not bigger models?** Couldn't fit them. That's the point — test what runs locally.

**Why did 1B beat 8B?** My guess: 8B tried longer chains and wandered off. Smaller stayed literal. Still just a guess.

**Is 31 tasks enough?** For a first cut, yes. It won't be the final word.

**Why zero for DeepSeek?** It fought the ReAct template. That's a scaffold mismatch, not a capability proof.

**Can I rerun this?** Yes: `python code/run_experiments.py --help`. Takes about a day on a 3060.

## Numbers I keep in my head

9 models, 31 tasks, ~855 trials, 47.0% mean accuracy, 92.8% re-run agreement, 10 tools, 5 figures, 25 refs.

If I forget everything else, those six numbers tell the story.
