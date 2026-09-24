# Audit notes, Aug 4-6

Before I called the paper done, I went through it line by line for three days. This is what I checked and fixed.

## What I looked at

- Did the numbers in the paper match the raw JSON? Mostly yes. A few places where I'd written claims the data didn't fully support — I softened those.
- Did the code do what the paper said? Yes for consistency, robustness, faults, safety. But the consistency formula wasn't written down anywhere — I added it.
- Were the citations real? 37 entries, 7 had junk in the title field. Fixed. One wrong arXiv ID (PAIR), one wrong year (DeepSeek-R1). Fixed.
- Did the stats hold up? n=9 is small, so I added Spearman + permutation checks next to the Pearson numbers. Safety per-cell counts are tiny — I say so now.

## Things I found

1. Gemma 2 9B: 60% consistency at 0% success. It fails the same way every time. That's determinism, not reliability — I say that now.
2. Llama 3.1 8B: some scores were single-task estimates. Flagged with a dagger, explained in text.
3. Safety: 14/54 overall. Confidentiality 0/9, scope 1/18. I don't present weak cells as firm anymore.
4. "First comprehensive" type claims — qualified with "to our knowledge, as of Aug 2026".
5. Missing keywords + data availability — added.

## End state

24 pages, 40 refs, compiles clean, all claims traceable to data or code. Good enough to share.
