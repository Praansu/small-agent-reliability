Where I left off.

I went through the paper line by line against the raw JSONs before I called it done:
- all 54 Table-1 cells match raw data
- the 9 Table-2 accuracies match
- reran the three Spearman pairs in scipy, got the same numbers
- fixed one real mistake: DS-R1 run-success line said "90-100% for others" but Llama 1B is 83.9%, so now it says 83.9-100%

Also added two figures from real data (radar panel, latency-vs-accuracy with DS-R1 at 114.7s), added the error-rate column (DS-R1 38.7% vs <=9.7% others), framed limitations as threats to validity.

Leftovers if I ever come back to it:
- bias/confidentiality are still n=1 per model, needs more prompts
- temperature sweep only covers 3 models
- consistency is 3 runs, code defaults to 5
- sanitizer ~20% is still an estimate, not measured

To rebuild the PDF: cd paper, pdflatex main, bibtex main, pdflatex twice.
