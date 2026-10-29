# Paper draft (LaTeX)

`main.tex` is the CounterPrune draft: generic `article` class so it compiles anywhere; swap in the venue class later.
Every number is copied from `docs/results-and-execution-plan.md` (which is recomputed from `results/` by
`tools/analysis/`); when results change, update the doc first, then the tables here. `\todo{}` marks the parts that
wait for the LLaVA-NeXT follow-up run or the analysis-section condensation.

Build: `pdflatex main.tex` (twice). No bibtex needed yet (inline bibliography).
