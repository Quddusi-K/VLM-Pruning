# CLAUDE.md — working conventions for this repo

Project: training-free, per-image visual-token exit for LLaVA-style VLMs (kernel-compatible). Start with HANDOFF.md,
then docs/results-and-execution-plan.md (numbers) and docs/research-and-literature.md (concepts, literature).

## Rules
- Notebooks are generated: edit `tools/notebook_builders/build_*.py`, then `cd tools/notebook_builders && python build_X.py`.
  Do not hand-edit the .ipynb files (except to inspect outputs).
- Every experimental number in docs/ must come from files in results/ (recompute; never copy from chat).
  Use the scripts in tools/analysis/ and extend them rather than doing one-off math.
- Paired comparisons use exact McNemar; report accuracy, agreement with the unpruned model, prefill FLOPs (FastV
  formula over actual per-layer sequence lengths) and measured latency. Compare at matched FLOPs or matched fidelity.
- MME ids repeat (two questions per image): align records by position, not id.
- Eval loops are resumable JSONL keyed by example id; result file names encode the method
  (`{bench}__{method}.jsonl`, e.g. `gqa__lens_t0.05_m1_e1.jsonl`). Changing method code without changing the name
  silently reuses stale results — rename or delete.
- GPU work runs on Kaggle (2×T4, fp16, SDPA). This repo holds code, docs and analysis; results/ is untracked.
- After a Kaggle run: zip results/cache before the session ends; attach the zip as a dataset for restore cells.
- Keep docs/results-and-execution-plan.md's status table and "open questions" current whenever results land.
- The method name "InfoPrune" is taken by another paper (arXiv 2511.19518); do not use it for the exit method.

## Default method settings
lens exit: tau=0.05, m=1, e_min=1 (one global threshold; no per-benchmark tuning). cf exit: tau=0.05, m=2, e_min=12.
No uniform subsampling in the main method.
