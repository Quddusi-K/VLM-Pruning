# vlm-visual-exit

Training-free, per-image **visual-token exit** for LLaVA-style vision-language models: drop all visual tokens at a
layer chosen per image by a cheap signal (logit-lens stability or a one-row counterfactual probe), with SDPA/
FlashAttention and KV-cache compatibility. Grew out of an analysis of attention-reliability-guided pruning
("InfoPrune", negative result) and a FastV K×R sweep.

- `HANDOFF.md` — read first: state, key results, layout, conventions, next steps
- `docs/` — results & plan, literature & concepts
- `notebooks/` — Kaggle notebooks (generated from `tools/notebook_builders/`)
- `tools/analysis/` — CPU analysis over result JSONL
- `results/` — untracked; see `results/README.md`
