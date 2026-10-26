# Handoff: training-free per-image visual-token exit for VLMs

*Written 26 October 2026 for continuing the project in Claude Code. Read this first, then
`docs/results-and-execution-plan.md` (all numbers) and `docs/research-and-literature.md` (concepts, literature,
positioning).*

---

## 1. The project in one paragraph

We accelerate LLaVA-style VLMs by **dropping all visual tokens partway through the decoder, at a layer chosen per
image by a cheap, training-free signal**, while staying compatible with fused attention kernels (SDPA /
FlashAttention) and the KV cache. The signal is the **logit-lens stability** of the last token (KL between consecutive
layers' next-token distributions ≤ τ) or a **one-row counterfactual probe** (how much the attention update changes when
visual keys are removed). On LLaVA-1.5-7B with one global τ=0.05, the exit has no statistically significant accuracy
loss on POPE, MME, GQA or TextVQA at 68% of baseline prefill FLOPs on average (59–60% on POPE/MME, 73% GQA, 80%
TextVQA); the only fixed exit layer that is equally safe everywhere (VTW-style, E=28) costs 88%. Measured prefill
latency tracks FLOPs. The project started as "InfoPrune" (choosing *which layer's attention to trust* for FastV-style
token ranking); that hypothesis failed cleanly and became the motivation/analysis section.

## 2. Where things stand (status table)

| Stage | Status | Evidence |
|---|---|---|
| Phase 0: FastV / SparseVLM reproduction (POPE) | done | results doc §2 |
| Phase 1: InfoScore port, InfoPrune (per-image attention-reliability layer choice) | done, **negative** | §3–§6, §8 findings 1–9 |
| FastV K×R sweep (K∈{1,2,3,4,7,10,17}, R∈{.5,.75,.9}) | done | §5–§6 |
| Phase A: oracle exit layers + exit-signal profiling, 4 benchmarks | done, **pass** in the one-threshold setting | §8A, §8C |
| Phase B: SDPA+KV-cache engine, full-generation eval, latency | done | §8D |
| Phase B sweep: fixed-E vs lens-τ vs cf-τ vs uniform, 17 methods × 4 benchmarks | done | §8E |
| Engine v2: cache-based cf probe, FastV-in-engine, LLaVA-NeXT-7B with same τ | **running on Kaggle now** (`notebooks/phaseB2-engine-v2.ipynb`) | — |
| FlashAttention-2 timings (needs Ampere GPU), larger subsets, RefCOCO, write-up | not started | — |

## 3. Key results to remember

- **Fixed exit layers fail on tasks that need the image late**: fixed E16 keeps 99.6% of POPE answers but 27% of
  TextVQA answers. TextVQA needs E31 (97% FLOPs) for 98% fidelity; POPE/MME need E15 (49%).
- **lens τ≤0.1 / cf τ≤0.05: no significant accuracy change on any benchmark** (exact McNemar). Fixed E≤24 is
  significantly worse on GQA and/or TextVQA.
- **At matched fidelity, per-image exit beats fixed exit by 8–13 pt FLOPs on GQA and 3–4 pt on TextVQA, and ties on
  POPE/MME** (where visual dependency is homogeneous — nothing to adapt to).
- **Adaptivity is real**: cf exit layer vs per-example oracle exit layer, Spearman 0.51 (GQA), 0.57 (TextVQA); GQA
  open-ended "query" questions exit at layer ~26, yes/no "verify/logical" at ~18.
- **Uniform visual subsampling is an ablation, not part of the method**: −2.3 pt mean accuracy, −6.5 on TextVQA at 50%.
- **Latency**: lens probe is free (336 ms prefill @ 59% FLOPs vs fixed E20 343 ms @ 64%). The *old* cf probe cost
  +70 ms (recomputed full K/V); engine v2 fixes this by reading the layer's cached K/V.
- **InfoScore (layer-level) is a clean diagnostic but does not improve pruning**: layers 0–1 are image-independent
  (InfoScore≈1), layers 2–3 most content-adaptive, yet ranking tokens with any layer gives the same accuracy; depth
  dominates. Per-image attention entropy has AUC≈0.5 for predicting pruning damage. (Keep as motivation.)
- Prior work to position against: VTW (training-free, dataset-level exit), DyVTE and AdaVSkip (per-instance but
  trained), FEATHER / Wen et al. (critiques of attention-based ranking). Name collision: "InfoPrune" is taken
  (arXiv 2511.19518) — **the method needs a new name.**

## 4. Repository layout

```
HANDOFF.md                      this file
CLAUDE.md                       conventions for Claude Code sessions
docs/
  results-and-execution-plan.md all numbers, findings, decisions, problem log (sections 0–12; 8A–8E are the exit work)
  research-and-literature.md    definitions (K, R, FLOPs, H_img, H_data, InfoScore), literature, gap table, draft related work
  archive-InfoPrune-project-overview-2026-09-28.md   early overview, superseded
notebooks/
  infoprune-phase1-consolidated.ipynb   Phase 1 (FastV fork, 8-bit, eager): InfoScore calibration, InfoPrune, K×R sweep
  infoprune-offline-analysis.ipynb      CPU analysis of Phase 1 JSONL (McNemar, oracle, AUC, router, InfoScore curves)
  phaseA-exit-profiling.ipynb           Phase A: per-example oracle exit layers + signals (POPE, GQA)
  phaseA-exit-profiling-v2.ipynb        + TextVQA, MME, pooled/transfer analysis (§8.5)
  phaseB-exit-engine.ipynb              Engine v1 (transformers 4.49, SDPA, DynamicCache), unit tests, latency, eval
  phaseB2-engine-v2.ipynb               Engine v2: cache-based cf probe, FastV-in-engine, MODEL_TAG llava15|llavanext
  archive/                              original Phase 0 / early Phase 1 notebooks
tools/
  notebook_builders/build_*.py          Python scripts that generate the notebooks (edit these, then regenerate)
  analysis/phaseB_sweep_tables.py       acc/agree/FLOPs/latency/McNemar tables for a phaseB results dir
  analysis/phaseB_matched_fidelity.py   fixed-E FLOPs at the same agreement as each adaptive point (+ MME by-position fix)
  analysis/phaseB_adaptivity.py         exit layer vs Phase A oracle (Spearman), exit by question type
results/                                untracked; see results/README.md for the expected layout
```

Regenerate a notebook: `cd tools/notebook_builders && python build_phaseB2.py` (writes to `../../notebooks/`).
`build_phaseB2.py` reads `build_phaseB.py` to reuse its data cells.

## 5. How the code works (engine v2, the part that matters)

- Model loaded with `AutoModelForImageTextToText`, fp16, `attn_implementation="sdpa"`, `device_map="auto"` (2×T4).
- **Merged embeddings** (text + projected image features) are captured from the library's own forward via a
  forward-pre-hook on `model.language_model` that raises to abort. This keeps the image-merge code the library's.
- **Visual span**: positions of `image_token_index` in `input_ids` (processor expands `<image>` to 576 tokens in
  transformers ≥4.47; LLaVA-NeXT: ~2k–2.9k, contiguous).
- **Layer loop**: our own prefill over `CORE.layers` with `attention_mask=None` (SDPA uses `is_causal`), explicit
  `position_ids`/`position_embeddings`, `DynamicCache`. Dropping tokens = slicing hidden states + keeping original
  position ids; later layers' caches simply hold fewer tokens; decode steps need no special masking.
- **lens exit** (pre-layer): `KL(lens_E || lens_{E-1}) <= tau` for `m` consecutive layers, `E >= e_min` → drop all
  visual tokens before layer E. lens = `lm_head(norm(h_last))`.
- **cf exit** (post-layer, v2): read layer E's cached K/V, recompute the last token's query, one attention row with and
  without visual keys, `||o_proj(Δ)|| / ||h_last||` ≤ τ → exit before layer E+1.
- **FastV mode**: same attention row at layer K−1, head-averaged, keep top (1−R) visual tokens from layer K on.
- Unit tests compare engine logits/greedy text to the stock model and the probe to a direct computation. Run them
  first after any engine change.
- FLOPs: FastV paper formula per layer, `4nd² + 2n²d + 2ndm`, summed over layers with the actual per-layer sequence
  lengths, relative to the unpruned model.

## 6. Conventions and gotchas (learned the hard way)

- **Always zip results after a Kaggle run** (`zip -rq ... results cache`); `/kaggle/working` is wiped when the session
  ends. All eval loops are resumable from JSONL by example id.
- **MME ids are not unique** (two questions per image). Compute agreement by record position, not by id.
- **TextVQA**: first-token proxy for "same answer" is unreliable (82%); use full-answer agreement (Phase B does).
- **fp16 entropy**: `eps=1e-8` underflows to 0 → NaN. Use fp32 / `torch.special.entr`.
- **Calibration must use images disjoint from evaluation** and benchmark-style prompts.
- **Compare at matched FLOPs or matched fidelity**, never raw accuracy; use exact McNemar for paired comparisons.
- Standard errors: ±1.1 (POPE 900), ±2.0 (GQA 600), ±2.2 (TextVQA 500), ±1.7 (MME 660) points.
- `IMAGE_START` is 4 in the HF prompt for the FastV fork (single `<image>` token); irrelevant with the expanded
  processor (span is detected).
- FlashAttention-2 does not run on T4 (Turing); SDPA is the kernel path on Kaggle. Rent an Ampere GPU for FA2 numbers.
- The FastV fork (`transformers 4.39.0.dev0`) needs `attn_implementation="eager"` and `use_cache=False`; it is only
  used by the Phase 1 notebook now.

## 7. Open items / next steps (in order)

1. Collect `phaseB2_llava15.zip` and `phaseB2_llavanext.zip`; run `tools/analysis/*.py` on them; add §8F to the results
   doc. Questions they answer: Is the v2 cf probe overhead gone? FastV vs exit on the same engine (accuracy, FLOPs,
   latency)? Does τ=0.05 transfer to LLaVA-NeXT unchanged, and how large are the savings with ~2.9k visual tokens?
2. Decide the paper's method name and default (currently: lens exit, τ=0.05, m=1, E_min=1; cf as variant).
3. Baselines on the same engine for the main table: unpruned, fixed exit (VTW), FastV K2/R.5 and R.75, uniform-only.
   Cite DyVTE / AdaVSkip numbers as trained references.
4. Robustness: larger GQA/TextVQA subsets (1000+), a calibration protocol on a held-out image set (τ chosen there, not
   on benchmarks), batch>1 note (per-image exits break batching → report batch 1 or pad).
5. Optional: RefCOCO/localization, Ampere FA2 latency, Qwen2.5-VL port.
6. Literature re-check for 2025–26 training-free per-instance visual exit before submission (AdaVSkip, Sept 2026, is
   the closest; it is trained).
7. Write-up: motivation = InfoScore analysis + FastV sweep (depth dominates, attention-ranking quality does not);
   method = per-image exit; results = §8E tables/curves, latency, adaptivity; ablations = τ, signal, uniform stage.

## 8. Kaggle setup reminders

- Accelerator GPU T4×2, Internet on. Attach: GQA questions dataset (`testdev_balanced_questions.json`), GQA images
  dataset, and the previous results zip (for restore cells).
- Phase A notebooks install the FastV fork; Phase B/B2 notebooks pin stock `transformers==4.49.0` — use separate
  Kaggle notebooks.
- Rough costs on 2×T4: Phase A profiling ~2–4 s/example; engine eval ~0.5 s/example (LLaVA-1.5), ~2.5 s (NeXT).
