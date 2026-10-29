# results/

Not tracked in git (large). Put the Kaggle result zips here, unzipped, so the analysis scripts find them:

```
results/
  infoprune_results/      <- infoprune_results.zip   (Phase 0-1: results/*.jsonl, cache/calib_*.json)
  phaseA/                 <- phaseA_results_v2.zip   (profile_{pope,gqa,textvqa,mme}.jsonl, verify_*.jsonl)
  phaseB/                 <- phaseB_results.zip (20 Oct 2026; 17 methods x 4 benchmarks, LLaVA-1.5, 77 files)
  phaseB2/llava15/        <- phaseB2_llava15.zip     (engine v2: FastV-in-engine, cheap cf probe; 27 Oct 2026)
  phaseB2/llavanext/      <- phaseB2_llavanext.zip   (LLaVA-NeXT-7B, 300/benchmark, same tau; 27 Oct 2026)
  phaseb2-engine-v2.log   Kaggle log of the LLaVA-1.5 engine-v2 run (unit tests, latency table)
  data/, llavanext/data/  stray dataset checkouts (POPE repo, COCO subset) — not needed by the scripts
```

The zips unpack to these paths directly (`unzip -o phaseB2_llava15.zip` → `phaseB2/llava15/`). Then e.g.
`python -I tools/analysis/phaseB_sweep_tables.py results/phaseB2/llava15` (also `phaseB_matched_fidelity.py`,
`phaseB_latency_overhead.py`, `phaseB_calibration.py results/phaseB`; `phaseB_adaptivity.py <phaseB dir> results/phaseA`).
