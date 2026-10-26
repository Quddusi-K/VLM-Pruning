# results/

Not tracked in git (large). Put the Kaggle result zips here, unzipped, so the analysis scripts find them:

```
results/
  infoprune_results/      <- infoprune_results.zip   (Phase 0-1: results/*.jsonl, cache/calib_*.json)
  phaseA/                 <- phaseA_results_v2.zip   (profile_{pope,gqa,textvqa,mme}.jsonl, verify_*.jsonl)
  phaseB/                 <- phaseB_results.zip (20 Oct 2026; 20 methods x 4 benchmarks, LLaVA-1.5)
  phaseB2/llava15/        <- phaseB2_llava15.zip     (engine v2: FastV-in-engine, cheap cf probe)
  phaseB2/llavanext/      <- phaseB2_llavanext.zip
```

Then e.g. `python -I tools/analysis/phaseB_sweep_tables.py results/phaseB/phaseB`.
