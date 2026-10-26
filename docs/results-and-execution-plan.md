# Visual Token Pruning in VLMs: Results, Findings and Execution Plan

*All experimental results to date, what they mean, the problems we hit, and the execution plan for the method paper.*

*Companion file: `research-and-literature.md`, which holds the concepts, definitions and literature.*

*Last updated: 20 October 2026. Every number below was recomputed directly from the result files in `infoprune_results.zip` (Phases 0–1), `phaseA_results_v2.zip` (Phase A, 4 benchmarks) and `phaseB_results.zip` (Phase B).*

---

## 0. Status at a glance

| Item | Status |
|---|---|
| Phase 0: reproduce FastV and SparseVLM on POPE | ✅ done |
| Phase 1: InfoScore ported to LLaVA's decoder; calibration; InfoPrune (argmax and threshold) | ✅ done |
| Per-image adaptive layer selection (the original hypothesis) | ❌ **not supported** (AUC ≈ 0.5 at R = 0.5, 0.75, 0.9) |
| Label-free choice of K via layer-level InfoScore | ❌ **not supported** (predicted K=3/4 is no better than K=2) |
| FastV K × R sweep (K ∈ {1,2,3,4,7,10,17}, R ∈ {0.5, 0.75}; K ∈ {2,17} at R=0.9) | ✅ done |
| Random-ranking control | ⏸ deferred: Wen et al. 2025 already show random ≈ FastV; it returns as a Phase D baseline |
| **Phase A: oracle exit layers + exit-signal profiling (POPE 900, GQA 600)** | ✅ done — **PASS in the one-threshold-for-all-tasks setting** (§8A) |
| Phase A v2: TextVQA (500) + MME (660) profiling, pooled analysis | ✅ done (§8C) |
| Phase B: SDPA + KV-cache engine with per-image exit | ✅ engine verified (logit diff 0.008 vs stock); POPE/GQA evaluated with real latency (§8D) |
| Phase B sweep: τ sweep + fixed-E curve + uniform ablation on 4 benchmarks | ✅ done (§8E): lens τ≤0.1 has no significant accuracy loss anywhere; beats fixed exit on GQA/TextVQA, ties on POPE/MME |
| Engine v2 (cache-based cf probe, FastV-in-engine) + LLaVA-NeXT-7B | ⏳ next |

---

## 1. Experimental setup (state this exactly in the paper)

- **Model:** `llava-hf/llava-1.5-7b-hf`, 8-bit (bitsandbytes), `attn_implementation="eager"`, `device_map="auto"` across 2× T4 (16 GB each) on Kaggle.
- **Code base:** FastV's patched `transformers 4.39.0.dev0`, plus a multi-GPU device fix to `fastv_forward`. InfoPrune is monkeypatched in through a dispatcher on `LlamaModel.fastv_forward`. The model is loaded once, and methods are switched by setting `model.config.fastv_config`.
- **Decoding:** greedy, `use_cache=False`, so every generated token reruns prefill with pruning. POPE uses `max_new_tokens=3` and GQA uses 10.
- **Image token index:** `IMAGE_START=4`, `IMAGE_LEN=576`. Phase 0 used 5, which is off by one with negligible effect.
- **POPE eval subset:** 150 questions per split (random, popular, adversarial), 450 in total over 73 COCO val2014 images, seed 0, the same sampling as Phase 0. Prompt: `USER: <image>\n{question}\nASSISTANT:`.
- **GQA eval subset:** 300 testdev-balanced questions over 209 images (shuffle seed 0). Prompt suffix: `\nAnswer the question using a single word or phrase.` Scoring: exact match after lowercasing and stripping punctuation and articles. A lenient contains-match is also reported; it is almost identical.
- **Calibration sets:** disjoint from evaluation images.
  - POPE: 150 images not in the eval subset, each with one POPE-style question (seed 1).
  - GQA: 150 testdev images not in the eval subset, with their questions and the GQA suffix.
- **FLOPs:** relative prefill FLOPs using FastV's per-layer formula (see the literature file, §3.2).
- **Statistics:** exact two-sided McNemar test for paired comparisons; bootstrap 95% CI for AUC (2000 resamples). Standard error of accuracy is about ±1.5 points on POPE (n=450) and about ±2.8 on GQA (n=300).

---

## 2. Phase 0: baseline reproduction

### 2.1 FastV on POPE (450 questions, `IMAGE_START=5`)
| Setting | Acc | F1 | Yes % |
|---|---|---|---|
| Baseline | 88.7 | 88.7 | 50.2 |
| FastV K=2, R=0.5 | 87.6 | 87.0 | 46.0 |
| FastV K=2, R=0.75 | 83.6 | 81.2 | 37.6 |
| FastV K=3, R=0.5 | 88.0 | 87.6 | 46.4 |

The single-image smoke test gave "Two cats sleeping on a couch." for both baseline and FastV K=2, R=0.5.

### 2.2 SparseVLM on POPE
These numbers come from the original LLaVA codebase, the `liuhaotian/llava-v1.5-7b` checkpoint, SDPA, 8-bit and `use_cache=True`, with 192 tokens kept. They are **not directly comparable** to the HF rows.

| Setting | Acc | F1 | Yes % |
|---|---|---|---|
| Baseline | 87.8 | 88.2 | 53.3 |
| SparseVLM | 87.6 | 87.8 | 52.2 |

### 2.3 MME (pipeline check only)
200 shuffled examples covering only two categories: code reasoning (40) and artwork (160).

| Setting | code_reasoning | artwork | Overall |
|---|---|---|---|
| Baseline | 57.5 | 75.6 | 72.0 |
| FastV K=2, R=0.5 | 55.0 | 75.6 | 71.5 |

---

## 3. Layer-level InfoScore (full depth, 150 calibration images each)

| Layer | POPE H_data | POPE mean H_img | **POPE InfoScore** | GQA H_data | GQA mean H_img | **GQA InfoScore** |
|---|---|---|---|---|---|---|
| 0 | 6.303 | 6.290 | **1.002** | 6.311 | 6.298 | **1.002** |
| 1 | 6.272 | 6.044 | **1.038** | 6.286 | 6.050 | **1.039** |
| 2 | 5.905 | 3.129 | **1.887** ← peak | 5.946 | 3.230 | 1.841 |
| 3 | 5.926 | 3.218 | 1.841 | 5.927 | 3.116 | **1.902** ← peak |
| 4 | 6.005 | 3.780 | 1.589 | 6.002 | 3.658 | 1.641 |
| 5 | 6.121 | 4.474 | 1.368 | 6.119 | 4.277 | 1.431 |
| 6 | 6.179 | 4.512 | 1.369 | 6.175 | 4.343 | 1.422 |
| 7 | 6.207 | 4.735 | 1.311 | 6.220 | 4.678 | 1.330 |
| 8 | 6.219 | 5.042 | 1.233 | 6.235 | 4.968 | 1.255 |
| 10 | 6.199 | 5.138 | 1.207 | 6.227 | 5.085 | 1.225 |
| 12 | 6.173 | 5.058 | 1.220 | 6.225 | 5.016 | 1.241 |
| 16 | 6.263 | 5.044 | 1.241 | 6.261 | 4.967 | 1.261 |
| 18 | 6.243 | 4.840 | 1.290 | 6.241 | 4.732 | 1.319 |
| 23 | 6.235 | 4.737 | 1.316 | 6.229 | 4.582 | 1.360 |
| 29 | 6.264 | 5.451 | 1.149 | 6.267 | 5.474 | 1.145 |
| 31 | 6.245 | 5.393 | 1.158 | 6.237 | 5.237 | 1.191 |

The maximum possible entropy is ln 576 = 6.356. Full per-layer values are in `cache/calib_{pope,gqa}_L0-31_n150.json`.

**Reading the table:**
- **Layers 0–1:** attention is near-uniform (H_img ≈ 6.0–6.3) and identical across images (InfoScore ≈ 1.0). Their attention follows position, not content.
- **Layers 2–3:** attention is sharply concentrated (about 22 effective tokens) and content-dependent. These are the peak InfoScore layers.
- **From layer 8 on:** attention is diffuse (about 150 effective tokens), with InfoScore around 1.2–1.3.
- POPE and GQA calibrations give almost identical curves.
- **Caveat:** H_data sits near its maximum at every layer from 2 on (5.9–6.3), so the ratio is driven mostly by 1/H_img.

---

## 4. Phase 1: InfoPrune results (R = 0.5)

Two modes:
- **Argmax:** score layers 2–16 per image, rank tokens with the best layer's attention, prune after layer 16 (the same compute as FastV K=17).
- **Threshold:** prune at the first layer with focus score ≥ τ. τ = 0.50 was the lowest value on the grid; the target mean K of 4 was unreachable with the floor at layer 2.

### 4.1 Paired comparisons (McNemar)
| Benchmark | A | B | Acc A | Acc B | Only A correct | Only B correct | p |
|---|---|---|---|---|---|---|---|
| POPE | InfoPrune argmax | FastV K=17 | 88.4 | 88.4 | 0 | 0 | 1.000 (identical predictions) |
| POPE | InfoPrune threshold | FastV K=7 (matched FLOPs) | 87.8 | 89.1 | 2 | 8 | 0.109 |
| POPE | InfoPrune threshold | FastV K=2 | 87.8 | 87.6 | 7 | 6 | 1.000 |
| GQA | InfoPrune argmax | FastV K=17 | 60.7 | 62.0 | 0 | 4 | 0.125 |
| GQA | InfoPrune threshold | FastV K=7 | 59.7 | 58.0 | 11 | 6 | 0.332 |
| GQA | InfoPrune threshold | FastV K=2 | 59.7 | 59.0 | 7 | 5 | 0.774 |

### 4.2 Chosen-layer distributions
- **POPE argmax:** {2:44, 3:59, 4:27, 5:33, 6:41, 7:16, 8:10, 9:9, 10:32, 11:35, 12:12, 13:13, 14:42, 15:25, 16:52}
- **POPE threshold:** {2:228, 3:37, 4:28, 5:5, 6:15, 7:11, 8:5, 9:10, 10:21, 11:29, 12:7, 13:9, 14:14, 15:8, 16:23}. Mean K\* is about 7.4.
- **GQA argmax:** {2:56, 3:33, 4:25, 5:36, 6:15, 7:15, 8:12, 9:6, 10:20, 11:13, 12:10, 13:14, 14:21, 15:4, 16:20}
- **GQA threshold:** {2:169, 3:32, 4:13, 5:21, 6:11, 7:6, 8:6, 9:3, 10:15, 11:3, 12:6, 13:2, 14:9, 16:4}. Mean K\* is about 6.1.

### 4.3 Is the chosen layer an image property?
- **Share of chosen-layer variance between images (POPE):** argmax 0.63, threshold 0.79, against 0.16 for a shuffled baseline (p < 0.001). POPE uses the same question template for every image, so the choice is a **stable property of the image**.
- **Across POPE splits (Kruskal–Wallis):** argmax p = 0.001, threshold p = 0.012. But the splits use different images, so this is confounded with image identity.
- **GQA question types:** threshold-mode mean chosen layer ranges from 3.2 (compare) to 4.4 (verify).

### 4.4 Smoke test (consolidated notebook)
- **Answers on the two-cats image:**
  - Baseline, FastV K=2, and argmax: "Two cats sleeping on a couch."
  - Threshold (pruned after layer 2): "A cat and a remote control."
- **Max logit difference vs baseline:** FastV K=2 2.70, argmax 1.07, threshold 2.83.
- **Threshold-mode chosen layers on five demo images:** 2, 2, 2, 3, 2.

---

## 5. FastV K × R sweep: full results

FLOPs are relative prefill FLOPs. Agreement is the share of predictions identical to the baseline's.

### 5.1 POPE: all runs (n=450)

| Run | Acc | F1 | Yes % | Agree w/ baseline | Prefill FLOPs |
|---|---|---|---|---|---|
| `baseline` | 88.7 | 88.7 | 50.2 | 100.0 | 100.0% |
| `fastv_K1_R0.5` | 86.7 | 85.8 | 44.2 | 91.8 | 52.4% |
| `fastv_K2_R0.5` | 87.6 | 87.0 | 46.0 | 94.4 | 53.9% |
| `fastv_K3_R0.5` | 88.2 | 87.8 | 46.7 | 94.7 | 55.4% |
| `fastv_K4_R0.5` | 86.2 | 85.4 | 44.7 | 93.6 | 57.0% |
| `fastv_K7_R0.5` | 89.1 | 88.8 | 47.6 | 96.0 | 61.6% |
| `fastv_K10_R0.5` | 88.4 | 88.3 | 49.1 | 98.0 | 66.2% |
| `fastv_K17_R0.5` | 88.4 | 88.5 | 50.4 | 99.8 | 76.9% |
| `fastv_K1_R0.75` | 80.0 | 75.8 | 32.7 | 81.6 | 29.0% |
| `fastv_K2_R0.75` | 83.3 | 80.9 | 37.3 | 87.1 | 31.3% |
| `fastv_K3_R0.75` | 83.6 | 82.0 | 41.1 | 88.7 | 33.6% |
| `fastv_K4_R0.75` | 83.6 | 81.5 | 38.9 | 86.4 | 35.9% |
| `fastv_K7_R0.75` | 86.0 | 85.1 | 44.0 | 92.4 | 42.8% |
| `fastv_K10_R0.75` | 87.1 | 86.7 | 46.9 | 94.4 | 49.6% |
| `fastv_K17_R0.75` | 88.4 | 88.5 | 50.4 | 99.8 | 65.7% |
| `fastv_K2_R0.9` | 74.4 | 66.7 | 26.7 | 74.2 | 18.0% |
| `fastv_K17_R0.9` | 88.4 | 88.5 | 50.4 | 99.8 | 59.0% |
| `infoprune_argmax_L2-16_R0.5` | 88.4 | 88.5 | 50.4 | 99.8 | 76.9% |
| `infoprune_thr0.50_L2-16_R0.5` | 87.8 | 87.5 | 47.6 | 96.0 | 62.2% |

### 5.2 GQA testdev-balanced: all runs (n=300)

| Run | Exact | Lenient | Agree w/ baseline | Prefill FLOPs |
|---|---|---|---|---|
| `baseline` | 61.7 | 61.7 | 100.0 | 100.0% |
| `fastv_K1_R0.5` | 58.3 | 58.7 | 86.0 | 53.3% |
| `fastv_K2_R0.5` | 59.0 | 59.0 | 87.3 | 54.8% |
| `fastv_K3_R0.5` | 57.3 | 57.3 | 88.3 | 56.3% |
| `fastv_K4_R0.5` | 56.7 | 57.0 | 83.3 | 57.8% |
| `fastv_K7_R0.5` | 58.0 | 58.3 | 88.3 | 62.4% |
| `fastv_K10_R0.5` | 62.0 | 62.0 | 94.7 | 66.9% |
| `fastv_K17_R0.5` | 62.0 | 62.0 | 98.3 | 77.4% |
| `fastv_K1_R0.75` | 49.7 | 49.7 | 71.7 | 30.5% |
| `fastv_K2_R0.75` | 52.3 | 52.3 | 77.0 | 32.7% |
| `fastv_K3_R0.75` | 52.7 | 52.7 | 78.7 | 35.0% |
| `fastv_K4_R0.75` | 52.7 | 52.7 | 75.7 | 37.2% |
| `fastv_K7_R0.75` | 56.0 | 56.3 | 81.0 | 43.9% |
| `fastv_K10_R0.75` | 58.0 | 58.0 | 87.3 | 50.7% |
| `fastv_K17_R0.75` | 61.0 | 61.0 | 97.3 | 66.4% |
| `fastv_K2_R0.9` | 50.7 | 50.7 | 69.7 | 19.7% |
| `fastv_K17_R0.9` | 59.0 | 59.0 | 94.3 | 59.8% |
| `infoprune_argmax_L2-16_R0.5` | 60.7 | 60.7 | 96.0 | 77.4% |
| `infoprune_thr0.50_L2-16_R0.5` | 59.7 | 59.7 | 90.7 | 61.1% |

### 5.3 POPE accuracy per split

| Run | random | popular | adversarial |
|---|---|---|---|
| `baseline` | 93.3 | 90.0 | 82.7 |
| `fastv_K1_R0.5` | 92.7 | 85.3 | 82.0 |
| `fastv_K2_R0.5` | 93.3 | 88.0 | 81.3 |
| `fastv_K3_R0.5` | 92.7 | 87.3 | 84.7 |
| `fastv_K4_R0.5` | 91.3 | 86.0 | 81.3 |
| `fastv_K7_R0.5` | 95.3 | 88.0 | 84.0 |
| `fastv_K10_R0.5` | 93.3 | 90.0 | 82.0 |
| `fastv_K17_R0.5` | 93.3 | 89.3 | 82.7 |
| `fastv_K1_R0.75` | 83.3 | 79.3 | 77.3 |
| `fastv_K2_R0.75` | 85.3 | 83.3 | 81.3 |
| `fastv_K3_R0.75` | 90.0 | 81.3 | 79.3 |
| `fastv_K4_R0.75` | 88.7 | 82.0 | 80.0 |
| `fastv_K7_R0.75` | 90.7 | 86.7 | 80.7 |
| `fastv_K10_R0.75` | 92.0 | 88.0 | 81.3 |
| `fastv_K17_R0.75` | 93.3 | 89.3 | 82.7 |
| `fastv_K2_R0.9` | 79.3 | 76.0 | 68.0 |
| `fastv_K17_R0.9` | 93.3 | 89.3 | 82.7 |
| `infoprune_argmax_L2-16_R0.5` | 93.3 | 89.3 | 82.7 |
| `infoprune_thr0.50_L2-16_R0.5` | 92.7 | 88.0 | 82.7 |

### 5.4 POPE accuracy: K × R grid (accuracy % / prefill FLOPs %)

| K | R=0.5 | R=0.75 | R=0.9 |
|---|---|---|---|
| 1 | 86.7 / 52.4 | 80.0 / 29.0 | — |
| 2 | 87.6 / 53.9 | 83.3 / 31.3 | 74.4 / 18.0 |
| 3 | 88.2 / 55.4 | 83.6 / 33.6 | — |
| 4 | 86.2 / 57.0 | 83.6 / 35.9 | — |
| 7 | 89.1 / 61.6 | 86.0 / 42.8 | — |
| 10 | 88.4 / 66.2 | 87.1 / 49.6 | — |
| 17 | 88.4 / 76.9 | 88.4 / 65.7 | 88.4 / 59.0 |

### 5.5 GQA exact match: K × R grid (accuracy % / prefill FLOPs %)

| K | R=0.5 | R=0.75 | R=0.9 |
|---|---|---|---|
| 1 | 58.3 / 53.3 | 49.7 / 30.5 | — |
| 2 | 59.0 / 54.8 | 52.3 / 32.7 | 50.7 / 19.7 |
| 3 | 57.3 / 56.3 | 52.7 / 35.0 | — |
| 4 | 56.7 / 57.8 | 52.7 / 37.2 | — |
| 7 | 58.0 / 62.4 | 56.0 / 43.9 | — |
| 10 | 62.0 / 66.9 | 58.0 / 50.7 | — |
| 17 | 62.0 / 77.4 | 61.0 / 66.4 | 59.0 / 59.8 |

---

## 6. Statistical analyses

### 6.1 Selected paired tests across the sweep (McNemar)
| Benchmark | A | B | Acc A | Acc B | Only A | Only B | p |
|---|---|---|---|---|---|---|---|
| POPE | K=2 R=0.5 | baseline | 87.6 | 88.7 | 10 | 15 | 0.424 |
| POPE | K=3 R=0.5 | K=2 R=0.5 | 88.2 | 87.6 | 9 | 6 | 0.607 |
| POPE | K=1 R=0.5 | K=2 R=0.5 | 86.7 | 87.6 | 9 | 13 | 0.523 |
| POPE | K=2 R=0.75 | baseline | 83.3 | 88.7 | 17 | 41 | **0.002** |
| POPE | K=3 R=0.75 | K=2 R=0.75 | 83.6 | 83.3 | 15 | 14 | 1.000 |
| POPE | K=1 R=0.75 | K=2 R=0.75 | 80.0 | 83.3 | 16 | 31 | **0.040** |
| POPE | K=2 R=0.9 | K=17 R=0.9 | 74.4 | 88.4 | 26 | 89 | **<0.001** |
| GQA | K=2 R=0.5 | K=17 R=0.5 | 59.0 | 62.0 | 6 | 15 | 0.078 |
| GQA | K=3 R=0.5 | K=2 R=0.5 | 57.3 | 59.0 | 4 | 9 | 0.267 |
| GQA | K=2 R=0.75 | baseline | 52.3 | 61.7 | 5 | 33 | **<0.001** |
| GQA | K=3 R=0.75 | K=2 R=0.75 | 52.7 | 52.3 | 9 | 8 | 1.000 |
| GQA | K=1 R=0.75 | K=2 R=0.75 | 49.7 | 52.3 | 16 | 24 | 0.268 |
| GQA | K=2 R=0.9 | K=17 R=0.9 | 50.7 | 59.0 | 17 | 42 | **0.002** |
| GQA | K=17 R=0.9 | baseline | 59.0 | 61.7 | 0 | 8 | **0.008** |

### 6.2 Headroom: fidelity oracle
For each example, take the cheapest available K that reproduces the baseline answer.

| Benchmark, R | Oracle mean FLOPs | FastV K=2 FLOPs | Examples needing K>2 |
|---|---|---|---|
| POPE R=0.5 | 53.0% | 53.9% | 20 / 450 |
| POPE R=0.75 | 31.3% | 31.3% | 47 / 450 |
| POPE R=0.9 (only K=2, 17 available) | 28.7% | 18.0% | 116 / 450 |
| GQA R=0.5 | 54.5% | 54.8% | 25 / 300 |
| GQA R=0.75 | 34.3% | 32.7% | 52 / 300 |
| GQA R=0.9 (only K=2, 17 available) | 33.5% | 19.7% | 91 / 300 |

**Reading:** at R=0.9, a perfect per-example router reproduces every baseline answer at about 29% (POPE) and 34% (GQA) of the FLOPs, versus about 59–60% for K=17. That is real headroom, if a signal can find the hard examples.

### 6.3 Can the per-image score predict which examples K=2 breaks? (AUC)
The target is "FastV K=2 changes the baseline answer". The feature is the negative focus score from the argmax run, whose scores are computed before pruning and so don't depend on R. Layer 2 was pre-registered; the other layers are exploratory.

| Benchmark, R | Answers changed by K=2 | AUC, layer 2 [95% CI] | AUC range over layers 2–16 |
|---|---|---|---|
| POPE 0.5 | 25 | 0.476 [0.36, 0.59] | 0.445 – 0.629 |
| POPE 0.75 | 58 | 0.452 [0.38, 0.53] | 0.402 – 0.496 |
| POPE 0.9 | 116 | 0.472 [0.41, 0.53] | 0.376 – 0.501 |
| GQA 0.5 | 38 | 0.483 [0.39, 0.57] | 0.467 – 0.554 |
| GQA 0.75 | 69 | 0.532 [0.46, 0.60] | 0.460 – 0.532 |
| GQA 0.9 | 91 | 0.497 [0.43, 0.56] | 0.465 – 0.536 |

**Side observation (exploratory, untested):** on POPE at R ≥ 0.75, AUCs sit systematically *below* 0.5 (as low as 0.38). Examples with *more focused* attention break more often under heavy pruning. This is worth a look in the analysis paper, but it has not been confirmed.

### 6.4 Offline router: K=2 vs K=17 by the layer-2 score, against random routing at the same rate
The table gives the range of the router's accuracy gain over random routing across 7 budgets.

| Benchmark, R | Gain range (points) |
|---|---|
| POPE 0.5 | −0.7 to +0.4 |
| POPE 0.75 | −1.0 to +0.5 |
| POPE 0.9 | −0.3 to +0.7 |
| GQA 0.5 | −0.6 to +0.9 |
| GQA 0.75 | −0.5 to +2.0 (+2.0 is 6 questions) |
| GQA 0.9 | −1.0 to +0.8 |

There is no consistent gain. The per-image score has no usable signal.

### 6.5 Accuracy-vs-FLOPs Pareto frontier (all runs)
Points marked * are Pareto-optimal.

**POPE:**
- \* K2 R0.9 18.0% → 74.4
- \* K1 R0.75 29.0% → 80.0
- \* K2 R0.75 31.3% → 83.3
- \* K3 R0.75 33.6% → 83.6
- \* K7 R0.75 42.8% → 86.0
- \* K10 R0.75 49.6% → 87.1
- \* K2 R0.5 53.9% → 87.6
- \* K3 R0.5 55.4% → 88.2
- \* K17 R0.9 59.0% → 88.4
- \* K7 R0.5 61.6% → 89.1
- Dominated: InfoPrune threshold (62.2% → 87.8) and InfoPrune argmax (76.9% → 88.4).

**GQA:**
- \* K2 R0.9 19.7% → 50.7
- \* K2 R0.75 32.7% → 52.3
- \* K3 R0.75 35.0% → 52.7
- \* K7 R0.75 43.9% → 56.0
- \* K10 R0.75 50.7% → 58.0
- \* K1 R0.5 53.3% → 58.3
- \* K2 R0.5 54.8% → 59.0
- \* InfoPrune threshold 61.1% → 59.7 (on the frontier, but within noise)
- \* K17 R0.75 66.4% → 61.0
- \* K10 R0.5 66.9% → 62.0

**Late and aggressive vs early and mild.** K=10 at R=0.75 roughly matches FastV's default (K=2, R=0.5) at less compute:
- POPE: 87.1% at 49.6% FLOPs vs 87.6% at 53.9%.
- GQA: 58.0% at 50.7% vs 59.0% at 54.8%.

---

## 7. Pending: random-ranking control (code ready, not run)

**Purpose:** if random token selection at the same K and R matches FastV, the attention ranking adds nothing at these settings. That closes the analysis story.

**Edit in the InfoPrune module cell.** Replace:
```python
        top = vecs[chosen].to(dev).topk(round(img_n * (1 - r))).indices + img_s
```
with:
```python
        rank_vec = vecs[chosen]
        if cfg.get("rank") == "random":   # control: same prune point and budget, random tokens (fixed per image)
            seed = int(inputs_embeds[0, img_s:img_s + img_n].float().abs().sum().item() * 1e3) % (2**31)
            rank_vec = torch.rand(img_n, generator=torch.Generator().manual_seed(seed))
        top = rank_vec.to(dev).topk(round(img_n * (1 - r))).indices + img_s
```

**New cell after the evaluation sections:**
```python
def random_cfg(calib, k, r):
    cfg = infoprune_cfg(calib, "threshold", r, threshold=0.0)    # threshold 0 = always prune at the first window layer
    cfg.update(layer_floor=k - 1, layer_ceiling=k - 1, rank="random")
    return cfg

CAL = {"pope": calibrate(pope_calib, "pope", layer_floor=0, layer_ceiling=31),
       "gqa":  calibrate(gqa_calib, "gqa", layer_floor=0, layer_ceiling=31, suffix=GQA_SUFFIX)}
BENCH = {"pope": (pope_eval, "", 3, pope_score), "gqa": (gqa_eval, GQA_SUFFIX, 10, gqa_score)}
for r, k in [(0.75, 2), (0.75, 3), (0.9, 2), (0.5, 2)]:
    for bench, (items, suffix, mnt, score) in BENCH.items():
        run_method(bench, f"random_K{k}_R{r}", random_cfg(CAL[bench], k, r), items, mnt, suffix, score)
```
Compare each `random_K{k}_R{r}` run against `fastv_K{k}_R{r}` with McNemar.

---

## 8. Key findings (paper-ready statements)

1. **Per-image attention entropy does not predict pruning damage.** AUC is 0.45–0.53 at layer 2, with every CI including 0.5, at R = 0.5, 0.75 and 0.9 on POPE and GQA. Routing by the score never beats random routing reliably.
2. **The chosen layer is nonetheless a stable image property.** 63–79% of the variance is between images, against 16% when shuffled. The score measures something consistent about each image that is unrelated to pruning damage.
3. **Layer-level InfoScore cleanly characterizes the decoder.** Layers 0–1 are image-independent (InfoScore ≈ 1.00–1.04, near-uniform attention). Layers 2–3 are the most content-adaptive (≈ 1.9). Deeper layers are diffuse (≈ 1.2–1.3).
4. **The most reliable layer does not prune better.** InfoScore-predicted K=3/4 ties FastV's K=2, whose ranking uses image-independent layer-1 attention: K3 vs K2 at R=0.75 gives 15/14 discordant answers on POPE and 9/8 on GQA, p = 1.0. Only K=1 (layer 0) is measurably worse (POPE R=0.75, p = 0.04).
5. **Depth dominates.** At R ≥ 0.75, accuracy rises steadily with K and is highest at the latest K tested (17). There is no interior optimum.
6. **POPE doesn't use image tokens after layer 16.** K=17 predictions are *identical* at R = 0.5, 0.75 and 0.9. GQA still loses a little (K=17: 62.0 → 61.0 → 59.0).
7. **There is real headroom for per-example adaptivity under heavy pruning.** At R=0.9, an oracle reproduces all baseline answers at about 29% (POPE) and 34% (GQA) of FLOPs, versus about 59–60% for K=17. A useful signal has to come from somewhere other than attention entropy.
8. **Late and aggressive pruning roughly matches early and mild pruning at less compute** (K=10, R=0.75 vs K=2, R=0.5).
9. **Consistency with the literature:** findings 3–5 agree with Feather the Throttle (positional bias in early attention; ranking criterion matters little on standard benchmarks) and with Wen et al. 2025 (random selection is competitive).

---

## 8A. Phase A results: per-image visual-token exit (oracle + signals)

**Setup.** `phaseA-exit-profiling.ipynb`. LLaVA-1.5-7B in **fp16** (not 8-bit), eager attention, 2×T4. One prefill per
example computes (i) the exact oracle: for every exit layer E = 0…31 (layers 0…E−1 see the image, layers E…31 see text
only) whether the model's **first answer token** is unchanged; (ii) candidate exit signals at every layer, computed
before layer E from 1–30 query rows without the attention matrix. Subsets: POPE 900 (300/split, superset of Phase 1),
GQA 600 testdev-balanced (superset of Phase 1). Proxy check: full generated answers at the first-token oracle exit
matched the full model on **47/47 POPE and 57/57 GQA** verifiable examples. POPE first-token accuracy 88.6% (matches
Phase 1). Thresholds are always chosen on half the data and evaluated on the other half (2-fold × 20 random splits).

### 8A.1 How deep the image is needed
| | POPE (900) | GQA (600) |
|---|---|---|
| Answer unchanged with the image removed at **every** layer (E=0) | 51.9%* | 44.8%* |
| Answer unchanged with the image removed at E=0 **and** at all later layers (stable) | 10.1% (91/900) | 28.3% (170/600) |
| Oracle earliest-stable exit layer: median / mean | 7 / 7.0 | 11 / 10.8 |
| Oracle FLOPs (earliest-stable exit) | 24.2% | 37.0% |
| Smallest fixed exit layer with ≥98% fidelity | **E=15 → 48.5% FLOPs** | **E=30 → 94.1% FLOPs** |
| Smallest fixed exit layer with ≥95% fidelity | E=12 → 39.4% | E=25 → 79.2% |

\*E=0 "unchanged" is inflated by yes/no priors: half of POPE answers come out the same without any image at all, but
most of those flip again at some later exit layer. Exit safety is **not monotone in E** for a given question, so
exit rules require the signal to persist for *m* layers.

Fixed-exit fidelity curve (fidelity % / FLOPs %):
- POPE: E8 68.6/27 · E10 81.4/33 · **E12 97.3/39** · E14 97.6/45 · E16 99.4/52 · E20 100/64
- GQA: E12 63.0/41 · E16 74.7/53 · E20 87.3/64 · E24 93.3/76 · E28 96.2/88 · E30 98.0/94

### 8A.2 Which signals carry information (AUC of "safe to exit at E", 0.5 = chance)
| Signal | POPE E8 | POPE E11 | POPE E14 | GQA E11 | GQA E17 | GQA E20 | GQA E26 |
|---|---|---|---|---|---|---|---|
| cf_last_res (counterfactual, last token) | 0.48 | 0.57 | 0.74 | 0.69 | 0.79 | **0.82** | 0.78 |
| cf_text_res (counterfactual, question tokens) | 0.62 | 0.72 | 0.72 | 0.51 | 0.69 | 0.69 | 0.65 |
| mass_last (visual attention share, last token) | 0.49 | 0.53 | 0.72 | 0.65 | 0.79 | **0.82** | 0.78 |
| lens_kl_prev (logit-lens KL to previous layer) | 0.37 | 0.50 | 0.41 | 0.35 | 0.68 | 0.69 | 0.64 |
| lens_p1 (logit-lens top-1 prob) | 0.38 | 0.50 | 0.53 | 0.70 | 0.77 | 0.78 | 0.46 |

Signals are at chance before layer ~10 and informative from layer ~11 on (POPE has too few unsafe examples past E=14
to measure). **Per-image attention entropy is not among them** (ruled out in Phase 1).

### 8A.3 Per-benchmark calibration (held-out) — matched fidelity, FLOPs
| Target fidelity | POPE: fixed E | POPE: best signal | GQA: fixed E | GQA: best signal |
|---|---|---|---|---|
| 99% | 48.9% | 46.3% (cf_text_res) | 96.2% | **83.4%** (cf_last_res) |
| 98% | 46.5% | 42.7% (cf_text_res) | 93.2% | **78.6%** (lens_kl_prev) |
| 95% | 39.4% | 38.4% (cf_text_res) | 79.8% | **70.5%** (lens_p1) |

Shuffled control (same exit-layer distribution, random assignment) at 98%: GQA 98.7% → **94.0%**, POPE 98.6% → 97.8%.
→ **GQA passes clearly (−14.6 pt FLOPs); POPE only marginally (−3.8 pt)**, because a fixed layer already works on POPE
(almost every question is settled by layer 15). Combining signals (cf AND lens) added nothing.

### 8A.4 One global threshold for all tasks (the deployment setting) — **headline result**
Calibrated on the pooled POPE+GQA data (half), evaluated per benchmark on the other half, 2-fold × 10. A single
fixed exit layer must be set for the hardest task and therefore wastes compute on easy ones; the per-image signals
adapt automatically.

| Method (one setting for both tasks) | POPE fidelity / FLOPs | GQA fidelity / FLOPs |
|---|---|---|
| Fixed exit layer (VTW-style) | 100.0% / 79.7% | 94.9% / 80.1% |
| **lens_kl_prev** | 99.7% / **57.8%** | **96.0%** / **70.2%** |
| cf_last_res | 99.8% / 58.5% | 95.3% / 73.9% |
| lens_p1 | 99.7% / 59.9% | 95.1% / 72.5% |
| mass_last | 99.8% / 82.9% | 96.2% / 87.2% |

Per-image exit saves **~22 pt FLOPs on POPE and ~10 pt on GQA** at equal-or-better fidelity than the best single fixed
layer. This matches DyVTE's finding that the right exit point is task- and instance-dependent, achieved here with
no training.

Pooled calibration on **all** 1500 profiles (deployment values used by Phase B): `lens_kl_prev` τ=0.0916, m=1, E_min=1
→ 98.1% fidelity @ 62.2% FLOPs; `cf_last_res` τ=0.0639, m=3, E_min=12 → 98.0% @ 61.6%; fixed layer for the same
pooled fidelity: **E=25 @ 79.0%**.

### 8A.5 Cross-benchmark transfer (calibrate on one, apply to the other, target 98%)
| Calibrated on → applied to | Fixed E | lens_kl_prev | cf_last_res |
|---|---|---|---|
| GQA → POPE | 99.8% / 93.9% | 100% / 65.3% | 100% / 60.5% |
| POPE → GQA | 75.0% / 49.5% | 88.3% / 62.6% | 87.0% / 62.7% |

Settings from the harder task transfer to the easier one at large savings; settings from the easy task are too
aggressive for the hard one, but the fixed layer fails worse (75%).

### 8A.6 Cost of the signals
Logit lens: one RMSNorm + LM-head on a single vector per layer ≈ 0.05% of prefill FLOPs. Counterfactual probe: one
query row of attention per layer (Q/K/V projections for one row + the key/value projections already needed by the
layer) — negligible, and neither needs the attention matrix, so SDPA/FlashAttention and the KV cache stay usable.

### 8A.7 Caveats
- "Fidelity" is agreement with the unpruned model, not task accuracy (Phase B measures both).
- n = 900 / 600; the POPE per-benchmark gain is within noise.
- Two benchmarks only; TextVQA and MME profiling (Phase A v2) tests whether one threshold still serves text-heavy
  and broader task mixes.
- Thresholds were chosen on the pooled POPE+GQA data; the paper should calibrate on a held-out calibration set.

**Verdict: Phase A passes in the one-threshold setting → proceed to Phase B.**

---

## 8B. Updated plan status (11 Oct 2026)

- **Method fixed:** keep all tokens in layers 0–1 (Phase C may add uniform 50% subsampling from layer 2), exit all
  visual tokens at the first layer where the **logit-lens KL to the previous layer ≤ τ** (τ from pooled calibration);
  counterfactual probe as the alternative signal. One global threshold.
- **Phase A v2** (`phaseA-exit-profiling-v2.ipynb`): TextVQA (500) + MME (≤50/category) profiles; pooled and transfer
  analyses across all four benchmarks.
- **Phase B** (`phaseB-exit-engine.ipynb`): SDPA + `DynamicCache` engine, exits by slicing hidden states while keeping
  original position ids (later layers simply cache fewer tokens); unit tests vs the stock model; thresholds recomputed
  from Phase A profiles; latency table (stock HF generate vs engine baseline vs fixed E vs adaptive); resumable
  evaluation with accuracy, agreement, FLOPs, exit-layer distribution, measured prefill/e2e latency; Phase C variants
  (`uniform 0.5 @ layer 2`) included.
- **Open:** FlashAttention-2 numbers need an Ampere GPU (T4 → SDPA only); second model (LLaVA-NeXT-7B); RefCOCO.

---

## 8C. Phase A v2: TextVQA and MME (first-token oracle, same protocol as §8A)

| | GQA (600) | MME (660) | POPE (900) | TextVQA (500) |
|---|---|---|---|---|
| Oracle earliest-stable exit: median | 11 | 6 | 7 | **17** |
| Oracle FLOPs | 37.0% | 23.0% | 24.2% | 46.8% |
| Fixed E for ≥98% fidelity | E30 → 94% | E15 → 49% | E15 → 49% | **E31 → 97%** |
| Per-benchmark best signal vs fixed E at 98% (held-out) | −14.6 pt | −1.3 pt | −3.8 pt | −6.0 pt |
| Full-answer agreement at the first-token oracle exit (proxy check) | 57/57 | 51/51 | 47/47 | **81.8% (54/66)** |

**One global threshold for all four benchmarks** (pooled calibration, held-out, T=98%; fidelity / FLOPs):

| Method | GQA | MME | POPE | TextVQA |
|---|---|---|---|---|
| Fixed E | 96.0 / 87.7 | 99.3 / 87.6 | 99.8 / 87.4 | 95.7 / 87.7 |
| cf_last_res | 95.8 / **75.7** | 99.2 / **58.1** | 99.8 / **59.4** | 96.6 / 86.6 |
| lens_kl_prev | 97.1 / 76.7 | 99.4 / 62.8 | 99.7 / 61.3 | 94.7 / 84.5 |

Reading: MME behaves like POPE (settled by layer ~15; a fixed layer is enough), TextVQA needs the image almost to
the end (fixed exit buys nothing), GQA sits in between. Per-image exit helps on the heterogeneous tasks (GQA,
TextVQA) and, with one global setting, saves 25–30 pt FLOPs on the easy tasks that a single fixed layer must
over-provision for. **TextVQA caveat:** the first-token proxy is unreliable there (multi-token answers), so TextVQA
numbers must come from Phase B full-answer evaluation.

---

## 8D. Phase B: engine verification, latency, and full-generation evaluation

**Setup.** `phaseB-exit-engine.ipynb`, stock `transformers` 4.49, fp16, SDPA attention, `DynamicCache`, 2×T4, batch 1.
Thresholds from pooled 4-benchmark Phase A calibration (T=98%): lens τ=0.0245 (m=1, E_min=1), cf τ=0.0501 (m=2,
E_min=12), fixed-E equivalent E=28.

**Engine correctness.** Max |logit diff| vs stock forward 0.0078; identical greedy text; fixed-exit path exact (0.0000).
Engine baseline e2e 935 ms vs HF `generate` 929 ms → no overhead.

**Latency (20 POPE images, 8 new tokens, EOS ignored; ms):**

| Method | Prefill | End-to-end | FLOPs | Mean exit |
|---|---|---|---|---|
| HF generate (SDPA + cache) | — | 929 | 100% | — |
| Engine baseline | 490 | 935 | 100% | — |
| Fixed E28 | 442 | 878 | 88.0% | 28 |
| Fixed E16 | 295 | 725 | 51.9% | 16 |
| Lens exit | 489 | 929 | 92.5% | 22.0 |
| Lens + uniform 50% @ L2 | 330 | 761 | 50.2% | 22.8 |

Prefill time tracks FLOPs (the residual is the vision tower's fixed cost). Note the "describe" prompt used here
exits later than the POPE questions below.

**Full-generation evaluation (acc / agreement with baseline / FLOPs / mean exit / prefill ms):**

| Method | POPE (900) | GQA (600) |
|---|---|---|
| baseline | 88.6 / 100 / 100% / — / 498 | 60.2 / 100 / 100% / — / 504 |
| fixed E16 | 88.8 / 99.6 / 51.7% / 16 / 308 | 53.0 / 73.3 / 52.7% / 16 / 315 (p<0.001 vs base) |
| fixed E20 | 88.6 / 100 / 63.8% / 20 / 343 | 57.8 / 85.8 / 64.5% / 20 / 347 (p=0.02) |
| fixed E24 | 88.6 / 100 / 75.8% / 24 / 400 | 59.2 / 93.0 / 76.3% / 24 / 403 |
| fixed E28 | 88.6 / 100 / 87.9% / 28 / 453 | 59.8 / 96.7 / 88.2% / 28 / 458 |
| **lens exit** | 88.6 / 99.8 / **61.0%** / 19.1 / 345 | 59.5 / **96.7** / **76.8%** / 21.9 / 421 |
| cf exit | 88.7 / 99.9 / 54.0% / 16.8 / 376 | 60.0 / 95.8 / 73.4% / 22.4 / 482 |
| uniform 50% @ L2 only | 87.6 / 96.6 / 54.0% / — / 321 | 58.7 / 92.0 / 55.0% / — / 320 |
| lens + uniform 50% @ L2 | 87.9 / 96.7 / **34.8%** / 19.1 / 244 | 58.8 / 91.0 / **43.4%** / 22.0 / 283 |

Findings:
1. **GQA, matched FLOPs:** lens exit at 76.8% FLOPs keeps 96.7% of answers; fixed E24 at 76.3% keeps 93.0%.
   **Matched fidelity:** fixed exit needs E28 (88.2%) for 96.7% → lens saves 11 pt. No accuracy change for lens/cf is
   significant (p ≥ 0.42); fixed E16/E20 are significantly worse.
2. **POPE:** lens ≈ fixed E20; a per-benchmark-tuned E16 is better (99.6% @ 51.7%). Same as Phase A: no per-image
   headroom on POPE. The one-global-setting advantage: a fixed layer good enough for GQA (E28) costs POPE 88% FLOPs vs
   61% with lens.
3. **Exit-layer distributions:** lens never fires before layer 17 (POPE mode 18–20; GQA spread 17–30 with 133/600 never
   exiting); cf fires at 12–20 on POPE and bimodally (12 or 28–32) on GQA.
4. **Phase C uniform stage costs ~1–1.5 pt accuracy** (agreement 92–97%) with or without exit; the exit adds almost no
   further loss. Combined point: POPE 87.9% @ 34.8% FLOPs, GQA 58.8% @ 43.4% — vs FastV K=2 R=0.5 (Phase 1, other
   subsets) 87.6% @ 53.9% and 59.0% @ 54.8%.
5. The pooled threshold is strict (set by TextVQA); a τ sweep is needed to trace the full accuracy-vs-FLOPs curve.

**Next (Phase B sweep):** fixed E ∈ {12,16,20,24,28}; lens τ ∈ {0.01,0.0245,0.05,0.1,0.2}; cf τ ∈ {0.025,0.05,0.1};
uniform ablation (0.5@L2, 0.75@L2, 0.5@L8) with lens τ=0.05; all on POPE, GQA, TextVQA, MME with full-answer scoring.

---

## 8E. Phase B sweep: 17 methods × 4 benchmarks (full-answer scoring, one global setting)

**Setup.** As §8D (fp16, SDPA, KV cache, 2×T4, batch 1). POPE 900, MME 660 (≤50/category), GQA 600, TextVQA 500
(VQA soft accuracy, no OCR tokens; baseline 50.8). MME ids are not unique (two questions per image), so MME
agreement is computed by record position. Data: `phaseB_results.zip` (20 Oct 2026).

### 8E.1 Accuracy / agreement with the unpruned model / prefill FLOPs (%)
| Method | POPE | MME | GQA | TextVQA | Mean FLOPs | Mean acc Δ |
|---|---|---|---|---|---|---|
| baseline | 88.6 / 100 / 100 | 74.4 / 100 / 100 | 60.2 / 100 / 100 | 50.8 / 100 / 100 | 100 | — |
| fixed E12 | 89.1 / 97.0 / 40 | 71.7\* / 92.7 / 41 | 47.5\* / 63.2 / 41 | 12.9\* / 20.2 / 41 | 40 | −13.2 |
| fixed E16 | 88.8 / 99.6 / 52 | 74.7 / 98.8 / 53 | 53.0\* / 73.3 / 53 | 17.1\* / 27.0 / 53 | 52 | −10.1 |
| fixed E20 | 88.6 / 100 / 64 | 75.0 / 98.8 / 64 | 57.8\* / 85.8 / 65 | 38.1\* / 60.8 / 64 | 64 | −3.6 |
| fixed E24 | 88.6 / 100 / 76 | 74.2 / 98.6 / 76 | 59.2 / 93.0 / 76 | 48.5\* / 82.8 / 76 | 76 | −0.9 |
| fixed E28 | 88.6 / 100 / 88 | 74.2 / 98.9 / 88 | 59.8 / 96.7 / 88 | 50.2 / 93.0 / 88 | 88 | −0.3 |
| lens τ=0.2 | 88.7 / 99.7 / 56 | 75.2 / 98.6 / 57 | 59.3 / 91.2 / 65 | 47.5\* / 77.0 / 70 | 62 | −0.8 |
| lens τ=0.1 | 88.6 / 99.8 / 58 | 75.0 / 98.8 / 58 | 60.2 / 94.0 / 69 | 49.5 / 84.6 / 75 | 65 | −0.2 |
| **lens τ=0.05** | 88.6 / 99.8 / 59 | 75.2 / 98.6 / 60 | 59.3 / 95.5 / 73 | 49.9 / 89.8 / 80 | **68** | **−0.25** |
| lens τ=0.0245 | 88.6 / 99.8 / 61 | 75.0 / 98.8 / 63 | 59.5 / 96.7 / 77 | 50.3 / 92.4 / 85 | 71 | −0.1 |
| lens τ=0.01 | 88.7 / 99.9 / 66 | 74.5 / 99.2 / 67 | 59.2 / 97.7 / 82 | 49.9 / 94.4 / 89 | 76 | −0.4 |
| cf τ=0.1 | 89.1 / 97.0 / 40 | 71.7\* / 92.7 / 41 | 55.7\* / 84.0 / 56 | 47.7\* / 85.0 / 75 | 53 | −2.5 |
| cf τ=0.05 | 88.7 / 99.9 / 54 | 73.6 / 95.6 / 49 | 60.0 / 96.0 / 73 | 50.3 / 96.0 / 91 | 67 | −0.3 |
| cf τ=0.025 | 88.4 / 99.9 / 67 | 74.4 / 98.5 / 67 | 60.2 / 99.3 / 89 | 50.8 / 100 / 98 | 80 | 0.0 |
| lens 0.05 + uniform 50% @ L2 | 87.9 / 96.7 / 34 | 73.3 / 92.6 / 35 | 59.0 / 91.0 / 42 | 44.3\* / 68.4 / 45 | 39 | −2.3 |
| lens 0.05 + uniform 50% @ L8 | 88.2 / 97.2 / 43 | 73.6 / 94.4 / 44 | 59.0 / 90.0 / 51 | 44.3\* / 70.6 / 55 | 48 | −2.2 |
| lens 0.05 + uniform 75% @ L2 | 89.1 / 98.8 / 46 | 73.9 / 96.5 / 47 | 58.7 / 92.3 / 57 | 48.8 / 78.4 / 63 | 53 | −0.9 |

\* significantly worse than baseline (exact McNemar, p < 0.05). **No lens or cf point with τ ≤ 0.1 (cf ≤ 0.05) is
significantly worse on any benchmark.**

### 8E.2 Matched-fidelity comparison (FLOPs a fixed exit layer needs for the same agreement, interpolated)
- **GQA:** adaptive saves +8 to +13 pt at every operating point (lens τ=0.05: 95.5% @ 73% vs fixed 84%; cf τ=0.05:
  96.0% @ 73% vs 86%).
- **TextVQA:** +3 to +4 pt; and lens τ≤0.0245 / cf τ≤0.05 reach 92–100% agreement, which no fixed E ≤ 28 reaches
  below 97% FLOPs.
- **POPE, MME:** fixed E16 (99.6% / 98.8% @ 52%) is as good or better than any adaptive point (−1 to −5 pt). No
  per-image headroom, as Phase A predicted.
- **One global setting:** the only fixed layer safe on all four tasks is E28 (88% FLOPs); lens τ=0.05 matches its
  accuracy at 68% mean FLOPs (59–60% on POPE/MME, 73% GQA, 80% TextVQA). Fixed E24 (76%) is significantly worse on
  TextVQA.

### 8E.3 Adaptivity
- Spearman correlation between the method's exit layer and the Phase A oracle (earliest stable exit) per example:
  cf τ=0.05: **+0.51 (GQA), +0.57 (TextVQA)**, +0.24 (POPE); lens τ=0.05: +0.42 (GQA), +0.22 (TextVQA).
- Exit layer by GQA question type (lens τ=0.05): query 25.6 · choose 22.0 · compare 19.3 · verify 18.6 · logical 18.3.
  MME categories all 17.7–19.7 (uniform, like POPE).
- Share of examples never exiting (lens τ=0.05): POPE 0%, MME 0%, GQA 13%, TextVQA 23%.
- The lens signal never fires before layer ~17 (logit lens is unstable earlier); cf fires from 12.

### 8E.4 Latency (prefill ms, same engine)
Lens probe is free: lens τ=0.05 336 ms @ 59% FLOPs vs fixed E20 343 ms @ 64%. The cf probe as implemented is not:
cf τ=0.05 377 ms @ 54% vs fixed E16 308 ms @ 52% (+70 ms), because it recomputes full-sequence K/V projections at
every probed layer. Fix: reuse the layer's cached K/V (one query row) → engine v2.

### 8E.5 Decisions
1. **Method = lens exit, default τ=0.05**, no uniform stage. cf as the stronger variant once the probe is cheap.
2. **Uniform subsampling is an ablation only** (−2.3 pt mean; −6.5 on TextVQA at 50%; −0.9 at 75%).
3. Baselines on the same engine: fixed exit (VTW), FastV K/R (to be added to the engine via the same one-row
   attention), uniform-only, unpruned.
4. Next: engine v2 (cache-based cf, FastV-in-engine); LLaVA-NeXT-7B with the **same τ, no recalibration**;
   Ampere GPU for FlashAttention timings; larger GQA/TextVQA subsets.

---

## 9. Execution plan: training-free per-image visual token exit (original plan; see 8B for status)

### 9.1 Method sketch
1. **Layers 0–1:** keep all 576 visual tokens. They are cheap, and image information hasn't been absorbed yet.
2. **Layer 2 until exit:** keep a **spatially uniform** subset (e.g. 50%, a regular grid). This needs no attention scores, avoids positional bias, and is supported by Findings 3–4, FEATHER and Wen et al.
3. **Per-image exit:** after each layer, run a cheap test of whether vision still matters for this question. When it says no, drop **all** remaining visual tokens.
4. **Kernel compatibility:** never materialize the attention matrix. Any signal that needs attention computes it manually for 1–2 query rows. Prefill shortens the sequence at layer boundaries. The KV cache stores fewer tokens in later layers, with original position ids preserved, so decoding needs no special mask.
5. **Training-free:** one global threshold, set once on a held-out calibration set.

Rough cost if the exit lands around layer 16: about a third of baseline prefill FLOPs. **To be confirmed in Phase A.**

### 9.2 Candidate exit signals
- **(a) Local counterfactual test (preferred).** At layer *l*, compute the last token's attention output with and without the visual keys and values. Exit when the relative change falls below τ. It is a per-image analogue of VTW's KL criterion and costs one query row.
- **(b) Visual attention mass.** The fraction of the last token's attention on image tokens, computed for one row. Watch for attention sinks.
- **(c) Answer stability (logit lens).** Project the last token through the final norm and LM head; exit once the top-1 prediction (or the KL between consecutive layers) has been stable for *m* layers. Cheap, but noisy in early layers. A tuned lens would need light training.

Per-image attention entropy is **excluded** by Finding 1.

### 9.3 Phases

**Phase A: go/no-go profiling (about 1 week, Kaggle T4).**
- Compute the **oracle exit layer** for every POPE and GQA example: the earliest layer after which dropping all visual tokens keeps the baseline answer. Binary search keeps this to about 5 forward passes per example.
- Log signals (a), (b) and (c) at every layer in the same pass.
- Offline: rank correlation and AUC between each signal and the oracle; a simulated router against VTW's single best fixed exit layer at matched FLOPs.
- Also report the true exit headroom (oracle-exit FLOPs vs VTW).
- **Gate:** no signal beats VTW at matched FLOPs → stop and write the analysis paper (framing B). Otherwise continue.

**Phase B: inference engine (about 2 weeks).**
- Move off the FastV fork to current `transformers`. Write our own prefill plus greedy decode loop over decoder layers with `DynamicCache` and SDPA.
- Prune or exit by slicing hidden states and their position embeddings while keeping original position ids. Later layers' caches hold fewer tokens.
- Load the model in fp16 split across both T4s, not 8-bit, because bitsandbytes distorts timings.
- **Unit test:** with pruning disabled, logits must match stock HF (max abs diff about 1e-3 in fp16).
- Latency harness: prefill time, end-to-end time and peak memory at batch size 1.

**Phase C: the method (1–2 weeks).**
- Uniform stage plus the winning exit signal; threshold calibrated on held-out data; determinism checks.
- A schedule ablation: no uniform stage vs uniform 25%, 50% or 75%.

**Phase D: evaluation (2–3 weeks).**
- **Models:** LLaVA-1.5-7B plus one more (LLaVA-NeXT-7B is easiest; Qwen2.5-VL-7B has more impact but needs more porting).
- **Benchmarks:** POPE, GQA, MME, TextVQA and a RefCOCO subset. TextVQA and RefCOCO stress fine detail and localization.
- **Baselines:** full model, FastV, VTW, random, uniform-only, and PDrop or FEATHER if time allows. Cite DyVTE and AdaVSkip numbers as trained references.
- **Metrics:** accuracy vs FLOPs (Pareto), measured prefill and end-to-end latency, memory, and exit-layer distributions per benchmark and question type.
- **Larger subsets:** about 1500 POPE and 1000 GQA questions, giving standard errors of about ±0.8 and ±1.5.

**Phase E: ablations and write-up (1–2 weeks).**
- Signal choice, threshold sensitivity, uniform ratio, overhead of the signal itself, and calibration transfer across benchmarks.
- Sections 3–6 of this file become the motivation and analysis section.

### 9.4 Feasibility and risks
| Piece | Difficulty | Main risk / mitigation |
|---|---|---|
| Uniform early stage | Easy | Low novelty alone (FEATHER, HiDrop). Use it as a component, not the claim. |
| KV-cache + SDPA engine | Medium | Bugs in positions and masks → equivalence unit test. Per-image exits break batching → report batch=1 or pad. |
| Per-image exit signal | Hard | No training-free signal beats VTW → Phase A decides this cheaply, before engineering. |
| Latency claims | Medium | FlashAttention-2 doesn't run on T4 (Turing) → SDPA on T4; final FA2 numbers on a rented Ampere GPU (A100/L4). |
| Novelty | Medium | AdaVSkip (Sept 2026) and DyVTE are close but trained → defend "training-free + kernel-compatible"; redo the literature search before submission. |
| Compute | Low–medium | Kaggle T4×2 is enough for Phases A–D accuracy runs; zip results after every session. |

### 9.5 Fallback (if Phase A fails)
Write the **analysis paper**: Findings 1–9, plus the random control (§7), a second model, larger n, and the per-benchmark "visual dependency depth" audit (the layer after which removing all visual tokens changes nothing). Target the NeurIPS "I Can't Believe It's Not Better" workshop, TMLR, or efficient-ML workshops.

---

## 10. Problems encountered and fixes (log)

| Problem | Cause | Fix |
|---|---|---|
| Notebooks "syntax" errors | Not syntax: out-of-order cells, missing imports (`re`), state from other notebooks | One self-contained notebook that runs top to bottom |
| `FileNotFoundError` loading a "pristine" `modeling_llama.py` | FastV not cloned in that session; the file also uses relative imports | Stash the original `fastv_forward` once on the class and assert its origin |
| `_ORIGINAL_FASTV_FORWARD` captured a patched function | Cell re-run after `enable_infoprune()` | Same stash + assert; a dispatcher replaces enable/disable toggling |
| `importlib.reload(modeling_llama)` breakage | Reload creates new classes mid-session | Removed; load the model once and switch `config.fastv_config` |
| Multi-GPU device mismatch in FastV | `keep_indexs` built on the wrong device | Patched `fastv_forward` to use `hidden_states.device` |
| NaN entropies (5 of 73 calibration images) | fp16 `eps=1e-8` → 0; 0·log 0 = NaN | fp32 + `torch.special.entr` |
| Biased NaN filtering | Dropped NaN traces were the sharpest distributions | Fixed by the item above |
| Calibration leakage | Calibration used the POPE eval images and a different prompt | Disjoint calibration images with benchmark-style prompts |
| H_data had no effect | Constant per layer, cancels under percentile scoring | Documented; score rewritten as `focus_score` |
| "InfoPrune matches baseline" overclaim | Argmax prunes after layer 16 (K=17 compute) | Added FastV K=17 and matched-K controls |
| `IMAGE_START` off by one | FastV default 5; HF prompt puts the image at 4 | Set to 4, with an assertion in `build_inputs` |
| Analysis mixed drop ratios | `fixed_k_runs` keyed by K only | R-aware `fixed_k_runs(runs, r)` and `ratios()` |
| Lost results after a Kaggle reset | `/kaggle/working` wiped when the session ended | Zip `results/` and `cache/` after every run, or Save Version |
| Latency not meaningful | `use_cache=False` (FastV's cache path is broken) | Report FLOPs now; Phase B engine adds KV cache |

---

## 11. Open questions
- Does random token selection match FastV at K=2 and K=3 (§7)? (deferred; literature says yes)
- ~~True exit headroom~~ → answered in §8A.1 (oracle 24% POPE / 37% GQA FLOPs).
- ~~Does any training-free signal track the oracle exit?~~ → yes from layer ~11 on (§8A.2); strongest in the one-threshold setting (§8A.4).
- Does the one-threshold result hold on TextVQA and MME (Phase A v2)?
- Does fidelity translate into task accuracy and *measured* latency with the KV cache (Phase B)?
- Does uniform 50% subsampling from layer 2 keep fidelity (Phase C)?
- Is the "focused attention breaks more at high R" effect on POPE real (§6.3)?
- Do findings 3–5 hold on a second model and on localization benchmarks?

---

## 12. Artifact inventory
- **Notebooks:** `fastv-phase0.ipynb`, `sparsevlm-phase0.ipynb`, `phase1-infoprune.ipynb` (old), `consolidated-infoprune.ipynb` (old), `infoprune-phase1-consolidated.ipynb` (Phase 1, run on Kaggle), `infoprune-offline-analysis.ipynb` (CPU analysis), `phaseA-exit-profiling.ipynb` (run), `phaseA-exit-profiling-v2.ipynb` (adds TextVQA/MME + pooled analysis), `phaseB-exit-engine.ipynb` (engine + latency + eval).
- **Results:** `infoprune_results.zip`, containing:
  - `results/{pope,gqa}__*.jsonl`: 19 runs per benchmark; one JSON line per example with answer, prediction, correctness, k_equiv, rel_flops, chosen_layer and per-layer scores.
  - `cache/calib_{pope,gqa}_L{2-16,0-31}_n150.json`: H_data, sorted per-layer H_img, and per-image traces.
- **Phase A results:** `phaseA_results_v2.zip` → `phaseA/profile_{pope,gqa,textvqa,mme}.jsonl` (per example: `preserved[E]`, `exit_top1`, `exit_kl`, per-layer `cf_*`, `mass_*`, `lens_*`), `verify_{pope,gqa}.jsonl`.
- **Phase B results:** `phaseB_results.zip` (20 Oct) → `phaseB/{pope,mme,gqa,textvqa}__{method}.jsonl` (20 methods) (answer, exit_layer, rel_flops, t_prefill, t_total, pred, correct), `calib.json`, `latency.json`.
- **Docs:** `research-and-literature.md` and this file.
