# Visual Token Pruning in VLMs: Research Notes and Literature

*A reference document for paper writing. It covers the problem, the concepts and definitions we use, how the project's direction evolved, the literature we have reviewed, where our work sits relative to it, and draft text for related work and positioning.*

*Companion file: `results-and-execution-plan.md`, which holds all numbers, findings and the execution plan.*

*Last updated: 4 October 2026.*

---

## 1. Project summary

We study **training-free visual token reduction** in vision-language models (VLMs), using LLaVA-1.5-7B as the test model. LLaVA turns every image into 576 visual tokens that pass through all 32 decoder layers, and these tokens dominate the cost of the prefill stage. Pruning methods remove some of these tokens partway through the decoder.

**How the project evolved:**

1. **Original idea (InfoPrune).** Borrow the *InfoScore* attention-reliability metric from attention-based segmentation work. Use it to choose, for each image, which decoder layer's attention to trust when pruning, replacing FastV's fixed layer.
2. **Finding.** The per-image entropy score does not predict which examples pruning breaks (AUC about 0.5 at every pruning ratio). The layer-level InfoScore cleanly separates layers, but the layer it recommends prunes no better than FastV's default. Accuracy is driven by *depth*, meaning how many layers process the full token set, not by the quality of the ranking layer.
3. **Current direction.** A method paper on **training-free, per-image "visual token exit"** that stays compatible with FlashAttention and the KV cache. The fallback is an analysis or negative-results paper built from what we already have.

> **Naming:** the method is **CounterPrune** (decided 29 October 2026; a web search for "CounterPrune" / "Counter-Prune" found no paper, code or product). Do **not** call it "InfoPrune": arXiv 2511.19518 already uses that name for an information-theoretic structural pruning method (see §5.7).

---

## 2. Problem setting

- **Model:** LLaVA-1.5-7B (`llava-hf/llava-1.5-7b-hf`). It has a CLIP ViT-L/336 vision encoder, an MLP projector and a Vicuna-7B (LLaMA-architecture) decoder with 32 layers, hidden size 4096 and FFN size 11008.
- **Visual tokens:** 576 per image (a 24×24 patch grid).
  - In the HF prompt `USER: <image>\n...`, the image block starts at sequence index **4** (`IMAGE_START=4`).
  - FastV's default config uses 5. Our Phase 0 runs used 5, which is off by one, with negligible effect on results.
- **Why prune:** prefill cost grows with sequence length (about 600 tokens, 96% of them visual). Attention is quadratic in length, and the FFN and projections are linear.
- **Two decisions every pruning method makes:**
  1. *Which* tokens to drop (the ranking criterion).
  2. *Where* and *how much* to drop (the layer K and the ratio R, or a schedule).

---

## 3. Concepts and definitions

### 3.1 FastV knobs
- **K:** the number of decoder layers that run on the full sequence before pruning. FastV with K ranks image tokens using **layer K−1's** attention and prunes after that layer. So K=2 uses layer 1's attention.
- **R:** the fraction of image tokens dropped. R=0.5 drops 288 of 576; R=0.75 keeps 144; R=0.9 keeps 58.

### 3.2 Prefill FLOPs (FastV paper's formula)
Per layer, with sequence length *n*, hidden size *d* and FFN size *m*:

> FLOPs(n) = 4·n·d² + 2·n²·d + 2·n·d·m

The relative prefill cost of pruning after K layers is:

> [K·FLOPs(n) + (32−K)·FLOPs(n − R·576)] / [32·FLOPs(n)]

Reference values for R=0.5: K=2 ≈ 54%, K=3 ≈ 55–56%, K=7 ≈ 62%, K=17 ≈ 77%.

### 3.3 Attention distribution over image tokens
At layer *l*, take the attention weights from the *query* tokens to the 576 image tokens. Average them over heads (and over query rows), then normalize to sum to 1. Call the result **a**ₗ.
- **Query rows (text-rater filter):** following SparseVLM, we keep only "relevant" text tokens. The filter scores each text token by the dot-product similarity between image and text embeddings before layer 0: softmax over the text tokens, mean over image patches, keep tokens scoring above the mean. If no token qualifies, we fall back to the last token.

### 3.4 Entropy quantities
- **Entropy:** Ent(**a**) = −Σⱼ aⱼ log aⱼ. The maximum over 576 tokens is ln 576 ≈ 6.356, reached by uniform attention.
- **H_img(l, i):** the entropy of layer *l*'s image-attention distribution for image *i*. Low means attention is concentrated on a few patches.
- **H_data(l):** the entropy of **a**ₗ *averaged over a calibration set*. High means the layer looks at different places for different images.
- **InfoScore(l) = H_data(l) / mean_i H_img(l, i).**
  - Because entropy is concave, H_data ≥ mean H_img, so **InfoScore ≥ 1 always**.
  - **InfoScore = 1 exactly when every image gets the same attention pattern**, i.e. the pattern is positional and independent of content.
  - High InfoScore means attention is both content-adaptive and decisive.
- **Effective number of attended tokens:** exp(H). Useful for intuition; for example, H = 3.1 gives about 22 tokens.

### 3.5 Per-image score used by InfoPrune ("focus score")
The score was originally defined as the percentile of H_data(l)/H_img(l,i) within layer *l*'s calibration distribution.

H_data(l) is constant within a layer, and percentiles don't change under monotone transforms, so **H_data cancels exactly**. The score reduces to: the fraction of calibration images whose attention at layer *l* was more diffuse than this image's. We checked numerically on 2000 random cases that the two forms give identical values.

### 3.6 Numerical note (worth a footnote)
In fp16, `eps = 1e-8` rounds to exactly 0, so 0·log(0) = NaN. Compute entropy in fp32, e.g. with `torch.special.entr`. Discarding NaN traces instead is biased, because it removes exactly the sharp, low-entropy distributions.

### 3.7 Evaluation terms
- **AUC:** the probability that a randomly chosen "broken" example scores higher than a randomly chosen "unbroken" one. 0.5 is chance.
- **McNemar test:** a paired test on discordant predictions. It is the right test when two methods answer the same questions.
- **Fidelity oracle:** for each example, the cheapest configuration that still reproduces the baseline answer. It measures headroom, i.e. how much an ideal router could save.
- **Visual token exit:** dropping *all* visual tokens after some layer, as in VTW and DyVTE.

---

## 4. Original hypothesis: InfoPrune (archived)

- **Hypothesis:** the right layer to prune at is per-image. A cheap, training-free entropy reliability score computed during the forward pass can find it.
- **Calibration:** estimate H_data(l) once on about 150 unlabeled images. At inference, compute the per-image score at layers 2–16 and use one of two rules:
  - **Argmax:** rank tokens with the best-scoring layer, and prune after layer 16.
  - **Threshold:** prune at the first layer whose score is at least τ, giving a per-image K\*.
- **Pruning mechanics:** FastV's in-place index-based token dropping, with R as a controllable knob.
- **Outcome:** the hypothesis was not supported. The evidence is in the companion file. The machinery and the layer-level InfoScore analysis are reused as **motivation and analysis** in the new direction.

---

## 5. Literature

Status key: ✅ abstract or paper read and the key claim checked in this project · 🔎 seen in search results or a user note, not yet read · ⚠️ a claim we must verify before citing.

### 5.1 Attention-based pruning (the methods we compare against)

**FastV.** Chen et al., *An Image is Worth 1/2 Tokens After Layer 2: Plug-and-Play Inference Acceleration for Large Vision-Language Models*, ECCV 2024. Code: https://github.com/pkunlp-icler/FastV. ✅ code read.
- **Method:** after a fixed layer K (default 2), rank image tokens by the attention they receive from the last token, averaged over heads, and drop the lowest fraction R. K and R are global and set by grid search.
- **Code notes:**
  - The ranking uses layer K−1's attention.
  - Requires `attn_implementation="eager"`.
  - The KV-cache path carries a comment that it causes a CUDA error, so we use `use_cache=False`.
  - The repo ships a patched `transformers==4.39.0.dev0`.
- **Use:** primary baseline and our pruning-mechanics backbone.
- ⚠️ Check FastV's own K/R ablation tables before claiming anything about K=3 vs K=2.

**SparseVLM.** Zhang et al., *SparseVLM: Visual Token Sparsification for Efficient Vision-Language Model Inference*, arXiv 2410.04417. Code: https://github.com/Gumpest/SparseVLMs. ✅ code used. ⚠️ confirm the venue (believed ICML 2025).
- **Method:** text-guided. It selects "relevant" text tokens as raters, scores visual tokens with their attention, and prunes progressively at several layers with adaptive ratios. It also recycles pruned tokens.
- **Code notes:** uses the original LLaVA codebase and checkpoint (`liuhaotian/llava-v1.5-7b`) and SDPA. The loader attaches FlashAttention-2 submodules, which don't run on T4, but they are not on the eval path.
- **Use:** baseline, and the source of our text-rater filter.
- **Limitation for our framing:** treats every layer's attention as equally trustworthy.

**PyramidDrop (PDrop).** arXiv 2410.17247, CVPR 2025. Code: https://github.com/Cooperx521/PyramidDrop. 🔎
- **Method:** splits the LLM into stages and drops a fraction of image tokens at the end of each stage, giving a pyramid-shaped token count.
- **Use:** progressive-pruning baseline, and prior art for the "progressive" part of the user's idea.

**STAR-Pro.** arXiv 2609.05916, 2026. ✅ abstract.
- **Method:** training-free and stage-wise. A pivoted-QR coverage candidate pool is followed by progressive pruning using evolving text-to-visual attention, with nested survivor sets.
- **Notes:** never drops all visual tokens. FlashAttention compatibility is not discussed.

**Other attention- or graph-based methods to read:** 🔎
- Fit and Prune (AAAI 2025).
- ATP-LLaVA (CVPR 2025): adaptive token pruning; check whether it is trained.
- Conical Visual Concentration (CVPR 2025).
- ViTCoP (arXiv 2601.17818).
- *Capturing Token Tendencies for Training-Free Token Pruning* (arXiv 2607.28341).
- *Rethinking Visual Token Reduction in LVLMs under Cross-Modal Misalignment* (AAAI 2026).
- *Beyond Text-Visual Attention: Exploiting Visual Cues for Effective Token Pruning* (ICCV 2025).

### 5.2 Critiques and analyses (most important for framing)

**Feather the Throttle.** Endo et al., *Feather the Throttle: Revisiting Visual Token Pruning for Vision-Language Model Acceleration*, ICCV 2025, arXiv 2412.13180. ✅ abstract and key claims.
- **Findings:**
  - FastV's early-layer attention has a strong positional bias: because of RoPE's long-term decay, it disproportionately removes tokens from the top of the image.
  - Most benchmarks barely change under aggressive pruning, but localization tasks collapse by up to 91%.
  - Performance stays strong even with suboptimal pruning criteria, which suggests that many benchmarks don't need fine visual detail.
- **FEATHER:** removes RoPE from the ranking attention, adds uniform spatial sampling, and prunes in two stages (layers 8 and 16).
- **Relation to us:**
  - Our InfoScore ≈ 1 at layers 0–1 independently confirms that early attention is image-independent.
  - The user's "uniform early" idea closely follows FEATHER.
  - Localization benchmarks are where token choice actually matters.

**Wen et al.** *Token Pruning in Multimodal Large Language Models: Are We Solving the Right Problem?*, ACL Findings 2025, arXiv 2502.11501. ✅ abstract and key claims.
- **Findings:**
  - Random selection and average pooling beat carefully designed methods on about 2/3 of benchmarks.
  - Attention-based scores (FastV) suffer from position bias.
  - Text guidance helps mainly on language-heavy tasks.
  - FLOPs and token counts are misleading proxies, so latency should be reported.
  - The authors recommend simple, hardware-friendly operations in shallow layers.
- **Relation to us:** our "ranking layer doesn't matter, depth dominates" result fits this. It is also why our method must report real latency with FlashAttention or SDPA and KV cache.

### 5.3 Visual token exit (closest to the new method)

**VTW.** Lin et al., *Boosting Multimodal Large Language Models with Visual Tokens Withdrawal for Rapid Inference*, arXiv 2405.05803 (AAAI 2025 ⚠️ verify). ✅ abstract and key claims.
- **Method:** removes *all* visual tokens after a layer chosen **per dataset**. The layer is picked by the KL divergence of outputs on a small sample, taking the first layer that meets the criterion. Training-free and plug-and-play, and compatible with FlashAttention and KV cache.
- **Results:** more than 40% FLOPs reduction with minimal loss on AI2D, SQA-IMG, MMMU, MME, POPE and others.
- **Relation to us:** the training-free, *dataset-level* baseline that our per-image exit must beat at matched FLOPs.

**DyVTE.** *Accelerating Multimodal Large Language Models via Dynamic Visual-Token Exit and the Empirical Findings*, arXiv 2411.19628. ✅ abstract and key claims.
- **Findings:** MLLM processing goes through three stages: early fusion, intra-modality modeling, and multimodal reasoning. Visual tokens stop contributing once text tokens have absorbed enough image information, typically after layers 16–25. **The optimal exit layer differs across examples, tasks and models.**
- **Method:** lightweight hypernetworks read text-token states and decide the exit per instance. They are trained cheaply with binary labels (whether the answer with exit matches the answer without), not by fine-tuning the MLLM. Up to 45.7% compute reduction without loss.
- **Relation to us:** the main *trained*, per-instance competitor. Our angle is to remove the training.

**AdaVSkip.** *Adaptive Visual Token Skipping Across Layers for Efficient MLLMs Inference*, arXiv 2609.15131 (September 2026). ✅ abstract and key claims.
- **Method:** two lightweight routers per layer decide whether visual tokens skip self-attention and/or the MLP, per instance and per module. Training has two stages: necessity-guided supervision using KL between original and intervened outputs, then RL with correctness rewards and skip-consistency regularization. Prefill only.
- **Results:**
  - 53.2% FLOPs reduction on LLaVA-NeXT-7B with performance preserved; 91.2% when combined with CDPruner.
  - Evaluated on LLaVA-1.5-7B, LLaVA-NeXT-7B and Qwen3-VL-8B.
  - Compares against VTW, DyVTE, ShortV, CDPruner, VisionZip, MMTok and ZOO-Prune.
- **Relation to us:** the closest and most recent work. It is trained, so training-free per-instance exit is still open, but the area is moving fast. **Must cite and compare.**

**HiDrop.** *Hierarchical Vision Token Reduction in MLLMs*, arXiv 2602.23699. ✅ abstract and key claims.
- **Method:** injects visual tokens late, prunes progressively in middle layers with a concave pyramid, and drops all visual tokens before the language-dominant deep layers. Pruning points are chosen by inter-layer attention similarity; it uses a differentiable top-k and is **trained** during instruction fine-tuning. The schedule is fixed, not per-instance.
- **Relation to us:** prior art for the "uniform/progressive, then exit" schedule, but trained and fixed.

### 5.4 Kernel-compatible and pre-LLM reduction

**TopV.** *Compatible Token Pruning with Inference Time Optimization for Fast and Low-Memory Multimodal Vision Language Model*, CVPR 2025, arXiv 2503.18278. ✅ title and abstract via search.
- Pruning designed to be compatible with FlashAttention and KV cache.
- **Use:** related work for our compatibility requirement.

**FasterVLM.** *[CLS] Token Tells Everything Needed for Training-free Efficient MLLMs*, arXiv 2412.05819. 🔎
- Ranks tokens by the vision encoder's [CLS] attention *before* the LLM, which keeps the LLM path kernel-compatible.

**VisionZip** (CVPR 2025). 🔎 Pre-LLM token selection and merging; frequently used as a baseline.

### 5.5 Learned pruning

**TPRL.** *Language-Guided Token Compression with Reinforcement Learning in Large Vision-Language Models*, arXiv 2603.13394. 🔎 (from the user's notes)
- Treats pruning as a multi-step decision problem trained with PPO. Prunes up to 66.7% of tokens (54.2% FLOPs reduction) with about a 0.7% average drop.
- **Use:** a learned upper reference point; it is not training-free.

### 5.6 Source of the reliability metric

**InfoScore / Segmentation-From-Attention.** Repo: https://github.com/rayat137/Segmentation-From-Attention (`compute_infoscore_blip.py`). ⚠️ The paper title, venue and claims are not yet verified.
- **Metric:** InfoScore = dataset-level entropy of attention ÷ mean image-level entropy. The layer with the highest score is judged most reliable for localizing content.
- **Reported (⚠️ verify):** the top InfoScore layers track the best-mIoU layers, and the paper includes LLaVA-1.5-7B experiments; the public code covers BLIP only.
- **Use:** the source of our layer-reliability analysis. We reimplemented it for LLaVA's causal decoder.

### 5.7 Name collision

**InfoPrune (another paper).** *Towards Efficient VLMs: Information-Theoretic Driven Compression via Adaptive Structural Pruning*, arXiv 2511.19518. 🔎 (from the user's notes)
- Structural (weight) pruning of VLMs; it plans to extend to language components together with quantization and distillation.
- **Action:** our method needs a different name.

### 5.8 Models and benchmarks (cite in the setup section)
- **LLaVA-1.5:** Liu et al., *Improved Baselines with Visual Instruction Tuning*, CVPR 2024.
- **POPE:** Li et al., *Evaluating Object Hallucination in Large Vision-Language Models*, EMNLP 2023. Splits: random, popular, adversarial. Repo: https://github.com/AoiDragon/POPE.
- **GQA:** Hudson & Manning, CVPR 2019. We use the testdev-balanced split and LLaVA's prompt "Answer the question using a single word or phrase."
- **MME:** Fu et al., 2023. We used the HF mirror `lmms-lab-encoder/MME` (⚠️ check the exact dataset name in the notebook).
- **Planned:** TextVQA, RefCOCO (localization), and a second model (LLaVA-NeXT-7B or Qwen2.5-VL-7B).

### 5.9 Interpretability tools (for exit signal (c))
- Logit lens: nostalgebraist, 2020 (blog post).
- Tuned lens: Belrose et al., 2023, *Eliciting Latent Predictions from Transformers with the Tuned Lens*. ⚠️ verify the citation.

---

## 6. Gap analysis and positioning

| Method | Training-free | Per-instance decision | Drops all visual tokens | FlashAttention + KV cache | Ranking uses attention |
|---|---|---|---|---|---|
| FastV | ✅ | ❌ | ❌ | ❌ (needs eager) | ✅ |
| SparseVLM | ✅ | partly (adaptive ratio) | ❌ | ❌ | ✅ |
| PDrop | ✅ | ❌ | ❌ | partly | ✅ |
| FEATHER | ✅ | ❌ | ❌ | partly | partly (+ uniform) |
| STAR-Pro | ✅ | partly | ❌ | not discussed | ✅ |
| VTW | ✅ | ❌ (per dataset) | ✅ | ✅ | ❌ |
| DyVTE | ❌ (hypernets) | ✅ | ✅ | partly | — |
| AdaVSkip | ❌ (routers + RL) | ✅ | skips modules | prefill only | — |
| HiDrop | ❌ (trained) | ❌ | ✅ | — | ✅ |
| **Ours (planned)** | ✅ | ✅ | ✅ | ✅ (target) | ❌ (uniform + 1-row probe) |

**Positioning in one sentence:** a training-free, per-instance visual-token exit that needs no attention matrices, so it stays compatible with FlashAttention, SDPA and KV cache. It is motivated by an analysis showing that attention-ranking quality matters far less than how deep the visual tokens are kept.

---

## 7. Draft related-work text (edit freely)

> **Attention-guided token pruning.** FastV [Chen et al., 2024] prunes low-attention visual tokens after a fixed early layer, and SparseVLM [Zhang et al., 2024] uses text-selected raters with progressive, layer-wise pruning. PyramidDrop [2025] and STAR-Pro [2026] refine the schedule into multiple stages. All of these rank tokens with decoder attention, which requires materializing attention weights and is therefore incompatible with fused attention kernels.
>
> **Critiques.** Recent analyses question the premise of attention-based ranking. Endo et al. [2025] show that FastV's early-layer attention is dominated by RoPE-induced positional bias and fails on localization, and Wen et al. [2025] find that random selection and pooling match or beat elaborate criteria on most benchmarks. Our layer-wise InfoScore analysis gives an information-theoretic account of this: attention in layers 0–1 is nearly image-independent (InfoScore ≈ 1), yet ranking with the most content-adaptive layer yields no accuracy gain over ranking with layer 1.
>
> **Visual token exit.** VTW [Lin et al., 2024] removes all visual tokens after a dataset-level layer chosen by a KL criterion. DyVTE [2024] and AdaVSkip [2026] make this decision per instance, but require trained hypernetworks or routers. HiDrop [2026] combines late injection, progressive pruning and early exit in a trained, fixed schedule. Our method is, to our knowledge, the first training-free, per-instance exit that is also compatible with fused attention kernels and KV caching. *(⚠️ Re-check this claim with a full literature search before submission.)*

---

## 8. Paper framings

**A. Method paper (primary).**
- *Claim:* a training-free per-instance visual exit beats VTW's dataset-level exit at matched FLOPs and real latency, approaches trained per-instance methods (DyVTE, AdaVSkip), and degrades gracefully on localization.
- *Venues:* CVPR, ICCV, ECCV, NeurIPS or ICLR (main), or efficient-ML workshops.

**B. Analysis or negative-results paper (fallback, mostly done).**
- *Claim:* attention reliability, measured by InfoScore, cleanly characterizes layers but does not improve pruning. Neither per-image entropy nor the most reliable layer helps; depth dominates. A random-ranking control would complete it, plus a "visual dependency depth" benchmark audit.
- *Venues:* the NeurIPS "I Can't Believe It's Not Better" workshop, TMLR, efficient-ML workshops.

**C. Benchmark audit (can be a section of A or B).**
- *Claim:* per-benchmark "visual dependency depth", the layer after which removing all visual tokens changes nothing. For example, POPE answers don't depend on image tokens after layer 16 in our runs.

---

## 9. To-read and verify list
- [ ] FastV K/R ablation tables, to see whether K=3 vs K=2 is already reported.
- [ ] Segmentation-From-Attention paper: exact title, venue, the InfoScore definition, and whether it includes LLaVA experiments.
- [ ] SparseVLM and VTW venues.
- [ ] AdaVSkip in full: router inputs, cost, exact baselines and numbers.
- [ ] DyVTE: exit-layer distributions per task, to compare with our oracle.
- [ ] VisionZip, FasterVLM, CDPruner, ShortV, MMTok, ZOO-Prune (AdaVSkip's baselines).
- [ ] Any training-free per-instance visual-exit paper from 2025–26. **This is the critical novelty check.**
- [ ] ATP-LLaVA, Fit and Prune, Conical Visual Concentration, ViTCoP, Capturing Token Tendencies, the AAAI 2026 misalignment paper.
- [ ] Logit lens and tuned lens citations.
