# InfoPrune: Reliability-Guided Adaptive-Layer Visual Token Pruning

*Project overview: what we are doing, why, what is new, and where things stand.*

---

## 1. The problem

Vision-language models like LLaVA-1.5 turn every image into 576 visual tokens and feed them through
the language decoder alongside the text prompt. Most of the prefill compute goes to these visual
tokens, and many of them contribute little to the answer. **Visual token pruning** drops the
unimportant ones partway through the decoder, which cuts FLOPs and latency with small accuracy loss.

Every pruning method has to answer two questions:

1. **Which tokens are unimportant?** Usually answered with attention scores: tokens that receive little
   attention from the text are dropped.
2. **Which layer's attention do we trust to make that call?** This is the question current methods
   answer poorly, and it is the focus of this project.

## 2. What existing methods do

### FastV ([pkunlp-icler/FastV](https://github.com/pkunlp-icler/FastV))
After a fixed layer K (default K=2), FastV ranks visual tokens by the attention they receive from the
last token and drops the lowest-ranked fraction R. K and R are the same for every image and every
dataset, tuned once by grid search (the repo ships an ablation script for sweeping them).

**Limitation:** one global cut point. It never asks whether layer K's attention is a good importance
signal for *this* image.

### SparseVLM ([Gumpest/SparseVLMs](https://arxiv.org/abs/2410.04417))
SparseVLM is text-guided. It first filters out uninformative text tokens (the "text raters" step,
e.g. dropping prepositions), then uses the remaining relevant text tokens to score visual-token
importance through the decoder's self-attention. It prunes progressively at several layers with an
adaptive per-layer ratio.

**Limitation:** more sophisticated than FastV, but it treats every layer's attention as equally
trustworthy. It never checks whether a given layer's attention is a reliable importance signal.

### InfoScore / Segmentation-From-Attention ([rayat137/Segmentation-From-Attention](https://github.com/rayat137/Segmentation-From-Attention))
This work asks which layer's attention is most reliable for localizing image content, in the context of
zero-shot segmentation. Its **InfoScore** metric combines two entropies:

- **Image-level entropy:** the average per-image entropy of the attention distribution. Low means each
  image's attention is concentrated and decisive.
- **Dataset-level entropy:** the entropy of the attention distribution aggregated across a whole
  dataset. High means the attention pattern changes with image content, rather than always landing on
  the same positions (a positional bias).

InfoScore is the ratio of dataset-level to image-level entropy. The layer with the highest score is
judged most reliable. Their ablations report that the top InfoScore-ranked layers track the layers
with the best segmentation mIoU, and the paper reports experiments on LLaVA-1.5-7B as well as BLIP and
ALBEF. The public code, `compute_infoscore_blip.py`, covers only BLIP, so the LLaVA version has to be
reimplemented. (Both of these points should be checked against the paper before we cite them.)

## 3. The gap and our idea

**The gap:** no pruning method we have looked at uses a reliability signal to decide *which layer's
attention to trust* for pruning. FastV hardcodes the layer; SparseVLM trusts all layers equally.

> A dedicated literature search for adaptive or instance-dependent pruning-layer selection is still an
> open to-do before we claim this gap in writing.

**InfoPrune** combines the three lines of work:

- InfoScore's entropy-based reliability signal decides **where** to prune.
- FastV's in-place token-dropping mechanics do the **pruning**.
- SparseVLM's relevant-text-token filtering defines the **query** whose attention we measure.

**Core hypothesis:** the right layer to prune at is neither global (FastV) nor "every layer"
(SparseVLM). It depends on the image, and an entropy-based reliability score can estimate it on the fly
without training, reusing attention computed during the forward pass.

## 4. Formulation

For layer *l* and image *i*, let **a**ₗ be the attention from the relevant instruction tokens (selected
with SparseVLM's text-rater filter) over the V = 576 visual tokens, averaged over heads and normalized to
a probability distribution. Entropy is Ent(**a**) = −Σⱼ aⱼ log aⱼ.

| Quantity | Definition | Interpretation |
|---|---|---|
| H_img(l, i) | Ent(**a**ₗ) for image *i* | Low: layer *l* concentrates decisively on a few tokens for this image |
| H_data(l) | Entropy of **a**ₗ averaged over a calibration set | High: layer *l*'s attention varies with content instead of following a fixed positional pattern |
| InfoScore(l) | H_data(l) / mean_i H_img(l, i) | High: content-adaptive and decisive, so a trustworthy pruning signal |

This adaptation to pruning is our extension, not something the InfoScore paper does, so the exact
form is a hypothesis to validate.

### Decision rule at inference
H_data(l) is estimated once, offline, on a small calibration set (a few hundred images). At inference,
layers run in order anyway, so after each candidate layer we compute that image's score and **prune at
the first layer whose score clears a threshold τ**. This gives a per-image cut point K\* instead of
FastV's fixed K, with no second forward pass.

The drop ratio R stays an explicit knob, so we can sweep the accuracy-versus-compute trade-off exactly
as FastV does.

## 5. What we have learned so far

### Implementation findings
- **Attention weights require eager attention.** SDPA and FlashAttention do not return attention
  weights, so we run `attn_implementation="eager"` on FastV's patched transformers fork
  (4.39.0.dev0), with LLaVA-1.5-7B loaded in 8-bit on Kaggle 2×T4.
- **NaN root cause.** Early calibration runs produced NaN entropies for some images. The cause:
  `eps = 1e-8` rounds to exactly 0 in fp16, so a zero attention weight gives 0·log 0 = NaN. Discarding
  those traces was biased, because it removed exactly the sharp, low-entropy layers the method is
  looking for. Entropy is now computed in fp32 with `torch.special.entr`.
- **H_data cancels under percentile scoring.** Our per-image score ranks H_data(l)/H_img(l, i) as a
  percentile within layer *l*'s calibration distribution. H_data(l) is constant within a layer, so it
  drops out, and the score reduces to "how much sharper is this image's attention at layer *l* than
  typical calibration images." This is still a sensible signal, but the dataset-level half of
  InfoScore currently does no work. See the open questions in §7.
- **Calibration hygiene.** The first calibration used the POPE evaluation images and a different
  prompt than evaluation. It now uses disjoint images with benchmark-style questions, separately for
  each benchmark.

### Results to date (small subsets, 8-bit, `use_cache=False`)

**POPE, 450 questions pooled across the random, popular, and adversarial splits (HF LLaVA-1.5-7B):**

| Setting | Acc | F1 | Yes % | Approx. prefill FLOPs |
|---|---|---|---|---|
| Baseline | 88.7 | 88.7 | 50.2 | 100% |
| FastV K=2, R=0.5 | 87.6 | 87.0 | 46.0 | ~54% |
| FastV K=2, R=0.75 | 83.6 | 81.2 | 37.6 | lower |
| FastV K=3, R=0.5 | 88.0 | 87.6 | 46.4 | ~56% |
| InfoPrune (argmax over layers 2–16), R=0.5 | 88.7 | 88.7 | — | ~77% |

**Important caveat on the InfoPrune row.** That version picks the best-scoring layer from layers 2–16
but always prunes *after layer 16*. Its compute matches FastV at K=17, not K=2. Pruning that late is
expected to be nearly lossless, so matching baseline (450/450 identical predictions) does **not** show
that InfoPrune beats FastV. The fair comparisons, now built into the notebook, are:

- InfoPrune-argmax versus FastV K=17 (same prune point);
- InfoPrune-threshold versus FastV at the matched mean K\*.

**SparseVLM (original LLaVA codebase and checkpoint, same POPE subset):** baseline 87.8 acc / 88.2 F1,
SparseVLM 87.6 / 87.8. Because this uses a different codebase and checkpoint, these numbers are not
directly comparable to the HF rows above.

**MME:** a 200-example check covered only two categories (artwork and code reasoning), where baseline
scored 72.0% and FastV K=2 scored 71.5%. That is a pipeline check, not a benchmark result.

**Calibration signal:** H_data(l) over layers 2–16 ranged from about 5.15 to 5.73, against a maximum of
ln 576 ≈ 6.36. On five demo images, the chosen layer varied (2, 3, 4, 5, 14), which is early evidence
that the choice depends on the image.

## 6. Research plan and status

| Phase | Goal | Status |
|---|---|---|
| 0: Baselines | Reproduce FastV and SparseVLM on POPE, GQA, MME subsets | Done for POPE; MME partial; GQA data ready |
| 1: Port InfoScore to LLaVA | Per-layer entropy in the causal decoder, calibration set, online H_img | Done; consolidated notebook with the fixes above |
| 2: Integrate into pruning | Per-image K\* via threshold, SparseVLM text filtering, R as a knob | Threshold mode implemented; needs evaluation; KV-cache support pending |
| 3: Evaluation | GQA, VQAv2, TextVQA, POPE, MME at matched FLOPs; layer-choice analysis | Next |
| 4: Ablations | τ sensitivity, head aggregation, combining with progressive pruning, overhead | Planned |
| 5: Write-up | Paper | Planned |

### Evaluation design (Phase 3)
- **Baselines:** the full model; FastV at K=2 and several R values; SparseVLM; and an **oracle**, the
  best fixed K per dataset from grid search. The oracle tests whether adaptive selection beats even a
  fully tuned static baseline, not just FastV's default setting.
- **Metrics:** accuracy at matched FLOPs (Pareto curves over R); POPE F1 as the hallucination measure;
  and an analysis of which layers are chosen per image and benchmark. The analysis should show whether
  the choice correlates with image clutter, object count, or question type, or whether it just
  reproduces the layer FastV already uses.

## 7. Open questions and risks

1. **Making H_data matter.** Under percentile scoring, the dataset-level entropy cancels. Options:
   - use the raw ratio H_data(l)/H_img(l, i) with a threshold, so H_data weights the layers;
   - use the dataset-level InfoScore to *pre-select* a few reliable candidate layers, then choose among
     them per image;
   - accept the simpler "relative sharpness" score and describe it honestly.

   Earlier experiments suggested raw-ratio thresholding did not discriminate between layers, so this
   needs a careful ablation.
2. **Is adaptivity real or noise?** If the chosen layer does not correlate with image or task
   properties, the honest claim becomes "matches a tuned fixed K without the tuning." That is still a
   contribution, but a different one.
3. **Overhead is not zero.** Reading attention weights forces eager attention, which gives up
   FlashAttention and SDPA speedups. A practical version would compute only the few query rows we need
   (relevant text tokens × visual tokens) directly from Q and K. This makes the Phase 4 overhead
   measurement essential, since the pitch depends on the score being nearly free.
4. **The query-token definition.** The text filter currently scores all tokens after the image,
   including "ASSISTANT:" and generated tokens. SparseVLM restricts it to the instruction.
5. **KV cache.** Everything so far runs with `use_cache=False`, so wall-clock latency is not
   meaningful yet. Until cached decoding works, we report FLOPs.
6. **Calibration transfer.** Does one calibration set (for example, COCO) work across benchmarks, or
   does each benchmark need its own? This is a cheap ablation worth reporting.

## 8. End goal

A training-free, drop-in pruning method for vision-language models that chooses, per image, **when**
to prune using a cheap reliability signal. The target claim for the paper:

> At matched FLOPs, reliability-guided per-image layer selection matches or beats both a fixed-layer
> pruner (FastV, including its best per-dataset K) and a uniform multi-layer pruner (SparseVLM),
> reduces hallucination (POPE F1), and adds negligible overhead. Analysis shows the selected layer
> genuinely adapts to image and task content.

**Novelty in one sentence:** we treat *which layer's attention to trust* as a per-instance decision,
and answer it with an entropy-based reliability score borrowed from attention-based segmentation.
Prior pruning methods either fix that layer globally or trust every layer equally.
