import json

cells = []
def md(s): cells.append(("markdown", s.strip("\n")))
def code(s): cells.append(("code", s.strip("\n")))

md(r'''
# InfoPrune — offline analysis of the Phase 1 results (no GPU needed)

Reads the per-example JSONL files and calibration caches the consolidated notebook already wrote to
`/kaggle/working/results` and `/kaggle/working/cache`. Run it in the same Kaggle session (or attach
that notebook's output as a dataset and point `ROOT` at it). CPU only; takes seconds.

It answers one gating question before we spend more GPU time: **does the per-image InfoPrune score carry
any information about which examples are hurt by early pruning?**

1. Paired significance of the headline comparisons (exact McNemar).
2. Headroom: how cheap could a *perfect* per-example K router be?
3. Does the score predict which examples early pruning breaks? (AUC)
4. Offline router: send each example to FastV K=2 or K=17 by its score, and compare against random routing
   at the same rate. Every example already has K=2 and K=17 predictions on disk, so this is exact.
5. Dataset-level InfoScore per layer from the calibration caches.
6. Is the chosen layer driven by the image, the question, or neither?
7. Accuracy vs K from the FastV sweep, next to the full-depth InfoScore curve.

§2–§4 repeat for every drop ratio R that has FastV K=2 (and K=17) runs on disk.
''')

code(r'''
import json, math, re
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

ROOT = Path("/kaggle/working")          # change if the files live elsewhere
RESULTS, CACHE, DATA = ROOT / "results", ROOT / "cache", ROOT / "data"

def load_runs(bench):
    runs = {}
    for p in sorted(RESULTS.glob(f"{bench}__*.jsonl")):
        recs = [json.loads(l) for l in open(p)]
        runs[p.stem.split("__", 1)[1]] = {r["id"]: r for r in recs}
    common = set.intersection(*(set(r) for r in runs.values()))
    return {n: {i: r[i] for i in common} for n, r in runs.items()}, sorted(common)

def is_correct(bench, r):
    return r["pred"] == r["label"] if bench == "pope" else bool(r["exact"])

def find(runs, pattern):
    hits = [n for n in runs if re.fullmatch(pattern, n)]
    return hits[0] if hits else None

def fixed_k_runs(runs, r):
    """{K: run name} for FastV runs at drop ratio r, plus the baseline as K=32."""
    out = {32: "baseline"}
    for n in runs:
        m = re.fullmatch(r"fastv_K(\d+)_R([\d.]+)", n)
        if m and float(m.group(2)) == float(r):
            out[int(m.group(1))] = n
    return dict(sorted(out.items()))

def ratios(runs):
    return sorted({float(m.group(1)) for n in runs if (m := re.fullmatch(r"fastv_K\d+_R([\d.]+)", n))})

def run_name(runs, k, r):
    return fixed_k_runs(runs, r).get(k)

BENCHES = [b for b in ("pope", "gqa") if list(RESULTS.glob(f"{b}__*.jsonl"))]
DATASETS = {b: load_runs(b) for b in BENCHES}
for b, (runs, ids) in DATASETS.items():
    print(f"{b}: {len(ids)} examples common to {len(runs)} runs -> {list(runs)}")
''')

md(r'''
## 1. Paired significance
Every method answers the same questions, so the right test is paired. `b01` = only A correct, `b10` =
only B correct. With 300–450 questions, a 1–2 point gap is usually 5–8 discordant questions.
''')
code(r'''
def mcnemar(a, b):
    b01 = sum(x and not y for x, y in zip(a, b))
    b10 = sum(y and not x for x, y in zip(a, b))
    n = b01 + b10
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b01, b10) + 1)) / 2 ** n)
    return b01, b10, p

for bench, (runs, ids) in DATASETS.items():
    acc = {n: [is_correct(bench, runs[n][i]) for i in ids] for n in runs}
    thr, arg = find(runs, r"infoprune_thr.*"), find(runs, r"infoprune_argmax.*")
    kmatch = [n for n in runs if n.startswith("fastv_K") and n not in ("fastv_K2_R0.5", "fastv_K17_R0.5")]
    pairs = [(arg, "fastv_K17_R0.5"), (thr, kmatch[0] if kmatch else None), (thr, "fastv_K2_R0.5"),
             ("fastv_K2_R0.5", "baseline"), ("fastv_K17_R0.5", "baseline")]
    print(f"\n== {bench.upper()} (n={len(ids)}) ==")
    print(f"{'A':<32}{'B':<24}{'acc A':>7}{'acc B':>7}{'onlyA':>7}{'onlyB':>7}{'p':>8}")
    for a, b in pairs:
        if a in acc and b in acc:
            b01, b10, p = mcnemar(acc[a], acc[b])
            print(f"{a:<32}{b:<24}{np.mean(acc[a])*100:>6.1f}%{np.mean(acc[b])*100:>6.1f}%{b01:>7}{b10:>7}{p:>8.3f}")
''')

md(r'''
## 2. Headroom: a perfect per-example router over the fixed-K runs we already have
**Fidelity oracle:** for each example, pick the smallest K whose answer equals the baseline answer.
Its accuracy equals baseline by construction, so the only question is how cheap it gets. If that is close
to FastV K=2's cost, adaptive K has a lot of room. If the oracle itself needs K=17 often, there is little
to gain.
''')
code(r'''
for bench, (runs, ids) in DATASETS.items():
  for R in ratios(runs):
    ks = fixed_k_runs(runs, R)
    if 2 not in ks:
        continue
    base = runs["baseline"]
    picks, flops = [], []
    for i in ids:
        for k, name in ks.items():
            if runs[name][i]["pred"] == base[i]["pred"]:
                picks.append(k); flops.append(runs[name][i]["rel_flops"]); break
    print(f"\n{bench.upper()} R={R}: fixed-K runs available {list(ks)}")
    print(f"  fidelity oracle: mean FLOPs {np.mean(flops)*100:.1f}%  (FastV K=2 alone: "
          f"{np.mean([runs[ks[2]][i]['rel_flops'] for i in ids])*100:.1f}%)")
    print(f"  oracle K distribution: {dict(sorted(Counter(picks).items()))}")
    print(f"  -> {sum(k == 2 for k in picks)/len(picks)*100:.1f}% of examples are already fine at K=2; "
          f"the router's job is to find the other {sum(k != 2 for k in picks)} examples")
''')

md(r'''
## 3. Does the score predict which examples early pruning breaks?
Target: FastV K=2 changes the baseline answer. Feature: the InfoPrune focus score at each layer
(from the argmax run, which records scores for every layer 2–16). Hypothesis: diffuse attention (low
score) means the example is hurt by early pruning, so we test `-score`. AUC 0.5 = no information.
Layer 2 is the pre-registered test, because it is the first layer threshold mode looks at; other layers
are exploratory (multiple comparisons).
''')
code(r'''
def auc(pos, neg):
    pos, neg = np.asarray(pos, dtype=float), np.asarray(neg, dtype=float)
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))

def auc_ci(pos, neg, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    pos, neg = np.asarray(pos), np.asarray(neg)
    bs = [auc(rng.choice(pos, len(pos)), rng.choice(neg, len(neg))) for _ in range(n_boot)]
    return np.percentile(bs, [2.5, 97.5])

for bench, (runs, ids) in DATASETS.items():
  arg = find(runs, r"infoprune_argmax.*")      # its scores are computed before pruning, so they don't depend on R
  for R in ratios(runs):
    if run_name(runs, 2, R) is None:
        continue
    k2 = runs[run_name(runs, 2, R)]; base = runs["baseline"]
    flip = {i: k2[i]["pred"] != base[i]["pred"] for i in ids}
    print(f"\n{bench.upper()} R={R}: K=2 changes {sum(flip.values())}/{len(ids)} baseline answers")
    layers = sorted(int(l) for l in runs[arg][ids[0]]["scores"])
    for l in layers:
        feat = {i: -runs[arg][i]["scores"][str(l)] for i in ids}
        pos = [feat[i] for i in ids if flip[i]]; neg = [feat[i] for i in ids if not flip[i]]
        a = auc(pos, neg)
        ci = auc_ci(pos, neg) if l == layers[0] else None
        tag = f"  95% CI [{ci[0]:.2f}, {ci[1]:.2f}]  <- pre-registered" if ci is not None else ""
        print(f"  layer {l:2d}: AUC {a:.3f}{tag}")
''')

md(r'''
## 4. Offline router: K=2 vs K=17 by the layer-2 score, against random routing at the same rate
Examples with score ≥ t go to K=2, the rest to K=17. Random routing with the same K=2 fraction has an
expected accuracy that is just the linear mix of the two. If the score carries signal, the router beats
that line at every budget. (The router uses InfoPrune's score but FastV's pruning for execution, so this
tests the *signal*, not the full method.)
''')
code(r'''
LAYER = 2
for bench, (runs, ids) in DATASETS.items():
  arg = find(runs, r"infoprune_argmax.*")
  for R in ratios(runs):
    if run_name(runs, 2, R) is None or run_name(runs, 17, R) is None:
        continue
    a2, a17 = runs[run_name(runs, 2, R)], runs[run_name(runs, 17, R)]
    s = {i: runs[arg][i]["scores"][str(LAYER)] for i in ids}
    acc2 = np.mean([is_correct(bench, a2[i]) for i in ids]); acc17 = np.mean([is_correct(bench, a17[i]) for i in ids])
    f2 = np.mean([a2[i]["rel_flops"] for i in ids]); f17 = np.mean([a17[i]["rel_flops"] for i in ids])
    print(f"\n{bench.upper()} R={R}: K=2 acc {acc2*100:.1f}% @ {f2*100:.1f}% FLOPs | K=17 acc {acc17*100:.1f}% @ {f17*100:.1f}% FLOPs")
    print(f"{'frac->K2':>9}{'router acc':>12}{'random acc':>12}{'gain':>8}{'FLOPs':>8}")
    for q in (0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3):
        t = np.quantile(list(s.values()), 1 - q)
        route = {i: (a2 if s[i] >= t else a17)[i] for i in ids}
        frac = np.mean([s[i] >= t for i in ids])
        racc = np.mean([is_correct(bench, route[i]) for i in ids])
        rand = frac * acc2 + (1 - frac) * acc17
        print(f"{frac:>9.2f}{racc*100:>11.1f}%{rand*100:>11.1f}%{(racc-rand)*100:>+7.1f}{np.mean([route[i]['rel_flops'] for i in ids])*100:>7.1f}%")
''')

md(r'''
## 5. Dataset-level InfoScore per layer (the original, layer-level form)
`InfoScore(l) = H_data(l) / mean_i H_img(l, i)`. This is the quantity the Segmentation-From-Attention
paper ranks layers by. If it peaks at the earliest layers, InfoScore independently recovers FastV's
empirically chosen early K.
''')
code(r'''
for p in sorted(CACHE.glob("calib_*.json")):
    c = json.load(open(p))
    rows = []
    for l, hd in sorted(((int(k), v) for k, v in c["H_data"].items())):
        rows.append((l, hd, float(np.mean(c["calib_h"][str(l)])), hd / float(np.mean(c["calib_h"][str(l)]))))
    best = max(rows, key=lambda r: r[3])
    print(f"\n{p.name}: peak InfoScore at layer {best[0]} ({best[3]:.3f})")
    for l, hd, mh, s in rows:
        print(f"  layer {l:2d}  H_data {hd:.3f}  mean H_img {mh:.3f}  InfoScore {s:.3f}  {'#' * int(round((s - 1) * 40))}")
''')

md(r'''
## 6. What drives the chosen layer: the image or the question?
POPE asks about 6 questions per image. If the chosen layer is an image property, questions about the
same image should mostly agree (high share of variance between images). If the variance is mostly
within images, the choice is driven by the question text or is noise.
''')
code(r'''
pope_img = {}
pope_dir = DATA / "POPE/output/coco"
if pope_dir.exists() and "pope" in DATASETS:
    for f in pope_dir.glob("coco_pope_*.json"):
        split = f.stem.split("_")[2]
        for line in open(f):
            if line.strip():
                l = json.loads(line)
                pope_img[f"{split}:{l['question_id']}"] = l["image"]

    runs, ids = DATASETS["pope"]
    for name in [n for n in runs if n.startswith("infoprune")]:
        layer = {i: runs[name][i]["chosen_layer"] for i in ids}
        groups = defaultdict(list)
        for i in ids:
            groups[pope_img[i]].append(layer[i])
        allv = np.array(list(layer.values()), dtype=float)
        ss_tot = ((allv - allv.mean()) ** 2).sum()
        ss_between = sum(len(v) * (np.mean(v) - allv.mean()) ** 2 for v in groups.values())
        # permutation baseline: same group sizes, shuffled layers
        rng = np.random.default_rng(0); sizes = [len(v) for v in groups.values()]; perm = []
        for _ in range(1000):
            sh = rng.permutation(allv); k = 0; sb = 0
            for n in sizes:
                sb += n * (sh[k:k + n].mean() - allv.mean()) ** 2; k += n
            perm.append(sb / ss_tot)
        eta2 = ss_between / ss_tot
        print(f"{name}: share of chosen-layer variance between images = {eta2:.2f} "
              f"(shuffled baseline {np.mean(perm):.2f}, p={np.mean(np.array(perm) >= eta2):.3f}; {len(groups)} images)")
else:
    print("POPE question files not found; skip.")

try:
    from scipy.stats import kruskal
    runs, ids = DATASETS["pope"]
    for name in [n for n in runs if n.startswith("infoprune")]:
        by = defaultdict(list)
        for i in ids:
            by[i.split(":")[0]].append(runs[name][i]["chosen_layer"])
        print(f"{name}: Kruskal-Wallis across POPE splits p={kruskal(*by.values()).pvalue:.3f}")
except ImportError:
    pass
''')

md(r'''
## 7. Accuracy vs K (FastV sweep) against the layer-level InfoScore curve
FastV at K ranks tokens with layer K−1's attention, so an InfoScore peak at layer *l* corresponds to K = *l*+1.
"Knee K" = smallest K whose accuracy is within 1 point of the best K at that R (a rough stand-in for the
point after which more layers stop helping; with n=300–450 the curve is noisy, so read it with §1's p-values).
''')
code(r'''
for bench, (runs, ids) in DATASETS.items():
    base_acc = np.mean([is_correct(bench, runs["baseline"][i]) for i in ids])
    print(f"\n== {bench.upper()} (baseline {base_acc*100:.1f}%) ==")
    for R in ratios(runs):
        ks = {k: n for k, n in fixed_k_runs(runs, R).items() if k != 32}
        if len(ks) < 2:
            continue
        acc = {k: np.mean([is_correct(bench, runs[n][i]) for i in ids]) for k, n in ks.items()}
        fl = {k: np.mean([runs[n][i]["rel_flops"] for i in ids]) for k, n in ks.items()}
        best = max(acc.values())
        knee = min(k for k in acc if acc[k] >= best - 0.01)
        row = "  ".join(f"K{k}:{acc[k]*100:.1f}%({fl[k]*100:.0f}%)" for k in acc)
        print(f"  R={R}: {row}\n         knee K={knee}")
    for p in sorted(CACHE.glob(f"calib_{bench}_L0-31_*.json")):
        c = json.load(open(p))
        sc = {int(l): hd / float(np.mean(c["calib_h"][l])) for l, hd in c["H_data"].items()}
        peak = max(sc, key=sc.get)
        print(f"  full-depth InfoScore peak: layer {peak} -> FastV K={peak + 1}")
        print("  " + "  ".join(f"L{l}:{v:.2f}" for l, v in sorted(sc.items())))
''')

md(r'''
## Reading the output
- **§3 AUC near 0.5 and §4 gains near 0** → the per-image score carries no usable information about which
  examples need more layers. Stop tuning τ; the score itself needs to change (see §5 and the
  localization check in the next-steps plan).
- **§3 AUC clearly above 0.5 and §4 gains positive at most budgets** → the signal is real and the current
  threshold rule is just a poor way of using it. Next: a better rule, larger N, and the shuffled-K* control.
- **§2** tells you the ceiling: how much FLOPs saving a perfect router would buy at baseline accuracy.
''')

nb = {"cells": [({"cell_type": "markdown", "metadata": {}, "source": s.splitlines(keepends=True)} if t == "markdown" else
                 {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.splitlines(keepends=True)})
                for t, s in cells],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.12"}},
      "nbformat": 4, "nbformat_minor": 5}
for i, c in enumerate(nb["cells"]):
    c["id"] = f"cell-{i:02d}"
json.dump(nb, open("../../notebooks/infoprune-offline-analysis.ipynb", "w"), indent=1)
print(len(cells), "cells")
