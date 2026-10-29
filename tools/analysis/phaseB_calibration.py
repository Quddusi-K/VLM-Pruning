"""Calibration protocol for the one-global-threshold setting (no benchmark tuning).

Two modes, both over the Phase B sweep files (full-answer agreement with the unpruned model):
  split    : for each of S random seeds, hold out a calibration fraction of every benchmark; pool the calibration
             records (mean of per-benchmark agreement, so POPE does not dominate); pick, per method family, the
             cheapest setting (lowest mean FLOPs) whose pooled calibration agreement >= target T; report that
             setting's accuracy / agreement / FLOPs / McNemar on the remaining evaluation records.
  loo      : leave-one-benchmark-out: choose the setting on the other three benchmarks (all records), evaluate on the
             held-out one.
Families: fixed_E* (VTW-style dataset-level exit chosen by the same protocol), lens_t*, cf_t*.
usage: python -I phaseB_calibration.py results/phaseB [--targets 0.95,0.97,0.98] [--frac 0.2] [--seeds 50]
"""
import json, math, re, sys, argparse, numpy as np
from collections import Counter, defaultdict
from pathlib import Path
ap = argparse.ArgumentParser(); ap.add_argument("dir"); ap.add_argument("--targets", default="0.95,0.97,0.98")
ap.add_argument("--frac", type=float, default=0.2); ap.add_argument("--seeds", type=int, default=50)
A = ap.parse_args(); D = Path(A.dir); BENCH = ["pope", "mme", "gqa", "textvqa"]
TARGETS = [float(t) for t in A.targets.split(",")]
R = {b: {p.stem.split("__", 1)[1]: [json.loads(l) for l in open(p)] for p in sorted(D.glob(f"{b}__*.jsonl"))} for b in BENCH}
def is_grid(n): return (n.startswith(("fixed_E", "lens_t", "cf_t")) and "uni" not in n and "+" not in n and n != "cf_t0.0501_m2_e12")
names = sorted(n for n in set.intersection(*(set(R[b]) for b in BENCH)) if is_grid(n))
FAM = {"fixed": [n for n in names if n.startswith("fixed")], "lens": [n for n in names if n.startswith("lens")], "cf": [n for n in names if n.startswith("cf")]}
def corr(r): return r["correct"] >= 1 if isinstance(r["correct"], float) else bool(r["correct"])
def occ_keys(recs):   # (id, k-th occurrence): MME ids repeat; files may be in a different item order after a resumed run
    seen=Counter(); out=[]
    for r in recs: out.append((r["id"],seen[r["id"]])); seen[r["id"]]+=1
    return out
def aligned(recs, base):   # method records reordered to the baseline's (id, occurrence) order
    d=dict(zip(occ_keys(recs),recs)); bk=occ_keys(base); assert set(d)==set(bk), "record sets differ"
    return [d[k] for k in bk]
# per-record arrays (aligned by position with baseline; MME ids repeat)
X = {}
for b in BENCH:
    base = R[b]["baseline"]
    for n in names + ["baseline"]:
        recs = aligned(R[b][n], base)
        X[b, n] = dict(agree=np.array([x["pred"] == y["pred"] for x, y in zip(recs, base)], float),
                       acc=np.array([r["correct"] for r in recs], float), c=np.array([corr(r) for r in recs], bool),
                       fl=np.array([r["rel_flops"] for r in recs], float))
def mcnemar(a, b):
    b01 = int(np.sum(a & ~b)); b10 = int(np.sum(b & ~a)); n = b01 + b10
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b01, b10) + 1)) / 2 ** n); return b01, b10, p
def pooled(n, idx):   # mean over benchmarks of per-benchmark means on index sets idx[b]
    ag = np.mean([X[b, n]["agree"][idx[b]].mean() for b in BENCH]); fl = np.mean([X[b, n]["fl"][idx[b]].mean() for b in BENCH]); return ag, fl
def choose(fam, idx, T):
    cands = [(pooled(n, idx), n) for n in FAM[fam]]
    ok = [(fl, n) for (ag, fl), n in cands if ag >= T]
    if ok: return min(ok)[1], True
    return max(cands)[1], False          # nothing reaches T: most conservative (highest agreement) setting
def evaluate(n, idx):
    out = {}
    for b in BENCH:
        i = idx[b]; x = X[b, n]; base = X[b, "baseline"]
        b01, b10, p = mcnemar(x["c"][i], base["c"][i])
        out[b] = dict(acc=x["acc"][i].mean(), dacc=(x["acc"][i] - base["acc"][i]).mean(), agree=x["agree"][i].mean(), fl=x["fl"][i].mean(), p=p)
    return out
print(f"grid: fixed {FAM['fixed']}\n      lens {FAM['lens']}\n      cf {FAM['cf']}")
print(f"records: " + ", ".join(f"{b} {len(R[b]['baseline'])}" for b in BENCH))

print(f"\n=== split mode: calibration fraction {A.frac:.0%} of every benchmark, {A.seeds} seeds; evaluation on the rest ===")
for T in TARGETS:
    print(f"\n--- target pooled agreement >= {T:.0%} ---")
    print(f"{'family':<7}{'chosen (seeds)':<46}{'met':>5}{'FLOPs':>7}{'agree':>7}{'minAg':>7}{'accΔ':>7}  per-benchmark acc Δ (share of seeds with p<0.05)")
    for fam in FAM:
        picks = Counter(); met = 0; ev = defaultdict(list)
        for s in range(A.seeds):
            rng = np.random.default_rng(s)
            cal = {b: np.sort(rng.permutation(len(R[b]["baseline"]))[:int(round(A.frac * len(R[b]["baseline"])))]) for b in BENCH}
            evl = {b: np.setdiff1d(np.arange(len(R[b]["baseline"])), cal[b]) for b in BENCH}
            n, ok = choose(fam, cal, T); picks[n] += 1; met += ok
            e = evaluate(n, evl)
            ev["fl"].append(np.mean([e[b]["fl"] for b in BENCH])); ev["ag"].append(np.mean([e[b]["agree"] for b in BENCH]))
            ev["min"].append(min(e[b]["agree"] for b in BENCH)); ev["dacc"].append(np.mean([e[b]["dacc"] for b in BENCH]))
            for b in BENCH: ev["d_" + b].append(e[b]["dacc"]); ev["sig_" + b].append(e[b]["p"] < 0.05)
        pk = ", ".join(f"{k.replace('_m1_e1','').replace('_m2_e12','')}×{v}" for k, v in picks.most_common())
        print(f"{fam:<7}{pk:<46}{met:>3}/{A.seeds:<2}{np.mean(ev['fl'])*100:>6.1f}%{np.mean(ev['ag'])*100:>6.1f}%{np.mean(ev['min'])*100:>6.1f}%{np.mean(ev['dacc'])*100:>+6.2f}  "
              + "  ".join(f"{b} {np.mean(ev['d_'+b])*100:+.1f} ({np.mean(ev['sig_'+b])*100:.0f}%)" for b in BENCH))

print(f"\n=== leave-one-benchmark-out: choose on the other three (all records), evaluate on the held-out benchmark ===")
for T in TARGETS:
    print(f"\n--- target pooled agreement >= {T:.0%} ---")
    print(f"{'held-out':<9}{'family':<7}{'chosen':<22}{'met':>4}{'acc':>7}{'accΔ':>7}{'agree':>7}{'FLOPs':>7}{'p':>7}")
    for hb in BENCH:
        others = {b: np.arange(len(R[b]["baseline"])) for b in BENCH if b != hb}
        for fam in FAM:
            cands = [(np.mean([X[b, n]["agree"].mean() for b in others]), np.mean([X[b, n]["fl"].mean() for b in others]), n) for n in FAM[fam]]
            ok = [(fl, n) for ag, fl, n in cands if ag >= T]
            n, met = (min(ok)[1], True) if ok else (max(cands)[2], False)
            e = evaluate(n, {b: np.arange(len(R[b]["baseline"])) for b in BENCH})[hb]
            print(f"{hb:<9}{fam:<7}{n.replace('_m1_e1','').replace('_m2_e12',''):<22}{'y' if met else 'n':>4}{e['acc']*100:>6.1f}%{e['dacc']*100:>+6.1f}{e['agree']*100:>6.1f}%{e['fl']*100:>6.1f}%{e['p']:>7.3f}{'*' if e['p']<0.05 else ''}")
