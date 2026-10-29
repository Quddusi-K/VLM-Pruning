"""Probe overhead from measured prefill latency.

For each benchmark, fit t_prefill = a + b * rel_flops per example on the runs whose cost is pure compute (baseline +
fixed_E*), then report each method's mean residual (ms) against that fit: positive = time not explained by FLOPs, i.e.
the cost of the exit probe / pruning step.  Also prints the latency.json table (20 POPE images, 8 tokens).
usage: python -I phaseB_latency_overhead.py results/phaseB2/llava15
"""
import json, sys, numpy as np
from pathlib import Path
D = Path(sys.argv[1]); BENCH = ["pope", "mme", "gqa", "textvqa"]
R = {b: {p.stem.split("__", 1)[1]: [json.loads(l) for l in open(p)] for p in sorted(D.glob(f"{b}__*.jsonl"))} for b in BENCH}
lat = D / "latency.json"
if lat.exists():
    rows = json.load(open(lat))
    print(f"latency.json: {'method':<36}{'prefill ms':>11}{'e2e ms':>9}{'FLOPs':>8}{'mean exit':>10}")
    for name, rs in rows.items():
        pf = np.nanmean([float(r["t_prefill"]) for r in rs]) * 1000; tot = np.mean([float(r["t_total"]) for r in rs]) * 1000
        ex = [r["exit_layer"] for r in rs if r["exit_layer"] is not None]
        print(f"              {name:<36}{pf:>11.0f}{tot:>9.0f}{np.mean([float(r['rel_flops']) for r in rs])*100:>7.1f}%{(np.mean(ex) if ex else float('nan')):>10.1f}")
print("\nprefill-time residual vs the FLOPs fit of baseline+fixed runs (ms; + = probe/pruning overhead)")
print(f"{'method':<34}" + "".join(f"{b:>16}" for b in BENCH) + f"{'mean':>10}")
names = sorted(set.intersection(*(set(R[b]) for b in BENCH)))
fits = {}
for b in BENCH:
    x = []; y = []
    for n, recs in R[b].items():
        if n == "baseline" or n.startswith("fixed_E"):
            x += [r["rel_flops"] for r in recs]; y += [r["t_prefill"] * 1000 for r in recs]
    bcoef, a = np.polyfit(x, y, 1); fits[b] = (a, bcoef)
print(f"{'fit a + b*FLOPs (a ms, b ms/100%)':<34}" + "".join(f"{fits[b][0]:>7.0f}+{fits[b][1]:>7.0f}" for b in BENCH))
for n in names:
    res = []
    for b in BENCH:
        a, bc = fits[b]; recs = R[b][n]
        res.append(np.mean([r["t_prefill"] * 1000 - (a + bc * r["rel_flops"]) for r in recs]))
    print(f"{n:<34}" + "".join(f"{v:>+16.1f}" for v in res) + f"{np.mean(res):>+10.1f}")
