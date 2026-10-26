import json, sys, numpy as np
from pathlib import Path
from collections import defaultdict
from scipy.stats import spearmanr
B=Path(sys.argv[1]); A=Path(sys.argv[2])
for b in ("pope","gqa","textvqa"):
    prof={json.loads(l)["id"]:json.loads(l) for l in open(A/f"profile_{b}.jsonl")}
    def stable(p):
        pres=np.array(p["preserved"],bool); tail=np.flip(np.cumprod(np.flip(pres)))
        return int(np.argmax(tail))
    for n in ("lens_t0.05_m1_e1","cf_t0.05_m2_e12"):
        recs=[json.loads(l) for l in open(B/f"{b}__{n}.jsonl")]
        x=[r["exit_layer"] if r["exit_layer"] is not None else 32 for r in recs]
        y=[stable(prof[r["id"]]) for r in recs]
        rho,p=spearmanr(x,y)
        # fraction of examples where exit >= oracle stable layer (i.e., safe by oracle)
        safe=np.mean([xi>=yi for xi,yi in zip(x,y)])
        print(f"{b:<8} {n:<20} Spearman(exit, oracle stable E) = {rho:+.2f} (p={p:.1e}) | exit >= oracle for {safe*100:.1f}%")
# exit by GQA structural type and MME category
gq={json.loads(l)["id"]:json.loads(l) for l in open(A/"profile_gqa.jsonl")}
recs=[json.loads(l) for l in open(B/"gqa__lens_t0.05_m1_e1.jsonl")]
by=defaultdict(list)
for r in recs: by[gq[r["id"]]["structural"]].append(r["exit_layer"] or 32)
print("\nGQA lens exit by question type:", {k:(round(np.mean(v),1),len(v)) for k,v in sorted(by.items())})
mm=[json.loads(l) for l in open(A/"profile_mme.jsonl")]
recs=[json.loads(l) for l in open(B/"mme__lens_t0.05_m1_e1.jsonl")]
by=defaultdict(list)
for p,r in zip(mm,recs): by[p["category"]].append(r["exit_layer"] or 32)
print("MME lens exit by category:", {k:round(np.mean(v),1) for k,v in sorted(by.items(), key=lambda kv: np.mean(kv[1]))})
