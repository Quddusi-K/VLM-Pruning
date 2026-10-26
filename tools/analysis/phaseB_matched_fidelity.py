import json, sys, numpy as np, re
from pathlib import Path
D=Path(sys.argv[1]); BENCH=["pope","mme","gqa","textvqa"]
R={b:{p.stem.split("__",1)[1]:[json.loads(l) for l in open(p)] for p in sorted(D.glob(f"{b}__*.jsonl"))} for b in BENCH}
# duplicate-id check
for b in BENCH:
    ids=[r["id"] for r in R[b]["baseline"]]; print(b, "records", len(ids), "unique ids", len(set(ids)))
def agree_pos(b,n):  # by position (files written in item order)
    a=R[b][n]; base=R[b]["baseline"]; assert len(a)==len(base)
    return np.mean([x["pred"]==y["pred"] for x,y in zip(a,base)])
def fl(b,n): return np.mean([r["rel_flops"] for r in R[b][n]])
print("\nMME agreement by position:")
for n in sorted(R["mme"]):
    if n!="baseline": print(f"  {n:<32} agree {agree_pos('mme',n)*100:5.1f}  FLOPs {fl('mme',n)*100:5.1f}")
print("\nFixed-E FLOPs needed for the same agreement as each adaptive point (linear interpolation on the fixed-E curve):")
for b in BENCH:
    fixed=sorted([(fl(b,n),agree_pos(b,n)) for n in R[b] if n.startswith("fixed")])
    xs=[f for f,_ in fixed]; ys=[a for _,a in fixed]
    print(f"== {b} == fixed curve: "+"  ".join(f"{a*100:.1f}@{f*100:.0f}" for f,a in fixed))
    for n in sorted(R[b], key=lambda n:-fl(b,n)):
        if n.startswith(("lens","cf")) and "uni" not in n:
            a=agree_pos(b,n); f=fl(b,n)
            if a<=max(ys): f_fix=np.interp(a, ys, xs)   # ys monotone? check
            else: f_fix=float("nan")
            print(f"   {n:<24} agree {a*100:5.1f} @ {f*100:5.1f}%  | fixed-E needs {f_fix*100:5.1f}%  -> saving {(f_fix-f)*100:+5.1f} pt")
