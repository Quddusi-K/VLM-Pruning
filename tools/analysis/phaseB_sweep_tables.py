import json, math, re, sys, numpy as np
from collections import Counter
from pathlib import Path
D = Path(sys.argv[1])
def mcnemar(a,b):
    b01=sum(x and not y for x,y in zip(a,b)); b10=sum(y and not x for x,y in zip(a,b)); n=b01+b10
    p=1.0 if n==0 else min(1.0,2*sum(math.comb(n,i) for i in range(min(b01,b10)+1))/2**n); return b01,b10,p
BENCH=["pope","mme","gqa","textvqa"]
R={}
for b in BENCH:
    R[b]={p.stem.split("__",1)[1]:[json.loads(l) for l in open(p)] for p in sorted(D.glob(f"{b}__*.jsonl"))}
def key(n):
    m=re.match(r"(baseline|fixed_E(\d+)|lens_t([\d.]+)|cf_t([\d.]+))",n)
    fam={"b":0,"f":1,"l":2,"c":3}[n[0]]
    num=float(m.group(2) or m.group(3) or m.group(4) or 0)
    return (("uni" in n), fam, -num if fam>1 else num, n)
def stats(b,n):
    recs=R[b][n]; base=R[b]["baseline"]; bp={r["id"]:r["pred"] for r in base}
    acc=np.mean([r["correct"] for r in recs]); ag=np.mean([bp[r["id"]]==r["pred"] for r in recs])
    fl=np.mean([r["rel_flops"] for r in recs]); ex=[r["exit_layer"] if r["exit_layer"] is not None else 32 for r in recs]
    bc=[ (r["correct"]>=1 if isinstance(r["correct"],float) else bool(r["correct"])) for r in base]
    c=[ (r["correct"]>=1 if isinstance(r["correct"],float) else bool(r["correct"])) for r in recs]
    b01,b10,p=mcnemar(c,bc)
    pf=np.mean([r["t_prefill"] for r in recs])*1000
    return acc,ag,fl,np.mean(ex),pf,b01,b10,p,ex
names=sorted(set.intersection(*(set(R[b]) for b in BENCH)),key=key)
print("acc / agree / FLOPs%  (prefill ms)   -- 4 benchmarks")
print(f"{'method':<32}"+"".join(f"{b:>26}" for b in BENCH))
for n in names:
    row=f"{n:<32}"
    for b in BENCH:
        acc,ag,fl,ex,pf,b01,b10,p,_=stats(b,n)
        row+=f"{acc*100:6.1f}/{ag*100:5.1f}/{fl*100:5.1f} ({pf:4.0f})"
    print(row)
print("\nmean exit layer (32 = no exit) and share never exiting")
print(f"{'method':<32}"+"".join(f"{b:>18}" for b in BENCH))
for n in names:
    if n.startswith(("lens","cf")):
        row=f"{n:<32}"
        for b in BENCH:
            *_,ex=stats(b,n); row+=f"{np.mean(ex):9.1f} {np.mean(np.array(ex)==32)*100:6.1f}%"
        print(row)
print("\nMcNemar vs baseline (onlyMethod/onlyBase, p) -- * = p<0.05")
print(f"{'method':<32}"+"".join(f"{b:>16}" for b in BENCH))
for n in names:
    if n=="baseline": continue
    row=f"{n:<32}"
    for b in BENCH:
        acc,ag,fl,ex,pf,b01,b10,p,_=stats(b,n); row+=f"{b01:>5}/{b10:<4}{p:5.3f}{'*' if p<0.05 else ' '}"
    print(row)
# pooled avg FLOPs and min-agreement across benchmarks
print("\nAcross all 4 benchmarks: mean FLOPs, mean agreement, min agreement, mean acc drop (pts)")
for n in names:
    if n=="baseline": continue
    ss=[stats(b,n) for b in BENCH]; base=[stats(b,"baseline")[0] for b in BENCH]
    print(f"{n:<32} FLOPs {np.mean([s[2] for s in ss])*100:5.1f}%  agree mean {np.mean([s[1] for s in ss])*100:5.1f}  min {min(s[1] for s in ss)*100:5.1f}  acc drop {np.mean([b-s[0] for s,b in zip(ss,base)])*100:+5.2f}")
