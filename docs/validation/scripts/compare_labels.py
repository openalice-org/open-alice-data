import glob, json, collections
SP=str(__import__("pathlib").Path(__file__).resolve().parents[1] / "data") + "/"
L=["CORE","PERIPHERAL","NOT"]
mp=json.load(open(SP+"sample_map.json"))
haiku={g["item"]:g["label"] for p in sorted(glob.glob(SP+"labels_haiku/batch_*.json")) for g in json.load(open(p))}
sonnet={g["item"]:g["label"] for g in json.load(open(SP+"labels_sonnet/batch_01.json"))}
eik={}
for x in json.load(open(SP+"eikos_labels.json")): eik[x["item"]]=x["label"]
def kappa(a,b,keys):
    n=len(keys); po=sum(a[k]==b[k] for k in keys)/n
    pa=collections.Counter(a[k] for k in keys); pb=collections.Counter(b[k] for k in keys)
    pe=sum((pa[l]/n)*(pb[l]/n) for l in L)
    return po,(po-pe)/(1-pe) if pe<1 else float("nan")
def conf(a,b,keys,an,bn):
    c=collections.Counter((a[k],b[k]) for k in keys)
    print(f"  rows={an} cols={bn}   "+"".join(f"{l:>11s}" for l in L))
    for r in L: print(f"  {r:>14s}      "+"".join(f"{c[(r,cc)]:>11d}" for cc in L))
def mix(lab,title):
    print(f"\n{title}  (n labeled={len(lab)})")
    for g in ("old_only","new_only","shared"):
        ks=[k for k in lab if mp[k]["group"]==g]; c=collections.Counter(lab[k] for k in ks)
        print(f"  {g:9s} n={len(ks):3d}  "+"  ".join(f"{l}={c[l]/len(ks):5.1%}" for l in L))
mix(haiku,"HAIKU label mix by group")
if eik: mix(eik,"EIKOS label mix by group")
ks=[k for k in sonnet if k in haiku]
po,kp=kappa(haiku,sonnet,ks); print(f"\nHAIKU vs SONNET on {len(ks)} items: agreement={po:.1%} kappa={kp:.2f}"); conf(haiku,sonnet,ks,"haiku","sonnet")
ks=[k for k in eik if k in haiku]
if ks:
    po,kp=kappa(haiku,eik,ks); print(f"\nHAIKU vs EIKOS on {len(ks)} items: agreement={po:.1%} kappa={kp:.2f}"); conf(haiku,eik,ks,"haiku","eikos")
    k4=[k for k in ks if k in sonnet]
    if k4:
        po,kp=kappa(sonnet,eik,k4); print(f"SONNET vs EIKOS on {len(k4)}: agreement={po:.1%} kappa={kp:.2f}")
# binary view that matters for the corpus question: is it quantum research at all (CORE or PERIPHERAL) vs NOT
def binar(lab): return {k:("Q" if v!="NOT" else "NOT") for k,v in lab.items()}
print("\nShare judged NOT quantum research, by group (Haiku | Eikos):")
for g in ("old_only","new_only","shared"):
    ks=[k for k in haiku if mp[k]["group"]==g]
    h=sum(haiku[k]=="NOT" for k in ks)/len(ks)
    ek=[k for k in ks if k in eik]; e=(sum(eik[k]=="NOT" for k in ek)/len(ek)) if ek else float("nan")
    print(f"  {g:9s} haiku={h:5.1%}   eikos={e:5.1%}")
