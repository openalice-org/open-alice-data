import json, glob, collections, math
H=str(__import__("pathlib").Path(__file__).resolve().parents[3].parent) + "/"
SP=str(__import__("pathlib").Path(__file__).resolve().parents[1] / "data") + "/"
new=json.load(open(H+"open-alice-data/pipeline/data/paper_counts_geo.json")); old=json.load(open(H+"open-alice/public/data/paper_counts_geo.json")) if False else None
import subprocess
old=json.loads(subprocess.check_output(["git","-C",H+"open-alice","show","main:public/data/paper_counts_geo.json"]))
print("keys:",list(new))
print("TOTAL distinct papers  deployed:",old.get("total"),"new:",new.get("total"))
def table(key,label):
    o,n=old[key],new[key]; ks=sorted(set(o)|set(n),key=lambda k:-(o.get(k,0) if not isinstance(o.get(k),dict) else o[k].get("papers",0)))
    return o,n,ks
o,n=old["by_country"],new["by_country"]
g=lambda d,k:(d.get(k,{}).get("papers") if isinstance(d.get(k),dict) else d.get(k,0)) or 0
print("\nBy country (papers): deployed -> new")
for k in sorted(set(o)|set(n),key=lambda k:-g(o,k)):
    a,b=g(o,k),g(n,k); print(f"  {k}: {a:6d} -> {b:6d}  ({(b-a)/a:+.0%})" if a else f"  {k}: {a} -> {b}")
oi,ni=old["by_institution"],new["by_institution"]
ks=[k for k in oi if k in ni]; a=[g(oi,k) for k in ks]; b=[g(ni,k) for k in ks]
def rank(v): 
    r=sorted(range(len(v)),key=lambda i:-v[i]); out=[0]*len(v)
    for pos,i in enumerate(r): out[i]=pos
    return out
ra,rb=rank(a),rank(b); n_=len(ks); rho=1-6*sum((x-y)**2 for x,y in zip(ra,rb))/(n_*(n_*n_-1))
print(f"\nInstitutions in both: {n_} | Spearman rank correlation of paper counts: {rho:.3f}")
top=lambda v:set(ks[i] for i in sorted(range(len(v)),key=lambda i:-v[i])[:10])
print("top-10 institutions identical sets:",top(a)==top(b),"| overlap",len(top(a)&top(b)),"/10")
chg=sorted(((abs(y-x)/x if x else 9,k,x,y) for k,x,y in zip(ks,a,b)),reverse=True)[:5]
print("largest relative institution changes:",[(k,x,y) for _,k,x,y in chg])
dm=lambda d:{k:v for k,v in d.items()}
print("\nDomain mix by country (computing/communication/sensing), deployed vs new:")
for k in ("SE","DK","FI","NO"):
    ov,nv=old["domain_by_country"].get(k),new["domain_by_country"].get(k); print(" ",k,json.dumps(ov)[:110],"|",json.dumps(nv)[:110])
# ---- were the dropped works good drops? labels of OLD-ONLY sample joined with the live check cause
mp=json.load(open(SP+"sample_map.json")); cause=json.load(open(SP+"live_all_result.json"))
hk={g["item"]:g["label"] for p in glob.glob(SP+"labels_haiku/batch_*.json") for g in json.load(open(p))}
ek={x["item"]:x["label"] for x in json.load(open(SP+"eikos_labels.json"))}
rows=collections.defaultdict(list)
for it,m in mp.items():
    if m["group"]!="old_only": continue
    c=cause.get(m["work_id"]); key=("NOT retrieved: "+c.split(" (")[0].split(":")[0][:48]) if c else "retrieved but EXCLUDED by new rules"
    rows[key].append(it)
print("\nOLD-ONLY sample (n=200): were the drops good? share judged NOT / CORE  (Haiku | Eikos)")
for k,its in sorted(rows.items(),key=lambda kv:-len(kv[1])):
    f=lambda lab,L:sum(lab[i]==L for i in its)/len(its)
    print(f"  n={len(its):3d}  NOT {f(hk,'NOT'):4.0%}|{f(ek,'NOT'):4.0%}   CORE {f(hk,'CORE'):4.0%}|{f(ek,'CORE'):4.0%}   {k}")
