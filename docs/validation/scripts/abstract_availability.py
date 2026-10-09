import csv, json, collections, glob
H=str(__import__("pathlib").Path(__file__).resolve().parents[3].parent) + "/"
SP=str(__import__("pathlib").Path(__file__).resolve().parents[1] / "data") + "/"
newids={json.loads(l)["id"] for l in open(H+"open-alice-data/corpus/data/sweden_quantum_works.jsonl",encoding="utf-8")}
cand={}
for r in csv.DictReader(open(H+"open-alice-data/corpus/data/sweden_quantum_candidate_audit.csv",encoding="utf-8")):
    wid="https://openalex.org/"+r["work_id"] if not r["work_id"].startswith("http") else r["work_id"]
    cand[wid]=r
old={}
for l in open(H+"Qsip/SPRINT 13/data/all_universities_works.jsonl",encoding="utf-8"):
    w=json.loads(l); i=w["id"]
    if i in old or i in newids or i not in cand: continue
    old[i]=bool(w.get("abstract_inverted_index"))
print("old-only & candidate-excluded works:",len(old))
newabs={}
for l in open(H+"open-alice-data/corpus/data/sweden_quantum_candidate_union.jsonl",encoding="utf-8"):
    w=json.loads(l)
    if w["id"] in old: newabs[w["id"]]=(bool(w.get("abstract_inverted_index")), len(w.get("keywords") or []))
c=collections.Counter((old[i], newabs.get(i,(None,))[0]) for i in old)
print("(old had abstract, new has abstract) -> count")
for k,v in sorted(c.items(),key=lambda kv:-kv[1]): print(f"   {str(k):16s} {v:6d} {v/len(old):6.1%}")
lost=[i for i in old if old[i] and newabs.get(i,(None,))[0] is False]
print("LOST their abstract between builds:",len(lost),f"({len(lost)/len(old):.1%} of retrieved-but-excluded old-only works)")
# of the sampled items labeled CORE by haiku, how many lost their abstract?
mp=json.load(open(SP+"sample_map.json")); hk={g["item"]:g["label"] for p in glob.glob(SP+"labels_haiku/batch_*.json") for g in json.load(open(p))}
core=[m["work_id"] for it,m in mp.items() if m["group"]=="old_only" and m["work_id"] in old and hk[it]=="CORE"]
print("sampled retrieved-but-excluded CORE works:",len(core),"| of which lost abstract:",sum(1 for i in core if i in lost))
