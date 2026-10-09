import csv, json, glob
H=str(__import__("pathlib").Path(__file__).resolve().parents[3].parent) + "/"
SP=str(__import__("pathlib").Path(__file__).resolve().parents[1] / "data") + "/"
mp=json.load(open(SP+"sample_map.json"))
hk={g["item"]:g["label"] for p in glob.glob(SP+"labels_haiku/batch_*.json") for g in json.load(open(p))}
ek={x["item"]:x["label"] for x in json.load(open(SP+"eikos_labels.json"))}
audit={}
for r in csv.DictReader(open(H+"open-alice-data/corpus/data/sweden_quantum_candidate_audit.csv",encoding="utf-8")):
    audit["https://openalex.org/"+r["work_id"] if not r["work_id"].startswith("http") else r["work_id"]]=r
want=[m["work_id"] for it,m in mp.items() if m["group"]=="old_only" and m["work_id"] in audit and hk[it]=="CORE" and ek[it]=="CORE"][:8]
print("both-labelers-CORE, retrieved-but-excluded examples:",len(want))
old={}
for l in open(H+"Qsip/SPRINT 13/data/all_universities_works.jsonl",encoding="utf-8"):
    w=json.loads(l)
    if w["id"] in want and w["id"] not in old: old[w["id"]]=w
newrow={}
for l in open(H+"open-alice-data/corpus/data/sweden_quantum_candidate_union.jsonl",encoding="utf-8"):
    w=json.loads(l)
    if w["id"] in want: newrow[w["id"]]=w
for i in want:
    o,n,a=old[i],newrow[i],audit[i]
    kw=lambda w:[k.get("display_name") for k in (w.get("keywords") or [])][:5]
    print(f"\n{(o.get('display_name') or '')[:90]}")
    print("   OLD terms:",o.get("paper_level_quantum_terms"),"| reasons:",(o.get("quantum_inclusion_reasons") or [""])[0][:60])
    print("   NEW terms:",a["paper_level_quantum_terms"][:60] or "[]","| NEW topics:",[t for t in n.get("matched_topic_ids",[])][:2])
    print("   old keywords:",kw(o)); print("   new keywords:",kw(n))
