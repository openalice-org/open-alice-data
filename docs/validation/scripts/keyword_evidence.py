"""Where did the old build get its quantum term? (explains why the new rules drop 13,709 earlier works)

For the works the earlier build included and the current run retrieves but excludes, checks whether a
quantum term (patterns from corpus/config/sweden_quantum_paper_filter.json) was in the title or
abstract, or only in the OpenAlex keywords. Needs the previously deployed corpus (legacy data).
"""
import csv, json, re, collections
H=str(__import__("pathlib").Path(__file__).resolve().parents[3].parent) + "/"
cfg=json.load(open(H+"open-alice-data/corpus/config/sweden_quantum_paper_filter.json"))
pats=[re.compile(p,re.I) for p in cfg["paper_level_term_patterns"]]
hit=lambda s: any(p.search(s or "") for p in pats)
def abstract(inv):
    if not inv: return ""
    pos=[(p,w) for w,ps in inv.items() for p in ps]; return " ".join(w for _,w in sorted(pos))
newids={json.loads(l)["id"] for l in open(H+"open-alice-data/corpus/data/sweden_quantum_works.jsonl",encoding="utf-8")}
cand=set()
for r in csv.DictReader(open(H+"open-alice-data/corpus/data/sweden_quantum_candidate_audit.csv",encoding="utf-8")):
    cand.add("https://openalex.org/"+r["work_id"] if not r["work_id"].startswith("http") else r["work_id"])
c=collections.Counter(); seen=set()
for l in open(H+"Qsip/SPRINT 13/data/all_universities_works.jsonl",encoding="utf-8"):
    w=json.loads(l); i=w["id"]
    if i in seen or i in newids or i not in cand: continue
    seen.add(i)
    text=hit((w.get("display_name") or "")+"\n"+abstract(w.get("abstract_inverted_index")))
    kw=hit("\n".join(k.get("display_name") or "" for k in (w.get("keywords") or []) if isinstance(k,dict)))
    c[("title/abstract has a quantum term" if text else ("ONLY the old keywords had one" if kw else "no quantum term anywhere (old build)"))]+=1
n=sum(c.values()); print("old-only works retrieved but excluded now:",n)
for k,v in c.most_common(): print(f"  {v:6d} {v/n:6.1%}  {k}")
