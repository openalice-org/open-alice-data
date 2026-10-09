import csv, json, collections, glob
H=str(__import__("pathlib").Path(__file__).resolve().parents[3].parent) + "/"
SP=str(__import__("pathlib").Path(__file__).resolve().parents[1] / "data") + "/"
gates=json.load(open(H+"open-alice-data/corpus/config/sweden_quantum_record_gates.json"))
newids={json.loads(l)["id"] for l in open(H+"open-alice-data/corpus/data/sweden_quantum_works.jsonl",encoding="utf-8")}
rows={}
for r in csv.DictReader(open(H+"open-alice-data/corpus/data/sweden_quantum_candidate_audit.csv",encoding="utf-8")):
    rows["https://openalex.org/"+r["work_id"] if not r["work_id"].startswith("http") else r["work_id"]]=r
old=set()
for l in open(H+"Qsip/SPRINT 13/data/all_universities_works.jsonl",encoding="utf-8"):
    old.add(json.loads(l)["id"])
def cause(r):
    lang=r["language"]; typ=r["work_type"]
    ex=set(gates.get("excluded_work_types") or []); acc=gates.get("accepted_languages") or gates.get("languages")
    if acc is not None and lang not in acc: return "record gate: language not accepted"
    if typ in ex: return f"record gate: work type '{typ}' excluded"
    if r["keyword_only_blocked_primary_domain"]=="true": return "derived-keyword-only domain safeguard"
    return "content rule (broad topic, no paper-level quantum term / no title evidence)"
tot=collections.Counter(); old_only=[i for i in old if i not in newids and i in rows]
for i in old_only: tot[cause(rows[i])]+=1
print("\nOLD-ONLY & retrieved-but-excluded: by actual exclusion cause (n=%d)"%len(old_only))
for k,v in tot.most_common(): print(f"  {v:6d} {v/len(old_only):6.1%}  {k}")
mp=json.load(open(SP+"sample_map.json"))
hk={g["item"]:g["label"] for p in glob.glob(SP+"labels_haiku/batch_*.json") for g in json.load(open(p))}
ek={x["item"]:x["label"] for x in json.load(open(SP+"eikos_labels.json"))}
by=collections.defaultdict(list)
for it,m in mp.items():
    if m["group"]=="old_only" and m["work_id"] in rows: by[cause(rows[m["work_id"]])].append(it)
print("\nsampled (n=%d): label mix by cause  [CORE/PERIPH/NOT  Haiku | Eikos]"%sum(len(v) for v in by.values()))
for k,its in sorted(by.items(),key=lambda kv:-len(kv[1])):
    f=lambda lab:"/".join(f"{sum(lab[i]==L for i in its)/len(its):.0%}" for L in ("CORE","PERIPHERAL","NOT"))
    print(f"  n={len(its):3d}  {f(hk)} | {f(ek)}   {k}")
