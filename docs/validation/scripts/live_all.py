"""Exact attribution for ALL old-only works that the new run no longer retrieves (no sampling)."""
import csv, json, time, collections, urllib.request, urllib.parse
H=str(__import__("pathlib").Path(__file__).resolve().parents[3].parent) + "/"
SP=str(__import__("pathlib").Path(__file__).resolve().parents[1] / "data") + "/"
key=[l.split("=",1)[1].strip() for l in open(H+"open-alice-data/.env") if l.startswith("OPENALEX_API_KEY=")][0]
SCOPE={"SE","NO","DK","IS","FI","EE","LV","LT"}
newids={json.loads(l)["id"] for l in open(H+"open-alice-data/corpus/data/sweden_quantum_works.jsonl",encoding="utf-8")}
cand={("https://openalex.org/"+r["work_id"]) if not r["work_id"].startswith("http") else r["work_id"] for r in csv.DictReader(open(H+"open-alice-data/corpus/data/sweden_quantum_candidate_audit.csv",encoding="utf-8"))}
routed={r["route"]["topic_id"].split("/")[-1] for r in json.load(open(H+"open-alice-data/corpus/data/sweden_quantum_candidate_union_manifest.json"))["routes"] if r["route"]["query_channel"]=="topic"}
todo={}
for l in open(H+"Qsip/SPRINT 13/data/all_universities_works.jsonl",encoding="utf-8"):
    w=json.loads(l); i=w["id"]
    if i in newids or i in cand or i in todo: continue
    todo[i]=[t.split("/")[-1] for t in (w.get("matched_topic_ids") or [])]
ids=list(todo); print("works to check:",len(ids),"| routed topics in new run:",len(routed),flush=True)
cats=collections.Counter(); by_year=collections.Counter(); res={}
def fetch(batch):
    f="|".join(b.split("/")[-1] for b in batch)
    url="https://api.openalex.org/works?"+urllib.parse.urlencode({"filter":"openalex:"+f,"per-page":len(batch),"select":"id,topics,authorships,publication_year","api_key":key})
    for attempt in range(4):
        try: return json.load(urllib.request.urlopen(url,timeout=60))["results"]
        except Exception as e: time.sleep(2*(attempt+1))
    return None
for s in range(0,len(ids),50):
    batch=ids[s:s+50]; got=fetch(batch)
    if got is None: cats["API FAILED (batch)"]+=len(batch); continue
    cur={g["id"]:g for g in got}
    for wid in batch:
        g=cur.get(wid)
        if g is None: c="deleted or merged (not returned by OpenAlex)"
        else:
            ct={t["id"].split("/")[-1] for t in g.get("topics") or []}
            cc={x for a in g.get("authorships") or [] for x in (a.get("countries") or [])}
            keeps=ct & set(todo[wid]); inscope=bool(cc & SCOPE)
            if not inscope: c="no in-scope country now (affiliation/country changed)"
            elif not keeps: c="RE-TAGGED: old topic no longer on the work" + (" (now carries a routed topic: unexpected)" if ct & routed else "")
            elif not (keeps & routed): c="old topic still present but NOT routed in new run (different broad-topic set)"
            else: c="old topic present AND routed: should have been retrieved (anomaly)"
        cats[c]+=1; res[wid]=c
    if (s//50)%40==0: print(s,"/",len(ids),dict(cats.most_common(3)),flush=True)
    time.sleep(0.12)
print("\nFINAL (n=%d):"%len(ids))
for k,v in cats.most_common(): print(f"  {v:6d} {v/len(ids):6.1%}  {k}")
json.dump(res,open(SP+"live_all_result.json","w"))
