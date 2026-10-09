import json, random, collections, os
H=str(__import__("pathlib").Path(__file__).resolve().parents[3].parent) + "/"
SP=str(__import__("pathlib").Path(__file__).resolve().parents[1] / "data") + "/"
rnd=random.Random(42)
def abstract(inv):
    if not inv: return ""
    pos=[]
    for w,ps in inv.items():
        for p in ps: pos.append((p,w))
    return " ".join(w for _,w in sorted(pos))
def item(w):
    return {"work_id":w["id"],"title":(w.get("display_name") or "").strip(),"abstract":abstract(w.get("abstract_inverted_index"))[:1400],"year":w.get("publication_year")}
newids={json.loads(l)["id"] for l in open(H+"open-alice-data/corpus/data/sweden_quantum_works.jsonl",encoding="utf-8")}
def reservoir(res,k,x,n):
    if len(res)<k: res.append(x)
    else:
        j=rnd.randrange(n)
        if j<k: res[j]=x
oldids=set(); old_only=[]; n_old=0; noabs=collections.Counter(); tot=collections.Counter()
for l in open(H+"Qsip/SPRINT 13/data/all_universities_works.jsonl",encoding="utf-8"):
    w=json.loads(l); i=w["id"]
    if i in oldids: continue
    oldids.add(i)
    if i in newids: continue
    it=item(w); tot["old_only"]+=1; noabs["old_only"]+= len(it["abstract"])<200
    if len(it["abstract"])>=200 and it["title"]:
        n_old+=1; reservoir(old_only,200,it,n_old)
new_only=[]; shared=[]; n_new=n_sh=0
for l in open(H+"open-alice-data/corpus/data/sweden_quantum_works.jsonl",encoding="utf-8"):
    w=json.loads(l); it=item(w); g="shared" if w["id"] in oldids else "new_only"
    tot[g]+=1; noabs[g]+= len(it["abstract"])<200
    if len(it["abstract"])<200 or not it["title"]: continue
    if g=="new_only": n_new+=1; reservoir(new_only,200,it,n_new)
    else: n_sh+=1; reservoir(shared,100,it,n_sh)
print("share with <200-char abstract (unlabelable):",{g:f"{noabs[g]/tot[g]:.1%} of {tot[g]}" for g in tot})
rows=[("old_only",x) for x in old_only]+[("new_only",x) for x in new_only]+[("shared",x) for x in shared]
rnd.shuffle(rows)
os.makedirs(SP+"batches",exist_ok=True); mp={}
for k,(g,x) in enumerate(rows,1):
    iid=f"item_{k:04d}"; mp[iid]={"group":g,"work_id":x["work_id"]}; x["item"]=iid
for b in range(0,len(rows),50):
    with open(SP+f"batches/batch_{b//50+1:02d}.jsonl","w",encoding="utf-8") as f:
        for g,x in rows[b:b+50]:
            f.write(json.dumps({"item":x["item"],"year":x["year"],"title":x["title"],"abstract":x["abstract"]},ensure_ascii=False)+"\n")
json.dump(mp,open(SP+"sample_map.json","w"))
print("items:",len(rows),"| groups:",collections.Counter(g for g,_ in rows),"| batches:",len(os.listdir(SP+"batches")))
