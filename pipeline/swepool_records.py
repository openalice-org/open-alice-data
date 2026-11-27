#!/usr/bin/env python3
"""Build OpenAlex-API-shaped work records from the SweMetrics `oa` mirror.

The rest of the pipeline (split_nordic_corpus.py, extract_authors.py, build_people_data.py, ...) was written
for records as the OpenAlex API returns them: authorships with nested institutions, topics with nested
subfield/field/domain, a primary location, an abstract as an inverted index. `fetch_records` rebuilds exactly
that shape from the `oa` tables, so works found through SweMetrics can join the corpus without touching
those steps.

Queries are grouped by work_id range (`between lo and hi` lets the server skip row groups) because a bare
`work_id in (...)` over a billion-row table scans all of it.

    python swepool_records.py --validate 60     # rebuild 60 corpus works and compare with their API records
"""

import argparse
import collections
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from swemetrics import in_list, query  # noqa: E402

CHUNK_IDS = 600
CHUNK_SPAN = 250_000_000


def oa(kind: str, n) -> str | None:
    return None if n is None else f"https://openalex.org/{kind}{int(n)}"


def doi_url(doi):
    if not doi:
        return None
    return doi if str(doi).startswith("http") else f"https://doi.org/{doi}"


def orcid_url(o):
    if not o:
        return None
    return o if str(o).startswith("http") else f"https://orcid.org/{o}"


def inverted_index(text: str) -> dict:
    """Plain abstract -> OpenAlex inverted index (word -> positions); reconstruct_abstract() inverts it back."""
    out: dict[str, list[int]] = {}
    for pos, word in enumerate(text.split()):
        out.setdefault(word, []).append(pos)
    return out


def id_chunks(ids):
    ids = sorted({int(i) for i in ids})
    chunk: list[int] = []
    for i in ids:
        if chunk and (len(chunk) >= CHUNK_IDS or i - chunk[0] > CHUNK_SPAN):
            yield chunk
            chunk = []
        chunk.append(i)
    if chunk:
        yield chunk


def where(chunk, col="work_id") -> str:
    return f"{col} between {chunk[0]} and {chunk[-1]} and {col} in (select unnest({in_list(chunk)}))"


WORK_COLS = """work_id, doi, title, display_name, publication_date, publication_year, language, type, cited_by_count,
  is_retracted, updated_date, id_pmid, id_pmcid, id_mag,
  primary_topic_id, primary_topic_display_name, primary_topic_score, primary_topic_subfield_id,
  primary_topic_subfield_display_name, primary_topic_field_id, primary_topic_field_display_name,
  primary_topic_domain_id, primary_topic_domain_display_name,
  open_access_is_oa, open_access_oa_status, open_access_any_repository_has_fulltext, open_access_oa_url,
  primary_location_id, primary_location_is_oa, primary_location_landing_page_url, primary_location_pdf_url,
  primary_location_is_published, primary_location_is_accepted, primary_location_raw_source_name,
  primary_location_raw_type, primary_location_license, primary_location_version, primary_location_source_id,
  primary_location_source_display_name, primary_location_source_issn_l, primary_location_source_issn,
  primary_location_source_is_oa, primary_location_source_is_in_doaj, primary_location_source_is_core,
  primary_location_source_host_organization_id, primary_location_source_host_organization_name,
  primary_location_source_host_organization_lineage, primary_location_source_host_organization_lineage_names,
  primary_location_source_type"""


def _build(wid, w, abstract, auths, insts, affs, topics, kws) -> dict:
    topic = None
    if w["primary_topic_id"] is not None:
        topic = {
            "id": oa("T", w["primary_topic_id"]), "display_name": w["primary_topic_display_name"], "score": w["primary_topic_score"],
            "subfield": {"id": f"https://openalex.org/subfields/{w['primary_topic_subfield_id']}", "display_name": w["primary_topic_subfield_display_name"]},
            "field": {"id": f"https://openalex.org/fields/{w['primary_topic_field_id']}", "display_name": w["primary_topic_field_display_name"]},
            "domain": {"id": f"https://openalex.org/domains/{w['primary_topic_domain_id']}", "display_name": w["primary_topic_domain_display_name"]},
        }
    source = None
    if w["primary_location_source_id"] is not None:
        source = {
            "id": oa("S", w["primary_location_source_id"]), "display_name": w["primary_location_source_display_name"],
            "issn_l": w["primary_location_source_issn_l"], "issn": w["primary_location_source_issn"],
            "is_oa": w["primary_location_source_is_oa"], "is_in_doaj": w["primary_location_source_is_in_doaj"],
            "is_core": w["primary_location_source_is_core"], "type": w["primary_location_source_type"],
            "host_organization": oa("P", w["primary_location_source_host_organization_id"]),
            "host_organization_name": w["primary_location_source_host_organization_name"],
            "host_organization_lineage": [oa("P", x) for x in (w["primary_location_source_host_organization_lineage"] or [])],
            "host_organization_lineage_names": w["primary_location_source_host_organization_lineage_names"],
        }
    authorships = []
    # the table keeps no authorship order: restore first / middle / last from author_position (the order among
    # middle authors is not recoverable from OpenAlex's tables)
    for a in sorted(auths, key=lambda r: {"first": 0, "last": 2}.get(r["author_position"], 1)):
        aid = a["author_id"]
        raw_affs = a["raw_affiliation_strings"] or []
        aff_by_string: dict[str, list[str]] = collections.OrderedDict((s, []) for s in raw_affs)
        for f in affs.get(aid, []):
            aff_by_string.setdefault(f["raw_affiliation_string"], [])
            if f["institution_id"] is not None:
                aff_by_string[f["raw_affiliation_string"]].append(oa("I", f["institution_id"]))
        authorships.append({
            "author_position": a["author_position"],
            "author": {"id": oa("A", aid), "display_name": a["author_display_name"], "orcid": orcid_url(a["orcid"]),
                       "observed_orcids": [orcid_url(a["raw_orcid"])] if a["raw_orcid"] else []},
            "institutions": [
                {"id": oa("I", i["institution_id"]), "display_name": i["institution_display_name"], "ror": i["institution_ror"],
                 "country_code": i["institution_country_code"], "type": i["institution_type"],
                 "lineage": [oa("I", x) for x in (i["lineage"] or [])]}
                for i in insts.get(aid, [])
            ],
            "countries": a["countries"] or [],
            "is_corresponding": a["is_corresponding"],
            "raw_author_name": a["raw_author_name"],
            "raw_affiliation_strings": raw_affs,
            "affiliations": [{"raw_affiliation_string": s, "institution_ids": ids} for s, ids in aff_by_string.items()],
        })
    ids = {"openalex": oa("W", wid)}
    if w["doi"]:
        ids["doi"] = doi_url(w["doi"])
    if w["id_mag"]:
        ids["mag"] = str(w["id_mag"])
    if w["id_pmid"]:
        ids["pmid"] = str(w["id_pmid"])
    return {
        "id": oa("W", wid), "doi": doi_url(w["doi"]), "ids": ids, "display_name": w["display_name"] or w["title"],
        "publication_year": w["publication_year"], "publication_date": (w["publication_date"] or "")[:10] or None,
        "type": w["type"], "work_type": w["type"], "language": w["language"], "cited_by_count": w["cited_by_count"],
        "is_retracted": bool(w["is_retracted"]), "updated_date": w["updated_date"],
        "authorships": authorships, "primary_topic": topic,
        "topics": [{"id": oa("T", t["topic_id"]), "display_name": t["display_name"], "score": t["score"],
                    "subfield": {"id": f"https://openalex.org/subfields/{t['subfield_id']}", "display_name": t["subfield_display_name"]},
                    "field": {"id": f"https://openalex.org/fields/{t['field_id']}", "display_name": t["field_display_name"]},
                    "domain": {"id": f"https://openalex.org/domains/{t['domain_id']}", "display_name": t["domain_display_name"]}}
                   for t in topics],
        "keywords": [{"id": f"https://openalex.org/keywords/{k['keyword_id']}", "display_name": k["display_name"], "score": k["score"]} for k in kws],
        "abstract_inverted_index": inverted_index(abstract) if abstract else None,
        "open_access": {"is_oa": w["open_access_is_oa"], "oa_status": w["open_access_oa_status"],
                        "any_repository_has_fulltext": w["open_access_any_repository_has_fulltext"], "oa_url": w["open_access_oa_url"]},
        "primary_location": {
            "id": w["primary_location_id"], "is_oa": w["primary_location_is_oa"], "landing_page_url": w["primary_location_landing_page_url"],
            "pdf_url": w["primary_location_pdf_url"], "is_published": w["primary_location_is_published"],
            "is_accepted": w["primary_location_is_accepted"], "raw_source_name": w["primary_location_raw_source_name"],
            "raw_type": w["primary_location_raw_type"], "license": w["primary_location_license"],
            "version": w["primary_location_version"], "source": source,
        },
        "locations": [],
    }


def iter_records(work_ids, progress=False):
    """Yield {work_id: record} one id-chunk at a time, so a caller can checkpoint after every chunk."""
    for n, chunk in enumerate(id_chunks(work_ids), 1):
        w_where = where(chunk)
        works = {r["work_id"]: r for r in query(f"select {WORK_COLS} from oa.works where {w_where}", timeout=290)}
        abstracts = {r["work_id"]: r["abstract"] for r in query(f"select work_id, abstract from oa.works_abstracts where {w_where}", timeout=290)}
        auths = collections.defaultdict(list)
        for r in query(f"select work_id, author_position, author_id, author_display_name, orcid, raw_orcid, raw_author_name, "
                       f"is_corresponding, raw_affiliation_strings, countries from oa.works_authorships where {w_where}", timeout=290):
            auths[r["work_id"]].append(r)
        insts = collections.defaultdict(lambda: collections.defaultdict(list))
        for r in query(f"select work_id, author_id, institution_id, institution_display_name, institution_ror, institution_country_code, "
                       f"institution_type, lineage from oa.works_authorship_institutions where {w_where}", timeout=290):
            insts[r["work_id"]][r["author_id"]].append(r)
        affs = collections.defaultdict(lambda: collections.defaultdict(list))
        for r in query(f"select work_id, author_id, raw_affiliation_string, institution_id from oa.works_authorship_affiliations where {w_where}", timeout=290):
            affs[r["work_id"]][r["author_id"]].append(r)
        topics = collections.defaultdict(list)
        for r in query(f"select work_id, topic_id, display_name, score, subfield_id, subfield_display_name, field_id, field_display_name, "
                       f"domain_id, domain_display_name from oa.works_topics where {w_where} order by score desc", timeout=290):
            topics[r["work_id"]].append(r)
        kws = collections.defaultdict(list)
        for r in query(f"select work_id, keyword_id, display_name, score from oa.works_keywords where {w_where} order by score desc", timeout=290):
            kws[r["work_id"]].append(r)
        out = {wid: _build(wid, w, abstracts.get(wid), auths[wid], insts[wid], affs[wid], topics[wid], kws[wid]) for wid, w in works.items()}
        if progress:
            print(f"  records: chunk {n}: {len(out)} of {len(chunk)} built", flush=True)
        yield out


def fetch_records(work_ids, progress=False) -> dict[int, dict]:
    """OpenAlex-shaped records for OpenAlex work ids (integers), from the `oa` mirror."""
    out: dict[int, dict] = {}
    for part in iter_records(work_ids, progress):
        out.update(part)
    return out


def validate(n: int) -> None:
    """Rebuild N accepted corpus works from SQL and compare with their API records."""
    from build_domain_specter import reconstruct_abstract
    from build_taxonomy_data import ACCEPTED_LABELS, SPRINT2_WORKS

    api = {}
    for line in open(SPRINT2_WORKS, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            w = json.loads(line)
        except json.JSONDecodeError:
            continue
        if w.get("classification_label") in ACCEPTED_LABELS and w.get("id"):
            wid = int(w["id"].rsplit("/W", 1)[1])
            if 2_000_000_000 <= wid < 2_250_000_000:  # one dense id window keeps the validation to a single quick chunk
                api[wid] = w
    rnd = random.Random(7)
    pick = rnd.sample(sorted(api), min(n, len(api)))
    print(f"validating {len(pick)} works (of {len(api)} accepted corpus works in the id window)", flush=True)
    built = fetch_records(pick, progress=True)
    stats = collections.Counter()
    for wid in pick:
        a, b = api[wid], built.get(wid)
        stats["works"] += 1
        if b is None:
            stats["missing in sql"] += 1
            continue
        stats["abstract equal"] += reconstruct_abstract(a.get("abstract_inverted_index")).split() == reconstruct_abstract(b.get("abstract_inverted_index")).split()
        stats["primary_topic id equal"] += (a.get("primary_topic") or {}).get("id") == (b.get("primary_topic") or {}).get("id")
        stats["keyword set equal"] += {k["display_name"] for k in a.get("keywords") or []} == {k["display_name"] for k in b.get("keywords") or []}
        stats["topic ids equal"] += {t["id"] for t in a.get("topics") or []} == {t["id"] for t in b.get("topics") or []}
        stats["authorship count equal"] += len(a["authorships"]) == len(b["authorships"])
        aa = {x["author"]["id"]: x for x in a["authorships"]}
        bb = {x["author"]["id"]: x for x in b["authorships"]}
        stats["author id set equal"] += set(aa) == set(bb)
        common = set(aa) & set(bb)
        stats["authorships compared"] += len(common)
        stats["institution ids equal (per authorship)"] += sum({i["id"] for i in aa[k]["institutions"]} == {i["id"] for i in bb[k]["institutions"]} for k in common)
        stats["countries equal (per authorship)"] += sum(sorted(aa[k].get("countries") or []) == sorted(bb[k].get("countries") or []) for k in common)
        stats["orcid equal (per authorship)"] += sum((aa[k]["author"].get("orcid") or None) == (bb[k]["author"].get("orcid") or None) for k in common)
        stats["same author order"] += [x["author"]["id"] for x in a["authorships"]] == [x["author"]["id"] for x in b["authorships"]]
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--validate", type=int, metavar="N", help="rebuild N known corpus works and compare with the API records")
    args = ap.parse_args()
    if args.validate:
        validate(args.validate)
    else:
        ap.print_help()
