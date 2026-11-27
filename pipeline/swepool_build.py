#!/usr/bin/env python3
"""SweMetrics enrichment, stage S1: find candidate works the API-based corpus never saw.

WHY
    The corpus (corpus/ steps 01-05) was built from OpenAlex API routes: works whose authorships carry a
    Nordic or Baltic country AND that sit in a quantum-related topic or derived keyword. The SweMetrics
    DuckDB gives us a full OpenAlex mirror (`oa`) plus the Swedish national index (`swepub`), so we can
    cast a broader net by TEXT, whatever the topic, and also use Swepub's curated records as evidence.

WHAT THIS STAGE DOES (resumable; each part writes a raw file, `assemble` joins them into the pool)

    swepub     Swepub works (the Swedish national index) whose title, abstract or author-supplied keywords
               match the same patterns, with their abstract, keywords, the matched OpenAlex id (if any) and the
               curated author records (ORCID, name parts, local person id, matched OpenAlex author id).
               Written to swe_swepub_raw.jsonl and swe_swepub_authors.jsonl. The `swepub` tables are small, so
               this needs no slicing.

    textnet    Nordic/Baltic works (authorship country, publication-time) whose title, abstract or
               OpenAlex keywords match the corpus's own paper-level quantum term patterns
               (corpus/config/sweden_quantum_paper_filter.json). The big tables are sliced by work_id range
               because a single full scan exceeds the server's per-query deadline; every slice is appended
               to corpus/data/swe_textnet_raw.jsonl and recorded in swe_textnet_raw.done.json, so an
               interrupted run continues where it stopped.

    assemble   Join both raw files into the pool corpus/data/swe_pool.jsonl (+ manifest):
                 * drop works the API-based corpus already evaluated (the step-05 candidate audit);
                 * apply the same record gates (language, work type) as corpus step 05;
                 * enrich: when OpenAlex has no abstract (or a short one) take Swepub's; keep Swepub's
                   author-supplied keywords;
                 * keep only works with a usable abstract (>= 200 characters), like the rescue pool.
               Swepub-only works (no OpenAlex id) are included under the id `swepub:<internal_work_id>` so the
               gate can already tell us how many of them pass; they cannot join the corpus until their
               authors are linked to people (a later stage).

Usage
    python swepool_build.py assemble                   # seconds, local only
    python swepool_build.py swepub                     # about a minute
    python swepool_build.py textnet                    # all slices (about 30-60 min), resumable
    python swepool_build.py textnet --from 2000000000 --to 2100000000   # one range (trial)
"""

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import swe_state  # noqa: E402
from swemetrics import NORDIC_BALTIC, SweMetricsError, chunks, in_list, query  # noqa: E402

CORPUS_DATA = HERE.parent / "corpus" / "data"
CONFIG = HERE.parent / "corpus" / "config" / "sweden_quantum_paper_filter.json"
RAW = CORPUS_DATA / "swe_textnet_raw.jsonl"
DONE = CORPUS_DATA / "swe_textnet_raw.done.json"
SWEPUB_RAW = CORPUS_DATA / "swe_swepub_raw.jsonl"
SWEPUB_AUTHORS = CORPUS_DATA / "swe_swepub_authors.jsonl"

MAX_WORK_ID = 7_300_000_000  # oa.works runs up to about 7.22e9
DEFAULT_SLICE = 100_000_000
YEAR_MIN, YEAR_MAX = 1950, 2026


def quantum_regex() -> str:
    """One case-insensitive alternation of the corpus's paper-level term patterns (RE2 syntax, SQL-quoted)."""
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    pats = [*cfg["paper_level_term_patterns"], *cfg.get("paper_level_only_term_patterns", [])]
    return "|".join(f"(?:{p})" for p in pats).replace("'", "''")


def textnet_sql(lo: int, hi: int, rx: str) -> str:
    nb = "[" + ",".join(f"'{c}'" for c in NORDIC_BALTIC) + "]"
    rng = lambda col: f"{col} >= {lo} and {col} < {hi}"  # noqa: E731  (a range filter lets the server prune row groups)
    return f"""
with nordic as (
  select distinct work_id from oa.works_authorships where {rng('work_id')} and list_has_any(countries, {nb})
),
base as (
  select w.work_id, w.title, w.publication_year, w.type, w.language, w.is_retracted, w.cited_by_count,
         w.primary_topic_display_name as topic, w.primary_topic_field_display_name as field,
         w.primary_topic_domain_display_name as domain, ab.abstract,
         regexp_matches(coalesce(w.title, ''), '{rx}', 'i') as in_title,
         regexp_matches(coalesce(ab.abstract, ''), '{rx}', 'i') as in_abstract,
         exists (select 1 from oa.works_keywords k where {rng('k.work_id')} and k.work_id = w.work_id
                 and regexp_matches(coalesce(k.display_name, ''), '{rx}', 'i')) as in_keywords
  from nordic
  join oa.works w on w.work_id = nordic.work_id and {rng('w.work_id')}
  left join oa.works_abstracts ab on ab.work_id = w.work_id and {rng('ab.work_id')}
  where w.publication_year between {YEAR_MIN} and {YEAR_MAX} and not coalesce(w.is_paratext, false)
)
select * from base where in_title or in_abstract or in_keywords
"""


def run_slice(lo: int, hi: int, rx: str, depth: int = 0) -> list[dict]:
    """One slice; if the server's deadline interrupts it, split it in half and try again (up to 3 times)."""
    try:
        return query(textnet_sql(lo, hi, rx), timeout=290, retries=1)
    except SweMetricsError as e:
        if depth >= 3 or hi - lo < 2:
            raise
        print(f"    slice {lo}-{hi} failed ({str(e)[:60]}), splitting", flush=True)
        mid = (lo + hi) // 2
        return run_slice(lo, mid, rx, depth + 1) + run_slice(mid, hi, rx, depth + 1)


def swepub_sql(rx: str) -> str:
    return f"""
with kw as (
  select record_id, list(distinct display_name) as keywords,
         bool_or(regexp_matches(coalesce(display_name, ''), '{rx}', 'i')) as in_keywords
  from swepub.works_keywords group by record_id
),
base as (
  select w.internal_work_id, w.record_id, w.work_id, w.match_method, w.is_ambiguous, w.doi, w.id_pmid,
         w.title, w.publication_year, w.type, w.language, ab.abstract, kw.keywords,
         regexp_matches(coalesce(w.title, ''), '{rx}', 'i') as in_title,
         regexp_matches(coalesce(ab.abstract, ''), '{rx}', 'i') as in_abstract,
         coalesce(kw.in_keywords, false) as in_keywords
  from swepub.works w
  left join swepub.works_abstracts ab on ab.record_id = w.record_id
  left join kw on kw.record_id = w.record_id
  where w.publication_year between {YEAR_MIN} and {YEAR_MAX}
)
select * from base where in_title or in_abstract or in_keywords
"""


def cmd_swepub(args) -> None:
    rx = quantum_regex()
    t = time.time()
    rows = query(swepub_sql(rx), timeout=290)
    print(f"swepub works with quantum signal: {len(rows)} ({time.time() - t:.0f}s)", flush=True)
    with open(SWEPUB_RAW, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    ids = [r["internal_work_id"] for r in rows]
    n = 0
    with open(SWEPUB_AUTHORS, "w", encoding="utf-8") as f:
        for part in chunks(ids, 4000):
            for a in query(f"""
                select a.internal_work_id, a.seq, a.author_position, a.author_display_name, a.raw_author_name,
                       a.orcid, a.raw_orcid, a.raw_affiliation_strings, a.ext_given_name, a.ext_family_name,
                       a.ext_local_person_id, a.internal_author_id,
                       sa.author_id as oa_author_id, sa.match_method as author_match_method,
                       sa.is_ambiguous as author_ambiguous, sa.orcid as curated_orcid, sa.display_name as curated_name
                from swepub.works_authorships a
                left join swepub.authors sa on sa.internal_author_id = a.internal_author_id
                where a.internal_work_id in (select unnest({in_list(part)}))""", timeout=290):
                f.write(json.dumps(a, ensure_ascii=False) + "\n")
                n += 1
    matched = sum(1 for r in rows if r["work_id"] is not None)
    print(f"  matched to OpenAlex: {matched} | swepub-only: {len(rows) - matched} | author rows: {n}")
    print(f"  wrote {SWEPUB_RAW.name} and {SWEPUB_AUTHORS.name}")


def full_id(work_id) -> str:
    return f"https://openalex.org/W{int(work_id)}"


LANG_FIX = {"eng": "en", "swe": "sv", "nor": "no", "dan": "da", "fin": "fi", "ger": "de", "fre": "fr", "und": None}
MIN_ABSTRACT_CHARS = 200


def read_jsonl(path: Path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def cmd_assemble(args) -> None:
    import collections
    import csv
    import hashlib

    gates = json.loads((HERE.parent / "corpus" / "config" / "sweden_quantum_record_gates.json").read_text())
    accepted_lang = set(gates.get("accepted_languages") or [])
    excluded_types = set(gates.get("excluded_work_types") or [])

    known = set()
    with open(CORPUS_DATA / "sweden_quantum_candidate_audit.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            known.add(int(row["work_id"][1:]))

    pool: dict[str, dict] = {}
    count = collections.Counter()

    def lang_ok(code):
        code = LANG_FIX.get(code, code)
        return (not code and gates.get("accept_missing_language", True)) or (code in accepted_lang if accepted_lang else True)

    # --- OpenAlex text net
    latest = {}
    for r in read_jsonl(RAW):  # an incremental harvest appends refreshed rows: the last one for a work wins
        latest[int(r["work_id"])] = r
    for wid, r in latest.items():
        count["textnet hits"] += 1
        if wid in known:
            count["textnet: already evaluated by the API corpus"] += 1
            continue
        pool[full_id(wid)] = {
            "id": full_id(wid), "display_name": (r["title"] or "").strip(), "abstract": r["abstract"] or "",
            "publication_year": r["publication_year"], "type": r["type"], "language": LANG_FIX.get(r["language"], r["language"]),
            "is_retracted": bool(r["is_retracted"]), "cited_by_count": r["cited_by_count"],
            "topic": r["topic"], "field": r["field"], "domain": r["domain"],
            "sources": ["oa_textnet"], "text_hits": [k for k in ("title", "abstract", "keywords") if r[f"in_{k}"]],
            "abstract_source": "openalex" if r["abstract"] else None, "keywords": [],
        }

    # --- Swepub
    for r in read_jsonl(SWEPUB_RAW):
        count["swepub hits"] += 1
        wid = r["work_id"]
        key = full_id(wid) if wid is not None else f"swepub:{r['internal_work_id']}"
        if wid is not None and int(wid) in known:
            count["swepub: already evaluated by the API corpus"] += 1
            continue
        sw_abs = r["abstract"] or ""
        hits = [k for k in ("title", "abstract", "keywords") if r[f"in_{k}"]]
        row = pool.get(key)
        if row is None:  # not seen by the OpenAlex text net: either swepub-only, or OpenAlex lacks the Nordic country / the text
            row = pool[key] = {
                "id": key, "display_name": (r["title"] or "").strip(), "abstract": sw_abs,
                "publication_year": r["publication_year"], "type": r["type"], "language": LANG_FIX.get(r["language"], r["language"]),
                "is_retracted": False, "cited_by_count": None, "topic": None, "field": None, "domain": None,
                "sources": ["swepub_country" if wid is not None else "swepub_only"], "text_hits": hits,
                "abstract_source": "swepub" if sw_abs else None, "keywords": [],
            }
        else:
            row["sources"].append("swepub")
            row["text_hits"] = sorted(set(row["text_hits"]) | set(hits))
            if len(sw_abs) >= MIN_ABSTRACT_CHARS and len(row["abstract"]) < MIN_ABSTRACT_CHARS:
                row["abstract"], row["abstract_source"] = sw_abs, "swepub (OpenAlex had none or a short one)"
        row["keywords"] = sorted({*row["keywords"], *(k.strip() for k in (r["keywords"] or []) if k and k.strip())})
        row["swepub_internal_work_id"] = r["internal_work_id"]
        row["swepub_record_id"] = r["record_id"]
        row["doi"] = r.get("doi")

    # --- gates and abstract requirement
    kept, dropped = [], collections.Counter()
    for key, row in pool.items():
        src = row["sources"][0]
        if not row["display_name"]:
            dropped["no title"] += 1
        elif accepted_lang and not lang_ok(row["language"]):
            dropped["record gate: language"] += 1
        elif row["type"] in excluded_types:
            dropped["record gate: work type"] += 1
        elif len(row["abstract"]) < MIN_ABSTRACT_CHARS:
            dropped["no usable abstract (< 200 chars)"] += 1
        else:
            kept.append(row)
            count[f"pool by source: {src}"] += 1
    kept.sort(key=lambda x: x["id"])
    out = CORPUS_DATA / "swe_pool.jsonl"
    sha = hashlib.sha256()
    with open(out, "w", encoding="utf-8") as f:
        for row in kept:
            line = json.dumps(row, ensure_ascii=False) + "\n"
            f.write(line)
            sha.update(line.encode("utf-8"))
    manifest = {
        "schema_version": 1, "stage": "S1 candidate pool (SweMetrics)", "pool_size": len(kept),
        "counts": dict(count), "dropped_before_pool": dict(dropped),
        "abstract_filled_from_swepub": sum(1 for r in kept if (r["abstract_source"] or "").startswith("swepub")),
        "min_abstract_chars": MIN_ABSTRACT_CHARS, "output": str(out), "output_sha256": sha.hexdigest(),
        "inputs": {"textnet_slices_done": len(json.loads(DONE.read_text())) if DONE.exists() else 0},
    }
    (CORPUS_DATA / "swe_pool_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


def snapshots() -> dict:
    """The SweMetrics snapshot dates (the small `institutions` tables carry them, so this is instant)."""
    out = {}
    for name, table in (("oa", "oa.institutions"), ("swepub", "swepub.institutions")):
        out[name] = str(query(f"select max(snapshot_date) as d from {table}")[0]["d"])[:10]
    return out


def cmd_snapshots(args) -> None:
    now = snapshots()
    state = swe_state.load()
    for k, v in now.items():
        seen = state["snapshots"].get(k)
        print(f"{k}: SweMetrics snapshot {v} | last processed {seen or 'never'} | {'CHANGED' if seen != v else 'unchanged'}")


def cmd_textnet(args) -> None:
    """Harvest the text net. FULL (all slices) the first time or with --full; afterwards INCREMENTAL: a new
    snapshot only adds works created since the last harvest, and those have the highest work ids, so only the
    slices from the old watermark (minus a margin for late-created old ids) upward are queried. Old works whose
    text or authorships changed are only picked up by a periodic --full run (the state records when)."""
    CORPUS_DATA.mkdir(parents=True, exist_ok=True)
    state = swe_state.load()
    tn = state.setdefault("textnet", {})
    snap = snapshots()["oa"]
    full = args.full or args.fresh or not tn.get("watermark")
    if not full and tn.get("snapshot") == snap and not args.force:
        print(f"text net is up to date for the OpenAlex snapshot {snap} (watermark work id {tn['watermark']}); use --force to rerun")
        return
    if args.fresh and RAW.exists():
        RAW.unlink()
    max_id = int(query("select max(work_id) as m from oa.works")[0]["m"])
    if full:
        lo0, hi0 = args.start, max(args.end, max_id + 1)
        done = set() if args.fresh else set(json.loads(DONE.read_text())) if DONE.exists() else set()
        what = f"FULL harvest, {len(range(lo0, hi0, args.slice))} slices"
    else:
        lo0 = max(0, (tn["watermark"] - args.margin) // args.slice * args.slice)
        hi0 = max_id + 1
        done = set()
        what = f"INCREMENTAL harvest from work id {lo0} (old watermark {tn['watermark']}, margin {args.margin})"
    print(f"{what}; snapshot {snap}", flush=True)
    rx = quantum_regex()
    slices = [(lo, min(lo + args.slice, hi0)) for lo in range(lo0, hi0, args.slice)]
    total_hits = 0
    t_all = time.time()
    for n, (lo, hi) in enumerate(slices, 1):
        key = f"{lo}-{hi}"
        if key in done:
            continue
        t = time.time()
        rows = run_slice(lo, hi, rx)
        with open(RAW, "a", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        done.add(key)
        if full:
            DONE.write_text(json.dumps(sorted(done)), encoding="utf-8")
        total_hits += len(rows)
        print(f"[{n}/{len(slices)}] work_id {lo:>10}-{hi:<10} {len(rows):>6} hits  {time.time() - t:5.1f}s  "
              f"(this run: {total_hits} hits, {time.time() - t_all:.0f}s)", flush=True)
        if args.limit_slices and n >= args.limit_slices:
            break
    if not args.limit_slices:  # a trial run must not move the watermark
        tn.update({"snapshot": snap, "watermark": max_id, "slice": args.slice})
        if full:
            tn["last_full"] = time.strftime("%Y-%m-%d")
        state["snapshots"]["oa"] = snap
        swe_state.save(state)
    print(f"done: {len(slices)} slices; raw file {RAW}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    snp = sub.add_parser("snapshots", help="show the SweMetrics snapshot dates against the last processed ones")
    snp.set_defaults(fn=cmd_snapshots)
    asm = sub.add_parser("assemble", help="join the raw files into corpus/data/swe_pool.jsonl")
    asm.set_defaults(fn=cmd_assemble)
    sw = sub.add_parser("swepub", help="Swepub works matching the quantum term patterns, with curated authors")
    sw.set_defaults(fn=cmd_swepub)
    t = sub.add_parser("textnet", help="Nordic/Baltic works matching the quantum term patterns")
    t.add_argument("--from", dest="start", type=int, default=0)
    t.add_argument("--to", dest="end", type=int, default=MAX_WORK_ID)
    t.add_argument("--slice", type=int, default=DEFAULT_SLICE, help="work_id range per query")
    t.add_argument("--fresh", action="store_true", help="discard the raw file and harvest everything again")
    t.add_argument("--full", action="store_true", help="harvest all slices (default after the first run: only new work ids)")
    t.add_argument("--force", action="store_true", help="rerun even if the snapshot is unchanged")
    t.add_argument("--margin", type=int, default=100_000_000, help="incremental runs go back this many work ids from the watermark (one slice)")
    t.add_argument("--limit-slices", type=int, default=0, help="stop after N slices (trial)")
    t.set_defaults(fn=cmd_textnet)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
