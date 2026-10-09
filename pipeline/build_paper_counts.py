"""Distinct quantum-paper counts (and domain mix) per institution and per country.

Reads the concatenated corpus (all_universities_works.jsonl), where a paper
co-authored across tracked institutions is duplicated once per institution
by split_nordic_corpus.py. Dedupes by OpenAlex work id, then attributes each
distinct paper to every tracked institution (and its country) it touches, so
a co-authored paper is counted once per institution/country rather than once
per author — replacing the person-level sum in index.html's
geoAggregatesByUni/geoAggregatesByCountry, which double-counts collaborations.

Domain mix (computing/communication/sensing) is hard-assigned per paper from
data/work_domain_labels.json (build_domain_specter.py's per-work argmax over
its BGE-embedding domain distribution — the SAME classifier behind
people_viz.json's computing_share/communication_share/sensing_share, just
applied per paper instead of averaged per person). A cheaper stdlib lexicon
classifier (build_taxonomy_data.py's classify_work_domains) was tried first
and rejected: it only has signal on ~9% of papers, which would leave ~91%
domain-unclassified — not a usable "papers per domain" breakdown. The BGE
model classifies every paper (dense embedding similarity, no "no signal"
case), at the cost of needing build_domain_specter.py to have been run first.
That file is still optional here: missing it just means domain_by_institution/
domain_by_country are omitted and index.html falls back to its old
person-summed slice weighting — only the paper *count* (not the domain mix)
gets fixed until build_domain_specter.py has produced it.

Note: work_domain_labels.json's coverage (works matched to at least one
tracked PERSON) is narrower than this script's own institution-matched paper
set (works matched to at least one tracked INSTITUTION) — a paper can touch a
tracked institution's affiliation string without its specific author having
been successfully resolved to a person record. Papers missing a label are
still counted in total/by_institution/by_country, just not in the domain
breakdown.
"""

from __future__ import annotations

import json
from pathlib import Path

QUANTUM_LABELS = {"core_quantum", "likely_quantum"}
REAL_DOMAINS = ("computing", "communication", "sensing")


def load_institution_index(universities: dict) -> dict[str, str]:
    """Maps every OpenAlex institution id (canonical + aliases) to its label."""
    index: dict[str, str] = {}
    for entry in universities.values():
        label = entry["label"]
        index[entry["openalex_institution_id"]] = label
        for alias in entry.get("openalex_institution_id_aliases", []):
            index[alias] = label
    return index


def _empty_domain_bucket() -> dict:
    return {d: 0 for d in REAL_DOMAINS}


# Time-window cutoffs, mirroring the frontend's filterByTimeWindow exactly
# (src/lib/people.ts): cutoff = currentYear - years + 1, so "last N years"
# includes the current year itself. Windows narrow by a paper's OWN
# publication_year here (not a person's last_year span like the frontend's
# people-side filter) — the literal, correct meaning of "papers published in
# the last N years", and still fully deduped per window the same way `all`
# already is (a co-authored paper counts once per institution/country, not
# once per author).
def _window_cutoffs(current_year: int) -> dict[str, int | None]:
    return {"all": None, "last5": current_year - 5 + 1, "last10": current_year - 10 + 1}


def count_distinct_papers(works_path: Path, institution_index: dict[str, str],
                           work_domain: dict[str, str] | None, current_year: int) -> dict:
    cutoffs = _window_cutoffs(current_year)
    windows = {
        w: {"total": 0, "by_institution": {}, "by_country": {},
            "domain_by_institution": {}, "domain_by_country": {}}
        for w in cutoffs
    }
    seen_ids: set[str] = set()

    with works_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            pid = rec.get("id")
            if not pid or pid in seen_ids:
                continue
            seen_ids.add(pid)
            if rec.get("classification_label") not in QUANTUM_LABELS:
                continue

            labels = set()
            countries = set()
            for ship in rec.get("authorships", []) or []:
                for inst in ship.get("institutions", []) or []:
                    label = institution_index.get(inst.get("id"))
                    if not label:
                        continue
                    labels.add(label)
                    cc = inst.get("country_code")
                    if cc:
                        countries.add(cc)

            if not labels:
                continue
            domain = work_domain.get(pid) if work_domain else None
            pub_year = rec.get("publication_year")
            pub_year = pub_year if isinstance(pub_year, int) else None

            for w, cutoff in cutoffs.items():
                if cutoff is not None and (pub_year is None or pub_year < cutoff):
                    continue
                bucket = windows[w]
                bucket["total"] += 1
                for label in labels:
                    bucket["by_institution"][label] = bucket["by_institution"].get(label, 0) + 1
                    if domain:
                        db = bucket["domain_by_institution"].setdefault(label, _empty_domain_bucket())
                        db[domain] += 1
                for cc in countries:
                    bucket["by_country"][cc] = bucket["by_country"].get(cc, 0) + 1
                    if domain:
                        db = bucket["domain_by_country"].setdefault(cc, _empty_domain_bucket())
                        db[domain] += 1

    if not work_domain:
        for bucket in windows.values():
            del bucket["domain_by_institution"]
            del bucket["domain_by_country"]

    # Backward-compatible shape: "all"'s own fields stay flat at the top
    # level (unchanged keys/meaning existing consumers already read), with
    # "last5"/"last10" added as new, optional, same-shape siblings.
    result: dict = dict(windows["all"])
    result["last5"] = windows["last5"]
    result["last10"] = windows["last10"]
    return result


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Compute distinct paper counts (and domain mix) per institution/country")
    parser.add_argument("--works", type=Path, default=Path("data/all_universities_works.jsonl"))
    parser.add_argument("--work-domain", type=Path, default=Path("data/work_domain_labels.json"),
                         help="optional per-work hard domain labels from build_domain_specter.py")
    parser.add_argument("--out", type=Path, default=Path("data/paper_counts_geo.json"))
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    universities = json.loads((root / "universities.json").read_text())
    institution_index = load_institution_index(universities)

    work_domain = None
    if args.work_domain.exists():
        work_domain = json.loads(args.work_domain.read_text())
    else:
        print(f"No {args.work_domain} found — domain-mix dedup skipped, "
              f"paper counts still computed. Run build_domain_specter.py to enable it.")

    from datetime import datetime, timezone
    current_year = datetime.now(timezone.utc).year
    counts = count_distinct_papers(args.works, institution_index, work_domain, current_year)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(counts, indent=2, sort_keys=True))
    print(f"Wrote distinct paper counts for {len(counts['by_institution'])} institutions, "
          f"{len(counts['by_country'])} countries to {args.out}"
          + (" (with domain mix)" if work_domain else ""))


if __name__ == "__main__":
    main()
