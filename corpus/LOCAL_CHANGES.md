# Local changes to the upstream corpus tool

`corpus/` is a copy of `KTH.Tools.QuantumAuthors` (commit `4795a62`, "Topics quantum (#3)").
Four things differ from upstream. None of them is cosmetic, and all of them change what the
corpus contains or whether it can be rebuilt, so they are recorded here.

| # | File | Change | Why |
| --- | --- | --- | --- |
| 1 | `config/sweden_quantum_scope.json` | Country filters `SE` became `SE\|NO\|DK\|IS\|FI\|EE\|LV\|LT` (primary and audit), and the two definitions were reworded. | The dataset behind open-alice covers the Nordics **and** the Baltics. Upstream still commits a Sweden-only scope; the Nordic scope that produced the earlier builds was never committed. With the Sweden-only config the corpus has 35,482 works, all Swedish. |
| 2 | `src/02_inventory_quantum_topic_candidates.py` | Topic `keywords` are read through `topic_keyword_names()`, which accepts both strings and `{display_name, id, score}` objects. | OpenAlex changed the topics API from a list of strings to a list of objects. Without this, step 02 stops with `TypeError: normalize() argument 2 must be str, not dict`. |
| 3 | `src/03_download_sweden_quantum_candidates.py` | The union of the downloaded route files is built in two passes (`_write_candidate_union`): pass 1 keeps only `{work -> (file, byte offset, line digest, updated_date)}` and the route hits, pass 2 seeks back to each chosen record in id order. | The old version held every candidate record in one dict. On the Nordic and Baltic scope (205,464 candidates) that needed far more RAM than a 22 GB workstation has free. Output is byte-identical to the old code (checked on real route files and on synthetic duplicates, variants, newer and older copies). |
| 4 | `.env`, `data/` | Credentials live in the git-ignored `.env`; `data/` is git-ignored and holds the regenerated corpus. | Not a code change; listed because the data cannot be restored from git. |

The corpus built with these changes is **not** byte-identical to the one behind the previously
deployed data, and it should not be expected to be: OpenAlex re-tags topics and regenerates keywords
over time. `docs/METHODOLOGY.md` quantifies the difference and explains the rescue pass that was
added to compensate for the part of it that costs recall.
