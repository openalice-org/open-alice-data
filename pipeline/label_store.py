#!/usr/bin/env python3
"""A persistent store of paper labels, so no paper is ever labelled twice.

Labelling a paper with Eikos-4B costs about 0.6 s on the iGPU. Label files used to be keyed by an item
number that changes whenever a sample is redrawn, and the sample, band and audit runs shared nothing, so
the same paper could be labelled again and again. This store keeps one record per
(work id, text hash, labeller) in an append-only JSONL file:

    {"id", "text_hash", "labeller", "label", "probs", "tokens", "labelled_at"}

* `id`        the work id (an OpenAlex url or `swepub:<n>`),
* `text_hash` a hash of the text the labeller saw (title + the abstract as given), so an edited abstract is
              labelled again but an unchanged one never is,
* `labeller`  a short key for the model AND rubric, e.g. `eikos-4b-int4-ov:<rubric hash>`: a new model or
              a changed rubric invalidates old labels on purpose.

Append-only means an interrupted run loses nothing: every label is written the moment it exists.

    python label_store.py import --items corpus/data/rescue_label_sample.jsonl \\
        --labels corpus/data/rescue_label_sample_eikos.json --labeller-model eikos-4b-int4-ov
    python label_store.py stats
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_STORE = HERE.parent / "corpus" / "data" / "label_store.jsonl"


def text_hash(title: str, abstract: str) -> str:
    return hashlib.sha1(f"{title}\n{abstract}".encode("utf-8")).hexdigest()[:16]


def labeller_key(model: str, question) -> str:
    """Model + rubric: changing either makes earlier labels stale. `question` is the rubric object
    (rescue_rubric.QUESTION, a dict) or a string; it is hashed in a stable form."""
    text = question if isinstance(question, str) else json.dumps(question, sort_keys=True, ensure_ascii=False)
    return f"{model}:{hashlib.sha1(text.encode('utf-8')).hexdigest()[:8]}"


class LabelStore:
    def __init__(self, path: Path = DEFAULT_STORE):
        self.path = Path(path)
        self.rows: dict[tuple[str, str, str], dict] = {}
        if self.path.exists():
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        self.rows[(r["id"], r["text_hash"], r["labeller"])] = r

    def get(self, wid: str, thash: str, labeller: str):
        return self.rows.get((wid, thash, labeller))

    def add(self, row: dict) -> None:
        row = {**row, "labelled_at": row.get("labelled_at") or time.strftime("%Y-%m-%dT%H:%M:%S")}
        self.rows[(row["id"], row["text_hash"], row["labeller"])] = row
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def __len__(self):
        return len(self.rows)


def cmd_import(args) -> None:
    sys.path.insert(0, str(HERE))
    from rescue_rubric import QUESTION

    key = labeller_key(args.labeller_model, QUESTION)
    store = LabelStore(args.store)
    items = {}
    for line in open(args.items, encoding="utf-8"):
        if line.strip():
            it = json.loads(line)
            items[it["item"]] = it
    n_new = n_have = n_orphan = 0
    for lab in json.load(open(args.labels, encoding="utf-8")):
        it = items.get(lab["item"])
        if it is None:
            n_orphan += 1
            continue
        th = text_hash(it["title"], it["abstract"])
        if store.get(it["id"], th, key):
            n_have += 1
            continue
        store.add({"id": it["id"], "text_hash": th, "labeller": key, "label": lab["label"],
                   "probs": lab["probs"], "tokens": lab.get("tokens"), "labelled_at": "imported"})
        n_new += 1
    print(f"imported {n_new} labels ({n_have} already in the store, {n_orphan} without a matching item) "
          f"-> {args.store} now holds {len(store)}; labeller key {key}")


def cmd_stats(args) -> None:
    store = LabelStore(args.store)
    by = {}
    for (_, _, lk), r in store.rows.items():
        by.setdefault(lk, {}).setdefault(r["label"], 0)
        by[lk][r["label"]] += 1
    print(f"{len(store)} labels in {args.store}")
    for lk, c in by.items():
        print(f"  {lk}: {c}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", type=Path, default=DEFAULT_STORE)
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("import", help="add labels from an existing {item,label,probs} file + its items file")
    i.add_argument("--items", type=Path, required=True)
    i.add_argument("--labels", type=Path, required=True)
    i.add_argument("--labeller-model", default="eikos-4b-int4-ov")
    i.set_defaults(fn=cmd_import)
    s = sub.add_parser("stats")
    s.set_defaults(fn=cmd_stats)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
