"""Run state for the SweMetrics enrichment: what has already been processed, so a run only does the rest.

One small JSON file (corpus/data/swe_state.json, git-ignored), written atomically after every step:

    {
      "snapshots":  {"oa": "2026-10-10", "swepub": "2026-10-23"},   # the SweMetrics snapshots last fully processed
      "textnet":    {"snapshot", "watermark", "slice", "done": [...], "last_full": "2026-11-27"},
      "stamps":     {"<stage>": "<fingerprint of the inputs it last ran on>"},
      "runs":       [{"started", "finished", "stats": {...}}, ...]   # the last 30
    }

Per-work progress is NOT kept here: embeddings live in the embedding caches, labels in label_store.py,
decisions in swe_decisions.jsonl, accepted records in swe_accepted_works.jsonl. This file only holds the
run-level bookkeeping (snapshots, harvest watermark, stage fingerprints).
"""

import hashlib
import json
import os
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STATE_PATH = HERE.parent / "corpus" / "data" / "swe_state.json"


def load(path: Path = STATE_PATH) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"snapshots": {}, "textnet": {}, "stamps": {}, "runs": []}


def save(state: dict, path: Path = STATE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def fingerprint(*parts) -> str:
    """A short hash of file paths (hashed by size + mtime) and plain values: the inputs a stage ran on."""
    h = hashlib.sha1()
    for p in parts:
        if isinstance(p, Path) or (isinstance(p, str) and os.path.exists(p)):
            if os.path.exists(p):
                st = os.stat(p)
                h.update(f"{p}:{st.st_size}:{int(st.st_mtime)}".encode())
            else:
                h.update(f"{p}:missing".encode())
        else:
            h.update(repr(p).encode())
    return h.hexdigest()[:16]


def record_run(state: dict, started: float, stats: dict) -> None:
    state.setdefault("runs", []).append({
        "started": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(started)),
        "finished": time.strftime("%Y-%m-%dT%H:%M:%S"), "seconds": round(time.time() - started), "stats": stats,
    })
    state["runs"] = state["runs"][-30:]
