"""Embed candidate titles and abstracts with BGE-large-en-v1.5"""

import datetime as dt
import json
import os
import re

import numpy as np

from utils.config_digest import json_digest
from utils.openalex_utils import file_sha256

CORE_CONFIG = "config/sweden_quantum_core_and_outliers.json"
FEATURES_INPUT = "data/sweden_quantum_candidate_features.jsonl"
CORE_MANIFEST = "data/sweden_quantum_core_manifest.json"

MASKED_OUTPUT = "data/sweden_quantum_embeddings_masked.npy"
UNMASKED_OUTPUT = "data/sweden_quantum_embeddings_unmasked.npy"
IDS_OUTPUT = "data/sweden_quantum_embedding_work_ids.txt"
MANIFEST_OUTPUT = "data/sweden_quantum_embeddings_manifest.json"

MODEL_NAME = "BAAI/bge-large-en-v1.5"
MAX_TOKENS = 512
BATCH_SIZE = 64

# BGE reads ordinary prose, so title and abstract are joined as a sentence break rather
# than with the [SEP] marker SPECTER2 was trained to expect.
SEPARATOR = ". "


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def verify_features(features_path, core_manifest_path, config_path):
    manifest = load_json(core_manifest_path)
    artifact = (manifest.get("artifacts") or {}).get("candidate_features") or {}
    if artifact.get("path") != features_path:
        raise RuntimeError("Core manifest points to an unexpected feature table")
    if artifact.get("sha256") != file_sha256(features_path):
        raise RuntimeError(
            "Feature table hash does not match the core manifest; rerun "
            "src/06_build_quantum_core_set.py"
        )
    if not artifact.get("candidate_text_sha256"):
        raise RuntimeError(
            "Core manifest predates candidate_text_sha256; rerun step 06"
        )
    if manifest.get("core_rule_sha256") != json_digest(
        load_json(config_path).get("core_rule") or {}
    ):
        raise RuntimeError("The core set was built from a different core rule")
    return manifest


def read_records(path):
    ids = []
    titles = []
    abstracts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            ids.append(row["work_id"])
            titles.append(row.get("title") or "")
            abstracts.append(row.get("abstract") or "")
    if not ids:
        raise RuntimeError("Feature table is empty")
    return ids, titles, abstracts


def join_text(title, abstract):
    """One passage per work, with no dangling separator when a side is empty."""
    title = title.strip()
    abstract = abstract.strip()
    if not title or not abstract:
        return title or abstract
    if title[-1] in ".?!":
        return title + " " + abstract
    return title + SEPARATOR + abstract


def mask_terms(text, patterns):
    for pattern in patterns:
        text = pattern.sub(" ", text)
    return re.sub(r"\s{2,}", " ", text).strip()


def encode_all(texts, model, tokenizer, device, torch, label):
    """Return L2-normalized CLS embeddings for every text, in input order."""
    vectors = np.empty((len(texts), model.config.hidden_size), dtype=np.float32)
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    done = 0
    for start in range(0, len(order), BATCH_SIZE):
        positions = order[start : start + BATCH_SIZE]
        batch = [texts[position] or "." for position in positions]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_TOKENS,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            output = model(**encoded)
        cls = output.last_hidden_state[:, 0, :].float()
        cls = cls / cls.norm(dim=1, keepdim=True).clamp_min(1e-12)
        vectors[positions] = cls.cpu().numpy()
        done += len(positions)
        if done % (BATCH_SIZE * 40) == 0 or done == len(order):
            print(f"  {label}: {done}/{len(order)}", flush=True)
    return vectors


def embed_candidates(
    features_path=FEATURES_INPUT,
    core_manifest_path=CORE_MANIFEST,
    config_path=CORE_CONFIG,
    masked_path=MASKED_OUTPUT,
    unmasked_path=UNMASKED_OUTPUT,
    ids_path=IDS_OUTPUT,
    manifest_output=MANIFEST_OUTPUT,
):
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "PyTorch and transformers are required. Run this step with the project "
            "virtual environment: .venv/bin/python src/07_embed_candidate_texts.py"
        ) from exc

    core_manifest = verify_features(features_path, core_manifest_path, config_path)
    patterns = [
        re.compile(pattern, flags=re.IGNORECASE)
        for pattern in core_manifest["core_rule"]["core_term_patterns"]
    ]
    ids, titles, abstracts = read_records(features_path)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    unmasked_texts = [
        join_text(title, abstract) for title, abstract in zip(titles, abstracts)
    ]
    masked_texts = [
        join_text(mask_terms(title, patterns), mask_terms(abstract, patterns))
        for title, abstract in zip(titles, abstracts)
    ]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval().to(device)
    if device == "cuda":
        model.half()
    print(f"Model {MODEL_NAME} on {device}; {len(ids)} records", flush=True)

    masked = encode_all(masked_texts, model, tokenizer, device, torch, "masked")
    unmasked = encode_all(unmasked_texts, model, tokenizer, device, torch, "unmasked")

    tmp = {
        path: path + ".tmp"
        for path in (masked_path, unmasked_path, ids_path, manifest_output)
    }
    for path in tmp:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    try:
        for path, vectors in ((masked_path, masked), (unmasked_path, unmasked)):
            with open(tmp[path], "wb") as f:
                np.save(f, vectors.astype(np.float16))
        with open(tmp[ids_path], "w", encoding="utf-8") as f:
            f.write("\n".join(ids) + "\n")
        manifest = {
            "schema_version": 1,
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "model": MODEL_NAME,
            "pooling": "cls",
            "text_join": "title" + SEPARATOR + "abstract",
            "normalized": True,
            "stored_dtype": "float16",
            "max_tokens": MAX_TOKENS,
            "device": device,
            "dimensions": int(masked.shape[1]),
            "records": len(ids),
            "masked_core_terms": core_manifest["core_rule"]["core_term_patterns"],
            "inputs": {
                "candidate_features": {
                    "path": features_path,
                    "sha256": file_sha256(features_path),
                    "candidate_text_sha256": (
                        core_manifest["artifacts"]["candidate_features"][
                            "candidate_text_sha256"
                        ]
                    ),
                },
                "core_manifest": {
                    "path": core_manifest_path,
                    "sha256": file_sha256(core_manifest_path),
                },
            },
            "artifacts": {
                "masked_embeddings": {
                    "path": masked_path,
                    "sha256": file_sha256(tmp[masked_path]),
                },
                "unmasked_embeddings": {
                    "path": unmasked_path,
                    "sha256": file_sha256(tmp[unmasked_path]),
                },
                "work_ids": {"path": ids_path, "sha256": file_sha256(tmp[ids_path])},
            },
        }
        with open(tmp[manifest_output], "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.write("\n")
        for path in (masked_path, unmasked_path, ids_path, manifest_output):
            os.replace(tmp[path], path)
        return manifest
    except Exception:
        for path in tmp.values():
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
        raise


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/..")
    manifest = embed_candidates()
    print(f"Records:  {manifest['records']} x {manifest['dimensions']}")
    print(f"Masked:   {MASKED_OUTPUT}")
    print(f"Unmasked: {UNMASKED_OUTPUT}")
    print(f"Manifest: {MANIFEST_OUTPUT}")


if __name__ == "__main__":
    main()
