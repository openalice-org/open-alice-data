"""Place every embedded work on a relative affinity to three quantum research areas."""

import csv
import json
import os

import numpy as np

from utils.openalex_utils import file_sha256

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANCHOR_CONFIG = "config/sweden_quantum_area_anchors.json"
EMBEDDING_MANIFEST = "data/sweden_quantum_embeddings_manifest.json"
EMBEDDINGS = "data/sweden_quantum_embeddings_unmasked.npy"
WORK_IDS = "data/sweden_quantum_embedding_work_ids.txt"
SCORES = "data/sweden_quantum_outlier_scores.csv"
OUTLIER_MANIFEST = "data/sweden_quantum_outlier_manifest.json"
OUTPUT = "data/sweden_quantum_area_affinities.csv"
PROTOTYPES_OUTPUT = "data/sweden_quantum_area_prototypes.npy"
MANIFEST_OUTPUT = "data/sweden_quantum_area_affinities_manifest.json"

MAX_TOKENS = 512  # the same truncation step 07 applied to the works


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def verify(path, manifest, keys):
    node = manifest
    for key in keys:
        node = node[key]
    if node.get("sha256") != file_sha256(path):
        raise RuntimeError(f"{path} does not match its manifest hash; rerun the stage")


def embed_anchors(texts, model_name):
    """L2-normalized CLS embeddings, the same pooling and truncation step 07 used.

    The unmasked works were encoded as plain passages with no instruction prefix, so the
    anchors are too: a prefix on one side only would tilt every cosine in the same
    direction.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    encoded = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=MAX_TOKENS,
        return_tensors="pt",
    )
    with torch.no_grad():
        output = model(**encoded)
    cls = output.last_hidden_state[:, 0, :].float()
    cls = cls / cls.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return cls.numpy().astype(np.float32)


def prototypes_from(anchor_vectors, group_sizes):
    """One prototype per area: the mean of its anchor vectors, renormalized."""
    rows = []
    start = 0
    for size in group_sizes:
        mean = anchor_vectors[start : start + size].mean(axis=0)
        rows.append(mean / max(float(np.linalg.norm(mean)), 1e-12))
        start += size
    return np.vstack(rows).astype(np.float32)


def temperature_softmax(values, temperature):
    scaled = np.asarray(values, dtype=np.float64) / temperature
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    weights = np.exp(scaled)
    return weights / weights.sum(axis=1, keepdims=True)


def normalized_entropy(weights):
    """Entropy in [0, 1]: 0 = all weight on one area, 1 = an even three-way split."""
    weights = np.clip(np.asarray(weights, dtype=np.float64), 1e-12, None)
    entropy = -(weights * np.log(weights)).sum(axis=1)
    return entropy / np.log(weights.shape[1])


def read_work_context(path):
    """Year, inclusion and verdict per work, so the affinity table stands on its own."""
    context = {}
    with open(path, encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            context[row["work_id"]] = (
                row["publication_year"],
                "true" if row["currently_included"] in ("true", "True") else "false",
                "true" if row["is_core"] in ("true", "True") else "false",
            )
    return context


def score_affinities():
    config = load_json(ANCHOR_CONFIG)
    embedding_manifest = load_json(EMBEDDING_MANIFEST)
    outlier_manifest = load_json(OUTLIER_MANIFEST)

    verify(EMBEDDINGS, embedding_manifest, ("artifacts", "unmasked_embeddings"))
    verify(WORK_IDS, embedding_manifest, ("artifacts", "work_ids"))
    verify(SCORES, outlier_manifest, ("artifacts", "outlier_scores"))
    if embedding_manifest["model"] != config["encoder"]["model"]:
        raise RuntimeError(
            f"The works were embedded with {embedding_manifest['model']} but the anchors "
            f"are configured for {config['encoder']['model']}; the cosines would be "
            f"between two different spaces"
        )
    if not embedding_manifest["normalized"]:
        raise RuntimeError("The stored work embeddings are not L2-normalized")

    areas = config["areas"]
    texts = [text for area in areas for text in area["anchors"]]
    sizes = [len(area["anchors"]) for area in areas]
    anchor_vectors = embed_anchors(texts, config["encoder"]["model"])
    prototypes = prototypes_from(anchor_vectors, sizes)

    with open(WORK_IDS, encoding="utf-8") as handle:
        ids = [line.strip() for line in handle if line.strip()]
    vectors = np.load(EMBEDDINGS).astype(np.float32)
    if vectors.shape[0] != len(ids):
        raise RuntimeError(
            "The embedding matrix and the work-ID list disagree on length"
        )

    # Both sides are unit vectors, so the dot product is the cosine.
    similarities = vectors @ prototypes.T
    affinities = temperature_softmax(similarities, config["softmax_temperature"])
    entropy = normalized_entropy(affinities)
    leaning = affinities.argmax(axis=1)
    ordered = np.sort(affinities, axis=1)
    margin = ordered[:, -1] - ordered[:, -2]

    context = read_work_context(SCORES)
    short_names = [area["short_name"] for area in areas]
    header = ["work_id", "publication_year", "currently_included", "is_core"]
    header += [f"cosine_{name}" for name in short_names]
    header += [f"affinity_{name}" for name in short_names]
    header += ["leaning", "affinity_margin", "normalized_entropy"]

    missing_context = 0
    tmp = OUTPUT + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for index, work_id in enumerate(ids):
            year, included, is_core = context.get(work_id, ("", "false", "false"))
            if work_id not in context:
                missing_context += 1
            writer.writerow(
                [work_id, year, included, is_core]
                + [f"{value:.6f}" for value in similarities[index]]
                + [f"{value:.6f}" for value in affinities[index]]
                + [
                    short_names[leaning[index]],
                    f"{margin[index]:.6f}",
                    f"{entropy[index]:.6f}",
                ]
            )
    os.replace(tmp, OUTPUT)
    np.save(PROTOTYPES_OUTPUT, prototypes)

    manifest = {
        "schema_version": 1,
        "artifacts": {
            "affinities": {
                "path": OUTPUT,
                "row_count": len(ids),
                "sha256": file_sha256(OUTPUT),
            },
            "prototypes": {
                "path": PROTOTYPES_OUTPUT,
                "sha256": file_sha256(PROTOTYPES_OUTPUT),
            },
        },
        "inputs": {
            "anchor_config": {
                "path": ANCHOR_CONFIG,
                "sha256": file_sha256(ANCHOR_CONFIG),
            },
            "embedding_manifest": {
                "path": EMBEDDING_MANIFEST,
                "sha256": file_sha256(EMBEDDING_MANIFEST),
            },
            "outlier_scores": {"path": SCORES, "sha256": file_sha256(SCORES)},
        },
        "model": config["encoder"]["model"],
        "softmax_temperature": config["softmax_temperature"],
        "areas": [
            {
                "name": area["name"],
                "short_name": area["short_name"],
                "anchors": len(area["anchors"]),
            }
            for area in areas
        ],
        "counts": {
            "works": len(ids),
            "works_without_score_row": missing_context,
            "leanings": {
                name: int((leaning == index).sum())
                for index, name in enumerate(short_names)
            },
        },
        "median_top_two_cosine_gap": float(
            np.median(
                np.sort(similarities, axis=1)[:, -1]
                - np.sort(similarities, axis=1)[:, -2]
            )
        ),
    }
    tmp = MANIFEST_OUTPUT + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, MANIFEST_OUTPUT)
    return manifest


def main():
    os.chdir(ROOT)
    manifest = score_affinities()
    print(f"Affinities:  {OUTPUT} ({manifest['counts']['works']:,} works)")
    for name, count in manifest["counts"]["leanings"].items():
        print(f"  leaning {name:7s} {count:,}")
    print(f"Median top-two cosine gap: {manifest['median_top_two_cosine_gap']:.4f}")


if __name__ == "__main__":
    main()
