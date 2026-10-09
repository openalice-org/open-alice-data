"""Unified token clustering for papers and patents."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize


@dataclass
class ClusterResult:
    token_to_cluster: dict[str, int]
    cluster_labels: dict[int, str]  # id → "tok1/tok2/tok3"
    doc_clusters: dict[str, set[int]]  # doc_id → {cluster_ids}
    n_clusters: int


def cluster_documents(
    docs: list[dict[str, Any]],
    max_features: int = 500,
    k_range: range = range(5, 41, 5),
) -> ClusterResult:
    """Cluster tokens via doc-membership vectors and KMeans with auto-k selection."""
    assert all("doc_id" in d and "tokens" in d for d in docs), "docs must have doc_id and tokens"

    # Build token → doc indices mapping
    token_to_docs: dict[str, set[int]] = {}
    for idx, doc in enumerate(docs):
        for token in doc.get("tokens", []):
            token_to_docs.setdefault(token, set()).add(idx)

    # Vocabulary: top max_features tokens by document frequency
    vocab = sorted(token_to_docs.keys(), key=lambda t: len(token_to_docs[t]), reverse=True)[:max_features]
    vocab_set = set(vocab)
    token_to_idx = {t: i for i, t in enumerate(vocab)}

    n_docs = len(docs)
    n_vocab = len(vocab)

    assert n_vocab > 0, "Empty vocabulary — cannot cluster"

    # Build binary matrix: rows = tokens, cols = docs
    X = np.zeros((n_vocab, n_docs), dtype=np.float32)
    for token, doc_indices in token_to_docs.items():
        if token in vocab_set:
            row_idx = token_to_idx[token]
            for doc_idx in doc_indices:
                X[row_idx, doc_idx] = 1.0

    # L2-normalize rows (token vectors)
    X = normalize(X, norm="l2", axis=1)

    # Auto-k sweep
    best_score = -1.0
    best_labels: np.ndarray | None = None
    best_k = k_range.start

    for k in k_range:
        if k >= n_vocab:
            break
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(X, labels)
        if score > best_score:
            best_score = score
            best_labels = labels
            best_k = k

    assert best_labels is not None, "KMeans failed to produce any valid clustering"

    # Token → cluster
    token_to_cluster = {token: int(best_labels[token_to_idx[token]]) for token in vocab}

    # Cluster label = top-3 tokens by doc frequency in cluster
    cluster_token_counts: dict[int, Counter[str]] = {}
    for token, cluster_id in token_to_cluster.items():
        counter = cluster_token_counts.setdefault(cluster_id, Counter())
        counter[token] = len(token_to_docs[token])

    cluster_labels: dict[int, str] = {}
    for cluster_id, counter in cluster_token_counts.items():
        top3 = [t for t, _ in counter.most_common(3)]
        cluster_labels[cluster_id] = "/".join(top3)

    # Doc → clusters (simple threshold: ≥1 token from cluster)
    doc_clusters: dict[str, set[int]] = {}
    for doc in docs:
        doc_id = doc["doc_id"]
        clusters: set[int] = set()
        for token in doc.get("tokens", []):
            if token in vocab_set:
                clusters.add(token_to_cluster[token])
        doc_clusters[doc_id] = clusters

    return ClusterResult(
        token_to_cluster=token_to_cluster,
        cluster_labels=cluster_labels,
        doc_clusters=doc_clusters,
        n_clusters=best_k,
    )


def write_clusters_csv(
    cluster_result: ClusterResult,
    paper_doc_ids: set[str],
    patent_doc_ids: set[str],
    output_path: Path,
) -> None:
    """Write cluster metadata to CSV."""
    import csv

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Count docs per cluster
    paper_counts: dict[int, int] = {i: 0 for i in range(cluster_result.n_clusters)}
    patent_counts: dict[int, int] = {i: 0 for i in range(cluster_result.n_clusters)}
    for doc_id, clusters in cluster_result.doc_clusters.items():
        for c in clusters:
            if doc_id in paper_doc_ids:
                paper_counts[c] += 1
            elif doc_id in patent_doc_ids:
                patent_counts[c] += 1

    fieldnames = ["cluster_id", "label", "members", "paper_doc_count", "patent_doc_count"]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for cid in sorted(cluster_result.cluster_labels.keys()):
            members = [t for t, c in cluster_result.token_to_cluster.items() if c == cid]
            writer.writerow({
                "cluster_id": cid,
                "label": cluster_result.cluster_labels[cid],
                "members": " ".join(members),
                "paper_doc_count": paper_counts[cid],
                "patent_doc_count": patent_counts[cid],
            })


if __name__ == "__main__":
    # Inline assert with synthetic corpus
    docs = [
        {"doc_id": "p1", "tokens": ["quantum", "computing", "qubit"]},
        {"doc_id": "p2", "tokens": ["quantum", "computing", "algorithm"]},
        {"doc_id": "t1", "tokens": ["quantum", "device", "superconducting"]},
        {"doc_id": "t2", "tokens": ["qubit", "superconducting", "circuit"]},
    ]
    result = cluster_documents(docs, max_features=50, k_range=range(2, 5))
    assert result.n_clusters >= 2
    assert len(result.token_to_cluster) > 0
    assert all(isinstance(v, int) for v in result.token_to_cluster.values())
    assert all(isinstance(v, str) for v in result.cluster_labels.values())
    assert set(result.doc_clusters.keys()) == {"p1", "p2", "t1", "t2"}
    print("cluster.py inline assertions passed")
