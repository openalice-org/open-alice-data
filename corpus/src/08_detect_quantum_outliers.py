"""Score every candidate work by its distance from the explicitly-quantum core."""

import csv
import datetime as dt
import json
import os
import re

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.preprocessing import normalize

from utils.config_digest import json_digest
from utils.openalex_utils import file_sha256

CORE_CONFIG = "config/sweden_quantum_core_and_outliers.json"
FEATURES_INPUT = "data/sweden_quantum_candidate_features.jsonl"
CORE_MANIFEST = "data/sweden_quantum_core_manifest.json"
AUDIT_INPUT = "data/sweden_quantum_candidate_audit.csv"
FILTER_MANIFEST = "data/sweden_quantum_works_manifest.json"
EMBEDDING_MANIFEST = "data/sweden_quantum_embeddings_manifest.json"

SCORES_OUTPUT = "data/sweden_quantum_outlier_scores.csv"
CLUSTERS_OUTPUT = "data/sweden_quantum_outlier_clusters.csv"
CENTERS_OUTPUT = "data/sweden_quantum_core_centers.csv"
MANIFEST_OUTPUT = "data/sweden_quantum_outlier_manifest.json"

SCORE_FIELDS = (
    "work_id",
    "publication_year",
    "title",
    "work_type",
    "primary_topic_name",
    "primary_topic_domain",
    "is_core",
    "currently_included",
    "verdict",
    "views_used",
    "low_text_evidence",
    "text_distance",
    "text_percentile",
    "topic_distance",
    "topic_percentile",
    "nearest_center",
    "nearest_center_label",
    "unmasked_text_percentile",
    "local_outlier_factor",
    "local_outlier_factor_flag",
    "far_cluster",
    "nearest_core_works",
)

CENTER_FIELDS = (
    "view",
    "center",
    "kept",
    "core_works",
    "label",
    "median_core_distance",
    "assigned_candidates",
    "assigned_currently_included",
    "top_terms",
    "top_primary_topics",
    "example_titles",
)

CLUSTER_FIELDS = (
    "cluster",
    "works",
    "currently_included",
    "top_terms",
    "top_primary_topics",
    "example_titles",
)

VERDICT_CORE = "core"
VERDICT_NEAR = "near_core"
VERDICT_BORDERLINE = "borderline"
VERDICT_FAR = "far_outlier"
VERDICT_UNSCORED = "unscored"
VERDICTS = (
    VERDICT_CORE,
    VERDICT_NEAR,
    VERDICT_BORDERLINE,
    VERDICT_FAR,
    VERDICT_UNSCORED,
)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def validate_config(config):
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported core/outlier schema_version")
    text = config.get("text_view") or {}
    topic = config.get("topic_view") or {}
    model = config.get("outlier_model") or {}
    if text.get("backend") not in ("embeddings", "tfidf"):
        raise ValueError("text_view.backend must be 'embeddings' or 'tfidf'")
    if not isinstance(text.get("svd_components"), int) or text["svd_components"] < 2:
        raise ValueError("text_view.svd_components must be an integer of at least 2")
    for key in ("minimum_text_tokens", "text_evidence_floor_tokens"):
        if not isinstance(text.get(key), int) or text[key] < 1:
            raise ValueError(f"text_view.{key} must be a positive integer")
    if text["text_evidence_floor_tokens"] > text["minimum_text_tokens"]:
        raise ValueError("The text floor cannot exceed minimum_text_tokens")
    if text.get("mask_core_terms") is not True:
        raise ValueError(
            "text_view.mask_core_terms must be true for a non-circular score"
        )
    if topic.get("mask_explicit_quantum_topics") is not True:
        raise ValueError("topic_view.mask_explicit_quantum_topics must be true")
    if not isinstance(model.get("k_neighbors"), int) or model["k_neighbors"] < 1:
        raise ValueError("outlier_model.k_neighbors must be a positive integer")
    borderline = model.get("borderline_percentile")
    far = model.get("far_percentile")
    for value in (borderline, far):
        if not isinstance(value, (int, float)) or not 0 < value < 100:
            raise ValueError("Percentile bands must be between 0 and 100")
    if borderline >= far:
        raise ValueError("borderline_percentile must be below far_percentile")
    if not isinstance(model.get("lof_neighbors"), int) or model["lof_neighbors"] < 2:
        raise ValueError("outlier_model.lof_neighbors must be an integer of at least 2")
    if not isinstance(model.get("random_state"), int):
        raise ValueError("outlier_model.random_state must be an integer")
    validate_center_settings(config.get("core_centers") or {})


def validate_center_settings(centers):
    """The centres label a work's region; they never decide a verdict."""
    if centers.get("method") != "spherical_kmeans":
        raise ValueError("core_centers.method must be 'spherical_kmeans'")
    fixed = centers.get("n_centers")
    if fixed is not None and (not isinstance(fixed, int) or fixed < 2):
        raise ValueError(
            "core_centers.n_centers must be null or an integer of at least 2"
        )
    grid = centers.get("center_count_grid")
    if fixed is None:
        if not isinstance(grid, list) or not grid:
            raise ValueError("core_centers.center_count_grid must be a non-empty list")
        if any(not isinstance(value, int) or value < 2 for value in grid):
            raise ValueError("Every centre count must be an integer of at least 2")
    for key in (
        "minimum_center_works",
        "silhouette_sample",
        "top_terms",
        "example_titles",
    ):
        if not isinstance(centers.get(key), int) or centers[key] < 1:
            raise ValueError(f"core_centers.{key} must be a positive integer")
    if not isinstance(centers.get("random_state"), int):
        raise ValueError("core_centers.random_state must be an integer")


def verify_inputs(features_path, core_manifest_path, config_path):
    manifest = load_json(core_manifest_path)
    artifact = (manifest.get("artifacts") or {}).get("candidate_features") or {}
    if artifact.get("path") != features_path:
        raise RuntimeError("Core manifest points to an unexpected feature table")
    if artifact.get("sha256") != file_sha256(features_path):
        raise RuntimeError(
            "Feature table hash does not match the core manifest; rerun "
            "src/06_build_quantum_core_set.py"
        )
    if manifest.get("core_config") != config_path:
        raise RuntimeError("Core manifest points to an unexpected configuration file")
    if manifest.get("core_rule_sha256") != json_digest(
        load_json(config_path).get("core_rule") or {}
    ):
        raise RuntimeError(
            "The core set was built from a different core rule; rerun "
            "src/06_build_quantum_core_set.py"
        )
    return manifest


def load_inclusion_flags(audit_path, filter_manifest_path):
    """Read the current production decision so the score can be compared with it."""
    manifest = load_json(filter_manifest_path)
    artifact = (manifest.get("artifacts") or {}).get("candidate_audit") or {}
    if artifact.get("path") != audit_path:
        raise RuntimeError("Filter manifest points to an unexpected candidate audit")
    if artifact.get("sha256") != file_sha256(audit_path):
        raise RuntimeError(
            "Candidate audit hash does not match its manifest; rerun "
            "src/05_filter_sweden_quantum_candidates.py"
        )
    flags = {}
    with open(audit_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            flags[row["work_id"]] = row["included"].strip().casefold() == "true"
    return flags, manifest


def read_features(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Invalid JSON in {path} line {line_number}"
                ) from exc
    if not rows:
        raise RuntimeError("Feature table is empty")
    return rows


def work_text(row, fields):
    parts = []
    for field in fields:
        if field == "keywords":
            parts.append(" ".join(row.get("keywords") or []))
        else:
            parts.append(row.get(field) or "")
    return "\n".join(parts)


def mask_terms(text, patterns):
    for pattern in patterns:
        text = pattern.sub(" ", text)
    return text


def build_text_view(texts, settings, random_state):
    """Return (tfidf matrix, feature names, L2-normalized SVD embedding)."""
    vectorizer = TfidfVectorizer(
        strip_accents="unicode",
        lowercase=True,
        stop_words="english",
        ngram_range=(1, settings["ngram_max"]),
        min_df=settings["min_df"],
        max_df=settings["max_df"],
        max_features=settings["max_features"],
        sublinear_tf=bool(settings.get("sublinear_tf", True)),
    )
    matrix = vectorizer.fit_transform(texts)
    components = min(settings["svd_components"], min(matrix.shape) - 1)
    svd = TruncatedSVD(n_components=components, random_state=random_state)
    embedding = normalize(svd.fit_transform(matrix))
    return (
        matrix,
        vectorizer.get_feature_names_out(),
        embedding,
        svd.explained_variance_ratio_.sum(),
    )


def build_label_view(texts, settings):
    """A light TF-IDF used only to name the far clusters with their strongest terms."""
    vectorizer = TfidfVectorizer(
        strip_accents="unicode",
        lowercase=True,
        stop_words="english",
        ngram_range=(1, settings["ngram_max"]),
        min_df=settings["cluster_label_min_df"],
        max_features=settings["cluster_label_max_features"],
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(texts)
    return matrix, vectorizer.get_feature_names_out()


def load_embeddings(work_ids, manifest_path, text_digest):
    """Load the step-21 embedding matrices and confirm they describe these texts.

    The check is against the identifier/title/abstract digest rather than the whole
    feature file, so changing the core rule does not force the GPU step to be rerun.
    """
    manifest = load_json(manifest_path)
    if manifest.get("schema_version") != 1:
        raise RuntimeError("Unsupported embedding manifest schema_version")
    features = (manifest.get("inputs") or {}).get("candidate_features") or {}
    if features.get("candidate_text_sha256") != text_digest:
        raise RuntimeError(
            "The embeddings were built from different candidate texts; rerun "
            ".venv/bin/python src/07_embed_candidate_texts.py"
        )
    artifacts = manifest.get("artifacts") or {}
    stored_ids = open(artifacts["work_ids"]["path"], encoding="utf-8").read().split()
    if stored_ids != list(work_ids):
        raise RuntimeError("Embedding work IDs do not match the feature table order")
    matrices = []
    for key in ("masked_embeddings", "unmasked_embeddings"):
        path = artifacts[key]["path"]
        if artifacts[key]["sha256"] != file_sha256(path):
            raise RuntimeError(f"{path} does not match its embedding manifest hash")
        with open(path, "rb") as f:
            matrices.append(normalize(np.load(f).astype(np.float32)))
    if matrices[0].shape[0] != len(work_ids):
        raise RuntimeError("Embedding row count does not match the feature table")
    return matrices[0], matrices[1], manifest


def build_topic_view(rows, masked_topic_ids):
    """Return (L2-normalized sparse topic-score matrix, available mask, dimension count)."""
    columns = {}
    for row in rows:
        for topic_id, _ in row.get("topics") or []:
            if topic_id in masked_topic_ids:
                continue
            if topic_id not in columns:
                columns[topic_id] = len(columns)
    indptr = [0]
    indices = []
    values = []
    for row in rows:
        seen = {}
        for topic_id, score in row.get("topics") or []:
            column = columns.get(topic_id)
            if column is None or score <= 0:
                continue
            seen[column] = max(seen.get(column, 0.0), float(score))
        for column, score in sorted(seen.items()):
            indices.append(column)
            values.append(score)
        indptr.append(len(indices))
    matrix = csr_matrix(
        (np.asarray(values, dtype=np.float64), np.asarray(indices), np.asarray(indptr)),
        shape=(len(rows), max(len(columns), 1)),
    )
    available = np.asarray(matrix.getnnz(axis=1) > 0)
    return normalize(matrix), available, len(columns)


def neighbour_distances(embedding, core_index, k, return_neighbours=False):
    """Mean cosine distance from every row to its k nearest core rows.

    Core rows exclude themselves by discarding their first neighbour.
    """
    core_matrix = embedding[core_index]
    model = NearestNeighbors(
        n_neighbors=min(k + 1, core_matrix.shape[0]), metric="cosine"
    )
    model.fit(core_matrix)
    distances, neighbours = model.kneighbors(embedding)
    is_core = np.zeros(embedding.shape[0], dtype=bool)
    is_core[core_index] = True
    kept = np.empty((embedding.shape[0], k), dtype=np.float64)
    kept_neighbours = np.empty((embedding.shape[0], k), dtype=np.int64)
    for position in range(embedding.shape[0]):
        row_distances = distances[position]
        row_neighbours = neighbours[position]
        if is_core[position]:
            row_distances = row_distances[1:]
            row_neighbours = row_neighbours[1:]
        kept[position] = row_distances[:k]
        kept_neighbours[position] = row_neighbours[:k]
    scores = kept.mean(axis=1)
    if return_neighbours:
        return scores, core_index[kept_neighbours]
    return scores


def fit_core_centers(matrix, reference_index, settings, view_name):
    """Cluster the core into centres and keep the ones with real support behind them.

    The rows are L2-normalized, so a Euclidean k-means orders points the same way cosine
    does; the centroids are re-normalized afterwards to make the cosine distance to a
    centre well defined.  With n_centers set, that count is used and its silhouette is
    still reported; with n_centers null the configured grid is searched instead, which is
    how the silhouette was found to express no preference on this core.
    """
    reference = matrix[reference_index]
    if settings["n_centers"] is not None:
        counts = [settings["n_centers"]]
    else:
        counts = sorted(set(settings["center_count_grid"]))
    counts = [count for count in counts if 2 <= count < reference.shape[0]]
    if not counts:
        raise RuntimeError(f"No usable centre count for the {view_name} view")

    sample = min(settings["silhouette_sample"], reference.shape[0])
    grid = []
    best = None
    for count in counts:
        kmeans = KMeans(
            n_clusters=count, random_state=settings["random_state"], n_init=10
        )
        labels = kmeans.fit_predict(reference)
        score = float(
            silhouette_score(
                reference,
                labels,
                metric="cosine",
                sample_size=sample,
                random_state=settings["random_state"],
            )
        )
        grid.append({"centers": count, "silhouette": round(score, 6)})
        print(f"  {view_name} view: {count:3d} centres  silhouette {score:+.4f}")
        if best is None or score > best["silhouette"]:
            best = {
                "silhouette": score,
                "selected": count,
                "labels": labels,
                "centers": np.asarray(kmeans.cluster_centers_, dtype=np.float64),
            }

    sizes = np.bincount(best["labels"], minlength=best["selected"])
    keep = sizes >= settings["minimum_center_works"]
    if int(keep.sum()) < 2:
        print(
            f"  {view_name} view: minimum_center_works would leave fewer than two "
            "centres, so every centre is kept"
        )
        keep = np.ones(best["selected"], dtype=bool)
    dropped = [
        {"center": int(center), "core_works": int(sizes[center])}
        for center in np.flatnonzero(~keep)
    ]
    if dropped:
        print(
            f"  {view_name} view: dropped {len(dropped)} centre(s) below "
            f"{settings['minimum_center_works']} core works "
            f"({sum(item['core_works'] for item in dropped)} works)"
        )
    return {
        "centers": normalize(best["centers"][keep]),
        "kept_ids": np.flatnonzero(keep),
        "labels": best["labels"],
        "sizes": sizes,
        "selected": best["selected"],
        "silhouette": best["silhouette"],
        "grid": grid,
        "dropped": dropped,
        "reference_index": reference_index,
    }


def center_distances(matrix, centers):
    """Cosine distance from every row to its nearest centre, and which centre that is.

    Both the rows and the centres are L2-normalized, so the cosine distance is one minus
    the dot product and the whole scoring step is a single matrix product.
    """
    similarity = np.asarray(matrix @ centers.T)
    nearest = np.argmax(similarity, axis=1)
    distance = 1.0 - similarity[np.arange(similarity.shape[0]), nearest]
    return np.clip(distance, 0.0, 2.0), nearest


def describe_centers(
    view_name,
    result,
    rows,
    label_matrix,
    feature_names,
    nearest,
    distance,
    included,
    settings,
):
    """One row per centre: what it is made of, and what it went on to attract."""
    slot_of = {int(center): slot for slot, center in enumerate(result["kept_ids"])}
    summaries = []
    for center in range(result["selected"]):
        members = result["reference_index"][np.flatnonzero(result["labels"] == center)]
        centroid = np.asarray(label_matrix[members].mean(axis=0)).ravel()
        ranked = np.argsort(centroid)[::-1][: settings["top_terms"]]
        terms = [str(feature_names[index]) for index in ranked]
        topics = {}
        for position in members:
            name = rows[position]["primary_topic_name"] or "(none)"
            topics[name] = topics.get(name, 0) + 1
        ranked_topics = sorted(topics.items(), key=lambda item: -item[1])[:5]
        slot = slot_of.get(center)
        assigned = (
            np.flatnonzero(nearest == slot)
            if slot is not None
            else np.zeros(0, dtype=int)
        )
        summaries.append(
            {
                "view": view_name,
                "center": int(center),
                "kept": str(slot is not None).lower(),
                "core_works": int(members.size),
                "label": " ".join(terms[:3]),
                "median_core_distance": round(float(np.median(distance[members])), 6),
                "assigned_candidates": int(assigned.size),
                "assigned_currently_included": int(included[assigned].sum()),
                "top_terms": " | ".join(terms),
                "top_primary_topics": " | ".join(
                    f"{name} ({count})" for name, count in ranked_topics
                ),
                "example_titles": " || ".join(
                    (rows[position]["title"] or "")[:90]
                    for position in members[: settings["example_titles"]]
                ),
            }
        )
    return summaries


def percentile_of(core_scores, scores):
    """Percentile of each score within the sorted core score distribution."""
    reference = np.sort(core_scores)
    positions = np.searchsorted(reference, scores, side="right")
    return 100.0 * positions / reference.size


def band_of(percentile, borderline, far):
    if percentile > far:
        return VERDICT_FAR
    if percentile > borderline:
        return VERDICT_BORDERLINE
    return VERDICT_NEAR


def decide_verdicts(
    is_core, text_pct, topic_pct, text_usable, topic_usable, borderline, far
):
    """Combine the two views.

    A work is far only when both usable views agree, so a single blunt signal cannot
    condemn a paper on its own.  A record with too little text is judged on its topic
    profile, and a record with neither usable view is left unscored rather than being
    silently called near the core.
    """
    verdicts = np.empty(is_core.size, dtype=object)
    views = np.empty(is_core.size, dtype=object)
    for position in range(is_core.size):
        has_text = bool(text_usable[position])
        has_topic = bool(topic_usable[position])
        views[position] = (
            "text+topic"
            if has_text and has_topic
            else "text"
            if has_text
            else "topic"
            if has_topic
            else "none"
        )
        if is_core[position]:
            verdicts[position] = VERDICT_CORE
            continue
        if has_text and has_topic:
            text_band = band_of(text_pct[position], borderline, far)
            topic_band = band_of(topic_pct[position], borderline, far)
            if text_band == VERDICT_FAR and topic_band == VERDICT_FAR:
                verdicts[position] = VERDICT_FAR
            elif VERDICT_FAR in (text_band, topic_band) or (
                text_band == VERDICT_BORDERLINE and topic_band == VERDICT_BORDERLINE
            ):
                verdicts[position] = VERDICT_BORDERLINE
            else:
                verdicts[position] = VERDICT_NEAR
        elif has_text:
            verdicts[position] = band_of(text_pct[position], borderline, far)
        elif has_topic:
            verdicts[position] = band_of(topic_pct[position], borderline, far)
        else:
            verdicts[position] = VERDICT_UNSCORED
    return verdicts, views


def describe_clusters(labels, rows, matrix, feature_names, included, model):
    """Summarise each far cluster by its strongest terms and primary topics."""
    summaries = []
    for cluster in sorted(set(labels.tolist())):
        members = np.flatnonzero(labels == cluster)
        centroid = np.asarray(matrix[members].mean(axis=0)).ravel()
        top = np.argsort(centroid)[::-1][: model["far_cluster_top_terms"]]
        topics = {}
        for position in members:
            name = rows[position]["primary_topic_name"] or "(none)"
            topics[name] = topics.get(name, 0) + 1
        ranked_topics = sorted(topics.items(), key=lambda item: -item[1])[:5]
        summaries.append(
            {
                "cluster": int(cluster),
                "works": int(members.size),
                "currently_included": int(sum(1 for p in members if included[p])),
                "top_terms": " | ".join(str(feature_names[index]) for index in top),
                "top_primary_topics": " | ".join(
                    f"{name} ({count})" for name, count in ranked_topics
                ),
                "example_titles": " || ".join(
                    (rows[position]["title"] or "")[:90]
                    for position in members[: model["neighbor_examples"]]
                ),
            }
        )
    return summaries


def detect_outliers(
    features_path=FEATURES_INPUT,
    core_manifest_path=CORE_MANIFEST,
    config_path=CORE_CONFIG,
    audit_path=AUDIT_INPUT,
    filter_manifest_path=FILTER_MANIFEST,
    scores_path=SCORES_OUTPUT,
    clusters_path=CLUSTERS_OUTPUT,
    centers_path=CENTERS_OUTPUT,
    manifest_output=MANIFEST_OUTPUT,
):
    config = load_json(config_path)
    validate_config(config)
    core_manifest = verify_inputs(features_path, core_manifest_path, config_path)
    inclusion_flags, filter_manifest = load_inclusion_flags(
        audit_path, filter_manifest_path
    )

    rows = read_features(features_path)
    missing = [row["work_id"] for row in rows if row["work_id"] not in inclusion_flags]
    if missing:
        raise RuntimeError(
            f"{len(missing)} feature rows are absent from the candidate audit; "
            "the audit and the feature table describe different candidate sets"
        )
    included = np.array([inclusion_flags[row["work_id"]] for row in rows], dtype=bool)
    is_core = np.array([bool(row["is_core"]) for row in rows], dtype=bool)
    core_index = np.flatnonzero(is_core)
    model = config["outlier_model"]
    if core_index.size <= model["k_neighbors"]:
        raise RuntimeError("The core set is too small for the configured k_neighbors")

    text_settings = config["text_view"]
    core_patterns = [
        re.compile(pattern, flags=re.IGNORECASE)
        for pattern in core_manifest["core_rule"]["core_term_patterns"]
    ]
    raw_texts = [work_text(row, text_settings["fields"]) for row in rows]
    masked_texts = [mask_terms(text, core_patterns) for text in raw_texts]

    print(
        f"Rows {len(rows)}  core {core_index.size}  currently included {int(included.sum())}"
    )
    backend = text_settings["backend"]
    masked_variance = None
    embedding_manifest = None
    if backend == "embeddings":
        print("Loading the embedded text view ...")
        masked_embedding, unmasked_embedding, embedding_manifest = load_embeddings(
            [row["work_id"] for row in rows],
            EMBEDDING_MANIFEST,
            core_manifest["artifacts"]["candidate_features"]["candidate_text_sha256"],
        )
        print(
            f"  model {embedding_manifest['model']}  dims {masked_embedding.shape[1]}"
        )
        label_matrix, feature_names = build_label_view(masked_texts, text_settings)
    else:
        print("Building the masked TF-IDF text view ...")
        label_matrix, feature_names, masked_embedding, masked_variance = (
            build_text_view(masked_texts, text_settings, model["random_state"])
        )
        print(
            f"  terms {label_matrix.shape[1]}  svd dims {masked_embedding.shape[1]}  "
            f"explained variance {masked_variance:.3f}"
        )
        print("Building the unmasked TF-IDF view (lexical diagnostic) ...")
        _, _, unmasked_embedding, _ = build_text_view(
            raw_texts, text_settings, model["random_state"]
        )
    own_tokens = np.array(
        [
            len(mask_terms(f"{row['title']} {row['abstract']}", core_patterns).split())
            for row in rows
        ]
    )
    low_text_evidence = own_tokens < text_settings["minimum_text_tokens"]
    text_usable = own_tokens >= text_settings["text_evidence_floor_tokens"]
    print(
        f"  works with thin masked text: {int(low_text_evidence.sum())}  "
        f"below the text floor: {int((~text_usable).sum())}"
    )
    print("Building the topic view ...")
    topic_matrix, topic_available, topic_dimensions = build_topic_view(
        rows, set(core_manifest["explicit_quantum_topic_ids"])
    )
    print(
        f"  topic dims {topic_dimensions}  works with a usable topic vector "
        f"{int(topic_available.sum())}"
    )

    k = model["k_neighbors"]
    usable = np.flatnonzero(topic_available)
    core_topic_index = np.flatnonzero(is_core & topic_available)
    if core_topic_index.size <= k:
        raise RuntimeError("Too few core works have a usable topic vector")

    print("Scoring distances to the core ...")
    text_distance, text_neighbours = neighbour_distances(
        masked_embedding, core_index, k, return_neighbours=True
    )
    unmasked_distance = neighbour_distances(unmasked_embedding, core_index, k)
    topic_distance = neighbour_distances(topic_matrix, core_topic_index, k)
    topic_distance[~topic_available] = np.nan

    text_percentile = percentile_of(text_distance[core_index], text_distance)
    unmasked_percentile = percentile_of(
        unmasked_distance[core_index], unmasked_distance
    )
    topic_percentile = np.full(len(rows), np.nan)
    topic_percentile[usable] = percentile_of(
        topic_distance[core_topic_index], topic_distance[usable]
    )

    # The centres name the region a work sits in; they never produce a distance, a band,
    # or a verdict.  Scoring a work by its distance to the nearest of a handful of
    # centroids was measured against this k-nearest-core-works score on held-out core
    # papers and lost by roughly five AUC points, because collapsing 5,114 reference
    # papers into a few centroids discards the shape of the core.
    center_settings = config["core_centers"]
    print("Clustering the core into centres for labelling ...")
    text_centers = fit_core_centers(
        masked_embedding, core_index, center_settings, "text"
    )
    center_distance, center_nearest = center_distances(
        masked_embedding, text_centers["centers"]
    )

    print("Fitting the local outlier factor on the core ...")
    # LOF compares a work with the local density around its core neighbours, so a work
    # near a sparse part of the core is not punished for the core's uneven density.
    lof = LocalOutlierFactor(
        n_neighbors=min(model["lof_neighbors"], core_index.size - 1),
        metric="cosine",
        novelty=True,
    )
    lof.fit(masked_embedding[core_index])
    local_outlier_factor = -lof.score_samples(masked_embedding)
    lof_flag = lof.predict(masked_embedding) == -1

    verdicts, views = decide_verdicts(
        is_core,
        text_percentile,
        topic_percentile,
        text_usable,
        topic_available,
        model["borderline_percentile"],
        model["far_percentile"],
    )

    far_positions = np.flatnonzero(verdicts == VERDICT_FAR)
    far_cluster = np.full(len(rows), -1, dtype=int)
    cluster_summaries = []
    if far_positions.size >= model["far_cluster_count"]:
        clusters = min(model["far_cluster_count"], far_positions.size)
        print(f"Clustering {far_positions.size} far works into {clusters} groups ...")
        kmeans = KMeans(
            n_clusters=clusters, random_state=model["random_state"], n_init=10
        )
        labels = kmeans.fit_predict(masked_embedding[far_positions])
        far_cluster[far_positions] = labels
        cluster_summaries = describe_clusters(
            labels,
            [rows[position] for position in far_positions],
            label_matrix[far_positions],
            feature_names,
            included[far_positions],
            model,
        )

    center_summaries = describe_centers(
        "text",
        text_centers,
        rows,
        label_matrix,
        feature_names,
        center_nearest,
        center_distance,
        included,
        center_settings,
    )
    # center_distances returns a slot into the kept centres; the CSVs name centres by the
    # cluster id they were given, so translate once here.
    text_center_ids = [int(center) for center in text_centers["kept_ids"]]
    text_center_label = {
        summary["center"]: summary["label"] for summary in center_summaries
    }

    core_thresholds = {
        "text_borderline_distance": float(
            np.percentile(text_distance[core_index], model["borderline_percentile"])
        ),
        "text_far_distance": float(
            np.percentile(text_distance[core_index], model["far_percentile"])
        ),
        "topic_borderline_distance": float(
            np.percentile(
                topic_distance[core_topic_index], model["borderline_percentile"]
            )
        ),
        "topic_far_distance": float(
            np.percentile(topic_distance[core_topic_index], model["far_percentile"])
        ),
        "core_median_text_distance": float(np.median(text_distance[core_index])),
        "core_median_topic_distance": float(
            np.median(topic_distance[core_topic_index])
        ),
    }

    tmp = {
        path: path + ".tmp"
        for path in (scores_path, clusters_path, centers_path, manifest_output)
    }
    for path in tmp:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    try:
        with open(tmp[scores_path], "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=SCORE_FIELDS)
            writer.writeheader()
            order = np.argsort(-text_percentile)
            for position in order:
                row = rows[position]
                neighbours = " | ".join(
                    rows[index]["work_id"]
                    for index in text_neighbours[position][: model["neighbor_examples"]]
                )
                writer.writerow(
                    {
                        "work_id": row["work_id"],
                        "publication_year": row["publication_year"],
                        "title": row["title"],
                        "work_type": row["type"],
                        "primary_topic_name": row["primary_topic_name"],
                        "primary_topic_domain": row["primary_topic_domain"],
                        "is_core": str(bool(is_core[position])).lower(),
                        "currently_included": str(bool(included[position])).lower(),
                        "verdict": verdicts[position],
                        "views_used": views[position],
                        "low_text_evidence": str(
                            bool(low_text_evidence[position])
                        ).lower(),
                        "text_distance": round(float(text_distance[position]), 6),
                        "text_percentile": round(float(text_percentile[position]), 3),
                        "topic_distance": (
                            ""
                            if np.isnan(topic_distance[position])
                            else round(float(topic_distance[position]), 6)
                        ),
                        "topic_percentile": (
                            ""
                            if np.isnan(topic_percentile[position])
                            else round(float(topic_percentile[position]), 3)
                        ),
                        "nearest_center": text_center_ids[
                            int(center_nearest[position])
                        ],
                        "nearest_center_label": text_center_label[
                            text_center_ids[int(center_nearest[position])]
                        ],
                        "unmasked_text_percentile": round(
                            float(unmasked_percentile[position]), 3
                        ),
                        "local_outlier_factor": round(
                            float(local_outlier_factor[position]), 6
                        ),
                        "local_outlier_factor_flag": str(
                            bool(lof_flag[position])
                        ).lower(),
                        "far_cluster": int(far_cluster[position]),
                        "nearest_core_works": neighbours,
                    }
                )
        with open(tmp[clusters_path], "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CLUSTER_FIELDS)
            writer.writeheader()
            writer.writerows(cluster_summaries)
        with open(tmp[centers_path], "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CENTER_FIELDS)
            writer.writeheader()
            writer.writerows(center_summaries)

        counts = {
            "candidate_works": len(rows),
            "core_works": int(core_index.size),
            "currently_included": int(included.sum()),
            "verdicts": {
                verdict: int(np.count_nonzero(verdicts == verdict))
                for verdict in VERDICTS
            },
            "included_and_far": int(
                np.count_nonzero(included & (verdicts == VERDICT_FAR))
            ),
            "included_and_borderline": int(
                np.count_nonzero(included & (verdicts == VERDICT_BORDERLINE))
            ),
            "excluded_and_near_core": int(
                np.count_nonzero(~included & (verdicts == VERDICT_NEAR))
            ),
            "core_internal_outliers": int(
                np.count_nonzero(is_core & (text_percentile > model["far_percentile"]))
            ),
            "local_outlier_factor_flags": int(np.count_nonzero(lof_flag)),
            "works_without_topic_vector": int(np.count_nonzero(~topic_available)),
            "works_with_thin_masked_text": int(np.count_nonzero(low_text_evidence)),
            "works_below_text_floor": int(np.count_nonzero(~text_usable)),
            "far_and_thin_masked_text": int(
                np.count_nonzero(low_text_evidence & (verdicts == VERDICT_FAR))
            ),
        }
        manifest = {
            "schema_version": 1,
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "method": "unsupervised_novelty_scoring_against_explicit_quantum_core",
            "model": {
                "text_view": {
                    "backend": backend,
                    "dimensions": int(masked_embedding.shape[1]),
                    "embedding_model": (embedding_manifest or {}).get("model"),
                    "cluster_label_terms": int(label_matrix.shape[1]),
                    "explained_variance_ratio": (
                        None
                        if masked_variance is None
                        else round(float(masked_variance), 6)
                    ),
                    "masked_core_terms": True,
                    "minimum_text_tokens": text_settings["minimum_text_tokens"],
                    "text_evidence_floor_tokens": text_settings[
                        "text_evidence_floor_tokens"
                    ],
                },
                "topic_view": {
                    "dimensions": topic_dimensions,
                    "masked_explicit_quantum_topics": True,
                },
                "scoring": "mean_distance_to_k_nearest_core_works",
                "core_centers": {
                    "role": "labelling_only",
                    "method": center_settings["method"],
                    "minimum_center_works": center_settings["minimum_center_works"],
                    "selected": text_centers["selected"],
                    "kept": int(text_centers["centers"].shape[0]),
                    "silhouette": round(text_centers["silhouette"], 6),
                    "grid": text_centers["grid"],
                    "dropped": text_centers["dropped"],
                    "core_works_per_center": [
                        int(size) for size in text_centers["sizes"]
                    ],
                },
                "k_neighbors": k,
                "distance": "cosine",
                "local_outlier_factor_neighbors": model["lof_neighbors"],
                "random_state": model["random_state"],
                "bands": {
                    "borderline_percentile": model["borderline_percentile"],
                    "far_percentile": model["far_percentile"],
                },
                "core_distance_thresholds": core_thresholds,
            },
            "inputs": {
                "candidate_features": {
                    "path": features_path,
                    "sha256": file_sha256(features_path),
                },
                "core_manifest": {
                    "path": core_manifest_path,
                    "sha256": file_sha256(core_manifest_path),
                    "core_works": (core_manifest.get("counts") or {}).get("core_works"),
                },
                "candidate_audit": {
                    "path": audit_path,
                    "sha256": file_sha256(audit_path),
                    "filter_manifest_created_at": filter_manifest.get("created_at"),
                },
                "config": {"path": config_path, "sha256": file_sha256(config_path)},
                "embeddings": (
                    None
                    if embedding_manifest is None
                    else {
                        "path": EMBEDDING_MANIFEST,
                        "sha256": file_sha256(EMBEDDING_MANIFEST),
                        "model": embedding_manifest["model"],
                        "created_at": embedding_manifest.get("created_at"),
                    }
                ),
            },
            "counts": counts,
            "artifacts": {
                "outlier_scores": {
                    "path": scores_path,
                    "row_count": len(rows),
                    "sha256": file_sha256(tmp[scores_path]),
                },
                "outlier_clusters": {
                    "path": clusters_path,
                    "row_count": len(cluster_summaries),
                    "sha256": file_sha256(tmp[clusters_path]),
                },
                "core_centers": {
                    "path": centers_path,
                    "row_count": len(center_summaries),
                    "sha256": file_sha256(tmp[centers_path]),
                },
            },
        }
        with open(tmp[manifest_output], "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.write("\n")
        for path in (scores_path, clusters_path, centers_path, manifest_output):
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
    manifest = detect_outliers()
    counts = manifest["counts"]
    thresholds = manifest["model"]["core_distance_thresholds"]
    centers = manifest["model"]["core_centers"]
    print()
    print(f"Scoring: {manifest['model']['scoring']}")
    print(
        f"Labelling centres: {centers['kept']} kept of {centers['selected']}  "
        f"silhouette {centers['silhouette']:+.4f}"
    )
    print(f"Core median text distance:  {thresholds['core_median_text_distance']:.4f}")
    print(f"Core median topic distance: {thresholds['core_median_topic_distance']:.4f}")
    print("Verdicts:")
    for verdict, count in counts["verdicts"].items():
        print(f"  {verdict:14s} {count}")
    print(f"Currently included and far:        {counts['included_and_far']}")
    print(f"Currently included and borderline: {counts['included_and_borderline']}")
    print(f"Currently excluded and near core:  {counts['excluded_and_near_core']}")
    print(f"Core-internal outliers:            {counts['core_internal_outliers']}")
    print(f"Scores:   {SCORES_OUTPUT}")
    print(f"Clusters: {CLUSTERS_OUTPUT}")
    print(f"Centres:  {CENTERS_OUTPUT}")
    print(f"Manifest: {MANIFEST_OUTPUT}")


if __name__ == "__main__":
    main()
