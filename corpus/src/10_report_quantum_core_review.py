"""Render a short self-contained HTML report for the stage 8 review instrument.

The report states how the data is processed and shows the statistics that support each
claim. Most charts are drawn with matplotlib and inlined as SVG; the two time series are
plotly, and plotly.js is embedded in the page rather than fetched, so the file still opens
offline with no external requests.

"""

import csv
import html
import importlib.util
import io
import itertools
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from matplotlib.patches import Patch
from plotly.io import to_html
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots

from utils.openalex_utils import file_sha256

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORE_CONFIG = "config/sweden_quantum_core_and_outliers.json"
FEATURES = "data/sweden_quantum_candidate_features.jsonl"
CORE_MANIFEST = "data/sweden_quantum_core_manifest.json"
OUTLIER_MANIFEST = "data/sweden_quantum_outlier_manifest.json"
EMBEDDING_MANIFEST = "data/sweden_quantum_embeddings_manifest.json"
FILTER_MANIFEST = "data/sweden_quantum_works_manifest.json"
SCORES = "data/sweden_quantum_outlier_scores.csv"
CLUSTERS = "data/sweden_quantum_outlier_clusters.csv"
AUDIT = "data/sweden_quantum_candidate_audit.csv"
UNION_MANIFEST = "data/sweden_quantum_candidate_union_manifest.json"
TOPIC_CANDIDATES = "data/sweden_quantum_topic_candidates.csv"
RECORD_GATES = "config/sweden_quantum_record_gates.json"
TOPIC_SNAPSHOT = "data/openalex_topics_snapshot.jsonl"
TOPIC_SNAPSHOT_MANIFEST = "data/openalex_topics_snapshot_manifest.json"
DERIVED_KEYWORDS = "data/sweden_quantum_derived_keywords.csv"
TOPIC_MANIFEST = "data/sweden_quantum_topic_candidates_manifest.json"
AFFINITIES = "data/sweden_quantum_area_affinities.csv"
AFFINITY_MANIFEST = "data/sweden_quantum_area_affinities_manifest.json"
OUTPUT = "quantum_core_review_report.html"

# Validated with the data-viz palette validator: a 3-slot categorical scale and a
# 5-step ordinal ramp, both against a light surface.
CAT = ("#2a78d6", "#eb6834", "#1baf7a")
ORDINAL = ("#86b6ef", "#5598e7", "#2a78d6", "#184f95", "#0d366b")
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_3 = "#77756f"
EDGE = "#e4e3de"
NEUTRAL = "#9c9b93"
MUTED = "#c3c1bb"

# A per-year mean over a handful of works is noise drawn as a trend, so the area series
# start at the first year clearing this floor. Same floor the QSIP study used.
MIN_WORKS_PER_YEAR = 30

# The distance axis is derived from the scores, never hardcoded. The two views live on
# different scales, and an embedding backend puts them somewhere else again than TF-IDF
# does, so a fixed window silently clamps most of the corpus into the last bin.
DISTANCE_BINS = 70

FONT_STACK = (
    "DejaVu Sans, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif"
)

# How the text encoder is described in prose, keyed by the model recorded in the
# embedding manifest, so swapping the model in step 07 also changes what the page claims.
MODEL_NOTES = {
    "BAAI/bge-large-en-v1.5": (
        "BGE-large-en-v1.5",
        "a general-purpose English text encoder trained contrastively on large-scale text pairs",
        "was trained on general English text rather than on scientific writing in particular",
    ),
    "allenai/specter2_base": (
        "SPECTER2",
        "a transformer trained on citation links between papers",
        "was trained on English scientific text",
    ),
}


def model_notes(name):
    """(display name, what it is, what it was trained on) for the configured encoder."""
    return MODEL_NOTES.get(
        name, (name, "a sentence encoder", "was trained on English text")
    )


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_core_module():
    """Reuse the real core rule so the funnel cannot drift from the pipeline."""
    path = os.path.join(ROOT, "src", "06_build_quantum_core_set.py")
    spec = importlib.util.spec_from_file_location("quantum_core_set", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify(path, manifest, keys):
    node = manifest
    for key in keys:
        node = node[key]
    if node.get("sha256") != file_sha256(path):
        raise RuntimeError(f"{path} does not match its manifest hash; rerun the stage")


def core_funnel(features_path, rule, core_module):
    """Apply the core checks in order, count the survivors, and keep the closest calls.

    The examples are the highest-scoring work each check still rejects, so they are the
    strongest case against the rule rather than a flattering one.
    """
    explicit = set(load_json(CORE_MANIFEST)["explicit_quantum_topic_ids"])
    core_patterns = core_module.compile_patterns(rule["core_term_patterns"])
    ambiguous = core_module.compile_patterns(rule.get("ambiguous_term_patterns") or [])
    excluded_types = set(rule["excluded_work_types"])
    stages = [
        ["In-window candidates", 0],
        ["Explicit-quantum primary topic", 0],
        ["Primary-topic score at least 0.5", 0],
        ["Quantum term in its own text", 0],
        ["Record type reports its own science", 0],
    ]
    channels = {"Topic routes only": 0, "Both channels": 0, "Keyword routes only": 0}
    examples = {}

    def remember(key, row, score, note, keep="highest"):
        current = examples.get(key)
        better = current is None or (
            score > current[0] if keep == "highest" else score < current[0]
        )
        if better:
            examples[key] = (score, row["title"], row["primary_topic_name"], note)

    with open(features_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            stages[0][1] += 1
            by_topic = bool(
                row["matched_explicit_quantum_topic_ids"]
                or row["matched_broad_related_topic_ids"]
            )
            by_keyword = bool(row["matched_derived_keyword_ids"])
            if by_topic and by_keyword:
                channels["Both channels"] += 1
            elif by_topic:
                channels["Topic routes only"] += 1
            else:
                channels["Keyword routes only"] += 1
            if row["primary_topic_id"] not in explicit:
                continue
            stages[1][1] += 1
            score = row["primary_topic_score"]
            if (
                not isinstance(score, (int, float))
                or score < rule["minimum_primary_topic_score"]
            ):
                if isinstance(score, (int, float)):
                    remember("weak_topic", row, score, "just under the score gate")
                    remember(
                        "weakest_topic",
                        row,
                        score,
                        "the weakest label in the data",
                        "lowest",
                    )
                continue
            stages[2][1] += 1
            own = "\n".join(row[field] for field in sorted(rule["core_term_fields"]))
            terms = core_module.matched_terms(own, core_patterns)
            if not terms:
                remember(
                    "no_term", row, score, "no quantum term in its title or abstract"
                )
                continue
            fuzzy = core_module.matched_terms(own, ambiguous)
            if fuzzy and not [
                t for t in terms if not any(t in phrase for phrase in fuzzy)
            ]:
                remember("ambiguous", row, score, f'only "{fuzzy[0].strip()}"')
                continue
            stages[3][1] += 1
            if row["type"] in excluded_types:
                remember("record_type", row, score, f"record type {row['type']}")
                continue
            stages[4][1] += 1
            remember("accepted", row, score, f'says "{terms[0]}" itself')
    return stages, list(channels.items()), examples


def topic_inventory(path):
    """Topic routes by inclusion group, plus the topics that are never queried."""
    from utils.quantum_topic_groups import candidate_group

    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    groups = {"Explicit quantum": 0, "Broad related": 0, "Blocked, never queried": 0}
    for row in rows:
        if row["taxonomy_gate"] == "blocked":
            groups["Blocked, never queried"] += 1
        elif candidate_group(row) == "explicit_quantum":
            groups["Explicit quantum"] += 1
        else:
            groups["Broad related"] += 1
    return len(rows), list(groups.items())


def topic_examples(path):
    """The largest topic in each group, with the term that put it there."""
    from utils.quantum_topic_groups import candidate_group

    picks = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["taxonomy_gate"] == "blocked":
                group = "blocked"
            elif candidate_group(row) == "explicit_quantum":
                group = "explicit_quantum"
            else:
                group = "broad_related"
            works = int(row["works_count"] or 0)
            if group not in picks or works > picks[group][0]:
                picks[group] = (works, row)
    rows = []
    for group, label in (
        ("explicit_quantum", "explicit_quantum"),
        ("broad_related", "broad_related"),
        ("blocked", "blocked"),
    ):
        if group not in picks:
            continue
        works, row = picks[group]
        name_terms = [
            t.strip() for t in row["matched_name_terms"].split("|") if t.strip()
        ]
        keyword_terms = [
            t.strip() for t in row["matched_keyword_terms"].split("|") if t.strip()
        ]
        where = (
            f'name: "{name_terms[0]}"'
            if name_terms
            else f'taxonomy keyword: "{keyword_terms[0]}"'
        )
        note = label if group != "blocked" else f"blocked · {row['field_name']}"
        rows.append(
            [
                (f"{row['topic_name']} ({row['topic_id']})", row["topic_name"]),
                where,
                note,
            ]
        )
    return rows


def keyword_examples(path, thresholds):
    """One selected keyword per tier and the two most instructive rejections."""
    low = thresholds["title_confirmed"]

    def number(row, key):
        try:
            return float(row[key])
        except (TypeError, ValueError):
            return 0.0

    def clears_numbers(row):
        return (
            number(row, "explicit_seed_work_count")
            >= low["minimum_explicit_seed_works"]
            and number(row, "explicit_topic_count") >= low["minimum_explicit_topics"]
            and number(row, "explicit_share_pct")
            >= 100 * low["minimum_explicit_share_of_topic_universe"]
        )

    picks = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["selected"] == "true":
                key = row["selection_tier"]
            elif row["name_matches_quantum_pattern"] != "true":
                continue
            elif row["identifier_matches_quantum_pattern"] != "true":
                # Only interesting when the identifier is the single reason it failed.
                if not clears_numbers(row):
                    continue
                key = "rejected_identifier"
            else:
                key = "rejected_share"
            works = number(row, "explicit_seed_work_count")
            if key not in picks or works > picks[key][0]:
                picks[key] = (works, row)
    outcomes = {
        "high_confidence": "accepted automatically",
        "title_confirmed": "kept, but the title must confirm",
        "rejected_share": "rejected: too common outside quantum topics",
        "rejected_identifier": "rejected: identifier carries no quantum signal",
    }
    rows = []
    for key, outcome in outcomes.items():
        if key not in picks:
            continue
        works, row = picks[key]
        rows.append(
            [
                (f"{row['keyword_name']} ({row['keyword_id']})", row["keyword_name"]),
                f"{int(works):,}",
                f"{number(row, 'explicit_share_pct'):.1f}%",
                outcome,
            ]
        )
    return rows


def keyword_points(path):
    """Every keyword with a quantum-specific name and OpenAlex identifier.

    These are the keywords the numeric thresholds actually judge, with the two numbers
    the rule reads and the outcome it reached.
    """
    points = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if (
                row["name_matches_quantum_pattern"] != "true"
                or row["identifier_matches_quantum_pattern"] != "true"
            ):
                continue
            try:
                works = int(row["explicit_seed_work_count"])
                share = float(row["explicit_share_pct"])
            except (TypeError, ValueError):
                continue
            points.append(
                {
                    "name": row["keyword_name"],
                    "works": works,
                    "share": share,
                    "group": (
                        row["selection_tier"]
                        if row["selected"] == "true"
                        else "rejected"
                    ),
                }
            )
    return points


def channel_label(reason):
    """Name the channel a single audit reason describes, using the step-4 chart labels."""
    # The keyword reasons also contain the phrase "explicit quantum topics", so the
    # keyword wording must be tested first.
    if "lacked an independent title" in reason:
        return "Lower tier, no title evidence"
    if "multi-signal safeguard" in reason:
        return "Keyword-only domain safeguard"
    if "lower-confidence" in reason:
        return "Title-confirmed keyword"
    if "high-confidence" in reason:
        return "High-confidence keyword"
    if "matched only a broad-related topic" in reason:
        return "Broad-related topic, no term"
    if "broad-related topic route confirmed" in reason:
        return "Broad-related topic + paper term"
    if "matched an explicit quantum topic" in reason:
        return "Explicit-quantum topic"
    return None


def channel_examples(audit_path, scores_path):
    """The newest article admitted or rejected through each single channel.

    Only works that arrived through exactly one channel are eligible, so the example
    cannot be carried by a second route, and titles carrying raw LaTeX are skipped
    because they are unreadable out of context.
    """
    types = {}
    with open(scores_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            types[row["work_id"]] = row["work_type"]
    picks = {}
    with open(audit_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            reasons = [
                r.strip() for r in row["inclusion_reasons"].split("|") if r.strip()
            ]
            if len(reasons) != 1 or types.get(row["work_id"]) != "article":
                continue
            title = row["title"]
            if "$" in title or "\\" in title:
                continue
            key = channel_label(reasons[0])
            if key is None:
                continue
            year = int(row["publication_year"] or 0)
            if key not in picks or (year, row["work_id"]) > (
                picks[key][0],
                picks[key][1]["work_id"],
            ):
                picks[key] = (year, row)
    return picks


def keyword_selection(path, thresholds):
    """The two gates that turn observed paper keywords into retrieval routes."""
    high = thresholds["high_confidence"]
    low = thresholds["title_confirmed"]

    def number(row, key):
        try:
            return float(row[key])
        except (TypeError, ValueError):
            return 0.0

    observed = pattern = selected = high_tier = 0
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            observed += 1
            if (
                row["name_matches_quantum_pattern"] != "true"
                or row["identifier_matches_quantum_pattern"] != "true"
            ):
                continue
            pattern += 1
            if (
                number(row, "explicit_seed_work_count")
                >= low["minimum_explicit_seed_works"]
                and number(row, "explicit_topic_count")
                >= low["minimum_explicit_topics"]
                and number(row, "explicit_share_pct")
                >= 100 * low["minimum_explicit_share_of_topic_universe"]
            ):
                selected += 1
                if (
                    number(row, "explicit_seed_work_count")
                    >= high["minimum_explicit_seed_works"]
                    and number(row, "explicit_share_pct")
                    >= 100 * high["minimum_explicit_share_of_topic_universe"]
                ):
                    high_tier += 1
    gate = [
        ("Observed on explicit-topic works", observed),
        ("Quantum pattern in name and ID", pattern),
    ]
    kept = [
        ("Quantum pattern in name and ID", pattern),
        ("Meets the retrieval thresholds", selected),
        ("Meets the automatic-acceptance tier", high_tier),
    ]
    return gate, kept


GROUPS = ("core", "included", "excluded")


def group_of(row):
    if row["is_core"] == "true":
        return "core"
    return "included" if row["currently_included"] == "true" else "excluded"


def read_distances(scores_path):
    """Both views' distances per group, and how many works each group holds.

    A blank cell means the view was not usable for that work, so the totals are counted
    separately rather than taken from the length of a distance list.
    """
    values = {view: {key: [] for key in GROUPS} for view in ("text", "topic")}
    totals = dict.fromkeys(GROUPS, 0)
    for row in csv.DictReader(open(scores_path, encoding="utf-8")):
        key = group_of(row)
        totals[key] += 1
        for view in ("text", "topic"):
            cell = row[f"{view}_distance"]
            if cell != "":
                values[view][key].append(float(cell))
    return values, totals


def histogram(values, bins=DISTANCE_BINS):
    """Share-of-group per bin over a range read off the data, so nothing is clamped.

    Each group is normalized to its own size because the three differ by an order of
    magnitude; the counts ride along for the hover.
    """
    low = min(min(series) for series in values.values() if series)
    high = max(max(series) for series in values.values() if series)
    width = (high - low) / bins
    centers = [round(low + width * (index + 0.5), 5) for index in range(bins)]
    shares, counts = {}, {}
    for key, series in values.items():
        binned = [0] * bins
        for value in series:
            binned[min(int((value - low) / width), bins - 1)] += 1
        counts[key] = binned
        total = len(series) or 1
        shares[key] = [100.0 * count / total for count in binned]
    return centers, shares, counts


def flagged_evidence(scores_path, audit_path):
    """Why the currently-included far outliers were admitted."""
    flagged = set()
    lof_agrees = 0
    with open(scores_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["currently_included"] == "true" and row["verdict"] == "far_outlier":
                flagged.add(row["work_id"])
                if row["local_outlier_factor_flag"] == "true":
                    lof_agrees += 1
    channels = {}
    terms = {
        "No quantum term at all": 0,
        "Only quantization/quantizer": 0,
        "Other quantum wording": 0,
    }
    quantiz = re.compile(r"quanti[sz]\w*", re.IGNORECASE)
    single = multi = 0
    with open(audit_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["included"] == "true":
                # How many inclusion channels this work arrived through.
                if (
                    len([r for r in row["inclusion_reasons"].split("|") if r.strip()])
                    > 1
                ):
                    multi += 1
                else:
                    single += 1
            if row["work_id"] not in flagged:
                continue
            reasons = [
                r.strip() for r in row["inclusion_reasons"].split("|") if r.strip()
            ]
            label = short_reason(reasons)
            channels[label] = channels.get(label, 0) + 1
            found = [
                t.strip()
                for t in row["paper_level_quantum_terms"].split("|")
                if t.strip()
            ]
            if not found:
                terms["No quantum term at all"] += 1
            elif all(quantiz.fullmatch(t) for t in found):
                terms["Only quantization/quantizer"] += 1
            else:
                terms["Other quantum wording"] += 1
    ordered = sorted(channels.items(), key=lambda item: -item[1])
    if len(ordered) > 4:
        rest = sum(count for _, count in ordered[4:])
        ordered = ordered[:4] + [("Several channels at once", rest)]
    return len(flagged), ordered, list(terms.items()), single, multi, lof_agrees


def short_reason(reasons):
    if len(reasons) > 1:
        return "Several channels at once"
    reason = reasons[0] if reasons else "unknown"
    # The keyword reasons also contain the phrase "explicit quantum topics", so the
    # keyword tiers must be tested before the topic route.
    if "lower-confidence" in reason:
        return "Lower-confidence keyword tier"
    if "high-confidence" in reason:
        return "High-confidence keyword tier"
    if "broad-related" in reason:
        return "Broad-related topic plus a term"
    if "matched an explicit quantum topic" in reason:
        return "Explicit-quantum topic route alone"
    return reason[:40]


def inclusion_families(audit_path):
    """How many corpus works were reached by topic routes, keyword routes, or both."""
    families = {"topic only": 0, "keyword only": 0, "both": 0}
    with open(audit_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["included"] != "true":
                continue
            reasons = row["inclusion_reasons"]
            by_topic = (
                "matched an explicit quantum topic" in reasons
                or "broad-related topic route confirmed" in reasons
            )
            by_keyword = "high-confidence" in reasons or "lower-confidence" in reasons
            if by_topic and by_keyword:
                families["both"] += 1
            elif by_keyword:
                families["keyword only"] += 1
            else:
                families["topic only"] += 1
    return families


def flagged_examples(scores_path, audit_path, limit=5):
    """The in-corpus far outliers that sit farthest from the core, with their evidence."""
    flagged = []
    with open(scores_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["currently_included"] == "true" and row["verdict"] == "far_outlier":
                flagged.append(row)
    flagged.sort(key=lambda row: (-float(row["text_distance"]), row["work_id"]))
    chosen = {row["work_id"]: row for row in flagged[:limit]}
    evidence = {}
    with open(audit_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["work_id"] in chosen:
                evidence[row["work_id"]] = row
    rows = []
    for work_id, row in chosen.items():
        audit = evidence.get(work_id, {})
        reasons = [
            r.strip()
            for r in audit.get("inclusion_reasons", "").split("|")
            if r.strip()
        ]
        terms = [
            t.strip()
            for t in audit.get("paper_level_quantum_terms", "").split("|")
            if t.strip()
        ]
        rows.append(
            [
                (clip(row["title"], 58), row["title"]),
                (clip(row["primary_topic_name"], 34), row["primary_topic_name"]),
                f"{float(row['text_distance']):.3f}",
                channel_label(reasons[0]) if reasons else "unknown",
                ", ".join(terms) if terms else "none",
            ]
        )
    return rows


def gated_by_language(row, languages):
    """A missing language is not evidence of a foreign one, so only a label can gate."""
    language = row["language"].strip()
    return bool(language) and language not in languages


def exclusion_reason(row, languages, work_types):
    """Why step 4 rejected a candidate, in the order step 4 itself attributes exclusions.

    The record gates are tested first because they run first: a gated record never reaches
    the topic and keyword rule, so no later reason can be the one that actually applied.
    """
    if gated_by_language(row, languages):
        return f"language read as {row['language'].strip()}"
    if row["work_type"] in work_types:
        return f"record type is {row['work_type']}"
    if row["keyword_only_blocked_primary_domain"] == "true":
        return "keyword only, blocked domain"
    if (
        row["matched_title_confirmed_derived_keyword_ids"].strip()
        and not row["title_level_quantum_terms"].strip()
    ):
        return "keyword needed title confirmation"
    return "broad-related topic only, no quantum wording"


def distinct_titles(rows, limit):
    """One row per paper; OpenAlex keeps duplicate records of the same work."""
    seen = set()
    chosen = []
    for row in rows:
        key = plain(row["title"]).casefold()
        if key in seen:
            continue
        seen.add(key)
        chosen.append(row)
        if len(chosen) == limit:
            break
    return chosen


def excluded_examples(scores_path, audit_path, gates, limit=8):
    """The excluded candidates nearest the core and the ones farthest from it.

    Works stopped by the language gate are left out of both ends. OpenAlex reads language
    off the title, so that gate turns on a metadata guess rather than on the paper, and
    the encoder cannot read a non-English abstract either way: such a work says nothing
    about how the filter handles subject matter, which is what both ends are here to show.

    Both ends are ranked on the text view. The far end additionally requires real text: a
    record with a bare title and no abstract sits far from everything, so without that
    condition the table would only show works the text view never really measured.
    """
    languages = set(gates["accepted_languages"])
    work_types = set(gates["excluded_work_types"])
    reasons = {}
    with open(audit_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["included"] == "true" or gated_by_language(row, languages):
                continue
            reasons[row["work_id"]] = exclusion_reason(row, languages, work_types)

    rows = []
    with open(scores_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["currently_included"] == "true" or row["is_core"] == "true":
                continue
            if not row["text_distance"] or not row["topic_distance"]:
                continue
            if row["work_id"] not in reasons:
                continue
            rows.append(row)
    rows.sort(key=lambda row: (float(row["text_distance"]), row["work_id"]))
    near = distinct_titles(rows, limit)
    far = distinct_titles(
        [row for row in reversed(rows) if row["low_text_evidence"] != "true"], limit
    )

    def render(chosen):
        return [
            [
                (clip(plain(row["title"]), 54), plain(row["title"])),
                (clip(row["primary_topic_name"], 30), row["primary_topic_name"]),
                row["publication_year"],
                f"{float(row['text_distance']):.3f}",
                reasons.get(row["work_id"], "unknown"),
            ]
            for row in chosen
        ]

    return render(near), render(far)


def read_clusters(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            terms = [t.strip() for t in row["top_terms"].split("|")]
            label = ", ".join(terms[:3])
            rows.append(
                {
                    "label": label,
                    "works": int(row["works"]),
                    "included": int(row["currently_included"]),
                    "topics": row["top_primary_topics"].split("|")[0].strip(),
                }
            )
    rows.sort(key=lambda item: -item["included"])
    return rows


# ------------------------------------------------------------------- matplotlib

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 9.5,
        "text.color": INK_2,
        "svg.fonttype": "none",
        "svg.hashsalt": "kth-quantum-report",
        "figure.dpi": 96,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9.5,
    }
)


def figure_svg(fig, dpi=96):
    """Serialise a figure as inline SVG that inherits the page's font stack.

    Text and axes stay vector; a scatter marked rasterized=True is embedded as a PNG at
    this dpi, which is what keeps an 83,488-point map to a sane file size.
    """
    buffer = io.StringIO()
    fig.savefig(
        buffer,
        format="svg",
        bbox_inches="tight",
        pad_inches=0.02,
        transparent=True,
        dpi=dpi,
    )
    plt.close(fig)
    markup = buffer.getvalue()
    markup = markup[markup.index("<svg") :]
    markup = markup.replace("DejaVu Sans", FONT_STACK)
    # Let CSS control the rendered width; the viewBox keeps the aspect ratio.
    markup = re.sub(r'<svg width="[^"]*" height="[^"]*"', "<svg", markup, count=1)
    return markup


def bare(ax):
    """A bar axis with no frame: the value labels replace the x scale."""
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.set_xticks([])
    ax.tick_params(axis="y", length=0, pad=6, labelcolor=INK)
    ax.xaxis.set_visible(False)


def draw_bars(ax, rows, colors, title=None):
    """rows: [(label, value)]; colors: one colour, or one per row."""
    labels = [label for label, _ in rows]
    values = [value for _, value in rows]
    if isinstance(colors, str):
        colors = [colors] * len(rows)
    positions = range(len(rows))
    ax.barh(list(positions), values, height=0.5, color=list(colors))
    ax.set_yticks(list(positions))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, max(values) * 1.16 or 1)
    for y, value in zip(positions, values):
        ax.text(value + max(values) * 0.015, y, f"{value:,}", va="center", fontsize=9)
    bare(ax)
    if title:
        ax.set_title(title, loc="left", color=INK_3, pad=8)


def draw_stacked(ax, rows, series_labels, title=None):
    """rows: [(label, [value_a, value_b])] on a shared 2-slot categorical scale."""
    labels = [label for label, _ in rows]
    totals = [sum(values) for _, values in rows]
    positions = list(range(len(rows)))
    span = max(totals) or 1
    gap = span * 0.004  # a surface gap instead of a border between segments
    for slot in range(len(rows[0][1])):
        left = [
            sum(values[:slot]) + (gap if slot and sum(values[:slot]) else 0)
            for _, values in rows
        ]
        ax.barh(
            positions,
            [values[slot] for _, values in rows],
            left=left,
            height=0.5,
            color=CAT[slot],
            label=series_labels[slot],
        )
    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, span * 1.16)
    for y, total in zip(positions, totals):
        ax.text(total + span * 0.015, y, f"{total:,}", va="center", fontsize=9)
    bare(ax)
    ax.legend(
        loc="lower right",
        bbox_to_anchor=(1.0, 1.0),
        ncols=len(series_labels),
        frameon=False,
        fontsize=9,
        handlelength=0.9,
        handleheight=0.9,
        borderpad=0,
        columnspacing=1.4,
        handletextpad=0.5,
    )
    if title:
        ax.set_title(title, loc="left", color=INK_3, pad=8)


def bars_chart(panels, width=8.4, row_height=0.30):
    """One or more bar panels stacked vertically, each with its own scale."""
    counts = [len(rows) for _, rows, _ in panels]
    height = sum(count * row_height + 0.34 for count in counts)
    fig, axes = plt.subplots(
        len(panels), 1, figsize=(width, height), gridspec_kw={"height_ratios": counts}
    )
    axes = [axes] if len(panels) == 1 else list(axes)
    for ax, (title, rows, colors) in zip(axes, panels):
        draw_bars(ax, rows, colors, title=title)
    fig.subplots_adjust(hspace=0.9)
    return figure_svg(fig)


def split_bars_chart(rows, colors, legend_labels, width=8.4, row_height=0.34):
    """A single scale with two meanings, distinguished by colour and a legend."""
    fig, ax = plt.subplots(figsize=(width, len(rows) * row_height + 0.5))
    draw_bars(ax, rows, colors)
    ax.legend(
        handles=[
            Patch(color=CAT[slot], label=label)
            for slot, label in enumerate(legend_labels)
        ],
        loc="lower right",
        bbox_to_anchor=(1.0, 1.0),
        ncols=2,
        frameon=False,
        fontsize=9,
        handlelength=0.9,
        handleheight=0.9,
        borderpad=0,
        columnspacing=1.4,
        handletextpad=0.5,
    )
    return figure_svg(fig)


def stacked_chart(rows, series_labels, width=8.4, row_height=0.34):
    fig, ax = plt.subplots(figsize=(width, len(rows) * row_height + 0.5))
    draw_stacked(ax, rows, series_labels)
    return figure_svg(fig)


KEYWORD_GROUPS = (
    ("high_confidence", "Accepted automatically", CAT[0]),
    ("title_confirmed", "Kept, title must confirm", CAT[2]),
    ("rejected", "Not used as a route", CAT[1]),
)


def keyword_scatter(points, works_floor, share_floor, width=8.4, height=4.0):
    """One dot per candidate keyword, on the two numbers the selection rule reads."""
    fig, ax = plt.subplots(figsize=(width, height))
    dashes = (0, (4, 3))
    ax.axvline(share_floor, color=INK_3, linewidth=0.9, linestyle=dashes)
    ax.axhline(works_floor, color=INK_3, linewidth=0.9, linestyle=dashes)
    for key, label, colour in KEYWORD_GROUPS:
        rows = [point for point in points if point["group"] == key]
        if not rows:
            continue
        ax.scatter(
            [point["share"] for point in rows],
            [max(point["works"], 1) for point in rows],
            s=13,
            c=colour,
            linewidths=0,
            alpha=0.8,
            label=f"{label} · {len(rows)}",
        )
        # Name the largest keyword of each outcome, so the cloud can be read. The label
        # turns inward near the right edge, where there is no room beside the point.
        biggest = max(rows, key=lambda point: point["works"])
        inward = biggest["share"] >= 70
        ax.annotate(
            biggest["name"],
            (biggest["share"], max(biggest["works"], 1)),
            textcoords="offset points",
            xytext=(-7 if inward else 7, -3),
            ha="right" if inward else "left",
            color=colour,
            fontsize=9,
        )
    ax.set_yscale("log")
    ax.set_xlim(0, 104)
    ax.set_xlabel(
        "% of all works carrying the keyword that sit on an explicit quantum topic",
        color=INK_3,
        labelpad=6,
    )
    ax.set_ylabel("explicit-topic works carrying it", color=INK_3, labelpad=6)
    ax.text(
        share_floor - 1.5,
        ax.get_ylim()[1],
        f"{share_floor:.0f}% share",
        ha="right",
        va="top",
        color=INK_3,
        fontsize=9,
    )
    ax.text(
        1,
        works_floor * 1.2,
        f"{works_floor} works",
        ha="left",
        va="bottom",
        color=INK_3,
        fontsize=9,
    )
    ax.grid(color=EDGE, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(EDGE)
    ax.tick_params(colors=INK_3, length=0, pad=5)
    ax.legend(
        loc="upper left",
        frameon=False,
        fontsize=9,
        markerscale=1.6,
        borderpad=0,
        handletextpad=0.4,
        labelspacing=0.4,
    )
    return figure_svg(fig)


def topic_names(snapshot_path):
    """Topic id -> display name, from the taxonomy snapshot step 01 downloaded."""
    names = {}
    for line in open(snapshot_path, encoding="utf-8"):
        record = json.loads(line)
        names[record["id"].rsplit("/", 1)[-1]] = record["display_name"]
    return names


def topic_view_example(features_path, scores_path, explicit_ids):
    """One real work's topic vector, before and after the explicit-quantum topics are cut.

    The example is the in-corpus work with the largest topic distance that still carries a
    masked topic, so the reader sees the case the topic view actually acts on rather than a
    flattering one.
    """
    best = None
    for row in csv.DictReader(open(scores_path, encoding="utf-8")):
        if row["currently_included"] != "true" or row["is_core"] == "true":
            continue
        if row["topic_distance"] == "":
            continue
        distance = float(row["topic_distance"])
        if best is None or distance > best[0]:
            best = (distance, row["work_id"])
    wanted = best[1]
    for line in open(features_path, encoding="utf-8"):
        record = json.loads(line)
        if record["work_id"] != wanted:
            continue
        kept = [(t, s) for t, s in record["topics"] if t not in explicit_ids]
        dropped = [(t, s) for t, s in record["topics"] if t in explicit_ids]
        return record, kept, dropped, best[0]
    raise RuntimeError(f"{wanted} is scored but absent from {features_path}")


GROUP_LABELS = {
    "core": "Core set",
    "included": "In corpus, not core",
    "excluded": "Excluded candidates",
}


def distance_plot(distances, thresholds, k):
    """Both views, one row each. The two are separate measurements on separate scales.

    Sharing an x-axis would imply the two distances are comparable numbers, which they are
    not: the text view spreads a corpus across roughly 0.08 to 0.55, while half the core
    sits at a topic distance of about zero.
    """
    figure = make_subplots(
        rows=2,
        cols=1,
        vertical_spacing=0.17,
        subplot_titles=(
            f"text view — cosine distance to the {k} nearest core works, "
            "on what the authors wrote",
            f"topic view — the same {k}-neighbour distance on the subjects OpenAlex assigned",
        ),
    )
    panel_titles(figure)
    for row, view in ((1, "text"), (2, "topic")):
        centers, shares, counts = histogram(distances[view])
        for slot, key in enumerate(GROUPS):
            if row == 1:
                y, template = shares[key], "%{y:.2f}% of group · %{customdata:,} works"
            else:
                # The topic distances are quantized, so a histogram of them is spikes
                # rather than a shape, and more than half the core lands in one bin. The
                # running total is the same information without either artefact.
                y = list(itertools.accumulate(shares[key]))
                template = "%{y:.1f}% of group at or below · %{customdata:,} works here"
            figure.add_trace(
                go.Scatter(
                    x=centers,
                    y=y,
                    mode="lines",
                    name=GROUP_LABELS[key],
                    legendgroup=key,
                    showlegend=row == 1,
                    line=dict(color=CAT[slot], width=2),
                    customdata=counts[key],
                    hovertemplate=template + "<extra></extra>",
                ),
                row=row,
                col=1,
            )
        # The bands the verdicts come from, drawn where they actually fall.
        for percentile, dash in ((95, "dash"), (99, "dot")):
            key = "borderline" if percentile == 95 else "far"
            figure.add_vline(
                x=thresholds[f"{view}_{key}_distance"],
                row=row,
                col=1,
                line=dict(color=NEUTRAL, width=1.1, dash=dash),
                annotation_text=f"core p{percentile}",
                annotation_font=dict(size=9, color=INK_3),
                # Row 2's curves run along the top, so its labels go underneath them.
                annotation_position=("top " if row == 1 else "bottom ")
                + ("right" if percentile == 99 else "left"),
            )
    expected = 4
    if len(figure.layout.shapes) != expected:
        raise RuntimeError(
            f"Expected {expected} band rules on the distance figure, found "
            f"{len(figure.layout.shapes)}"
        )
    figure.update_layout(
        height=520,
        margin=dict(l=58, r=24, t=44, b=52),
        legend=dict(orientation="h", y=-0.13, x=0.0, font=dict(size=10)),
        **PLOT_BASE,
    )
    figure.update_xaxes(**AXIS)
    figure.update_yaxes(
        title_text="% of group", rangemode="tozero", row=1, col=1, **AXIS
    )
    figure.update_yaxes(
        title_text="cumulative % of group", range=[0, 100], row=2, col=1, **AXIS
    )
    figure.update_xaxes(title_text="cosine distance", row=2, col=1)
    return figure


# --------------------------------------------------- three areas across the years


def read_area_affinities(path, short_names):
    """Per year, over the works currently in the corpus: mean affinity, mean entropy, leanings.

    Sums rather than stored rows: this is 83k lines and the report only ever needs the
    per-year means.
    """
    years = {}
    for row in csv.DictReader(open(path, encoding="utf-8")):
        if row["currently_included"] != "true":
            continue
        year = int(row["publication_year"] or 0)
        if not year:
            continue
        block = years.setdefault(
            year,
            {
                "works": 0,
                "affinity": dict.fromkeys(short_names, 0.0),
                "leaning": dict.fromkeys(short_names, 0),
                "entropy": 0.0,
            },
        )
        block["works"] += 1
        for name in short_names:
            block["affinity"][name] += float(row[f"affinity_{name}"])
        block["leaning"][row["leaning"]] += 1
        block["entropy"] += float(row["normalized_entropy"])
    return years


def area_means(years, short_names, floor, last_year):
    """Per-year means over the whole year axis, blank where a year cannot carry one.

    The axis keeps every year the corpus has so this figure and the volume figure below it
    line up; a year under the floor, or the incomplete last one, contributes None and the
    line simply breaks there rather than the year vanishing from the scale.
    """
    axis = sorted(years)
    drawn = [
        year for year in axis if years[year]["works"] >= floor and year <= last_year
    ]
    dropped = sum(
        years[year]["works"] for year in axis if year not in drawn and year <= last_year
    )

    def per_year(pick):
        return [
            pick(years[year]) / years[year]["works"] if year in drawn else None
            for year in axis
        ]

    means = {name: per_year(lambda b, n=name: b["affinity"][n]) for name in short_names}
    # Fractions, not percentages: these ride the same axis as the mean affinities, so a
    # 0-100 scale would flatten the lines the panel is actually about.
    shares = {name: per_year(lambda b, n=name: b["leaning"][n]) for name in short_names}
    entropy = per_year(lambda b: b["entropy"])
    return axis, drawn, means, shares, entropy, dropped


def area_block(years, short_names, span):
    """Pooled means and leaning shares over a block of years, for the decade table."""
    works = sum(years[year]["works"] for year in span)
    if not works:
        return None
    return {
        "works": works,
        "per_year": works / len(span),
        "affinity": {
            name: sum(years[year]["affinity"][name] for year in span) / works
            for name in short_names
        },
        "leaning": {
            name: 100 * sum(years[year]["leaning"][name] for year in span) / works
            for name in short_names
        },
        "entropy": sum(years[year]["entropy"] for year in span) / works,
    }


PLOT_CONFIG = {"displaylogo": False, "responsive": True, "scrollZoom": False}

# Plotly draws on its own canvas, so the page palette is handed to it explicitly. The
# layout below follows the QSIP study's years figure so the two read as the same chart:
# the surface rather than a transparent panel, unified hover, the legend under the plot,
# and reference marks drawn as shapes in the neutral rather than the grid colour.
PLOT_BASE = dict(
    paper_bgcolor=SURFACE,
    plot_bgcolor=SURFACE,
    font=dict(family=FONT_STACK, color=INK_2, size=11),
    hovermode="x unified",
    hoverlabel=dict(font_family=FONT_STACK, font_size=11),
)
AXIS = dict(gridcolor=EDGE, linecolor=EDGE, zeroline=False)


def panel_titles(figure):
    """Left-align the subplot titles, as the reference figure does."""
    for annotation in figure.layout.annotations:
        annotation.update(
            x=0.0, xanchor="left", xref="paper", font=dict(size=11, color=INK)
        )


def end_labels(axis, means, short_names):
    """One label per line at its right-hand end, in that line's own colour.

    Two lines can finish on top of each other — Communication and Sensing do, in the last
    few years — so a pair closer than this gets pushed apart. Colour alone would leave the
    reader guessing which label belongs to which line.
    """
    ends = []
    for slot, name in enumerate(short_names):
        drawn = [(x, y) for x, y in zip(axis, means[name]) if y is not None]
        if drawn:
            ends.append([drawn[-1][0], drawn[-1][1], name, CAT[slot]])
    ends.sort(key=lambda end: end[1])
    shifts = [0.0] * len(ends)
    for lower, upper in zip(range(len(ends) - 1), range(1, len(ends))):
        if ends[upper][1] - ends[lower][1] < 0.035:
            shifts[lower] -= 7.0
            shifts[upper] += 7.0
    return [
        dict(
            x=x,
            y=y,
            xref="x",
            yref="y",
            text=f" {name}",
            yshift=shift,
            showarrow=False,
            xanchor="left",
            yanchor="middle",
            font=dict(size=10.5, color=colour),
        )
        for (x, y, name, colour), shift in zip(ends, shifts)
    ]


def area_plot(
    axis, means, shares, entropy, works, short_names, last_complete, tilt_end=None
):
    """The reference figure: mean affinity over works published over mean entropy.

    Three panels on one shared year axis. The volume panel is the weakest of the three
    readings, so it gets the least height; the affinity panel carries the point and gets
    the most.
    """
    figure = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        row_heights=[0.54, 0.16, 0.30],
        subplot_titles=(
            "mean affinity to each area",
            "works published",
            "mean normalized entropy — how mixed a typical paper is",
        ),
    )
    panel_titles(figure)

    for slot, name in enumerate(short_names):
        figure.add_trace(
            go.Scatter(
                x=axis,
                y=means[name],
                mode="lines",
                name=name,
                legendgroup=name,
                line=dict(color=CAT[slot], width=2.2),
                hovertemplate=f"<b>{name}</b> %{{y:.3f}} mean affinity in %{{x}}<extra></extra>",
            ),
            row=1,
            col=1,
        )
    # The fragile version of the same story: a leaning is a winner-takes-all reading of
    # three numbers that are usually close, so these are off until the reader asks.
    for slot, name in enumerate(short_names):
        figure.add_trace(
            go.Scatter(
                x=axis,
                y=shares[name],
                mode="lines",
                name=f"{name} — share leaning",
                legendgroup=f"share {name}",
                visible="legendonly",
                line=dict(color=CAT[slot], width=1.6, dash="dot"),
                hovertemplate=(
                    f"<b>{name}</b> %{{y:.1%}} of %{{x}}'s works lean this way<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )
    figure.add_trace(
        go.Bar(
            x=axis,
            y=works,
            name="works",
            showlegend=False,
            marker=dict(
                color=[NEUTRAL if year <= last_complete else MUTED for year in axis],
                line=dict(width=0),
            ),
            hovertemplate="%{y:,} works in %{x}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=axis,
            y=entropy,
            mode="lines",
            name="mean entropy",
            showlegend=False,
            line=dict(color=INK_2, width=2.2),
            hovertemplate="mean entropy %{y:.3f} in %{x}<extra></extra>",
        ),
        row=3,
        col=1,
    )

    # After the traces: given row/col, add_hline resolves its axis from what is already in
    # the subplot and silently adds nothing to an empty one.
    even = 1.0 / len(short_names)
    figure.add_hline(
        y=even, row=1, col=1, line=dict(color=NEUTRAL, width=1.1, dash="dash")
    )
    if tilt_end is not None:
        for row in (1, 2, 3):
            figure.add_vline(
                x=tilt_end,
                row=row,
                col=1,
                line=dict(color=NEUTRAL, width=1.1, dash="dot"),
            )
    expected = 1 + (3 if tilt_end is not None else 0)
    if len(figure.layout.shapes) != expected:
        raise RuntimeError(
            f"the years figure carries {len(figure.layout.shapes)} reference shapes, not "
            f"{expected}; add_hline/add_vline must be called after the traces"
        )

    notes = [
        # At the right-hand end, where the three series have left the third of the panel
        # empty; at the left they start close enough to it to be written over.
        dict(
            text="an even three-way split",
            x=float(axis[-1]),
            y=even,
            xref="x",
            yref="y",
            showarrow=False,
            xanchor="right",
            yanchor="bottom",
            yshift=3,
            font=dict(size=9, color=INK_2),
        ),
        # A zoom that does not say so is a lie by axis.
        dict(
            text="note the zoom — normalized entropy runs 0 to 1",
            x=1.0,
            y=1.0,
            xref="x domain",
            yref="y3 domain",
            showarrow=False,
            xanchor="right",
            yanchor="top",
            yshift=-2,
            font=dict(size=9, color=INK_2),
        ),
    ]
    if tilt_end is not None:
        notes.append(
            dict(
                text=f"{tilt_end}",
                x=float(tilt_end),
                y=1.0,
                xref="x",
                yref="y domain",
                showarrow=False,
                xanchor="left",
                yanchor="top",
                xshift=3,
                font=dict(size=9, color=INK_2),
            )
        )
    figure.update_layout(
        height=780,
        bargap=0.25,
        margin=dict(l=60, r=70, t=46, b=56),
        legend=dict(orientation="h", y=-0.11, x=0.0, font=dict(size=10)),
        annotations=list(figure.layout.annotations)
        + notes
        + end_labels(axis, means, short_names),
        **PLOT_BASE,
    )
    figure.update_xaxes(**AXIS)
    figure.update_yaxes(**AXIS)
    figure.update_yaxes(
        rangemode="tozero",
        title=dict(text="mean affinity", font=dict(size=10)),
        row=1,
        col=1,
    )
    figure.update_yaxes(
        rangemode="tozero", title=dict(text="works", font=dict(size=10)), row=2, col=1
    )
    # Autorange puts the last point on the axis line, and this panel is already a zoom, so
    # it gets an explicit padded range instead.
    drawn_entropy = [value for value in entropy if value is not None]
    low, high = min(drawn_entropy), max(drawn_entropy)
    pad = (high - low) * 0.10 or 0.01
    figure.update_yaxes(
        title=dict(text="entropy", font=dict(size=10)),
        range=[low - pad, high + pad],
        row=3,
        col=1,
    )
    return figure


# ------------------------------------------------------------------- page bits


def esc(value):
    return html.escape(str(value), quote=True)


def interactive(plot, caption):
    """A plotly figure as a page figure. The library itself is embedded once, in <head>."""
    markup = to_html(plot, include_plotlyjs=False, full_html=False, config=PLOT_CONFIG)
    return f'<figure class="interactive">{markup}<figcaption>{caption}</figcaption></figure>'


def figure(svg, caption, narrow=False):
    css = ' class="narrow"' if narrow else ""
    return f"<figure{css}>{svg}<figcaption>{caption}</figcaption></figure>"


def plain(value):
    """Drop markup from a title. OpenAlex stores MathML inline, and the tags are noise."""
    return re.sub(r"\s+", " ", re.sub(r"<[^>]*>", " ", str(value))).strip()


def clip(value, limit):
    """Shorten a label; the full text still travels in the cell's tooltip."""
    value = str(value)
    return value if len(value) <= limit else value[: limit - 1].rstrip(" ,") + "…"


def table(headers, rows):
    """A cell may be a (display, full) pair, in which case the full text is a tooltip."""
    head = "".join(f"<th>{esc(h)}</th>" for h in headers)
    cells = []
    for row in rows:
        rendered = []
        for cell in row:
            if isinstance(cell, tuple):
                rendered.append(f'<td title="{esc(cell[1])}">{esc(cell[0])}</td>')
            else:
                rendered.append(f"<td>{esc(cell)}</td>")
        cells.append("<tr>" + "".join(rendered) + "</tr>")
    return (
        f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(cells)}</tbody></table>"
    )


def details(summary, content):
    return f"<details><summary>{esc(summary)}</summary>{content}</details>"


STYLE = f"""
:root {{ color-scheme: light; }}
* {{ box-sizing: border-box; }}
body {{
  margin: 0; padding: 56px 24px 96px; background: {SURFACE}; color: {INK};
  font: 15px/1.6 {FONT_STACK};
}}
main {{ max-width: 820px; margin: 0 auto; }}
h1 {{ font-size: 25px; line-height: 1.25; margin: 0 0 6px; letter-spacing: -0.01em; font-weight: 620; }}
h2 {{ font-size: 16px; margin: 52px 0 6px; letter-spacing: -0.005em; font-weight: 620; }}
p {{ margin: 10px 0; color: {INK_2}; max-width: 72ch; }}
p.lede {{ color: {INK}; }}
p.flow, p.caption {{ color: {INK_3}; font-size: 12.5px; }}
p.flow {{ margin: 4px 0; }}
figure {{ margin: 18px 0 4px; }}
figure.interactive .plotly-graph-div {{ width: 100% !important; }}
figure.interactive .js-plotly-plot {{ margin: 0 -8px; }}
figure svg {{ display: block; width: 100%; height: auto; }}
figure.narrow svg {{ max-width: 470px; }}
figcaption {{ color: {INK_3}; font-size: 12.5px; margin-top: 6px; max-width: 72ch; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; margin: 10px 0 2px; }}
th, td {{ text-align: left; padding: 7px 12px 7px 0; border-bottom: 1px solid {EDGE}; color: {INK_2}; }}
th {{ color: {INK_3}; font-weight: 600; font-size: 12px; }}
td:not(:first-child) {{ font-variant-numeric: tabular-nums; }}
details {{ margin: 10px 0 0; }}
summary {{ cursor: pointer; color: {INK_3}; font-size: 12.5px; }}
ul {{ color: {INK_2}; margin: 10px 0; padding-left: 20px; max-width: 72ch; }}
li {{ margin: 4px 0; }}
code {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12.5px; color: {INK}; }}
"""


def build_report():
    core_manifest = load_json(CORE_MANIFEST)
    outlier_manifest = load_json(OUTLIER_MANIFEST)
    embedding_manifest = load_json(EMBEDDING_MANIFEST)
    filter_manifest = load_json(FILTER_MANIFEST)
    union_manifest = load_json(UNION_MANIFEST)
    config = load_json(CORE_CONFIG)
    keyword_config = load_json(union_manifest["keyword_derivation_config"])
    topic_manifest = load_json(TOPIC_MANIFEST)
    discovery_config = load_json(topic_manifest["discovery_config"])
    record_gates = load_json(RECORD_GATES)

    verify(FEATURES, core_manifest, ("artifacts", "candidate_features"))
    verify(SCORES, outlier_manifest, ("artifacts", "outlier_scores"))
    verify(CLUSTERS, outlier_manifest, ("artifacts", "outlier_clusters"))
    verify(AUDIT, filter_manifest, ("artifacts", "candidate_audit"))
    # Each file matching its own manifest is not enough. The scores carry a frozen
    # currently_included column, so if the filter has been rerun since step 08, every
    # corpus-versus-excluded count on this page would describe a corpus that no longer
    # exists while still passing all of the checks above.
    scored_audit = (
        (outlier_manifest.get("inputs") or {}).get("candidate_audit") or {}
    ).get("sha256")
    if scored_audit != filter_manifest["artifacts"]["candidate_audit"]["sha256"]:
        raise RuntimeError(
            "The outlier scores were built against a different filter run than the audit on "
            "disk; rerun python src/08_detect_quantum_outliers.py"
        )
    if core_manifest["topic_inventory_sha256"] != file_sha256(TOPIC_CANDIDATES):
        raise RuntimeError("The topic inventory changed since the core set was built")
    if load_json(TOPIC_MANIFEST)["output_sha256"] != file_sha256(TOPIC_CANDIDATES):
        raise RuntimeError("The topic inventory does not match its own manifest")
    if union_manifest["derived_keywords_sha256"] != file_sha256(DERIVED_KEYWORDS):
        raise RuntimeError(
            "The derived-keyword inventory changed since the union was built"
        )
    if filter_manifest["record_gates_config_sha256"] != file_sha256(RECORD_GATES):
        raise RuntimeError(
            "The record gates changed since the filter ran; rerun step 4"
        )

    if not os.path.exists(AFFINITY_MANIFEST):
        raise RuntimeError(
            "The area affinities are missing; run "
            ".venv/bin/python src/09_score_quantum_area_affinities.py"
        )
    affinity_manifest = load_json(AFFINITY_MANIFEST)
    verify(AFFINITIES, affinity_manifest, ("artifacts", "affinities"))
    if affinity_manifest["inputs"]["outlier_scores"]["sha256"] != file_sha256(SCORES):
        raise RuntimeError(
            "The affinities were scored against a different verdict table; rerun "
            "src/09_score_quantum_area_affinities.py"
        )
    if affinity_manifest["model"] != embedding_manifest["model"]:
        raise RuntimeError(
            "The affinities and the work embeddings use different encoders"
        )

    if not os.path.exists(AFFINITY_MANIFEST):
        raise RuntimeError(
            "The area affinities are missing; run "
            ".venv/bin/python src/09_score_quantum_area_affinities.py"
        )
    affinity_manifest = load_json(AFFINITY_MANIFEST)
    verify(AFFINITIES, affinity_manifest, ("artifacts", "affinities"))
    if affinity_manifest["inputs"]["outlier_scores"]["sha256"] != file_sha256(SCORES):
        raise RuntimeError(
            "The affinities were scored against a different verdict table; rerun "
            "src/09_score_quantum_area_affinities.py"
        )
    if affinity_manifest["model"] != embedding_manifest["model"]:
        raise RuntimeError(
            "The affinities and the work embeddings use different encoders"
        )

    encoder_name, encoder_note, encoder_training = model_notes(
        embedding_manifest["model"]
    )

    core = core_manifest["counts"]
    counts = outlier_manifest["counts"]
    verdicts = counts["verdicts"]
    model = outlier_manifest["model"]
    thresholds = model["core_distance_thresholds"]

    stages, retrieval_channels, core_cases = core_funnel(
        FEATURES, config["core_rule"], load_core_module()
    )
    topic_total, topic_groups = topic_inventory(TOPIC_CANDIDATES)
    topic_rows = topic_examples(TOPIC_CANDIDATES)
    high = keyword_config["selection_tiers"]["high_confidence"]
    low = keyword_config["selection_tiers"]["title_confirmed"]
    keyword_gate, keyword_kept = keyword_selection(
        DERIVED_KEYWORDS, keyword_config["selection_tiers"]
    )
    keyword_rows = keyword_examples(DERIVED_KEYWORDS, keyword_config["selection_tiers"])
    keyword_candidates = keyword_points(DERIVED_KEYWORDS)
    rejected_candidates = [p for p in keyword_candidates if p["group"] == "rejected"]
    if not rejected_candidates:
        raise RuntimeError(
            "No quantum-pattern keyword was rejected, so the step 3 figure and its prose "
            "no longer describe the data"
        )
    top_rejected = max(rejected_candidates, key=lambda point: point["works"])
    # Rejections the two drawn axes cannot explain: they clear both floors and still fail
    # the topic-spread condition, which is the chart's one blind spot.
    corner_rejects = sum(
        1
        for point in rejected_candidates
        if point["works"] >= low["minimum_explicit_seed_works"]
        and point["share"] >= 100 * low["minimum_explicit_share_of_topic_universe"]
    )
    channel_cases = channel_examples(AUDIT, SCORES)
    related_terms = len(discovery_config["term_groups"]["related"])
    explicit_group_count = dict(topic_groups)["Explicit quantum"]
    core_term_count = len(config["core_rule"]["core_term_patterns"])
    families = inclusion_families(AUDIT)
    keyword_only_included = families["keyword only"]
    keyword_and_topic = families["both"]
    without_explicit = (
        filter_manifest["counts"]["included_works"]
        - filter_manifest["counts"]["inclusion_channels_overlap"]["explicit_quantum"]
    )
    taxonomy_size = topic_manifest["input_topic_count"]
    channel_counts = filter_manifest["counts"]["inclusion_channels_overlap"]
    filter_rows = [
        ("Explicit-quantum topic", channel_counts["explicit_quantum"]),
        (
            "Broad-related topic + paper term",
            channel_counts["broad_related_paper_term_confirmation"],
        ),
        ("High-confidence keyword", channel_counts["derived_keyword"]),
        ("Title-confirmed keyword", channel_counts["title_confirmed_derived_keyword"]),
        (
            "Broad-related topic, no term",
            filter_manifest["counts"]["excluded_broad_topic_only"],
        ),
        (
            "Lower tier, no title evidence",
            filter_manifest["counts"][
                "excluded_title_confirmed_keyword_without_title_evidence"
            ],
        ),
        (
            "Keyword-only domain safeguard",
            filter_manifest["counts"]["excluded_keyword_only_blocked_domain"],
        ),
        (
            "Language not accepted",
            filter_manifest["counts"]["excluded_non_accepted_language"],
        ),
        (
            "Record is not a paper",
            filter_manifest["counts"]["excluded_non_paper_work_type"],
        ),
    ]
    filter_colors = [CAT[0]] * 4 + [CAT[1]] * 5
    distances, totals = read_distances(SCORES)
    if load_json(TOPIC_SNAPSHOT_MANIFEST)["output_sha256"] != file_sha256(
        TOPIC_SNAPSHOT
    ):
        raise RuntimeError(
            "The topic taxonomy snapshot does not match its own manifest"
        )
    all_topic_names = topic_names(TOPIC_SNAPSHOT)
    explicit_topic_ids = set(core_manifest["explicit_quantum_topic_ids"])
    topic_case, topic_kept, topic_dropped, topic_case_distance = topic_view_example(
        FEATURES, SCORES, explicit_topic_ids
    )
    near_excluded, far_excluded = excluded_examples(SCORES, AUDIT, record_gates)
    topic_rows = [
        [
            all_topic_names.get(topic_id, topic_id),
            f"{score:.2f}",
            "dropped — explicitly quantum"
            if topic_id in explicit_topic_ids
            else "kept",
        ]
        for topic_id, score in topic_dropped + topic_kept
    ]
    (
        flagged_total,
        channels,
        term_split,
        included_single,
        included_multi,
        lof_agrees,
    ) = flagged_evidence(SCORES, AUDIT)
    clusters = read_clusters(CLUSTERS)

    flagged_rows = flagged_examples(SCORES, AUDIT)

    area_short = [area["short_name"] for area in affinity_manifest["areas"]]
    area_years = read_area_affinities(AFFINITIES, area_short)
    # The corpus was pulled part-way through the manifest's year, so that year is short a
    # publication season and is kept out of every mean.
    last_complete_year = int(filter_manifest["created_at"][:4]) - 1
    year_axis, drawn_years, area_series, area_shares, area_entropy, thin_works = (
        area_means(area_years, area_short, MIN_WORKS_PER_YEAR, last_complete_year)
    )
    partial_year = max(area_years)
    # Where the early tilt stops: the first decade whose successor no longer gains on the
    # leading area. Read off the data rather than asserted.
    decade_means = []
    for decade in sorted({(year // 10) * 10 for year in drawn_years}):
        span = [year for year in drawn_years if (year // 10) * 10 == decade]
        block = area_block(area_years, area_short, span)
        if block:
            decade_means.append((decade, block["affinity"][area_short[0]]))
    settled = next(
        (
            decade
            for (decade, mean), (_, later) in zip(decade_means, decade_means[1:])
            if later - mean < 0.01
        ),
        decade_means[-1][0],
    )
    tilt_end_year = settled + 10

    channel_rows = []
    for label, verdict in (
        ("Explicit-quantum topic", "in"),
        ("Broad-related topic + paper term", "in"),
        ("High-confidence keyword", "in"),
        ("Title-confirmed keyword", "in"),
        ("Broad-related topic, no term", "out"),
        ("Keyword-only domain safeguard", "out"),
    ):
        if label not in channel_cases:
            continue
        year, audit_row = channel_cases[label]
        terms = [
            t.strip()
            for t in audit_row["paper_level_quantum_terms"].split("|")
            if t.strip()
        ]
        channel_rows.append(
            [
                label,
                (clip(audit_row["title"], 52), f"{audit_row['title']} ({year})"),
                ", ".join(terms[:3]) if terms else "none",
                verdict,
            ]
        )

    core_rows = []
    for key, outcome in (
        ("accepted", "in the core"),
        ("weak_topic", "rejected"),
        ("weakest_topic", "rejected"),
        ("no_term", "rejected"),
        ("ambiguous", "rejected"),
        ("record_type", "rejected"),
    ):
        if key not in core_cases:
            continue
        score, title, topic_name, note = core_cases[key]
        core_rows.append(
            [
                outcome,
                (clip(title, 56), title),
                (clip(topic_name, 32), topic_name),
                f"{score:.3f}",
                note,
            ]
        )

    verdict_rows = [
        ("Core set", [counts["core_works"], 0]),
        (
            "Near core",
            [
                verdicts["near_core"] - counts["excluded_and_near_core"],
                counts["excluded_and_near_core"],
            ],
        ),
        (
            "Borderline",
            [
                counts["included_and_borderline"],
                verdicts["borderline"] - counts["included_and_borderline"],
            ],
        ),
        (
            "Far outlier",
            [
                counts["included_and_far"],
                verdicts["far_outlier"] - counts["included_and_far"],
            ],
        ),
        ("Unscored", [verdicts["unscored"], 0]),
    ]
    cluster_rows = [
        (item["label"], [item["included"], item["works"] - item["included"]])
        for item in clusters
    ]

    page = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Swedish quantum corpus: how it is built and reviewed</title>
<style>{STYLE}</style>
<script>{get_plotlyjs()}</script></head><body><main>

<h1>Swedish quantum corpus: how it is built and reviewed</h1>

<p class="lede">OpenAlex assigns each work a domain, field and subfield, up to three scored
<em>topics</em>, and a set of <em>keywords</em> derived from its text. This corpus uses those labels
to find candidates; the review checks where they lead to incorrect results.</p>

<p>Candidate works are found through selected quantum topics and keywords derived from papers in
those topics. A match through a broad topic is kept only when the paper also uses quantum language
in its title, abstract or keywords. Steps 1–4 explain this selection.</p>
<p>The rule has two limits: an incorrect topic can admit an unrelated paper, and a yes-or-no
decision cannot rank relevance. Steps 5–8 compare candidates with a small core of clearly quantum
papers to flag likely false positives and omissions. This review does not change corpus membership.</p>

<h2>Steps 1–2 · Which OpenAlex topics count as quantum</h2>
<p>OpenAlex publishes {
        taxonomy_size:,} topics. Every one of them is matched against a small
written-down vocabulary: the word <em>quantum</em> itself, plus {
        related_terms
    } spellings of twelve
quantum-specific concepts (qubit, entanglement, decoherence, Bell inequality,
single photon, squeezed light, trapped ions, Rydberg atoms, Bose-Einstein condensate, Josephson
junction, nitrogen-vacancy centre, atom interferometry). The match is made against the topic's name
and its taxonomy keywords, never against the topic description, because OpenAlex writes those with
AI. {
        topic_total
    } topics match, and what happens next depends on where the match landed:</p>
<ul>
  <li>the name contains the literal word <em>quantum</em>: an <code>explicit_quantum</code> route,
  accepted automatically.</li>
  <li>only the taxonomy keywords matched: a <code>broad_related</code> route, where the paper itself
  must confirm the term.</li>
  <li>the topic sits in Social Sciences: recorded for the audit trail and never queried.</li>
</ul>
{
        figure(
            bars_chart(
                [
                    ("Matched topics by group", topic_groups, CAT[0]),
                    ("How each candidate was reached", retrieval_channels, CAT[0]),
                ]
            ),
            '''Every route shares the same base filter (<code>authorships.countries:SE</code>, 1950–2026),
and a work retrieved by several routes is counted once.''',
        )
    }
<p>Each accepted topic then becomes one query: every Swedish work OpenAlex files under that topic in
the year window. Three worked examples,
the largest topic in each group:</p>
{table(["Topic", "Matched on", "Group"], topic_rows)}
<p class="caption">The first is <code>explicit_quantum</code> for one reason only: the literal word
<em>quantum</em> appears in the topic's own name. The second never says quantum in its name. It was found through a taxonomy keyword,
and plenty of particle-physics papers under it never discuss a quantum system as such, so its papers
must confirm the term themselves. The third is why the domain gate exists: archaeology uses
"entanglement" for an entirely unrelated idea, so that topic is recorded and never queried.</p>
<p>The {explicit_group_count} explicitly-named quantum topics on their own account 
for {channel_counts["explicit_quantum"]:,} of the
{filter_manifest["counts"]["included_works"]:,} works in the corpus. The other
{
        without_explicit:,}, a third of it, are reachable only through a broad topic or a derived keyword,
and they are not marginal papers: superconductivity, particle physics, atomic clocks and black-hole
theory all sit under topics whose names never say quantum. Dropping those
routes would cost a third of the corpus. Taking them without a condition would drag in most of
physics, which is why they carry the paper-level requirement instead.</p>

<h2>Step 3 · Where the extra keywords come from</h2>
<p>A topic is coarse. OpenAlex has only {
        taxonomy_size:,} of them for all of science and gives a work
about three, so a quantum paper can easily end up filed under a topic that is not in our list at all.
OpenAlex also attaches <em>keywords</em> to each work, which are far more specific: a keyword is a
concept with a stable identifier such as <code>quantum-dot</code>, extracted from the paper's own
text. Those make a second, finer retrieval channel.</p>
<p>The list of keywords to use is not written by hand. It is read off the papers the explicit topics
already returned: every keyword appearing on any of those papers is a candidate, and each one is then
measured on four things:</p>
<ul>
  <li>how often it appears among papers returned by explicit quantum topics. It must appear in at
  least {low["minimum_explicit_seed_works"]} papers.</li>
  <li>how many different explicit topics it spans, so a keyword that only ever appears in one
  subfield does not become a route for the whole corpus.</li>
  <li>what share of all the works carrying it anywhere are explicit-topic works. This is the real
  test of whether the keyword means something quantum. <em>{
        top_rejected["name"]
    }</em> sits on
  {top_rejected["works"]:,} of the seed papers, but only {
        top_rejected["share"]:.0f}% of the works
  carrying it anywhere are on a quantum topic, so as a route it would retrieve half of physics.</li>
  <li>whether the OpenAlex identifier is quantum-specific. Generic identifiers such as
  <code>charm</code> or <code>correlation-function</code> are excluded even when the displayed name
  includes a quantum qualifier.</li>
</ul>
{
        figure(
            keyword_scatter(
                keyword_candidates,
                low["minimum_explicit_seed_works"],
                100 * low["minimum_explicit_share_of_topic_universe"],
            ),
            f'''The chart includes {len(keyword_candidates)} of the {keyword_gate[0][1]:,} keywords observed
on the seed papers. These passed the name and identifier check. The dashed lines mark the minimum
of {low['minimum_explicit_seed_works']} seed papers and the minimum explicit-topic share.
{corner_rejects} further keywords are excluded because they span fewer than
{low['minimum_explicit_topics']} explicit topics. The vertical axis is logarithmic.''',
        )
    }
<p>Two tiers come out of it. {
        keyword_kept[2][1]
    } keywords are accepted as routes outright, which
takes at least {
        high["minimum_explicit_seed_works"]
    } explicit-topic works and a share of at least
{100 * high["minimum_explicit_share_of_topic_universe"]:.0f}%. Another
{
        keyword_kept[1][1] - keyword_kept[2][1]
    } sit between the floors and that stricter bar, and they
still retrieve candidates, but a paper found only through one of them has to carry a quantum term in
its own title before it is kept.</p>
{table(["Keyword", "Explicit-topic works", "Share explicit", "Outcome"], keyword_rows)}
<p class="caption">The largest keyword in each outcome. The rejected examples either appear too
often outside explicit quantum topics or have a generic OpenAlex identifier.</p>
<p>The keyword routes reach {
        keyword_only_included:,} works in the corpus that no topic route found,
and another {
        keyword_and_topic:,} that a topic route found as well. Keywords are extracted from the
text by machine rather than assigned to the paper as a whole, so they are the weaker of the two
signals, which is why a tier of them has to be confirmed by the title.</p>

<h2>Step 4 · The inclusion rule</h2>
<p>At this point every candidate carries a list of the routes that found it. The filter turns that
list into one yes or no, by reading the routes in a fixed order:</p>
<ul>
  <li>the record is not an English-language paper: out before any quantum test is applied. The
  OpenAlex language must be <code>en</code> and the type must not be one of the ten that describe
  something other than a paper, such as an erratum, a dataset or a peer review. A record with no
  language or no type is kept, because an absent label is not evidence.</li>
  <li>an explicit-quantum topic matched: in, with no further test.</li>
  <li>a high-confidence derived keyword matched: in, with no further test.</li>
  <li>only a broad-related topic matched: in only if a quantum term appears in the paper's own title,
  abstract or keywords.</li>
  <li>only a title-confirmed keyword matched: in only if the <em>title</em> matches a quantum
  pattern, which is a harder test than the previous one.</li>
  <li>keywords were the only route and OpenAlex filed the paper in Social Sciences: in only with at
  least two quantum keywords and a strong quantum phrase in the title or abstract, because
  "entanglement" and "quantum leap" are ordinary metaphors there.</li>
</ul>
<p>Each check is fixed: a route or text pattern either matches or it does not. The filter uses no
probability or manual decisions, and the same candidate file always produces the same corpus.</p>
{
        figure(
            split_bars_chart(
                filter_rows,
                filter_colors,
                ["Included through this channel", "Excluded for this reason"],
            ),
            f'''Inclusion channels overlap. A work can arrive through several, so these bars do not sum to
the corpus. {included_single:,} works arrived through exactly one channel and {included_multi:,}
through more than one. The exclusions are exclusive, and the largest bar on the chart is one of
them: works that only matched a broad topic and never mentioned anything quantum in their own
text. The two gate bars count every candidate the gate rejected, including works the quantum rule
would have excluded anyway.''',
        )
    }
<p>Six worked examples, each the newest article that arrived through that channel and no other:</p>
{
        table(
            [
                "Channel",
                "Newest article through it alone",
                "Its own quantum wording",
                "Corpus",
            ],
            channel_rows,
        )
    }
<p class="caption">The first and last rows show the rule's main limitation. An explicit-quantum
topic can admit a work whose text says nothing quantum. A keyword-only match in Social Sciences can
be rejected even when the text does. The review stage checks both kinds of case.</p>

<h2>Step 5 · What survives the core rule</h2>
<p>Steps 5–8 compare candidates with a conservative core of clearly quantum papers. The core is
selected by fixed rules because there is no manually labelled reference set. It is used only for
comparison, not as a gold standard or as the final corpus. A work enters it only when:</p>
<ul>
  <li>its <em>primary</em> topic, the one OpenAlex thinks the paper is mostly about, is one of the
  {explicit_group_count} explicitly-named quantum topics.</li>
  <li>OpenAlex assigned that topic with a score of at least
  {
        config["core_rule"]["minimum_primary_topic_score"]
    }, so weak guesses are dropped.</li>
  <li>the paper's own title or abstract contains an unambiguous quantum term, not just
  "quantum efficiency" or "quantum yield", which are routine photodetector and photochemistry
  vocabulary.</li>
  <li>the record reports science of its own, so errata, prefaces, peer reviews and datasets are
  out.</li>
</ul>
{
        figure(
            bars_chart(
                [(None, [(label, count) for label, count in stages], ORDINAL)],
                row_height=0.34,
            ),
            f'''{stages[0][1]:,} candidates in, {stages[-1][1]:,} core works out, or
{100 * stages[-1][1] / stages[0][1]:.1f}%. {core['core_without_abstract']:,} of them have no
abstract, so their evidence is in the title.''',
        )
    }
<p>The wording must be in the title or abstract, not in the paper keywords, because those keywords
are themselves machine-generated and would make the evidence circular. Below is the closest call at
each check, the work with the highest primary-topic score the check still rejects, plus the
weakest topic label in the whole candidate set:</p>
{table(["Outcome", "Example", "Primary topic", "Score", "Reason"], core_rows)}
<p class="caption">The score gate looks harmless at its boundary, where it rejects a work OpenAlex
placed on a quantum topic with a score just under 0.5, but the second rejected row shows what it is
really for: OpenAlex will attach an explicit-quantum topic to a work with a score of a few
thousandths. The ambiguous-wording rule removes semiconductor papers whose only quantum word is
"quantum efficiency", and the record-type rule removes errata, prefaces, peer reviews and datasets,
whose text describes someone else's science.</p>

<h2>Steps 6–7 · Distance from the core</h2>
<p>Each candidate is scored by its mean cosine distance from the
{
        model["k_neighbors"]
    } nearest core papers. A lower score means greater similarity. Core papers are
not counted as their own neighbours. The report compares three groups:</p>
<ul>
  <li><b>core</b> ({totals["core"]:,}): the reference papers selected in step 5.</li>
  <li><b>in corpus, not core</b> ({
        totals["included"]:,}): accepted works that do not meet the core
  criteria.</li>
  <li><b>excluded candidates</b> ({
        totals["excluded"]:,}): retrieved works rejected in step 4. They
  are scored to identify possible omissions.</li>
</ul>
<p>Distance is calculated separately from the paper's text and its OpenAlex topics:</p>
<ul>
  <li><b>text view:</b> the title and abstract encoded with {encoder_name}, {
        encoder_note
    }. Shared
  vocabulary can make unrelated papers appear similar.</li>
  <li><b>topic view:</b> a vector of the work's scored OpenAlex topics, from a taxonomy of
  {len(all_topic_names):,} topics. Incorrect topic assignments affect this view.</li>
</ul>
<p>To avoid repeating the core selection rule, the {
        core_term_count
    } core-defining terms are removed
before encoding the text, and the {
        explicit_group_count
    } explicit-quantum topics are removed from
the topic vector. The table shows the remaining topics for
"{clip(topic_case["title"], 60)}", the in-corpus work with the largest topic distance
({topic_case_distance:.3f}):</p>
{table(["OpenAlex topic on the work", "Score", "In the topic view?"], topic_rows)}
<p class="caption">After explicit-quantum topics are removed, the remaining topics determine the
topic distance. A work with no remaining topics shared by the core has a distance near 1.</p>
{
        interactive(
            distance_plot(distances, thresholds, model["k_neighbors"]),
            f'''Dashed and dotted lines mark the core's 95th and 99th percentiles, used in step 8. The
median core text distance is {thresholds['core_median_text_distance']:.3f}; non-core and excluded
works generally lie farther away. Topic distances contain many ties because works have few topics:
about 60% of the core has a distance near zero. Both views overlap, so the scores are used for
ranking and review rather than automatic inclusion.''',
        )
    }
<p>The excluded group is scored but never re-admitted, so both ends of it are worth reading. Works
stopped by the language gate are left out of both tables below: OpenAlex reads language off the
title, so that gate turns on a metadata guess rather than on the paper, and the encoder cannot read
a non-English abstract either way. What remains are works the topic and keyword rule actually
judged. The {
        len(near_excluded)
    } closest to the core are the ones step 4 is most likely to have been
wrong about:</p>
{
        table(
            [
                "Excluded candidate",
                "Primary topic",
                "Year",
                "Text distance",
                "Why step 4 said no",
            ],
            near_excluded,
        )
    }
<p class="caption">These sit inside the core's own distance range, and several are quantum papers on
any ordinary reading. The broad-topic rule accounts for most of them: they were retrieved by a
broad-related topic and their own title and abstract never used a quantum word the rule recognises,
which is the price of requiring paper-level wording rather than trusting the topic label. The rest
are record-type rejections. Peer reviews belong there, since their text describes someone else's
paper, but the gate also removes records OpenAlex typed as <code>other</code>, and some of those are
ordinary papers.</p>
<p>The {
        len(far_excluded)
    } farthest are the opposite case. They are restricted to records with a
real abstract, because a bare title sits far from everything and would fill the table with works
the text view never really measured:</p>
{
        table(
            [
                "Excluded candidate",
                "Primary topic",
                "Year",
                "Text distance",
                "Why step 4 said no",
            ],
            far_excluded,
        )
    }
<p class="caption">Here the distance and the exclusion agree. Almost all were retrieved by a
broad-related topic and carry no quantum wording of their own, and the subjects — palaeoclimate,
constitutional law, radiology, obituaries — are plainly unrelated. The travel-writing book shows the
blocked-domain rule doing its job, since "entanglement" in a literary title is not a physical claim.
That these are the farthest the filter let through is the useful part: the excluded group has no
tail of quantum work sitting out at this end.</p>

<h2>Step 8 · Verdicts against the current corpus</h2>
<p>Each distance is converted to a percentile of the core distribution, calculated with each core
paper left out of its own neighbourhood. A candidate at the 99th percentile is farther from the
core than 99% of core papers. In the text view, the 95th percentile is
{thresholds["text_borderline_distance"]:.3f} and the 99th is
{thresholds["text_far_distance"]:.3f}. The two views produce four verdicts:</p>
<ul>
  <li><b>near core</b>: neither view is past the 95th.</li>
  <li><b>borderline</b>: one view is past the 99th, or both are past the 95th.</li>
  <li><b>far outlier</b>: both views are past the 99th.</li>
  <li><b>unscored</b>: neither view is available.</li>
</ul>
{
        figure(
            stacked_chart(verdict_rows, ["In the corpus", "Not in the corpus"]),
            f'''{counts['included_and_far']:,} corpus works are far from the core in both views. A local
outlier factor also flags {lof_agrees:,} of them. The recall list contains
{counts['excluded_and_near_core']:,} excluded candidates near the core.
{counts['works_below_text_floor']:,} records have too little text and use only the topic view;
{verdicts['unscored']:,} have neither view.''',
        )
    }

<h2>What the far group is made of</h2>
<p>The {counts["verdicts"]["far_outlier"]:,} far outliers are divided into {
        len(clusters)
    } k-means
clusters using their text embeddings. Each cluster is labelled with terms that are unusually common
among its papers. These labels describe the results; they were not predefined.</p>
{
        figure(
            stacked_chart(cluster_rows, ["In the corpus", "Not in the corpus"]),
            f'''The {len(clusters)} clusters are ordered by the number of members currently in the corpus.
Most corrosion and steel works were already excluded; more speech-coding and surfactant papers were
included.''',
        )
    }
{
        details(
            "Clusters with their most common primary topic",
            table(
                ["Cluster terms", "Works", "In corpus", "Most common primary topic"],
                [
                    [
                        item["label"],
                        f"{item['works']:,}",
                        f"{item['included']:,}",
                        item["topics"],
                    ]
                    for item in clusters
                ],
            ),
        )
    }

<h2>Why those {flagged_total} works were admitted</h2>
<p>The audit file records each candidate's retrieval routes, matched terms and inclusion decision.
Linking those records to the outlier flags shows which parts of the inclusion rule admitted them.</p>
{
        figure(
            bars_chart(
                [
                    ("Inclusion channel", channels, CAT[0]),
                    ("Paper-level quantum wording", term_split, CAT[0]),
                ]
            ),
            '''Most flagged inclusions come from two checks. Some works are accepted through an incorrect
explicit-quantum topic even though their text contains no quantum terms. The pattern
<code>\\bquanti[sz]\\w*\\b</code> also matches "quantization" in signal-processing papers.''',
        )
    }
{
        table(
            [
                "Work in the corpus",
                "Its primary topic",
                "Distance",
                "Admitted by",
                "Its quantum wording",
            ],
            flagged_rows,
        )
    }
<p class="caption">The five corpus works farthest from the core. Several received incorrect
explicit-quantum topics and were therefore accepted automatically. The scores CSV includes the
three nearest core works for each row.</p>

<h2>Research areas over time</h2>
<p>Each work is compared with descriptions of
{", ".join(area["name"] for area in affinity_manifest["areas"][:-1])} and
{affinity_manifest["areas"][-1]["name"]} from
<code>{
        affinity_manifest["inputs"]["anchor_config"]["path"]
    }</code>. The descriptions and papers use
the same encoder. A softmax with T&nbsp;=&nbsp;{
        affinity_manifest["softmax_temperature"]
    } converts the
similarities into three relative affinities that sum to one. They show which description is closest,
not whether a paper belongs to an area.</p>
{
        interactive(
            area_plot(
                year_axis,
                area_series,
                area_shares,
                area_entropy,
                [area_years[year]["works"] for year in year_axis],
                area_short,
                last_complete_year,
                tilt_end_year,
            ),
            f"Annual means for corpus works with a valid year. The dashed line marks equal affinity; the "
            f"vertical dotted line marks {tilt_end_year}, when the early trend levels off. Means are omitted "
            f"for years with fewer than {MIN_WORKS_PER_YEAR} works ({thin_works:,} earlier works) and for "
            f"the partial year {partial_year}; the volume panel still includes them. Data were collected on "
            f"{filter_manifest['created_at'][:10]}. The median gap between the two closest areas is "
            f"{affinity_manifest['median_top_two_cosine_gap']:.4f}, so many assignments are weak.",
        )
    }

<h2>Limits</h2>
<ul>
  <li>The core favours papers that state their quantum subject explicitly. It is not a gold
  standard.</li>
  <li>OpenAlex assigns only about three topics per work, producing many tied topic distances.</li>
  <li>Shared boilerplate can place prefaces and editorials near the core. Check
  <code>work_type</code> and <code>low_text_evidence</code>.</li>
  <li>{encoder_name} truncates at {embedding_manifest["max_tokens"]} tokens and
  {encoder_training}.</li>
  <li>The percentile thresholds, {model["k_neighbors"]} neighbours and {
        len(clusters)
    } clusters are
  design choices, not calibrated error rates.</li>
</ul>

</main></body></html>
"""
    tmp = OUTPUT + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(page)
    os.replace(tmp, OUTPUT)
    return {
        "output": OUTPUT,
        "bytes": os.path.getsize(OUTPUT),
        "core_works": core["core_works"],
        "flagged": flagged_total,
        "clusters": len(clusters),
    }


def main():
    os.chdir(ROOT)
    result = build_report()
    print(f"Report:      {result['output']} ({result['bytes'] / 1024:.0f} KB)")
    print(f"Core works:  {result['core_works']:,}")
    print(f"Flagged:     {result['flagged']:,} in-corpus far outliers")
    print(f"Clusters:    {result['clusters']}")


if __name__ == "__main__":
    main()
