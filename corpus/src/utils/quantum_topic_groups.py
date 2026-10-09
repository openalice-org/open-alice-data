"""Shared names and routing rules for quantum topic candidate groups."""

EXPLICIT_QUANTUM = "explicit_quantum"
INTRINSICALLY_QUANTUM = "intrinsically_quantum"
BROAD_RELATED = "broad_related"

TOPIC_GROUPS = {
    EXPLICIT_QUANTUM,
    INTRINSICALLY_QUANTUM,
    BROAD_RELATED,
}


def _pipe_values(value):
    return {
        item.strip().casefold() for item in (value or "").split("|") if item.strip()
    }


def candidate_group(candidate):
    """Return the inclusion group for one topic-inventory row.

    Conditional taxonomy domains retain the broad paper-level confirmation gate.
    Otherwise a literal ``quantum`` match in the topic name is explicit, another
    configured name match is intrinsically quantum, and keyword-only discovery
    is broad related.
    """
    if candidate.get("taxonomy_gate") == "conditional":
        return BROAD_RELATED
    matched_name_terms = _pipe_values(candidate.get("matched_name_terms"))
    if "quantum" in matched_name_terms:
        return EXPLICIT_QUANTUM
    if matched_name_terms:
        return INTRINSICALLY_QUANTUM
    return BROAD_RELATED
