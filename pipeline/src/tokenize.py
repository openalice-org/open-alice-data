"""Tokenization utilities for papers and patents."""

import re

STOPWORDS: set[str] = {
    "about", "above", "according", "across", "after", "again", "against", "all",
    "also", "among", "and", "another", "any", "application", "applications", "are",
    "around", "based", "because", "been", "being", "between", "both", "can",
    "claim", "claims", "comprising", "consist", "consists", "containing", "could",
    "device", "devices", "does", "each", "effect", "embodiment", "embodiments",
    "fig", "figure", "first", "for", "from", "further", "has", "have", "having",
    "herein", "into", "invention", "inventions", "includes", "including", "least",
    "may", "method", "methods", "more", "one", "only", "other", "over", "paper",
    "patent", "patents", "plurality", "present", "provided", "provides", "related",
    "said", "same", "second", "such", "system", "systems", "than", "that", "the",
    "their", "thereof", "these", "this", "those", "through", "using", "where",
    "wherein", "which", "with", "within", "without", "work", "works", "would",
    "configured", "configuration", "corresponding", "determine", "determined",
    "determining", "module", "modules", "obtained", "obtaining", "order", "output",
    "outputs", "process", "processor", "processors", "receive", "received",
    "receiving", "storage", "target",
}

GENERIC_SCIENCE_WORDS: set[str] = {
    "analysis", "approach", "article", "case", "comprise", "comprises", "data",
    "different", "energy", "experimental", "high", "information", "large", "low",
    "model", "models", "new", "novel", "performance", "physic", "physics",
    "properties", "results", "review", "sample", "samples", "science", "show",
    "shown", "study", "technology", "they", "traditional", "use", "used",
    # Generic quantifiers / pronouns / vague modifiers
    "two", "three", "four", "five", "six", "ten", "several", "many", "much",
    "most", "less", "few", "single", "multiple", "various", "some", "every",
    "our", "ours", "here", "there", "his", "her", "hers", "him", "she", "they",
    "them", "its", "it", "you", "your", "yours", "we", "us", "any",
    # Vague verbs / states
    "allow", "allows", "allowed", "allowing", "due", "was", "were", "been",
    "found", "find", "finds", "finding", "findings", "give", "gives", "given",
    "gave", "giving", "obtain", "obtains", "obtained", "obtaining",
    "provide", "provides", "providing", "described", "describe", "describes",
    "describing", "description", "show", "shows", "showed", "showing",
    "demonstrate", "demonstrates", "demonstrated", "demonstrating",
    "consider", "considers", "considered", "considering", "explore", "explored",
    "explores", "exploring", "exploration", "investigate", "investigates",
    "investigated", "investigating",
    # Generic adverbs / adjectives / connectives
    "very", "however", "but", "not", "none", "well", "finally", "thus", "hence",
    "therefore", "respectively", "generally", "particular", "particularly",
    "specifically", "essentially", "typically", "moreover", "furthermore",
    "previously", "currently", "recently",
    # Generic measurement / shape vocabulary
    "small", "smaller", "smallest", "big", "bigger", "biggest", "large",
    "larger", "largest", "magnitude", "magnitudes", "increase", "increased",
    "increases", "increasing", "decrease", "decreased", "decreases",
    "decreasing", "value", "values", "boundary", "boundaries",
    # Generic academic vocabulary
    "abstract", "pure", "applied", "theoretical", "practical", "technique",
    "techniques", "control", "controls", "controlled", "controlling",
    # Web / URL artifacts
    "org", "http", "https", "www", "com", "edu", "html", "pdf", "doi", "url",
    # Other noise observed in clusters
    "own", "collapse", "respectively",
    # Added after cluster review
    "lead", "leads", "led", "leading", "discuss", "discussed", "discussing",
    "discussion", "presence", "measured", "developed", "performed",
    "shift", "shifts", "shifted", "condition", "conditions",
    "scheme", "schemes", "proposed", "propose", "proposes",
    "compared", "compare", "compares", "comparison",
    "enhancement", "enhanced", "enhance", "enhances",
    # MathML / XML artifacts
    "xmln", "mathml", "mml", "mrow", "mtext", "mfrac", "msub", "msup",
}


def tokenize(text: str) -> list[str]:
    """Lowercase, alpha+digit tokens of length >= 3, with stopword removal and light stemming."""
    words = re.findall(r"[a-z][a-z0-9\-]{2,}", (text or "").lower())
    tokens: list[str] = []
    for word in words:
        word = word.strip("-")
        if len(word) < 3:
            continue
        if word in STOPWORDS or word in GENERIC_SCIENCE_WORDS:
            continue
        if word.endswith("ies") and len(word) > 5:
            word = word[:-3] + "y"
        elif word.endswith("s") and not word.endswith("ss") and len(word) > 4:
            word = word[:-1]
        tokens.append(word)
    return tokens


def keyword_tokens(terms: list[str]) -> list[str]:
    """Extract 1–3-gram phrases and individual tokens from keyword/topic strings."""
    tokens: list[str] = []
    for term in terms:
        phrase = re.sub(r"\([^)]*\)", "", term).strip()
        pieces = tokenize(phrase)
        if not pieces:
            continue
        if 1 <= len(pieces) <= 3:
            tokens.append(" ".join(pieces))
        tokens.extend(pieces)
    return tokens


if __name__ == "__main__":
    # Sanity: junk words filtered, scientific terms preserved.
    assert tokenize("two qubits") == ["qubit"]
    assert tokenize("pure mathematics") == ["mathematic"]
    assert tokenize("its own collapse due") == []
    assert tokenize("photonic crystal") == ["photonic", "crystal"]
    assert tokenize("ground state quantum") == ["ground", "state", "quantum"]
    assert keyword_tokens(["Quantum Computing"]) == ["quantum computing", "quantum", "computing"]
    print("tokenize.py assertions passed.")
