"""The labelling rubric of the rescue pass: one definition, used by every labeller.

The same three-way judgement ("is this paper quantum research?") was given, word for word,
to the Haiku and Sonnet sub-agents of the audit (docs/validation/) and to Eikos-4B in the
rescue pass, so their answers are comparable. Judgements use the title and abstract only.

    CORE        quantum physics or quantum technology is the central subject
    PERIPHERAL  quantum mechanics is only a tool, or lives inside another field
    NOT         "quantum" is incidental, metaphorical, or absent

The rescue pass admits CORE works only. PERIPHERAL is deliberately not rescued: the audit
showed the rule-excluded pool is full of quantum-chemistry, nanomaterial and collider papers
that the original corpus definition ("quantum is central rather than merely mentioned",
corpus/config/sweden_quantum_paper_filter.json) was written to leave out.
"""

LABELS = ("CORE", "PERIPHERAL", "NOT")

# Option descriptions shown to the model next to each letter.
CRITERIA = {
    "CORE": "central subject is quantum physics or quantum technology",
    "PERIPHERAL": "quantum mechanics only as a tool or incidental to another field",
    "NOT": "not quantum research",
}

INSTRUCTIONS = (
    "Decide from the title and abstract alone whether this paper is quantum research. "
    "CORE: the central subject is quantum physics or quantum technology (quantum computing, information, "
    "communication, cryptography, qubits, quantum sensing or metrology, quantum optics, quantum many-body "
    "or condensed-matter physics, cold atoms, superconducting or trapped-ion devices, quantum field theory or gravity). "
    "PERIPHERAL: quantum mechanics appears only as a computational or incidental tool or in another field "
    "(quantum-chemistry or DFT calculations, quantum dots as nanomaterials for bio or optoelectronic uses, "
    "particle-physics or QCD collider analyses, quantum yield). "
    "NOT: 'quantum' is incidental, metaphorical or absent."
)

QUESTION = {"type": "score", "instructions": INSTRUCTIONS, "criteria": CRITERIA}
