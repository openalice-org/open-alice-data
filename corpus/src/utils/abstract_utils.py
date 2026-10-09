def reconstruct_abstract(abstract_inverted_index):
    """Rebuild a plain string from an OpenAlex abstract_inverted_index."""
    if not abstract_inverted_index:
        return ""
    words = {}
    for token, positions in abstract_inverted_index.items():
        for pos in positions:
            words[pos] = token
    return " ".join(words[i] for i in sorted(words))


if __name__ == "__main__":
    idx = {"quantum": [0, 4], "computing": [1], "is": [2], "hard": [3]}
    assert reconstruct_abstract(idx) == "quantum computing is hard quantum"
    print("Example 1:", reconstruct_abstract(idx))

    assert reconstruct_abstract(None) == ""
    assert reconstruct_abstract({}) == ""
    print("Example 2 (empty):", repr(reconstruct_abstract(None)))

    print("All checks passed.")
