"""Hash one configuration section instead of a whole configuration file.

Steps that consume the explicitly-quantum core bind to the core rule that produced it.
Hashing only that section means re-tuning an unrelated setting, such as the outlier
model, does not invalidate an unchanged core set.
"""

import hashlib
import json


def json_digest(value):
    """SHA-256 of a JSON value in a canonical, key-sorted encoding."""
    canonical = json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    assert json_digest({"a": 1, "b": 2}) == json_digest({"b": 2, "a": 1})
    assert json_digest({"a": 1}) != json_digest({"a": 2})
    print("json_digest({'a': 1}) =", json_digest({"a": 1}))
