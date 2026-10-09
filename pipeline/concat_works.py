"""Concatenate all three universities' classified-works JSONL into one file for the taxonomy pipeline."""

from __future__ import annotations

import json
from pathlib import Path


def concat_works(paths: list[Path], out_path: Path) -> int:
    n = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as out_f:
        for path in paths:
            with path.open(encoding="utf-8") as in_f:
                for line in in_f:
                    if line.strip():
                        out_f.write(line if line.endswith("\n") else line + "\n")
                        n += 1
    return n


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Concatenate per-university classified-works JSONL files")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    universities = json.loads((root / "universities.json").read_text())
    paths: list[Path] = []
    for key in universities:
        path = root / "data" / key / "quantum_classified_works_specter_reviewed.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Missing works file for {key}: {path}")
        paths.append(path)
    n = concat_works(paths, args.out)
    print(f"Wrote {n} works to {args.out}")


if __name__ == "__main__":
    import sys

    if "--out" in sys.argv:
        main()
    else:
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            a = td_path / "a.jsonl"
            b = td_path / "b.jsonl"
            c = td_path / "c.jsonl"
            a.write_text('{"id": "W1"}\n')
            b.write_text('{"id": "W2"}\n{"id": "W3"}\n')
            c.write_text('{"id": "W4"}\n')
            out = td_path / "merged.jsonl"
            n = concat_works([a, b, c], out)
            assert n == 4, n
            assert len(out.read_text().splitlines()) == 4
        print("concat_works smoke test passed")
