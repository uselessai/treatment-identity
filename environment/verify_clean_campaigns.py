#!/usr/bin/env python3
"""Compare a clean current run with the retained 1.6.0 logical outcomes."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


HERE = Path(__file__).resolve().parent
REFERENCE = HERE.parent / "campaigns" / "data_1.6.0"


def selected(path: Path, keys: tuple[str, ...]) -> list[tuple[str, ...]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return [tuple(row[key] for key in keys) for row in csv.DictReader(stream)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    comparisons = (
        (
            "M_seeded_defects_matrix.csv",
            ("defect", "targeted_gate", "gate", "status", "is_target", "detected"),
        ),
        ("P_portability_matrix.csv", ("subject", "gate", "status")),
    )
    for relative, keys in comparisons:
        assert selected(REFERENCE / relative, keys) == selected(
            args.results / relative, keys), f"logical outcome drift: {relative}"
    print("OK: two clean-run logical matrices match retained 1.6.0 evidence")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
