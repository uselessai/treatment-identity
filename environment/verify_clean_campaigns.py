#!/usr/bin/env python3
"""Compare a clean-environment campaign run with retained RC outcomes."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


HERE = Path(__file__).resolve().parent
REFERENCE = HERE.parent / "campaigns" / "data_1.5.0"


def selected(path: Path, keys: tuple[str, ...]) -> list[tuple[str, ...]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return [tuple(row[key] for key in keys) for row in csv.DictReader(stream)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    args = parser.parse_args()

    comparisons = (
        (
            "default/M_seeded_defects_matrix.csv",
            ("defect", "targeted_gate", "gate", "status", "is_target", "detected"),
        ),
        ("default/P_portability_matrix.csv", ("subject", "gate", "status")),
        (
            "escalation/P_portability_clip_escalation.csv",
            ("subject", "gate", "status"),
        ),
    )
    for relative, keys in comparisons:
        reference = REFERENCE / relative
        observed = args.results / relative
        assert selected(reference, keys) == selected(observed, keys), (
            f"logical outcome drift: {relative}"
        )

    print(f"OK: {len(comparisons)} clean-run logical matrices identical to RC")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
