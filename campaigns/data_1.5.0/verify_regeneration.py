#!/usr/bin/env python3
"""Verify RC hashes and compare logical outcomes with archived 1.4.2 data."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


HERE = Path(__file__).resolve().parent
ARCHIVED = HERE.parent / "data"


def sha256(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def selected(path: Path, keys: tuple[str, ...]) -> list[tuple[str, ...]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return [tuple(row[key] for key in keys) for row in csv.DictReader(stream)]


def main() -> int:
    hashes = 0
    for line in (HERE / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        expected, relative = line.split(maxsplit=1)
        assert sha256(HERE / relative) == expected, f"hash mismatch: {relative}"
        hashes += 1

    comparisons = (
        (
            ARCHIVED / "M_seeded_defects_matrix.csv",
            HERE / "default/M_seeded_defects_matrix.csv",
            ("defect", "targeted_gate", "gate", "status", "is_target", "detected"),
        ),
        (
            ARCHIVED / "P_portability_matrix.csv",
            HERE / "default/P_portability_matrix.csv",
            ("subject", "gate", "status"),
        ),
        (
            ARCHIVED / "P_portability_clip_escalation.csv",
            HERE / "escalation/P_portability_clip_escalation.csv",
            ("subject", "gate", "status"),
        ),
    )
    for archived, regenerated, keys in comparisons:
        assert selected(archived, keys) == selected(regenerated, keys), (
            f"logical outcome drift: {regenerated.name}"
        )

    print(f"OK: {hashes} hashes; {len(comparisons)} logical matrices identical")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
