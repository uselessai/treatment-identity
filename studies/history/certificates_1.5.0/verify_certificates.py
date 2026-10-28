#!/usr/bin/env python3
"""Validate regenerated certificates, their hashes and logical continuity."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from treatment_identity import validate_certificate


HERE = Path(__file__).resolve().parent
HISTORICAL = HERE.parent


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def outcomes(document: dict) -> list[tuple[str, str]]:
    return [(gate["gate"], gate["status"]) for gate in document["gates"]]


def main() -> int:
    checked = 0
    for line in (HERE / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        expected, relative = line.split(maxsplit=1)
        candidate_path = HERE / relative
        assert digest(candidate_path) == expected, f"hash mismatch: {relative}"

        candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
        validate_certificate(candidate)
        assert candidate["schema"] == "treatment-certificate/1.2"

        mode, filename = Path(relative).parts
        old_dir = "certificates_released" if mode == "released" else "certificates_working"
        old_path = HISTORICAL / old_dir / filename
        historical = json.loads(old_path.read_text(encoding="utf-8"))
        assert outcomes(candidate) == outcomes(historical), (
            f"logical outcome drift: {relative}"
        )
        checked += 1

    print(f"OK: {checked} schema-1.2 certificates; hashes and outcomes verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
