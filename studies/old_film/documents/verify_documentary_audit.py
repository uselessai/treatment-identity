#!/usr/bin/env python3
"""Verify the frozen UNDECL corpus, search counts and decision linkage."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path


HERE = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest() -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in (HERE / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        expected, relative = line.split(maxsplit=1)
        entries[relative] = expected
    return entries


def count(pattern: str, text: str) -> int:
    return len(re.findall(pattern, text, flags=re.IGNORECASE))


def main() -> int:
    checks = 0
    for relative, expected in manifest().items():
        actual = sha256(HERE / relative)
        assert actual == expected, f"hash mismatch: {relative}"
        checks += 1

    with (HERE / "search_log.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 8
    context_pattern = (
        r"(?:target|ground.?truth|\bgt\b).{0,80}(?:sharpen|unsharp)"
        r"|(?:sharpen|unsharp).{0,80}(?:target|ground.?truth|\bgt\b)"
    )
    for row in rows:
        text = (HERE / row["document"]).read_text(encoding="utf-8")
        observed = {
            "sharpen_hits": count(r"\bsharpen(?:ed|ing|s)?\b", text),
            "unsharp_hits": count(r"\bunsharp(?:ness)?\b", text),
            "target_sharpen_context_hits": count(context_pattern, text),
        }
        for field, value in observed.items():
            assert value == int(row[field]), (
                f"search count drift: {row['document']} {field} "
                f"{value} != {row[field]}"
            )
            checks += 1

    with (HERE / "decisions.csv").open(encoding="utf-8", newline="") as stream:
        decisions = list(csv.DictReader(stream))
    assert {row["system"] for row in decisions} == {"RTN", "RRTN", "MambaOFR", "MgfrOFR"}
    for row in decisions:
        assert row["decision"] == "UNDECL"
        evidence = (HERE / row["executable_evidence"]).resolve()
        document = json.loads(evidence.read_text(encoding="utf-8"))
        gate = next(g for g in document["gates"] if g["gate"] == "target_transforms")
        assert gate["status"] == "UNDECL"
        assert float(gate["evidence"]["calls_per_frame"]) == float(
            row["observed_calls_per_target_frame"]
        )
        checks += 2

    print(f"OK: {checks} documentary-audit checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
