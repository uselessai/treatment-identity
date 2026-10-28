#!/usr/bin/env python3
"""Validate and cross-check the eight field certificates for release 1.6.0."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from treatment_identity import validate_certificate  # noqa: E402


EXPECTED = {
    "precomputed_input": {"released": "N/A"},
    "value_range": {"released": "PASS"},
    "channel_content": {"released": "PASS"},
    "geometry": {"released": "FAIL", "working": "FAIL"},
}


def main() -> int:
    manifest = {}
    for line in (HERE / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1)
        manifest[name] = digest
    for name, expected in manifest.items():
        path = HERE / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, name

    documents = []
    for scope in ("released", "working"):
        for path in sorted((HERE / scope).glob("*.json")):
            document = json.loads(path.read_text(encoding="utf-8"))
            validate_certificate(document)
            assert document["schema"] == "treatment-certificate/1.2"
            gates = {gate["gate"]: gate["status"] for gate in document["gates"]}
            assert len(gates) == 10
            assert sum(name.startswith("separability[") for name in gates) == 3
            for gate, by_scope in EXPECTED.items():
                if scope in by_scope:
                    assert gates[gate] == by_scope[scope], (path, gate, gates[gate])
            documents.append(document)
    assert len(documents) == 8
    assert any(document["pipeline"] == "MgfrOFR" for document in documents)
    print("OK: 8 schema-1.2 certificates; 80 gate records; MgfrOFR present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
