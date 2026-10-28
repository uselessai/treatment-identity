#!/usr/bin/env python3
"""Verify the complete logical evidence deposited for mechanism 1.6.0."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def rows(name: str) -> list[dict[str, str]]:
    with (HERE / name).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> int:
    manifest = {}
    for line in (HERE / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1)
        manifest[name] = digest
    for name, expected in manifest.items():
        observed = hashlib.sha256((HERE / name).read_bytes()).hexdigest()
        assert observed == expected, f"hash mismatch: {name}"

    summary = json.loads((HERE / "M_summary.json").read_text(encoding="utf-8"))
    assert summary["targeted_pairs"] == summary["targeted_detected"] == 10
    assert summary["expected_misses"] == []
    assert summary["collateral_failures"] == 0
    assert summary["false_alarm_runs"] == 120
    assert summary["false_alarms"] == 0
    assert summary["one_pass_measurements"] == 20

    matrix = rows("M_seeded_defects_matrix.csv")
    targeted = [row for row in matrix if row["is_target"] == "1"]
    assert len(targeted) == 10
    assert all(row["status"] == "FAIL" and row["detected"] == "1"
               for row in targeted)
    assert rows("M_collateral.csv") == []
    false = rows("M_false_alarms.csv")
    assert len(false) == 120
    assert len({row["seed"] for row in false}) == 20
    assert all(row["status"] == "PASS" and row["false_alarm"] == "0"
               for row in false)

    portability = rows("P_portability_matrix.csv")
    effort = rows("P_effort.csv")
    assert len(portability) == 48
    assert len({row["subject"] for row in portability}) == 8
    assert {row["gate"] for row in portability} == {
        "precomputed_input", "temporal_window", "target_transforms",
        "operator_trace", "value_range", "channel_content",
    }
    assert all(row["status"] in {"PASS", "N/A", "SKIP"}
               for row in portability)
    assert len(effort) == 8 and all(row["status"] == "RUN" for row in effort)
    assert all(row["gates_failed"] == "0" and row["gates_undecl"] == "0"
               for row in effort)
    p_summary = json.loads((HERE / "P_summary.json").read_text(encoding="utf-8"))
    assert p_summary["subjects_run"] == p_summary["subjects_registered"] == 8
    assert p_summary["total_failures"] == p_summary["total_undeclared"] == 0
    assert p_summary["gate_rows"] == 48
    assert p_summary["clip_escalation"] == 1

    print(f"OK: {len(manifest)} hashes; M 10/10 with 0/120 alarms; "
          "P 8 complete subjects / 48 logical cells")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
