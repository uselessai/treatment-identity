#!/usr/bin/env python3
"""Verify the logical invariants of the 1.7.0 evidence.

Run from the repository root:

    python3 campaigns/data_1.7.0/verify_evidence.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FAIL: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"{'OK  ' if ok else 'FAIL'}  {label}" + (f" --- {detail}" if detail and not ok else ""))
    if not ok:
        FAIL.append(label)


def main() -> int:
    rows = list(csv.DictReader((HERE / "C_static_analysis.csv").open()))
    check("C: four subjects analysed statically", len(rows) == 4, str(len(rows)))
    ruff_default = sum(int(r["ruff_default"]) for r in rows)
    unused = sum(int(r["pylint_unused_variable"]) for r in rows)
    pylint_all = sum(int(r["pylint_EWF"]) for r in rows)
    check("C: ruff default reports 447 diagnostics", ruff_default == 447, str(ruff_default))
    check("C: 139 unused-variable messages", unused == 139, str(unused))
    check("C: unused-variable is a small share of pylint output",
          unused < pylint_all / 10, f"{unused}/{pylint_all}")

    c = json.loads((HERE / "C_tool_comparison.json").read_text())
    d = c["delivery_side"]
    check("C: the two deliveries are identical", d["identical"] and d["max_abs_delta"] == 0.0)
    check("C: deepchecks reports no failed condition, paired",
          d["deepchecks"]["paired"]["failed"] == 0)
    check("C: deepchecks reports no failed condition, each alone",
          all(v["failed"] == 0 for v in d["deepchecks"]["each"].values()))
    check("C: evidently reports no drifted column",
          d["evidently"]["drifted"]["count"] == 0.0)
    check("C: a hand-written assertion does detect the collision",
          d["handwritten_pytest"]["detects_D3"] is True)

    t = json.loads((HERE / "C_experiment_tracking.json").read_text())
    check("C: tracking records different configurations for identical data",
          t["every_recorded_field_differs"] and t["delivered_data_identical"])

    m = json.loads((HERE / "T_modality.json").read_text())
    g = m["gates"]
    check("T: separability reports the seeded collision", g["separability"]["status"] == "FAIL")
    check("T: the operator order is fixed across 12 draws",
          g["operator_trace"]["status"] == "FAIL"
          and g["operator_trace"]["evidence"]["distinct_orders"] == 1)
    check("T: the content predicates pass on the delivered window",
          g["value_range"]["status"] == "PASS" and g["channel_content"]["status"] == "PASS")
    check("T: three predicates ported unmodified", m["summary"]["ported_unmodified"] == 3)
    check("T: two gates remain modality-bound", m["summary"]["modality_bound"] == 2)

    print()
    if FAIL:
        print(f"{len(FAIL)} check(s) failed")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
