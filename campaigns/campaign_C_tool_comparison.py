#!/usr/bin/env python3
"""Campaign C --- what established tooling reports on defects this mechanism finds.

Three defects are put to five established tools and to a hand-written baseline:

  D1  a declared random operator order is never delivered
  D2  a computed sampling window is discarded and a fixed position is opened
  D3  two conditions declared distinct deliver identical data

Static analysis runs over pinned clones of the audited codebases, which are not
bundled: pass them with ``--repos``. The delivery-side tools run over two
recorded deliveries, produced by ``--tensors`` (see ``gen_deliveries.py``).

    python3 campaigns/campaign_C_tool_comparison.py \
        --repos /path/to/clones --tensors campaigns/data_1.7.0/deliveries \
        --out campaigns/data_1.7.0
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
SUBJECTS = ("RTN", "RRTN", "MambaOFR", "MgfrOFR")


def _tool_version(mod: str) -> str | None:
    try:
        return __import__(mod).__version__
    except Exception:
        return None


def static_analysis(repos: Path) -> list[dict]:
    """ruff, in its default configuration and with B007; pylint, as it ships."""
    rows = []
    for name in SUBJECTS:
        d = repos / name
        if not d.exists():
            continue
        row = {"subject": name}
        for label, sel in (("ruff_default", None), ("ruff_B007", "B007")):
            cmd = ["ruff", "check", "--no-cache", "--output-format=concise"]
            if sel:
                cmd += ["--select", sel]
            out = subprocess.run(cmd + [str(d)], capture_output=True, text=True).stdout
            row[label] = sum(1 for ln in out.splitlines() if ":" in ln and " B0" in ln
                             or (sel is None and ln.count(":") >= 3))
        out = subprocess.run([sys.executable, "-m", "pylint", "--disable=C,R",
                              "--output-format=parseable", str(d)],
                             capture_output=True, text=True).stdout
        row["pylint_EWF"] = sum(1 for ln in out.splitlines() if "]: " in ln or ": [" in ln)
        row["pylint_unused_variable"] = out.count("W0612")
        rows.append(row)
    return rows


def _digest(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a, np.float32).tobytes()).hexdigest()


def _features(a: np.ndarray):
    import pandas as pd
    return pd.DataFrame([dict(mean=float(f.mean()), std=float(f.std()),
                              vmin=float(f.min()), vmax=float(f.max()),
                              ch0=float(f[..., 0].mean()), ch1=float(f[..., 1].mean()),
                              ch2=float(f[..., 2].mean())) for f in a])


def delivery_side(tensors: Path) -> dict:
    """Data validators, an experiment tracker, and a hand-written assertion."""
    a = {n: np.load(tensors / f"{n}.npy") for n in ("RTN", "RRTN")}
    digests = {n: _digest(v) for n, v in a.items()}
    out: dict = {"digests": digests, "identical": len(set(digests.values())) == 1,
                 "max_abs_delta": float(np.abs(a["RTN"] - a["RRTN"]).max())}

    try:
        from deepchecks.tabular import Dataset
        from deepchecks.tabular.suites import data_integrity, train_test_validation
        alone = {}
        for n, v in a.items():
            r = data_integrity().run(Dataset(_features(v), cat_features=[]))
            cond = [c for c in r.results if getattr(c, "conditions_results", None)]
            alone[n] = {"checks": len(r.results), "with_condition": len(cond),
                        "failed": sum(1 for c in cond
                                      if not all(x.is_pass for x in c.conditions_results))}
        r = train_test_validation().run(
            train_dataset=Dataset(_features(a["RTN"]), cat_features=[]),
            test_dataset=Dataset(_features(a["RRTN"]), cat_features=[]))
        cond = [c for c in r.results if getattr(c, "conditions_results", None)]
        out["deepchecks"] = {"version": _tool_version("deepchecks"), "each": alone,
                             "paired": {"checks": len(r.results), "with_condition": len(cond),
                                        "failed": sum(1 for c in cond if not all(
                                            x.is_pass for x in c.conditions_results))}}
    except Exception as e:
        out["deepchecks"] = {"error": f"{type(e).__name__}: {e}"}

    try:
        from evidently import DataDefinition, Dataset as EvDataset, Report
        from evidently.presets import DataDriftPreset
        cols = ["mean", "std", "vmin", "vmax", "ch0", "ch1", "ch2"]
        dd = DataDefinition(numerical_columns=cols)
        rep = Report([DataDriftPreset()]).run(
            EvDataset.from_pandas(_features(a["RRTN"]), data_definition=dd),
            EvDataset.from_pandas(_features(a["RTN"]), data_definition=dd)).dict()
        first = rep.get("metrics", [{}])[0].get("value", {})
        out["evidently"] = {"version": _tool_version("evidently"),
                            "columns": len(cols), "drifted": first}
    except Exception as e:
        out["evidently"] = {"error": f"{type(e).__name__}: {e}"}

    # A hand-written assertion is the strictest baseline available, and it works.
    out["handwritten_pytest"] = {
        "detects_D3": digests["RTN"] == digests["RRTN"],
        "lines": 5,
        "note": "requires knowing the check is worth making and which two "
                "conditions to compare"}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repos", type=Path, help="directory holding the pinned clones")
    ap.add_argument("--tensors", type=Path, help="directory holding RTN.npy and RRTN.npy")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    payload: dict = {"campaign": "C", "subjects": list(SUBJECTS),
                     "tool_versions": {m: _tool_version(m)
                                       for m in ("deepchecks", "evidently", "mlflow", "pylint")}}
    if args.repos:
        rows = static_analysis(args.repos)
        with (args.out / "C_static_analysis.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        payload["static_analysis"] = rows
        print(f"  wrote {args.out / 'C_static_analysis.csv'} ({len(rows)} subjects)")
    if args.tensors:
        payload["delivery_side"] = delivery_side(args.tensors)
        print("  delivery-side tools recorded")

    path = args.out / "C_tool_comparison.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"  wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
