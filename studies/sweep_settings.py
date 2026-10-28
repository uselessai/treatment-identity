#!/usr/bin/env python3
"""Sweep a loader's own settings and report which of them change what it delivers.

``separability`` asks whether two conditions an experiment declares to be
different arrive different, and it needs the declaration: an equality is only a
failure against an intention. Half of that question needs no declaration at all.
Flipping a setting the loader itself exposes and comparing the delivered tensors
is a measurement, and what it returns is useful on its own --- a setting that
changes nothing is worth knowing about whether or not anyone declared it should.

An adapter opts in by exposing ``settings()``: a mapping from the name of an
option to the two values to compare. Everything else is held fixed, including
the seed, so a difference cannot come from the random state.

Run:  python studies/sweep_settings.py
"""
from __future__ import annotations

import argparse
import csv
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE / "adapters"))

from treatment_identity.seeding import seed_all                   # noqa: E402
import campaign_P_portability as camp                             # noqa: E402


def sweep(adapter, workdir: Path, seeds: int) -> list[dict]:
    """One row per exposed setting: does flipping it change the delivered item?

    A subject that raises while a setting is exercised is recorded rather than
    allowed to end the sweep. An option that cannot be turned on without an
    exception is a result about that subject, and losing the remaining settings
    to it would be losing the rest of the description as well.
    """
    rows = []
    for name, (value_a, value_b) in adapter.settings().items():
        changed = entered = total = raised = 0
        changed_when_entered = same_when_entered = 0
        changed_without_entry = same_without_entry = 0
        raised_after_entry = 0
        first_error = ""
        for seed in range(seeds):
            try:
                seed_all(seed)
                a, reached_a = adapter.deliver_under(workdir, name, value_a)
                seed_all(seed)
                b, reached_b = adapter.deliver_under(workdir, name, value_b)
            except Exception as exc:                              # noqa: BLE001
                raised += 1
                if bool(getattr(adapter, "last_branch_reached", False)):
                    raised_after_entry += 1
                first_error = first_error or f"{type(exc).__name__}: {exc}"[:120]
                continue
            total += 1
            reached = bool(reached_a or reached_b)
            differs = not np.array_equal(np.asarray(a), np.asarray(b))
            if reached:
                entered += 1
            if differs:
                changed += 1
            if reached and differs:
                changed_when_entered += 1
            elif reached:
                same_when_entered += 1
            elif differs:
                changed_without_entry += 1
            else:
                same_without_entry += 1
        rows.append(dict(setting=name, value_a=str(value_a), value_b=str(value_b),
                         requested_seeds=seeds, compared=total,
                         delivered_differs=changed, branch_reached=entered,
                         changed_when_reached=changed_when_entered,
                         unchanged_when_reached=same_when_entered,
                         changed_without_reach=changed_without_entry,
                         unchanged_without_reach=same_without_entry,
                         raised=raised, raised_after_reach=raised_after_entry,
                         first_error=first_error))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=HERE / "data")
    ap.add_argument("--seeds", type=int, default=16)
    ap.add_argument("--subject", action="append")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows = []
    for key in (args.subject or list(camp.SUBJECTS)):
        cfg = camp.SUBJECTS[key]
        camp._isolate_shared_package_names(cfg["root"])
        module = __import__(cfg["module"])
        adapter = getattr(module, cfg["cls"])()
        if not hasattr(adapter, "settings"):
            print(f"{key}: exposes no settings to sweep")
            continue
        with tempfile.TemporaryDirectory() as td:
            for r in sweep(adapter, Path(td), args.seeds):
                r["subject"] = key
                rows.append(r)
                if r["compared"] == 0:
                    verdict = f"the subject raised on every draw ({r['first_error']})"
                elif r["delivered_differs"] < r["compared"]:
                    verdict = ("delivers the same item in "
                               f"{r['compared'] - r['delivered_differs']}/"
                               f"{r['compared']} completed pairs")
                else:
                    verdict = "changes what is delivered"
                if r["raised"] and r["compared"]:
                    verdict += f"; the subject raised on {r['raised']} further draws"
                if r["unchanged_when_reached"]:
                    verdict += (f"; {r['unchanged_when_reached']} unchanged "
                                "pair(s) entered the enabled branch")
                print(f"{key}: {r['setting']:12} {r['value_a']} vs {r['value_b']:6} "
                      f"-> {verdict}")

    if rows:
        out = args.out / "P_settings_sweep.csv"
        with out.open("w", newline="") as fh:
            w = csv.DictWriter(
                fh, fieldnames=["subject", "setting", "value_a", "value_b",
                                "requested_seeds", "compared",
                                "delivered_differs", "branch_reached",
                                "changed_when_reached", "unchanged_when_reached",
                                "changed_without_reach", "unchanged_without_reach",
                                "raised", "raised_after_reach", "first_error"])
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
