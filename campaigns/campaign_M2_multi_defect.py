#!/usr/bin/env python3
"""Campaign M2 --- do the gates still work when more than one thing is broken?

Campaign M seeds one defect at a time and asks whether the gate that covers it
turns red and the others stay green. That establishes each gate reads its own
property and no other, but it says nothing about a pipeline with two problems,
which is the ordinary case: a loader that ignores its window is not thereby
guaranteed to get its colour channels right.

This campaign seeds defects in *pairs*. For every pair whose two defects are
covered by different gates it asks two things:

  * do **both** covering gates turn red, or does one problem hide the other?
  * do the gates covering neither turn red?

Pairs whose defects target the same gate are skipped, and so are pairs that
contradict each other at the source --- applying a transformation twice and not
at all cannot both hold.

Run:  python campaigns/campaign_M2_multi_defect.py [--seeds 5]
"""
from __future__ import annotations

import argparse
import csv
import itertools
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import campaign_M_seeded_defects as m                            # noqa: E402


class MultiDefectLoader(m.SubjectLoader):
    """The reference loader with more than one defect injected at once.

    The base class applies each defect through an independent branch keyed on
    ``self.defect``. Making that attribute compare equal to any member of a set
    turns those branches on together without touching the branches themselves,
    so what each defect does is exactly what Campaign M seeds.
    """

    class _AnyOf(str):
        def __new__(cls, members):
            obj = super().__new__(cls, "+".join(sorted(members)))
            obj.members = frozenset(members)
            return obj

        def __eq__(self, other):
            return other in self.members

        def __ne__(self, other):
            return not self.__eq__(other)

        def __hash__(self):
            return str.__hash__(self)

    def __init__(self, defects, num_frames: int = 5):
        super().__init__(defect="none", num_frames=num_frames)
        self.defect = self._AnyOf(defects)


#: Pairs that cannot both hold in one loader: one undoes the other.
CONTRADICTORY = {frozenset({"double_sharpen", "no_sharpen"}),
                 frozenset({"prefix_window", "frozen_window"})}


def pairs():
    targeted = {d: t for d, (t, _) in m.DEFECTS.items() if t}
    for a, b in itertools.combinations(sorted(targeted), 2):
        if targeted[a] == targeted[b] or frozenset({a, b}) in CONTRADICTORY:
            continue
        yield a, b, targeted[a], targeted[b]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=HERE / "data")
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows = []
    with tempfile.TemporaryDirectory() as td:
        for a, b, ga, gb in pairs():
            both, spurious = 0, 0
            for seed in range(args.seeds):
                loader = MultiDefectLoader({a, b})
                wd = Path(td) / f"{a}_{b}_{seed}"
                wd.mkdir(parents=True, exist_ok=True)
                res = m.run_delivery_gates(loader, wd, seed=seed)
                status = {g: r.status for g, (r, _) in res.items()}
                if status.get(ga) == "FAIL" and status.get(gb) == "FAIL":
                    both += 1
                spurious += sum(1 for g, st in status.items()
                                if st == "FAIL" and g not in (ga, gb))
            rows.append(dict(defect_a=a, defect_b=b, gate_a=ga, gate_b=gb,
                             seeds=args.seeds, both_detected=both,
                             spurious_failures=spurious))
            print(f"{a:20} + {b:20} both red {both}/{args.seeds}"
                  f"   others red {spurious}")

    out = args.out / "M2_multi_defect.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    todos = sum(r["both_detected"] for r in rows)
    posibles = sum(r["seeds"] for r in rows)
    print(f"\n{len(rows)} pairs; both defects detected in {todos}/{posibles} runs; "
          f"{sum(r['spurious_failures'] for r in rows)} failures outside the two")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
