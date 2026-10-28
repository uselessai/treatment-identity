#!/usr/bin/env python
"""Derive fragility results from the frozen cell scores, correctly.

The sweeps write one row per (subject, scale, nominal distortion) with the five
metric values. Everything else --- which comparisons lose their winner, at what
distortion, how often --- is derivation, and three defects were found in how the
sweeps did it inline. This module redoes that derivation from their CSVs, so no
scoring is repeated and the raw measurements stay untouched.

What is corrected:

1. Realized geometry, not nominal. The target height is rounded to whole pixels,
   so the 21 nominal distortions collapse to 13 distinct heights at s=1 and 17 at
   s=2; five nominal values all produce exactly 180 px, which is the d=0 geometry
   under another name. Magnitudes are therefore reported against the realized
   distortion H/(H0 s) - 1, and duplicate geometries are collapsed before the
   smallest one is taken. The smallest realizable step at s=1 is 1/180, about
   0.56%, not the 0.1% the nominal grid suggests.

2. Two verdicts, not one. A comparison can lose its decisive winner without
   acquiring the opposite one. The sweeps skipped tie-band crossings and kept
   looking for a strict inversion, which hid the weaker failure. Both are
   reported: `order_lost` at the first geometry where the winner is no longer
   decisive, and `inverted` at the first geometry where the opposite side wins
   decisively.

3. Denominators. A pair with no decisive winner at d=0 cannot invert, so a flip
   rate over all pairs understates itself. Rates are reported over the pairs
   that have a decisive winner at zero distortion, with the tie count alongside.

    python fragility_derive.py --cells paper/data/geometry_fragility_v2_cells.csv
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import statistics
from collections import defaultdict
from pathlib import Path

METRICS = ("PSNR", "SSIM", "LPIPS", "DISTS", "STLPIPS")
HIGHER = {"PSNR": True, "SSIM": True, "LPIPS": False,
          "DISTS": False, "STLPIPS": False}
TIE = {"PSNR": 0.02, "SSIM": 0.0005, "LPIPS": 0.0005,
       "DISTS": 0.0005, "STLPIPS": 0.0005}
#: The inherited stretch as REALIZED in whole pixels: 368/360 at the
#: evaluator's scale, 184/180 at the reference's. Both are exactly 1/45.
#: Comparing against the nominal 0.0222 excluded the inherited geometry
#: itself by four decimal places.
INHERITED_D = 368.0 / 360.0 - 1.0


def load(path: Path):
    """{subject: {(scale, realized_d): {metric: value}}}, duplicates collapsed.

    Subject identity is whatever the file's leading non-numeric columns encode,
    so the same reader serves the E18, mature, external and budget sweeps.
    """
    rows = list(csv.DictReader(path.open(newline="")))
    if not rows:
        raise SystemExit(f"no rows in {path}")
    numeric = {"scale", "distortion", "target_w", "target_h", *METRICS}
    key_cols = [c for c in rows[0] if c not in numeric]

    # Base height per scale, read from the zero-distortion row. Deriving it from
    # each row instead would just return the nominal distortion, which is the
    # quantity being corrected away from.
    base_h: dict[float, float] = {}
    for r in rows:
        if abs(float(r["distortion"])) < 1e-12:
            base_h[float(r["scale"])] = float(r["target_h"]) / float(r["scale"])
    if not base_h:
        raise SystemExit(f"no zero-distortion row in {path}")

    subjects: dict[tuple, dict] = defaultdict(dict)
    for r in rows:
        subj = tuple(r[c] for c in key_cols)
        s = float(r["scale"])
        realized = float(r["target_h"]) / (base_h[s] * s) - 1.0
        pt = (s, round(realized, 6))
        # A repeated realized geometry is the same measurement; keep one.
        subjects[subj].setdefault(pt, {m: float(r[m]) for m in METRICS})
    return key_cols, dict(subjects)


def classify(A: dict, B: dict, s: float, m: str):
    """(verdict, realized magnitude) for one pair, one scale, one metric."""
    base_pt = (s, 0.0)
    if base_pt not in A or base_pt not in B:
        return "no_zero_point", None
    base = A[base_pt][m] - B[base_pt][m]
    if base == 0.0 or abs(base) < TIE[m]:
        return "tie_at_zero", None
    left_wins = (base > 0) == HIGHER[m]
    points = sorted((p for p in A if p[0] == s and p != base_pt and p in B),
                    key=lambda p: abs(p[1]))
    lost = inverted = None
    for pt in points:
        diff = A[pt][m] - B[pt][m]
        if diff == 0.0 or abs(diff) < TIE[m]:
            if lost is None:
                lost = abs(pt[1])
            continue
        if ((diff > 0) == HIGHER[m]) != left_wins:
            if lost is None:
                lost = abs(pt[1])
            inverted = abs(pt[1])
            break
    if inverted is not None:
        return "inverted", (lost, inverted)
    if lost is not None:
        return "order_lost", (lost, None)
    return "stable", None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--exclude", default="",
                    help="comma-separated subject keys to drop, for a subject "
                         "that turns out to duplicate another")
    ap.add_argument("--group-by", default="",
                    help="comma-separated key columns that must MATCH for two "
                         "subjects to be compared; e.g. 'seed' so that cells are "
                         "only paired within a seed, never across seeds")
    args = ap.parse_args()

    path = Path(args.cells)
    key_cols, subjects = load(path)
    drop = {x for x in args.exclude.split(",") if x}
    names = sorted(n for n in subjects if not (set(n) & drop))
    if drop:
        print(f"excluded subjects containing {sorted(drop)}: "
              f"{len(subjects) - len(names)} dropped")
    scales = sorted({p[0] for v in subjects.values() for p in v})
    group_idx = [key_cols.index(c) for c in args.group_by.split(",") if c]

    def comparable(a: tuple, b: tuple) -> bool:
        return all(a[i] == b[i] for i in group_idx)

    print(f"{len(names)} subjects, scales {scales}")
    for s in scales:
        geoms = sorted({p[1] for v in subjects.values() for p in v if p[0] == s})
        print(f"  s={s:g}: {len(geoms)} realized geometries, "
              f"smallest non-zero |d| = {min(abs(g) for g in geoms if g):.5f}")

    rows = []
    for left, right in itertools.combinations(names, 2):
        if not comparable(left, right):
            continue
        for s in scales:
            for m in METRICS:
                verdict, mags = classify(subjects[left], subjects[right], s, m)
                lost, inv = (mags if mags else (None, None))
                rows.append({
                    "subject_a": "|".join(left), "subject_b": "|".join(right),
                    "scale": f"{s:g}", "metric": m, "verdict": verdict,
                    "realized_d_order_lost": "" if lost is None else f"{lost:.5f}",
                    "realized_d_inverted": "" if inv is None else f"{inv:.5f}",
                })

    out = Path(args.out) if args.out else path.with_name(
        path.name.replace("_cells.csv", "_derived.csv"))
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summary = {"source": str(path), "subjects": len(names), "by_scale_metric": {}}
    print(f"\n{'scale':>6} {'metric':>8} {'decisive':>9} {'ties':>5} "
          f"{'order_lost':>11} {'inverted':>9} {'inv<=2.22%':>11}")
    for s in scales:
        for m in METRICS:
            sub = [r for r in rows if r["scale"] == f"{s:g}" and r["metric"] == m]
            ties = sum(1 for r in sub if r["verdict"] == "tie_at_zero")
            dec = len(sub) - ties
            inv = [r for r in sub if r["verdict"] == "inverted"]
            lost = [r for r in sub if r["verdict"] in ("inverted", "order_lost")]
            low = sum(1 for r in inv
                      if float(r["realized_d_inverted"]) <= INHERITED_D)
            summary["by_scale_metric"][f"s{s:g}_{m}"] = {
                "decisive_at_zero": dec, "ties_at_zero": ties,
                "order_lost_or_inverted": len(lost), "inverted": len(inv),
                "inverted_at_or_below_inherited": low,
                "median_realized_d_inverted": (
                    statistics.median(float(r["realized_d_inverted"]) for r in inv)
                    if inv else None),
            }
            print(f"{s:>6g} {m:>8} {dec:>9} {ties:>5} {len(lost):>11} "
                  f"{len(inv):>9} {low:>11}")
    Path(str(out).replace(".csv", ".json")).write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
