#!/usr/bin/env python3
"""Campaign S2 --- does DDColor's CutMix switch reach the delivered tensor?

``do_cutmix`` is a treatment switch the subject exposes and ships turned off.
Turning it on declares a different treatment, so ``separability`` asks whether
the delivered tensor changes.  The branch is probabilistic, and that is the
whole difficulty: an identical pair proves nothing if the branch never ran.

The adapter therefore records entry into the branch, and only the draws that
*entered* it carry evidence.  Among those:

  * tensor differs  -> the declared distinction reached the model
  * tensor identical -> UNEXPECTED EQUALITY: the branch executed and delivered
    the same tensor, which is what ``separability`` exists to detect

Both arms read the same fixture at the same index, so the base image is
identical by construction and the only declared difference is the switch.

Run:  python campaigns/campaign_S_ddcolor_cutmix.py --seeds 16
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

from treatment_identity.checks import check_separability   # noqa: E402
from treatment_identity.seeding import seed_all            # noqa: E402
from adapters_ddcolor_lab import DDColorLabAdapter         # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=HERE / "data")
    ap.add_argument("--seeds", type=int, default=16)
    ap.add_argument("--index", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    ad = DDColorLabAdapter()
    rows = []
    with tempfile.TemporaryDirectory() as td:
        meta = ad.write_fixture(Path(td))

        for seed in range(args.seeds):
            seed_all(seed)
            before = ad.branch_entries
            b = ad.deliver(meta, do_cutmix=True, index=args.index)
            entered = ad.branch_entries - before

            seed_all(seed)
            a = ad.deliver(meta, do_cutmix=False, index=args.index)

            res = check_separability(lambda: a, lambda: b, declared="distinct",
                                     label="do_cutmix_false_vs_true")
            delta = res.evidence.get("max_abs_delta_0_255", 0.0)
            if not entered:
                verdict, why = "N/A", "branch not entered at this seed"
            elif res.status == "FAIL":
                verdict, why = "FAIL", "branch entered and delivered an identical tensor"
            else:
                verdict, why = "PASS", "branch entered and the tensor changed"
            rows.append(dict(seed=seed, branch_entered=bool(entered),
                             verdict=verdict, max_abs_delta_0_255=delta,
                             note=why))

    entered = [r for r in rows if r["branch_entered"]]
    fails = [r for r in entered if r["verdict"] == "FAIL"]
    out = args.out / "S2_ddcolor_cutmix.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"environment reconstruction: {DDColorLabAdapter.environment_reconstruction}")
    print(f"\n{args.seeds} matched seeds, index {args.index}")
    print(f"  branch entered:            {len(entered)}/{args.seeds}")
    print(f"  of those, UNEXPECTED EQUALITY: {len(fails)}/{len(entered)}"
          f"{'' if not entered else f'  ({100*len(fails)/len(entered):.0f}%)'}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
