#!/usr/bin/env python
"""Study 4: are the fixed window and the fixed operation order chance?

The certificates observe eight windows and twelve operation orders at one seed.
This repeats both observations with many more draws, under several seeds, and
gives the probability of seeing a single value in every draw if the loader drew
it at random as its code intends: (1/k)^(n-1) for k equally likely values.

    python studies/old_film/repeat_draws.py --repos <clones> --bank <noise_data>
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1])]

from audit_vp_lineage import LINEAGE, _adapter_kwargs        # noqa: E402
from vp_code import OPERATORS, VPCodeAdapter                 # noqa: E402
from treatment_identity.profile import (PROBE_FRAMES,        # noqa: E402
                                        observe_operators, observe_window)

#: The window fixture has this many frames; a window of PROBE_FRAMES frames can
#: start at CLIP - PROBE_FRAMES + 1 positions.
CLIP = 128


def log10_chance(k: int, n: int) -> float:
    """log10 of the probability that n independent uniform draws over k values agree."""
    return -(n - 1) * math.log10(k)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repos", required=True, type=Path)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--draws", type=int, default=100)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", type=Path, default=HERE / "certificates" / "repeat_draws.csv")
    args = ap.parse_args()

    rows = []
    with tempfile.TemporaryDirectory() as td:
        for name, meta in LINEAGE.items():
            ad = VPCodeAdapter(name, args.repos / meta["pristine"], texture_bank=args.bank,
                               channels=meta["channels"], **_adapter_kwargs(meta))
            for seed in range(args.seeds):
                w = observe_window(ad, Path(td) / f"{name}_w{seed}", draws=args.draws, seed=seed)
                o = observe_operators(ad, Path(td) / f"{name}_o{seed}", draws=args.draws, seed=seed)
                rows.append(dict(
                    system=name, seed=seed, draws=args.draws,
                    distinct_windows=w.get("distinct_windows"),
                    distinct_orders=o.get("distinct_orders"),
                    log10_chance_same_window=round(
                        log10_chance(CLIP - PROBE_FRAMES + 1, w.get("draws", 0)), 1),
                    log10_chance_same_order=round(
                        log10_chance(math.factorial(len(OPERATORS)), o.get("draws", 0)), 1)))
                print(rows[-1])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
