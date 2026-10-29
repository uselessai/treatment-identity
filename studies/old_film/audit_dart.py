#!/usr/bin/env python
"""Study 4: DART, an old-film restoration system from another group, with its
own loader and degradation library, checked against what its own documents state.

DART runs in its own environment, so this script is run with that Python:

    <DART>/.venv/bin/python studies/old_film/audit_dart.py --out <dir>

Requirements and their sources (``documents_dart``):

* frame window: "7 frames" per training clip, sampled over the clip (paper,
  Table 1; ``configs/train.yaml`` ``data.num_frames``);
* values in [-1, 1]: ``configs/train.yaml`` normalisation mean and std 0.5;
* grey content in three channels: "Inputs and targets are grayscale"
  (AbsoluteDegradation README), so equal channels are what is declared;
* target transformations: no inspected document states that the target is
  transformed, so the observed count is reported as undeclared.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[1]), str(HERE.parent / "adapters")]

from adapters_dart import DartAdapter                         # noqa: E402
from treatment_identity import (Certificate, ContentContract,  # noqa: E402
                                StreamContract, check_channel_content,
                                check_target_transforms, check_temporal_window,
                                check_value_range)

GREY = StreamContract(value_range=(-1.0, 1.0), channels=3,
                      require_distinct_channels=False)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=HERE / "certificates" / "released")
    args = ap.parse_args()
    ad = DartAdapter()
    cert = Certificate(
        pipeline="DART",
        expected_treatment=("DART training loader under configs/train.yaml: windows of "
                            "frames sampled over the clip, values in [-1, 1], grey "
                            "content in three channels; target transformation "
                            "unspecified in the inspected documents"),
    ).with_repository(ad.root)
    with tempfile.TemporaryDirectory() as td:
        wd = Path(td)
        cert.add(check_temporal_window(ad, wd / "w"))
        cert.add(check_target_transforms(ad, wd / "t", ad.recorder, "GroundTruthSharpenStep",
                                         expected=0, declared=False))
        content = ContentContract(lq=GREY, gt=GREY)
        cert.add(check_value_range(ad, wd / "r", content))
        cert.add(check_channel_content(ad, wd / "c", content))
    cert.describe(ad)
    # DART's environment has no jsonschema; the document is validated where
    # DDC's own environment is used.
    path = cert.write(args.out / "dart.json", validate=False)
    print(cert.summary())
    print("certificate:", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
