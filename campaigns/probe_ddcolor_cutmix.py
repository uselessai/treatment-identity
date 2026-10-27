#!/usr/bin/env python3
"""Probe --- DDColor's CutMix axis order, and what it costs.

``lab_dataset.py`` builds ``img_gt`` with ``cv2.resize``, so the array is HWC:
(gt_size, gt_size, 3).  ``rand_bbox`` returns a box in (H, W) coordinates, and
line 103 applies it as::

    img_gt[:, bbx1:bbx2, bby1:bby2] = cmix_img[:, bbx1:bbx2, bby1:bby2]

which leaves axis 0 (rows) untouched, slices axis 1 (columns) with an
x-coordinate, and slices axis 2 --- the three colour channels --- with a
y-coordinate in [0, 256].  Whenever ``bby1 >= 3`` the channel slice is empty
and the augmentation writes nothing at all.

This probe transcribes ``rand_bbox`` and the indexing line and measures the
consequence.  It does not execute the subject's dataset class: that path also
calls ``np.int``, removed in NumPy 1.24, so it cannot run as published on a
current environment at all.

Both effects are dormant under the shipped configuration
(``options/train/train_ddcolor.yml`` sets ``do_cutmix: False``); they describe
what the released code does when the flag it exposes is switched on.
"""
from __future__ import annotations

import argparse

import numpy as np

GT_SIZE = 256
SOURCE = "zDDColor/basicsr/data/lab_dataset.py:92-104,137-158"


def rand_bbox(size, lam):
    """Transcribed from lab_dataset.py:137-158.

    ``np.int`` in the original is replaced by ``int`` here purely so the
    function runs; that substitution is itself one of the findings.
    """
    W, H = size[0], size[1]
    cut_rat = np.sqrt(1. - lam)
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    cx, cy = np.random.randint(W), np.random.randint(H)
    return (np.clip(cx - cut_w // 2, 0, W), np.clip(cy - cut_h // 2, 0, H),
            np.clip(cx + cut_w // 2, 0, W), np.clip(cy + cut_h // 2, 0, H))


def trial(as_released: bool) -> int:
    """One CutMix draw; returns how many tensor entries changed."""
    img = np.zeros((GT_SIZE, GT_SIZE, 3), np.float32)
    cmix = np.ones((GT_SIZE, GT_SIZE, 3), np.float32)
    lam = np.clip(np.random.beta(1.0, 1.0), 0.3, 0.4)     # alpha=1.0, their config
    bbx1, bby1, bbx2, bby2 = rand_bbox(cmix.shape[:2], lam)
    if as_released:
        img[:, bbx1:bbx2, bby1:bby2] = cmix[:, bbx1:bbx2, bby1:bby2]
    else:
        img[bbx1:bbx2, bby1:bby2, :] = cmix[bbx1:bbx2, bby1:bby2, :]
    return int((img != 0).sum())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--draws", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    total = GT_SIZE * GT_SIZE * 3
    print(f"source: {SOURCE}")
    print(f"{args.draws} draws on a {GT_SIZE}x{GT_SIZE}x3 HWC array\n")
    for as_released, label in ((True, "as released"), (False, "spatial axes")):
        np.random.seed(args.seed)
        changed = [trial(as_released) for _ in range(args.draws)]
        noop = sum(1 for c in changed if c == 0)
        fired = [c for c in changed if c > 0]
        print(f"{label:14} no-op {noop/args.draws:7.2%}   "
              f"tensor replaced when it fires "
              f"{(np.mean(fired) / total if fired else 0):7.2%}")
    print("\nintended cut ratio, lam in [0.3, 0.4]: the nominal replaced area is 60-70% before clipping.\n"
          "Boundary clipping reduces realised area; these are box-rule draws, not full-loader runs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
