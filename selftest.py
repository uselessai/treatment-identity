#!/usr/bin/env python
"""Self-test: seven delivery gates plus one evaluation check must discriminate.

A suite that only ever reports FAIL proves nothing. This file defines a minimal
reference loader that honours its contract, and a family of deliberately
defective variants --- one per delivery-divergence class documented in the
paper. The separate geometry check is exercised against faithful and distorted
shapes.

    python selftest.py

Exit status 0 means the suite discriminates.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from treatment_identity import (CheckResult, ContentContract, StreamContract,
                                check_channel_content, check_geometry,
                                check_operator_trace,                         # noqa: E402
                                check_precomputed_input, check_separability,
                                check_target_transforms, check_temporal_window,
                                check_value_range)
from treatment_identity.adapter import CallRecorder, LoaderSpec, Sample  # noqa: E402


class ReferenceLoader:
    """A loader that does everything the protocol asks of it.

    Deliberately tiny: it exists to show that the gates are satisfiable, and to
    document the contract in executable form.
    """

    name = "reference"

    #: which defect to inject, if any
    DEFECTS = ("none", "ignore_precomputed", "prefix_window",
               "double_sharpen", "fixed_order", "value_range",
               "channel_collapse")

    def __init__(self, defect: str = "none", num_frames: int = 5):
        assert defect in self.DEFECTS, defect
        self.defect = defect
        self.num_frames = num_frames
        self.recorder = CallRecorder()
        self._rng = np.random.default_rng(0)

    # -- adapter protocol -------------------------------------------------
    def build(self, spec: LoaderSpec):
        gt_dir = sorted(Path(spec.gt_root).iterdir())[0]
        lq_dir = sorted(Path(spec.lq_root).iterdir())[0] if spec.lq_root else gt_dir
        self._gt = sorted(gt_dir.glob("*.png"))
        self._lq = sorted(lq_dir.glob("*.png"))
        self._spec = spec
        self._rng = np.random.default_rng(spec.seed)
        return self

    def sample(self, loader, index: int) -> Sample:
        n = self.num_frames
        total = len(self._gt)
        if self.defect == "prefix_window":
            ids = list(range(n))                       # the window is discarded
        else:
            start = int(self._rng.integers(0, max(total - n, 1)))
            ids = list(range(start, start + n))

        gt = [cv2.imread(str(self._gt[i])) for i in ids]

        use_pre = self._spec.use_precomputed_lq and self.defect != "ignore_precomputed"
        if use_pre:
            lq = [cv2.imread(str(self._lq[i])) for i in ids]
        else:
            lq = [self._degrade(g) for g in gt]

        gt = [self._sharpen(g) for g in gt]
        if self.defect == "double_sharpen":
            gt = [self._sharpen(g) for g in gt]

        lq_a, gt_a = np.asarray(lq), np.asarray(gt)
        if self.defect == "value_range":
            lq_a = lq_a.astype(np.float32) / 255.0
            gt_a = gt_a.astype(np.float32) / 255.0
        if self.defect == "channel_collapse":
            grey = lq_a.mean(axis=-1, keepdims=True)
            lq_a = np.repeat(grey, 3, axis=-1)
        return Sample(lq=lq_a, gt=gt_a, frame_ids=ids)

    def __len__(self) -> int:
        return len(self._gt)

    # -- the "pipeline" ---------------------------------------------------
    def _sharpen(self, img: np.ndarray) -> np.ndarray:
        self.recorder.record("sharpen")
        return img

    def _degrade(self, img: np.ndarray) -> np.ndarray:
        ops = ["blur", "downsample", "noise", "jpeg"]
        if self.defect != "fixed_order":
            ops = list(self._rng.permutation(ops))
        out = img.astype(np.float32)
        for op in ops:
            self.recorder.record(op)
            if op == "blur":
                out = cv2.GaussianBlur(out, (3, 3), 0.8)
            elif op == "noise":
                out = out + self._rng.normal(0, 3, out.shape).astype(np.float32)
        return np.clip(out, 0, 255).astype(np.uint8)


OPS = {"blur", "downsample", "noise", "jpeg"}
CONTENT_CONTRACT = ContentContract(
    lq=StreamContract(value_range=(0.0, 255.0), require_range_extrema=True,
                      channels=3, require_distinct_channels=True),
    gt=StreamContract(value_range=(0.0, 255.0), require_range_extrema=True,
                      channels=3, require_distinct_channels=True),
)


def run_gates(loader: ReferenceLoader, workdir: Path) -> dict[str, CheckResult]:
    res: dict[str, CheckResult] = {}
    res["precomputed_input"] = check_precomputed_input(loader, workdir)
    res["temporal_window"] = check_temporal_window(loader, workdir)
    res["target_transforms"] = check_target_transforms(
        loader, workdir, loader.recorder, "sharpen", expected=1)
    res["operator_trace"] = check_operator_trace(
        loader, workdir, loader.recorder, OPS, declared_policy="random_permutation")
    res["value_range"] = check_value_range(
        loader, workdir, CONTENT_CONTRACT)
    res["channel_content"] = check_channel_content(
        loader, workdir, CONTENT_CONTRACT)
    return res


#: which gate each injected defect must turn red
TARGETED = {
    "ignore_precomputed": "precomputed_input",
    "prefix_window": "temporal_window",
    "double_sharpen": "target_transforms",
    "fixed_order": "operator_trace",
    "value_range": "value_range",
    "channel_collapse": "channel_content",
}


def main() -> int:
    wd = Path(tempfile.mkdtemp(prefix="ti_selftest_"))
    ok = True

    print("1. reference loader — every exercised delivery gate must PASS")
    base = run_gates(ReferenceLoader("none"), wd / "ref")
    for name, r in base.items():
        good = r.status == "PASS"
        ok &= good
        print(f"   {'OK ' if good else 'BAD'}  {r}")

    print("\n2. injected defects — each must turn its own gate red")
    for defect, gate in TARGETED.items():
        res = run_gates(ReferenceLoader(defect), wd / defect)
        r = res[gate]
        good = r.status == "FAIL"
        ok &= good
        print(f"   {'OK ' if good else 'BAD'}  {defect:20s} -> {gate:18s} {r.status}")
        # the other gates must stay green: a gate that fires on everything is useless
        for other, ro in res.items():
            if other != gate and ro.status == "FAIL":
                ok = False
                print(f"        BAD  collateral failure in {other}: {ro.message[:80]}")

    print("\n3. separability — must fail on unexpected equality, pass on real difference")
    seeds = tuple(range(16))

    def same(seed: int) -> np.ndarray:
        return np.random.default_rng(seed).normal(100, 10, (8, 8, 3))

    r_eq = check_separability(
        same, same, declared="distinct", label="inert_factor",
        seeds=seeds, distributional=True)
    r_ne = check_separability(
        same, lambda seed: same(seed) + 20,
        declared="distinct", label="real_factor",
        seeds=seeds, distributional=True)
    r_id = check_separability(
        same, same, declared="identical", label="declared_same", seeds=seeds)
    for r, want in ((r_eq, "FAIL"), (r_ne, "PASS"), (r_id, "PASS")):
        good = r.status == want
        ok &= good
        print(f"   {'OK ' if good else 'BAD'}  {r.name:26s} {r.status} (wanted {want})")

    print("\n4. geometry — aspect distortion must fail, faithful upscale must pass")
    r_bad = check_geometry((368, 640), (180, 320))
    r_good = check_geometry((360, 640), (180, 320))
    for r, want in ((r_bad, "FAIL"), (r_good, "PASS")):
        good = r.status == want
        ok &= good
        print(f"   {'OK ' if good else 'BAD'}  {r.status} (wanted {want}) — {r.message[:64]}")

    print("\n" + ("SELF-TEST PASSED: the suite discriminates."
                  if ok else "SELF-TEST FAILED."))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
