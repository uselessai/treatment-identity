"""The fixture seed must reach the subject, not stop at the gate.

Every delivery gate builds its own LoaderSpec. Three of them hardcoded
``seed=0``, and a subject re-seeds itself from the spec it is given, so a
campaign that set the subject's generator directly had that assignment
destroyed on the next line. Twenty labelled seeds were twenty repetitions of
seed zero: the CSV changed the label and not the experiment.

These tests fail if that returns.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from treatment_identity import (check_precomputed_input, check_target_transforms,
                                check_temporal_window)
from treatment_identity.adapter import LoaderSpec, Sample


class _SeedRecordingLoader:
    """Records the seed of every spec it is built with, and samples from it."""

    name = "seed recorder"

    def __init__(self):
        self.seeds: list[int] = []

    def build(self, spec: LoaderSpec):
        self.seeds.append(spec.seed)
        self._rng = np.random.default_rng(spec.seed)
        self._gt = sorted(Path(spec.gt_root).rglob("*.png"))
        self._n = spec.num_frames
        return self

    def sample(self, loader, index: int) -> Sample:
        import cv2
        start = int(self._rng.integers(0, max(len(self._gt) - self._n, 1)))
        ids = list(range(start, start + self._n))
        frames = np.asarray([cv2.imread(str(self._gt[i])) for i in ids])
        return Sample(gt=frames, lq=frames, frame_ids=ids)

    def __len__(self) -> int:
        return 8


class SeedPropagationTests(unittest.TestCase):
    def _fixture(self, d: str) -> Path:
        from treatment_identity import fixtures as fx
        root = Path(d)
        fx.make_pair(root, gt_kind="index", lq_kind="index", n_frames=32)
        return root

    def test_every_gate_passes_its_seed_to_the_subject(self) -> None:
        gates = (
            lambda a, w, s: check_precomputed_input(a, w, seed=s),
            lambda a, w, s: check_temporal_window(a, w, seed=s),
            lambda a, w, s: check_target_transforms(
                a, w, __import__("treatment_identity.adapter", fromlist=["x"])
                .CallRecorder(), "sharpen", expected=0, declared=False, seed=s),
        )
        for gate in gates:
            with tempfile.TemporaryDirectory(prefix="ti_seed_") as d:
                root = self._fixture(d)
                loader = _SeedRecordingLoader()
                gate(loader, root, 12345)
                self.assertTrue(loader.seeds, "the gate never built the loader")
                self.assertTrue(
                    all(s == 12345 for s in loader.seeds),
                    f"the gate built the subject with {loader.seeds}, not 12345")

    def test_different_seeds_produce_different_series(self) -> None:
        """A seed that arrives but changes nothing would be just as useless."""
        series = []
        for seed in (0, 1):
            with tempfile.TemporaryDirectory(prefix="ti_seed_") as d:
                root = self._fixture(d)
                loader = _SeedRecordingLoader()
                check_temporal_window(loader, root, seed=seed)
                loader.build(LoaderSpec(gt_root=root / "g2" / "GT",
                                        lq_root=root / "g2" / "LQ",
                                        num_frames=5, seed=seed))
                series.append([tuple(loader.sample(loader, i).frame_ids)
                               for i in range(4)])
        self.assertNotEqual(series[0], series[1],
                            "two different seeds delivered identical windows")


if __name__ == "__main__":
    unittest.main()
