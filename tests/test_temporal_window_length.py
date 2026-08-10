"""Regression tests for the window-length check of ``check_temporal_window``.

Lesson L6 of the article describing this package reports a gate that granted a
pass it had not earned: a single-image loader returning one frame where the
contract asked for five was accepted, because the gate asserted *which* frames
arrived and never *how many*. Eight one-frame samples are eight distinct
windows, so the coverage figure came out fine and nothing raised.

These tests exist so that it cannot come back, and so that the two repaired
outcomes stay distinguishable: a contradicted declaration is FAIL, and a
contract that declares no window at all is N/A rather than a pass.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from treatment_identity import check_temporal_window
from treatment_identity.adapter import LoaderSpec, Sample


class _SingleFrameLoader:
    """Returns one frame however many the contract asks for."""

    name = "single-frame subject"

    def __init__(self, frames_returned: int = 1):
        self.frames_returned = frames_returned

    def build(self, spec: LoaderSpec):
        self._gt = sorted((Path(spec.gt_root)).rglob("*.png"))
        return self

    def sample(self, loader, index: int) -> Sample:
        import cv2
        ids = [index % max(len(self._gt), 1)] * self.frames_returned
        frames = np.asarray([cv2.imread(str(self._gt[i])) for i in ids])
        return Sample(gt=frames, lq=frames, frame_ids=ids)

    def __len__(self) -> int:
        return 8


class TemporalWindowLengthTests(unittest.TestCase):
    def _run(self, declared: int, returned: int):
        from treatment_identity import fixtures as fx
        with tempfile.TemporaryDirectory(prefix="ti_l6_") as d:
            root = Path(d)
            fx.make_pair(root / "g2", gt_kind="index", lq_kind="index",
                         n_frames=32)
            return check_temporal_window(_SingleFrameLoader(returned), root,
                                         num_frames=declared)

    def test_a_contradicted_window_length_fails(self) -> None:
        """Five declared, one delivered: a declaration is contradicted."""
        res = self._run(declared=5, returned=1)
        self.assertEqual(res.status, "FAIL", res.message)
        self.assertIn("delivered 1", res.message)
        self.assertEqual(res.evidence["declared_num_frames"], 5)

    def test_no_declared_window_is_not_a_pass(self) -> None:
        """One declared: nothing to assert, and that is N/A, not PASS."""
        res = self._run(declared=1, returned=1)
        self.assertEqual(res.status, "N/A", res.message)
        self.assertNotEqual(res.status, "PASS")


if __name__ == "__main__":
    unittest.main()
