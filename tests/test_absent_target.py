"""Regression tests for the absent-target branch of ``check_target_transforms``.

The article describing this package reports a second gate that granted a pass
it had not earned, and this one survived into a published table before it was
found. Zero-DCE is zero-reference: it trains on low-light inputs with no ground
truth anywhere in its interface. ``Sample.gt`` was mandatory, so the only way to
adapt it was to report its single delivered tensor as both input and target.
``target_transforms`` then compared zero observed transformations of that
fabricated target against a declared zero, agreed, and returned ``PASS`` --- on
the one subject in the study chosen precisely because it has no target.

The gate was not wrong about what it observed. It was wrong about what the
observation was worth, which is what happens when declaredness is asked before
applicability. These tests pin the repaired ordering and all three outcomes it
now distinguishes:

* the arm declares no target                     -> ``N/A``
* the arm declares one and the loader returns none -> ``N/A``
* the arm declares one and the loader returns one  -> the gate still asserts

The third matters as much as the first two: a repair that made the gate answer
``N/A`` more readily would have replaced an unearned pass with an unearned
silence.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from treatment_identity import check_target_transforms
from treatment_identity.adapter import CallRecorder, LoaderSpec, Sample


class _Loader:
    """A minimal paired loader; ``target=False`` makes it zero-reference."""

    name = "absent-target subject"

    def __init__(self, target: bool = True, sharpen_calls: int = 0):
        self.target = target
        self.sharpen_calls = sharpen_calls

    def build(self, spec: LoaderSpec):
        self._gt = sorted(Path(spec.gt_root).rglob("*.png"))
        return self

    def sample(self, loader, index: int) -> Sample:
        import cv2
        frames = np.asarray([cv2.imread(str(self._gt[index % len(self._gt)]))])
        return Sample(gt=frames if self.target else None, lq=frames)

    def __len__(self) -> int:
        return 8


class AbsentTargetTests(unittest.TestCase):
    def _run(self, loader, *, has_target: bool, expected: int = 0,
             declared: bool = False):
        rec = CallRecorder()
        with tempfile.TemporaryDirectory(prefix="ti_notarget_") as d:
            return check_target_transforms(
                loader, Path(d), rec, "sharpen", expected=expected,
                declared=declared, has_target=has_target)

    def test_a_declared_zero_reference_arm_is_not_applicable(self) -> None:
        """No target declared: the gate asserts nothing, and that is not a pass."""
        res = self._run(_Loader(target=False), has_target=False)
        self.assertEqual(res.status, "N/A", res.message)
        self.assertNotEqual(res.status, "PASS")
        self.assertIn("zero-reference", res.message)

    def test_a_missing_target_is_not_applicable_either(self) -> None:
        """Contract claims a target, loader delivers none: N/A, not FAIL.

        Nothing the arm declared has been contradicted --- the gate simply has
        nothing to count transformations of.
        """
        res = self._run(_Loader(target=False), has_target=True)
        self.assertEqual(res.status, "N/A", res.message)
        self.assertNotIn(res.status, ("PASS", "FAIL"))

    def test_a_real_target_is_still_asserted_over(self) -> None:
        """The repair must not buy N/A everywhere: a real target still passes."""
        res = self._run(_Loader(target=True), has_target=True)
        self.assertEqual(res.status, "PASS", res.message)
        self.assertEqual(res.evidence["expected_per_frame"], 0)

    def test_the_sample_type_can_say_there_is_no_target(self) -> None:
        """The type-level half of the repair, which is what made it possible."""
        s = Sample(lq=np.zeros((1, 4, 4, 3)), gt=None)
        self.assertFalse(s.has_target)
        lq, gt = s.as_arrays()
        self.assertIsNone(gt)
        self.assertEqual(lq.shape, (1, 4, 4, 3))


if __name__ == "__main__":
    unittest.main()
