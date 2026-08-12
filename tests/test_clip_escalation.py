"""Declared and manufactured source-clip lengths remain separate claims."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from treatment_identity import check_precomputed_input, with_clip_escalation
from treatment_identity.adapter import LoaderSpec, Sample


class _NeedsThirtyTwoFrames:
    name = "minimum-length fixture loader"

    def __init__(self):
        self.specs: list[tuple[int | None, int | None]] = []

    def build(self, spec: LoaderSpec):
        self.specs.append((spec.clip_length, spec.fixture_clip_length))
        if (spec.fixture_clip_length or 0) < 32:
            raise FileNotFoundError("subject indexes frame 31")
        self.lq = sorted(Path(spec.lq_root).rglob("*.png"))
        self.gt = sorted(Path(spec.gt_root).rglob("*.png"))
        return self

    def sample(self, loader, index: int) -> Sample:
        lq = cv2.imread(str(self.lq[index]))
        gt = cv2.imread(str(self.gt[index]))
        return Sample(lq=np.asarray([lq]), gt=np.asarray([gt]))


class ClipEscalationTests(unittest.TestCase):
    def test_undeclared_minimum_is_found_without_rewriting_the_declaration(self) -> None:
        loader = _NeedsThirtyTwoFrames()
        with tempfile.TemporaryDirectory(prefix="ti_clip_") as directory:
            root = Path(directory)

            def run(length: int):
                return check_precomputed_input(
                    loader, root, clip_length=length,
                    declared_clip_length=None)

            result = with_clip_escalation(
                run, gate="precomputed_input", declared=None, default=16)
        self.assertEqual(result.status, "UNDECL", result.message)
        self.assertEqual(result.evidence["clip_length_used"], 32)
        self.assertEqual(loader.specs[0], (None, 16))
        self.assertEqual(loader.specs[-1], (None, 32))

    def test_declared_minimum_is_used_directly(self) -> None:
        loader = _NeedsThirtyTwoFrames()
        with tempfile.TemporaryDirectory(prefix="ti_clip_declared_") as directory:
            root = Path(directory)

            def run(length: int):
                return check_precomputed_input(
                    loader, root, clip_length=length,
                    declared_clip_length=32)

            result = with_clip_escalation(
                run, gate="precomputed_input", declared=32, default=16)
        self.assertEqual(result.status, "PASS", result.message)
        self.assertTrue(all(spec == (32, 32) for spec in loader.specs))

    def test_underlying_contradiction_dominates_undeclared_length(self) -> None:
        class WrongBranch(_NeedsThirtyTwoFrames):
            def sample(self, loader, index: int) -> Sample:
                gt = cv2.imread(str(self.gt[index]))
                return Sample(lq=np.asarray([gt]), gt=np.asarray([gt]))

        loader = WrongBranch()
        with tempfile.TemporaryDirectory(prefix="ti_clip_fail_") as directory:
            root = Path(directory)

            def run(length: int):
                return check_precomputed_input(
                    loader, root, clip_length=length,
                    declared_clip_length=None)

            result = with_clip_escalation(
                run, gate="precomputed_input", declared=None, default=16)
        self.assertEqual(result.status, "FAIL", result.message)
        self.assertEqual(result.evidence["underlying_status"], "FAIL")
        self.assertIn("undeclared minimum", result.message)


if __name__ == "__main__":
    unittest.main()
