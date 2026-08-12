"""Content-contract gates at loader output and immediately before the model."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from treatment_identity import (
    Certificate,
    ContentContract,
    LoaderSpec,
    Sample,
    StreamContract,
    TrainingStepGuard,
    TreatmentContractViolation,
    check_channel_content,
    check_sample_channel_content,
    check_sample_value_range,
    check_value_range,
)


def _contract(*, lo: float = 0.0, hi: float = 255.0) -> ContentContract:
    stream = StreamContract(
        value_range=(lo, hi),
        require_range_extrema=True,
        channels=3,
        require_distinct_channels=True,
    )
    return ContentContract(lq=stream, gt=stream)


class _ImageLoader:
    name = "content fixture loader"

    def __init__(self, transform=lambda value: value):
        self.transform = transform

    def build(self, spec: LoaderSpec):
        self.lq = sorted(Path(spec.lq_root).rglob("*.png"))
        self.gt = sorted(Path(spec.gt_root).rglob("*.png"))
        return self

    def sample(self, loader, index: int) -> Sample:
        lq = cv2.imread(str(self.lq[index]))[..., ::-1].copy()
        gt = cv2.imread(str(self.gt[index]))[..., ::-1].copy()
        return Sample(lq=self.transform(lq)[None],
                      gt=self.transform(gt)[None])


class ContentGateTests(unittest.TestCase):
    def test_signature_satisfies_range_and_channel_contracts(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ti_content_") as directory:
            loader = _ImageLoader()
            value = check_value_range(loader, Path(directory), _contract())
            channels = check_channel_content(loader, Path(directory), _contract())
        self.assertEqual(value.status, "PASS", value.message)
        self.assertEqual(channels.status, "PASS", channels.message)

    def test_silent_rescaling_is_rejected_by_fixture_anchors(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ti_range_") as directory:
            result = check_value_range(
                _ImageLoader(lambda value: value.astype(np.float32) / 255.0),
                Path(directory), _contract())
        self.assertEqual(result.status, "FAIL", result.message)
        self.assertIn("do not realise declared extrema", result.message)

    def test_colour_to_luminance_replication_is_rejected(self) -> None:
        def collapse(value: np.ndarray) -> np.ndarray:
            grey = np.mean(value, axis=-1, keepdims=True)
            return np.repeat(grey, 3, axis=-1)

        with tempfile.TemporaryDirectory(prefix="ti_channels_") as directory:
            result = check_channel_content(
                _ImageLoader(collapse), Path(directory), _contract())
        self.assertEqual(result.status, "FAIL", result.message)
        self.assertIn("collapsed", result.message)

    def test_nonfinite_values_are_rejected(self) -> None:
        sample = Sample(lq=np.full((1, 4, 4, 3), np.nan), gt=None)
        contract = ContentContract(
            lq=StreamContract(value_range=(0.0, 1.0), channels=3))
        result = check_sample_value_range(sample, contract)
        self.assertEqual(result.status, "FAIL")
        self.assertIn("NaN or infinity", result.message)

    def test_intended_grayscale_is_an_explicit_one_channel_contract(self) -> None:
        sample = Sample(lq=np.zeros((1, 4, 4, 1)), gt=None)
        contract = ContentContract(lq=StreamContract(channels=1))
        result = check_sample_channel_content(sample, contract)
        self.assertEqual(result.status, "PASS", result.message)


class TrainingBoundaryTests(unittest.TestCase):
    def test_corruption_after_loader_is_blocked_before_model_call(self) -> None:
        called = False
        contract = ContentContract(
            lq=StreamContract(value_range=(0.0, 1.0), channels=3,
                              require_distinct_channels=True))
        guard = TrainingStepGuard(contract)

        def model(batch: Sample) -> str:
            nonlocal called
            called = True
            return "ran"

        checked = guard.wrap(model)
        grey = np.full((1, 4, 4, 3), 0.5, np.float32)
        with self.assertRaises(TreatmentContractViolation):
            checked(Sample(lq=grey, gt=None))
        self.assertFalse(called)

    def test_valid_batch_runs_and_records_boundary_evidence(self) -> None:
        cert = Certificate("fixture", "RGB values in [0, 1]")
        contract = ContentContract(
            lq=StreamContract(value_range=(0.0, 1.0), channels=3,
                              require_distinct_channels=True))
        guard = TrainingStepGuard(contract, certificate=cert)
        batch = {"lq": np.stack([
            np.zeros((3, 3)), np.full((3, 3), 0.5), np.ones((3, 3))
        ], axis=-1)[None], "gt": None}
        checked = guard.wrap(lambda value: value["lq"].shape)
        self.assertEqual(checked(batch), (1, 3, 3, 3))
        self.assertEqual([gate["status"] for gate in cert.gates],
                         ["PASS", "PASS"])
        self.assertTrue(all(
            gate["evidence"]["boundary"] == "training_step_input"
            for gate in cert.gates))


if __name__ == "__main__":
    unittest.main()
