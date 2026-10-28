"""The profile is part of every run, not only of a run without a contract."""
import json
import unittest
from pathlib import Path

import cv2
import numpy as np

from treatment_identity import CheckResult, Certificate, Sample, validate_certificate
from treatment_identity.profile import describe_loader

ROOT = Path(__file__).resolve().parents[1]


class _Crops:
    """Reads the stored pair and returns a moving window, cropped to 128x128."""

    name = "stand-in.crops"

    def build(self, spec):
        gt = sorted(next(Path(spec.gt_root).iterdir()).glob("*.png"))
        lq = sorted(next(Path(spec.lq_root).iterdir()).glob("*.png"))
        self._pairs, self._n = list(zip(gt, lq)), spec.num_frames
        self._rng = np.random.default_rng(spec.seed)
        return self

    def sample(self, loader, index):
        start = int(self._rng.integers(0, len(self._pairs) - self._n))
        pairs = self._pairs[start:start + self._n]
        read = lambda p: cv2.imread(str(p))[:128, :128]          # noqa: E731
        return Sample(gt=np.stack([read(g) for g, _ in pairs]),
                      lq=np.stack([read(q) for _, q in pairs]))


class ProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profile = describe_loader(_Crops())

    def test_profile_describes_every_part(self):
        p = self.profile
        self.assertEqual(p["streams"]["gt"]["shape"], [128, 128, 3])
        self.assertEqual(p["content"]["behaviour"], "crop")
        self.assertEqual(p["window"]["frames"], 5)
        self.assertGreater(p["window"]["distinct_windows"], 1)
        self.assertEqual(p["input_source"]["source"], "reads a pre-rendered input")
        self.assertFalse(p["operators"]["observed"])

    def test_a_checked_certificate_carries_the_profile(self):
        cert = Certificate("stand-in", "values in [0, 255]")
        cert.add(CheckResult("value_range", "PASS", "stand-in verdict", {}))
        cert.profile = self.profile
        document = json.loads(json.dumps(cert.to_dict()))
        validate_certificate(document)
        self.assertEqual(document["schema"], "treatment-certificate/1.3")
        self.assertEqual(document["profile"]["content"]["behaviour"], "crop")

    def test_older_certificates_validate_against_their_own_schema(self):
        old = ROOT / "studies/history/certificates_1.6.0/released/rtn.json"
        document = json.loads(old.read_text())
        self.assertEqual(document["schema"], "treatment-certificate/1.2")
        validate_certificate(document)


if __name__ == "__main__":
    unittest.main()
