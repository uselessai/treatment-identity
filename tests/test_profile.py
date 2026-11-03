"""The profile is part of every run, and the certificate carries it."""
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
        self.assertFalse(p["streams"]["gt"]["identical_channels"])

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


class OperationCountTests(unittest.TestCase):
    def test_counts_per_delivered_frame(self):
        from treatment_identity import CallRecorder

        class _Sharpens(_Crops):
            recorder = CallRecorder()

            def sample(self, loader, index):
                s = super().sample(loader, index)
                for _ in range(len(s.gt)):
                    self.recorder.record("sharpen")
                if index % 2 == 0:
                    self.recorder.record("jitter")
                return s

        from treatment_identity.profile import observe_operators
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            o = observe_operators(_Sharpens(), Path(td), draws=4)
        self.assertEqual(o["operations"]["sharpen"], {"per_frame": 1.0, "draws": 4})
        self.assertEqual(o["operations"]["jitter"]["draws"], 2)


class CallTraceTests(unittest.TestCase):
    """Calls recorded without being named: the project's own functions, and
    the functions of other code they call, but nothing inside that code."""

    PROJECT = (
        "import random\n"
        "import numpy as np\n"
        "def blur(x): return np.flip(x, 0)\n"
        "def noise(x): return x + 1\n"
        "def jpeg(x): return x\n"
        "def degrade(x, shuffle):\n"
        "    steps = [blur, noise, jpeg]\n"
        "    if shuffle: random.shuffle(steps)\n"
        "    for s in steps: x = s(x)\n"
        "    return x\n")

    def _draws(self, shuffle: bool):
        import importlib.util
        import random
        import tempfile
        import numpy as np
        from treatment_identity import CallRecorder
        from treatment_identity.profile import _call_orders
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / "proj_steps.py").write_text(self.PROJECT)
            spec = importlib.util.spec_from_file_location("proj_steps", Path(td) / "proj_steps.py")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            rec, nested = CallRecorder(roots=[td]), []
            random.seed(0)
            with rec.traced():                      # not watching: nothing recorded
                mod.degrade(np.zeros((2, 2)), shuffle)
            self.assertEqual(rec.calls, [])
            rec.watching = True
            for _ in range(8):
                rec.reset()
                with rec.traced():
                    mod.degrade(np.zeros((2, 2)), shuffle)
                nested.append(list(rec.nested))
            return rec.calls, _call_orders(nested)

    def test_project_functions_and_their_library_calls(self):
        calls, orders = self._draws(shuffle=False)
        self.assertEqual(calls, ["proj_steps.py:degrade", "proj_steps.py:blur",
                                 "numpy.flip", "proj_steps.py:noise", "proj_steps.py:jpeg"])
        self.assertEqual(orders["order"], "fixed")

    def test_an_order_drawn_at_random_changes(self):
        calls, orders = self._draws(shuffle=True)
        self.assertIn("random.shuffle", calls)
        self.assertEqual(orders["order"], "random")
        self.assertEqual(list(orders["changing"]), ["proj_steps.py:degrade"])


class RecorderWrapTests(unittest.TestCase):
    def test_wrap_records_and_keeps_behaviour(self):
        import types
        from treatment_identity import CallRecorder
        module = types.SimpleNamespace(blur=lambda x: x + 1)
        rec = CallRecorder()
        self.assertTrue(rec.wrap(module, "blur"))
        self.assertTrue(rec.wrap(module, "blur"))            # idempotent
        self.assertEqual(module.blur(1), 2)
        self.assertEqual(rec.calls, ["blur"])
        self.assertFalse(rec.wrap(module, "missing"))


if __name__ == "__main__":
    unittest.main()
