"""Instrument defects found by applying the gates to six further subjects.

ProPainter, E2FGVI, DiffuEraser and LaMa exposed them in October 2026; each
test reproduces the counterexample with a stand-in loader, not the subject.
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from treatment_identity import Sample, check_operator_trace, check_temporal_window
from treatment_identity.adapter import CallRecorder
from treatment_identity.checks import _bimodality, _to_255


class _Noise:
    """A loader whose frames carry no decodable index, as after a resize."""

    def build(self, spec):
        return spec

    def sample(self, spec, index):
        rng = np.random.default_rng(index)
        frames = [rng.integers(0, 256, (16, 16, 3)).astype(np.float64)
                  for _ in range(spec.num_frames)]
        return Sample(lq=frames, gt=frames, frame_ids=None)


class _Empty:
    """A loader that drops every clip the fixture offers."""

    recorder = CallRecorder()

    def build(self, spec):
        return spec

    def sample(self, spec, index):
        raise IndexError("list index out of range")


class ExtensionRegressions(unittest.TestCase):
    def test_declared_range_reads_minus_one_to_one_grey_as_grey(self):
        grey = np.full((8, 8, 3), 128 / 127.5 - 1.0)       # 0.0039 in [-1, 1]
        self.assertLess(float(_to_255(grey).max()), 2.0)    # the guessed reading
        self.assertAlmostEqual(float(_to_255(grey, (-1.0, 1.0)).max()), 128.0)
        self.assertEqual(_bimodality(grey, (-1.0, 1.0)), 0.0)

    def test_black_alone_is_not_the_checkerboard(self):
        holes = np.zeros((8, 8, 3))
        holes[:2] = 128.0                                   # an inpainting hole
        self.assertEqual(_bimodality(holes), 0.0)
        board = np.indices((8, 8)).sum(0) % 2 * 255.0
        self.assertAlmostEqual(_bimodality(board), 1.0)

    def test_unreadable_index_is_skip_not_an_inert_sampler(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = check_temporal_window(_Noise(), Path(tmp), clip_length=8)
        self.assertEqual(r.status, "SKIP")
        self.assertIn("no readable index", r.message)

    def test_empty_loader_reaches_the_clip_escalation(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(IndexError):
                check_temporal_window(_Empty(), Path(tmp), clip_length=8)
            with self.assertRaises(IndexError):
                check_operator_trace(_Empty(), Path(tmp), _Empty.recorder,
                                     {"a"}, clip_length=8)

    def test_adapter_loc_ignores_docstrings(self):
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "campaigns"))
        import campaign_P_portability as camp
        with tempfile.TemporaryDirectory() as tmp:
            mod = Path(tmp) / "probe_adapter.py"
            mod.write_text('"""Module.\n\nTwo lines."""\n# note\n\n'
                           'def f():\n    """Doc."""\n    return 1\n')
            camp.HERE, here = Path(tmp), camp.HERE
            try:
                self.assertEqual(camp.adapter_loc("probe_adapter"), 2)
            finally:
                camp.HERE = here


if __name__ == "__main__":
    unittest.main()
