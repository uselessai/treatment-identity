"""Per-item content scale: a uniform loader against one that does two things.

The single-reading gate answers "what does this loader do to the picture",
which is the right question only while the loader does one thing. These tests
pin the blind spot that follows from that, and the per-item reading that closes
it.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from treatment_identity import check_content_scale, check_content_scale_by_item
from treatment_identity import fixtures as fx

PATCH = 256


def _crop(img: np.ndarray, patch: int = PATCH) -> np.ndarray:
    y = (img.shape[0] - patch) // 2
    x = (img.shape[1] - patch) // 2
    return img[y:y + patch, x:x + patch]


def _resize(img: np.ndarray, patch: int = PATCH) -> np.ndarray:
    return cv2.resize(img, (patch, patch), interpolation=cv2.INTER_AREA)


def _always_resize(img: np.ndarray) -> np.ndarray:
    return _resize(img)


def _crop_if_large_enough(img: np.ndarray) -> np.ndarray:
    """The ordinary shape-dependent policy: crop what fits, resize what does not."""
    h, w = img.shape[:2]
    return _crop(img) if h >= PATCH and w >= PATCH else _resize(img)


def _probes_by_item(root: Path, clips, deliver) -> dict:
    """One item per clip, each with its own probe, as the gate expects."""
    probes = {}
    for i, clip in enumerate(clips):
        source = clip.read(0)
        h, w = source.shape[:2]
        probes[i] = [(deliver(source), clip.period_x, clip.period_y, w / h)]
    return probes


class ContentScaleByItemTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="ti_scale_mixed_")
        self.root = Path(self._tmp.name)
        self.clips = fx.make_scale_mixed(self.root, n_frames=2)
        self.addCleanup(self._tmp.cleanup)

    def test_uniform_loader_reads_as_one_behaviour(self) -> None:
        res = check_content_scale_by_item(
            _probes_by_item(self.root, self.clips, _always_resize))
        self.assertEqual(res.evidence["behaviour"], "reshape")
        self.assertEqual(res.evidence["items_read"], len(self.clips))
        self.assertEqual(set(res.evidence["counts"]), {"reshape"})

    def test_shape_dependent_loader_reads_as_mixed(self) -> None:
        res = check_content_scale_by_item(
            _probes_by_item(self.root, self.clips, _crop_if_large_enough))
        self.assertEqual(res.evidence["behaviour"], "mixed")
        self.assertEqual(set(res.evidence["counts"]), {"crop", "reshape"})
        self.assertEqual(sum(res.evidence["counts"].values()), len(self.clips))
        self.assertIn("does not treat every item alike", res.message)

    def test_single_reading_misses_the_split(self) -> None:
        """Why the per-item gate exists: one item names one behaviour."""
        by_item = _probes_by_item(self.root, self.clips, _crop_if_large_enough)
        first = check_content_scale(by_item[0])
        last = check_content_scale(by_item[len(self.clips) - 1])
        self.assertEqual(first.evidence["behaviour"], "crop")
        self.assertEqual(last.evidence["behaviour"], "reshape")

    def test_declaration_turns_the_split_into_a_verdict(self) -> None:
        by_item = _probes_by_item(self.root, self.clips, _crop_if_large_enough)
        undeclared = check_content_scale_by_item(by_item)
        declared = check_content_scale_by_item(by_item, declared="crop")
        self.assertEqual(undeclared.status, "UNDECL")
        self.assertEqual(declared.status, "FAIL")

    def test_unreadable_item_is_counted_not_assumed(self) -> None:
        by_item = _probes_by_item(self.root, self.clips, _always_resize)
        flat = np.full((PATCH, PATCH, 3), 128, np.uint8)
        by_item[len(self.clips)] = [(flat, 12, 20, 1.0)]
        res = check_content_scale_by_item(by_item)
        self.assertEqual(res.evidence["items_unreadable"], 1)
        self.assertEqual(res.evidence["items_read"], len(self.clips))
        self.assertEqual(res.evidence["behaviour"], "reshape")


if __name__ == "__main__":
    unittest.main()
