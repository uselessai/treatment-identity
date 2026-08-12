import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from treatment_identity.fixtures import decode_index, make_index_encoded


class IndexFixtureTests(unittest.TestCase):
    def test_index_one_is_unambiguous_in_three_numerical_conventions(self):
        with tempfile.TemporaryDirectory() as tmp:
            clip = make_index_encoded(Path(tmp), n_frames=3)
            level = cv2.imread(str(clip.frame_path(1)), cv2.IMREAD_UNCHANGED)
            self.assertEqual(decode_index(level), 1)
            self.assertEqual(decode_index(level.astype(np.float32)), 1)
            self.assertEqual(decode_index(level.astype(np.float32) / 255.0), 1)
            self.assertEqual(
                decode_index(level.astype(np.float32) / 127.5 - 1.0), 1)

    def test_signature_survives_offset_crop_and_chw_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            clip = make_index_encoded(Path(tmp), n_frames=18)
            frame = cv2.imread(str(clip.frame_path(17)), cv2.IMREAD_UNCHANGED)
            crop = frame[3:38, 2:41]
            self.assertEqual(decode_index(np.moveaxis(crop, -1, 0)), 17)

    def test_corrupted_signature_is_rejected(self):
        corrupted = np.arange(16 * 16 * 3, dtype=np.float32).reshape(16, 16, 3)
        self.assertIsNone(decode_index(corrupted))


if __name__ == "__main__":
    unittest.main()
