"""The wizard serves fixtures to a project's own reads and finds its data.

Each test builds a tiny project in a temporary folder. Nothing outside it is
read or written.
"""

from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from treatment_identity import wizard
from treatment_identity.checks import measure_content_scale
from treatment_identity.fixtures import make_scale_grid


def _answers(**data):
    a = wizard.Answers(None)
    a.data.update(data)
    return a


class ServingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.root = self.tmp / "data" / "gt"
        self.root.mkdir(parents=True)
        self.data = wizard.VirtualData({str(self.root): "gt"})
        self.frames = [np.full((8, 8, 3), v, np.uint8) for v in (10, 20, 30)]
        self.data.frames = {"gt": self.frames}

    def test_reads_under_the_data_folder_are_served(self):
        with self.data.active():
            img = cv2.imread(str(self.root / "clip" / "00000002.png"))
            with open(self.root / "clip" / "00000001.png", "rb") as f:
                raw = f.read()
        self.assertEqual(int(img[0, 0, 0]), 30)
        decoded = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(int(decoded[0, 0, 0]), 20)
        # and nothing outside it
        with self.data.active():
            self.assertIsNone(cv2.imread(str(self.tmp / "elsewhere.png")))

    def test_an_archive_holds_at_least_a_fixture_clip(self):
        with self.data.active():
            with open(self.root / "clip.zip", "rb") as f:
                names = zipfile.ZipFile(io.BytesIO(f.read())).namelist()
        self.assertEqual(len(names), 100)

    def test_a_video_is_served_as_rgb_frames(self):
        frames = self.data.video_for(self.root / "a" / "clip.mp4")
        self.assertEqual(len(frames), 100)
        self.assertEqual(frames[3].shape, (8, 8, 3))
        self.assertIsNone(self.data.video_for(self.tmp / "clip.mp4"))

    def test_described_file_is_read_in_place_of_the_real_one(self):
        index = self.tmp / "train.json"
        index.write_text(json.dumps({"a": 1, "b": 2}))
        self.data.served[str(index.resolve())] = json.dumps({"a": 1})
        with self.data.active():
            with open(index) as f:
                self.assertEqual(json.load(f), {"a": 1})
        self.assertEqual(json.loads(index.read_text()), {"a": 1, "b": 2})


class PaddingTests(unittest.TestCase):
    def test_padding_keeps_the_scale_of_the_content(self):
        with tempfile.TemporaryDirectory() as td:
            make_scale_grid(Path(td) / "GT", n_frames=2, shape=(96, 160),
                            period_x=12, period_y=20)
            data = wizard.VirtualData({})
            data.min_size, data.pad = 256, True
            data.load("gt", Path(td) / "GT")
        frame = data.frames["gt"][0]
        self.assertEqual(frame.shape[:2], (256, 256))
        # the grid's periods, 12 and 20 pixels, are what is measured after padding
        mx, my = measure_content_scale(frame[None])
        self.assertAlmostEqual(mx / 12, 1.0, delta=0.05)
        self.assertAlmostEqual(my / 20, 1.0, delta=0.05)

    def test_fixtures_are_served_at_their_size_until_a_loader_fails(self):
        with tempfile.TemporaryDirectory() as td:
            make_scale_grid(Path(td) / "GT", n_frames=2, shape=(96, 160),
                            period_x=12, period_y=20)
            data = wizard.VirtualData({})
            data.min_size = 256
            data.load("gt", Path(td) / "GT")
        self.assertEqual(data.frames["gt"][0].shape[:2], (96, 160))


class DiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.cwd = Path(tempfile.mkdtemp())

    def test_stand_ins_are_not_taken_for_data(self):
        d = self.cwd / "frames"
        d.mkdir()
        (d / "00000000.png").touch()
        self.assertFalse(wizard._holds_data(d))
        cv2.imwrite(str(d / "00000001.png"), np.zeros((4, 4, 3), np.uint8))
        self.assertTrue(wizard._holds_data(d))

    def test_validation_folders_follow_command_line_references(self):
        (self.cwd / "train.yaml").write_text(
            "root: /nowhere\ndata:\n  train_dir: ${root}/train\n"
            "  val_gt: ${root}/val/gt\n  val_lq: ${root}/val/lq\n")
        out = wizard.other_roots(self.cwd, ["python", "train.yaml", f"root={self.cwd}/d"],
                                 {str(self.cwd / "d" / "train"): "gt"})
        self.assertEqual(sorted(Path(p).name for p in out), ["gt", "lq"])

    def test_model_folders_are_not_data(self):
        weights = self.cwd / "weights" / "vae"
        weights.mkdir(parents=True)
        (weights / "model.safetensors").touch()
        (self.cwd / "data" / "train").mkdir(parents=True)
        roots = wizard.discover_roots(
            self.cwd, ["python", "x.py", "--vae_path=weights/vae", "--train_data_dir=data/train"],
            _answers(data_roots="detected"))
        self.assertEqual(list(roots), [str((self.cwd / "data" / "train").resolve())])


    def test_subfolders_named_for_the_streams(self):
        for sub in ("target", "input"):
            (self.cwd / "data" / "train" / sub).mkdir(parents=True)
        roots = wizard.discover_roots(self.cwd, ["python", "x.py", "--train_dir=data/train"],
                                      _answers())
        base = (self.cwd / "data" / "train").resolve()
        self.assertEqual(roots, {str(base / "input"): "lq", str(base / "target"): "gt"})


class PreloadTests(unittest.TestCase):
    def test_a_dataset_that_reads_nothing_per_item_is_reported(self):
        from treatment_identity.adapter import AdapterError
        data = wizard.VirtualData({})
        data.frames = {"gt": [np.zeros((8, 8, 3), np.uint8)]}
        loaded = [{"gt": np.zeros((3, 8, 8), np.float32), "lq": np.zeros((3, 8, 8), np.float32)}]
        adapter = wizard.WizardAdapter(loaded, data, _answers(target_key="gt", input_key="lq"))
        adapter._reads_at_build = len(data.trace)
        with self.assertRaises(AdapterError):
            adapter.sample(loaded, 0)


class KeyTests(unittest.TestCase):
    def _keys(self, item):
        adapter = wizard.WizardAdapter(None, wizard.VirtualData({}), _answers())
        return adapter.keys(item)

    def test_names_of_input_and_target(self):
        a = np.zeros((3, 8, 8), np.float32)
        self.assertEqual(self._keys({"lq": a, "gt": a}), ("lq", "gt"))

    def test_an_image_and_its_mask(self):
        a = np.zeros((3, 8, 8), np.float32)
        self.assertEqual(self._keys({"image": a, "mask": a}), ({"masked_by": "mask"}, "image"))

    def test_conditioning_input_and_a_mask(self):
        a = np.zeros((3, 8, 8), np.float32)
        item = {"pixel_values": a, "conditioning_pixel_values": a, "masks": a}
        self.assertEqual(self._keys(item), ("conditioning_pixel_values", "pixel_values"))

    def test_a_target_held_one_level_down(self):
        import types
        a = np.zeros((3, 8, 8), np.float32)
        item = {"inputs": a, "data_samples": types.SimpleNamespace(gt_img=a)}
        self.assertEqual(self._keys(item), ("inputs", "data_samples.gt_img"))


if __name__ == "__main__":
    unittest.main()
