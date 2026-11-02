"""Requirements from the configuration and from a contract, judged on the profile."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from treatment_identity import wizard
from treatment_identity.requirements import combine, from_configuration, judge

PROFILE = {
    "streams": {"gt": {"shape": [256, 256, 3], "range": [0.0, 1.0], "channels": 3},
                "lq": {"shape": [64, 64, 3], "range": [0.0, 1.0], "channels": 3}},
    "window": {"frames": 15, "distinct_windows": 8, "note": "8 distinct windows"},
    "content": {"behaviour": "crop", "note": "crop",
                "sizes": [{"stored": [480, 854], "delivered": [256, 256]}],
                "evidence": {"probes": 5, "rotated_probes": 2,
                             "readings": [{"rotation_deg": 90.0}, {"rotation_deg": 90.0},
                                          {}, {}, {}]}},
    "input_source": {"source": "reads a pre-rendered input", "note": "stored"},
}


class ConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.cwd = Path(tempfile.mkdtemp())
        (self.cwd / "train.yml").write_text(
            "scale: 4\ndatasets:\n  train:\n    gt_size: 256\n    num_frame: 15\n"
            "    use_rot: true\n    dataroot_lq: data/lq\n  val:\n    num_frame: 100\n")

    def test_training_values_become_requirements(self):
        found, conflicting = from_configuration(self.cwd, ["python", "x.py", "-opt", "train.yml"])
        self.assertEqual({k: v["declared"] for k, v in found.items()},
                         {"scale": 4, "patch": 256, "frames": 15, "rotation": "allowed",
                          "input": "stored"})
        self.assertEqual(conflicting, {})
        self.assertIn("datasets.train.gt_size", found["patch"]["source"])

    def test_a_size_given_as_two_numbers(self):
        (self.cwd / "rtn.yml").write_text("datasets:\n  train:\n    gt_size: [256, 256]\n"
                                          "  val:\n    gt_size: [640, 368]\n")
        found, _ = from_configuration(self.cwd, ["python", "x.py", "-opt", "rtn.yml"])
        self.assertEqual(found["patch"]["declared"], [256, 256])

    def test_the_command_line_comes_first(self):
        found, _ = from_configuration(
            self.cwd, ["python", "x.py", "-opt", "train.yml", "--gt_size", "128"])
        self.assertEqual(found["patch"]["declared"], 128)
        self.assertEqual(found["patch"]["source"], "command line: gt_size")


class CombineTests(unittest.TestCase):
    configured = {"frames": {"declared": 15, "source": "train.yml: num_frame"},
                  "patch": {"declared": 256, "source": "train.yml: gt_size"}}

    def test_the_contract_adds_and_replaces(self):
        req = combine(self.configured, {"requirements": {"frames": 5, "window": "moves"}},
                      "mine.yml")
        self.assertEqual(req["frames"]["declared"], 5)
        self.assertEqual(req["frames"]["replaces"]["declared"], 15)
        self.assertEqual(req["window"]["source"], "mine.yml")
        self.assertEqual(req["patch"]["source"], "train.yml: gt_size")

    def test_ignore_and_contract_only(self):
        self.assertNotIn("patch", combine(self.configured, {"ignore": ["patch"]}))
        self.assertEqual(combine(self.configured, {"from_config": False}), {})

    def test_unknown_names_are_refused(self):
        with self.assertRaises(ValueError):
            combine({}, {"requirements": {"frame": 5}})


class JudgeTests(unittest.TestCase):
    def _status(self, **req):
        r = judge(PROFILE, {k: {"declared": v, "source": "t"} for k, v in req.items()})
        return {x.name.split(":")[1]: x.status for x in r}

    def test_verdicts_read_from_the_profile(self):
        self.assertEqual(self._status(frames=15, scale=4, patch=256, input="stored",
                                      window="moves", content="crop", rotation="present"),
                         dict.fromkeys(["frames", "scale", "patch", "input", "window",
                                        "content", "rotation"], "PASS"))
        self.assertEqual(self._status(frames=5, patch=128, rotation="none", input="made"),
                         dict.fromkeys(["frames", "patch", "rotation", "input"], "FAIL"))

    def test_every_stored_size_counts_for_the_patch(self):
        profile = dict(PROFILE, content=dict(PROFILE["content"], sizes=[
            {"stored": [480, 854], "delivered": [480, 854]}]))
        r = judge(profile, {"patch": {"declared": 256, "source": "t"}})[0]
        self.assertEqual(r.status, "FAIL")
        self.assertIn("stored [480, 854] delivered [480, 854]", r.message)


class ServedSettingTests(unittest.TestCase):
    def test_a_setting_is_served_and_the_file_left_alone(self):
        cwd = Path(tempfile.mkdtemp())
        (cwd / "train.yml").write_text("datasets:\n  train:\n    gt_size: 256\n")
        (cwd / "cfg.py").write_text("scale = 4\nmodel = dict(scale=scale)\n")
        a = wizard.Answers(None)
        a.data["set_in"] = {"train.yml:datasets.train.gt_size": 128, "cfg.py:scale": 2}
        served = wizard.served_settings(cwd, a)
        self.assertIn("gt_size: 128", served[str((cwd / "train.yml").resolve())])
        self.assertIn("scale = 2", served[str((cwd / "cfg.py").resolve())])
        self.assertIn("gt_size: 256", (cwd / "train.yml").read_text())


if __name__ == "__main__":
    unittest.main()
