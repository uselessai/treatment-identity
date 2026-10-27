"""Adapter for VRT's training loader --- a further portability subject.

VRT (Video Restoration Transformer, Liang et al., arXiv:2201.12288,
https://github.com/JingyunLiang/VRT) is a transformer for video SR, deblurring,
denoising and interpolation. Its repository ships no training loader: it holds the network,
the test script and a test dataset, and its README says "The training code is
at KAIR" (README.md:154). VRT is trained by KAIR's ``main_train_vrt.py`` from
the configurations KAIR ships under ``options/vrt/``. The one audited here is
``001_train_vrt_videosr_bi_reds_6frames.json`` (video SR x4 on REDS, the
first task of VRT's README), whose training set is ``VideoRecurrentTrainDataset``
(json line 21; ``data/select_dataset.py:59-60``).

That is the class subject ``kair_video`` already adapts, from the same KAIR
file and commit. What this subject adds is the configuration a published model
was trained under: scale 4 with a pre-rendered bicubic input, 256-pixel target
patches, six frames, flips and rotations on. ``kair_video`` audits the class
under a neutral scale-1 setting; this one audits what VRT asked of it.

Two adaptation costs are specific to this subject and are recorded because
adaptation effort is a response variable of the study:

* The protocol's fixtures are same-size pairs; at scale 4 the loader requires
  ``GT = 4 x LQ`` exactly. :meth:`build` renders one stream at the other's
  resolution with exact pixel replication or decimation, so no value changes.
* VRT and KAIR both ship a top-level ``utils`` with no ``__init__.py``. With
  VRT first on the path --- where the harness puts a subject's root --- KAIR's
  loader imports cleanly and binds VRT's ``utils/utils_video.py``, which lacks
  ``FileClient`` and ``paired_random_crop``; the failure surfaces only at the
  first item. The adapter therefore loads KAIR's two files by path.

The subject is not modified to make a gate pass.
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
import types
from pathlib import Path

import cv2
import numpy as np

KAIR_ROOT = Path("/home/laura/02ImproveData/zKAIR")       # executed code: fc1732f4a451

from treatment_identity.adapter import LoaderSpec, Sample  # noqa: E402

# options/vrt/001_train_vrt_videosr_bi_reds_6frames.json; line numbers there.
SCALE = 4                    # :9
GT_SIZE = 256                # :31


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class VrtTrainAdapter:
    """Expose VRT's REDS x4 training loader to the treatment-identity gates."""

    name = "vrt.VideoRecurrentTrainDataset(kair)"
    project = "VRT"
    url = "https://github.com/JingyunLiang/VRT"

    environment_note = (
        "KAIR utils/utils_video.py and data/dataset_video_train.py loaded by file "
        "path; 'utils' and 'utils.utils_video' are aliased to KAIR's copy in "
        "sys.modules only while the loader module executes, then restored")

    def __init__(self, kair_root: Path = KAIR_ROOT):
        root = Path(kair_root)
        self.uv = _load("_vrt_kair_utils_video", root / "utils" / "utils_video.py")
        pkg = types.ModuleType("utils")
        pkg.__path__, pkg.utils_video = [], self.uv
        saved = {k: sys.modules.get(k) for k in ("utils", "utils.utils_video")}
        sys.modules.update({"utils": pkg, "utils.utils_video": self.uv})
        try:
            mod = _load("_vrt_kair_dataset_video_train",
                        root / "data" / "dataset_video_train.py")
        finally:
            for k, v in saved.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v
        self._cls = mod.VideoRecurrentTrainDataset
        # Observes the augmentation draw so the sweep can say whether a
        # setting's branch was entered; delegates to the original unchanged.
        self.last_status = (False, False, False)
        original = self.uv.augment

        def augment(imgs, hflip=True, rotation=True):
            out, self.last_status = original(imgs, hflip, rotation, return_status=True)
            return out
        self.uv.augment = augment

    # -- adapter protocol ---------------------------------------------------
    def _stage(self, gt_root: Path, lq_root: Path) -> Path:
        """Write the pair at the 4:1 resolution ratio the configuration needs.

        REDS stores GT at full size and LQ as an offline bicubic x4 render. A
        fixture large enough for one 256-pixel GT patch keeps its GT, cropped
        to a multiple of 4 as KAIR's LR-generation scripts do, and its stored
        LQ is decimated x4. A smaller fixture keeps its stored LQ and has its GT
        replicated x4, so the input the gate wrote is the input the loader
        opens. Replication magnifies the GT, but only on fixtures too small to
        be cropped at all; the content-scale probes always take the first route.
        """
        stage = gt_root.parent / "vrt_stage"
        if stage.exists():
            shutil.rmtree(stage)
        for gt_png in sorted(gt_root.rglob("*.png")):
            rel = gt_png.relative_to(gt_root)
            gt, lq = cv2.imread(str(gt_png)), cv2.imread(str(lq_root / rel))
            h, w = gt.shape[:2]
            if min(h, w) >= GT_SIZE:
                gt = gt[:h - h % SCALE, :w - w % SCALE]
                lq = lq[:h - h % SCALE:SCALE, :w - w % SCALE:SCALE]
            else:
                gt = np.repeat(np.repeat(gt, SCALE, 0), SCALE, 1)
            for side, img in (("GT", gt), ("LQ", lq)):
                (stage / side / rel.parent).mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(stage / side / rel), img)
        return stage

    def build(self, spec: LoaderSpec, **overrides):
        gt_root = Path(spec.gt_root)
        stage = self._stage(gt_root, Path(spec.lq_root) if spec.lq_root else gt_root)
        clips = sorted(p for p in (stage / "GT").iterdir() if p.is_dir())
        if not clips:
            raise NotImplementedError("fixture has no clip directories")
        h, w = cv2.imread(str(next(clips[0].glob("*.png")))).shape[:2]
        # Four-field meta-info, as the parser at dataset_video_train.py:65 reads.
        meta = stage / "meta_info_vrt.txt"
        meta.write_text("".join(
            f"{c.name} {len(list(c.glob('*.png')))} ({h},{w},3) 0\n" for c in clips))
        opt = {
            "name": "train_dataset",              # json :20
            "dataroot_gt": str(stage / "GT"),
            "dataroot_lq": str(stage / "LQ"),
            "meta_info_file": str(meta),
            "filename_tmpl": "08d",               # :25
            "filename_ext": "png",                # :26
            "val_partition": "REDS4",             # :27
            "test_mode": False,                   # :28
            # :29 asks for lmdb. The fixtures are PNG folders; the disk backend
            # reaches the same imfrombytes call with the same bytes.
            "io_backend": {"type": "disk"},
            "num_frame": spec.num_frames,         # :30 asks for 6; the probe's request is honoured
            "gt_size": GT_SIZE,
            "interval_list": [1],                 # :32
            "random_reverse": False,              # :33
            "use_hflip": True,                    # :34
            "use_rot": True,                      # :35
            "scale": SCALE,                       # :9, broadcast by utils_option.py:58
            "n_channels": 3,                      # :10, broadcast by utils_option.py:59
            "phase": "train",
        }
        opt.update(overrides)
        self._ds = self._cls(opt)
        return self._ds

    def sample(self, loader, index: int) -> Sample:
        item = loader[index % len(loader)]
        # Delivered as the loader returns it: (T, C, H, W), float32 in [0, 1].
        return Sample(lq=item["L"].numpy(), gt=item["H"].numpy(), frame_ids=None,
                      extra={"key": item["key"]})

    def __len__(self) -> int:
        return len(self._ds)

    # -- settings sweep -----------------------------------------------------
    #: The three augmentation switches the configuration names (json :33-35).
    def settings(self):
        return {"use_hflip": (False, True), "use_rot": (False, True),
                "random_reverse": (False, True)}

    def deliver_under(self, workdir: Path, name: str, value):
        """Deliver item 0 of an index-encoded clip with one switch changed."""
        from treatment_identity.fixtures import decode_index, make_pair
        root = Path(workdir) / "vrt_sweep"
        if not (root / "GT").exists():
            make_pair(root, n_frames=32, gt_kind="index", lq_kind="index")
        ds = self.build(LoaderSpec(gt_root=root / "GT", lq_root=root / "LQ",
                                   num_frames=6), **{name: value})
        lq = ds[0]["L"].numpy()
        hflip, vflip, rot90 = self.last_status
        ids = [decode_index(f) for f in lq]
        reached = {"use_hflip": hflip, "use_rot": vflip or rot90,
                   "random_reverse": ids == sorted(ids, reverse=True)}[name]
        return lq, reached
