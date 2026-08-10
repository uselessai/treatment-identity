"""Adapter for KAIR's ``VideoRecurrentTrainDataset`` --- Campaign P subject 1.

KAIR (``https://github.com/cszn/KAIR``) is an independently developed training
toolbox with no code relationship to the family audited in the article. It is
therefore a portability subject rather than another member of the same lineage.

What this file exists to measure, besides the gate outcomes, is **adaptation
effort**: what a third party has to write to make an existing loader auditable.
Everything below the header is that cost. Keep it honest --- do not "fix" the
subject loader to make a gate pass, and do not add glue that the subject does
not need in normal use.

The dataset expects:
  * ``dataroot_gt`` / ``dataroot_lq`` directories of clip folders;
  * a meta-info text file listing ``<clip> <n_frames> (h,w,c)`` per line;
  * ``num_frame`` and ``gt_size``.

The protocol's fixtures already write clip folders in that shape, so the whole
adapter is: synthesise the meta-info file the subject wants, build it, and
convert one item into a :class:`Sample`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

KAIR_ROOT = Path("/home/laura/02ImproveData/zKAIR")

from treatment_identity.adapter import LoaderSpec, Sample  # noqa: E402


class KairVideoTrainAdapter:
    """Expose ``VideoRecurrentTrainDataset`` to the treatment-identity gates."""

    name = "kair.VideoRecurrentTrainDataset"
    project = "KAIR"
    url = "https://github.com/cszn/KAIR"

    def __init__(self, kair_root: Path = KAIR_ROOT):
        self.kair_root = Path(kair_root)
        if str(self.kair_root) not in sys.path:
            sys.path.insert(0, str(self.kair_root))
        from data.dataset_video_train import VideoRecurrentTrainDataset  # noqa: E402
        self._cls = VideoRecurrentTrainDataset

    # -- adapter protocol ---------------------------------------------------
    def build(self, spec: LoaderSpec):
        gt_root = Path(spec.gt_root)
        lq_root = Path(spec.lq_root) if spec.lq_root else gt_root

        clips = sorted(p for p in gt_root.iterdir() if p.is_dir())
        if not clips:
            raise NotImplementedError("fixture has no clip directories")
        first = clips[0]
        frames = sorted(first.glob("*.png"))
        h, w = 64, 64
        try:
            import cv2
            h, w = cv2.imread(str(frames[0])).shape[:2]
        except Exception:                                    # pragma: no cover
            pass

        # The class docstring documents three fields per line; the parser at
        # dataset_video_train.py:65 unpacks four (folder, frame_num, shape,
        # start_frame). Discovering that cost one run. Recorded here because
        # adaptation effort is one of this study's response variables, and an
        # undocumented input convention is exactly the kind of cost it measures.
        meta = gt_root.parent / "meta_info_fixture.txt"
        meta.write_text("".join(
            f"{c.name} {len(sorted(c.glob('*.png')))} ({h},{w},3) 0\n" for c in clips))

        opt = {
            "dataroot_gt": str(gt_root),
            "dataroot_lq": str(lq_root),
            "meta_info_file": str(meta),
            "val_partition": "official",
            "num_frame": spec.num_frames,
            "gt_size": min(h, w),
            "scale": 1,
            "interval_list": [1],
            "random_reverse": False,
            "use_hflip": False,
            "use_rot": False,
            "filename_tmpl": "08d",
            "filename_ext": "png",
            "io_backend": {"type": "disk"},
            "phase": "train",
            "sigma": 0,
            "name": "fixture",          # not "REDS": no validation partition
            "test_mode": False,
        }
        self._ds = self._cls(opt)
        return self._ds

    def sample(self, loader, index: int) -> Sample:
        item = loader[index % len(loader)]
        lq = _to_numpy(item["L"] if "L" in item else item["lq"])
        gt = _to_numpy(item["H"] if "H" in item else item["gt"])
        ids = item.get("key")
        return Sample(lq=lq, gt=gt, frame_ids=None,
                      extra={"key": ids} if ids is not None else {})

    def __len__(self) -> int:
        return len(self._ds)

    # ``disable_generator`` is deliberately NOT implemented: this subject has no
    # online degradation generator to rebind, so the pre-computed-input gate
    # falls back to inspecting the delivered tensor. The certificate records
    # that the weaker of the two forms was used, which is the honest outcome
    # and part of what Study 3 measures.


def _to_numpy(x) -> np.ndarray:
    """Torch (T,C,H,W) in [0,1] -> numpy (T,H,W,C) in [0,255]."""
    try:
        import torch
        if isinstance(x, torch.Tensor):
            a = x.detach().cpu().numpy()
        else:
            a = np.asarray(x)
    except ImportError:                                      # pragma: no cover
        a = np.asarray(x)
    if a.ndim == 4 and a.shape[1] in (1, 3):
        a = np.transpose(a, (0, 2, 3, 1))
    if a.dtype.kind == "f" and a.max() <= 1.0 + 1e-6:
        a = a * 255.0
    return a
