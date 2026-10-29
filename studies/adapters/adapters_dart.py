"""Adapter for DART's training loader --- a video restoration subject for old film.

DART (Jastrzebski et al., ACCV 2026,
https://github.com/TytanMikJas/DART-Degradation-Aware-Recurrent-Transformer)
restores old film with a recurrent transformer that is supervised with a mask of
the defects. Its training loader, ``VideoFrameSequenceDataset`` in
``src/data/video_frame_sequence_dataset.py``, takes clean clips and degrades
them on the fly with AbsoluteDegradation v1.0.0, which also returns the defect
mask. It is the first subject that delivers a third stream next to the input
and the target.

The loader is built the way ``scripts/train.py`` builds it: the Hydra
configuration ``configs/train.yaml`` is composed and its ``data`` node is
instantiated, with only the paths and the number of workers overridden. It runs
in DART's own environment (Python 3.12, NumPy 2), with AbsoluteDegradation
loaded from its clone rather than installed.
"""
from __future__ import annotations

import contextlib
import inspect
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DART_ROOT = Path("/home/laura/02ImproveData/zDART/repo")
DEGRADATION_ROOT = Path("/home/laura/02ImproveData/zDART/AbsoluteDegradation")
for base in (HERE, *HERE.parents):
    if (base / "treatment_identity" / "__init__.py").exists():
        sys.path.insert(0, str(base))
        break

from treatment_identity.adapter import CallRecorder, LoaderSpec, Sample  # noqa: E402


@contextlib.contextmanager
def _cwd(path: Path):
    """AbsoluteDegradation reads its blue-noise basis from a path relative to
    the working directory, ``data/blue_noise/...``; it is built from its own
    clone so that the basis it downloaded is the one it reads."""
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


class DartAdapter:
    """Expose DART's ``VideoFrameSequenceDataset`` as ``scripts/train.py`` builds it."""

    name = "dart.VideoFrameSequenceDataset"
    project = "DART"
    url = "https://github.com/TytanMikJas/DART-Degradation-Aware-Recurrent-Transformer"

    def __init__(self, root: Path = DART_ROOT, degradation_root: Path = DEGRADATION_ROOT):
        self.root, self.degradation_root = Path(root), Path(degradation_root)
        for p in (str(self.degradation_root), str(self.root)):
            if p not in sys.path:
                sys.path.insert(0, p)
        self.recorder = CallRecorder()
        self._instrument()

    def _instrument(self) -> None:
        """Record every degradation step that runs, by class name.

        The steps are composed in nested sequences, some of them permuted at
        random, so the order of the concrete steps is what the operation-order
        observation needs. Compositions themselves are not recorded.
        """
        from absolute_degradation.distortions import steps
        from absolute_degradation.distortions.steps.abstract_distort_step import (
            AbstractDistortStep)
        for name, cls in vars(steps).items():
            if (inspect.isclass(cls) and issubclass(cls, AbstractDistortStep)
                    and not inspect.isabstract(cls) and "Compose" not in name
                    and "distort" in vars(cls)):
                self.recorder.wrap(cls, "distort", name)

    def build(self, spec: LoaderSpec):
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        from hydra.utils import instantiate

        gt_root = Path(spec.gt_root)
        lq_root = Path(spec.lq_root or spec.gt_root)
        # Validation needs <val_root>/degraded and <val_root>/gt; the fixture
        # pair serves, since only the training dataset is probed.
        val = gt_root.parent / "dart_val"
        val.mkdir(parents=True, exist_ok=True)
        for name, src in (("degraded", lq_root), ("gt", gt_root)):
            link = val / name
            if not link.exists():
                link.symlink_to(src.resolve())
        overrides = [
            f"data.train_clean_root={gt_root}",
            f"val_root={val}",
            f"degradation.texture_path={self.degradation_root / 'data' / 'textures'}",
            f"data.num_frames={spec.num_frames}",
            "data.num_workers=0",
        ]
        if spec.use_precomputed_lq:
            # The loader's own option for stored inputs; the configuration's
            # online degradation is left as configured.
            overrides.append(f"+data.train_degraded_root={lq_root}")
        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=str(self.root / "configs"), version_base=None):
            cfg = compose(config_name="train", overrides=overrides)
        with _cwd(self.degradation_root):
            datamodule = instantiate(cfg.data)
            datamodule.setup("fit")
        self._ds = datamodule.train_dataset
        return self._ds

    def sample(self, loader, index: int) -> Sample:
        item = loader[index % len(loader)]
        lq = np.asarray(item["degraded"])
        gt = np.asarray(item["clean"]) if item.get("clean") is not None else None
        mask = item.get("degradation_mask")
        extra = {"mask_shape": list(np.asarray(mask).shape)} if mask is not None else {}
        return Sample(lq=lq, gt=gt, frame_ids=None, extra=extra)

    def __len__(self) -> int:
        return len(self._ds)
