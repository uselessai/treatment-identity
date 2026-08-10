"""Adapter for BasicSR's ``REDSDataset`` --- Campaign P subject 4.

Included for reach rather than for coverage. This is the most widely reused
video training loader in restoration research: a very large fraction of
published video super-resolution and restoration work either uses it or uses a
descendant of it. A result here therefore says more about the field than a
result on a one-off loader, even though it exercises no gate the earlier
subjects left untouched.

It also supports a small validity check the other subjects cannot. Subject~1
(``kair.VideoRecurrentTrainDataset``) is a descendant of this class. The two
share ancestry with each other but not with the audited family, so running both
asks whether the gates return the \\emph{same} verdict on two loaders that
should behave alike. Disagreement would be evidence about the instrument, not
about the subjects.

Note the ancestry carefully, because it is easy to state wrongly: BasicSR is
unrelated to the four pipelines the article audits. It is related to another
portability subject.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

BASICSR_ROOT = Path("/home/laura/02ImproveData/zBasicSR")

from treatment_identity.adapter import LoaderSpec, Sample  # noqa: E402


class BasicSRRedsAdapter:
    """Expose ``REDSDataset`` to the treatment-identity gates."""

    name = "basicsr.REDSDataset"
    project = "BasicSR"
    url = "https://github.com/XPixelGroup/BasicSR"

    #: Set by :meth:`_restore_generated_module`. Reported: this subject cannot
    #: be imported from a clone of its repository.
    environment_reconstruction: str | None = None

    def __init__(self, basicsr_root: Path = BASICSR_ROOT, *, shim: bool = True):
        self.basicsr_root = Path(basicsr_root)
        if str(self.basicsr_root) not in sys.path:
            sys.path.insert(0, str(self.basicsr_root))
        if shim:
            self._restore_generated_module()
        from basicsr.data.reds_dataset import REDSDataset             # noqa: E402
        self._cls = REDSDataset

    # -- environment reconstruction -----------------------------------------
    @classmethod
    def _restore_generated_module(cls, root: Path = BASICSR_ROOT) -> None:
        """Write the ``basicsr/version.py`` that the package generates on install.

        ``basicsr/__init__.py`` imports ``__version__`` and ``__gitsha__`` from
        a module that does not exist in the repository: ``setup.py`` writes it
        during installation. A clone of the repository is therefore not
        importable, and nothing in the tree says so --- the failure surfaces as
        ``ModuleNotFoundError: No module named 'basicsr.version'``.

        This is environment reconstruction, like the SciPy case in the blind
        super-resolution subject and unlike a modification of the subject: the
        file we write is the file the project's own build step writes. It is
        recorded because a third party auditing this loader has to discover it,
        and because the two cases together make one point --- the executable
        artefact is not the repository.
        """
        version_py = Path(root) / "basicsr" / "version.py"
        if version_py.exists():
            return
        version_py.write_text(
            "# GENERATED for auditing: setup.py writes this file on install,\n"
            "# so a clone of the repository is not importable without it.\n"
            "__version__ = '0.0.0+audit'\n"
            "__gitsha__ = 'unknown'\n"
            "version_info = (0, 0, 0)\n")
        cls.environment_reconstruction = (
            "basicsr/version.py written by hand; the package generates it at "
            "install time and the repository does not contain it")

    # -- adapter protocol ---------------------------------------------------
    def build(self, spec: LoaderSpec):
        gt_root = Path(spec.gt_root)
        lq_root = Path(spec.lq_root) if spec.lq_root else gt_root

        clips = sorted(p for p in gt_root.iterdir() if p.is_dir())
        if not clips:
            raise NotImplementedError("fixture has no clip directories")
        frames = sorted(clips[0].glob("*.png"))
        h, w = 64, 64
        try:
            import cv2
            h, w = cv2.imread(str(frames[0])).shape[:2]
        except Exception:                                    # pragma: no cover
            pass

        # This subject's documented meta-info format is the three-field one its
        # docstring shows, and its parser agrees. Subject 1, which descends from
        # this class, diverged to four fields without updating its docstring --
        # a small piece of evidence about how conventions drift inside a fork.
        meta = gt_root.parent / "meta_info_basicsr.txt"
        meta.write_text("".join(
            f"{c.name} {len(sorted(c.glob('*.png')))} ({h},{w},3)\n" for c in clips))

        opt = {
            "name": "fixture",
            "dataroot_gt": str(gt_root),
            "dataroot_lq": str(lq_root),
            "dataroot_flow": None,
            "meta_info_file": str(meta),
            "val_partition": "official",
            "io_backend": {"type": "disk"},
            "num_frame": spec.num_frames,
            "gt_size": min(h, w),
            "interval_list": [1],
            "random_reverse": False,
            "use_hflip": False,
            "use_rot": False,
            "scale": 1,
            "phase": "train",
        }
        self._ds = self._cls(opt)
        return self._ds

    def sample(self, loader, index: int) -> Sample:
        item = loader[index % len(loader)]
        lq = _to_numpy(item["lq"])
        gt = _to_numpy(item["gt"])
        if gt.ndim == 3:                      # REDSDataset returns a single GT
            gt = gt[None, ...]
        return Sample(lq=lq, gt=np.repeat(gt, lq.shape[0], axis=0)
                      if gt.shape[0] == 1 and lq.ndim == 4 else gt,
                      frame_ids=None)

    def __len__(self) -> int:
        return len(self._ds)


def _to_numpy(x) -> np.ndarray:
    """Torch (T,C,H,W) or (C,H,W) in [0,1] -> numpy (...,H,W,C) in [0,255]."""
    try:
        import torch
        a = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
    except ImportError:                                      # pragma: no cover
        a = np.asarray(x)
    if a.ndim == 4 and a.shape[1] in (1, 3):
        a = np.transpose(a, (0, 2, 3, 1))
    elif a.ndim == 3 and a.shape[0] in (1, 3):
        a = np.transpose(a, (1, 2, 0))
    if a.dtype.kind == "f" and a.max() <= 1.0 + 1e-6:
        a = a * 255.0
    return a
