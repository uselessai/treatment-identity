"""Adapter for Restormer's ``Dataset_PairedImage`` --- a further portability subject.

Restormer (CVPR 2022, Zamir et al., https://github.com/swz30/Restormer) ships a
vendored copy of BasicSR, so its paired training loader reads the same
``dataroot_gt``/``dataroot_lq`` convention the REDS subject uses. The two are
not redundant: this one is single-image and applies its own geometric
augmentations, so it exercises a different applicability profile with a
convention already known.

Adaptation effort is a response variable of the study. Everything below the
header is that cost, and the subject is not modified to make a gate pass.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESTORMER_ROOT = Path("/home/laura/02ImproveData/zRestormer")

for base in (HERE, *HERE.parents):
    for cand in (base, base / "treatment_identity"):
        if (cand / "treatment_identity" / "__init__.py").exists():
            sys.path.insert(0, str(cand))
            break

from treatment_identity.adapter import LoaderSpec, Sample  # noqa: E402


from _configured import fitted  # noqa: E402

class RestormerAdapter:
    """Expose Restormer's paired training loader to the gates."""

    name = "restormer.Dataset_PairedImage"
    project = "Restormer"
    url = "https://github.com/swz30/Restormer"

    #: Set by :meth:`_restore_missing_package`. Reported in the certificate and
    #: in the article: this subject cannot be imported as released.
    environment_reconstruction: str | None = None

    def __init__(self, root: Path = RESTORMER_ROOT):
        self.root = Path(root)
        self._restore_missing_package()
        from basicsr.data.paired_image_dataset import Dataset_PairedImage
        self._cls = Dataset_PairedImage

    @classmethod
    def _restore_missing_package(cls, root: Path = RESTORMER_ROOT) -> None:
        """Make the subject's vendored ``basicsr`` importable.

        The repository ships ``basicsr/data/__init__.py`` and does not track
        ``basicsr/__init__.py``; its ``setup.py`` generates ``version.py`` and
        not that file. A clean clone therefore has no importable ``basicsr``
        package, and ``import basicsr.data...`` does not fail: it resolves to
        whatever ``basicsr`` the environment has installed, which is a
        different codebase with a different API. The failure surfaces later, as
        a missing class name, and would not surface at all if the installed
        package happened to define the same name.

        The package is reconstructed in memory, bound to the subject's own
        directory. Nothing is written into the subject's tree.
        """
        import types

        vendored = Path(root) / "basicsr"
        existing = sys.modules.get("basicsr")
        if existing is not None and getattr(existing, "__path__", None) == [str(vendored)]:
            return
        pkg = types.ModuleType("basicsr")
        pkg.__path__ = [str(vendored)]                   # type: ignore[attr-defined]
        sys.modules["basicsr"] = pkg
        for sub in ("basicsr.data", "basicsr.utils", "basicsr.models"):
            sys.modules.pop(sub, None)
        cls.environment_reconstruction = (
            "basicsr/__init__.py reconstructed in memory over the subject's own "
            "vendored directory; the repository does not track it and setup.py "
            "does not generate it, so a clean clone silently imports the "
            "installed basicsr instead")

    def build(self, spec: LoaderSpec):
        # Two subjects in this campaign vendor a package called ``basicsr``.
        # Whichever imported last owns the name, so each reclaims it at build
        # time rather than at import time; the campaign runs subjects in
        # sequence and this keeps them from reading each other's code.
        self._restore_missing_package(self.root)
        import importlib
        mod = importlib.import_module("basicsr.data.paired_image_dataset")
        self._cls = mod.Dataset_PairedImage
        gt_root, lq_root = Path(spec.gt_root), Path(spec.lq_root or spec.gt_root)
        # The fixture writes one directory per clip; this loader wants a flat
        # pair of directories whose file names correspond.
        stage = gt_root.parent / "restormer_stage"
        for name, src in (("gt", gt_root), ("lq", lq_root)):
            d = stage / name
            d.mkdir(parents=True, exist_ok=True)
            for f in sorted(src.rglob("*.png")):
                link = d / f"{f.parent.name}_{f.name}"
                if not link.exists():
                    link.symlink_to(f.resolve())
        opt = {
            "name": "ti_probe",
            "phase": "train",
            "scale": 1,
            # Deraining/Options/Deraining_Restormer.yml (progressive maximum)
            "gt_size": fitted(384, gt_root),
            "geometric_augs": False,
            "dataroot_gt": str(stage / "gt"),
            "dataroot_lq": str(stage / "lq"),
            "io_backend": {"type": "disk"},
        }
        self._ds = self._cls(opt)
        return self._ds

    def sample(self, loader, index: int) -> Sample:
        item = loader[index % len(loader)]
        return Sample(gt=_to_numpy(item["gt"]), lq=_to_numpy(item["lq"]),
                      frame_ids=None)

    def __len__(self) -> int:
        return len(self._ds)


def _to_numpy(x) -> np.ndarray:
    """Torch (C,H,W) in [0,1] -> numpy (H,W,C) in [0,255]."""
    try:
        import torch
        a = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
    except ImportError:                                      # pragma: no cover
        a = np.asarray(x)
    if a.ndim == 3 and a.shape[0] in (1, 3):
        a = np.transpose(a, (1, 2, 0))
    if a.dtype.kind == "f" and a.max() <= 1.0 + 1e-6:
        a = a * 255.0
    return a
