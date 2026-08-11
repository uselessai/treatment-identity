"""Adapter for Uformer's paired training loader.

Uformer (CVPR 2022, Wang et al., https://github.com/ZhendongWang6/Uformer) is a
transformer for image restoration. Its training loader reads a directory holding
``groundtruth/`` and ``input/``, which makes it a pre-rendered pipeline: the
degraded input is read from disk rather than produced online, and that is what
``precomputed_input`` is written to check.

It shares no code with the audited family and none with the other subjects: it
does not depend on BasicSR, which two otherwise attractive candidates
(Restormer, DDColor) turned out to bundle.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
UFORMER_ROOT = Path("/home/laura/02ImproveData/zUformer")

for base in (HERE, *HERE.parents):
    for cand in (base, base / "treatment_identity"):
        if (cand / "treatment_identity" / "__init__.py").exists():
            sys.path.insert(0, str(cand))
            break

from treatment_identity.adapter import LoaderSpec, Sample  # noqa: E402


class UformerAdapter:
    """Expose Uformer's ``DataLoaderTrain`` to the treatment-identity gates."""

    name = "uformer.DataLoaderTrain"
    project = "Uformer"
    url = "https://github.com/ZhendongWang6/Uformer"

    def __init__(self, root: Path = UFORMER_ROOT):
        self.root = Path(root)
        if str(self.root) not in sys.path:
            sys.path.insert(0, str(self.root))
        # The loader lives under dataset/ and imports utils from the repo root,
        # so both have to be importable, not just the package directory.
        if str(self.root / "dataset") not in sys.path:
            sys.path.insert(0, str(self.root / "dataset"))
        # Uformer's loader imports natsort, which its repository does not pin
        # and which is absent from a current scientific environment. This is
        # the third subject in this study whose published form does not run as
        # published, by a third unrelated mechanism. The dependency is
        # reconstructed here rather than installed: natsorted over fixture
        # filenames of uniform width is ordinary sorted, the subject's source
        # is untouched, and the certificate records that the environment was
        # reconstructed so a passing gate is not mistaken for a subject that
        # runs as published.
        try:
            import natsort  # noqa: F401
            self.reconstructed_dependency = None
        except ModuleNotFoundError:
            import types
            shim = types.ModuleType("natsort")
            shim.natsorted = lambda seq, *a, **k: sorted(seq)
            shim.os_sorted = shim.natsorted
            sys.modules["natsort"] = shim
            self.reconstructed_dependency = "natsort"

        # Uformer's loader does "from utils import is_png_file". So does KAIR,
        # with a different utils. The campaign imports every adapter into one
        # process, so whichever subject was imported first wins and the other
        # fails with a name that does exist in its own tree. Drop the cached
        # module and put this subject's root first for the duration of the
        # import: an adaptation cost created by auditing several subjects at
        # once, which no single-subject run would ever show.
        for stale in [m for m in list(sys.modules) if m == "utils"
                      or m.startswith("utils.")]:
            del sys.modules[stale]
        sys.path.insert(0, str(self.root))
        try:
            from dataset.dataset_motiondeblur import DataLoaderTrain  # noqa: E402
        finally:
            sys.path.remove(str(self.root))
        self._cls = DataLoaderTrain

    def build(self, spec: LoaderSpec):
        gt_root, lq_root = Path(spec.gt_root), Path(spec.lq_root or spec.gt_root)

        # The loader expects one directory containing 'groundtruth' and 'input'.
        # The fixtures are laid out as GT/<clip>/*.png and LQ/<clip>/*.png, so
        # the adapter builds the layout the subject documents rather than
        # changing the subject: a flat pair of directories of symlinks.
        stage = gt_root.parent / "uformer_stage"
        for name, src in (("groundtruth", gt_root), ("input", lq_root)):
            d = stage / name
            d.mkdir(parents=True, exist_ok=True)
            for f in sorted(src.rglob("*.png")):
                link = d / f"{f.parent.name}_{f.name}"
                if not link.exists():
                    link.symlink_to(f.resolve())

        self._n = spec.num_frames
        return self._cls(str(stage), img_options={"patch_size": 32})

    def sample(self, loader, index: int) -> Sample:
        clean, noisy, *_ = loader[index % len(loader)]
        gt = np.asarray(clean)
        lq = np.asarray(noisy)
        return Sample(gt=gt, lq=lq, frame_ids=None)

    def __len__(self) -> int:
        return 8
