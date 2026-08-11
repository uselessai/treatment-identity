"""Adapter for MPRNet's paired denoising loader.

MPRNet (CVPR 2021, Zamir et al., https://github.com/swz30/MPRNet) is a
multi-stage restoration network. Its training loader, like Uformer's, reads a
directory holding ``groundtruth/`` and ``input/`` and is therefore a
pre-rendered pipeline.

Two subjects with the same input convention are not redundant here: they are
different codebases from different groups that arrived at the same convention,
and the gates should say so. If one passes and the other fails, the convention
is not what decides it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MPRNET_ROOT = Path("/home/laura/02ImproveData/zMPRNet")

for base in (HERE, *HERE.parents):
    for cand in (base, base / "treatment_identity"):
        if (cand / "treatment_identity" / "__init__.py").exists():
            sys.path.insert(0, str(cand))
            break

from treatment_identity.adapter import LoaderSpec, Sample  # noqa: E402


class MprnetAdapter:
    """Expose MPRNet's ``DataLoaderTrain`` to the treatment-identity gates."""

    name = "mprnet.DataLoaderTrain"
    project = "MPRNet"
    url = "https://github.com/swz30/MPRNet"

    def __init__(self, root: Path = MPRNET_ROOT):
        self.root = Path(root)
        task = self.root / "Denoising"
        if str(task) not in sys.path:
            sys.path.insert(0, str(task))
        from dataset_RGB import DataLoaderTrain  # noqa: E402
        self._cls = DataLoaderTrain

    def build(self, spec: LoaderSpec):
        gt_root, lq_root = Path(spec.gt_root), Path(spec.lq_root or spec.gt_root)
        stage = gt_root.parent / "mprnet_stage"
        # MPRNet reads 'input' and 'target'. Uformer reads 'input' and
        # 'groundtruth'. The two conventions look identical until one of them
        # raises, and neither repository documents its own: this cost one run,
        # and it is the kind of cost this study measures.
        for name, src in (("target", gt_root), ("input", lq_root)):
            d = stage / name
            d.mkdir(parents=True, exist_ok=True)
            for f in sorted(src.rglob("*.png")):
                link = d / f"{f.parent.name}_{f.name}"
                if not link.exists():
                    link.symlink_to(f.resolve())
        return self._cls(str(stage), img_options={"patch_size": 32})

    def sample(self, loader, index: int) -> Sample:
        clean, noisy, *_ = loader[index % len(loader)]
        return Sample(gt=np.asarray(clean), lq=np.asarray(noisy), frame_ids=None)

    def __len__(self) -> int:
        return 8
