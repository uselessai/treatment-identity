"""Adapter for Zero-DCE's unpaired low-light loader.

Zero-DCE (CVPR 2020, Guo et al., https://github.com/Li-Chongyi/Zero-DCE) is a
zero-reference method: it trains on low-light images with **no ground truth at
all**, and its loader returns a single tensor rather than a pair.

That is why it is here. Every other subject in this study, and every pipeline in
the audited family, manufactures an input from a target. This one has no target,
so the contract that describes the others cannot be written for it, and what the
gates do with that is a finding about the contract's reach rather than about the
loader. A subject that the vocabulary does not fit is more informative than a
fourth subject that fits comfortably.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ZERODCE_ROOT = Path("/home/laura/02ImproveData/zZero-DCE/Zero-DCE_code")

for base in (HERE, *HERE.parents):
    for cand in (base, base / "treatment_identity"):
        if (cand / "treatment_identity" / "__init__.py").exists():
            sys.path.insert(0, str(cand))
            break

from treatment_identity.adapter import LoaderSpec, Sample  # noqa: E402


class ZeroDceAdapter:
    """Expose Zero-DCE's ``lowlight_loader`` to the treatment-identity gates."""

    name = "zerodce.lowlight_loader"
    project = "Zero-DCE"
    url = "https://github.com/Li-Chongyi/Zero-DCE"

    def __init__(self, root: Path = ZERODCE_ROOT):
        self.root = Path(root)
        if str(self.root) not in sys.path:
            sys.path.insert(0, str(self.root))
        from dataloader import lowlight_loader  # noqa: E402
        self._cls = lowlight_loader

    def build(self, spec: LoaderSpec):
        # The loader globs "*.jpg" from a path it concatenates without a
        # separator, so the argument has to end in one. It reads no ground
        # truth: there is nothing in its interface to point at a target.
        src = Path(spec.lq_root or spec.gt_root)
        stage = Path(spec.gt_root).parent / "zerodce_stage"
        stage.mkdir(parents=True, exist_ok=True)
        try:
            from PIL import Image
            for f in sorted(src.rglob("*.png")):
                out = stage / f"{f.parent.name}_{f.stem}.jpg"
                if not out.exists():
                    Image.open(f).convert("RGB").save(out, quality=95)
        except ImportError as exc:                            # pragma: no cover
            raise NotImplementedError(f"Pillow unavailable: {exc}") from exc
        return self._cls(str(stage) + "/")

    def sample(self, loader, index: int) -> Sample:
        item = loader[index % len(loader)]
        # A single tensor, and `gt=None` says so.
        #
        # This adapter used to report the same array as both input and target,
        # because `Sample.gt` was mandatory and there was no way to express an
        # absent one. The consequence was a gate passing for want of the thing
        # it asserts over: `target_transforms` compared zero observed
        # transformations of that fabricated target against a declared zero and
        # returned PASS, on the one subject in the study chosen precisely
        # because it has no target. `None` is the honest answer, and the
        # target-dependent gates now report N/A against it.
        a = np.asarray(item[0] if isinstance(item, (tuple, list)) else item)
        return Sample(gt=None, lq=a, frame_ids=None)

    def __len__(self) -> int:
        return 8
