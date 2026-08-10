"""Adapter for a single-frame low-light loader --- Campaign P subject 3.

``HybridLowLightDataset`` from an unrelated low-light-enhancement project. It is
included for a reason the other two subjects cannot supply: it is a
**single-image** pipeline with a **fixed** augmentation order and an explicit
target-sharpening probability. Running the same gates against it shows which
of them are generic and which need the subject to expose something --- the
substance of RQ4.

Expected shape of the result, stated before running so that it is a prediction
rather than a description:

  * ``temporal_window``   --- no window exists; the gate should not claim a pass.
  * ``operator_trace``    --- the subject declares a fixed order and applies one.
  * ``target_transforms`` --- the subject transforms its target under a
    probability, so the declared count is a distribution, not an integer. This
    is a case the gate's contract does not currently express, and reporting it
    is more useful than forcing it into PASS or FAIL.
  * ``precomputed_input`` --- the loader always reads a stored low-light image;
    there is no online generator to bypass.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

LLIE_DATASET = Path("/home/laura/02ImproveData/zLLIE-arch415/dataset.py")

from treatment_identity.adapter import CallRecorder, LoaderSpec, Sample  # noqa: E402


class LlieSingleFrameAdapter:
    """Expose ``HybridLowLightDataset`` to the gates."""

    name = "arch415.HybridLowLightDataset"
    project = "NTIRE2026-LLIE arch415"
    url = "local"

    def __init__(self, dataset_py: Path = LLIE_DATASET):
        self.dataset_py = Path(dataset_py)
        spec = importlib.util.spec_from_file_location("llie_dataset", self.dataset_py)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["llie_dataset"] = mod
        spec.loader.exec_module(mod)                          # type: ignore[union-attr]
        self._cls = mod.HybridLowLightDataset
        self._mod = mod
        self.recorder = CallRecorder()
        self._instrument()

    def _instrument(self) -> None:
        """Record the target-sharpening call.

        The subject applies an unsharp mask to its target under a probability,
        through a static method. Wrapping it is the whole instrumentation cost
        for this subject: three lines.
        """
        cls = self._cls
        rec = self.recorder
        original = cls._unsharp_mask
        if getattr(original, "_ti_wrapped", False):
            return

        def wrapper(*a, **kw):
            rec.record("sharpen")
            return original(*a, **kw)

        wrapper._ti_wrapped = True                            # type: ignore[attr-defined]
        cls._unsharp_mask = staticmethod(wrapper)

    # -- adapter protocol ---------------------------------------------------
    def build(self, spec: LoaderSpec):
        gt_root = Path(spec.gt_root)
        lq_root = Path(spec.lq_root) if spec.lq_root else gt_root
        gt_clips = sorted(p for p in gt_root.iterdir() if p.is_dir())
        lq_clips = sorted(p for p in lq_root.iterdir() if p.is_dir())
        gt_dir = gt_clips[0] if gt_clips else gt_root
        lq_dir = lq_clips[0] if lq_clips else lq_root

        gts = sorted(gt_dir.glob("*.png"))
        lqs = sorted(lq_dir.glob("*.png"))
        samples = [(str(l), str(g), g.stem) for l, g in zip(lqs, gts)]
        if not samples:
            raise NotImplementedError("fixture produced no image pairs")

        self._ds = self._cls(
            samples,
            crop_size=32,
            is_train=True,
            # The declared treatment: photometric augmentation off, target
            # sharpening always on. Both are claims the gates try to falsify.
            low_aug_prob=0.0,
            gt_sharpen_prob=1.0,
            gt_sharpen_amount=0.2,
        )
        return self._ds

    def sample(self, loader, index: int) -> Sample:
        item = loader[index % len(loader)]
        low, gt = _unpack(item)
        return Sample(lq=_to_numpy(low)[None, ...],
                      gt=_to_numpy(gt)[None, ...], frame_ids=None)

    def __len__(self) -> int:
        return len(self._ds)


def _unpack(item):
    if isinstance(item, dict):
        keys = list(item)
        low = item.get("low", item.get("lq", item[keys[0]]))
        gt = item.get("gt", item.get("high", item[keys[1]]))
        return low, gt
    return item[0], item[1]


def _to_numpy(x) -> np.ndarray:
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
