"""Adapter for MMagic's ``BasicFramesDataset``.

MMagic (OpenMMLab, Apache-2.0, https://github.com/open-mmlab/mmagic) is a
different ecosystem from every other subject in this study: its own registry,
its own transform pipeline, its own dependency stack. Of 1287 Python files,
exactly one mentions BasicSR, in a kernel utility.

It is here for one reason. ``temporal_window`` had been exercised outside the
audited family on a single subject, because every other candidate with a
temporal dimension descends from the same two projects. This one samples a
window through a pipeline transform rather than inside the dataset, so it also
tests whether the gate's assumption -- that the loader decides the window --
survives a design that puts the decision somewhere else.

Environment reconstruction: MMagic asserts ``mmcv < 2.2.0`` and the audited
environment has exactly 2.2.0, so ``import mmagic`` raises before any code of
ours runs. That is a third mechanism by which a published artefact does not
execute as released, alongside a deleted dependency function and an untracked
generated file. The assertion is satisfied for the duration of the import and
restored afterwards; the subject's source is untouched, and the certificate
records that the environment was reconstructed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MMAGIC_ROOT = Path("/home/laura/02ImproveData/zmmagic")

for base in (HERE, *HERE.parents):
    for cand in (base, base / "treatment_identity"):
        if (cand / "treatment_identity" / "__init__.py").exists():
            sys.path.insert(0, str(cand))
            break

from treatment_identity.adapter import LoaderSpec, Sample  # noqa: E402


class MmagicFramesAdapter:
    """Expose MMagic's ``BasicFramesDataset`` to the treatment-identity gates."""

    name = "mmagic.BasicFramesDataset"
    project = "MMagic"
    url = "https://github.com/open-mmlab/mmagic"

    def __init__(self, root: Path = MMAGIC_ROOT):
        self.root = Path(root)
        if str(self.root) not in sys.path:
            sys.path.insert(0, str(self.root))
        self.reconstructed_dependency = None

        import mmcv
        real = mmcv.__version__
        try:
            import mmagic  # noqa: F401
        except AssertionError:
            # The version gate is a hard assert at import time. Satisfy it for
            # the import and put the real string back, so nothing else in the
            # process sees a version that is not installed.
            mmcv.__version__ = "2.1.0"
            self.reconstructed_dependency = f"mmcv version gate (real {real})"
            try:
                import mmagic  # noqa: F401
            finally:
                mmcv.__version__ = real

        # The pipeline is built from a registry, and the registry is empty
        # until the project populates it. Importing the dataset class alone
        # gets you a KeyError on the first transform: another convention that
        # is obvious once known and documented nowhere the caller looks.
        # Importing the transforms package is what populates the registry. The
        # project's own register_all_modules() also imports every model, which
        # pulls optional dependencies this audit does not need and does not
        # have -- so the narrow import is both sufficient and the honest one.
        import mmagic.datasets.transforms  # noqa: F401

        # Registering the transforms is not enough: the pipeline resolves names
        # through mmengine's registry, and MMagic's live in a child registry
        # under the scope "mmagic". Without the scope the lookup fails with a
        # KeyError naming a transform that is, in fact, registered. Two
        # conventions deep, and neither is where a caller would look.
        from mmengine.registry import DefaultScope
        if DefaultScope.get_current_instance() is None:
            DefaultScope.get_instance("ti-audit", scope_name="mmagic")

        from mmagic.datasets import BasicFramesDataset
        self._cls = BasicFramesDataset

    def build(self, spec: LoaderSpec):
        gt_root, lq_root = Path(spec.gt_root), Path(spec.lq_root or spec.gt_root)
        self._n = max(int(spec.num_frames), 1)

        pipeline = [
            dict(type="GenerateSegmentIndices", interval_list=[1]),
            dict(type="LoadImageFromFile", key="img", channel_order="rgb"),
            dict(type="LoadImageFromFile", key="gt", channel_order="rgb"),
            dict(type="PackInputs"),
        ]

        # An annotation file rather than directory scanning. The class accepts
        # either, and the scan produced an empty data_list against a layout its
        # own docstring describes -- the third convention this subject cost.
        # Case 1 of that docstring is "<clip> <length>", which is what the
        # temporal gate needs anyway: it names the clip and its length.
        clips = sorted(p for p in gt_root.iterdir() if p.is_dir())
        ann = gt_root.parent / "mmagic_ann.txt"
        ann.write_text("".join(
            f"{c.name} {len(sorted(c.glob('*.png')))}\n" for c in clips))

        return self._cls(
            ann_file=str(ann),
            data_root=str(gt_root.parent),
            data_prefix=dict(img=lq_root.name, gt=gt_root.name),
            pipeline=pipeline,
            depth=1,
            num_input_frames=self._n,
            search_key="gt",
        )

    def sample(self, loader, index: int) -> Sample:
        item = loader[index % len(loader)]
        inputs = item["inputs"]
        gt = item["data_samples"].gt_img.data

        def to_np(x):
            a = x.numpy() if hasattr(x, "numpy") else np.asarray(x)
            # MMagic packs (T, C, H, W); the gates read (T, H, W, C).
            if a.ndim == 4 and a.shape[1] in (1, 3):
                a = np.moveaxis(a, 1, -1)
            return a

        return Sample(gt=to_np(gt), lq=to_np(inputs), frame_ids=None)

    def __len__(self) -> int:
        return 8
