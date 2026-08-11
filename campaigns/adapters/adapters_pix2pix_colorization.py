"""Adapter for pix2pix's ``ColorizationDataset`` --- the online-degradation subject.

Chosen to close the one gap the portability study could not close by counting.
Six of the seven subjects that preceded it read a pre-rendered input from disk;
only KAIR's blind super-resolution loader manufactures its input online from the
target, which is what all four audited codebases do and what every divergence
the article reports is a defect of. One external example of the class under
study is not enough to say the gates travel *to that class*, and adding another
pre-rendered loader would not have helped.

Finding one is harder than it sounds, and the reason is a result in itself.
Online-degradation training loaders in restoration have largely converged on two
ancestors, BasicSR and KAIR, and the rule that keeps this study honest --- no
candidate that vendors a codebase already represented here --- removes most of
their descendants. Restormer and DDColor both vendor BasicSR. LaMa composes a
randomised mask pipeline and would have been ideal, but its released form needs
``webdataset``, ``omegaconf`` and an ``IAAAffine2`` that current albumentations
no longer provides, so auditing it would have produced a fourth
unexecutable-as-published subject rather than a subject of the class we lack.

``ColorizationDataset`` is the one that fits. It belongs to pix2pix
(Isola et al., https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix), which
shares no ancestry with anything else in this study or in the audited family,
and it manufactures its input the way the audited family does: it reads one RGB
image, converts it to Lab, and returns the L channel as the model's input and
the ab channels as the learning target. The degraded input is never on disk. It
is also a different task --- colourisation --- which the article names as a
setting the contract should reach and, until this subject, had not run.

The degradation is deterministic rather than randomised, so ``operator_trace``
has no sequence to observe here. That gate still rests on one external subject,
and the article says so rather than counting this one twice.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
PIX2PIX_ROOT = Path("/home/laura/02ImproveData/zpix2pix")

for base in (HERE, *HERE.parents):
    for cand in (base, base / "treatment_identity"):
        if (cand / "treatment_identity" / "__init__.py").exists():
            sys.path.insert(0, str(cand))
            break

from treatment_identity.adapter import LoaderSpec, Sample  # noqa: E402


class Pix2pixColorizationAdapter:
    """Expose pix2pix's ``ColorizationDataset`` to the treatment-identity gates."""

    name = "pix2pix.ColorizationDataset"
    project = "pytorch-CycleGAN-and-pix2pix"
    url = "https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix"

    def __init__(self, root: Path = PIX2PIX_ROOT):
        self.root = Path(root)
        if str(self.root) not in sys.path:
            sys.path.insert(0, str(self.root))
        # The subject's package is `data`, a name generic enough that another
        # subject's module could already own it in this process. The campaign
        # imports every adapter into one interpreter, so drop any cached `data`
        # and let this subject's root win for the duration of the import --- the
        # same collision two subjects already produce over `utils`, and an
        # adaptation cost that only appears when several subjects are audited
        # together.
        for stale in [m for m in list(sys.modules)
                      if m == "data" or m.startswith("data.")]:
            del sys.modules[stale]
        from data.colorization_dataset import ColorizationDataset  # noqa: E402
        self._cls = ColorizationDataset

    def build(self, spec: LoaderSpec):
        # The subject reads ONE directory of RGB sources under
        # <dataroot>/<phase>, and derives the input from them. It is pointed at
        # the fixture's target frames, because those are what it treats as
        # source material; the fixture's pre-rendered LQ directory is what the
        # contract says should reach the model, and observing that it does not
        # is the point of the precomputed_input gate for this subject.
        gt_root = Path(spec.gt_root)
        stage = gt_root.parent / "pix2pix_stage"
        phase = "train" if spec.train else "test"
        d = stage / phase
        d.mkdir(parents=True, exist_ok=True)
        for f in sorted(gt_root.rglob("*.png")):
            link = d / f"{f.parent.name}_{f.name}"
            if not link.exists():
                link.symlink_to(f.resolve())

        # `opt` is the subject's own configuration object. Building one by hand
        # is convention translation, not modification: these are the values its
        # own training script sets for this dataset, and `input_nc=1`,
        # `output_nc=2` and `direction='AtoB'` are asserted by the subject
        # itself in __init__.
        opt = SimpleNamespace(
            dataroot=str(stage), phase=phase, max_dataset_size=float("inf"),
            input_nc=1, output_nc=2, direction="AtoB",
            preprocess="resize_and_crop", load_size=32, crop_size=32,
            no_flip=True, serial_batches=True, num_threads=0, batch_size=1,
            isTrain=spec.train,
        )
        return self._cls(opt)

    def sample(self, loader, index: int) -> Sample:
        item = loader[index % len(loader)]
        # A is the L channel the model receives; B is the ab target. Both come
        # back as (C, H, W) torch tensors, and the checks normalise orientation
        # themselves. The delivered input is reported as it is delivered: this
        # subject's model never sees the fixture's pre-rendered LQ.
        lq = np.asarray(item["A"])
        gt = np.asarray(item["B"])
        return Sample(gt=gt, lq=lq, frame_ids=None)

    def __len__(self) -> int:
        return 8
