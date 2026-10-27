"""Adapter for LaMa's ``InpaintingTrainDataset`` --- a further portability subject.

LaMa (WACV 2022, Suvorov et al., https://github.com/advimman/lama) is an
inpainting model, and its training loader is structurally unlike the paired
restoration loaders of the other subjects: it delivers an image and a
runtime-generated *mask*, not a low-quality/ground-truth pair. The treatment it
manufactures is the mask, drawn per item from a mixture of mask kinds that the
data configuration names and weights separately.

The loader is built the way the trainer builds it,
``make_default_train_dataloader(**config.data.train)``
(``saicinpainting/training/trainers/base.py:130``), from the ``train`` section
of a data configuration the repository ships, read from disk rather than
transcribed. That configuration is ``abl-04-256-mh-dist-celeba.yaml``, the data
of ``lama-fourier-celeba.yaml:28`` and ``big-lama-celeba.yaml:49``: the
CelebA-HQ experiments. The Places configurations (``abl-04-256-mh-dist.yaml``)
select ``transform_variant: distortions``, whose ``IAAAffine2`` and
``IAAPerspective2`` subclass ``albumentations.DualIAATransform``; the subject
pins ``albumentations==0.5.2`` (``requirements.txt:12``) and that class no
longer exists in the albumentations of this environment, so those
configurations cannot execute as released here and are not the ones audited.

The image is delivered as the target and ``image * (1 - mask)`` as the input.
The loader returns the two parts; the product is composed by the trainer
(``saicinpainting/training/trainers/default.py:59``), which then also
concatenates the mask as a fourth channel (``:67-68``). The adapter reproduces
the first step and not the second, and carries the mask in ``extra``.

Adaptation effort is a response variable of the study. Everything below the
header is that cost, and the subject is not modified to make a gate pass.
"""

from __future__ import annotations

import contextlib
import importlib.machinery
import importlib.util
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LAMA_ROOT = Path("/home/laura/02ImproveData/zLaMa")
DATA_CONFIG = "configs/training/data/abl-04-256-mh-dist-celeba.yaml"

for base in (HERE, *HERE.parents):
    for cand in (base, base / "treatment_identity"):
        if (cand / "treatment_identity" / "__init__.py").exists():
            sys.path.insert(0, str(cand))
            break

from treatment_identity.adapter import (CallRecorder, GeneratorReached,  # noqa: E402
                                        LoaderSpec, Sample)

#: Mask primitive -> the configuration key that weights it in the mixture
#: (``abl-04-256-mh-dist-celeba.yaml:11`` and ``:19``).
MASK_KINDS = {"irregular": "irregular_proba", "rectangle": "box_proba"}


class LamaAdapter:
    """Expose LaMa's inpainting training loader to the gates."""

    name = "lama.InpaintingTrainDataset"
    project = "LaMa"
    url = "https://github.com/advimman/lama"

    #: Set by :meth:`_stub_absent_imports`. Reported in the certificate.
    environment_note: str | None = None

    def __init__(self, root: Path = LAMA_ROOT):
        import yaml

        self.root = Path(root)
        if str(self.root) not in sys.path:
            sys.path.insert(0, str(self.root))
        self._stub_absent_imports()
        from saicinpainting.training.data import datasets, masks
        self._datasets, self._masks = datasets, masks
        self.train_cfg = yaml.safe_load((self.root / DATA_CONFIG).read_text())["train"]
        self.recorder = CallRecorder()
        self.last_branch_reached = False
        self._instrument()

    @classmethod
    def _stub_absent_imports(cls) -> None:
        """Satisfy module-level imports on paths this audit does not execute.

        Importing ``datasets.py`` pulls in names that are bound at import time
        and used only elsewhere: ``webdataset`` (``InpaintingTrainWebDataset``,
        ``datasets.py:50``), ``omegaconf`` (the web and validation branches,
        ``:242``, ``:250``), ``imgaug`` and ``albumentations.DualIAATransform``
        (``aug.py:1-2``, the ``distortions*`` variants only), ``easydict``
        (``evaluation/utils.py:4``, reached through the evaluation package
        ``__init__``) and ``pytorch_lightning`` (``utils.py:12``). The subject
        pins all of them; they are absent here, or removed from the installed
        version. Each placeholder raises if it is ever reached, so a stub cannot
        stand in for behaviour; it only lets the module import.
        """
        class _Absent:
            def __init__(self, *a, **kw):
                raise RuntimeError("stubbed for this audit; path out of scope")

        stubbed = []
        for name, attrs in (("webdataset", ("Dataset",)),
                            ("omegaconf", ("open_dict", "OmegaConf")),
                            ("imgaug", ()), ("imgaug.augmenters", ()),
                            ("easydict", ("EasyDict",)),
                            ("pytorch_lightning", ("seed_everything",))):
            try:
                __import__(name)
            except ImportError:
                # A spec, because torch._dynamo calls find_spec on modules
                # already in sys.modules and rejects one without it.
                mod = importlib.util.module_from_spec(
                    importlib.machinery.ModuleSpec(name, None))
                for attr in attrs:
                    setattr(mod, attr, _Absent)
                sys.modules[name] = mod
                if "." in name:
                    parent, child = name.rsplit(".", 1)
                    setattr(sys.modules[parent], child, mod)
                stubbed.append(name)
        import albumentations
        for attr in ("DualIAATransform", "to_tuple"):
            if not hasattr(albumentations, attr):
                setattr(albumentations, attr, _Absent)
                stubbed.append(f"albumentations.{attr}")
        if stubbed:
            cls.environment_note = (
                "stubbed in memory, raising if reached: " + ", ".join(stubbed)
                + "; imported at module level, used only by the web dataset, "
                  "the validation builder, the distortions* augmentations and "
                  "evaluation utilities, none of which this audit executes")

    def _instrument(self) -> None:
        """Record which mask primitive runs, without changing what it does.

        The generators call these module functions (``masks.py:67``, ``:99``).
        An earlier adapter's wrapper is replaced, not wrapped again, so the
        counter always belongs to this instance.
        """
        rec = self.recorder
        for fn_name, label in (("make_random_irregular_mask", "irregular"),
                               ("make_random_rectangle_mask", "rectangle")):
            current = getattr(self._masks, fn_name)
            original = getattr(current, "_ti_original", current)

            def make(orig, lab):
                def wrapper(*a, **kw):
                    rec.record(lab)
                    return orig(*a, **kw)
                wrapper._ti_original = orig            # type: ignore[attr-defined]
                return wrapper

            setattr(self._masks, fn_name, make(original, label))

    @contextlib.contextmanager
    def disable_generator(self):
        """Make the online mask generator raise ``GeneratorReached``.

        The builder resolves it through the module-level ``get_mask_generator``
        (``datasets.py:210``). Without this sentinel the tensor test alone can
        be misled: masked pixels are zero, so a mostly masked flat image reads
        as the bimodal checkerboard.
        """
        def sentinel(*a, **kw):
            raise GeneratorReached("LaMa's mask generator was reached")

        original = self._datasets.get_mask_generator
        self._datasets.get_mask_generator = lambda **kw: sentinel
        try:
            yield ["get_mask_generator"]
        finally:
            self._datasets.get_mask_generator = original

    def build(self, spec: LoaderSpec, **override):
        # The subject globs '*.jpg' (datasets.py:27). cv2.imread decodes by
        # content, so the PNG fixture is linked under a .jpg name and read
        # losslessly; re-encoding it as JPEG would put the adapter's own
        # compression into the delivered tensor.
        gt_root = Path(spec.gt_root)
        stage = gt_root.parent / "lama_stage"
        stage.mkdir(parents=True, exist_ok=True)
        for f in sorted(gt_root.rglob("*.png")):
            link = stage / f"{f.parent.name}_{f.stem}.jpg"
            if not link.exists():
                link.symlink_to(f.resolve())
        # dataloader_kwargs holds unresolved ${data.*} interpolations, and the
        # gates read the dataset item, not the collated batch.
        train = dict(self.train_cfg, indir=str(stage), dataloader_kwargs=None)
        train.update(override)
        self._ds = self._datasets.make_default_train_dataloader(**train).dataset
        return self._ds

    def sample(self, loader, index: int) -> Sample:
        item = loader[index % len(loader)]
        img = np.asarray(item["image"])                       # (C, H, W)
        mask = np.asarray(item["mask"])                       # (1, H, W)
        return Sample(gt=np.transpose(img, (1, 2, 0)),
                      lq=np.transpose(img * (1 - mask), (1, 2, 0)),
                      frame_ids=None, extra={"mask": mask[0]})

    def __len__(self) -> int:
        return len(self._ds)

    # -- settings sweep -----------------------------------------------------
    #: The configuration weights each mask kind separately; switching one off
    #: leaves the other as the only kind drawn.
    def settings(self):
        return {key: (1, 0) for key in MASK_KINDS.values()}

    def deliver_under(self, workdir: Path, name: str, value):
        """Deliver item 0 with one mixture weight set, everything else fixed.

        Also returns whether the kind that weight guards was drawn: the mixture
        is sampled per item, so an unchanged item proves nothing without it.
        """
        gt_root = Path(workdir) / "fixture" / "GT"
        if not gt_root.exists():
            from treatment_identity.fixtures import make_scale_grid
            make_scale_grid(gt_root, n_frames=8, shape=(256, 256))
        kwargs = dict(self.train_cfg["mask_gen_kwargs"], **{name: value})
        ds = self.build(LoaderSpec(gt_root=gt_root), mask_gen_kwargs=kwargs)
        guarded = next(k for k, v in MASK_KINDS.items() if v == name)
        self.recorder.reset()
        try:
            lq = self.sample(ds, 0).lq
        finally:
            self.last_branch_reached = guarded in self.recorder.calls
        return lq, self.last_branch_reached
