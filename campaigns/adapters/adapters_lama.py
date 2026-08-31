"""Adapter for LaMa's ``InpaintingTrainDataset`` --- a further portability subject.

LaMa (WACV 2022, Suvorov et al., https://github.com/advimman/lama) is an
inpainting model, and its training loader is structurally unlike the paired
restoration loaders of the other subjects: it delivers an image and a
runtime-generated *mask*, not a low-quality/ground-truth pair. The treatment it
manufactures is the mask, and the subject ships several mask generators it
names and configures separately.

That difference is the reason to include it. It tests whether the contract
vocabulary reaches a pipeline whose declared treatment is a mask rather than a
degradation, and the certificate records which gates found something to assert.

The adapter delivers the image as the target and the masked image as the input.
The loader returns the two parts rather than the product, so this composition is
adapter-side work and is declared here rather than presented as the subject's
own output.

Adaptation effort is a response variable of the study. Everything below the
header is that cost, and the subject is not modified to make a gate pass.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LAMA_ROOT = Path("/home/laura/02ImproveData/zLaMa")

for base in (HERE, *HERE.parents):
    for cand in (base, base / "treatment_identity"):
        if (cand / "treatment_identity" / "__init__.py").exists():
            sys.path.insert(0, str(cand))
            break

from treatment_identity.adapter import CallRecorder, LoaderSpec, Sample  # noqa: E402

#: The mask generators the subject ships and names separately.
MASK_GENERATORS = ("RandomIrregularMaskGenerator", "RandomRectangleMaskGenerator")


class LamaAdapter:
    """Expose LaMa's inpainting training loader to the gates."""

    name = "lama.InpaintingTrainDataset"
    project = "LaMa"
    url = "https://github.com/advimman/lama"

    #: Set by :meth:`_stub_undeclared_import`. Reported in the certificate.
    environment_note: str | None = None

    def __init__(self, root: Path = LAMA_ROOT, *,
                 generator: str = "RandomIrregularMaskGenerator"):
        self.root = Path(root)
        if str(self.root) not in sys.path:
            sys.path.insert(0, str(self.root))
        self._stub_absent_import()
        from saicinpainting.training.data.datasets import InpaintingTrainDataset
        from saicinpainting.training.data import masks as mask_mod
        self._cls = InpaintingTrainDataset
        self._masks = mask_mod
        self.generator = generator
        self.recorder = CallRecorder()
        self._instrument()

    @classmethod
    def _stub_absent_import(cls) -> None:
        """Satisfy a module-level import this environment does not provide.

        ``datasets.py`` imports ``webdataset`` at module level and uses it on
        one line, inside ``InpaintingTrainWebDataset``, a class this audit does
        not exercise. The subject declares the dependency in its own
        ``requirements.txt`` and pins it in ``conda_env.yml``, so its absence
        here is our omission and not the subject's --- the distinction Study 3
        already draws for one other subject.

        It is stubbed in memory rather than installed, because the environment
        this study runs in is itself a pinned artefact of the article and
        adding a package to it would invalidate the claim that it rebuilds
        clean. The stub is reached by no code path this audit executes; were
        the WebDataset class audited, the dependency would have to be installed
        instead.
        """
        import types

        if "webdataset" in sys.modules:
            return
        try:
            import webdataset                                # noqa: F401
            return
        except ImportError:
            pass
        stub = types.ModuleType("webdataset")

        def _absent(*a, **kw):
            raise RuntimeError(
                "webdataset is stubbed for this audit and was not installed; "
                "InpaintingTrainWebDataset is out of scope")

        stub.Dataset = _absent                              # type: ignore[attr-defined]
        sys.modules["webdataset"] = stub
        cls.environment_note = (
            "webdataset stubbed in memory: declared by the subject's own "
            "requirements.txt and not installed here, used only by a class "
            "this audit does not exercise")

    def _instrument(self) -> None:
        """Record which mask primitive runs, without changing what it does."""
        rec = self.recorder
        for fn_name, label in (("make_random_irregular_mask", "irregular"),
                               ("make_random_rectangle_mask", "rectangle")):
            original = getattr(self._masks, fn_name, None)
            if original is None or getattr(original, "_ti_wrapped", False):
                continue

            def make(orig, lab):
                def wrapper(*a, **kw):
                    rec.record(lab)
                    return orig(*a, **kw)
                wrapper._ti_wrapped = True            # type: ignore[attr-defined]
                return wrapper

            setattr(self._masks, fn_name, make(original, label))

    def build(self, spec: LoaderSpec):
        # The subject globs '*.jpg' recursively; the fixture writes PNG, so the
        # adapter re-encodes rather than asking the subject to accept PNG.
        import cv2

        gt_root = Path(spec.gt_root)
        stage = gt_root.parent / "lama_stage"
        stage.mkdir(parents=True, exist_ok=True)
        for f in sorted(gt_root.rglob("*.png")):
            out = stage / f"{f.parent.name}_{f.stem}.jpg"
            if not out.exists():
                cv2.imwrite(str(out), cv2.imread(str(f)), [cv2.IMWRITE_JPEG_QUALITY, 100])

        gen_cls = getattr(self._masks, self.generator)
        gen = gen_cls()

        class _Identity:
            def __call__(self, image=None, **kw):
                return {"image": image}

        self._ds = self._cls(str(stage), mask_generator=gen, transform=_Identity())
        return self._ds

    def sample(self, loader, index: int) -> Sample:
        item = loader[index % len(loader)]
        img = np.asarray(item["image"], dtype=np.float64)     # (C, H, W)
        mask = np.asarray(item["mask"], dtype=np.float64)     # (1, H, W)
        gt = np.transpose(img, (1, 2, 0))
        lq = np.transpose(img * (1.0 - mask), (1, 2, 0))
        return Sample(gt=gt, lq=lq, frame_ids=None)

    def __len__(self) -> int:
        return len(self._ds)
