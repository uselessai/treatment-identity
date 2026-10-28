"""Adapter for ProPainter's ``TrainDataset`` --- a video-inpainting subject.

ProPainter (ICCV 2023, Zhou et al., https://github.com/sczhou/ProPainter) is a
video inpainting model. Its training loader (``core/dataset.py``, built by
``core/trainer.py`` from ``configs/train_propainter.json``) is a video loader
whose treatment is a *moving mask* generated online for every item, and whose
temporal window is not one contiguous run: ``num_local_frames`` consecutive
frames, randomly reversed, followed by ``num_ref_frames`` reference frames
sampled from the rest of the clip.

That is the reason to include it. It is the second inpainting subject and the
only video one, so it asks whether the window and mask vocabulary reach a loader
that composes both, at a full-frame resize rather than a crop.

The loader returns frames in ``[-1, 1]`` and the mask, and the trainer composes
the input (``masked_frames = frames * (1 - masks)``, ``core/trainer.py:363``).
The adapter performs that same product, so the composition is adapter-side work
and is declared here rather than presented as the loader's own output.

Adaptation effort is a response variable of the study. Everything below the
header is that cost, and the subject is not modified to make a gate pass.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROPAINTER_ROOT = Path("/home/laura/02ImproveData/zProPainter")

for base in (HERE, *HERE.parents):
    for cand in (base, base / "treatment_identity"):
        if (cand / "treatment_identity" / "__init__.py").exists():
            sys.path.insert(0, str(cand))
            break

from treatment_identity.adapter import CallRecorder, LoaderSpec, Sample  # noqa: E402

#: ``train_data_loader`` of ``configs/train_propainter.json:4-13``, minus the
#: paths. The adapter runs the subject's own configuration rather than mapping
#: ``spec.num_frames`` onto it: a single requested length has no faithful split
#: into local and reference frames.
TRAIN_LOADER = {"w": 432, "h": 240, "num_local_frames": 10,
                "num_ref_frames": 6, "load_flow": 0}


class ProPainterAdapter:
    """Expose ProPainter's ``TrainDataset`` and its mask/flip order to the gates."""

    name = "propainter.TrainDataset"
    project = "ProPainter"
    url = "https://github.com/sczhou/ProPainter"

    #: Set by :meth:`_restore_removed_method`. Reported in the certificate.
    environment_note: str | None = None

    def __init__(self, root: Path = PROPAINTER_ROOT):
        self.root = Path(root)
        # ``core`` and ``utils`` are top-level names in this repository and in
        # other subjects (KAIR has a ``utils``); the harness isolates only
        # ``basicsr``, so a cached copy from an earlier subject is dropped here.
        for name in [m for m in list(sys.modules)
                     if m.split(".")[0] in ("core", "utils")]:
            del sys.modules[name]
        while str(self.root) in sys.path:
            sys.path.remove(str(self.root))
        sys.path.insert(0, str(self.root))
        self._restore_removed_method()
        from core import dataset as dataset_mod
        self._mod = dataset_mod
        self.recorder = CallRecorder()
        self._instrument()

    @classmethod
    def _restore_removed_method(cls) -> None:
        """Reinstate ``FigureCanvasAgg.tostring_rgb``, removed in Matplotlib 3.10.

        ``get_random_shape`` (``core/utils.py:296``) rasterises every mask
        outline through it, so on a current environment no training item can be
        built. ``requirements.txt`` lists ``matplotlib`` without a version. The
        replacement drops the alpha channel of ``buffer_rgba()``, which is what
        the removed method returned on an opaque Agg canvas.
        """
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        if hasattr(FigureCanvasAgg, "tostring_rgb"):
            return
        FigureCanvasAgg.tostring_rgb = (                    # type: ignore[attr-defined]
            lambda self: np.asarray(self.buffer_rgba())[..., :3].tobytes())
        cls.environment_note = (
            "matplotlib FigureCanvasAgg.tostring_rgb reinstated in memory over "
            "buffer_rgba(); removed in Matplotlib 3.10, unpinned by the subject, "
            "and called by its mask generator on every item")

    def _instrument(self) -> None:
        """Record the mask generator and the flip, without changing either.

        Both are names bound in ``core.dataset`` and called once per item, in
        that order (``core/dataset.py:73`` and ``:123``). The random reversal of
        the local frames (``:109-112``) is inline list code and cannot be
        observed this way; the trace does not claim it. The module was imported
        afresh in ``__init__``, so these wrap the originals, never a wrapper.
        """
        mod, rec = self._mod, self.recorder
        masks, flip = (mod.create_random_shape_with_random_motion,
                       mod.GroupRandomHorizontalFlip)

        def mask_wrapper(*a, **kw):
            rec.record("mask")
            return masks(*a, **kw)

        class FlipWrapper(flip):
            def __call__(self, *a, **kw):
                rec.record("flip")
                return super().__call__(*a, **kw)

        mod.create_random_shape_with_random_motion = mask_wrapper
        mod.GroupRandomHorizontalFlip = FlipWrapper

    def build(self, spec: LoaderSpec):
        gt_root = Path(spec.gt_root)
        stage = gt_root.parent / "propainter_stage"
        stage.mkdir(parents=True, exist_ok=True)
        clips = sorted(p.name for p in gt_root.iterdir() if p.is_dir())
        (stage / "train.json").write_text(json.dumps(
            {c: len(list((gt_root / c).iterdir())) for c in clips}))
        # The split file is read from './datasets/<name>/train.json', relative
        # to the working directory (core/dataset.py:32). An absolute 'name'
        # makes os.path.join discard the relative prefix, so the subject can be
        # built from anywhere without a chdir.
        self._ds = self._mod.TrainDataset(dict(
            TRAIN_LOADER, name=str(stage.resolve()),
            video_root=str(gt_root), flow_root=""))
        # Clips of <= 16 frames are dropped silently (core/dataset.py:45), and
        # the dataset is then empty without saying so. Indexing it here lets the
        # subject's own IndexError (:71) report that at build time, before any
        # random draw, instead of every later sample failing one by one.
        if not len(self._ds):
            self._ds[0]
        return self._ds

    def sample(self, loader, index: int) -> Sample:
        frames, masks, *_ = loader[index % len(loader)]
        frames = frames.numpy()                              # (T, 3, H, W)
        masks = masks.numpy()                                # (T, 1, H, W), 1 = hole
        gt = np.transpose(frames, (0, 2, 3, 1))
        lq = np.transpose(frames * (1.0 - masks), (0, 2, 3, 1))
        return Sample(gt=gt, lq=lq, frame_ids=None)

    def __len__(self) -> int:
        return len(self._ds)
