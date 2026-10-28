"""Adapter for E2FGVI's ``TrainDataset`` --- a video-inpainting subject.

E2FGVI (CVPR 2022, Li et al., https://github.com/MCG-NKU/E2FGVI) is a
flow-guided video inpainting model from a group unrelated to the audited
family. Its training loader (``core/dataset.py:17``, built by
``core/trainer.py:30`` from ``configs/train_e2fgvi.json``) is unlike the other
video subjects in two ways, and those are the reasons to include it:

* its window is not one block of consecutive frames. Each item is five
  consecutive *local* frames at a random pivot followed by three *reference*
  frames drawn at random from the rest of the clip (``_sample_index``,
  ``core/dataset.py:43-50``);
* its treatment is a runtime-generated mask, as in LaMa, but for a whole clip
  and either fixed or moving (``core/utils.py:186-224``).

The loader returns the frames and the masks, not the masked frames. The input
the network receives is composed one step later, by the trainer
(``core/trainer.py:304``); the adapter reproduces that one line so the input
stream is the subject's own composition rather than one we chose.

Adaptation effort is a response variable of the study. Everything below the
header is that cost, and the subject is not modified to make a gate pass.
"""

from __future__ import annotations

import contextlib
import importlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
E2FGVI_ROOT = Path("/home/laura/02ImproveData/zE2FGVI")

for base in (HERE, *HERE.parents):
    for cand in (base, base / "treatment_identity"):
        if (cand / "treatment_identity" / "__init__.py").exists():
            sys.path.insert(0, str(cand))
            break

from treatment_identity.adapter import (CallRecorder, GeneratorReached,  # noqa: E402
                                        LoaderSpec, Sample)

#: The subject's own loader options, configs/train_e2fgvi.json:7-10 (identical
#: in train_e2fgvi_hq.json:7-10). ``spec.num_frames`` is not mapped onto them:
#: the window is two fields here, and the contract declares their sum.
LOADER_OPTIONS = {"w": 432, "h": 240, "num_local_frames": 5, "num_ref_frames": 3}


class E2fgviAdapter:
    """Expose E2FGVI's ``TrainDataset`` and its mask generator to the gates."""

    name = "e2fgvi.TrainDataset"
    project = "E2FGVI"
    url = "https://github.com/MCG-NKU/E2FGVI"

    #: Set by :meth:`_restore_removed_method`. Reported in the certificate.
    environment_note: str | None = None

    def __init__(self, root: Path = E2FGVI_ROOT):
        self.root = Path(root)
        self._restore_removed_method()
        # ``core`` has no __init__.py, and ProPainter, which descends from this
        # project, ships a ``core`` of its own. Whichever was imported first in
        # a process would answer ``core.dataset``, so the name is reclaimed.
        for m in [m for m in sys.modules if m == "core" or m.startswith("core.")]:
            del sys.modules[m]
        while str(self.root) in sys.path:
            sys.path.remove(str(self.root))
        sys.path.insert(0, str(self.root))
        self._mod = importlib.import_module("core.dataset")
        self._utils = importlib.import_module("core.utils")
        self._cls = self._mod.TrainDataset
        self.recorder = CallRecorder()
        self.claimed_ids: list[int] | None = None
        self._instrument()

    @classmethod
    def _restore_removed_method(cls) -> None:
        """Reinstate ``FigureCanvasAgg.tostring_rgb``, removed in Matplotlib 3.10.

        The mask generator draws its random shape with Matplotlib and reads the
        canvas back with it (``core/utils.py:255``), so on a current
        environment every item raises before a frame is read. The subject pins
        matplotlib 3.5.1 (``environment.yml:72``), so the absence is ours. The
        replacement is the one Matplotlib's deprecation notice points to,
        ``buffer_rgba`` with the alpha channel dropped; it is reached on every
        item and is therefore reconstruction, not a stub.
        """
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        # ProPainter's adapter reinstates the same method; a copy that is not
        # Matplotlib's own is still a reconstruction and is still reported.
        own = getattr(FigureCanvasAgg, "tostring_rgb", None)
        if own is not None and own.__module__.startswith("matplotlib"):
            return
        if own is None:
            FigureCanvasAgg.tostring_rgb = (                  # type: ignore[attr-defined]
                lambda self: np.asarray(self.buffer_rgba())[..., :3].tobytes())
        cls.environment_note = (
            "matplotlib FigureCanvasAgg.tostring_rgb reinstated in memory over "
            "buffer_rgba; removed in Matplotlib 3.10, used by the mask "
            "generator on every item, and the subject pins matplotlib 3.5.1")

    def _instrument(self) -> None:
        """Record the two randomised treatment steps in the order they run.

        ``core/dataset.py`` binds the generator into its own namespace at
        import (line 12), so that is where it is wrapped. The flip is recorded
        when it is called, not when it flips: a branch not taken is not a
        reordering. Wrappers delegate, and a later adapter replaces an earlier
        wrapper instead of stacking on it.
        """
        rec = self.recorder
        gen = self._mod.create_random_shape_with_random_motion
        gen = getattr(gen, "_ti_original", gen)
        flip_cls = self._utils.GroupRandomHorizontalFlip
        flip = getattr(flip_cls.__call__, "_ti_original", flip_cls.__call__)

        def mask(*a, **kw):
            rec.record("mask")
            return gen(*a, **kw)

        def hflip(*a, **kw):
            rec.record("hflip")
            return flip(*a, **kw)

        mask._ti_original, hflip._ti_original = gen, flip      # type: ignore[attr-defined]
        self._mod.create_random_shape_with_random_motion = mask
        flip_cls.__call__ = hflip

    @contextlib.contextmanager
    def disable_generator(self):
        """Replace the mask generator with a sentinel for the duration."""
        name = "create_random_shape_with_random_motion"
        current = getattr(self._mod, name)

        def sentinel(*a, **kw):
            raise GeneratorReached("TrainDataset generated its mask online")

        setattr(self._mod, name, sentinel)
        try:
            yield [name]
        finally:
            setattr(self._mod, name, current)

    # -- adapter protocol ---------------------------------------------------
    def build(self, spec: LoaderSpec):
        # The subject reads <data_root>/<name>/train.json ({clip: n_frames}) and
        # one zip per clip under JPEGImages/, indexing the sorted member list
        # (core/utils.py:49-56). PNG members are stored as written: the reader
        # decodes with PIL and never looks at the extension. ``lq_root`` is not
        # used because the subject reads one stream only.
        gt_root = Path(spec.gt_root)
        data = gt_root.parent / "e2fgvi_stage" / "fixture"
        (data / "JPEGImages").mkdir(parents=True, exist_ok=True)
        counts = {}
        for clip in sorted(p for p in gt_root.iterdir() if p.is_dir()):
            frames = sorted(clip.glob("*.png"))
            zpath = data / "JPEGImages" / f"{clip.name}.zip"
            with zipfile.ZipFile(zpath, "w") as z:
                for f in frames:
                    z.write(f, f.name)
            # The reader caches open handles by path, process-wide; a rewritten
            # fixture at the same path would otherwise be read through a stale one.
            self._utils.TrainZipReader.file_dict.pop(str(zpath), None)
            counts[clip.name] = len(frames)
        (data / "train.json").write_text(json.dumps(counts))
        self._ds = self._cls(dict(LOADER_OPTIONS, name="fixture",
                                  data_root=str(data.parent)))

        # The loader's own claim about which frames it selected is the return
        # value of ``_sample_index``; it is kept so temporal_window can compare
        # it with what arrived.
        pick = self._ds._sample_index

        def claim(*a, **kw):
            self.claimed_ids = list(pick(*a, **kw))
            return self.claimed_ids

        self._ds._sample_index = claim
        return self._ds

    def sample(self, loader, index: int) -> Sample:
        frames, masks, _ = loader[index % len(loader)]
        f = frames.numpy().astype(np.float64)        # (T, 3, H, W) in [-1, 1]
        m = masks.numpy().astype(np.float64)         # (T, 1, H, W), 1 = hole
        lq = f * (1.0 - m)                           # core/trainer.py:304
        return Sample(lq=np.transpose(lq, (0, 2, 3, 1)),
                      gt=np.transpose(f, (0, 2, 3, 1)),
                      frame_ids=self.claimed_ids,
                      extra={"hole_fraction": float(m.mean())})

    def __len__(self) -> int:
        return len(self._ds)

    # No ``settings()``: the loader's options are geometry and window sizes
    # only (configs/train_e2fgvi.json:5-10). Fixed versus moving masks is a
    # coin flip inside the generator (core/utils.py:207), not a switch.
