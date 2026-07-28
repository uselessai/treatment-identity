"""Adapter for the RTN / RRTN / MambaOFR family (``VP_code`` layout).

The three repositories share a dataset class name (``Film_dataset_1``) and a
degradation module, so one adapter covers all three; the differences it exposes
are exactly what the audit is about.

Nothing here modifies the repositories. The adapter imports their modules from
a path, instruments a few functions to observe which ones run, and reads back
what the dataset returns.
"""

from __future__ import annotations

import contextlib
import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from treatment_identity.adapter import (AdapterError, CallRecorder,  # noqa: E402
                                        GeneratorReached, LoaderSpec, Sample)
from treatment_identity.seeding import seed_all  # noqa: E402

# Operators whose realised order the trace gate inspects.
OPERATORS = {"blur", "downsample", "noise", "jpeg"}


def _load_module(pkg_name: str, path: Path, pkg_dir: Path):
    """Import a module file as part of a synthetic package, so relative imports work."""
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(pkg_dir)]
        sys.modules[pkg_name] = pkg
    full = f"{pkg_name}.{path.stem}"
    if full in sys.modules:
        return sys.modules[full]
    spec = importlib.util.spec_from_file_location(full, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full] = mod
    spec.loader.exec_module(mod)
    return mod


class VPCodeAdapter:
    """Wraps one repository of the RTN lineage."""

    def __init__(self, name: str, repo: str | Path, *, texture_bank: str | Path,
                 channels: int = 3, dataset_module: str = "VP_code.data.dataset",
                 degradation_module: str = "VP_code.data.Data_Degradation.util",
                 degradation_path: str | None = None):
        self.name = name
        self.repo = Path(repo)
        self.dataset_module = dataset_module
        self.degradation_module = degradation_module
        self.degradation_path = degradation_path
        self.texture_bank = str(texture_bank)
        self.channels = channels
        self.recorder = CallRecorder()
        self._util = None
        self._live_util = None
        self._dataset_cls = None

    # -- module loading -------------------------------------------------
    @property
    def util(self):
        if self._util is None:
            tag = f"ti_{self.name.lower()}_deg"
            rel = self.degradation_path or "VP_code/data/Data_Degradation/util.py"
            f = self.repo / rel
            if not f.exists():
                raise AdapterError(f"no degradation module at {f}")
            self._util = _load_module(tag, f, f.parent)
            self._instrument()
        return self._util

    def _instrument(self) -> None:
        self._instrument_module(self._util)

    def _instrument_module(self, u) -> None:
        """Observe operator calls and target transformations without changing behaviour."""
        rec = self.recorder

        def wrap(attr: str, label: str):
            orig = getattr(u, attr, None)
            if orig is None or getattr(orig, "_ti_wrapped", False):
                return
            def inner(*a, **k):
                rec.record(label)
                return orig(*a, **k)
            inner._ti_wrapped = True          # type: ignore[attr-defined]
            inner._ti_orig = orig             # type: ignore[attr-defined]
            setattr(u, attr, inner)

        wrap("add_blur_fixed", "blur")
        wrap("downsampling_artifact_v3_fixed", "downsample")
        wrap("gaussian_noise_artifact_v2", "noise")
        wrap("speckle_noise_artifact_v2", "noise")
        wrap("jpeg_artifact_v2", "jpeg")
        wrap("add_sharpening", "sharpen")
        wrap("color_jitter", "color_jitter")

    def degradation_fn(self, which: str = "4"):
        """Return the generator the training loader invokes."""
        u = self.util
        name = {"4": "degradation_video_list_4",
                "4_one_channel": "degradation_video_list_4_one_channel",
                "5": "degradation_video_list_5"}[which]
        fn = getattr(u, name, None)
        if fn is None:
            raise AdapterError(f"{self.name} has no {name}")
        return fn

    # -- dataset --------------------------------------------------------
    def _dataset(self):
        """Import this repository's dataset class in isolation.

        The three repositories all expose a top-level ``VP_code`` package, so
        importing one leaves a module that would shadow the next. We purge that
        namespace and pin ``sys.path`` for the duration of the import; the class
        object is then cached and the global state restored.
        """
        if self._dataset_cls is not None:
            return self._dataset_cls
        saved_path = list(sys.path)
        top = self.dataset_module.split(".")[0]
        saved_mods = {k: v for k, v in sys.modules.items() if k.split(".")[0] == top}
        for k in saved_mods:
            del sys.modules[k]
        try:
            sys.path.insert(0, str(self.repo))
            import importlib
            mod = importlib.import_module(self.dataset_module)
            cls = getattr(mod, "Film_dataset_1", None)
            if cls is None:
                raise AdapterError(f"{self.name}: no Film_dataset_1")
            self._dataset_cls = cls
            # Instrument the *live* degradation module --- the one the dataset's
            # imported functions resolve their globals against. Patching a
            # separately loaded copy would observe nothing, because the generator
            # looks its operators up in this module's dict at call time.
            self._live_util = sys.modules.get(self.degradation_module)
            if self._live_util is not None:
                self._instrument_module(self._live_util)
        finally:
            for k in [k for k in sys.modules if k.split(".")[0] == top]:
                del sys.modules[k]
            sys.modules.update(saved_mods)
            sys.path[:] = saved_path
        return self._dataset_cls

    def build(self, spec: LoaderSpec) -> Any:
        # A loader that samples its temporal window at random must start from a
        # known state, or every probe of it reports a different answer and the
        # certificate means nothing. See treatment_identity.seeding.
        seed_all(spec.seed)
        cls = self._dataset()
        cfg = {
            "name": "TI",
            "type": "Film_dataset_1",
            "dataroot_gt": str(spec.gt_root),
            "dataroot_lq": str(spec.lq_root or spec.gt_root),
            "num_frame": spec.num_frames,
            "interval_list": [1],
            "random_reverse": False,
            "use_hflip": False,
            "use_rot": False,
            "scale": 1,
            "gt_size": [64, 64],
            "use_flip": False,
            "texture_template": self.texture_bank,
            "normalizing": True,
            "is_train": spec.train,
            "use_precomputed_lq_in_train": spec.use_precomputed_lq,
            "channels": self.channels,
            "val_partition": "REDS4",
        }
        cfg.update(spec.options)
        try:
            return cls(cfg)
        except Exception as e:  # pragma: no cover - repo-specific signatures
            raise AdapterError(f"{self.name}: cannot construct dataset: {e}") from e

    def sample(self, loader: Any, index: int) -> Sample:
        item = loader[index % max(len(loader), 1)]
        lq, gt = item["lq"], item["gt"]
        to_np = lambda t: t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)
        return Sample(lq=to_np(lq), gt=to_np(gt),
                      frame_ids=list(item.get("frame_list", []) or []) or None,
                      extra={"key": item.get("key")})

    @contextlib.contextmanager
    def disable_generator(self):
        """Replace every online generator in the live module with a sentinel.

        The dataset's imported functions resolve their globals against this
        module at call time, so rebinding the names here is sufficient: a loader
        that honours its pre-computed-input flag never looks them up, and one
        that ignores the flag raises instead of returning a plausible tensor.
        """
        u = self._live_util or self.util
        names = [n for n in ("degradation_video_list_4",
                             "degradation_video_list_4_one_channel",
                             "degradation_video_list_5",
                             "standard_degradation_pipeline")
                 if hasattr(u, n)]
        if not names:
            raise AdapterError(f"{self.name}: no generator to disable")

        def sentinel(*a, **k):
            raise GeneratorReached(
                f"{self.name}: the loader called the online generator although "
                "the pre-computed-input flag was set")

        saved = {n: getattr(u, n) for n in names}
        for n in names:
            setattr(u, n, sentinel)
        try:
            yield names
        finally:
            for n, fn in saved.items():
                setattr(u, n, fn)

    # -- direct generator access, for the separability gate -------------
    def render(self, video: list[np.ndarray], *, which: str = "4", seed: int = 1234,
               neutralise_colour_jitter: bool = False) -> np.ndarray:
        """Run the generator itself under a fixed seed and return the degraded stream."""
        import random
        u = self.util
        saved = None
        if neutralise_colour_jitter:
            saved = u.color_jitter
            u.color_jitter = lambda img: img.convert("L")
        try:
            random.seed(seed)
            np.random.seed(seed)
            fn = self.degradation_fn(which)
            lq, _gt = fn(video, texture_url=self.texture_bank)
            return np.asarray(lq)
        finally:
            if saved is not None:
                u.color_jitter = saved


def make_adapters(root: str | Path, texture_bank: str | Path) -> dict[str, VPCodeAdapter]:
    """Build one adapter per repository found under ``root``."""
    root = Path(root)
    known = {
        "RTN": ("Bringing-Old-Films-Back-to-Life", 3),
        "RRTN": ("RRTN-old-film-restoration", 1),
        "MambaOFR": ("MambaOFR", 3),
    }
    out: dict[str, VPCodeAdapter] = {}
    for name, (folder, ch) in known.items():
        p = root / folder
        if p.exists():
            out[name] = VPCodeAdapter(name, p, texture_bank=texture_bank, channels=ch)
    return out
