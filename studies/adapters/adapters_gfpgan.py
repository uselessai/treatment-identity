"""Adapter for GFPGAN's ``FFHQDegradationDataset`` --- a further portability subject.

GFPGAN (CVPR 2021, Wang et al., https://github.com/TencentARC/GFPGAN) is a
blind face-restoration model. Its training loader reads only high-quality faces
and manufactures the low-quality input online from each one, following the
degradation model of the paper (blur, downsample, noise, JPEG), and then adds
colour jitter and conversion to grey that the paper does not describe but the
shipped configuration switches on (``options/train_gfpgan_v1.yml``).

It is included for two reasons. It is a third subject that builds its input
from the target, after KAIR's blind SR loader and DDColor, and it is the first
whose degradation operators are all named in a publication, so the operator
order is a published claim rather than a code reading. And it depends on the
*installed* ``basicsr`` rather than a vendored copy: the other two subjects
that ship a ``basicsr`` can shadow it on ``sys.path``, so this adapter binds
the name to the installed distribution and records which file it loaded.

Adaptation effort is a response variable of the study. Everything below the
header is that cost, and the subject is not modified to make a gate pass.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
GFPGAN_ROOT = Path("/home/laura/02ImproveData/zGFPGAN")

for base in (HERE, *HERE.parents):
    for cand in (base, base / "treatment_identity"):
        if (cand / "treatment_identity" / "__init__.py").exists():
            sys.path.insert(0, str(cand))
            break

from treatment_identity.adapter import CallRecorder, LoaderSpec, Sample  # noqa: E402

#: The recorded label of the branch each swept option guards.
BRANCH_OF = {"use_hflip": "hflip", "color_jitter_prob": "color_jitter",
             "color_jitter_pt_prob": "color_jitter_pt", "gray_prob": "gray"}


class _Observed:
    """Stand-in for a module the subject calls by attribute: records the named
    calls and delegates every attribute, so what runs is unchanged."""

    def __init__(self, target, labels: dict, rec: CallRecorder):
        self._target, self._labels, self._rec = target, labels, rec

    def __getattr__(self, name):
        attr = getattr(self._target, name)
        if name not in self._labels:
            return attr

        def call(*a, **kw):
            self._rec.record(self._labels[name])
            return attr(*a, **kw)
        return call


class GfpganAdapter:
    """Expose GFPGAN's FFHQ training loader and its degradation sequence."""

    name = "gfpgan.FFHQDegradationDataset"
    project = "GFPGAN"
    url = "https://github.com/TencentARC/GFPGAN"

    #: Set by :meth:`_bind_installed_basicsr`. Reported in the certificate.
    environment_note: str | None = None

    def __init__(self, root: Path = GFPGAN_ROOT):
        self.root = Path(root)
        mod = self._bind_installed_basicsr()
        self._mod = mod
        self._cls = mod.FFHQDegradationDataset
        self.basicsr_file = mod.degradations.__file__
        self.recorder = CallRecorder()
        self.last_branch_reached = False
        self._instrument()

    # -- environment --------------------------------------------------------
    def _bind_installed_basicsr(self):
        """Import the loader against the installed ``basicsr``, without facexlib.

        Three things are done in memory and nothing is written anywhere:

        * ``basicsr`` is imported with this environment's site-packages first
          on the path. Earlier subjects leave their own roots on ``sys.path``,
          two of which contain a ``basicsr`` directory, and the subject's
          ``requirements.txt`` asks for the installed package (>=1.4.2).
        * ``gfpgan/__init__.py`` is bypassed by a package object bound to the
          subject's directory. It imports ``gfpgan.utils``, whose line 6 needs
          ``facexlib`` for inference-time face detection; the training loader
          and ``gfpgan/data/__init__.py`` do not touch it.
        * the JPEG quality basicsr passes to OpenCV is made an integer (below).
        """
        import importlib
        import sysconfig
        import types

        for name in [m for m in list(sys.modules)
                     if m.split(".")[0] in ("basicsr", "gfpgan")]:
            del sys.modules[name]
        site = sysconfig.get_paths()["purelib"]
        sys.path.insert(0, site)
        try:
            basicsr = importlib.import_module("basicsr")
        finally:
            sys.path.remove(site)
        pkg = types.ModuleType("gfpgan")
        pkg.__path__ = [str(self.root / "gfpgan")]                 # type: ignore[attr-defined]
        sys.modules["gfpgan"] = pkg
        mod = importlib.import_module("gfpgan.data.ffhq_degradation_dataset")
        # basicsr's add_jpg_compression hands cv2.imencode the float quality
        # that random_add_jpg_compression draws; OpenCV 4.10 rejects a float
        # in that int list ("Sequence item with index 1 has a wrong type"), so
        # the JPEG step raises on every item. Neither package pins OpenCV. The
        # quality is passed as int(quality), the integer imencode accepts.
        degr = importlib.import_module("basicsr.data.degradations")
        jpeg = degr.add_jpg_compression
        degr.add_jpg_compression = lambda img, quality=90: jpeg(img, int(quality))
        type(self).environment_note = (
            f"basicsr {basicsr.__version__} bound from {Path(basicsr.__file__).parent} "
            "(installed package, as requirements.txt asks); gfpgan/__init__.py "
            "bypassed in memory because gfpgan/utils.py imports facexlib, which "
            "is not installed and which the training loader does not use; "
            "basicsr add_jpg_compression given int(quality) in memory, because "
            "OpenCV 4.10 imencode rejects the float quality it passes")
        return mod

    # -- instrumentation ----------------------------------------------------
    def _instrument(self) -> None:
        """Record the degradation calls of ``__getitem__`` without altering them.

        The subject calls ``cv2`` and ``degradations`` by attribute, so each is
        replaced in the subject's namespace by a recording stand-in. Both
        ``cv2.resize`` calls are labelled ``resize``: the first is the paper's
        downsampling (line 176), the second restores the original size after
        JPEG (line 185). The colour-jitter helpers and the four adjustments
        they call in a random order (``torch.randperm(4)``, line 101) are
        wrapped where the subject looks them up.
        """
        import cv2

        m, rec, cls = self._mod, self.recorder, self._cls
        m.cv2 = _Observed(cv2, {"filter2D": "blur", "resize": "resize",
                                "cvtColor": "gray"}, rec)
        m.degradations = _Observed(
            m.degradations, {"random_add_gaussian_noise": "gaussian_noise",
                             "random_add_jpg_compression": "jpeg"}, rec)

        def wrap(fn, label):
            def wrapper(*a, **kw):
                rec.record(label)
                return fn(*a, **kw)
            return wrapper

        for fn_name, label in (("adjust_brightness", "brightness"),
                               ("adjust_contrast", "contrast"),
                               ("adjust_saturation", "saturation"),
                               ("adjust_hue", "hue")):
            setattr(m, fn_name, wrap(getattr(m, fn_name), label))
        for fn_name in ("color_jitter", "color_jitter_pt"):
            setattr(cls, fn_name, staticmethod(wrap(getattr(cls, fn_name), fn_name)))

        augment = m.augment

        def observed_augment(*a, **kw):
            out = augment(*a, **kw)
            if out[1][0]:                    # return_status: (hflip, vflip, rot90)
                rec.record("hflip")
            return out
        m.augment = observed_augment

    # -- options ------------------------------------------------------------
    @staticmethod
    def opt(gt_folder: Path, **override) -> dict:
        """``datasets.train`` of ``options/train_gfpgan_v1.yml``, lines 13-45.

        One departure: ``crop_components`` (line 43) is off, because it loads
        a landmark file (line 44) this audit does not have. It adds facial
        component boxes to the item and leaves ``lq`` and ``gt`` unchanged
        (lines 160-162, 216-225). A fresh dict per build matters: the subject
        pops ``type`` from ``io_backend`` on first access (line 147).
        """
        opt = {
            "dataroot_gt": str(gt_folder), "io_backend": {"type": "disk"},
            "use_hflip": True, "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5],
            "out_size": 512,
            "blur_kernel_size": 41, "kernel_list": ["iso", "aniso"],
            "kernel_prob": [0.5, 0.5], "blur_sigma": [0.1, 10],
            "downsample_range": [0.8, 8], "noise_range": [0, 20],
            "jpeg_range": [60, 100],
            "color_jitter_prob": 0.3, "color_jitter_shift": 20,
            "color_jitter_pt_prob": 0.3, "gray_prob": 0.01,
            "crop_components": False,
        }
        opt.update(override)
        return opt

    # -- adapter protocol ---------------------------------------------------
    def build(self, spec: LoaderSpec):
        # The subject scans one flat folder of faces (paths_from_folder, not
        # recursive); the fixture writes one directory per clip.
        gt_root = Path(spec.gt_root)
        clips = sorted(p for p in gt_root.iterdir() if p.is_dir())
        self._ds = self._cls(self.opt(clips[0] if clips else gt_root))
        return self._ds

    def sample(self, loader, index: int) -> Sample:
        item = loader[index % len(loader)]
        return Sample(lq=_to_hwc(item["lq"])[None], gt=_to_hwc(item["gt"])[None],
                      frame_ids=None)

    def __len__(self) -> int:
        return len(self._ds)

    # -- settings sweep -----------------------------------------------------
    #: Treatment switches of the shipped configuration, each compared between
    #: off and its shipped value (yml lines 18, 32, 34, 35; the off values are
    #: the ones the yml's own comment at lines 37-40 suggests).
    def settings(self):
        return {"use_hflip": (False, True), "color_jitter_prob": (None, 0.3),
                "color_jitter_pt_prob": (None, 0.3), "gray_prob": (None, 0.01)}

    def deliver_under(self, workdir: Path, name: str, value):
        """Deliver the input of one item with one option set, the rest as shipped.

        Returns the input and whether the branch that option guards ran: every
        one of these branches is probabilistic.
        """
        gt_root = Path(workdir) / "fixture"
        if not gt_root.exists():
            from treatment_identity.fixtures import make_scale_grid
            make_scale_grid(gt_root, n_frames=8, shape=(128, 128))
        clips = sorted(p for p in gt_root.iterdir() if p.is_dir())
        ds = self._cls(self.opt(clips[0], **{name: value}))
        self.recorder.reset()
        self.last_branch_reached = False
        try:
            item = ds[0]
        finally:
            self.last_branch_reached = BRANCH_OF[name] in self.recorder.calls
        return _to_hwc(item["lq"]), self.last_branch_reached


def _to_hwc(x) -> np.ndarray:
    """Torch (C,H,W) -> numpy (H,W,C), values left as delivered ([-1, 1])."""
    a = x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
    if a.ndim == 3 and a.shape[0] in (1, 3):
        a = np.transpose(a, (1, 2, 0))
    return a
