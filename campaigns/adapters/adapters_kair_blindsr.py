"""Adapter for KAIR's ``DatasetBlindSR`` --- Campaign P subject 2.

Chosen deliberately: this loader composes a *randomised* degradation sequence
(``shuffle_order = random.sample(range(7), 7)`` in ``utils_blindsr``), which is
exactly the declared policy that ``operator_trace`` exists to check. Subject 1
returned ``SKIP`` on that gate because it composes no sequence of its own; this
one exercises it.

It is also a single-image loader, so the temporal gate has no window to assert.
Reporting that as ``SKIP`` rather than as a pass is part of the answer to RQ4:
which gates are generic, and which require the subject to expose something.

Adaptation effort is a response variable of the study. Everything below the
header is that cost, and the subject is not modified to make a gate pass.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

KAIR_ROOT = Path("/home/laura/02ImproveData/zKAIR")

from treatment_identity.adapter import CallRecorder, LoaderSpec, Sample  # noqa: E402

#: The seven branches of the BSRGAN degradation loop, named as the source
#: comments name them. The gate is told this set and the declared policy.
BSRGAN_OPS = {"blur_a", "blur_b", "downsample2", "downsample3",
              "gaussian_noise", "jpeg", "processed_camera_sensor_noise"}


class KairBlindSRAdapter:
    """Expose ``DatasetBlindSR`` and its operator sequence to the gates."""

    name = "kair.DatasetBlindSR"
    project = "KAIR"
    url = "https://github.com/cszn/KAIR"

    #: Set by :meth:`_restore_removed_dependency`. Reported in the certificate
    #: and in the article: this subject cannot be audited as released.
    environment_reconstruction: str | None = None

    def __init__(self, kair_root: Path = KAIR_ROOT, *, shim: bool = True):
        self.kair_root = Path(kair_root)
        if str(self.kair_root) not in sys.path:
            sys.path.insert(0, str(self.kair_root))
        if shim:
            self._restore_removed_dependency()
        from data.dataset_blindsr import DatasetBlindSR              # noqa: E402
        from utils import utils_blindsr as blindsr                   # noqa: E402
        self._cls = DatasetBlindSR
        self._blindsr = blindsr
        # The subject binds the name into its own namespace at import time, so
        # patching scipy alone is not enough if the module was imported first.
        if shim and type(self).environment_reconstruction:
            import scipy.interpolate as si
            blindsr.interp2d = si.interp2d
        self.recorder = CallRecorder()
        self._instrument()

    # -- environment reconstruction -----------------------------------------
    @classmethod
    def _restore_removed_dependency(cls) -> None:
        """Reinstate ``scipy.interpolate.interp2d``, removed in SciPy 1.14.

        The subject calls it in ``shift_pixel`` to translate a blur kernel by a
        sub-pixel offset, so under a current environment three of the four gates
        abort before asserting anything. Nothing in the repository pins a SciPy
        that still provides it.

        This is **environment reconstruction, not subject modification**, and
        the distinction is the whole point of reporting it. The subject's own
        source is untouched; what we supply is a dependency its authors could
        assume and we cannot. A practitioner auditing this pipeline today has to
        do exactly this, and the fact that they have to is a finding.

        The replacement is the one SciPy's own removal notice recommends,
        ``RegularGridInterpolator`` on a regular grid, restricted to the
        rectangular-output call pattern the subject uses.
        """
        import scipy.interpolate as si

        # SciPy 1.14 keeps the NAME and removes the FUNCTION: the attribute
        # exists and raises NotImplementedError when called. A hasattr check
        # therefore reports the dependency as present, which is how this
        # reconstruction failed on its first attempt. Probe by calling it.
        if getattr(si, "interp2d", None) is not None:
            try:
                si.interp2d(np.arange(2.0), np.arange(2.0), np.zeros((2, 2)))
                return                       # a real implementation is present
            except NotImplementedError:
                pass                         # the removal stub; replace it
            except Exception:
                return                       # something else; leave it alone

        from scipy.interpolate import RegularGridInterpolator

        def interp2d(x, y, z, kind="linear", **kwargs):
            method = {"linear": "linear", "cubic": "cubic"}.get(kind, "linear")
            grid = RegularGridInterpolator(
                (np.asarray(y), np.asarray(x)), np.asarray(z),
                method=method, bounds_error=False, fill_value=None)

            def call(xnew, ynew):
                xn, yn = np.asarray(xnew), np.asarray(ynew)
                yy, xx = np.meshgrid(yn, xn, indexing="ij")
                return grid(np.stack([yy, xx], axis=-1))

            return call

        si.interp2d = interp2d                                # type: ignore[attr-defined]
        cls.environment_reconstruction = (
            "scipy.interpolate.interp2d reinstated over RegularGridInterpolator; "
            "removed in SciPy 1.14 and not pinned by the subject repository")

    # -- instrumentation ----------------------------------------------------
    def _instrument(self) -> None:
        """Record which degradation branches execute, and in what order.

        The subject does not expose its operator sequence, so the adapter wraps
        the module-level functions the loop calls. This is the only place where
        the adapter reaches into the subject, and it observes rather than
        changes: every wrapper delegates to the original.

        Two of the seven branches cannot be separated this way --- branches 0
        and 1 both call ``add_blur`` --- so the trace sees ``blur`` twice rather
        than ``blur_a`` and ``blur_b``. That is a limit of black-box
        instrumentation and it is recorded rather than papered over.
        """
        b = self._blindsr
        rec = self.recorder
        for fn_name, label in (("add_blur", "blur"),
                               ("add_Gaussian_noise", "gaussian_noise"),
                               ("add_JPEG_noise", "jpeg"),
                               ("add_speckle_noise", "speckle_noise"),
                               ("add_Poisson_noise", "poisson_noise"),
                               ("add_resize", "resize")):
            original = getattr(b, fn_name, None)
            if original is None or getattr(original, "_ti_wrapped", False):
                continue

            def make(orig, lab):
                def wrapper(*a, **kw):
                    rec.record(lab)
                    return orig(*a, **kw)
                wrapper._ti_wrapped = True                    # type: ignore[attr-defined]
                return wrapper

            setattr(b, fn_name, make(original, label))

    # -- adapter protocol ---------------------------------------------------
    def build(self, spec: LoaderSpec):
        gt_root = Path(spec.gt_root)
        clips = sorted(p for p in gt_root.iterdir() if p.is_dir())
        root = clips[0] if clips else gt_root
        opt = {
            "dataroot_H": str(root),
            "n_channels": 3,
            "scale": 2,
            "shuffle_prob": 0.1,
            "use_sharp": False,
            "degradation_type": "bsrgan",
            "lq_patchsize": 16,
            "H_size": 32,
            "phase": "train",
        }
        self._ds = self._cls(opt)
        return self._ds

    def sample(self, loader, index: int) -> Sample:
        item = loader[index % len(loader)]
        lq = _to_numpy(item["L"])
        gt = _to_numpy(item["H"])
        return Sample(lq=lq[None, ...], gt=gt[None, ...], frame_ids=None)

    def __len__(self) -> int:
        return len(self._ds)


def _to_numpy(x) -> np.ndarray:
    """Torch (C,H,W) in [0,1] -> numpy (H,W,C) in [0,255]."""
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
