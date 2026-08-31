"""Adapter for DDColor's ``LabDataset`` --- the CutMix treatment pair.

The subject exposes ``do_cutmix`` as a switch on its training treatment and
ships it turned off (``options/train/train_ddcolor.yml``).  Turning it on is a
declared distinction: the loader is asked for a different treatment.  This
adapter delivers one item under each setting so ``separability`` can ask
whether that distinction reaches the delivered tensor.

Two things the subject cannot supply are supplied here, and both are recorded
rather than hidden:

* ``rand_bbox`` calls ``np.int``, removed in NumPy 1.24, so the CutMix branch
  raises before doing anything on a current environment.  This is environment
  reconstruction of the same kind Campaign P reports for SciPy, not a change
  to the subject's behaviour: ``np.int`` was an alias for the builtin ``int``.
* the branch is probabilistic (``np.random.uniform(...) > cutmix_p``), so an
  identical pair proves nothing unless we know the branch was entered.  The
  adapter wraps ``rand_bbox`` to record entry, which is the same reachability
  instrumentation the other adapters use, and reports the two cases apart.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for base in (HERE, *HERE.parents):
    for cand in (base, base / "treatment_identity"):
        if (cand / "treatment_identity" / "__init__.py").exists():
            sys.path.insert(0, str(cand))
            break

from treatment_identity.adapter import LoaderSpec, Sample  # noqa: E402

DDCOLOR_ROOT = Path("/home/laura/02ImproveData/zDDColor")


class DDColorLabAdapter:
    """Expose ``LabDataset`` under two CutMix settings."""

    name = "ddcolor.LabDataset"
    project = "DDColor"
    url = "https://github.com/piddnad/DDColor"

    environment_reconstruction: str | None = None

    def __init__(self, root: Path = DDCOLOR_ROOT):
        self.root = Path(root)
        if str(self.root) not in sys.path:
            sys.path.insert(0, str(self.root))
        self._restore_removed_alias()
        lab_dataset = self._claim_vendored_basicsr()
        self._mod = lab_dataset
        self._cls = lab_dataset.LabDataset
        self.branch_entries = 0
        self._instrument()

    def _claim_vendored_basicsr(self):
        """Bind ``basicsr`` to this subject's own vendored copy.

        Two subjects of this campaign vendor a package under that name, and in
        one process whichever imported last owns it. Reclaiming it here, rather
        than relying on import order, is what keeps one subject from being
        audited through another subject's code --- a confusion that would not
        raise, because both copies define most of the same names.
        """
        import importlib

        for name in [m for m in list(sys.modules)
                     if m == "basicsr" or m.startswith("basicsr.")]:
            del sys.modules[name]
        while str(self.root) in sys.path:
            sys.path.remove(str(self.root))
        sys.path.insert(0, str(self.root))
        return importlib.import_module("basicsr.data.lab_dataset")

    # -- environment reconstruction -----------------------------------------
    @classmethod
    def _restore_removed_alias(cls) -> None:
        """Reinstate ``np.int``, removed in NumPy 1.24.

        ``rand_bbox`` uses it to floor the box size. Without it the CutMix
        branch raises ``AttributeError`` before touching a pixel, so the
        published code cannot exercise its own augmentation on any current
        environment. The alias was for the builtin ``int`` and is restored as
        that, which changes no behaviour.
        """
        if getattr(np, "int", None) is not None:
            return
        np.int = int                                # type: ignore[attr-defined]
        cls.environment_reconstruction = (
            "numpy.int reinstated as the builtin int; removed in NumPy 1.24 "
            "and used by rand_bbox, so the CutMix branch cannot execute as "
            "released")

    # -- instrumentation ----------------------------------------------------
    def _instrument(self) -> None:
        """Record entry into the CutMix branch without altering it."""
        original = getattr(self._mod, "rand_bbox")
        if getattr(original, "_ti_wrapped", False):
            return

        adapter = self

        def wrapper(*a, **kw):
            adapter.branch_entries += 1
            return original(*a, **kw)

        wrapper._ti_wrapped = True                  # type: ignore[attr-defined]
        self._mod.rand_bbox = wrapper

    # -- fixture ------------------------------------------------------------
    @staticmethod
    def write_fixture(workdir: Path, n: int = 8, size: int = 320) -> Path:
        """A small corpus of distinguishable images plus the meta-info file
        the subject requires. Content differs per image so a CutMix paste
        from another index is detectable."""
        import cv2

        workdir = Path(workdir)
        img_dir = workdir / "gt"
        img_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for i in range(n):
            rng = np.random.default_rng(1000 + i)
            img = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
            # a flat band per image index, so a pasted region is unmistakable
            img[:, :, 0] = (i * 29) % 256
            p = img_dir / f"img_{i:03d}.png"
            cv2.imwrite(str(p), img)
            paths.append(str(p))
        meta = workdir / "meta_info.txt"
        meta.write_text("\n".join(paths) + "\n")
        return meta

    # -- delivery -----------------------------------------------------------
    def opt(self, meta: Path, *, do_cutmix: bool, gt_size: int = 256) -> dict:
        """The subject's own option names; only do_cutmix differs between arms."""
        return {
            "io_backend": {"type": "disk"},
            "dataroot_gt": str(meta.parent / "gt"),
            "meta_info_file": str(meta),
            "gt_size": gt_size,
            "do_fmix": False, "fmix_p": 0.5,
            "do_cutmix": do_cutmix, "cutmix_p": 0.5,
        }

    def deliver(self, meta: Path, *, do_cutmix: bool, index: int = 0):
        ds = self._cls(self.opt(meta, do_cutmix=do_cutmix))
        item = ds[index % len(ds)]
        gt = item["gt"] if isinstance(item, dict) and "gt" in item else item
        try:
            import torch
            if isinstance(gt, torch.Tensor):
                gt = gt.detach().cpu().numpy()
        except ImportError:                                       # pragma: no cover
            pass
        return np.asarray(gt)

    # -- adapter protocol ---------------------------------------------------
    def build(self, spec: LoaderSpec):
        """Drive the loader the way the six-gate suite expects.

        Like the Restormer subject, this one vendors a package called
        ``basicsr``; it reclaims the name here so the two cannot read each
        other's code when both run in one process.

        The subject reads a meta-info file of absolute paths, so the fixture is
        re-encoded once and listed there. Its own defaults are used for
        everything else, CutMix included, which the shipped configuration
        leaves off.
        """
        import cv2
        import importlib

        for name in [m for m in list(sys.modules) if m == "basicsr"
                     or m.startswith("basicsr.")]:
            del sys.modules[name]
        if str(self.root) in sys.path:
            sys.path.remove(str(self.root))
        sys.path.insert(0, str(self.root))
        self._cls = importlib.import_module("basicsr.data.lab_dataset").LabDataset

        gt_root = Path(spec.gt_root)
        stage = gt_root.parent / "ddcolor_stage"
        img_dir = stage / "gt"
        img_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for f in sorted(gt_root.rglob("*.png")):
            out = img_dir / f"{f.parent.name}_{f.stem}.png"
            if not out.exists():
                cv2.imwrite(str(out), cv2.imread(str(f)))
            paths.append(str(out))
        meta = stage / "meta_info.txt"
        meta.write_text("\n".join(paths) + "\n")
        self._meta = meta
        self._ds = self._cls(self.opt(meta, do_cutmix=False))
        return self._ds

    def sample(self, loader, index: int) -> Sample:
        item = loader[index % len(loader)]
        # The subject delivers luminance as the input and the ab pair as the
        # target; both are streams the gates can read.
        return Sample(lq=_to_hwc(item["lq"]), gt=_to_hwc(item["gt"]),
                      frame_ids=None)

    def __len__(self) -> int:
        return len(self._ds)


def _to_hwc(x) -> np.ndarray:
    try:
        import torch
        a = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
    except ImportError:                                       # pragma: no cover
        a = np.asarray(x)
    a = np.asarray(a, dtype=np.float64)
    if a.ndim == 3 and a.shape[0] in (1, 2, 3):
        a = np.transpose(a, (1, 2, 0))
    return a
