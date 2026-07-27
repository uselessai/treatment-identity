"""Synthetic fixtures whose content makes a delivery divergence unmistakable.

Every fixture is designed so that the failure mode it targets produces a value
that cannot be confused with a plausible result: a checkerboard that no
degradation of a flat field could produce, or a frame whose every pixel encodes
its own index.

All fixtures are written to disk as 8-bit PNG clips laid out the way film
restoration loaders expect::

    root/
      clip_000/
        00000000.png
        00000001.png
        ...

CPU only; a full fixture is a few hundred kilobytes.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np

__all__ = [
    "FixtureClip",
    "make_flat_gt",
    "make_checkerboard_lq",
    "make_index_encoded",
    "decode_index",
    "make_pair",
]

# Frame value reserved for the flat ground truth. Chosen mid-range so that a
# degradation of it stays far from the checkerboard extremes.
FLAT_VALUE = 128

# Checkerboard uses the extremes: no blur/noise/JPEG chain applied to a flat
# field can produce this bimodal histogram.
CHECKER_LO, CHECKER_HI = 0, 255


class FixtureClip:
    """A written-out clip plus the metadata needed to assert on it."""

    def __init__(self, root: Path, name: str, n_frames: int, shape: tuple[int, int]):
        self.root = Path(root)
        self.name = name
        self.n_frames = n_frames
        self.shape = shape

    @property
    def path(self) -> Path:
        return self.root / self.name

    def frame_path(self, i: int) -> Path:
        return self.path / f"{i:08d}.png"

    def read(self, i: int) -> np.ndarray:
        img = cv2.imread(str(self.frame_path(i)))
        if img is None:
            raise FileNotFoundError(self.frame_path(i))
        return img


def _write_clip(root: Path, name: str, frames: list[np.ndarray]) -> FixtureClip:
    d = Path(root) / name
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    for i, f in enumerate(frames):
        cv2.imwrite(str(d / f"{i:08d}.png"), f)
    h, w = frames[0].shape[:2]
    return FixtureClip(root, name, len(frames), (h, w))


def make_flat_gt(root: Path, name: str = "clip_000", n_frames: int = 16,
                 shape: tuple[int, int] = (64, 64), value: int = FLAT_VALUE) -> FixtureClip:
    """A constant grey clip. Any structure in a delivered tensor came from elsewhere."""
    h, w = shape
    frames = [np.full((h, w, 3), value, np.uint8) for _ in range(n_frames)]
    return _write_clip(root, name, frames)


def make_checkerboard_lq(root: Path, name: str = "clip_000", n_frames: int = 16,
                         shape: tuple[int, int] = (64, 64), cell: int = 8) -> FixtureClip:
    """A high-contrast checkerboard.

    Delivered as the low-quality stream, it answers one question with no
    ambiguity: did the loader open the pre-computed input, or did it generate
    its own from the target?
    """
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    board = np.where(((yy // cell) + (xx // cell)) % 2 == 0, CHECKER_HI, CHECKER_LO)
    frames = [np.dstack([board.astype(np.uint8)] * 3) for _ in range(n_frames)]
    return _write_clip(root, name, frames)


def make_index_encoded(root: Path, name: str = "clip_000", n_frames: int = 32,
                       shape: tuple[int, int] = (64, 64)) -> FixtureClip:
    """A clip whose frame *i* has every pixel equal to ``i``.

    A loader that computes a temporal window and then discards it cannot hide:
    the delivered tensor states which frames it opened.
    """
    if n_frames > 256:
        raise ValueError("index encoding needs n_frames <= 256")
    h, w = shape
    frames = [np.full((h, w, 3), i, np.uint8) for i in range(n_frames)]
    return _write_clip(root, name, frames)


def decode_index(frame: np.ndarray, tol: float = 2.0) -> int | None:
    """Recover the frame index written by :func:`make_index_encoded`.

    Accepts float tensors in [0,1] or [-1,1] as well as uint8, because loaders
    normalise differently. Returns ``None`` if the frame is not flat enough to
    be an index-encoded fixture (i.e. something degraded it).
    """
    raw = np.asarray(frame)
    a = raw.astype(np.float64)
    if a.ndim == 4:
        a = a[0]
    if a.ndim == 3 and a.shape[0] in (1, 3) and a.shape[-1] not in (1, 3):
        a = np.moveaxis(a, 0, -1)  # CHW -> HWC
    # Only floating-point tensors are rescaled. An integer frame is already in
    # level units, and rescaling it would turn index 1 into 255 --- the kind of
    # silent mis-decode this suite exists to prevent.
    if np.issubdtype(raw.dtype, np.floating):
        amax, amin = float(a.max()), float(a.min())
        if amax <= 1.0 + 1e-6 and amin >= -1e-6:
            a = a * 255.0
        elif amax <= 1.0 + 1e-6:
            a = (a + 1.0) * 127.5
    if float(a.std()) > tol:
        return None
    return int(round(float(a.mean())))


def make_pair(root: Path, *, n_frames: int = 32, shape: tuple[int, int] = (64, 64),
              gt_kind: str = "flat", lq_kind: str = "checker",
              clip: str = "clip_000") -> tuple[FixtureClip, FixtureClip]:
    """Write a matched GT/LQ fixture pair under ``root/GT`` and ``root/LQ``."""
    root = Path(root)
    builders = {
        "flat": make_flat_gt,
        "checker": make_checkerboard_lq,
        "index": make_index_encoded,
    }
    gt = builders[gt_kind](root / "GT", clip, n_frames, shape)
    lq = builders[lq_kind](root / "LQ", clip, n_frames, shape)
    return gt, lq
