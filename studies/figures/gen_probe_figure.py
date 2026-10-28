#!/usr/bin/env python3
"""The fixtures DDC feeds a loader, and how the content check reads a delivery.

Top row: the frames written by treatment_identity.fixtures for the input,
frame-window, value/channel and content checks. Bottom row: the content
fixture at 16:9 after four operations a loader may apply, each read back by
treatment_identity.checks.measure_content_orientation. The numbers under the
bottom panels are those readings; nothing is drawn by hand.

    python gen_probe_figure.py DDC_REPO
"""
import sys
import tempfile
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(sys.argv[1])
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from treatment_identity import fixtures as fx                      # noqa: E402
from treatment_identity.checks import (measure_content_orientation,  # noqa: E402
                                       measure_content_scale)

PX, PY = 24, 40


def first_frame(builder, **kw):
    with tempfile.TemporaryDirectory() as tmp:
        clip = builder(Path(tmp), **kw)
        path = sorted(Path(clip.path if hasattr(clip, "path") else tmp).rglob("*.png"))[0]
        return cv2.imread(str(path), cv2.IMREAD_UNCHANGED)


with tempfile.TemporaryDirectory() as tmp:
    t = Path(tmp)
    fx.make_checkerboard_lq(t / "lq", n_frames=1, shape=(64, 64))
    fx.make_flat_gt(t / "gt", n_frames=1, shape=(64, 64))
    fx.make_index_encoded(t / "idx", n_frames=24, shape=(64, 64))
    fx.make_content_signature(t / "sig", n_frames=1, shape=(64, 64))
    fx.make_scale_grid(t / "grid", n_frames=1, shape=(480, 854), period_x=PX, period_y=PY)
    read = lambda d, i=0: cv2.imread(str(sorted((t / d).rglob("*.png"))[i]))  # noqa: E731
    checker, flat, sig, grid = read("lq"), read("gt"), read("sig"), read("grid")
    index = [read("idx", i) for i in (3, 11, 19)]

h, w = grid.shape[:2]
ops = {
    "Crop": grid[:256, :256],
    "Rescale": cv2.resize(grid, (455, 256), interpolation=cv2.INTER_AREA),
    "Reshape": cv2.resize(grid, (256, 256), interpolation=cv2.INTER_AREA),
    "Rotation": cv2.rotate(grid[:256, :256], cv2.ROTATE_90_CLOCKWISE),
}


def reading(img):
    o = measure_content_orientation(img, PX, PY)
    if o is None:
        px, py = measure_content_scale(img)
        o = {"rotation_deg": 0.0, "scale_x": px / PX, "scale_y": py / PY}
    return o


fig = plt.figure(figsize=(10.0, 4.7))
gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.08], hspace=0.42)
top = gs[0].subgridspec(1, 7, wspace=0.12)
bot = gs[1].subgridspec(1, 5, wspace=0.15, width_ratios=[1.6, 1, 1.6, 1, 1])


def show(ax, img, title, cap=None, grey=False, vmax=None):
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax.imshow(img, cmap="gray" if grey else None, vmin=0, vmax=vmax or 255,
              interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=8.5)
    if cap:
        ax.set_xlabel(cap, fontsize=7.5)


axes = [fig.add_subplot(top[0, i]) for i in range(7)]
show(axes[0], checker, "Stored input:\ninput", "checkerboard")
show(axes[1], flat, "Stored input:\ntarget", "uniform grey")
# Index frames hold pixel value i; the display is stretched so they can be told apart.
for ax, (i, img) in zip(axes[2:5], zip((3, 11, 19), index)):
    show(ax, img[..., 0], f"Frame window:\nframe {i}", f"pixels = {i}", grey=True, vmax=24)
show(axes[5], sig, "Range and\nchannels", "0/255 anchors,\none pattern per channel")
show(axes[6], grid[:240, :240], "Content scale", f"period {PX} px across,\n{PY} px down")

panels = [("Source 16:9", grid)] + list(ops.items())
for i, (name, img) in enumerate(panels):
    ax = fig.add_subplot(bot[0, i])
    if name.startswith("Source"):
        cap = f"${w}\\times{h}$"
    else:
        r = reading(img)
        cap = (f"${img.shape[1]}\\times{img.shape[0]}$\n"
               f"width $\\times${r['scale_x']:.2f}\nheight $\\times${r['scale_y']:.2f}")
        if abs(r["rotation_deg"]) > 1:
            cap += f", turned {abs(r['rotation_deg']):.0f}$^\\circ$"
    show(ax, img, name, cap)

fig.savefig(HERE / "fig_probes.pdf", bbox_inches="tight")
print("wrote fig_probes.pdf")
