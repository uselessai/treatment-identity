#!/usr/bin/env python3
"""One photograph at five source shapes, passed through the sixteen released loaders.

Each delivered panel is the array the loader returned through its DDC adapter,
unmodified. The scale factors printed under it are the content-scale readings
of the same loader on the spatial-period fixture at that source shape, read
from evidence/campaigns/data_1.11.0/P_profiles.json; nothing is drawn by hand.

    python gen_aspect_figure.py DDC_REPO PHOTO
"""
import json
import sys
import tempfile
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO, PHOTO = Path(sys.argv[1]), Path(sys.argv[2])
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(REPO), str(REPO / "studies"), str(REPO / "studies/adapters")]
import campaign_P_portability as camp                              # noqa: E402
from treatment_identity.adapter import LoaderSpec                  # noqa: E402
from treatment_identity.fixtures import SCALE_PROBE_SOURCES, _write_clip  # noqa: E402

# Reshapes and the rescale first, then crops. Projects that ship a top-level
# package of the same name (``data``) are imported in an order that works.
LOADERS = (("zerodce", "Zero-DCE"), ("ddcolor_lab", "DDColor"), ("e2fgvi", "E2FGVI"),
           ("propainter", "ProPainter"), ("diffueraser", "DiffuEraser"),
           ("kair_video", "KAIR video"), ("mmagic", "MMagic"), ("basicsr_reds", "BasicSR REDS"),
           ("gfpgan", "GFPGAN"), ("lama", "LaMa"), ("vrt", "VRT"),
           ("kair_blindsr", "KAIR BlindSR"), ("uformer", "Uformer"), ("mprnet", "MPRNet"),
           ("pix2pix_colorization", "pix2pix colour"), ("restormer", "Restormer"))
SHAPES = {"w169": "16:9", "w43": "4:3", "w11": "1:1", "w34": "3:4", "w916": "9:16"}
# The profiles of Study 3, from the DDC repository or the manuscript's copy.
_SRC = next(p for p in (REPO / "studies/data_1.11.0/P_profiles.json",
                        HERE.parent / "evidence/campaigns/data_1.11.0/P_profiles.json")
            if p.exists())
PROFILES = {p["subject"]: p for p in json.loads(_SRC.read_text())}


def reading(subject, aspect):
    for r in PROFILES[subject]["content"].get("evidence", {}).get("readings", []):
        if abs(r["source_aspect"] - aspect) < 0.02:
            return r
    return None


def to_image(a):
    a = np.asarray(a, dtype=np.float64)
    while a.ndim > 3:
        a = a[0]
    if a.shape[0] in (1, 3) and a.shape[-1] not in (1, 3):
        a = np.transpose(a, (1, 2, 0))
    lo, hi = (-1.0, 1.0) if a.min() < -0.01 else (0.0, 1.0) if a.max() <= 1.0 else (0.0, 255.0)
    if a.ndim == 2 or a.shape[-1] == 1:
        a = np.repeat(a.reshape(a.shape[0], a.shape[1], 1), 3, axis=-1)
        if a.max() > 1.0 and a.max() <= 100.0 and a.min() >= 0.0:   # Lab lightness
            lo, hi = 0.0, 100.0
    return np.clip((a - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


def channels(a):
    a = np.asarray(a)
    while a.ndim > 3:
        a = a[0]
    if a.ndim == 2:
        return 1
    return a.shape[0] if a.shape[0] in (1, 2, 3) and a.shape[-1] not in (1, 2, 3) else a.shape[-1]


def viewable(sample):
    """The target when it is a picture; a colourisation loader's target holds
    only the two colour channels, so its lightness input is shown instead."""
    for stream in (sample.gt, sample.lq):
        if stream is not None and channels(stream) in (1, 3):
            return stream
    return sample.lq


photo = cv2.imread(str(PHOTO))
sources = {}
for tag, (h, w) in SCALE_PROBE_SOURCES:
    ph, pw = photo.shape[:2]
    s = min(ph / h, pw / w)
    ch, cw = int(h * s), int(w * s)
    y, x = (ph - ch) // 2, (pw - cw) // 2
    sources[tag] = cv2.resize(photo[y:y + ch, x:x + cw], (w, h), interpolation=cv2.INTER_AREA)

delivered = {}
for key, _ in LOADERS:
    cfg = camp.SUBJECTS[key]
    camp._isolate_shared_package_names(cfg["root"])
    adapter = getattr(__import__(cfg["module"]), cfg["cls"])()
    for tag, img in sources.items():
        # Some loaders need longer clips than others, as in the campaign.
        for n in (cfg.get("clip_length") or 32, 100, 200):
            try:
                with tempfile.TemporaryDirectory() as tmp:
                    for side in ("GT", "LQ"):
                        _write_clip(Path(tmp) / side, "clip_000", [img.copy() for _ in range(n)])
                    spec = LoaderSpec(gt_root=Path(tmp) / "GT", lq_root=Path(tmp) / "LQ",
                                      num_frames=cfg.get("num_frames", 5), seed=0)
                    loader = adapter.build(spec)
                    sample = adapter.sample(loader, 0)
                break
            except Exception as exc:                             # noqa: BLE001
                print(f"  {key} {tag}: {n} frames not enough ({type(exc).__name__})")
        delivered[key, tag] = to_image(viewable(sample))

shapes = list(SCALE_PROBE_SOURCES)
# Two blocks of eight loaders side by side, each headed by the sources, so the
# sixteen fit one page.
blocks = [list(LOADERS[:8]), list(LOADERS[8:])]
nrow, ncol = 1 + max(len(b) for b in blocks), len(shapes)
fig, axes = plt.subplots(nrow, 2 * ncol + 1, figsize=(7.2, 1.0 * nrow),
                         gridspec_kw={"width_ratios": [1] * ncol + [1.05] + [1] * ncol})
for ax in axes[:, ncol]:
    ax.axis("off")
for bi, block in enumerate(blocks):
    rows = [("Source", "Source")] + block
    for r in range(nrow):
        for c, (tag, (h, w)) in enumerate(shapes):
            ax = axes[r][c + bi * (ncol + 1)]
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if r >= len(rows):
                ax.axis("off")
                continue
            key, name = rows[r]
            img = (cv2.cvtColor(sources[tag], cv2.COLOR_BGR2RGB) if key == "Source"
                   else delivered[key, tag])                  # the source is read by OpenCV
            # Every panel is drawn at its true proportions in a common box.
            ax.imshow(img, extent=(0, img.shape[1], img.shape[0], 0))
            side = max(img.shape[:2])
            ax.set_xlim((img.shape[1] - side) / 2, (img.shape[1] + side) / 2)
            ax.set_ylim((img.shape[0] + side) / 2, (img.shape[0] - side) / 2)
            if r == 0:
                ax.set_title(SHAPES[tag], fontsize=7)
            cap = f"{img.shape[1]}$\\times${img.shape[0]}"
            rd = reading(key, w / h) if key != "Source" else None
            if rd is not None:
                cap += f"\n$\\times${rd['scale_x']:.2f} / $\\times${rd['scale_y']:.2f}"
            ax.text(0.5, -0.03, cap, transform=ax.transAxes, ha="center", va="top",
                    fontsize=4.6, linespacing=1.0)
            if c == 0:
                ax.text(-0.1, 0.5, name, transform=ax.transAxes, ha="right", va="center",
                        fontsize=6.2)
fig.subplots_adjust(left=0.085, right=0.995, top=0.975, bottom=0.03, hspace=0.62, wspace=0.08)
fig.savefig(HERE / "fig_aspects.pdf", bbox_inches="tight")
print("wrote fig_aspects.pdf")
