#!/usr/bin/env python3
"""What a released training loader delivers, seen on a photograph and measured on a fixture.

Both rows pass through the *same* released code: Zero-DCE's own
``lowlight_loader`` is imported unmodified and asked for a sample, so the middle
column is the tensor the model would receive. The periods under the second row
are read by treatment_identity.checks.measure_content_scale, and the fixture is
built by treatment_identity.fixtures.make_scale_grid. Nothing is drawn by hand.

    python gen_fixture_figure.py MECHANISM_SRC ZERODCE_CODE PHOTO
"""
import sys, tempfile, shutil
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MECH, ZDCE, PHOTO = (Path(a) for a in sys.argv[1:4])
sys.path.insert(0, str(MECH)); sys.path.insert(0, str(ZDCE))
from treatment_identity.fixtures import make_scale_grid           # noqa: E402
from treatment_identity.checks import measure_content_scale       # noqa: E402
from dataloader import lowlight_loader                            # noqa: E402

OUT = Path(__file__).resolve().parent / "fig_zerodce.pdf"


def through_loader(img: Image.Image) -> np.ndarray:
    """One sample from the released loader, converted back to viewable pixels."""
    with tempfile.TemporaryDirectory() as d:
        img.convert("RGB").save(Path(d) / "frame.jpg", quality=97)
        item = lowlight_loader(str(Path(d)) + "/")[0]        # C,H,W in [0,1]
    return (item.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)


def keep_aspect(img: Image.Image) -> np.ndarray:
    """The same square, reached by cropping instead of squeezing."""
    w, h = img.size
    s = min(w, h)
    box = ((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s)
    return np.asarray(img.crop(box).resize((256, 256), Image.LANCZOS))


def portrait(img):
    """The same content cropped to 9:16 before the loader sees it."""
    w, h = img.size
    nw = round(h * 9 / 16)
    left = (w - nw) // 2
    return img.crop((left, 0, left + nw, h))


photo = Image.open(PHOTO).convert("RGB")
with tempfile.TemporaryDirectory() as tmp:
    clip = make_scale_grid(Path(tmp), period_x=40, period_y=60)
    grid = Image.fromarray(clip.read(0))
    px, py = clip.period_x, clip.period_y
with tempfile.TemporaryDirectory() as tmp:
    # The fixture's 9:16 source is the one the campaign probes, not a crop.
    grid_tall = Image.fromarray(make_scale_grid(Path(tmp), shape=(854, 480),
                                                period_x=40, period_y=60).read(0))

rows = []
for name, img, tall in (("A photograph", photo, portrait(photo)),
                        ("The fixture", grid, grid_tall)):
    rows.append((name, img, [
        ("Source, $16{:}9$", np.asarray(img), True),
        ("Delivered", through_loader(img), False),
        ("If proportions were kept", keep_aspect(img), False),
        ("The same source at $9{:}16$", np.asarray(tall), True),
        ("Delivered", through_loader(tall), False),
    ]))


DH, CW, CH = 260, 660, 285


def display(arr, nearest):
    """Panels are shown at a common height with their true proportions, on
    identical canvases, so a squeezed delivery looks squeezed. The measured
    periods printed below come from the delivered array, not from this view."""
    h0, w0 = arr.shape[:2]
    dw = max(1, round(DH * w0 / h0))
    small = Image.fromarray(arr).resize(
        (dw, DH), Image.NEAREST if nearest else Image.LANCZOS)
    pad = np.full((CH, CW, 3), 255, np.uint8)
    y0, x0 = (CH - DH) // 2, (CW - dw) // 2
    pad[y0 - 2:y0 + DH + 2, x0 - 2:x0 + dw + 2] = 60
    pad[y0:y0 + DH, x0:x0 + dw] = np.asarray(small)
    return pad


fig, axes = plt.subplots(2, 5, figsize=(13.0, 4.9))
for r, (rowname, _, cols) in enumerate(rows):
    # Ratios are taken against the measured source, not the nominal period, so
    # the estimator's own bias cancels and the reference column reads exactly one.
    ref = measure_content_scale(cols[0][1]) if r == 1 else (None, None)
    for c, (title, arr, is_source) in enumerate(cols):
        ax = axes[r][c]
        ax.imshow(display(arr, r == 1)); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        if r == 0:
            ax.set_title(title, fontsize=10, pad=6)
        h, w = arr.shape[:2]
        cap = f"${w}\\times{h}$"
        if r == 1:
            mx, my = measure_content_scale(arr)
            cap += f"\nperiods {mx:.1f} / {my:.1f} px"
            if is_source:
                ref = (mx, my)
                cap += "\nreference"
                print(f"  {title:32} {w}x{h}  periods {mx:6.1f}/{my:5.1f}  reference")
            else:
                fx, fy = mx / ref[0], my / ref[1]
                cap += f"\n$\\times${fx:.2f} / $\\times${fy:.2f}"
                print(f"  {title:32} {w}x{h}  periods {mx:6.1f}/{my:5.1f}  "
                      f"x{fx:.2f}/x{fy:.2f}")
        ax.set_xlabel(cap, fontsize=9)
    axes[r][0].set_ylabel(rowname, fontsize=10)
fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight")
print("wrote", OUT)
