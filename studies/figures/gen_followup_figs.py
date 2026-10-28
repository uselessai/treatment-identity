#!/usr/bin/env python3
"""Pictures for the follow-up investigations of Study 4.

Every panel comes from the released loader, built through its DDC adapter:

  fig_vrt_split.pdf     the clips KAIR's video loader keeps for training, from
                        the REDS meta-info file, under the dataset name of the
                        released VRT configuration and under the name REDS;
  fig_ddcolor_mix.pdf   DDColor draws with image mixing off, on and changing the
                        sample, and on but leaving it unchanged;
  fig_sizes.pdf         a photograph stored larger than the configured size,
                        as GFPGAN and LaMa deliver it.

    python gen_followup_figs.py DDC_REPO PHOTO [FIGURE ...]
"""
import random
import sys
import tempfile
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

REPO, PHOTO = Path(sys.argv[1]), Path(sys.argv[2])
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(REPO), str(REPO / "studies"), str(REPO / "studies/adapters")]
import campaign_P_portability as camp                              # noqa: E402

KAIR = Path(camp.SUBJECTS["kair_video"]["root"])
DDCOLOR_IMAGES = Path(camp.SUBJECTS["ddcolor_lab"]["root"]) / "assets/test_images"


def adapter(key):
    cfg = camp.SUBJECTS[key]
    camp._isolate_shared_package_names(cfg["root"])
    return getattr(__import__(cfg["module"]), cfg["cls"])()


# --------------------------------------------------------------------- VRT
def fig_vrt_split():
    ad = adapter("vrt")
    meta = KAIR / "data/meta_info/meta_info_REDS_GT.txt"
    clips = [line.split()[0] for line in meta.read_text().splitlines() if line.strip()]
    base = {"dataroot_gt": str(KAIR), "dataroot_lq": str(KAIR), "meta_info_file": str(meta),
            "filename_tmpl": "08d", "filename_ext": "png", "val_partition": "REDS4",
            "test_mode": False, "io_backend": {"type": "disk"}, "num_frame": 6,
            "gt_size": 256, "interval_list": [1], "random_reverse": False,
            "use_hflip": True, "use_rot": True, "scale": 4, "n_channels": 3,
            "phase": "train"}
    kept = {}
    for label, name in (("released", "train_dataset"), ("reds", "REDS")):
        ds = ad._cls(dict(base, name=name))
        kept[label] = {k.split("/")[0] for k in ds.keys}
    test = ["000", "011", "015", "020"]
    print(f"VRT: {len(clips)} clips listed; released keeps {len(kept['released'])}, "
          f"name REDS keeps {len(kept['reds'])}")

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 2.2))
    rows = [("Dataset named REDS", kept["reds"]),
            ("Released VRT configuration", kept["released"])]
    cols = 54
    for ax, (title, keep) in zip(axes, rows):
        for i, c in enumerate(clips):
            x, y = i % cols, i // cols
            colour = "#4C72B0" if c in keep else "#DDDDDD"
            ax.add_patch(Rectangle((x, -y), 0.9, 0.9, color=colour, lw=0))
            if c in test:
                ax.add_patch(Rectangle((x - 0.08, -y - 0.08), 1.06, 1.06, fill=False,
                                       ec="#C44E52", lw=1.6))
        ax.set_xlim(-0.3, cols + 0.2); ax.set_ylim(-len(clips) // cols - 0.2, 1.2)
        ax.set_aspect("equal"); ax.axis("off")
        n_test = sum(c in keep for c in test)
        ax.set_title(f"{title}: {len(keep)} clips for training, "
                     f"{n_test} of them test clips", fontsize=8.5, loc="left")
    fig.tight_layout()
    fig.savefig(HERE / "fig_vrt_split.pdf", bbox_inches="tight")


# ------------------------------------------------------------------ DDColor
def lab_to_rgb(item):
    """The delivered L and ab channels, back to an RGB picture."""
    from skimage import color            # what the loader's rgb2lab inverts
    l = item["lq"].numpy().transpose(1, 2, 0)[..., :1]
    ab = item["gt"].numpy().transpose(1, 2, 0)
    rgb = color.lab2rgb(np.concatenate([l, ab], axis=-1))
    return np.clip(np.asarray(rgb) * 255, 0, 255).astype(np.uint8)


def fig_ddcolor_mix():
    """Draws of the released loader only: mixing off, mixing on that changes
    the sample, and mixing on that leaves it identical."""
    ad = adapter("ddcolor_lab")
    photos = sorted(p for p in DDCOLOR_IMAGES.iterdir()
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png"))[:6]
    with tempfile.TemporaryDirectory() as tmp:
        gt = Path(tmp) / "gt"; gt.mkdir()
        paths = []
        for i, p in enumerate(photos):
            img = cv2.imread(str(p))
            s = min(img.shape[:2]); y, x = (img.shape[0] - s) // 2, (img.shape[1] - s) // 2
            q = gt / f"img_{i:03d}.png"
            cv2.imwrite(str(q), cv2.resize(img[y:y + s, x:x + s], (320, 320)))
            paths.append(str(q))
        meta = Path(tmp) / "meta_info.txt"
        meta.write_text("\n".join(paths) + "\n")

        def draw(seed, do_cutmix):
            random.seed(seed); np.random.seed(seed)
            ds = ad._cls(ad.opt(meta, do_cutmix=do_cutmix))
            ds.paths = ds.paths[:len(photos) - 1]   # randint(0, len) is inclusive
            return ds[0]

        changed = unchanged = None
        for seed in range(400):
            before = ad.branch_entries
            try:
                mixed = draw(seed, True)
            except Exception:
                continue
            if ad.branch_entries == before:
                continue
            plain = draw(seed, False)
            same = np.array_equal(mixed["gt"].numpy(), plain["gt"].numpy())
            if same and unchanged is None:
                unchanged = mixed
            if not same and changed is None:
                changed, base = mixed, plain
            if changed is not None and unchanged is not None:
                break
        print(f"DDColor: seeds searched up to {seed}")
        panels = [("Mixing off", lab_to_rgb(base)),
                  ("Mixing on: sample changed", lab_to_rgb(changed)),
                  ("Mixing on: sample unchanged", lab_to_rgb(unchanged))]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6))
    for ax, (title, img) in zip(axes, panels):
        ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(HERE / "fig_ddcolor_mix.pdf", bbox_inches="tight")


# ------------------------------------------------------------ GFPGAN, LaMa
def fig_sizes():
    from treatment_identity.adapter import LoaderSpec
    from treatment_identity.fixtures import _write_clip
    photo = cv2.imread(str(PHOTO))
    s = min(photo.shape[:2]); y, x = (photo.shape[0] - s) // 2, (photo.shape[1] - s) // 2
    stored = cv2.resize(photo[y:y + s, x:x + s], (768, 768), interpolation=cv2.INTER_AREA)
    rows = []
    for key, name, configured in (("gfpgan", "GFPGAN", 512), ("lama", "LaMa", 256)):
        ad = adapter(key)
        with tempfile.TemporaryDirectory() as tmp:
            for side in ("GT", "LQ"):
                _write_clip(Path(tmp) / side, "clip_000", [stored.copy() for _ in range(8)])
            spec = LoaderSpec(gt_root=Path(tmp) / "GT", lq_root=Path(tmp) / "LQ",
                              num_frames=1, seed=0)
            sample = ad.sample(ad.build(spec), 0)
        a = np.asarray(sample.gt if sample.gt is not None else sample.lq, dtype=np.float64)
        while a.ndim > 3:
            a = a[0]
        if a.shape[0] in (1, 3) and a.shape[-1] not in (1, 3):
            a = a.transpose(1, 2, 0)
        lo, hi = (-1.0, 1.0) if a.min() < -0.01 else (0.0, 1.0) if a.max() <= 1.0 else (0, 255)
        img = np.clip((a - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
        print(f"{name}: delivered {img.shape[1]}x{img.shape[0]}, configured {configured}")
        rows.append((name, img, configured))

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.3))
    for ax, (name, img, configured) in zip(axes, rows):
        # The adapters return RGB for GFPGAN (img2tensor bgr2rgb) and LaMa.
        ax.imshow(img, extent=(0, img.shape[1], img.shape[0], 0))
        ax.add_patch(Rectangle((0, 0), configured, configured, fill=False,
                               ec="#C44E52", lw=1.8, ls="--"))
        ax.text(configured - 6, configured - 10, f"configured\n{configured}$\\times${configured}",
                color="#C44E52", fontsize=8, ha="right", va="bottom",
                bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.5))
        ax.set_xlim(0, img.shape[1]); ax.set_ylim(img.shape[0], 0)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{name}: delivered {img.shape[1]}$\\times${img.shape[0]}", fontsize=9)
    fig.tight_layout()
    fig.savefig(HERE / "fig_sizes.pdf", bbox_inches="tight")


if __name__ == "__main__":
    for f in sys.argv[3:] or ["fig_vrt_split", "fig_ddcolor_mix", "fig_sizes"]:
        globals()[f]()
        print("wrote", f + ".pdf")
