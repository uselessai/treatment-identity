#!/usr/bin/env python
"""Study 6, second part: other tools on three delivery defects of other projects.

  zerodce   the loader resizes every image to 256x256, whatever its shape
  gfpgan    the configuration sets out_size 512, and the loader delivers the stored size
  ddcolor   with image mixing on, most mixed samples are identical to the unmixed ones

As in ``compare_tools.py``, each tool is given what it normally consumes, and
each released loader is set next to a copy with one line corrected, which
differs from it in the defect only. Every loader runs in its own process, from
a clean clone of the project at the audited commit, because the projects share
module names.

    python studies/compare_tools_external.py --projects /path/with/zZero-DCE,zGFPGAN,zDDColor \
        --natural /path/to/DAVIS --out studies/data_1.14.0/comparison
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE / "adapters"), str(HERE.parent)]

DRAWS = 100

CASES = {
    "zerodce": dict(
        folder="zZero-DCE", commit="e0f4adc54d0f", sub="Zero-DCE_code",
        source="Zero-DCE_code/dataloader.py",
        old="data_lowlight = data_lowlight.resize((self.size,self.size), Image.ANTIALIAS)",
        new=("data_lowlight = data_lowlight.crop(((data_lowlight.size[0]-min(data_lowlight.size))//2, "
             "(data_lowlight.size[1]-min(data_lowlight.size))//2, "
             "(data_lowlight.size[0]+min(data_lowlight.size))//2, "
             "(data_lowlight.size[1]+min(data_lowlight.size))//2))"
             ".resize((self.size,self.size), Image.ANTIALIAS)"),
        kinds=("fixture", "natural")),
    "gfpgan": dict(
        folder="zGFPGAN", commit="7552a7791caa", sub="",
        source="gfpgan/data/ffhq_degradation_dataset.py",
        old="        img_gt = imfrombytes(img_bytes, float32=True)\n",
        new=("        img_gt = imfrombytes(img_bytes, float32=True)\n"
             "        img_gt = cv2.resize(img_gt, (self.out_size, self.out_size), "
             "interpolation=cv2.INTER_AREA)\n"),
        kinds=("fixture", "natural_512", "natural_768")),
    "ddcolor": dict(
        folder="zDDColor", commit="2adb63f2656a", sub="",
        source="basicsr/data/lab_dataset.py",
        old="img_gt[:, bbx1:bbx2, bby1:bby2] = cmix_img[:, bbx1:bbx2, bby1:bby2]",
        new="img_gt[bbx1:bbx2, bby1:bby2, :] = cmix_img[bbx1:bbx2, bby1:bby2, :]",
        kinds=("fixture", "natural")),
}


# ------------------------------------------------------------------ inputs
def natural_images(davis: Path, n: int = DRAWS) -> list[Path]:
    """n frames of natural video, ten from each of the first clips."""
    out = []
    for clip in sorted(p for p in davis.iterdir() if p.is_dir()):
        frames = sorted(clip.glob("*.png")) + sorted(clip.glob("*.jpg"))
        out += frames[:: max(1, len(frames) // 10)][:10]
        if len(out) >= n:
            break
    return out[:n]


def write_images(paths: list[Path], dest: Path, square: int | None = None) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for i, p in enumerate(paths):
        img = cv2.imread(str(p))
        if square:
            h, w = img.shape[:2]
            s = min(h, w)
            img = img[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]
            img = cv2.resize(img, (square, square), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(dest / f"{i:08d}.png"), img)


def features(arrays: list[np.ndarray]):
    """Per-sample summary statistics and the delivered size."""
    import pandas as pd
    rows = []
    for a in arrays:
        a = np.asarray(a, dtype=np.float32)
        hwc = a if a.ndim == 3 and a.shape[-1] <= 4 else np.moveaxis(a, 0, -1)
        row = {"height": hwc.shape[0], "width": hwc.shape[1],
               "mean": float(hwc.mean()), "std": float(hwc.std()),
               "min": float(hwc.min()), "max": float(hwc.max()),
               "row_diff": float(np.abs(np.diff(hwc, axis=0)).mean()),
               "col_diff": float(np.abs(np.diff(hwc, axis=1)).mean())}
        for c in range(hwc.shape[-1]):
            row[f"channel{c}_mean"] = float(hwc[..., c].mean())
        rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ worker
def worker(case: str, root: Path, davis: Path, out: Path, seed: int) -> None:
    """Run one loader: deliveries on each input, the two tests, and DDC."""
    from treatment_identity.adapter import LoaderSpec
    from treatment_identity.fixtures import make_scale_grid
    from treatment_identity.profile import observe_content
    from treatment_identity.seeding import seed_all

    nat = natural_images(davis)
    res: dict = {"hand_tests": {}, "ddc": {}}
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)

        def squeeze(a):
            a = np.asarray(a)
            return a[0] if a.ndim == 4 else a

        if case == "zerodce":
            from adapters_zerodce import ZeroDceAdapter
            ad = ZeroDceAdapter(root=root / CASES[case]["sub"])
            for kind in CASES[case]["kinds"]:
                d = td / kind
                if kind == "fixture":
                    make_scale_grid(d / "GT", n_frames=DRAWS, shape=(480, 854),
                                    period_x=24, period_y=40)
                else:
                    write_images(nat, d / "GT" / "clip_000")
                seed_all(seed)
                loader = ad.build(LoaderSpec(gt_root=d / "GT", lq_root=d / "GT", seed=seed))
                arrays = [squeeze(ad.sample(loader, i).lq) for i in range(DRAWS)]
                features(arrays).to_csv(out / f"{kind}.csv", index=False)
                sizes = {tuple(features([a])[["height", "width"]].iloc[0]) for a in arrays}
                res["hand_tests"][kind] = {"delivered_size_is_256": sizes == {(256, 256)}}
            c = observe_content(ad, td / "ddc")
            res["ddc"] = {"behaviour": c.get("behaviour"), "note": c.get("note"),
                          "sizes": c.get("sizes")}

        elif case == "gfpgan":
            from adapters_gfpgan import GfpganAdapter
            ad = GfpganAdapter(root=root)
            for kind in CASES[case]["kinds"]:
                d = td / kind
                if kind == "fixture":
                    make_scale_grid(d / "GT", n_frames=DRAWS, shape=(768, 768),
                                    period_x=24, period_y=40)
                else:
                    write_images(nat, d / "GT" / "clip_000", square=int(kind.split("_")[1]))
                seed_all(seed)
                loader = ad.build(LoaderSpec(gt_root=d / "GT", lq_root=d / "GT", seed=seed))
                arrays = [squeeze(ad.sample(loader, i).gt) for i in range(DRAWS)]
                features(arrays).to_csv(out / f"{kind}.csv", index=False)
                sizes = {tuple(int(v) for v in a.shape[:2]) for a in arrays}
                res["hand_tests"][kind] = {"delivered_size_is_out_size": sizes == {(512, 512)},
                                           "delivered_sizes": sorted(sizes)}
            c = observe_content(ad, td / "ddc")
            res["ddc"] = {"behaviour": c.get("behaviour"), "sizes": c.get("sizes"),
                          "configured": [512, 512]}

        elif case == "ddcolor":
            from adapters_ddcolor_lab import DDColorLabAdapter, _to_hwc
            from sweep_settings import sweep
            ad = DDColorLabAdapter(root=root)
            for kind in CASES[case]["kinds"]:
                d = td / kind
                if kind == "fixture":
                    meta = ad.write_fixture(d)
                else:
                    write_images(nat, d / "gt")
                    meta = d / "meta_info.txt"
                    meta.write_text("\n".join(str(p) for p in sorted((d / "gt").glob("*.png"))) + "\n")
                on = ad._cls(ad.opt(meta, do_cutmix=True))
                off = ad._cls(ad.opt(meta, do_cutmix=False))
                arrays, changed, raised = [], 0, 0
                for i in range(DRAWS):
                    try:
                        seed_all(seed * 1000 + i)
                        a = _to_hwc(on[i % len(on)]["gt"])
                    except IndexError:
                        # the loader draws the partner image one past the end
                        raised += 1
                        continue
                    seed_all(seed * 1000 + i)
                    b = _to_hwc(off[i % len(off)]["gt"])
                    arrays.append(a)
                    changed += int(not np.array_equal(a, b))
                features(arrays).to_csv(out / f"{kind}.csv", index=False)
                res["hand_tests"][kind] = {"some_sample_differs_with_mixing": changed > 0,
                                           "samples_changed": changed,
                                           "draws": len(arrays), "raised": raised}
            row = next(r for r in sweep(ad, td / "ddc", DRAWS) if r["setting"] == "do_cutmix")
            res["ddc"] = {k: row[k] for k in ("compared", "branch_reached", "changed_when_reached",
                                              "unchanged_when_reached", "raised")}
    (out / "worker.json").write_text(json.dumps(res, indent=2, default=str) + "\n")


# ------------------------------------------------------------------ driver
def clone(project: Path, commit: str, dest: Path) -> Path:
    subprocess.run(["git", "clone", "-q", str(project), str(dest)], check=True)
    subprocess.run(["git", "-C", str(dest), "checkout", "-q", commit], check=True)
    return dest


def static_analysis(source: Path, old: str) -> dict:
    """Diagnostics of ruff and pylint on the loader file, and those on the defect line."""
    text = source.read_text().splitlines()
    first = old.strip().splitlines()[0].strip()
    line = next(i + 1 for i, ln in enumerate(text) if first in ln)

    def ruff(*select):
        cmd = ["ruff", "check", "--no-cache", "--isolated", "--output-format=json", *select, str(source)]
        return json.loads(subprocess.run(cmd, capture_output=True, text=True).stdout or "[]")

    default, everything = ruff(), ruff("--select", "ALL")
    pl = json.loads(subprocess.run([sys.executable, "-m", "pylint", "--output-format=json",
                                    "--disable=C,R", str(source)],
                                   capture_output=True, text=True).stdout or "[]")
    at = lambda ds, key: [f"{d.get('code') or d.get('message-id')} {d['message']}"
                          for d in ds if key(d) == line]
    return {"defect_line": line,
            "ruff_default": {"diagnostics": len(default),
                             "on_defect_line": at(default, lambda d: d["location"]["row"])},
            "ruff_all_rules": {"diagnostics": len(everything),
                               "on_defect_line": at(everything, lambda d: d["location"]["row"])},
            "pylint": {"diagnostics": len(pl), "on_defect_line": at(pl, lambda d: d["line"])}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--projects", type=Path)
    ap.add_argument("--natural", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--worker", nargs=3, metavar=("CASE", "ROOT", "SEED"))
    args = ap.parse_args()
    if args.worker:
        case, root, seed = args.worker
        args.out.mkdir(parents=True, exist_ok=True)
        worker(case, Path(root), args.natural, args.out, int(seed))
        return 0

    import pandas as pd
    from compare_tools import deepchecks_integrity, evidently_drift, tool_versions
    args.out.mkdir(parents=True, exist_ok=True)
    result = {"versions": tool_versions(), "draws": DRAWS, "cases": {}}
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for case, cfg in CASES.items():
            released = clone(args.projects / cfg["folder"], cfg["commit"], td / f"{case}_released")
            corrected = clone(args.projects / cfg["folder"], cfg["commit"], td / f"{case}_corrected")
            src = corrected / cfg["source"]
            text = src.read_text()
            assert text.count(cfg["old"]) == 1, (case, text.count(cfg["old"]))
            src.write_text(text.replace(cfg["old"], cfg["new"]))
            entry = {"static": static_analysis(released / cfg["source"], cfg["old"]), "runs": {}}
            feats = {}
            for label, root, seed in (("released", released, 0), ("corrected", corrected, 0),
                                      ("corrected_again", corrected, 1)):
                wdir = td / f"{case}_{label}_out"
                p = subprocess.run([sys.executable, __file__, "--natural", str(args.natural),
                                    "--out", str(wdir), "--worker", case, str(root), str(seed)],
                                   capture_output=True, text=True)
                if p.returncode:
                    print(p.stderr[-3000:])
                    raise SystemExit(f"{case} {label} failed")
                run = json.loads((wdir / "worker.json").read_text())
                for kind in cfg["kinds"]:
                    df = feats[(label, kind)] = pd.read_csv(wdir / f"{kind}.csv")
                    df.to_csv(args.out / f"features_{case}_{label}_{kind}.csv", index=False)
                    run.setdefault("deepchecks", {})[kind] = deepchecks_integrity(df)
                entry["runs"][label] = run
                print(case, label, json.dumps(run["ddc"])[:300], json.dumps(run["hand_tests"]))
            entry["evidently"] = {}
            for kind in cfg["kinds"]:
                ref = feats[("corrected", kind)]
                entry["evidently"][kind] = {
                    "released_vs_corrected": evidently_drift(ref, feats[("released", kind)]),
                    "corrected_two_runs": evidently_drift(ref, feats[("corrected_again", kind)])}
            result["cases"][case] = entry
    (args.out / "C_tool_comparison_external.json").write_text(json.dumps(result, indent=2) + "\n")
    print("wrote", args.out / "C_tool_comparison_external.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
