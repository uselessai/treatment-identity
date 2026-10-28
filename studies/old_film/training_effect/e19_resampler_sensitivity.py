#!/usr/bin/env python
"""Does E19's LPIPS reversal depend on the native-grid resampler?

``e19_geometry_sensitivity.py`` re-scores E19 in the ground truth's native
geometry and finds that all three paired LPIPS differences change sign. That
result rests on one implementation choice the article had not stated: the
corrected column brings the 640x368 output back to 320x180 with OpenCV's
``INTER_AREA``.

INTER_AREA is the defensible default for downsampling --- it integrates over the
source footprint instead of point-sampling it --- but a sign reversal that
depends on a single resampler is a different finding from one that does not, and
LPIPS is exactly the metric most sensitive to resampling. So the choice has to be
tested rather than asserted.

The legacy direction is *not* varied. Stretching the ground truth up to the
squeezed geometry with INTER_AREA is what the released harness does; it is the
published path, not a choice of ours, and it stays fixed as the control. Only the
corrected direction --- the one we introduce --- is swept:

    INTER_AREA      area-weighted average    (the reported column)
    INTER_CUBIC     4x4 bicubic
    INTER_LANCZOS4  8x8 Lanczos

If the LPIPS sign flips under all three, the reversal is a property of the
geometry rather than of the resampler. If it does not, the article must say so.

    python e19_resampler_sensitivity.py            # score and write the CSV
    python e19_resampler_sensitivity.py --check    # verify the CSV is current
    python e19_resampler_sensitivity.py --manifest-only --project-root PATH
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

import cv2

PAPER = Path(__file__).resolve().parents[1]
PROJECT = PAPER.parent
RESULTS = PROJECT / "metricas" / "fair_eval" / "e19_prefix_vs_window_strict_v1"
INFERENCE = RESULTS / "inference"
GT_ROOT = PROJECT / "dataset" / "fair_testset" / "GT"
FROZEN = RESULTS / "e19_paired_deltas_by_seed.csv"
OUT = PAPER / "data" / "e19_resampler_sensitivity.csv"

METRICS = ("PSNR", "SSIM", "LPIPS", "DISTS")
BETTER = {"PSNR": "higher", "SSIM": "higher", "LPIPS": "lower", "DISTS": "lower"}
SEED_ORDER = ("2021", "7", "424242")
ARMS = ("legacy", "window")
#: Corrected-direction resamplers. The first is the one the article reports.
RESAMPLERS = {
    "INTER_AREA": cv2.INTER_AREA,
    "INTER_CUBIC": cv2.INTER_CUBIC,
    "INTER_LANCZOS4": cv2.INTER_LANCZOS4,
}
TOL = {"PSNR": 5e-3, "SSIM": 5e-5, "LPIPS": 5e-5, "DISTS": 5e-5}
EXPECTED_OUTPUT_HW = (368, 640)
EXPECTED_REFERENCE_HW = (180, 320)


def configure_project(project_root: Path) -> None:
    """Resolve large, unbundled inputs from an explicit experiment root."""
    global PROJECT, RESULTS, INFERENCE, GT_ROOT, FROZEN
    PROJECT = Path(project_root).resolve()
    RESULTS = PROJECT / "metricas" / "fair_eval" / "e19_prefix_vs_window_strict_v1"
    INFERENCE = RESULTS / "inference"
    GT_ROOT = PROJECT / "dataset" / "fair_testset" / "GT"
    FROZEN = RESULTS / "e19_paired_deltas_by_seed.csv"


def arm_dir(arm: str, seed: str) -> Path:
    """Clip root for one arm, resolved by search rather than hard-coded."""
    base = INFERENCE / f"e19_{arm}_s{seed}"
    candidates = [path for path in sorted(base.glob("test_results_*/LQ"))
                  if path.is_dir()]
    if len(candidates) != 1:
        raise SystemExit(
            f"expected exactly one inference root below {base}, found "
            f"{len(candidates)}: {candidates}")
    return candidates[0]


def paired_manifest(root: Path) -> dict[str, list[tuple[Path, Path]]]:
    """Return every output/reference pair, rejecting incomplete coverage."""
    if not root.is_dir():
        raise SystemExit(f"missing inference root: {root}")
    if not GT_ROOT.is_dir():
        raise SystemExit(f"missing reference root: {GT_ROOT}")
    output_clips = {path.name: path for path in root.iterdir() if path.is_dir()}
    reference_clips = {path.name: path for path in GT_ROOT.iterdir() if path.is_dir()}
    if set(output_clips) != set(reference_clips):
        missing = sorted(set(reference_clips) - set(output_clips))
        extra = sorted(set(output_clips) - set(reference_clips))
        raise SystemExit(f"clip manifest mismatch at {root}: missing={missing}, extra={extra}")
    if not output_clips:
        raise SystemExit(f"empty clip manifest at {root}")

    manifest: dict[str, list[tuple[Path, Path]]] = {}
    for clip in sorted(output_clips):
        outputs = {path.name: path for path in output_clips[clip].glob("*.png")}
        references = {path.name: path for path in reference_clips[clip].glob("*.png")}
        if set(outputs) != set(references):
            missing = sorted(set(references) - set(outputs))
            extra = sorted(set(outputs) - set(references))
            raise SystemExit(
                f"frame manifest mismatch at {root}/{clip}: "
                f"missing={missing}, extra={extra}")
        if not outputs:
            raise SystemExit(f"empty frame manifest at {root}/{clip}")
        manifest[clip] = [(outputs[name], references[name]) for name in sorted(outputs)]
    return manifest


def load_pair(frame: Path, gt_path: Path):
    """Read one pair and enforce the two geometry states being contrasted."""
    out_bgr = cv2.imread(str(frame))
    gt_bgr = cv2.imread(str(gt_path))
    if out_bgr is None or gt_bgr is None:
        raise SystemExit(f"unreadable pair: {frame}, {gt_path}")
    if out_bgr.shape[:2] != EXPECTED_OUTPUT_HW:
        raise SystemExit(
            f"unexpected output shape at {frame}: {out_bgr.shape[:2]}, "
            f"expected {EXPECTED_OUTPUT_HW}")
    if gt_bgr.shape[:2] != EXPECTED_REFERENCE_HW:
        raise SystemExit(
            f"unexpected reference shape at {gt_path}: {gt_bgr.shape[:2]}, "
            f"expected {EXPECTED_REFERENCE_HW}")
    return out_bgr, gt_bgr


def validate_manifests() -> None:
    """Check exact pairing and image geometry without loading metric models."""
    for seed in SEED_ORDER:
        for arm in ARMS:
            manifest = paired_manifest(arm_dir(arm, seed))
            for pairs in manifest.values():
                for frame, gt_path in pairs:
                    load_pair(frame, gt_path)
            frames = sum(len(pairs) for pairs in manifest.values())
            print(f"{arm} seed {seed}: {len(manifest)} clips, {frames} paired "
                  f"frames, output {EXPECTED_OUTPUT_HW}, reference "
                  f"{EXPECTED_REFERENCE_HW}")


def load_engine():
    """The same Metrics class the released harness uses, not a reimplementation."""
    sys.path.insert(0, str(PROJECT / "herramientas"))
    import torch
    from eval_fair_testset import Metrics
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return Metrics(device), device


def score_frame(engine, out_bgr, gt_bgr) -> dict:
    """One frame under the legacy geometry and under each corrected resampler."""
    res = {}
    # legacy control: GT stretched up to the output's squeezed geometry, the way
    # the released harness does it. Fixed, never swept.
    gt_up = cv2.resize(gt_bgr, (out_bgr.shape[1], out_bgr.shape[0]),
                       interpolation=cv2.INTER_AREA)
    res["legacy"] = engine.compute(out_bgr, gt_up, allow_resize=False)
    # corrected: output back to the ground truth's own geometry, once, per resampler
    for name, flag in RESAMPLERS.items():
        out_native = cv2.resize(out_bgr, (gt_bgr.shape[1], gt_bgr.shape[0]),
                                interpolation=flag)
        res[name] = engine.compute(out_native, gt_bgr, allow_resize=False)
    return res


def score_arm(engine, arm: str, seed: str) -> dict:
    """Clip-level means for one arm, under every geometry."""
    root = arm_dir(arm, seed)
    geoms = ["legacy", *RESAMPLERS]
    per_clip = {g: [] for g in geoms}
    for clip, pairs in paired_manifest(root).items():
        acc = {g: [] for g in geoms}
        for frame, gt_path in pairs:
            out_bgr, gt_bgr = load_pair(frame, gt_path)
            r = score_frame(engine, out_bgr, gt_bgr)
            for g in geoms:
                acc[g].append(r[g])
        for g in geoms:
            per_clip[g].append(
                {m: statistics.fmean(x[m] for x in acc[g]) for m in METRICS})
    return {g: {m: statistics.fmean(c[m] for c in per_clip[g]) for m in METRICS}
            for g in geoms}


def frozen_deltas() -> dict:
    """The published window-minus-legacy deltas, by metric and seed."""
    out: dict[str, dict[str, float]] = {}
    with open(FROZEN) as fh:
        for r in csv.DictReader(fh):
            out.setdefault(r["metric"], {})[r["seed"]] = float(
                r["delta_window_minus_legacy"])
    return out


def build_rows() -> tuple[list[dict], list[str]]:
    engine, device = load_engine()
    print(f"scoring on {device}", file=sys.stderr)
    arms: dict[tuple[str, str], dict] = {}
    for seed in SEED_ORDER:
        for arm in ARMS:
            print(f"  {arm} s{seed} …", file=sys.stderr)
            arms[(arm, seed)] = score_arm(engine, arm, seed)

    published = frozen_deltas()
    rows, problems = [], []
    for m in METRICS:
        for seed in SEED_ORDER:
            leg_d = (arms[("window", seed)]["legacy"][m]
                     - arms[("legacy", seed)]["legacy"][m])
            pub = published[m][seed]
            drift = abs(leg_d - pub)
            if drift > TOL[m]:
                problems.append(
                    f"{m} seed {seed}: legacy reproduction {leg_d:+.6f} differs "
                    f"from the frozen {pub:+.6f} by {drift:.6f} (tol {TOL[m]})")
            row = {"metric": m, "seed": seed, "better_direction": BETTER[m],
                   "published_delta": f"{pub:.10f}",
                   "legacy_delta_reproduced": f"{leg_d:.10f}"}
            for name in RESAMPLERS:
                cor_d = (arms[("window", seed)][name][m]
                         - arms[("legacy", seed)][name][m])
                row[f"corrected_delta_{name}"] = f"{cor_d:.10f}"
                row[f"sign_flips_{name}"] = str(
                    (leg_d > 0) != (cor_d > 0) if leg_d != 0 and cor_d != 0 else False)
                row[f"favours_window_{name}"] = str(
                    (cor_d > 0) == (BETTER[m] == "higher"))
            rows.append(row)
    return rows, problems


def write_rows(rows: list[dict]) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def summarise(rows: list[dict]) -> dict:
    """Per metric: does every resampler agree on the direction after correction?"""
    out = {}
    for m in METRICS:
        sub = [r for r in rows if r["metric"] == m]
        entry = {"n_pairs": len(sub)}
        for name in RESAMPLERS:
            entry[name] = {
                "mean_corrected_delta": statistics.fmean(
                    float(r[f"corrected_delta_{name}"]) for r in sub),
                "signs_flipped_vs_legacy": sum(
                    1 for r in sub if r[f"sign_flips_{name}"] == "True"),
                "favours_window": sum(
                    1 for r in sub if r[f"favours_window_{name}"] == "True"),
            }
        entry["all_resamplers_agree_per_seed"] = all(
            len({r[f"favours_window_{n}"] for n in RESAMPLERS}) == 1 for r in sub)
        out[m] = entry
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="verify the released CSV reproduces from the frames")
    ap.add_argument("--project-root", type=Path, default=PROJECT,
                    help="root containing metricas/, dataset/ and herramientas/")
    ap.add_argument("--manifest-only", action="store_true",
                    help="verify exact clip/frame pairing without scoring metrics")
    args = ap.parse_args()
    configure_project(args.project_root)

    if args.manifest_only:
        validate_manifests()
        return 0

    rows, problems = build_rows()
    if problems:
        print("LEGACY REPRODUCTION FAILED --- corrected figures are not trustworthy:",
              file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        return 2

    if args.check:
        if not OUT.is_file():
            print(f"missing {OUT}", file=sys.stderr)
            return 1
        if [dict(r) for r in csv.DictReader(open(OUT))] != rows:
            print(f"STALE: {OUT} does not match a fresh scoring", file=sys.stderr)
            return 1
        print(f"{OUT.name} is current; legacy column reproduces the frozen deltas")
        return 0

    write_rows(rows)
    print(json.dumps(summarise(rows), indent=2))
    print(f"wrote {OUT}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
