#!/usr/bin/env python
"""Campaign M --- seeded-defect study for the treatment-identity gates.

Study 2 of the Software Quality Journal manuscript (Section "Seeded-defect
detection and cost"). The package's own ``selftest.py`` shows that four
injected defects turn four gates red; that is a qualitative demonstration. This
script turns it into a measurement:

  * a defect-class x gate matrix, with the status each gate returns;
  * per-gate wall-clock cost;
  * false alarms: the conforming loader re-run over many fixture seeds;
  * and, deliberately, the defect classes that NO gate detects.

The misses are the point of reporting the matrix rather than a headline number.
A suite whose diagonal is full and whose off-diagonal is empty would be
suspicious; publishing the empty cells is what bounds the contract.

Outputs (under ``revistas/claude/data/``):
    M_seeded_defects_matrix.csv   defect x gate, one row per pair
    M_gate_cost.csv               per-gate wall clock over all runs
    M_false_alarms.csv            conforming loader over N fixture seeds
    M_summary.json                headline counts, for the manuscript macros

Usage:
    python campaign_M_seeded_defects.py [--seeds 20] [--out DIR]

CPU only, no GPU, no data from the study. Takes seconds.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent


def _locate_package() -> None:
    """Put ``treatment_identity`` on the path from wherever this script lives.

    The previous version resolved it by counting parent directories, which
    works from exactly one location. This script is deposited alongside the
    package as well as run from the manuscript tree, and a deposited script
    that cannot import its own package is the defect this campaign is written
    to detect -- lesson L5, in the artefact that reports it. Prefer an
    installed package; otherwise walk up looking for a real one.
    """
    try:
        import treatment_identity  # noqa: F401
        return
    except ModuleNotFoundError:
        pass
    for base in (HERE, *HERE.parents):
        for cand in (base, base / "treatment_identity"):
            if (cand / "treatment_identity" / "__init__.py").exists():
                sys.path.insert(0, str(cand))
                return
    raise SystemExit(
        "treatment_identity not found: install it (pip install -e .) or run "
        "this script from a tree that contains the package")


def _default_out() -> Path:
    """Where the CSVs go, from either of the two trees this script lives in.

    Beside the script when it ships inside the package (``campaigns/data``),
    one level up when it runs from the manuscript tree (``../data``). Resolved
    by looking, not by counting directories.
    """
    return HERE / "data" if (HERE / "data").is_dir() else HERE.parent / "data"


_locate_package()

from treatment_identity import (CheckResult, check_geometry,          # noqa: E402
                                check_operator_trace, check_precomputed_input,
                                check_separability, check_target_transforms,
                                check_temporal_window)
from treatment_identity.adapter import CallRecorder, LoaderSpec, Sample  # noqa: E402

OPS = {"blur", "downsample", "noise", "jpeg"}
DELIVERY_GATES = ("precomputed_input", "temporal_window",
                  "target_transforms", "operator_trace")


# ---------------------------------------------------------------------------
# The subject: a conforming loader, plus one defect per class.
# ---------------------------------------------------------------------------

#: defect -> (gate that must fire, one-line description for the manuscript)
DEFECTS: dict[str, tuple[str | None, str]] = {
    "none": (None,
             "conforming reference loader (negative control)"),
    "ignore_precomputed": ("precomputed_input",
                           "configuration flag is set and the loader does not read it"),
    "prefix_window": ("temporal_window",
                      "sampling index is computed and then discarded; a fixed prefix is opened"),
    "frozen_window": ("temporal_window",
                      "window is sampled once and reused for every subsequent item"),
    "double_sharpen": ("target_transforms",
                       "the target is transformed twice where one transformation is declared"),
    "no_sharpen": ("target_transforms",
                   "a declared target transformation is not applied"),
    "fixed_order": ("operator_trace",
                    "an operator permutation is drawn and the loop indexes the unpermuted list"),
    # --- classes expected to be MISSED; reported as such -------------------
    "value_range": (None,
                    "output range silently changes from [0,1] to [0,255]"),
    "channel_collapse": (None,
                         "colour output silently collapses to replicated luminance"),
}


class SubjectLoader:
    """Reference loader with one injectable defect.

    Extends the reference loader shipped with the package (``selftest.py``)
    with the classes needed for a detection matrix, including two that no gate
    is expected to catch.
    """

    name = "subject"

    def __init__(self, defect: str = "none", num_frames: int = 5):
        if defect not in DEFECTS:
            raise ValueError(f"unknown defect: {defect}")
        self.defect = defect
        self.num_frames = num_frames
        self.recorder = CallRecorder()
        self._rng = np.random.default_rng(0)
        self._frozen_ids: list[int] | None = None

    # -- adapter protocol ---------------------------------------------------
    def build(self, spec: LoaderSpec):
        gt_dir = sorted(Path(spec.gt_root).iterdir())[0]
        lq_dir = sorted(Path(spec.lq_root).iterdir())[0] if spec.lq_root else gt_dir
        self._gt = sorted(gt_dir.glob("*.png"))
        self._lq = sorted(lq_dir.glob("*.png"))
        self._spec = spec
        self._rng = np.random.default_rng(spec.seed)
        self._frozen_ids = None
        return self

    def sample(self, loader, index: int) -> Sample:
        n, total = self.num_frames, len(self._gt)

        if self.defect == "prefix_window":
            ids = list(range(n))                       # the window is discarded
        elif self.defect == "frozen_window":
            if self._frozen_ids is None:
                start = int(self._rng.integers(0, max(total - n, 1)))
                self._frozen_ids = list(range(start, start + n))
            ids = list(self._frozen_ids)
        else:
            start = int(self._rng.integers(0, max(total - n, 1)))
            ids = list(range(start, start + n))

        gt = [cv2.imread(str(self._gt[i])) for i in ids]

        use_pre = self._spec.use_precomputed_lq and self.defect != "ignore_precomputed"
        lq = ([cv2.imread(str(self._lq[i])) for i in ids] if use_pre
              else [self._degrade(g) for g in gt])

        if self.defect != "no_sharpen":
            gt = [self._sharpen(g) for g in gt]
        if self.defect == "double_sharpen":
            gt = [self._sharpen(g) for g in gt]

        lq_a, gt_a = np.asarray(lq), np.asarray(gt)

        if self.defect == "value_range":
            # A real range change: the fixture and the contract are in level
            # units (frame i has every pixel equal to i), and this loader
            # silently returns [0,1] instead.
            #
            # The previous version only cast uint8 to float32 and left the
            # values alone, so it changed the dtype and not the range at all.
            # It nevertheless produced an alarm, from a heuristic in our own
            # decoder rather than from the defect, and the article reported
            # that alarm as a detection by the wrong route. The mutant is now
            # what it was described as, and whatever the gates do with it is
            # what gets reported.
            lq_a = lq_a.astype(np.float32) / 255.0
            gt_a = gt_a.astype(np.float32) / 255.0
        if self.defect == "channel_collapse":
            lq_a = np.repeat(lq_a.mean(axis=-1, keepdims=True), 3, axis=-1).astype(np.uint8)

        return Sample(lq=lq_a, gt=gt_a, frame_ids=ids)

    def __len__(self) -> int:
        return len(self._gt)

    # -- the "pipeline" -----------------------------------------------------
    def _sharpen(self, img: np.ndarray) -> np.ndarray:
        self.recorder.record("sharpen")
        return img

    def _degrade(self, img: np.ndarray) -> np.ndarray:
        ops = ["blur", "downsample", "noise", "jpeg"]
        if self.defect != "fixed_order":
            ops = list(self._rng.permutation(ops))
        out = img.astype(np.float32)
        for op in ops:
            self.recorder.record(op)
            if op == "blur":
                out = cv2.GaussianBlur(out, (3, 3), 0.8)
            elif op == "noise":
                out = out + self._rng.normal(0, 3, out.shape).astype(np.float32)
        return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Running the gates, with timing
# ---------------------------------------------------------------------------

def _timed(fn, *a, **kw) -> tuple[CheckResult, float]:
    t0 = time.perf_counter()
    res = fn(*a, **kw)
    return res, time.perf_counter() - t0


def run_delivery_gates(loader: SubjectLoader, workdir: Path,
                       seed: int = 0) -> dict[str, tuple[CheckResult, float]]:
    """Drive every delivery gate at one fixture seed.

    ``seed`` reaches the subject, which is the whole point and was not true
    before: the false-alarm arm assigned ``loader._rng`` directly, and then each
    gate built a LoaderSpec with a hardcoded seed of 0, and the subject's
    ``build`` re-seeded itself from the spec. The assignment was destroyed on
    the next line, so twenty labelled seeds were twenty repetitions of seed
    zero and the CSV changed the label rather than the experiment.
    """
    out: dict[str, tuple[CheckResult, float]] = {}
    out["precomputed_input"] = _timed(check_precomputed_input, loader, workdir,
                                      seed=seed)
    out["temporal_window"] = _timed(check_temporal_window, loader, workdir,
                                    seed=seed)
    out["target_transforms"] = _timed(
        check_target_transforms, loader, workdir, loader.recorder, "sharpen",
        expected=1, seed=seed)
    out["operator_trace"] = _timed(
        check_operator_trace, loader, workdir, loader.recorder, OPS,
        declared_policy="random_permutation", seed=seed)
    return out


def run_separability(inert: bool) -> tuple[CheckResult, float]:
    """Two arms declared distinct. ``inert=True`` makes the factor do nothing."""
    a = np.full((8, 8, 3), 100, np.uint8)
    b = a.copy()
    if not inert:
        b[0, 0] = 0
    return _timed(check_separability, lambda: a, lambda: b,
                  declared="distinct", label="arm_A_vs_arm_B")


def run_geometry(distorted: bool) -> tuple[CheckResult, float]:
    delivered = (368, 640) if distorted else (360, 640)
    return _timed(check_geometry, delivered, (180, 320))


# ---------------------------------------------------------------------------
# Campaign
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=20,
                    help="fixture seeds for the false-alarm arm (default 20)")
    ap.add_argument("--out", type=Path, default=_default_out())
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    wd = Path(tempfile.mkdtemp(prefix="campaign_M_"))
    rows: list[dict] = []
    cost: dict[str, list[float]] = defaultdict(list)

    # --- 1. delivery gates over every defect class -------------------------
    for defect, (target, description) in DEFECTS.items():
        results = run_delivery_gates(SubjectLoader(defect), wd / defect)
        for gate, (res, dt) in results.items():
            cost[gate].append(dt)
            rows.append(dict(
                defect=defect, defect_description=description,
                targeted_gate=target or "", gate=gate, status=res.status,
                is_target=int(gate == target),
                detected=int(res.status == "FAIL"),
                seconds=round(dt, 6),
                message=res.message[:200]))

    # --- 2. separability and geometry, which take builders, not adapters ---
    for label, (res, dt) in (
            ("inert_factor", run_separability(inert=True)),
            ("real_factor", run_separability(inert=False))):
        cost["separability"].append(dt)
        rows.append(dict(
            defect=label,
            defect_description=("an experimental factor that does nothing"
                                if label == "inert_factor"
                                else "a factor that really changes the tensor (control)"),
            targeted_gate="separability" if label == "inert_factor" else "",
            gate="separability", status=res.status,
            is_target=int(label == "inert_factor"),
            detected=int(res.status == "FAIL"),
            seconds=round(dt, 6), message=res.message[:200]))

    for label, (res, dt) in (
            ("geometry_squeeze", run_geometry(distorted=True)),
            ("geometry_faithful", run_geometry(distorted=False))):
        cost["geometry"].append(dt)
        rows.append(dict(
            defect=label,
            defect_description=("evaluation resize distorts the declared aspect ratio"
                                if label == "geometry_squeeze"
                                else "evaluation resize preserves it (control)"),
            targeted_gate="geometry" if label == "geometry_squeeze" else "",
            gate="geometry", status=res.status,
            is_target=int(label == "geometry_squeeze"),
            detected=int(res.status == "FAIL"),
            seconds=round(dt, 6), message=res.message[:200]))

    # --- 3. false alarms: the conforming loader, many fixture seeds --------
    fa_rows: list[dict] = []
    false_alarms = 0
    for s in range(args.seeds):
        loader = SubjectLoader("none")
        results = run_delivery_gates(loader, wd / f"fa_{s}", seed=s)
        for gate, (res, dt) in results.items():
            cost[gate].append(dt)
            alarm = int(res.status == "FAIL")
            false_alarms += alarm
            fa_rows.append(dict(seed=s, gate=gate, status=res.status,
                                false_alarm=alarm, seconds=round(dt, 6)))

    # --- write ------------------------------------------------------------
    def write(path: Path, data: list[dict]) -> None:
        with path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
        print(f"wrote {path}  ({len(data)} rows)")

    write(args.out / "M_seeded_defects_matrix.csv", rows)
    write(args.out / "M_false_alarms.csv", fa_rows)
    write(args.out / "M_gate_cost.csv", [
        dict(gate=g, runs=len(v),
             mean_seconds=round(statistics.fmean(v), 6),
             median_seconds=round(statistics.median(v), 6),
             max_seconds=round(max(v), 6))
        for g, v in sorted(cost.items())])

    # --- headline counts --------------------------------------------------
    targeted = [r for r in rows if r["is_target"]]
    hits = [r for r in targeted if r["detected"]]
    expected_miss = [d for d, (t, _) in DEFECTS.items() if t is None and d != "none"]
    collateral = [r for r in rows
                  if not r["is_target"] and r["detected"] and r["defect"] != "none"]
    total_seconds = sum(r["seconds"] for r in rows)

    # Collateral firing is a declared response variable of Study 2 and it needs
    # its own source file: a number that only exists inside a summary cannot be
    # checked against anything. One row per gate that fired outside its class.
    # Cost of ONE full pass, measured rather than derived from a total.
    #
    # This replaces a number that was wrong by a factor of forty. The summary
    # used to publish ``suite_seconds_one_pass`` = the sum of EVERY gate
    # execution in the seeded matrix -- forty executions across twelve loader
    # variants -- under a name that says "one pass", and the manuscript
    # repeated it as the cost of one pass. The false-alarm campaign already
    # runs the four adapter-driven gates over the conforming loader once per
    # seed, so twenty independent passes are available at no extra cost: group
    # them by seed and report the distribution.
    #
    # Reporting the median and the maximum rather than the mean is not a
    # stylistic choice. The mean of ``temporal_window`` is dominated by a
    # single first-call file-system warm-up, and measuring the same campaign on
    # a busy machine moved that mean by about an order of magnitude while the
    # median moved by seven per cent. A cost a reader cannot reproduce is not a
    # cost.
    pass_cost: dict[str, float] = {}
    for r in fa_rows:
        pass_cost[r["seed"]] = pass_cost.get(r["seed"], 0.0) + r["seconds"]
    passes = sorted(pass_cost.values())
    write(args.out / "M_pass_cost.csv",
          [dict(seed=s, gates=len(DELIVERY_GATES), pass_seconds=round(v, 6))
           for s, v in sorted(pass_cost.items())])

    collateral_rows = [dict(defect=r["defect"], gate=r["gate"],
                            status=r["status"], targeted_gate=r["targeted_gate"],
                            message=r["message"]) for r in collateral]
    if collateral_rows:
        write(args.out / "M_collateral.csv", collateral_rows)
    else:
        (args.out / "M_collateral.csv").write_text(
            "defect,gate,status,targeted_gate,message\n")
        print(f"wrote {args.out / 'M_collateral.csv'}  (0 rows)")

    summary = dict(
        defect_classes=len(DEFECTS) + 3,          # + inert/real factor, + geometry pair
        targeted_defects=len([d for d, (t, _) in DEFECTS.items() if t]) + 2,
        positive_controls=2,                      # real_factor, geometry_faithful
        targeted_pairs=len(targeted),
        targeted_detected=len(hits),
        expected_misses=expected_miss,
        collateral_failures=len(collateral),
        false_alarm_runs=len(fa_rows),
        false_alarms=false_alarms,
        one_pass_median_seconds=round(statistics.median(passes), 4),
        one_pass_min_seconds=round(min(passes), 4),
        one_pass_max_seconds=round(max(passes), 4),
        one_pass_measurements=len(passes),
        # Renamed 2026-08-10. This is the sum of EVERY gate execution in the
        # seeded matrix, not the cost of one pass, and its old name said the
        # opposite. It is load-dependent -- 0.80 s idle, 7.68 s with a training
        # job on the machine -- because one first-call stall dominates it.
        seeded_matrix_seconds_total=round(total_seconds, 4),
        slowest_gate=max(cost, key=lambda g: statistics.fmean(cost[g])),
        slowest_gate_median_seconds=round(
            statistics.median(cost[max(cost, key=lambda g: statistics.fmean(cost[g]))]), 6),
    )
    (args.out / "M_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print("\n--- Campaign M summary ---")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    if collateral:
        print("\n  collateral failures (a gate firing outside its class):")
        for r in collateral:
            print(f"    {r['defect']} -> {r['gate']}: {r['message'][:70]}")
    print(f"\n  expected misses, reported as such: {', '.join(expected_miss)}")

    ok = len(hits) == len(targeted) and false_alarms == 0
    print("\nCAMPAIGN M " + ("PASSED" if ok else "NEEDS REVIEW"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
