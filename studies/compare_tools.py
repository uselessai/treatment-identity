#!/usr/bin/env python
"""What other tools report on two delivery defects that DDC finds in released code.

Subjects: the released training loaders of RTN and MambaOFR, whose frame window
never moves, and of RRTN, whose window moves; in all three the degradation
operations run in one order although the code draws a random permutation.

Each tool is given what it normally consumes:

  static analysis    ruff and pylint, on the two source files that hold the defects
  data validation    Deepchecks (tabular data-integrity suite) on features of the
                     samples each loader delivers
  drift detection    Evidently, deliveries of each released loader against those of
                     a copy of it in which one line is corrected
  hand-written tests two assertions a developer could write without fixtures
  DDC                the frame-window and operation-order observations of the profile

For RTN and MambaOFR, two corrected copies are made in a temporary folder, each
with one defect removed and nothing else changed: in one the frames are read at
the drawn positions, in the other the operations run in the drawn order. They
give every tool a reference that differs from the released loader in the defect
only, and show that each tool is run in a way in which it can respond.

The loaders are run on two inputs: the DDC fixture whose frames carry their own
index, and a natural video clip, which is what a data validator would be given
in practice.

    python studies/compare_tools.py --repos <clones> --bank <noise_data> \
        --natural <folder of frames> --out studies/data_1.14.0/comparison
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE / "old_film"), str(HERE.parent)]
warnings.filterwarnings("ignore")

from audit_old_film import SYSTEMS, _adapter_kwargs             # noqa: E402
from vp_code import VPCodeAdapter                                # noqa: E402
from treatment_identity.adapter import LoaderSpec                # noqa: E402
from treatment_identity.fixtures import decode_index, make_index_encoded  # noqa: E402
from treatment_identity.profile import (PROBE_FRAMES, observe_operators,  # noqa: E402
                                        observe_window)
from treatment_identity.seeding import seed_all                  # noqa: E402

SUBJECTS = ("RTN", "MambaOFR", "RRTN")
#: Where each defect is in the source, relative to the repository: the loop that
#: draws a value and does not use it.
DATASET = "VP_code/data/dataset.py"
DEGRADATION = "VP_code/data/Data_Degradation/util.py"


# ------------------------------------------------------------------ deliveries
def write_natural(src: Path, root: Path, short_side: int = 128) -> int:
    """Copy a natural clip into the layout the loaders read, at a small size."""
    d = root / "clip_000"
    d.mkdir(parents=True)
    names = sorted(p for p in src.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    for i, p in enumerate(names):
        img = cv2.imread(str(p))
        h, w = img.shape[:2]
        s = short_side / min(h, w)
        img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(d / f"{i:08d}.png"), img)
    return len(names)


def deliveries(adapter, workdir: Path, kind: str, natural: Path, draws: int, seed: int):
    """Draw samples from one loader on one input; return per-draw records."""
    seed_all(seed)
    if kind == "fixture":
        make_index_encoded(workdir / "GT", n_frames=128, shape=(128, 128))
        make_index_encoded(workdir / "LQ", n_frames=128, shape=(128, 128))
    else:
        write_natural(natural, workdir / "GT")
        write_natural(natural, workdir / "LQ")
    spec = LoaderSpec(gt_root=workdir / "GT", lq_root=workdir / "LQ",
                      num_frames=PROBE_FRAMES, seed=seed)
    loader = adapter.build(spec)
    out = []
    for i in range(draws):
        s = adapter.sample(loader, i)
        gt = np.asarray(s.gt, dtype=np.float32)
        frames = gt if gt.ndim == 4 else gt[None]
        window = tuple(decode_index(f) for f in frames) if kind == "fixture" else None
        out.append({"gt": frames, "reported": tuple(s.frame_ids or ()), "window": window,
                    "digest": hashlib.sha256(np.ascontiguousarray(frames).tobytes()).hexdigest()})
    return out


def features(records):
    """Per-sample summary statistics, the table a data validator is given."""
    import pandas as pd
    rows = []
    for r in records:
        a = r["gt"]
        row = {"mean": float(a.mean()), "std": float(a.std()),
               "min": float(a.min()), "max": float(a.max()),
               "frame_diff": float(np.abs(np.diff(a, axis=0)).mean()) if len(a) > 1 else 0.0}
        for t in range(len(a)):
            row[f"frame{t}_mean"] = float(a[t].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def corrected_copy(repo: Path, dest: Path, defect: str) -> Path:
    """A copy of the repository with one defect removed and nothing else changed."""
    shutil.copytree(repo, dest, ignore=shutil.ignore_patterns(".git", "__pycache__"))
    if defect == "window":
        f = dest / DATASET
        old = "img_gt_path = os.path.join(current_gt_root, clip_name, new_clip_sequence[tmp_id])"
        new = "img_gt_path = os.path.join(current_gt_root, clip_name, new_clip_sequence[frame])"
        text = f.read_text()
        i = text.index("for tmp_id, frame in enumerate(frame_list):")
        j = text.index(old, i)
        f.write_text(text[:j] + new + text[j + len(old):])
    else:
        f = dest / DEGRADATION
        old = ("for i,distortion_index in enumerate(distortion_sequence):\n"
               "        distortion = distortion_types[i]")
        text = f.read_text()
        assert old in text
        f.write_text(text.replace(old, old.replace("distortion_types[i]",
                                                   "distortion_types[distortion_index]")))
    return dest


# ------------------------------------------------------------------ tools
def deepchecks_integrity(df) -> dict:
    from deepchecks.tabular import Dataset
    from deepchecks.tabular.suites import data_integrity
    res = data_integrity().run(Dataset(df, cat_features=[]))
    failed = []
    for c in res.results:
        for cond in getattr(c, "conditions_results", None) or []:
            if not cond.is_pass():
                failed.append({"check": c.get_header(), "condition": cond.name,
                               "details": str(cond.details)[:200]})
    return {"checks": len(res.results), "failed_conditions": failed}


def evidently_drift(reference, current) -> dict:
    from evidently import DataDefinition, Dataset, Report
    from evidently.presets import DataDriftPreset
    cols = list(reference.columns)
    dd = DataDefinition(numerical_columns=cols)
    rep = Report([DataDriftPreset()]).run(
        Dataset.from_pandas(current, data_definition=dd),
        Dataset.from_pandas(reference, data_definition=dd)).dict()
    first = rep.get("metrics", [{}])[0].get("value", {})
    return {"columns": len(cols), "drifted_columns": first.get("count"),
            "share": first.get("share")}


def hand_tests(records) -> dict:
    """Two assertions written without fixtures or knowledge of the defect."""
    return {
        # "the loader returns different samples in different draws"
        "samples_differ": {"distinct_samples": len({r["digest"] for r in records}),
                           "passes": len({r["digest"] for r in records}) > 1},
        # "the frame list the loader reports changes between draws"
        "reported_window_varies": {"distinct_reported": len({r["reported"] for r in records}),
                                   "passes": len({r["reported"] for r in records}) > 1},
    }


def static_analysis(repo: Path) -> dict:
    """Diagnostics of ruff and pylint on the two files, and those on the defect loops."""
    files = [str(repo / DATASET), str(repo / DEGRADATION)]

    def ruff(*select):
        cmd = ["ruff", "check", "--no-cache", "--isolated", "--output-format=json", *select, *files]
        return json.loads(subprocess.run(cmd, capture_output=True, text=True).stdout or "[]")

    def unused_loop(diags, key_file, key_line, key_code, key_msg):
        return [f"{Path(d[key_file]).name}:{key_line(d)} {d[key_code]} {d[key_msg]}"
                for d in diags if d[key_code] in ("B007", "W0612", "unused-variable")]

    default, bugbear, everything = ruff(), ruff("--select", "B007"), ruff("--select", "ALL")
    out = subprocess.run([sys.executable, "-m", "pylint", "--output-format=json",
                          "--disable=C,R", *files], capture_output=True, text=True).stdout
    pl = json.loads(out or "[]")
    line = lambda d: d["location"]["row"]
    return {
        "ruff_default": {"diagnostics": len(default),
                         "unused_loop_variable": unused_loop(default, "filename", line, "code", "message")},
        "ruff_B007": {"diagnostics": len(bugbear),
                      "unused_loop_variable": unused_loop(bugbear, "filename", line, "code", "message")},
        "ruff_all_rules": {"diagnostics": len(everything)},
        "pylint": {"diagnostics": len(pl),
                   "unused_variable": [f"{Path(d['path']).name}:{d['line']} {d['message-id']} {d['message']}"
                                       for d in pl if d["message-id"] == "W0612"]},
    }


def tool_versions() -> dict:
    v = {}
    for m in ("deepchecks", "evidently", "treatment_identity"):
        try:
            v[m] = __import__(m).__version__
        except Exception:
            v[m] = None
    v["ruff"] = subprocess.run(["ruff", "--version"], capture_output=True, text=True).stdout.strip()
    v["pylint"] = subprocess.run([sys.executable, "-m", "pylint", "--version"],
                                 capture_output=True, text=True).stdout.splitlines()[0]
    return v


# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repos", required=True, type=Path)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--natural", required=True, type=Path)
    ap.add_argument("--draws", type=int, default=100)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    result = {"versions": tool_versions(), "draws": args.draws,
              "natural_clip": args.natural.name, "subjects": {}, "evidently": {}}
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)

        def run(label, name, repo, seed=0, static=True):
            meta = SYSTEMS[name]
            ad = VPCodeAdapter(label.replace(" ", "_"), repo, texture_bank=args.bank,
                               channels=meta["channels"], **_adapter_kwargs(meta))
            w = observe_window(ad, td / f"{label}_w{seed}", draws=args.draws, seed=seed)
            o = observe_operators(ad, td / f"{label}_o{seed}", draws=args.draws, seed=seed)
            sub = {"ddc": {"distinct_windows": w.get("distinct_windows"),
                           "window_note": w.get("note"),
                           "distinct_orders": o.get("distinct_orders"),
                           "order_note": o.get("note")}, "inputs": {}}
            if static:
                sub["static"] = static_analysis(repo)
            feats = {}
            for kind in ("fixture", "natural"):
                recs = deliveries(ad, td / f"{label}_{kind}{seed}", kind, args.natural,
                                  args.draws, seed)
                df = feats[kind] = features(recs)
                df.to_csv(args.out / f"features_{label.replace(' ', '_')}_{kind}_s{seed}.csv",
                          index=False)
                entry = {"hand_tests": hand_tests(recs), "deepchecks": deepchecks_integrity(df)}
                if kind == "fixture":
                    entry["delivered_windows"] = len({r["window"] for r in recs})
                sub["inputs"][kind] = entry
            print(label, seed, json.dumps(sub["ddc"]))
            return sub, feats

        for name in SUBJECTS:
            repo = args.repos / SYSTEMS[name]["pristine"]
            sub, released = run(name, name, repo)
            result["subjects"][name] = sub
            if name == "RRTN":
                continue
            for defect in ("window", "order"):
                label = f"{name} {defect} corrected"
                fixed_repo = corrected_copy(repo, td / f"repo_{name}_{defect}", defect)
                fixed, ref = run(label, name, fixed_repo, static=False)
                result["subjects"][label] = fixed
                _, again = run(label, name, fixed_repo, seed=1, static=False)
                for kind in ("fixture", "natural"):
                    result["evidently"].setdefault(kind, {})[f"{name} released vs {defect} corrected"] = \
                        evidently_drift(ref[kind], released[kind])
                    result["evidently"][kind][f"{name} {defect} corrected, two runs"] = \
                        evidently_drift(ref[kind], again[kind])
    (args.out / "C_tool_comparison.json").write_text(json.dumps(result, indent=2) + "\n")

    with (args.out / "C_tool_comparison.csv").open("w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["system", "ddc_distinct_windows", "ddc_distinct_orders",
                     "ruff_default", "ruff_default_unused_loop", "ruff_B007_hits",
                     "ruff_all_rules", "pylint", "pylint_unused_variable",
                     "input", "delivered_windows", "distinct_samples",
                     "reported_windows", "deepchecks_failed_conditions"])
        blank = {k: {"diagnostics": "", "unused_loop_variable": [], "unused_variable": []}
                 for k in ("ruff_default", "ruff_B007", "ruff_all_rules", "pylint")}
        for name, sub in result["subjects"].items():
            st = sub.get("static", blank)
            for kind, e in sub["inputs"].items():
                wr.writerow([name, sub["ddc"]["distinct_windows"], sub["ddc"]["distinct_orders"],
                             st["ruff_default"]["diagnostics"],
                             len(st["ruff_default"]["unused_loop_variable"]),
                             len(st["ruff_B007"]["unused_loop_variable"]),
                             st["ruff_all_rules"]["diagnostics"], st["pylint"]["diagnostics"],
                             len(st["pylint"]["unused_variable"]), kind,
                             e.get("delivered_windows", ""),
                             e["hand_tests"]["samples_differ"]["distinct_samples"],
                             e["hand_tests"]["reported_window_varies"]["distinct_reported"],
                             len(e["deepchecks"]["failed_conditions"])])
    print("wrote", args.out / "C_tool_comparison.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
