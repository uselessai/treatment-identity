#!/usr/bin/env python3
"""Campaign S --- separability on conditions the subjects themselves declare.

``separability`` was never exercised outside the audited lineage, because a
portability subject is one loader under one configuration and the gate needs a
pair.  The pair used here is not authored by us: each subject *ships* two
settings it names differently and documents as different treatments, and the
gate asks whether that declared distinction survives to the delivered tensor.

Both arms run under matched seeds, so a difference cannot come from the random
state.  ``declared="distinct"`` therefore fails on unexpected *equality*: two
settings the project separates by name delivering the same tensor.

A control arm runs each setting against itself under matched seeds with
``declared="identical"``.  If that control does not pass, the harness cannot
hold the seed fixed for this subject and no verdict on the real pair is
reported --- the gate would otherwise be reading our own nondeterminism.

Run:  python campaigns/campaign_S_separability.py --out campaigns/data
"""
from __future__ import annotations

import argparse
import csv
import importlib
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE / "adapters"))

from treatment_identity.checks import check_separability   # noqa: E402
from treatment_identity.fixtures import make_pair          # noqa: E402
from treatment_identity.seeding import seed_all            # noqa: E402

# --------------------------------------------------------------------------
# Subjects. ``arms`` are two option overrides the subject itself distinguishes;
# ``declared_by`` cites where the subject makes that distinction.
# --------------------------------------------------------------------------
SUBJECTS = {
    "kair_blindsr_degradation": dict(
        module="adapters_kair_blindsr", cls="KairBlindSRAdapter",
        root=Path("/home/laura/02ImproveData/zKAIR"),
        arm_a=("bsrgan", {"degradation_type": "bsrgan"}),
        arm_b=("bsrgan_plus", {"degradation_type": "bsrgan_plus"}),
        declared_by="zKAIR/data/dataset_blindsr.py:69-79 dispatches the two "
                    "names to degradation_bsrgan and degradation_bsrgan_plus",
        note="two degradation pipelines KAIR names separately and ships "
             "separate trained models for",
    ),
    "kair_blindsr_sharp": dict(
        module="adapters_kair_blindsr", cls="KairBlindSRAdapter",
        root=Path("/home/laura/02ImproveData/zKAIR"),
        arm_a=("use_sharp=False", {"degradation_type": "bsrgan_plus",
                                   "use_sharp": False}),
        arm_b=("use_sharp=True", {"degradation_type": "bsrgan_plus",
                                  "use_sharp": True}),
        declared_by="zKAIR/data/dataset_blindsr.py:72 forwards use_sharp to "
                    "degradation_bsrgan_plus as a documented option",
        note="a flag the subject exposes as a treatment switch on the same "
             "degradation pipeline",
    ),
}

#: Subjects that cannot supply a pair, and why. The gate needs a loader that
#: *manufactures* its input from a configurable treatment; a subject that reads
#: a pre-rendered pair has no second condition to declare. That is a property
#: of the subject, not a limit of the gate, and it is the same argument the
#: article already makes for operator_trace's SKIPs.
NOT_AUDITED = [
    dict(subject="uformer", project="Uformer",
         reason="reads pre-rendered pairs: the degradation happened offline, so "
                "the loader exposes no treatment to switch"),
    dict(subject="restormer", project="Restormer",
         reason="reads pre-rendered pairs: the degradation happened offline, so "
                "the loader exposes no treatment to switch"),
    dict(subject="mprnet", project="MPRNet",
         reason="reads pre-rendered pairs; same structure as Uformer"),
    dict(subject="basicsr_reds", project="BasicSR",
         reason="reads pre-rendered LQ/GT pairs: the loader composes no treatment"),
    dict(subject="kair_video", project="KAIR",
         reason="reads pre-rendered pairs; no treatment option in the loader"),
    dict(subject="zerodce", project="Zero-DCE",
         reason="zero-reference: one stream, no declared treatment, no pair"),
    dict(subject="mmagic", project="MMagic",
         reason="the treatment is composed by config pipeline transforms, not by "
                "the dataset, so no pair is selectable"),
    dict(subject="pix2pix_colorization", project="pytorch-CycleGAN-and-pix2pix",
         reason="its declared alternatives change the delivered shape, so a "
                "difference would come from shape, not treatment"),
]

BASE_OPT = {
    "n_channels": 3, "scale": 2, "shuffle_prob": 0.1, "use_sharp": False,
    "degradation_type": "bsrgan", "lq_patchsize": 16, "H_size": 32,
    "phase": "train",
}


def build_arm(adapter_cls, root: Path, gt_root: Path, overrides: dict,
              seed: int, index: int = 0):
    """Deliver one item under one setting, with the seed set immediately
    before delivery so both arms see the same random state."""
    def build():
        seed_all(seed)
        ad = adapter_cls(root)
        opt = dict(BASE_OPT)
        opt["dataroot_H"] = str(gt_root)
        opt.update(overrides)
        ds = ad._cls(opt)
        item = ds[index % len(ds)]
        import numpy as np
        lq = item["L"]
        try:
            import torch
            lq = lq.detach().cpu().numpy() if isinstance(lq, torch.Tensor) else np.asarray(lq)
        except ImportError:                                  # pragma: no cover
            lq = np.asarray(lq)
        return lq
    return build


def run_subject(key: str, cfg: dict, workdir: Path, seeds: list[int]) -> list[dict]:
    mod = importlib.import_module(cfg["module"])
    adapter_cls = getattr(mod, cfg["cls"])
    gt, _lq = make_pair(workdir, n_frames=4, shape=(128, 128), gt_kind="signature")
    gt_root = Path(gt.root) if hasattr(gt, "root") else workdir / "GT"
    clips = sorted(p for p in Path(gt_root).iterdir() if p.is_dir())
    data_root = clips[0] if clips else gt_root

    name_a, over_a = cfg["arm_a"]
    name_b, over_b = cfg["arm_b"]
    rows = []
    for seed in seeds:
        a = build_arm(adapter_cls, cfg["root"], data_root, over_a, seed)
        b = build_arm(adapter_cls, cfg["root"], data_root, over_b, seed)
        # control: the same arm twice, which must agree if the seed holds
        ctl = check_separability(a, build_arm(adapter_cls, cfg["root"], data_root,
                                              over_a, seed),
                                 declared="identical", label=f"{name_a}_self")
        res = check_separability(a, b, declared="distinct",
                                 label=f"{name_a}_vs_{name_b}")
        rows.append(dict(
            subject=key, seed=seed, arm_a=name_a, arm_b=name_b,
            control_status=ctl.status,
            control_delta=ctl.evidence.get("max_abs_delta_0_255", ""),
            status=(res.status if ctl.status == "PASS" else "NO_VERDICT"),
            max_abs_delta_0_255=res.evidence.get("max_abs_delta_0_255", ""),
            mae_0_255=res.evidence.get("mae_0_255", ""),
            declared_by=cfg["declared_by"], note=cfg["note"]))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=HERE / "data")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--subject", action="append")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.seeds))

    rows = []
    keys = args.subject or list(SUBJECTS)
    with tempfile.TemporaryDirectory() as td:
        for key in keys:
            wd = Path(td) / key
            wd.mkdir(parents=True, exist_ok=True)
            try:
                rows += run_subject(key, SUBJECTS[key], wd, seeds)
                print(f"{key}: ok")
            except Exception as exc:                        # noqa: BLE001
                print(f"{key}: ERROR {type(exc).__name__}: {exc}")
                rows.append(dict(subject=key, seed=-1, arm_a="", arm_b="",
                                 control_status="ERROR", control_delta="",
                                 status="ERROR", max_abs_delta_0_255="",
                                 mae_0_255="", declared_by="",
                                 note=f"{type(exc).__name__}: {exc}"[:200]))

    for sub in NOT_AUDITED:
        rows.append(dict(subject=sub["subject"], seed=-1, arm_a="", arm_b="",
                         control_status="", control_delta="", status="SKIP",
                         max_abs_delta_0_255="", mae_0_255="",
                         declared_by="", note=sub["reason"]))

    out = args.out / "S_separability_matrix.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
