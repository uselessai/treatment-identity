#!/usr/bin/env python
"""Run the six treatment-identity gates over the RTN / RRTN / MambaOFR lineage.

    python audit_vp_lineage.py --repos <dir> --bank <noise_data> --out certificates/

``--repos`` may point either at pristine clones (auditing the released code) or
at the working copies a study actually trained with (auditing the harness); the
certificates record which, via the commit hash and dirty flag.

Everything runs on CPU against synthetic fixtures and finishes in seconds.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

from adapters.vp_code import OPERATORS, VPCodeAdapter  # noqa: E402
from treatment_identity import (Certificate, check_geometry, check_operator_trace,  # noqa: E402
                                check_precomputed_input, check_separability,
                                check_target_transforms, check_temporal_window)
from treatment_identity.adapter import LoaderSpec  # noqa: E402
from treatment_identity import fixtures as fx  # noqa: E402

LINEAGE = {
    "RTN": {"pristine": "Bringing-Old-Films-Back-to-Life",
            "working": "zBringing-Old-Films-Back-to-Life", "channels": 3},
    "RRTN": {"pristine": "RRTN-old-film-restoration",
             "working": "zRRTN-old-film-restoration", "channels": 1},
    "MambaOFR": {"pristine": "MambaOFR", "working": "zMambaOFR", "channels": 3},
    # MgfrOFR reorganised the tree: the generator body is inherited but lives
    # under a different package, so the adapter is told where to look.
    # MgfrOFR was never retrained here, so there is no working copy distinct
    # from the clone: both keys name the same clean checkout, and the
    # certificate's dirty flag is what proves it.
    "MgfrOFR": {"pristine": "zMgfrOFR", "working": "zMgfrOFR", "channels": 3,
                "dataset_module": "basicofr.data.rtn_dataset",
                "degradation_module": "basicofr.data.degradations.core",
                "degradation_path": "basicofr/data/degradations/core.py"},
}


def _adapter_kwargs(meta: dict) -> dict:
    """Only the layout overrides a repository actually needs."""
    return {k: meta[k] for k in ("dataset_module", "degradation_module",
                                 "degradation_path") if k in meta}


def _clip_from_fixture(root: Path, n: int = 3, shape=(64, 64)) -> list[np.ndarray]:
    """A small colour clip, as the generators expect: BGR float in [0,1]."""
    rng = np.random.default_rng(7)
    h, w = shape
    return [rng.random((h, w, 3)).astype(np.float32) for _ in range(n)]


def audit(name: str, repo: Path, bank: str, channels: int, workdir: Path,
          peers: dict[str, VPCodeAdapter],
          declares_precomputed: bool = True,
          **adapter_kwargs) -> tuple[Certificate, VPCodeAdapter]:
    ad = VPCodeAdapter(name, repo, texture_bank=bank, channels=channels,
                       **adapter_kwargs)
    cert = Certificate(
        pipeline=name,
        expected_treatment=(f"{name} native degradation regime; pre-computed "
                            "inputs honoured when configured; windowed temporal "
                            "sampling; target NOT sharpened (no publication of "
                            "this lineage declares it); random operator order"),
    ).with_repository(repo)
    cert.uncontrolled_rng = ["albumentations (library-internal RNG, outside "
                             "random/numpy/torch seeds)"]

    # -- gates 1 and 2: delivery -------------------------------------------
    # `declares_precomputed`: does *this tree* configure the branch? Only then
    # is its absence a divergence rather than an unclaimed feature.
    g1 = check_precomputed_input(ad, workdir, declared=declares_precomputed)
    cert.add(g1)
    cert.delivery_modality = ("pre-rendered" if g1.ok else
                              "online (flag not honoured)" if g1.status == "FAIL" else
                              "online (no pre-computed branch declared)")
    cert.branch_reached = ("precomputed" if g1.ok else "online_generator")

    g2 = check_temporal_window(ad, workdir)
    cert.add(g2)
    if "unique_frames" in g2.evidence:
        cert.unique_frames_reachable = len(g2.evidence["unique_frames"])
        cert.clip_length = 32

    # -- gate 4: how often is the target transformed? ----------------------
    # expected=0: the declared count. No paper of the lineage mentions GT
    # sharpening, so any occurrence is the divergence the gate reports.
    # No publication in this lineage mentions sharpening the target, so the
    # declared count is zero and the declaration itself is absent: the gate
    # reports UNDECLARED rather than accusing the papers of a false statement.
    g4 = check_target_transforms(ad, workdir, ad.recorder, "sharpen",
                                 expected=0, declared=False)
    cert.add(g4)
    cert.target_transform_count = g4.evidence.get("calls_per_frame")

    # -- gate 5: is the declared random operator order the one that runs? ---
    cert.add(check_operator_trace(ad, workdir, ad.recorder, OPERATORS,
                                  declared_policy="random_permutation"))
    orders = [g for g in cert.gates if g["gate"] == "operator_trace"]
    if orders and orders[0]["evidence"].get("orders_observed"):
        cert.observed_operator_order = orders[0]["evidence"]["orders_observed"]

    # -- gate 3: is this pipeline's generator distinct from its peers? ------
    video = _clip_from_fixture(workdir)
    for other, other_ad in peers.items():
        if other == name:
            continue
        cert.add(check_separability(
            lambda: ad.render(video, seed=1234, neutralise_colour_jitter=True),
            lambda: other_ad.render(video, seed=1234, neutralise_colour_jitter=True),
            declared="distinct", label=f"{name}_vs_{other}"))

    # -- gate 6: does the evaluation contract survive? ----------------------
    # The only gate that does not drive a loader: the shapes are those recorded
    # from the study's evaluation path, and the certificate says so.
    g6 = check_geometry(delivered_shape=(368, 640), declared_shape=(180, 320),
                        shape_source="recorded from the evaluation loaders of "
                                     "this study; not re-executed by the gate")
    cert.add(g6)
    cert.geometry = {"declared": [180, 320], "delivered": [368, 640],
                     "source": g6.evidence.get("shape_source")}

    cert.channels = channels
    cert.gt_source = "synthetic fixture (flat field / index-encoded)"
    cert.lq_source = "synthetic fixture (checkerboard / index-encoded)"
    cert.training_seed = 0
    cert.render_seed = 1234

    # Hash the tensors as delivered, not the files on disk: the certificate is a
    # statement about what reached the optimiser. A pipeline that opens the right
    # files and then transforms them still has to declare the result.
    try:
        probe = Path(workdir) / "g7"
        fx.make_pair(probe, gt_kind="index", lq_kind="checker", n_frames=16)
        s7 = ad.sample(ad.build(LoaderSpec(gt_root=probe / "GT", lq_root=probe / "LQ",
                                           use_precomputed_lq=True, num_frames=5,
                                           seed=0, train=True)), 0)
        gt, lq = np.asarray(s7.gt), np.asarray(s7.lq)
        cert.gt_stream_sha256 = hashlib.sha256(
            np.ascontiguousarray(gt, dtype=np.float32).tobytes()).hexdigest()
        cert.lq_stream_sha256 = hashlib.sha256(
            np.ascontiguousarray(lq, dtype=np.float32).tobytes()).hexdigest()
        cert.value_range = {"gt": [float(gt.min()), float(gt.max())],
                            "lq": [float(lq.min()), float(lq.max())],
                            "note": "as delivered by the loader, over the fixture"}
    except Exception as e:  # pragma: no cover - repo-specific
        cert.value_range = {"error": f"{type(e).__name__}: {e}"}

    # Stated rather than left null: silence about a seed reads as "controlled".
    cert.data_order_seed = ("not exposed: this loader draws sample order from the "
                           "global RNG via the framework sampler")
    return cert, ad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # Repeatable: clones do not always live under one root, and the separability
    # gate needs every sibling present to compare against. Searching several
    # roots in order beats emitting a certificate whose comparisons are missing.
    ap.add_argument("--repos", required=True, type=Path, action="append",
                    help="directory containing the repository clones; repeatable")
    ap.add_argument("--bank", required=True, help="path to the noise_data template bank")
    ap.add_argument("--out", type=Path, default=Path("certificates"))
    ap.add_argument("--working", action="store_true",
                    help="audit the z-prefixed working copies instead of pristine clones")
    ap.add_argument("--only", action="append", metavar="NAME",
                    help="write certificates for these pipelines only; every "
                         "discovered pipeline is still loaded so that the "
                         "separability comparisons stay complete")
    args = ap.parse_args()

    key = "working" if args.working else "pristine"
    workdir = Path(tempfile.mkdtemp(prefix="treatment_identity_"))

    adapters: dict[str, VPCodeAdapter] = {}
    where: dict[str, Path] = {}
    for name, meta in LINEAGE.items():
        for root in args.repos:
            repo = root / meta[key]
            if repo.exists():
                where[name] = repo
                adapters[name] = VPCodeAdapter(name, repo, texture_bank=args.bank,
                                               channels=meta["channels"],
                                               **_adapter_kwargs(meta))
                break
    if not adapters:
        roots = ", ".join(str(r) for r in args.repos)
        print(f"no repositories found under {roots} (looking for the "
              f"{key} names)", file=sys.stderr)
        return 2

    wanted = [n for n in adapters if not args.only or n in args.only]
    if args.only and not wanted:
        print(f"--only {args.only} matched none of {sorted(adapters)}", file=sys.stderr)
        return 2

    print(f"auditing {len(wanted)} of {len(adapters)} discovered pipelines "
          f"({key} clones) — fixtures in {workdir}\n")
    failures = 0
    for name in wanted:
        meta = LINEAGE[name]
        # Only the working tree configures the pre-computed branch; in the
        # released trees nothing claims it, so gate 1 reports N/A there.
        cert, _ = audit(name, where[name], args.bank,
                        meta["channels"], workdir, adapters,
                        declares_precomputed=args.working,
                        **_adapter_kwargs(meta))
        path = cert.write(args.out / f"{name.lower()}.json")
        print(cert.summary())
        print(f"  certificate        : {path}\n")
        failures += int(cert.status != "PASS")

    print(f"{len(wanted) - failures}/{len(wanted)} pipelines passed every gate.")
    if failures:
        print("At least one gate reports a divergence; see the certificates.",
              file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
