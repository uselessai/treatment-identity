#!/usr/bin/env python
"""E19 — Paired pilot: do outcomes differ between the two indexing modes?

The audit established that RTN's loader computes a sampling window and then
opens the clip prefix, making approximately 7% of the source-frame positions
reachable. Whether this changes downstream metrics is a separate question, and
this paired pilot probes it.

    arm A   RTN_FRAME_INDEX=legacy_prefix    (the released behaviour)
    arm B   RTN_FRAME_INDEX=correct_window   (the frame the sampler chose)

The design intends to hold everything else fixed: the same frozen LQ/GT bytes
in both arms (RTN_PRECOMPUTED_LQ=1, so neither arm re-degrades and neither
sharpens a target twice), working tree, environment, losses, batch size, and
paired seeds. Closeout must hash the effective working tree and run artefacts;
the upstream commit alone is not sufficient provenance.

We do not predict the direction of the effect. With three paired seeds, every
delta is reported descriptively; no result is interpreted as significance,
equivalence, or a bound on the consequence.

    python e19_prefix_vs_window.py --dry-run
    python e19_prefix_vs_window.py --run --seeds 2021 7 424242

Before every training, the two gates relevant to this contrast --- the
pre-computed-input gate and the temporal-window gate --- are run against this
arm's environment, and the run is refused unless they report what the arm
claims. They run on the protocol's synthetic fixture, not on the training clips:
the fixture reproduces the defect at a smaller size, so this certifies the code
path the arm takes, not the exact tensors it will see. The generated YAML
records, but does not itself apply, the treatment environment; the launcher
checks that record against the variables passed to the child. Frozen data
manifests are re-hashed at campaign start and require final revalidation. No GPU
time is spent on an uncertified arm.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path("/home/laura/02ImproveData")
RTN = ROOT / "zBringing-Old-Films-Back-to-Life"
PAPER = ROOT / "paper"
PROTOCOL = PAPER / "treatment_identity"
LOGS = PAPER / "logs"
CERTS = PAPER / "data" / "e19_certificates"
PY = "/home/laura/miniconda3/envs/rrtn/bin/python"

# Both arms consume this frozen bank, byte for byte.
DATA_GT = ROOT / "cross_train_data_20260520_142038" / "rtn" / "GT" / "train"
DATA_LQ = ROOT / "cross_train_data_20260520_142038" / "rtn" / "LQ" / "train"
TEMPLATE_CFG = RTN / "configs" / "ms_rtn_data_rtn_s7.yaml"

ARMS = {
    "legacy": {"RTN_FRAME_INDEX": "legacy_prefix", "RTN_PRECOMPUTED_LQ": "1"},
    "window": {"RTN_FRAME_INDEX": "correct_window", "RTN_PRECOMPUTED_LQ": "1"},
}


def log(msg: str) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {msg}"
    print(line, flush=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    with open(LOGS / "e19_prefix_vs_window.log", "a") as fh:
        fh.write(line + "\n")


def run_name(arm: str, seed: int) -> str:
    return f"e19_{arm}_s{seed}"


def manifest(root: Path, out: Path) -> dict:
    """Re-hash every file both arms consume and compare with the frozen record.

    A configuration hash pins what was requested; this pins what is on disk. The
    two arms are only "the same data" if this manifest is the same manifest.
    """
    files = sorted(p for p in root.rglob("*") if p.is_file())
    h = hashlib.sha256()
    entries = []
    for f in files:
        d = hashlib.sha256(f.read_bytes()).hexdigest()
        h.update(f.relative_to(root).as_posix().encode())
        h.update(d.encode())
        entries.append({"path": f.relative_to(root).as_posix(), "sha256": d})
    rec = {"root": str(root), "n_files": len(entries), "manifest_sha256": h.hexdigest(),
           "files": entries}
    # Recomputed on every start and compared against the frozen copy. Returning a
    # cached hash would let the data change under a running experiment while the
    # certificate kept quoting the old digest --- a provenance claim that is true
    # of a file and false of the run.
    if out.exists():
        old = json.loads(out.read_text())
        if old.get("manifest_sha256") != rec["manifest_sha256"]:
            raise SystemExit(
                f"ABORT: {root} changed since it was frozen.\n"
                f"  frozen  {old.get('manifest_sha256')} ({old.get('n_files')} files)\n"
                f"  current {rec['manifest_sha256']} ({rec['n_files']} files)\n"
                "Both arms must consume identical bytes; refusing to continue.")
        return old
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1) + "\n")
    return rec


def write_config(arm: str, seed: int) -> Path:
    """Derive an immutable config, changing only what the arm requires."""
    import yaml
    cfg = yaml.safe_load(TEMPLATE_CFG.read_text())
    cfg["name"] = run_name(arm, seed)
    cfg["seed"] = seed
    ds = cfg["datasets"]["train"] if "datasets" in cfg else cfg["dataset"]["train"]
    ds["dataroot_gt"] = str(DATA_GT)
    ds["dataroot_lq"] = str(DATA_LQ)
    # Both arms consume the frozen pre-rendered inputs. Leaving this false while
    # the behaviour is switched by an environment variable would reproduce, in our
    # own experiment, the defect this paper reports: a treatment that is not
    # recoverable from the configuration. The switches are recorded too.
    ds["use_precomputed_lq_in_train"] = True
    cfg["treatment_env"] = dict(ARMS[arm])
    out = RTN / "configs" / f"{run_name(arm, seed)}.yaml"
    rendered = yaml.safe_dump(cfg, sort_keys=False)
    if out.exists():
        if out.read_text() != rendered:
            raise SystemExit(
                f"ABORT: refusing to overwrite non-matching E19 config {out}. "
                "Archive the old experiment explicitly before regenerating it.")
    else:
        out.write_text(rendered)
    return out


def certify(arm: str, seed: int) -> tuple[bool, dict]:
    """Run the identity gates under this arm's environment before training it."""
    env = dict(os.environ, **ARMS[arm], PYTHONPATH=str(PROTOCOL))
    code = f"""
import sys, json, tempfile, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, {str(PROTOCOL)!r})
from adapters.vp_code import VPCodeAdapter
from treatment_identity import check_precomputed_input, check_temporal_window
ad = VPCodeAdapter("RTN", {str(RTN)!r},
                   texture_bank={str(ROOT / 'zMambaOFR' / 'noise_data')!r}, channels=3)
wd = tempfile.mkdtemp()
out = {{}}
for fn in (check_precomputed_input, check_temporal_window):
    r = fn(ad, wd)
    out[r.name] = {{"status": r.status, "message": r.message, "evidence": r.evidence}}
print("___CERT___" + json.dumps(out))
"""
    p = subprocess.run([PY, "-c", code], capture_output=True, text=True, env=env, timeout=600)
    marker = "___CERT___"
    if marker not in p.stdout:
        return False, {"error": (p.stderr or p.stdout)[-800:]}
    cert = json.loads(p.stdout.split(marker, 1)[1].strip())

    # Both arms must deliver the frozen input; the arms differ only in the window.
    ok = cert.get("precomputed_input", {}).get("status") == "PASS"
    want = "PASS" if arm == "window" else "FAIL"
    ok &= cert.get("temporal_window", {}).get("status") == want
    cert["_expected_temporal_window"] = want
    cert["_arm"] = arm
    cert["_seed"] = seed
    return ok, cert


def check_config_matches_arm(cfg: Path, arm: str) -> None:
    """Refuse when the YAML and the environment disagree about the treatment.

    The paper's own finding is that a configuration can declare one thing while
    the loader does another. The least we can do is not ship that property in the
    experiment that tests it.
    """
    import yaml
    c = yaml.safe_load(cfg.read_text())
    ds = c["datasets"]["train"] if "datasets" in c else c["dataset"]["train"]
    want_pre = ARMS[arm]["RTN_PRECOMPUTED_LQ"] != "0"
    if bool(ds.get("use_precomputed_lq_in_train")) != want_pre:
        raise SystemExit(f"ABORT {cfg.name}: YAML says use_precomputed_lq_in_train="
                         f"{ds.get('use_precomputed_lq_in_train')} but the arm sets "
                         f"RTN_PRECOMPUTED_LQ={ARMS[arm]['RTN_PRECOMPUTED_LQ']}")
    if c.get("treatment_env") != dict(ARMS[arm]):
        raise SystemExit(f"ABORT {cfg.name}: recorded treatment_env does not match the arm")


def gpu_compute_pids() -> list[int]:
    """Return compute PIDs, failing closed when GPU state cannot be queried."""
    out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        capture_output=True, text=True, timeout=30)
    if out.returncode != 0:
        raise SystemExit(
            "ABORT: nvidia-smi could not verify exclusive GPU use: "
            + (out.stderr.strip() or f"return code {out.returncode}"))
    pids = []
    for line in out.stdout.splitlines():
        token = line.strip()
        if token:
            try:
                pids.append(int(token))
            except ValueError as exc:
                raise SystemExit(
                    f"ABORT: unexpected nvidia-smi PID output {token!r}") from exc
    return pids


def train(arm: str, seed: int, cfg: Path) -> bool:
    name = run_name(arm, seed)
    env = dict(os.environ, **ARMS[arm])
    cmd = [PY, "VP_code/main_gan.py", "--name", name,
           "--model_name", "RNN_Swin_4", "--discriminator_name", "discriminator",
           "--gpus", "1", "--flow_path", "pretrained_models/raft-sintel.pth"]
    logfile = LOGS / f"{name}.log"
    if logfile.exists() and logfile.stat().st_size:
        raise SystemExit(
            f"ABORT: refusing to append a new run to existing log {logfile}. "
            "Archive the prior attempt and use a clean log.")
    log(f"[train] {name}  env={ARMS[arm]}  -> {logfile}")
    t0 = time.time()
    with open(logfile, "a") as fh:
        p = subprocess.run(cmd, cwd=RTN, env=env, stdout=fh, stderr=subprocess.STDOUT)
    hours = (time.time() - t0) / 3600
    ckpt = RTN / "OUTPUT" / name / "models" / "net_G_20000.pth"
    if p.returncode == 0 and ckpt.exists():
        log(f"[ok] {name} trained in {hours:.1f}h -> {ckpt}")
        return True
    log(f"[FAIL] {name} rc={p.returncode} after {hours:.1f}h; checkpoint present: {ckpt.exists()}")
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--run", action="store_true")
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--certify-only", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[2021, 7, 424242])
    ap.add_argument("--force", action="store_true",
                    help="deprecated; E19 refuses shared-GPU execution")
    args = ap.parse_args()

    if not DATA_GT.exists() or not DATA_LQ.exists():
        log(f"frozen data missing: {DATA_GT} / {DATA_LQ}")
        return 2

    # Alternate which arm leads, so execution order is not confounded with the
    # treatment (the implemented E19 design alternates the leading arm).
    pairs = []
    for i, sd in enumerate(args.seeds):
        order = ("legacy", "window") if i % 2 == 0 else ("window", "legacy")
        pairs.extend((arm, sd) for arm in order)
    log(f"E19: {len(pairs)} trainings = {len(args.seeds)} seeds x 2 arms")

    if args.dry_run:
        for arm, seed in pairs:
            cfg = write_config(arm, seed)
            log(f"[dry] {run_name(arm, seed)}  env={ARMS[arm]}  cfg={cfg.name}")
        log("dry run complete; no GPU used, configs written")
        return 0

    if args.certify_only:
        bad = 0
        for arm in ("legacy", "window"):
            ok, cert = certify(arm, args.seeds[0])
            status = {k: v.get("status") for k, v in cert.items() if isinstance(v, dict)}
            log(f"[cert] arm={arm}: {'PASS' if ok else 'REFUSED'}  {status}")
            bad += int(not ok)
        return 1 if bad else 0

    if args.force:
        log("REFUSED: --force is incompatible with the recorded "
            "exclusive-GPU design")
        return 3
    busy_pids = gpu_compute_pids()
    if busy_pids:
        log(f"GPU is in use by compute PIDs {busy_pids}; paired runs must not "
            "contend for the device.")
        return 3

    gt_man = manifest(DATA_GT, PAPER / "data" / "e19_manifests" / "gt.json")
    lq_man = manifest(DATA_LQ, PAPER / "data" / "e19_manifests" / "lq.json")
    log(f"data frozen: GT {gt_man['n_files']} files sha256 {gt_man['manifest_sha256'][:16]}; "
        f"LQ {lq_man['n_files']} files sha256 {lq_man['manifest_sha256'][:16]}")

    failures = 0
    for arm, seed in pairs:
        name = run_name(arm, seed)
        # Detect bank mutation before every arm, not only when the launcher starts.
        gt_man = manifest(DATA_GT, PAPER / "data" / "e19_manifests" / "gt.json")
        lq_man = manifest(DATA_LQ, PAPER / "data" / "e19_manifests" / "lq.json")
        cfg = write_config(arm, seed)
        check_config_matches_arm(cfg, arm)
        cfg_sha = hashlib.sha256(cfg.read_bytes()).hexdigest()
        ok, cert = certify(arm, seed)
        cert["_config"] = {"path": str(cfg), "sha256": cfg_sha}
        cert["_data"] = {"gt": gt_man["manifest_sha256"], "gt_files": gt_man["n_files"],
                         "lq": lq_man["manifest_sha256"], "lq_files": lq_man["n_files"]}
        cert["_order_index"] = pairs.index((arm, seed))
        cert_path = CERTS / f"{run_name(arm, seed)}.json"
        if cert_path.exists():
            old = json.loads(cert_path.read_text())
            comparable = ("_arm", "_seed", "_expected_temporal_window",
                          "_config", "_data", "_order_index")
            changed = [key for key in comparable if old.get(key) != cert.get(key)]
            for gate in ("precomputed_input", "temporal_window"):
                if old.get(gate, {}).get("status") != cert.get(gate, {}).get("status"):
                    changed.append(f"{gate}.status")
            if changed:
                raise SystemExit(
                    f"ABORT: existing certificate {cert_path} disagrees in "
                    f"{', '.join(changed)}; refusing to overwrite provenance.")
        else:
            CERTS.mkdir(parents=True, exist_ok=True)
            cert_path.write_text(json.dumps(cert, indent=2) + "\n")
        if not ok:
            log(f"[REFUSED] {name}: the identity gates do not report the treatment "
                f"this arm claims. No GPU time spent. Certificate in {CERTS}")
            failures += 1
            continue
        ckpt = RTN / "OUTPUT" / name / "models" / "net_G_20000.pth"
        if ckpt.exists():
            log(f"[skip] {name} checkpoint already present; certification and "
                "configuration were revalidated, but scientific closeout must "
                "still verify the clean run log and checkpoint hash")
            continue
        busy_pids = gpu_compute_pids()
        if busy_pids:
            log(f"[REFUSED] {name}: GPU became busy with compute PIDs {busy_pids}")
            failures += 1
            continue
        log(f"[cert] {name} certified: "
            f"{ {k: v.get('status') for k, v in cert.items() if isinstance(v, dict)} }")
        failures += int(not train(arm, seed, cfg))

    # Scientific closeout still happens in e19_evaluate.py, but the launcher
    # performs a final byte-level bank check before declaring training complete.
    manifest(DATA_GT, PAPER / "data" / "e19_manifests" / "gt.json")
    manifest(DATA_LQ, PAPER / "data" / "e19_manifests" / "lq.json")
    log(f"E19 finished with {failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
