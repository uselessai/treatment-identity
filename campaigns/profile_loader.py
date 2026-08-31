#!/usr/bin/env python3
"""Profile a training loader without writing a contract for it.

Every gate of this mechanism answers a question of the form "does the loader do
what was declared?", and someone has to write the declaration first. Seven of
the eight need one, which is a real cost: it has to be written per subject, and
writing it by reading the subject's own code makes the check partly circular.

This tool asks the other half of each question --- "what does the loader do?"
--- and needs nothing declared. It points at an adapter, delivers a few samples
from instrumented fixtures, and reports what came back. Nothing here passes or
fails: a profile is a description, and deciding whether a described behaviour is
acceptable is the reader's job, not the tool's.

Run:  python campaigns/profile_loader.py --subject ddcolor_lab
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE / "adapters"))

from treatment_identity.adapter import CallRecorder, LoaderSpec   # noqa: E402
from treatment_identity.checks import (check_content_scale,       # noqa: E402
                                       measure_content_scale)
from treatment_identity.fixtures import make_pair, make_scale_grid  # noqa: E402

import campaign_P_portability as camp                             # noqa: E402


def _hwc(a):
    a = np.asarray(a, dtype=np.float64)
    while a.ndim > 3:
        a = a[0]
    if a.ndim == 3 and a.shape[0] in (1, 2, 3) and a.shape[2] not in (1, 2, 3):
        a = np.transpose(a, (1, 2, 0))
    return a


def observe_streams(adapter, workdir: Path, cfg: dict) -> dict:
    """Shape, numeric range and channel count of what is delivered."""
    make_pair(workdir, n_frames=100, shape=(128, 128), gt_kind="signature")
    spec = LoaderSpec(gt_root=workdir / "GT", lq_root=workdir / "LQ",
                      num_frames=cfg.get("num_frames", 5), seed=0)
    loader = adapter.build(spec)
    sample = adapter.sample(loader, 0)
    out = {}
    for name in ("lq", "gt"):
        stream = getattr(sample, name, None)
        if stream is None:
            out[name] = None
            continue
        a = _hwc(stream)
        out[name] = {
            "shape": [int(v) for v in a.shape],
            "range": [round(float(a.min()), 3), round(float(a.max()), 3)],
            "channels": int(a.shape[2]) if a.ndim == 3 else 1,
        }
    return out


def observe_operators(adapter, workdir: Path, cfg: dict, draws: int = 12) -> dict:
    """How many distinct operator orders the loader realises over N draws."""
    recorder = getattr(adapter, "recorder", None)
    if recorder is None:
        return {"observed": False,
                "note": "the adapter exposes no operator instrumentation"}
    make_pair(workdir, n_frames=100, shape=(128, 128))
    spec = LoaderSpec(gt_root=workdir / "GT", lq_root=workdir / "LQ",
                      num_frames=cfg.get("num_frames", 5), seed=0)
    loader = adapter.build(spec)
    orders = []
    for i in range(draws):
        try:
            recorder.reset()
        except AttributeError:
            pass
        adapter.sample(loader, i)
        seen = recorder.calls if hasattr(recorder, "calls") else []
        if seen:
            orders.append(tuple(seen))
    if not orders:
        return {"observed": False, "note": "no operator calls were seen"}
    distinct = len({o for o in orders})
    return {"observed": True, "draws": len(orders), "distinct_orders": distinct,
            "example": list(orders[0]),
            "note": ("every draw realised the same order"
                     if distinct == 1 else
                     f"{distinct} distinct orders over {len(orders)} draws")}


def observe_content(adapter, workdir: Path, cfg: dict) -> dict:
    """Whether the loader crops the picture, rescales it, or reshapes it."""
    probes = []
    for tag, (sh, sw) in (("w169", (480, 854)), ("w43", (480, 640))):
        for px, py in ((6, 10), (12, 20), (24, 40), (40, 60)):
            d = workdir / f"{tag}_{px}"
            for side in ("GT", "LQ"):
                make_scale_grid(d / side, n_frames=100, shape=(sh, sw),
                                period_x=px, period_y=py)
            spec = LoaderSpec(gt_root=d / "GT", lq_root=d / "LQ",
                              num_frames=cfg.get("num_frames", 5), seed=0)
            loader = adapter.build(spec)
            sample = adapter.sample(loader, 0)
            stream = sample.gt if sample.gt is not None else sample.lq
            if stream is None:
                continue
            mx, my = measure_content_scale(stream)
            if mx is not None and my is not None:
                probes.append((stream, px, py, sw / sh))
                break
    res = check_content_scale(probes)
    return {"behaviour": res.evidence.get("behaviour"), "status": res.status,
            "note": res.message, "evidence": res.evidence}


def profile(key: str, cfg: dict) -> dict:
    module = __import__(cfg["module"])
    adapter = getattr(module, cfg["cls"])()   # su propia raiz por defecto
    out = {"subject": key, "project": getattr(adapter, "project", ""),
           "loader": getattr(adapter, "name", "")}
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        out["streams"] = observe_streams(adapter, base / "s", cfg)
        out["operators"] = observe_operators(adapter, base / "o", cfg)
        out["content"] = observe_content(adapter, base / "c", cfg)
    return out


def render(p: dict) -> str:
    lines = [f"{p['loader']}  ({p['project']})", "-" * 66]
    for name in ("lq", "gt"):
        s = p["streams"].get(name)
        if s is None:
            lines.append(f"  {name:9} not delivered")
        else:
            lines.append(f"  {name:9} shape {s['shape']}, "
                         f"range [{s['range'][0]}, {s['range'][1]}], "
                         f"{s['channels']} channel(s)")
    op = p["operators"]
    lines.append(f"  operators {op.get('note', '')}"
                 + (f" ({', '.join(op['example'])})" if op.get("example") else ""))
    c = p["content"]
    lines.append(f"  content   {c['behaviour'] or 'not measurable'}: {c['note']}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subject", action="append",
                    help="profile only these (default: every registered subject)")
    ap.add_argument("--json", type=Path, help="also write the profiles here")
    args = ap.parse_args()

    keys = args.subject or list(camp.SUBJECTS)
    results = []
    for key in keys:
        cfg = camp.SUBJECTS[key]
        camp._isolate_shared_package_names(cfg["root"])
        try:
            p = profile(key, cfg)
            results.append(p)
            print(render(p))
        except Exception as exc:                                  # noqa: BLE001
            print(f"{key}: could not be profiled: "
                  f"{type(exc).__name__}: {str(exc)[:120]}")
        print()
    if args.json:
        args.json.write_text(json.dumps(results, indent=2))
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
