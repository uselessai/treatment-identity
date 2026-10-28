#!/usr/bin/env python3
"""Campaign T --- does the mechanism reach a modality that is not image or video?

The gates were built for loaders that manufacture image or video inputs from
targets. This campaign asks which of their predicates survive a move to a
different shape of data, using a synthetic multivariate time-series loader that
carries the two execution-path defects the audited family carries: an operator
permutation that is computed and discarded, and two conditions declared distinct
that deliver the same window.

The loader and the defects are ours, as in campaign M. What this establishes is
that the predicates can be evaluated on a non-image modality --- not detection
over defects written by other people.

    python3 studies/campaign_T_modality.py --out studies/data_1.7.1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from treatment_identity.adapter import (CallRecorder, ContentContract,  # noqa: E402
                                        Sample, StreamContract)
from treatment_identity.checks import (check_sample_channel_content,  # noqa: E402
                                       check_sample_value_range,
                                       check_separability)

STEPS, SENSORS = 32, 6
AUGS = ("jitter", "scale", "warp", "dropout")


def _jitter(x, s):  return x + 0.01 * s * np.sin(np.arange(x.shape[0]))[:, None]
def _scale(x, s):   return x * (1.0 + 0.02 * s)
def _warp(x, s):    return np.roll(x, int(1 + s) % 3, axis=0)
def _dropout(x, s): y = x.copy(); y[::7] *= 0.5; return y


FN = {"jitter": _jitter, "scale": _scale, "warp": _warp, "dropout": _dropout}


class SensorWindowLoader:
    """A training loader over sensor windows, carrying two seeded defects."""

    #: Declared strengths. The strong condition never reads its own value.
    STRENGTH = {"weak": 1.0, "strong": 4.0}

    def __init__(self, condition: str = "weak", seed: int = 0, n: int = 8):
        self.condition = condition
        self.strength = self.STRENGTH[condition]
        self.seed = seed
        self.n = n

    def __len__(self) -> int:
        return self.n

    def _augment(self, x, rng):
        order = rng.permutation(len(AUGS))          # defect 1: computed...
        for i, _idx in enumerate(order):
            x = FN[AUGS[i]](x, 1.0)                 # ...indexed by i, and 1.0
        return x                                    # defect 2: strength ignored

    def __getitem__(self, i):
        rng = np.random.default_rng(self.seed + i)
        gt = rng.standard_normal((STEPS, SENSORS)).astype(np.float32)
        return self._augment(gt.copy(), rng), gt


def window(condition: str, seed: int = 0, index: int = 0):
    """One window in HWC form: steps x singleton spatial axis x sensors."""
    lq, gt = SensorWindowLoader(condition, seed=seed)[index]
    return lq[:, None, :], gt[:, None, :]


def run() -> dict:
    results: dict[str, dict] = {}

    r = check_separability(lambda: window("weak")[0], lambda: window("strong")[0],
                           declared="distinct", label="weak_vs_strong")
    results["separability"] = {"status": r.status, "message": r.message,
                               "evidence": r.evidence, "ported": "unmodified"}

    lq, gt = window("weak")
    sample = Sample(lq=[lq], gt=[gt])
    # The content gate interprets the last axis as channels.  Place the six
    # sensor variables on that axis and require their signatures to remain
    # distinct; a singleton synthetic image channel would test neither fact.
    stream = StreamContract(value_range=(-8.0, 8.0), channels=SENSORS,
                            require_distinct_channels=True)
    contract = ContentContract(lq=stream, gt=stream)
    for fn, gate in ((check_sample_value_range, "value_range"),
                     (check_sample_channel_content, "channel_content")):
        r = fn(sample, contract)
        results[gate] = {"status": r.status, "message": r.message,
                         "evidence": r.evidence,
                         "ported": "unmodified"}

    # The operator gate's logic is modality-agnostic; its fixture driver is not,
    # because that driver writes image files. The recorder is used directly.
    recorder = CallRecorder()
    original = dict(FN)
    for key in list(FN):
        def wrapped(f=original[key], name=key):
            def call(x, s):
                recorder.record(name)
                return f(x, s)
            return call
        FN[key] = wrapped()
    orders = []
    for seed in range(12):
        recorder.reset()
        window("weak", seed=seed)
        orders.append(tuple(recorder.order(set(original))))
    FN.update(original)
    distinct = sorted({o for o in orders})
    results["operator_trace"] = {
        "status": "FAIL" if len(distinct) == 1 else "PASS",
        "message": f"{len(orders)} draws produced {len(distinct)} distinct order(s)",
        "evidence": {"draws": len(orders), "distinct_orders": len(distinct),
                     "realised_order": list(distinct[0])},
        "ported": "logic only; fixture driver writes image files"}

    results["_not_run"] = {
        "temporal_window": "assumes a sequence of frames",
        "precomputed_input": "builds image fixtures on disk",
        "target_transforms": "builds image fixtures on disk",
        "geometry": "assumes an image aspect ratio"}
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    results = run()
    payload = {
        "campaign": "T",
        "subject": "synthetic multivariate time series",
        "logical_window_shape": [STEPS, SENSORS],
        "delivered_hwc_shape": [STEPS, 1, SENSORS],
        "declared_sensor_channels": SENSORS,
        "distinct_sensor_channels_required": True,
        "authored_by": "this study, as in campaign M",
        "gates": results,
        "summary": {
            "ported_unmodified": 3,
            "logic_ported_new_fixtures_needed": 3,
            "modality_bound": 2,
            "seeded_defects_reported": 2,
        },
    }
    path = args.out / "T_modality.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    for gate, r in results.items():
        if not gate.startswith("_"):
            print(f"  {gate:18s} {r['status']:6s} {r['message'][:70]}")
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
