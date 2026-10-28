"""Describe what a loader delivers, with or without a contract.

A check answers "does the loader do what was declared?", and needs the
declaration. A profile answers the other half --- "what does the loader do?"
--- and needs none. It still needs an execution protocol: an adapter, fixtures,
a requested sample length, a seed and a number of draws. Those settings are
recorded with the profile; they are not expectations the loader can fail.

The profile is produced on every run. Without a contract it is the whole
result; with one, the certificate carries it next to the verdicts, so a reader
sees what was delivered as well as whether it met the requirement.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from .adapter import LoaderSpec
from .checks import (check_content_scale, check_content_scale_by_item,
                     measure_content_scale)
from .fixtures import (SCALE_PROBE_SOURCES, decode_index, make_index_encoded,
                       make_pair, make_scale_grid)
from .seeding import seed_all

__all__ = ["describe_loader", "PROBE_FRAMES", "PROBE_SEED"]


#: What every probe asks for, the same for every subject. Reading this from a
#: per-subject registry would be reading a declaration, and the point of a
#: profile is that it reads none: a loader that ignores the request and returns
#: one frame is reporting its own behaviour, which is the observation wanted.
PROBE_FRAMES = 5
PROBE_SEED = 0
WINDOW_DRAWS = 8
OPERATOR_DRAWS = 12
INPUT_SOURCE_DRAWS = 8
CONTENT_ITEM_DRAWS = 8
STREAM_FIXTURE = (512, 512)


def _hwc(a):
    a = np.asarray(a, dtype=np.float64)
    while a.ndim > 3:
        a = a[0]
    if a.ndim == 3 and a.shape[0] in (1, 2, 3) and a.shape[2] not in (1, 2, 3):
        a = np.transpose(a, (1, 2, 0))
    return a


def _to_255(a):
    """Map the numeric conventions used by these adapters to level units."""
    a = np.asarray(a, dtype=np.float64)
    if a.size == 0:
        return a
    if a.max() <= 1.0 + 1e-6 and a.min() >= -1e-6:
        return a * 255.0
    if a.max() <= 1.0 + 1e-6 and a.min() >= -1.0 - 1e-6:
        return (a + 1.0) * 127.5
    return a


def _checkerboard_score(a) -> float:
    """Fraction near the black/white anchors of the stored checkerboard.

    This is the same discriminator used by ``precomputed_input``.  It tests
    for the fixture itself, rather than treating any high-variance output as
    evidence that the stored input was read.
    """
    # Both anchors, as in the gate: black alone is an inpainting hole or a
    # [-1, 1] grey read as [0, 1], not the checkerboard coming back.
    v = _to_255(a).ravel()
    black, white = float(np.mean(v < 15.0)), float(np.mean(v > 240.0))
    return black + white if min(black, white) >= 0.2 else 0.0


def observe_streams(adapter, workdir: Path) -> dict:
    """Per-frame shape, observed extrema and channels of each stream."""
    seed_all(PROBE_SEED)
    # Larger than the patch most loaders are configured to crop, so the shape
    # recorded is the loader's, not the fixture's.
    make_pair(workdir, n_frames=100, shape=STREAM_FIXTURE, gt_kind="signature")
    spec = LoaderSpec(gt_root=workdir / "GT", lq_root=workdir / "LQ",
                      num_frames=PROBE_FRAMES, seed=PROBE_SEED)
    loader = adapter.build(spec)
    sample = adapter.sample(loader, 0)
    out = {}
    for name in ("lq", "gt"):
        stream = getattr(sample, name, None)
        if stream is None:
            out[name] = None
            continue
        raw = np.asarray(stream)
        a = _hwc(raw)
        out[name] = {
            "full_shape": [int(v) for v in raw.shape],
            "shape": [int(v) for v in a.shape],
            "frames": int(raw.shape[0]) if raw.ndim == 4 else 1,
            "range": [round(float(a.min()), 3), round(float(a.max()), 3)],
            "channels": int(a.shape[2]) if a.ndim == 3 else 1,
        }
    return out


def observe_operators(adapter, workdir: Path,
                      draws: int = OPERATOR_DRAWS) -> dict:
    """How many distinct operator orders the loader realises over N draws."""
    recorder = getattr(adapter, "recorder", None)
    if recorder is None:
        return {"observed": False,
                "note": "the adapter exposes no operator instrumentation"}
    seed_all(PROBE_SEED)
    make_pair(workdir, n_frames=100, shape=(128, 128))
    spec = LoaderSpec(gt_root=workdir / "GT", lq_root=workdir / "LQ",
                      num_frames=PROBE_FRAMES, seed=PROBE_SEED)
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


def observe_content(adapter, workdir: Path) -> dict:
    """Whether the loader crops the picture, rescales it, or reshapes it."""
    seed_all(PROBE_SEED)
    probes = []
    by_item: dict = {}
    for tag, (sh, sw) in SCALE_PROBE_SOURCES:
        for px, py in ((6, 10), (12, 20), (24, 40), (40, 60)):
            d = workdir / f"{tag}_{px}"
            for side in ("GT", "LQ"):
                make_scale_grid(d / side, n_frames=100, shape=(sh, sw),
                                period_x=px, period_y=py)
            spec = LoaderSpec(gt_root=d / "GT", lq_root=d / "LQ",
                              num_frames=PROBE_FRAMES, seed=PROBE_SEED)
            loader = adapter.build(spec)
            sample = adapter.sample(loader, 0)
            stream = sample.gt if sample.gt is not None else sample.lq
            if stream is None:
                continue
            mx, my = measure_content_scale(stream)
            if mx is not None and my is not None:
                probes.append((stream, px, py, sw / sh))
                # The same fixture, further items. One reading says what the
                # loader did to the item that was looked at; it does not say
                # that every item was treated that way.
                for i in range(CONTENT_ITEM_DRAWS):
                    try:
                        item = adapter.sample(loader, i)
                    except IndexError:
                        break
                    stream_i = item.gt if item.gt is not None else item.lq
                    if stream_i is not None:
                        by_item[f"{tag}:{i}"] = [(stream_i, px, py, sw / sh)]
                break
    res = check_content_scale(probes)
    # ``check_content_scale`` also has an assertion-mode status vocabulary.
    # A contract-free profile must not leak PASS/UNDECL into its public record:
    # here the only distinction is whether the measurement was readable.
    note = res.message.removesuffix("; no declaration of this was found")
    per_item = check_content_scale_by_item(by_item) if by_item else None
    return {"behaviour": res.evidence.get("behaviour"),
            "measurement": ("unreadable" if res.status == "SKIP"
                            else "observed"),
            "note": note, "evidence": res.evidence,
            "per_item": ({"behaviour": per_item.evidence.get("behaviour"),
                          "counts": per_item.evidence.get("counts"),
                          "items_read": per_item.evidence.get("items_read"),
                          "items_unreadable": per_item.evidence.get(
                              "items_unreadable"),
                          "note": per_item.message.removesuffix(
                              "; no declaration of this was found")}
                         if per_item is not None else
                         {"behaviour": None, "items_read": 0,
                          "note": "no item delivered a readable pattern"})}


def observe_window(adapter, workdir: Path,
                   draws: int = WINDOW_DRAWS) -> dict:
    """How many frames arrive, and whether the window moves between draws.

    The fixture writes a recoverable index into every frame, so the window a
    loader opened can be read back from what it delivered. A window that never
    moves is the shape of the seeded ``frozen_window`` defect, and
    saying so needs nothing declared.
    """
    seed_all(PROBE_SEED)
    make_index_encoded(workdir / "GT", n_frames=128)
    make_index_encoded(workdir / "LQ", n_frames=128)
    spec = LoaderSpec(gt_root=workdir / "GT", lq_root=workdir / "LQ",
                      num_frames=PROBE_FRAMES, seed=PROBE_SEED)
    loader = adapter.build(spec)
    windows = []
    for i in range(draws):
        sample = adapter.sample(loader, i)
        stream = sample.gt if sample.gt is not None else sample.lq
        a = np.asarray(stream)
        frames = a.shape[0] if a.ndim == 4 else 1
        ids = []
        for f in range(frames):
            idx = decode_index(a[f] if a.ndim == 4 else a)
            if idx is not None:
                ids.append(idx)
        windows.append(tuple(ids))
    legible = [w for w in windows if w]
    if not legible:
        return {"frames": None,
                "note": "the delivered frames carry no recoverable index"}
    distinct = len(set(legible))
    n = len(legible[0])
    if n <= 1:
        detalle = "single image: there is no window to move"
        distinct_value = None
    elif distinct == 1:
        detalle = f"the same window in all {len(legible)} draws"
        distinct_value = distinct
    else:
        detalle = f"{distinct} distinct windows over {len(legible)} draws"
        distinct_value = distinct
    return {"frames": n, "draws": len(legible),
            "distinct_windows": distinct_value,
            "note": f"{n} frame(s) per item; {detalle}"}


def observe_input_source(adapter, workdir: Path) -> dict:
    """Whether the input is read from storage or manufactured from the target.

    The fixture writes a checkerboard low-quality stream that no degradation
    would produce. If it comes back, the loader read it; if it does not, the
    loader built its own input.
    """
    seed_all(PROBE_SEED)
    make_pair(workdir, n_frames=128, shape=(128, 128),
              gt_kind="flat", lq_kind="checker")
    spec = LoaderSpec(gt_root=workdir / "GT", lq_root=workdir / "LQ",
                      num_frames=PROBE_FRAMES, seed=PROBE_SEED,
                      use_precomputed_lq=True)
    loader = adapter.build(spec)
    scores = []
    has_target = None
    for i in range(INPUT_SOURCE_DRAWS):
        item = adapter.sample(loader, i)
        if item.lq is None:
            continue
        has_target = item.gt is not None
        scores.append(_checkerboard_score(item.lq))
    if not scores:
        return {"source": None, "note": "no comparable pair was delivered"}
    if not has_target:
        # A zero-reference loader has no target from which an alternative
        # input could be manufactured, so this two-source comparison does not
        # apply.  Do not turn one delivered stream into an inferred lineage.
        return {"source": "one stream; source not classified",
                "draws": len(scores),
                "checkerboard_scores": [round(v, 4) for v in scores],
                "note": "zero-reference: no target exists for the comparison"}
    votes = [score >= 0.80 for score in scores]
    reads = sum(votes)
    if reads not in (0, len(votes)):
        return {"source": "not decided",
                "draws": len(votes), "read_votes": reads,
                "checkerboard_scores": [round(v, 4) for v in scores],
                "note": (f"the stored checkerboard came back in {reads} of "
                         f"{len(votes)} "
                         "draws and not in the others, so this reading is not "
                         "settled for this subject")}
    read_all = bool(reads)
    return {"source": "reads a pre-rendered input" if read_all
            else "manufactures its input",
            "draws": len(votes), "read_votes": reads,
            "checkerboard_scores": [round(v, 4) for v in scores],
            "note": (f"the stored checkerboard was delivered in all "
                     f"{len(votes)} draws" if read_all
                     else "the stored checkerboard was delivered in none of "
                          f"{len(votes)} draws")}


def describe_loader(adapter) -> dict:
    """The profile of one loader: streams, operators, content, window, source.

    Each part is observed on its own fixture. A part the loader cannot be
    observed for is recorded as such, with the reason, rather than dropped:
    what could not be seen is part of the description.
    """
    out = {"loader": getattr(adapter, "name", ""),
           "probe": {"seed": PROBE_SEED,
                     "requested_frames": PROBE_FRAMES,
                     "window_draws": WINDOW_DRAWS,
                     "operator_draws": OPERATOR_DRAWS,
                     "input_source_draws": INPUT_SOURCE_DRAWS,
                     "stream_fixture": "100 frames, {}x{} signature".format(*STREAM_FIXTURE),
                     "content_sources": ["x".join(map(str, s))
                                         for _, s in SCALE_PROBE_SOURCES]}}
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        for name, fn, sub in (("streams", observe_streams, "s"),
                              ("operators", observe_operators, "o"),
                              ("content", observe_content, "c"),
                              ("window", observe_window, "w"),
                              ("input_source", observe_input_source, "i")):
            try:
                out[name] = fn(adapter, base / sub)
            except Exception as exc:                              # noqa: BLE001
                out[name] = {"note": f"not readable: {type(exc).__name__}"}
    return out
