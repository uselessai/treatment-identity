"""The six gates.

Each gate targets one class of divergence between the treatment a pipeline
declares and the tensor it delivers. Every gate is written so that its failure
mode yields ``FAIL`` rather than a plausible number --- including the one that
is easiest to miss, :func:`check_separability`, where the failure is two arms
being *identical* when they were declared distinct.

All gates run on CPU against synthetic fixtures and complete in seconds.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .adapter import (AdapterError, CallRecorder, GeneratorReached,
                      LoaderAdapter, LoaderSpec, Sample)
from . import fixtures as fx

__all__ = [
    "CheckResult",
    "check_precomputed_input",
    "check_temporal_window",
    "check_separability",
    "check_target_transforms",
    "check_operator_trace",
    "check_geometry",
    "ALL_CHECKS",
]

PASS, FAIL, SKIP, NA = "PASS", "FAIL", "SKIP", "N/A"


@dataclass
class CheckResult:
    name: str
    status: str
    message: str
    evidence: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == PASS

    @property
    def is_divergence(self) -> bool:
        """Only FAIL is a divergence: N/A and SKIP assert nothing."""
        return self.status == FAIL

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"[{self.status:4}] {self.name}: {self.message}"


def _to_hwc(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    if a.ndim == 4:
        a = a[0]
    if a.ndim == 3 and a.shape[0] in (1, 3) and a.shape[-1] not in (1, 3):
        a = np.moveaxis(a, 0, -1)
    return a


def _frames(seq) -> list[np.ndarray]:
    arr = np.asarray(seq)
    if arr.ndim == 4:
        return [_to_hwc(arr[i]) for i in range(arr.shape[0])]
    return [_to_hwc(arr)]


def _to_255(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    if a.max() <= 1.0 + 1e-6 and a.min() >= -1e-6:
        return a * 255.0
    if a.max() <= 1.0 + 1e-6:
        return (a + 1.0) * 127.5
    return a


def _bimodality(a: np.ndarray) -> float:
    """Fraction of pixels within 15 levels of pure black or pure white.

    A checkerboard scores ~1.0; any degradation of a flat mid-grey field scores
    near 0. The statistic is deliberately crude so that it cannot be gamed by a
    mild blur.
    """
    v = _to_255(a).ravel()
    return float(np.mean((v < 15) | (v > 240)))


# --------------------------------------------------------------------------
# Gate 1 — is the pre-computed input actually opened?
# --------------------------------------------------------------------------

def check_precomputed_input(adapter: LoaderAdapter, workdir: Path,
                            *, declared: bool = True) -> CheckResult:
    """Deliver a checkerboard as the pre-computed input over a flat target.

    Two independent discriminators, in order of strength:

    1. The online generator is replaced by a sentinel that raises
       :class:`GeneratorReached`. A loader honouring its pre-computed-input flag
       never reaches it; one ignoring the flag raises, and the exception is the
       finding rather than a plausible tensor. Adapters that cannot patch their
       generator skip this step.
    2. The delivered tensor is inspected: the checkerboard is bimodal, and no
       degradation of a flat mid-grey target can be.

    ``declared=False`` records that the pipeline never claims the feature, so
    its absence is ``N/A`` rather than a defect: only a configuration that sets
    a flag the loader ignores is a divergence.
    """
    root = Path(workdir) / "g1"
    fx.make_pair(root, gt_kind="flat", lq_kind="checker", n_frames=16)
    spec = LoaderSpec(gt_root=root / "GT", lq_root=root / "LQ",
                      use_precomputed_lq=True, num_frames=5, seed=0, train=True)
    ev: dict[str, Any] = {"declared_use_precomputed": declared,
                          "sentinel_generator": False}

    # -- discriminator 1: does the loader reach the generator at all? ------
    disable = getattr(adapter, "disable_generator", None)
    if callable(disable):
        try:
            with disable() as patched:
                ev["sentinel_generator"] = True
                ev["generators_disabled"] = list(patched or [])
                try:
                    adapter.sample(adapter.build(spec), 0)
                    ev["generator_reached"] = False
                except GeneratorReached as e:
                    ev["generator_reached"] = True
                    ev["sentinel_message"] = str(e)
        except (AdapterError, NotImplementedError) as e:
            ev["sentinel_error"] = str(e)

    # -- discriminator 2: is the delivered tensor the checkerboard? --------
    try:
        sample = adapter.sample(adapter.build(spec), 0)
    except NotImplementedError as e:
        return CheckResult("precomputed_input", SKIP, f"adapter cannot build: {e}", ev)

    lq = _frames(sample.lq)
    score = float(np.mean([_bimodality(f) for f in lq]))
    ev["bimodality"] = round(score, 4)
    ev["n_frames_delivered"] = len(lq)

    delivered = score >= 0.80 and not ev.get("generator_reached", False)
    if delivered:
        return CheckResult("precomputed_input", PASS,
                           "pre-computed input reaches the model", ev)

    how = ("the sentinel generator was reached, so the loader re-generates"
           if ev.get("generator_reached")
           else f"the delivered tensor is not the checkerboard "
                f"(bimodality {score:.3f}, expected >=0.80)")
    if not declared:
        return CheckResult(
            "precomputed_input", NA,
            f"this pipeline declares no pre-computed-input branch; {how}. "
            "Not a divergence: nothing claimed it.", ev)
    return CheckResult(
        "precomputed_input", FAIL,
        f"the loader did not deliver the pre-computed input: {how}. "
        "The configured flag does not reach the delivered tensor.", ev)


# --------------------------------------------------------------------------
# Gate 2 — does the computed temporal window reach the file system?
# --------------------------------------------------------------------------

def check_temporal_window(adapter: LoaderAdapter, workdir: Path,
                          n_probe: int = 8) -> CheckResult:
    """Ask one clip for several entries and read back which frames arrived.

    Every pixel of fixture frame *i* equals *i*, so the delivered tensor states
    its own provenance. A loader that computes a window and then opens the clip
    prefix returns the same indices for every entry.
    """
    root = Path(workdir) / "g2"
    n_frames = 32
    fx.make_pair(root, gt_kind="index", lq_kind="index", n_frames=n_frames)
    spec = LoaderSpec(gt_root=root / "GT", lq_root=root / "LQ",
                      use_precomputed_lq=True, num_frames=5, seed=0, train=True)
    try:
        loader = adapter.build(spec)
    except NotImplementedError as e:
        return CheckResult("temporal_window", SKIP, f"adapter cannot build: {e}")

    observed: list[tuple[int, ...]] = []
    claimed: list[tuple[int, ...] | None] = []
    for i in range(n_probe):
        try:
            s = adapter.sample(loader, i)
        except IndexError:
            break
        ids = tuple(fx.decode_index(f) for f in _frames(s.gt))
        observed.append(ids)
        claimed.append(tuple(s.frame_ids) if s.frame_ids is not None else None)

    if not observed:
        return CheckResult("temporal_window", SKIP, "no samples produced")

    unique_windows = {w for w in observed}
    unique_frames = {i for w in observed for i in w if i is not None}
    coverage = len(unique_frames) / n_frames
    ev = {"windows_observed": [list(w) for w in observed],
          "distinct_windows": len(unique_windows),
          "unique_frames": sorted(i for i in unique_frames if i is not None),
          "coverage_of_clip": round(coverage, 4)}

    # A loader that discards its window returns the prefix every time.
    prefix = tuple(range(len(observed[0])))
    if len(unique_windows) == 1 and observed[0] == prefix:
        return CheckResult(
            "temporal_window", FAIL,
            f"every one of the {len(observed)} entries returned the clip prefix "
            f"{list(prefix)}; the computed window does not reach the file system. "
            f"Only {len(unique_frames)}/{n_frames} frames "
            f"({coverage:.1%}) of the clip are reachable.", ev)

    if len(unique_windows) == 1:
        return CheckResult(
            "temporal_window", FAIL,
            f"all {len(observed)} entries returned the same window {list(observed[0])}; "
            "the sampler is inert.", ev)

    # If the loader states which frames it picked, the claim must match delivery.
    for obs, cl in zip(observed, claimed):
        if cl is not None and tuple(cl) != tuple(obs):
            ev["mismatch"] = {"claimed": list(cl), "delivered": list(obs)}
            return CheckResult(
                "temporal_window", FAIL,
                f"the loader reported frames {list(cl)} but delivered {list(obs)}.", ev)

    return CheckResult("temporal_window", PASS,
                       f"{len(unique_windows)} distinct windows over "
                       f"{len(unique_frames)}/{n_frames} frames "
                       f"({coverage:.1%} of the clip)", ev)


# --------------------------------------------------------------------------
# Gate 3 — are treatments declared distinct actually distinguishable?
# --------------------------------------------------------------------------

def check_separability(build_a: Callable[[], np.ndarray],
                       build_b: Callable[[], np.ndarray],
                       *, declared: str = "distinct",
                       tol: float = 1e-6,
                       label: str = "A_vs_B") -> CheckResult:
    """Compare two treatments under matched seeds.

    ``declared="distinct"`` fails on unexpected *equality*;
    ``declared="identical"`` fails on unexpected difference. This gate concerns
    one controlled realisation; it does not establish distributional difference
    or equivalence.
    """
    a = _to_255(np.asarray(build_a(), dtype=np.float64))
    b = _to_255(np.asarray(build_b(), dtype=np.float64))
    if a.shape != b.shape:
        ev = {"shape_a": list(a.shape), "shape_b": list(b.shape)}
        if declared == "distinct":
            return CheckResult(f"separability[{label}]", PASS,
                               "treatments differ in tensor shape", ev)
        return CheckResult(f"separability[{label}]", FAIL,
                           "treatments declared identical differ in shape", ev)

    delta = float(np.abs(a - b).max())
    mae = float(np.abs(a - b).mean())
    ev = {"max_abs_delta_0_255": round(delta, 10), "mae_0_255": round(mae, 10),
          "declared": declared, "tolerance": tol}

    if declared == "distinct":
        if delta <= tol:
            return CheckResult(
                f"separability[{label}]", FAIL,
                "UNEXPECTED EQUALITY: two treatments declared distinct produced "
                f"tensors identical to within {tol:g} (max delta {delta:.3e}). "
                "The declared distinction is not observable under this probe; "
                "verify treatment identity before interpreting a comparison.", ev)
        return CheckResult(f"separability[{label}]", PASS,
                           f"treatments are distinguishable (max delta {delta:.4g})", ev)

    if delta > tol:
        return CheckResult(
            f"separability[{label}]", FAIL,
            f"treatments declared identical differ by {delta:.3e} > {tol:g}", ev)
    return CheckResult(f"separability[{label}]", PASS,
                       f"treatments declared identical agree to {delta:.3e}", ev)


# --------------------------------------------------------------------------
# Gate 4 — how many times is the target transformed?
# --------------------------------------------------------------------------

def check_target_transforms(adapter: LoaderAdapter, workdir: Path,
                            recorder: CallRecorder, transform_name: str,
                            expected: int) -> CheckResult:
    """Count target-side transformations against the number the *paper* declares.

    ``expected`` is read off the publication, not off the code: it is the claim
    under test. A transformation the papers never mention is therefore
    ``expected=0``, and observing it once is a ``FAIL`` --- the undeclared
    transformation is the finding. Passing the observed count as ``expected``
    would make the gate assert the code against itself and it could never fail.

    The same gate catches the accidental second application that arises when a
    pipeline is fed a target another pipeline already transformed.
    """
    root = Path(workdir) / "g4"
    fx.make_pair(root, gt_kind="flat", lq_kind="checker", n_frames=8)
    spec = LoaderSpec(gt_root=root / "GT", lq_root=root / "LQ",
                      use_precomputed_lq=False, num_frames=3, seed=0,
                      train=True, target_transforms=expected)
    recorder.reset()
    try:
        loader = adapter.build(spec)
        sample = adapter.sample(loader, 0)
    except NotImplementedError as e:
        return CheckResult("target_transforms", SKIP, f"adapter cannot build: {e}")

    n_frames = len(_frames(sample.gt))
    observed = recorder.count(transform_name)
    per_frame = observed / n_frames if n_frames else 0.0
    gt_hash = hashlib.sha256(np.ascontiguousarray(
        _to_255(np.asarray(sample.gt)).round().astype(np.uint8)).tobytes()).hexdigest()[:16]
    ev = {"transform": transform_name, "calls": observed, "frames": n_frames,
          "calls_per_frame": round(per_frame, 4), "expected_per_frame": expected,
          "delivered_gt_sha256_16": gt_hash}

    if abs(per_frame - expected) < 1e-9:
        return CheckResult("target_transforms", PASS,
                           f"{transform_name} applied {expected}x per frame, as declared", ev)
    if expected == 0:
        return CheckResult(
            "target_transforms", FAIL,
            f"UNDECLARED TARGET TRANSFORM: {transform_name} is applied "
            f"{per_frame:g}x per target frame and no publication of this lineage "
            "declares it. Every model trained here is fitted to a target the "
            "paper does not describe.", ev)
    return CheckResult(
        "target_transforms", FAIL,
        f"{transform_name} applied {per_frame:g}x per target frame, declared {expected}x", ev)


# --------------------------------------------------------------------------
# Gate 5 — is the declared operator sampling policy the one that runs?
# --------------------------------------------------------------------------

def check_operator_trace(adapter: LoaderAdapter, workdir: Path,
                         recorder: CallRecorder, operators: set[str],
                         declared_policy: str = "random_permutation",
                         n_draws: int = 12) -> CheckResult:
    """Draw repeatedly and compare the realised operator order to the declared policy.

    A permutation that is computed and then discarded leaves exactly one
    observed order across every draw.
    """
    root = Path(workdir) / "g5"
    fx.make_pair(root, gt_kind="flat", lq_kind="checker", n_frames=8)
    spec = LoaderSpec(gt_root=root / "GT", lq_root=root / "LQ",
                      use_precomputed_lq=False, num_frames=3, seed=0,
                      train=True, operator_order=declared_policy)
    try:
        loader = adapter.build(spec)
    except NotImplementedError as e:
        return CheckResult("operator_trace", SKIP, f"adapter cannot build: {e}")

    orders: list[tuple[str, ...]] = []
    for i in range(n_draws):
        recorder.reset()
        try:
            adapter.sample(loader, i)
        except IndexError:
            break
        seq = recorder.order(operators)
        # collapse consecutive repeats: one order per frame, we want the pattern
        dedup: list[str] = []
        for c in seq:
            if not dedup or dedup[-1] != c:
                dedup.append(c)
        if dedup:
            orders.append(tuple(dedup[:len(operators)]))

    if not orders:
        return CheckResult("operator_trace", SKIP, "no operator calls observed")

    distinct = {o for o in orders}
    ev = {"declared_policy": declared_policy, "draws": len(orders),
          "distinct_orders": len(distinct),
          "orders_observed": sorted("->".join(o) for o in distinct)}

    if declared_policy == "random_permutation":
        if len(distinct) == 1:
            return CheckResult(
                "operator_trace", FAIL,
                f"policy declared '{declared_policy}' but all {len(orders)} draws "
                f"executed the same order: {'->'.join(next(iter(distinct)))}. "
                "The permutation is computed and discarded.", ev)
        return CheckResult("operator_trace", PASS,
                           f"{len(distinct)} distinct orders over {len(orders)} draws", ev)

    if len(distinct) > 1:
        return CheckResult("operator_trace", FAIL,
                           f"policy declared 'fixed' but {len(distinct)} orders observed", ev)
    return CheckResult("operator_trace", PASS, "fixed order, as declared", ev)


# --------------------------------------------------------------------------
# Gate 6 — does the evaluation contract survive the pipeline?
# --------------------------------------------------------------------------

def check_geometry(delivered_shape: tuple[int, int] | Callable[[tuple[int, int]],
                                                               tuple[int, int]],
                   declared_shape: tuple[int, int],
                   *, aspect_tol: float = 0.005,
                   shape_source: str = "recorded") -> CheckResult:
    """Assert the spatial contract instead of absorbing a resize.

    An evaluator that silently resizes one side to match the other reports
    metrics on a geometry no one specified.

    Unlike the other five, this gate does not drive a loader. Pass a callable to
    have it execute the pipeline's own resize rule on the declared shape
    (``shape_source`` becomes ``"measured"``); pass a tuple to assert against a
    shape observed elsewhere, in which case the certificate records that the
    shape was supplied rather than produced by the gate.
    """
    if callable(delivered_shape):
        delivered_shape = tuple(delivered_shape(declared_shape))  # type: ignore[arg-type]
        shape_source = "measured"
    dh, dw = delivered_shape  # type: ignore[misc]
    eh, ew = declared_shape
    da, ea = dw / dh, ew / eh
    rel = abs(da - ea) / ea
    ev = {"delivered": [dh, dw], "declared": [eh, ew],
          "delivered_aspect": round(da, 5), "declared_aspect": round(ea, 5),
          "aspect_error": round(rel, 5), "shape_source": shape_source}

    if (dh, dw) == (eh, ew):
        return CheckResult("geometry", PASS, f"delivered {dh}x{dw}, as declared", ev)
    if rel <= aspect_tol:
        return CheckResult("geometry", PASS,
                           f"delivered {dh}x{dw} preserves the declared aspect ratio", ev)
    return CheckResult(
        "geometry", FAIL,
        f"delivered {dh}x{dw} distorts the declared {eh}x{ew} aspect ratio by "
        f"{rel:.2%}; metrics computed here are not on the declared geometry", ev)


ALL_CHECKS = (
    "precomputed_input",
    "temporal_window",
    "separability",
    "target_transforms",
    "operator_trace",
    "geometry",
)
