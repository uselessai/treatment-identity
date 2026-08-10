"""Five treatment-delivery gates and one evaluation-integrity check.

The first five checks target divergences between the treatment a pipeline
declares and the sample its loader returns at the training-step-input boundary.
The sixth, :func:`check_geometry`, checks the separate evaluation boundary.
Each check is written so that its failure mode yields ``FAIL`` rather than a
plausible number --- including :func:`check_separability`, where the failure is
two arms being *identical* when they were declared distinct.

All six checks run on CPU against synthetic fixtures and complete in seconds.
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
from .seeding import seed_all

__all__ = [
    "CheckResult",
    "PASS", "FAIL", "SKIP", "NA", "UNDECLARED",
    "check_precomputed_input",
    "check_temporal_window",
    "check_separability",
    "check_target_transforms",
    "check_operator_trace",
    "check_geometry",
    "with_clip_escalation",
    "CLIP_LENGTH_LADDER",
    "ALL_CHECKS",
]

PASS, FAIL, SKIP, NA = "PASS", "FAIL", "SKIP", "N/A"
#: The publication is silent where the code acts. This is not the same as a
#: contradicted claim: an operation a paper never mentions has not been denied,
#: it has been left unspecified. Identity still cannot be established --- there
#: is no declaration for the delivered tensor to match --- so UNDECLARED is a
#: divergence, but calling it a FAIL would attribute to the authors a claim they
#: never made.
UNDECLARED = "UNDECL"


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
        """FAIL and UNDECLARED are divergences; N/A and SKIP assert nothing.

        Both block treatment identity, for different reasons: FAIL because the
        delivered tensor contradicts a declaration, UNDECLARED because there is
        no declaration to compare it against.
        """
        return self.status in (FAIL, UNDECLARED)

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

#: Clip lengths the escalation tries, in order, when nothing is declared.
#: 100 is in the ladder because it is the clip length of REDS, and a loader
#: written against one dataset tends to carry that dataset's shape as an
#: unstated constant.
CLIP_LENGTH_LADDER = (16, 32, 64, 100, 200)


def with_clip_escalation(run, *, gate: str, declared: int | None,
                          default: int) -> "CheckResult":
    """Run a gate body at a clip length, and find out when it needs a longer one.

    Three cases, and the third is the one worth having.

    The arm declares a clip length: honour it. The gate then reports on the
    loader, not on our fixture, which is the point of putting the field in the
    contract.

    Nothing is declared and the default works: nothing to say. Most loaders
    accept whatever length they are handed.

    Nothing is declared and the default does *not* work: escalate. If a longer
    clip makes the same gate run, the loader has a requirement about its input
    data that its interface never stated, and the shortest length that works is
    evidence for it. That is UNDECLARED, not FAIL --- nothing was contradicted,
    because nothing was claimed --- and it is emphatically not an ERROR, which
    is what an unexercisable gate used to report and what taught us nothing.

    Before this existed the harness caught a FileNotFoundError from inside the
    loader and recorded that the subject could not be audited. The subject was
    auditable; the fixture was short, and the reason it had to be long was
    itself the finding.
    """
    if declared is not None:
        return run(declared)
    try:
        return run(default)
    except (FileNotFoundError, IndexError, KeyError) as first:
        for length in (n for n in CLIP_LENGTH_LADDER if n > default):
            try:
                result = run(length)
            except (FileNotFoundError, IndexError, KeyError):
                continue
            result.evidence.update(
                declared_clip_length=None,
                clip_length_used=length,
                clip_length_default=default,
                first_error=f"{type(first).__name__}: {first}"[:200])
            return CheckResult(
                gate, UNDECLARED,
                f"the loader requires source clips of at least {length} frames "
                f"and nothing declared it: it failed on a {default}-frame clip "
                f"with {type(first).__name__} and ran on a {length}-frame one. "
                f"At that length the gate reports {result.status}.",
                result.evidence)
        raise


def check_precomputed_input(adapter: LoaderAdapter, workdir: Path,
                            *, declared: bool = True,
                          clip_length: int = 16) -> CheckResult:
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
    fx.make_pair(root, gt_kind="flat", lq_kind="checker", n_frames=clip_length)
    spec = LoaderSpec(gt_root=root / "GT", lq_root=root / "LQ",
                      use_precomputed_lq=True, num_frames=5, seed=0, train=True)
    ev: dict[str, Any] = {"declared_use_precomputed": declared,
                          "sentinel_generator": False}

    # Seed the probe series here rather than trusting whatever state the
    # subject's own construction left behind. check_temporal_window has done
    # this since 1.1.0; the other adapter-driven gates did not, and on a
    # subject that consumes global randomness that made them irreproducible --
    # this gate returned PASS, 32 and 64 across five runs of the same command
    # against BasicSR's REDS loader. An instrument that is not itself
    # reproducible cannot certify reproducibility, which is lesson L2 of the
    # article, arriving for the third time.
    seed_all(spec.seed)


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
                           "loader returns the pre-computed input at the "
                           "training-step-input boundary", ev)

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
                          n_probe: int = 8, seed: int = 0,
                          clip_length: int = 32,
                          num_frames: int = 5) -> CheckResult:
    """Ask one clip for several entries and read back which frames arrived.

    Every pixel of fixture frame *i* equals *i*, so the delivered tensor states
    its own provenance. A loader that computes a window and then opens the clip
    prefix returns the same indices for every entry.

    Two quantities come out of this and they are not the same kind of thing. A
    loader that always returns the prefix has an exactly known reachable set:
    the probe count does not matter, because no further probe can reach a
    different frame. A loader that samples its window at random has only the
    coverage *observed* in ``n_probe`` draws under ``seed`` --- a lower bound on
    what it can reach, reproducible but not exhaustive. The result reports
    which of the two it is; conflating them was a defect in an earlier version
    of this protocol.
    """
    root = Path(workdir) / "g2"
    n_frames = clip_length
    fx.make_pair(root, gt_kind="index", lq_kind="index", n_frames=n_frames)
    spec = LoaderSpec(gt_root=root / "GT", lq_root=root / "LQ",
                      use_precomputed_lq=True, num_frames=num_frames,
                      seed=seed, train=True)
    try:
        loader = adapter.build(spec)
    except NotImplementedError as e:
        return CheckResult("temporal_window", SKIP, f"adapter cannot build: {e}")

    # Build may consume randomness of its own, so the probe series is seeded
    # here rather than trusting the state build left behind. Without this a
    # loader that samples its window at random answers differently every run.
    rng_record = seed_all(seed)

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

    # Length before contents.
    #
    # This gate used to assert WHICH frames arrived and never WHETHER the right
    # number of them did, so a single-image loader returning one frame where the
    # contract asked for five was granted a pass: eight one-frame samples are
    # eight distinct windows, and the coverage figure came out fine. Reported as
    # lesson L6 of the article that describes this package, and repaired here.
    #
    # The two outcomes are deliberately different statuses. A contract that asks
    # for a window of n and receives one of m contradicts a declaration, which
    # is FAIL. A contract that asks for no window at all -- a single-image
    # pipeline honestly declared as num_frames=1 -- gives this gate nothing to
    # assert over, which is N/A and not a pass: the released loader never
    # declared the branch under test.
    delivered = len(observed[0])
    if spec.num_frames <= 1:
        return CheckResult(
            "temporal_window", NA,
            f"the arm declares a window of {spec.num_frames} frame(s): there is "
            "no temporal sampling for this gate to assert over",
            {"declared_num_frames": spec.num_frames,
             "delivered_window": delivered})
    if delivered != spec.num_frames:
        return CheckResult(
            "temporal_window", FAIL,
            f"the contract declares a window of {spec.num_frames} frames and "
            f"the loader delivered {delivered}",
            {"declared_num_frames": spec.num_frames,
             "delivered_window": delivered,
             "windows_observed": len({len(o) for o in observed})})

    unique_windows = {w for w in observed}
    unique_frames = {i for w in observed for i in w if i is not None}
    coverage = len(unique_frames) / n_frames
    ev = {"windows_observed": [list(w) for w in observed],
          "distinct_windows": len(unique_windows),
          "unique_frames": sorted(i for i in unique_frames if i is not None),
          "coverage_of_clip": round(coverage, 4),
          "n_probe": len(observed),
          "rng": rng_record,
          # Exhaustive when every probe returned the same window: no further
          # probe can widen the set. A sampling estimate otherwise.
          "coverage_is_exhaustive": len(unique_windows) == 1}

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
                       f"{len(unique_windows)} distinct windows; "
                       f"{len(unique_frames)}/{n_frames} frames "
                       f"({coverage:.1%} of the clip) reached in "
                       f"{len(observed)} probes at seed {seed} --- a "
                       f"reproducible lower bound on coverage, not the "
                       f"reachable set", ev)


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
                            expected: int, declared: bool = True,
                          clip_length: int = 8) -> CheckResult:
    """Count target-side transformations against the number the *paper* declares.

    ``expected`` is read off the publication, not off the code: it is the claim
    under test. Passing the observed count as ``expected`` would make the gate
    assert the code against itself, and it could never fail.

    ``declared`` says whether the publication addresses this transformation at
    all. It separates two findings that an earlier version of this gate merged:

    * ``declared=True``  --- the paper states a count and the code disagrees.
      The delivered tensor contradicts a claim: ``FAIL``.
    * ``declared=False`` --- the paper never mentions the transformation and the
      code applies it. Nothing has been contradicted, because nothing was
      claimed; what is missing is the declaration itself: ``UNDECLARED``.

    The distinction matters for what one may write about the audited authors.
    Silence is under-specification, not a false statement, and the audit should
    not report it as one. Both block identity, and both are divergences.

    The same gate catches the accidental second application that arises when a
    pipeline is fed a target another pipeline already transformed.
    """
    root = Path(workdir) / "g4"
    fx.make_pair(root, gt_kind="flat", lq_kind="checker", n_frames=clip_length)
    spec = LoaderSpec(gt_root=root / "GT", lq_root=root / "LQ",
                      use_precomputed_lq=False, num_frames=3, seed=0,
                      train=True, target_transforms=expected)
    seed_all(spec.seed)
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
          "publication_declares_it": declared,
          "delivered_gt_sha256_16": gt_hash}

    if abs(per_frame - expected) < 1e-9:
        return CheckResult("target_transforms", PASS,
                           f"{transform_name} applied {expected}x per frame, as declared", ev)
    if expected == 0 and not declared:
        return CheckResult(
            "target_transforms", UNDECLARED,
            f"UNDECLARED TARGET TRANSFORM: {transform_name} is applied "
            f"{per_frame:g}x per target frame and no publication of this lineage "
            "mentions it. The papers do not deny it --- they are silent --- so "
            "the target returned by the loader cannot be reconstructed from "
            "them.", ev)
    return CheckResult(
        "target_transforms", FAIL,
        f"{transform_name} applied {per_frame:g}x per target frame, declared {expected}x", ev)


# --------------------------------------------------------------------------
# Gate 5 — is the declared operator sampling policy the one that runs?
# --------------------------------------------------------------------------

def check_operator_trace(adapter: LoaderAdapter, workdir: Path,
                         recorder: CallRecorder, operators: set[str],
                         declared_policy: str = "random_permutation",
                         n_draws: int = 12,
                          clip_length: int = 8) -> CheckResult:
    """Draw repeatedly and compare the realised operator order to the declared policy.

    A permutation that is computed and then discarded leaves exactly one
    observed order across every draw.
    """
    root = Path(workdir) / "g5"
    fx.make_pair(root, gt_kind="flat", lq_kind="checker", n_frames=clip_length)
    spec = LoaderSpec(gt_root=root / "GT", lq_root=root / "LQ",
                      use_precomputed_lq=False, num_frames=3, seed=0,
                      train=True, operator_order=declared_policy)
    seed_all(spec.seed)
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
# Complementary evaluation-integrity check — does the shape contract survive?
# --------------------------------------------------------------------------

def check_geometry(delivered_shape: tuple[int, int] | Callable[[tuple[int, int]],
                                                               tuple[int, int]],
                   declared_shape: tuple[int, int],
                   *, aspect_tol: float = 0.005,
                   shape_source: str = "recorded") -> CheckResult:
    """Assert the spatial contract instead of absorbing a resize.

    An evaluator that silently resizes one side to match the other reports
    metrics on a geometry no one specified.

    Unlike the five treatment-delivery gates, this check does not drive a
    loader. Pass a callable to have it execute the pipeline's own resize rule on
    the declared shape
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
