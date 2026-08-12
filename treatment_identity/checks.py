"""Seven treatment-delivery gates and one evaluation-integrity check.

The first seven checks target divergences between the treatment a pipeline
declares and delivered data.  Loader-driven checks observe loader output;
content checks can additionally observe an adapter's explicit post-collate,
pre-model boundary.  The eighth, :func:`check_geometry`, checks the separate
evaluation boundary.
Each check is written so that its failure mode yields ``FAIL`` rather than a
plausible number --- including :func:`check_separability`, where the failure is
two arms being *identical* when they were declared distinct.

All eight checks run on CPU against synthetic fixtures and complete in seconds.
"""

from __future__ import annotations

import hashlib
import inspect
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from .adapter import (AdapterError, CallRecorder, ContentContract,
                      GeneratorReached, LoaderAdapter, LoaderSpec, Sample,
                      StreamContract)
from . import fixtures as fx
from .seeding import seed_all

__all__ = [
    "CheckResult",
    "PASS", "FAIL", "SKIP", "NA", "UNDECLARED",
    "check_precomputed_input",
    "check_temporal_window",
    "check_separability",
    "check_value_range",
    "check_channel_content",
    "check_sample_value_range",
    "check_sample_channel_content",
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
    a = _as_numpy(a)
    if a.ndim == 4:
        a = a[0]
    if (a.ndim == 3 and 1 <= a.shape[0] <= 4
            and not 1 <= a.shape[-1] <= 4):
        a = np.moveaxis(a, 0, -1)
    return a


def _frames(seq) -> list[np.ndarray]:
    arr = _as_numpy(seq)
    if arr.ndim == 4:
        return [_to_hwc(arr[i]) for i in range(arr.shape[0])]
    return [_to_hwc(arr)]


def _to_255(a: np.ndarray) -> np.ndarray:
    a = _as_numpy(a).astype(np.float64, copy=False)
    if a.max() <= 1.0 + 1e-6 and a.min() >= -1e-6:
        return a * 255.0
    if a.max() <= 1.0 + 1e-6:
        return (a + 1.0) * 127.5
    return a


def _as_numpy(value: Any) -> np.ndarray:
    """Convert CPU or accelerator tensors without making Torch a dependency."""
    current = value
    detach = getattr(current, "detach", None)
    if callable(detach):
        current = detach()
    cpu = getattr(current, "cpu", None)
    if callable(cpu):
        current = cpu()
    numpy = getattr(current, "numpy", None)
    if callable(numpy):
        return np.asarray(numpy())
    return np.asarray(current)


def _bimodality(a: np.ndarray) -> float:
    """Fraction of pixels within 15 levels of pure black or pure white.

    A checkerboard scores ~1.0; any degradation of a flat mid-grey field scores
    near 0. The statistic is deliberately crude so that it cannot be gamed by a
    mild blur.
    """
    v = _to_255(a).ravel()
    return float(np.mean((v < 15) | (v > 240)))


def _stream(sample: Sample, name: str) -> Any | None:
    if name == "lq":
        return sample.lq
    if name == "gt":
        return sample.gt
    raise ValueError(f"unknown stream: {name}")


def check_sample_value_range(sample: Sample, contract: ContentContract, *,
                             boundary: str = "loader_output",
                             name: str | None = None) -> CheckResult:
    """Assert finite values, bounds and optional fixture anchors on a sample."""
    gate = name or ("value_range" if boundary == "loader_output"
                    else f"value_range[{boundary}]")
    evidence: dict[str, Any] = {"boundary": boundary, "streams": {}}
    failures: list[str] = []
    asserted = 0

    for stream_name, stream_contract in (("lq", contract.lq),
                                         ("gt", contract.gt)):
        if stream_contract is None or stream_contract.value_range is None:
            continue
        asserted += 1
        raw = _stream(sample, stream_name)
        if raw is None:
            failures.append(f"{stream_name} is absent but its range is declared")
            evidence["streams"][stream_name] = {"absent": True}
            continue
        arr = _as_numpy(raw).astype(np.float64, copy=False)
        finite = bool(np.isfinite(arr).all())
        lo, hi = stream_contract.value_range
        ev = {
            "declared": [float(lo), float(hi)],
            "atol": float(stream_contract.range_atol),
            "require_finite": stream_contract.require_finite,
            "require_extrema": stream_contract.require_range_extrema,
            "finite": finite,
            "dtype": str(_as_numpy(raw).dtype),
        }
        if arr.size and finite:
            observed_lo, observed_hi = float(arr.min()), float(arr.max())
            ev["observed"] = [observed_lo, observed_hi]
            atol = stream_contract.range_atol
            if observed_lo < lo - atol or observed_hi > hi + atol:
                failures.append(
                    f"{stream_name} range [{observed_lo:g}, {observed_hi:g}] "
                    f"exceeds declared [{lo:g}, {hi:g}]")
            if stream_contract.require_range_extrema and (
                    abs(observed_lo - lo) > atol or abs(observed_hi - hi) > atol):
                failures.append(
                    f"{stream_name} anchors [{observed_lo:g}, {observed_hi:g}] "
                    f"do not realise declared extrema [{lo:g}, {hi:g}]")
        elif not arr.size:
            failures.append(f"{stream_name} is empty")
            ev["empty"] = True
        if stream_contract.require_finite and not finite:
            failures.append(f"{stream_name} contains NaN or infinity")
        evidence["streams"][stream_name] = ev

    if not asserted:
        return CheckResult(gate, NA, "no value range is declared", evidence)
    if failures:
        return CheckResult(gate, FAIL,
                           "VALUE-RANGE CONTRADICTION: " + "; ".join(failures),
                           evidence)
    return CheckResult(
        gate, PASS,
        f"{asserted} delivered stream(s) satisfy the declared finite range"
        + (" and fixture anchors" if any(
            c is not None and c.require_range_extrema
            for c in (contract.lq, contract.gt)) else ""),
        evidence)


def check_sample_channel_content(sample: Sample, contract: ContentContract, *,
                                 boundary: str = "loader_output",
                                 name: str | None = None) -> CheckResult:
    """Assert channel count and optional non-collapsed colour signatures."""
    gate = name or ("channel_content" if boundary == "loader_output"
                    else f"channel_content[{boundary}]")
    evidence: dict[str, Any] = {"boundary": boundary, "streams": {}}
    failures: list[str] = []
    asserted = 0

    for stream_name, stream_contract in (("lq", contract.lq),
                                         ("gt", contract.gt)):
        if stream_contract is None or stream_contract.channels is None:
            continue
        asserted += 1
        raw = _stream(sample, stream_name)
        if raw is None:
            failures.append(f"{stream_name} is absent but its channels are declared")
            evidence["streams"][stream_name] = {"absent": True}
            continue
        frames = _frames(raw)
        counts: list[int] = []
        pairwise: list[float] = []
        for frame in frames:
            if frame.ndim == 2:
                frame = frame[..., None]
            count = int(frame.shape[-1]) if frame.ndim == 3 else 0
            counts.append(count)
            if stream_contract.require_distinct_channels and count >= 2:
                f64 = frame.astype(np.float64, copy=False)
                pairwise.extend(float(np.mean(np.abs(f64[..., i] - f64[..., j])))
                                for i in range(count) for j in range(i + 1, count))
        ev = {
            "declared_channels": stream_contract.channels,
            "observed_channels": counts,
            "require_distinct_channels": stream_contract.require_distinct_channels,
            "channel_atol": stream_contract.channel_atol,
        }
        if any(c != stream_contract.channels for c in counts):
            failures.append(
                f"{stream_name} delivered channel counts {counts}, declared "
                f"{stream_contract.channels}")
        if stream_contract.require_distinct_channels:
            minimum = min(pairwise) if pairwise else 0.0
            ev["minimum_pairwise_channel_mae"] = minimum
            ev["pairwise_comparisons"] = len(pairwise)
            if minimum <= stream_contract.channel_atol:
                failures.append(
                    f"{stream_name} channel signature collapsed "
                    f"(minimum pairwise MAE {minimum:g} <= "
                    f"{stream_contract.channel_atol:g})")
        evidence["streams"][stream_name] = ev

    if not asserted:
        return CheckResult(gate, NA, "no channel contract is declared", evidence)
    if failures:
        return CheckResult(gate, FAIL,
                           "CHANNEL-CONTENT CONTRADICTION: " + "; ".join(failures),
                           evidence)
    return CheckResult(
        gate, PASS,
        f"{asserted} delivered stream(s) retain the declared channel content",
        evidence)


def _sample_at_boundary(adapter: LoaderAdapter, loader: Any, index: int,
                        boundary: str) -> Sample:
    if boundary == "loader_output":
        return adapter.sample(loader, index)
    if boundary == "training_step_input":
        method = getattr(adapter, "sample_training_step", None)
        if not callable(method):
            raise NotImplementedError(
                "adapter does not expose sample_training_step; loader output "
                "cannot be relabelled as training-step input")
        return method(loader, index)
    raise ValueError(f"unknown observation boundary: {boundary}")


def _content_probe_sample(adapter: LoaderAdapter, workdir: Path,
                          contract: ContentContract, *, clip_length: int,
                          declared_clip_length: int | None, seed: int,
                          boundary: str) -> Sample:
    root = Path(workdir) / "content"
    fx.make_pair(root, gt_kind="signature", lq_kind="signature",
                 n_frames=clip_length)
    spec = LoaderSpec(
        gt_root=root / "GT", lq_root=root / "LQ",
        use_precomputed_lq=True, num_frames=min(5, clip_length), seed=seed,
        train=True, clip_length=declared_clip_length,
        fixture_clip_length=clip_length, content=contract)
    seed_all(seed)
    loader = adapter.build(spec)
    return _sample_at_boundary(adapter, loader, 0, boundary)


def check_value_range(adapter: LoaderAdapter, workdir: Path,
                      contract: ContentContract, *, clip_length: int = 16,
                      declared_clip_length: int | None = None,
                      seed: int = 0,
                      boundary: str = "loader_output") -> CheckResult:
    """Drive a content-signature fixture and assert its numerical convention."""
    try:
        sample = _content_probe_sample(
            adapter, Path(workdir) / "g6_range", contract,
            clip_length=clip_length, declared_clip_length=declared_clip_length,
            seed=seed, boundary=boundary)
    except (NotImplementedError, AdapterError) as exc:
        return CheckResult(
            "value_range" if boundary == "loader_output"
            else f"value_range[{boundary}]", SKIP, str(exc),
            {"boundary": boundary})
    return check_sample_value_range(sample, contract, boundary=boundary)


def check_channel_content(adapter: LoaderAdapter, workdir: Path,
                          contract: ContentContract, *, clip_length: int = 16,
                          declared_clip_length: int | None = None,
                          seed: int = 0,
                          boundary: str = "loader_output") -> CheckResult:
    """Drive a colour-signature fixture and reject channel collapse."""
    try:
        sample = _content_probe_sample(
            adapter, Path(workdir) / "g7_channels", contract,
            clip_length=clip_length, declared_clip_length=declared_clip_length,
            seed=seed, boundary=boundary)
    except (NotImplementedError, AdapterError) as exc:
        return CheckResult(
            "channel_content" if boundary == "loader_output"
            else f"channel_content[{boundary}]", SKIP, str(exc),
            {"boundary": boundary})
    return check_sample_channel_content(sample, contract, boundary=boundary)


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
                underlying_status=result.status,
                first_error=f"{type(first).__name__}: {first}"[:200])
            if result.status == FAIL:
                return CheckResult(
                    gate, FAIL,
                    f"the loader has an undeclared minimum source length of "
                    f"{length} frames and, once that fixture runs, the gate "
                    f"also contradicts its declaration: {result.message}",
                    result.evidence)
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
                            clip_length: int = 16,
                            declared_clip_length: int | None = None,
                            seed: int = 0) -> CheckResult:
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
                      use_precomputed_lq=True, num_frames=5, seed=seed,
                      train=True, clip_length=declared_clip_length,
                      fixture_clip_length=clip_length)
    ev: dict[str, Any] = {"declared_use_precomputed": declared,
                          "sentinel_generator": False}

    # Seed the probe series here rather than trusting whatever state the
    # subject's construction left behind.  The instrument must reproduce its
    # own observations before it can certify a subject's treatment identity.
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
                           "loader-output boundary", ev)

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
                          declared_clip_length: int | None = None,
                          num_frames: int = 5) -> CheckResult:
    """Ask one clip for several entries and read back which frames arrived.

    Fixture frame *i* carries a crop-stable index plus numerical range anchors,
    so the delivered tensor states its own provenance without an ambiguity
    between raw level one and normalised one. A loader that computes a window
    and then opens the clip prefix returns the same indices for every entry.

    Two quantities come out of this and they are not the same kind of thing. A
    loader that always returns the prefix has an exactly known reachable set:
    the probe count does not matter, because no further probe can reach a
    different frame. A loader that samples its window at random has only the
    coverage *observed* in ``n_probe`` draws under ``seed`` --- a lower bound on
    what it can reach, reproducible but not exhaustive. The result reports
    which of the two it is.
    """
    root = Path(workdir) / "g2"
    n_frames = clip_length
    fx.make_pair(root, gt_kind="index", lq_kind="index", n_frames=n_frames)
    spec = LoaderSpec(gt_root=root / "GT", lq_root=root / "LQ",
                      use_precomputed_lq=True, num_frames=num_frames,
                      seed=seed, train=True,
                      clip_length=declared_clip_length,
                      fixture_clip_length=clip_length)
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
        # Read the index off the target when there is one, off the input when
        # there is not. The fixture encodes the frame index into both streams,
        # so either answers the question, and a zero-reference loader that
        # returns no target is a subject this gate can still speak about
        # instead of raising inside a probe.
        stream = s.lq if s.gt is None else s.gt
        ids = tuple(fx.decode_index(f) for f in _frames(stream))
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

def _builder_takes_seed(builder: Callable[..., np.ndarray]) -> bool:
    try:
        parameters = inspect.signature(builder).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD,
                          p.VAR_POSITIONAL) for p in parameters)


def _build_at_seed(builder: Callable[..., np.ndarray], seed: int) -> np.ndarray:
    seed_all(seed)
    value = builder(seed) if _builder_takes_seed(builder) else builder()
    return _to_255(_as_numpy(value).astype(np.float64, copy=False))


def _distance_matrix(vectors: np.ndarray) -> np.ndarray:
    """Root-mean-square Euclidean distance, computed from one Gram matrix."""
    z = np.asarray(vectors, dtype=np.float64)
    norms = np.einsum("ij,ij->i", z, z)
    squared = np.maximum(norms[:, None] + norms[None, :] - 2.0 * z @ z.T, 0.0)
    return np.sqrt(squared / max(z.shape[1], 1))


def _energy_from_distance(distance: np.ndarray,
                          x: np.ndarray, y: np.ndarray) -> float:
    return float(2.0 * distance[np.ix_(x, y)].mean()
                 - distance[np.ix_(x, x)].mean()
                 - distance[np.ix_(y, y)].mean())


def _paired_energy_permutation(a: list[np.ndarray], b: list[np.ndarray], *,
                               permutations: int,
                               random_seed: int) -> tuple[float, float, int]:
    """Paired randomisation test of equality of the two output distributions."""
    n = len(a)
    vectors = np.stack([x.reshape(-1) for x in (*a, *b)])
    distance = _distance_matrix(vectors)
    x0, y0 = np.arange(n), np.arange(n, 2 * n)
    observed = _energy_from_distance(distance, x0, y0)
    rng = np.random.default_rng(random_seed)
    exceed = 0
    for _ in range(permutations):
        swap = rng.integers(0, 2, size=n, dtype=np.int8).astype(bool)
        x = np.where(swap, y0, x0)
        y = np.where(swap, x0, y0)
        exceed += int(_energy_from_distance(distance, x, y) >= observed - 1e-15)
    p_value = (exceed + 1.0) / (permutations + 1.0)
    return observed, float(p_value), exceed


def check_separability(build_a: Callable[..., np.ndarray],
                       build_b: Callable[..., np.ndarray],
                       *, declared: str = "distinct",
                       tol: float = 1e-6,
                       label: str = "A_vs_B",
                       seeds: Sequence[int] | None = None,
                       distributional: bool = False,
                       alpha: float = 0.05,
                       min_effect: float = 0.0,
                       permutations: int = 4095,
                       permutation_seed: int = 0) -> CheckResult:
    """Compare two treatments under one or several matched seeds.

    Exact unexpected equality remains the strongest discriminator and is never
    replaced by a p-value.  When ``distributional=True``, the builders are run
    under several matched seeds and a paired energy-distance randomisation test
    is added.  A non-significant result is reported as failure to demonstrate
    the *declared operational distinction under this fixed design*, not as
    proof that the population distributions are equal.

    Builders may accept the seed as one positional argument or accept no
    arguments and draw from the global RNGs seeded by the gate.
    """
    if declared not in {"distinct", "identical"}:
        raise ValueError("declared must be 'distinct' or 'identical'")
    draws = tuple(int(s) for s in (seeds if seeds is not None else (0,)))
    if not draws:
        raise ValueError("at least one separability seed is required")
    if distributional and len(draws) < 4:
        raise ValueError("distributional separability requires at least four seeds")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")
    if permutations < 99:
        raise ValueError("at least 99 permutations are required")

    samples_a: list[np.ndarray] = []
    samples_b: list[np.ndarray] = []
    per_draw: list[dict[str, Any]] = []
    shapes_match = True
    finite = True
    for seed in draws:
        a = _build_at_seed(build_a, seed)
        b = _build_at_seed(build_b, seed)
        samples_a.append(a)
        samples_b.append(b)
        finite = finite and bool(np.isfinite(a).all() and np.isfinite(b).all())
        if a.shape != b.shape:
            shapes_match = False
            per_draw.append({"seed": seed, "shape_a": list(a.shape),
                             "shape_b": list(b.shape)})
            continue
        difference = np.abs(a - b)
        per_draw.append({
            "seed": seed,
            "max_abs_delta_0_255": float(difference.max()),
            "mae_0_255": float(difference.mean()),
        })

    gate = f"separability[{label}]"
    ev: dict[str, Any] = {
        "declared": declared,
        "tolerance": tol,
        "seeds": list(draws),
        "draws": per_draw,
        "distributional_test": distributional,
    }
    if not finite:
        return CheckResult(
            gate, FAIL,
            "SEPARABILITY UNDEFINED: at least one treatment delivered NaN or "
            "infinity under the fixed probe", ev)
    if not shapes_match:
        if declared == "distinct":
            return CheckResult(gate, PASS,
                               "treatments differ in tensor shape under at "
                               "least one matched-seed draw", ev)
        return CheckResult(gate, FAIL,
                           "treatments declared identical differ in tensor shape", ev)

    deltas = [d["max_abs_delta_0_255"] for d in per_draw]
    maes = [d["mae_0_255"] for d in per_draw]
    ev.update(
        max_abs_delta_0_255=max(deltas),
        mean_paired_mae_0_255=float(np.mean(maes)),
        draws_distinguishable=sum(delta > tol for delta in deltas),
    )

    if declared == "identical":
        if any(delta > tol for delta in deltas):
            return CheckResult(
                gate, FAIL,
                f"treatments declared identical differ in "
                f"{sum(delta > tol for delta in deltas)}/{len(draws)} matched draws",
                ev)
        return CheckResult(
            gate, PASS,
            f"treatments declared identical agree within {tol:g} in all "
            f"{len(draws)} matched draws", ev)

    if all(delta <= tol for delta in deltas):
        return CheckResult(
            gate, FAIL,
            "UNEXPECTED EQUALITY: two treatments declared distinct produced "
            f"tensors identical to within {tol:g} in all {len(draws)} "
            "matched-seed draws. The declared distinction is not observable "
            "under this probe; verify treatment identity before interpreting "
            "a comparison.", ev)

    if not distributional:
        return CheckResult(
            gate, PASS,
            f"treatments are distinguishable in "
            f"{sum(delta > tol for delta in deltas)}/{len(draws)} matched draws "
            f"(maximum delta {max(deltas):.4g})", ev)

    flattened_sizes = {sample.size for sample in (*samples_a, *samples_b)}
    if len(flattened_sizes) != 1:
        ev["flattened_sizes"] = sorted(flattened_sizes)
        return CheckResult(
            gate, FAIL,
            "DISTRIBUTIONAL SEPARABILITY NOT EVALUABLE: delivered tensor size "
            "varies across seeds, contradicting the fixed statistical design",
            ev)

    effect, p_value, exceed = _paired_energy_permutation(
        samples_a, samples_b, permutations=permutations,
        random_seed=permutation_seed)
    ev["energy_test"] = {
        "statistic_rms_0_255": effect,
        "p_value": p_value,
        "alpha": alpha,
        "minimum_effect": min_effect,
        "permutations": permutations,
        "permuted_statistics_ge_observed": exceed,
        "paired_exchangeability_assumption": True,
    }
    if p_value <= alpha and effect > min_effect:
        return CheckResult(
            gate, PASS,
            f"{len(draws)} matched draws distinguish the treatments and the "
            f"paired energy test rejects equality (effect {effect:.4g}, "
            f"p={p_value:.4g}, alpha={alpha:g})", ev)
    return CheckResult(
        gate, FAIL,
        "DISTRIBUTIONAL SEPARABILITY NOT ESTABLISHED: the fixed multiseed "
        f"probe did not meet its declared evidence threshold (effect "
        f"{effect:.4g}, p={p_value:.4g}, alpha={alpha:g}, minimum effect "
        f"{min_effect:g}). This is not proof that the population distributions "
        "are equal.", ev)


# --------------------------------------------------------------------------
# Gate 4 — how many times is the target transformed?
# --------------------------------------------------------------------------

def check_target_transforms(adapter: LoaderAdapter, workdir: Path,
                            recorder: CallRecorder, transform_name: str,
                            expected: int, declared: bool = True,
                            clip_length: int = 8,
                            declared_clip_length: int | None = None,
                            seed: int = 0,
                            has_target: bool = True) -> CheckResult:
    """Count target-side transformations against the number the *paper* declares.

    ``expected`` is read off the publication, not off the code: it is the claim
    under test. Passing the observed count as ``expected`` would make the gate
    assert the code against itself, and it could never fail.

    ``declared`` says whether the publication addresses this transformation at
    all. It separates two distinct findings:

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

    ``has_target=False`` declares a zero-reference arm, and the gate answers
    ``N/A`` without building anything. Applicability is asked before
    declaredness, so a target-less sample cannot pass against a fabricated zero
    count. A property that does not apply withholds nothing and asserts nothing,
    which is what ``N/A`` means.
    """
    if not has_target:
        return CheckResult(
            "target_transforms", NA,
            "the arm declares no learning target (zero-reference): there is no "
            "target for this gate to assert over, and a count of zero "
            "transformations of a target that does not exist is not a pass")
    root = Path(workdir) / "g4"
    fx.make_pair(root, gt_kind="flat", lq_kind="checker", n_frames=clip_length)
    spec = LoaderSpec(gt_root=root / "GT", lq_root=root / "LQ",
                      use_precomputed_lq=False, num_frames=3, seed=seed,
                      train=True, target_transforms=expected,
                      has_target=has_target,
                      clip_length=declared_clip_length,
                      fixture_clip_length=clip_length)
    seed_all(spec.seed)
    recorder.reset()
    try:
        loader = adapter.build(spec)
        sample = adapter.sample(loader, 0)
    except NotImplementedError as e:
        return CheckResult("target_transforms", SKIP, f"adapter cannot build: {e}")

    # Second line of defence, and a different question from the one above. The
    # contract says a target exists; the adapter delivered none. This is not
    # inapplicability: an explicit precondition has been contradicted, so the
    # gate must fail before attempting to count target-side operations.
    if sample.gt is None:
        return CheckResult(
            "target_transforms", FAIL,
            "the contract declares a learning target but the loader delivered "
            "none; target presence is a contradicted precondition",
            {"declared_has_target": True, "delivered_has_target": False})

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
                         clip_length: int = 8,
                         declared_clip_length: int | None = None,
                         seed: int = 0) -> CheckResult:
    """Draw repeatedly and compare the realised operator order to the declared policy.

    A permutation that is computed and then discarded leaves exactly one
    observed order across every draw.
    """
    root = Path(workdir) / "g5"
    fx.make_pair(root, gt_kind="flat", lq_kind="checker", n_frames=clip_length)
    spec = LoaderSpec(gt_root=root / "GT", lq_root=root / "LQ",
                      use_precomputed_lq=False, num_frames=3, seed=seed,
                      train=True, operator_order=declared_policy,
                      clip_length=declared_clip_length,
                      fixture_clip_length=clip_length)
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

    Unlike the seven treatment-delivery gates, this check does not drive a
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
    "value_range",
    "channel_content",
    "geometry",
)
