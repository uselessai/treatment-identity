"""Eight delivery predicates and one complementary geometry check.

Checks report bounded observations against explicit expectations. Distributional
separability can remain unassessed when assumptions or evidence are insufficient.
The historical campaign did not exercise every predicate in this revision.
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


def _to_255(a: np.ndarray,
            value_range: tuple[float, float] | None = None) -> np.ndarray:
    """Map a stream to 0..255 levels.

    With a declared ``value_range`` the mapping follows the declaration. Without
    one it is guessed from the observed extrema, and the guess is wrong for a
    [-1, 1] stream whose values happen to be non-negative: a flat mid-grey,
    0.0039 in [-1, 1], becomes level 1 under the [0, 1] reading.
    """
    a = _as_numpy(a).astype(np.float64, copy=False)
    if value_range is not None:
        lo, hi = (float(v) for v in value_range)
        return (a - lo) * (255.0 / (hi - lo))
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


def _bimodality(a: np.ndarray,
                value_range: tuple[float, float] | None = None) -> float:
    """Fraction of pixels within 15 levels of pure black or pure white.

    A checkerboard scores ~1.0; any degradation of a flat mid-grey field scores
    near 0. The statistic is deliberately crude so that it cannot be gamed by a
    mild blur.

    Both anchors must be present, each on at least a fifth of the pixels (a
    checkerboard has about half of each); the score is then the usual sum.
    Counting either anchor alone scored an inpainting hole (pure black) or a
    misread mid-grey as the checkerboard.
    """
    v = _to_255(a, value_range).ravel()
    black, white = float(np.mean(v < 15)), float(np.mean(v > 240))
    return black + white if min(black, white) >= 0.2 else 0.0


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
        return CheckResult(gate, UNDECLARED, "no value range is declared", evidence)
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
        return CheckResult(gate, UNDECLARED, "no channel contract is declared", evidence)
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
                            seed: int = 0,
                            value_range: tuple[float, float] | None = None
                            ) -> CheckResult:
    """Deliver a checkerboard as the pre-computed input over a flat target.

    ``value_range`` is the input stream's declared numeric convention; when
    given, the delivered values are read in it instead of a guessed one.

    Two independent discriminators, in order of strength:

    1. The online generator is replaced by a sentinel that raises
       :class:`GeneratorReached`. A loader honouring its pre-computed-input flag
       never reaches it; one ignoring the flag raises, and the exception is the
       finding rather than a plausible tensor. Adapters that cannot patch their
       generator skip this step.
    2. The delivered tensor is inspected: the checkerboard is bimodal, and no
       degradation of a flat mid-grey target can be.

    ``declared=False`` is an explicit contract that the feature is inapplicable,
    not an inference from documentary silence; its absence is ``N/A``: only a configuration that sets
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
    score = float(np.mean([_bimodality(f, value_range) for f in lq]))
    ev["bimodality"] = round(score, 4)
    ev["value_range"] = list(value_range) if value_range is not None else "inferred"
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
            if i == 0:
                raise          # empty loader: let the clip escalation see it
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
          "coverage_is_exhaustive": False}

    # An index the decoder could not read is not an index. A loader that
    # interpolates the fixture (a fixed-size resize, say) blurs the encoding
    # away, and a tuple of Nones is then the same "window" on every draw: the
    # gate used to read that as an inert sampler and fail a loader whose window
    # moves. Nothing can be asserted about which frames arrived when no frame
    # says which it is.
    unreadable = sum(i is None for w in observed for i in w)
    if unreadable:
        ev["unreadable_frames"] = unreadable
        return CheckResult(
            "temporal_window", SKIP,
            f"{unreadable} of {sum(len(w) for w in observed)} delivered frames "
            "carry no readable index (the loader's interpolation erased the "
            "fixture's encoding); which frames arrived cannot be stated", ev)

    # A loader that discards its window returns the prefix every time.
    prefix = tuple(range(len(observed[0])))
    if len(unique_windows) == 1 and observed[0] == prefix:
        return CheckResult(
            "temporal_window", FAIL,
            f"every one of the {len(observed)} entries returned the clip prefix "
            f"{list(prefix)}; no window variation was observed. "
            f"Observed {len(unique_frames)}/{n_frames} frames "
            f"({coverage:.1%}) under this probe; source inspection is required for an exhaustive claim.", ev)

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


def _build_at_seed(builder: Callable[..., np.ndarray], seed: int,
                   value_range: tuple[float, float]) -> np.ndarray:
    seed_all(seed)
    value = builder(seed) if _builder_takes_seed(builder) else builder()
    lo, hi = value_range
    return (_as_numpy(value).astype(np.float64, copy=False) - lo) * (255.0 / (hi - lo))


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
                       *, declared: str | None = "distinct",
                       value_range_a: tuple[float, float] = (0.0, 255.0),
                       value_range_b: tuple[float, float] = (0.0, 255.0),
                       exchangeability_justification: str | None = None,
                       tol: float = 1e-6,
                       label: str = "A_vs_B",
                       seeds: Sequence[int] | None = None,
                       distributional: bool = False,
                       alpha: float = 0.05,
                       min_effect: float = 0.0,
                       permutations: int = 4095,
                       permutation_seed: int = 0) -> CheckResult:
    """Compare observed arrays against an explicit bounded contract.

    Units default to raw 0--255 and never depend on observed extrema. Supply
    each stream's declared range to convert other conventions. ``distinct``
    without distributional inference requires a difference on this probe;
    ``identical`` requires agreement on every draw. ``None`` records UNDECL.
    Distributional inference additionally requires an explicit justification
    of within-pair exchangeability. Missing assumptions, insufficient evidence,
    or an unevaluable design produce SKIP, not a demonstrated contradiction.
    """
    if declared not in {None, "distinct", "identical"}:
        raise ValueError("declared must be None, 'distinct' or 'identical'")
    for bounds in (value_range_a, value_range_b):
        if len(bounds) != 2 or not np.isfinite(bounds).all() or bounds[0] >= bounds[1]:
            raise ValueError("value ranges must contain finite increasing bounds")
    if tol < 0 or min_effect < 0:
        raise ValueError("tolerance and minimum effect must be non-negative")
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
        a = _build_at_seed(build_a, seed, value_range_a)
        b = _build_at_seed(build_b, seed, value_range_b)
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
        "value_range_a": list(value_range_a),
        "value_range_b": list(value_range_b),
        "exchangeability_justification": exchangeability_justification,
    }
    if not finite:
        return CheckResult(
            gate, FAIL,
            "SEPARABILITY UNDEFINED: at least one treatment delivered NaN or "
            "infinity under the fixed probe", ev)
    if not shapes_match:
        if declared is None:
            return CheckResult(gate, UNDECLARED, "different shapes observed; no comparison contract", ev)
        if distributional and declared == "distinct":
            return CheckResult(gate, SKIP, "shape differences observed; fixed-shape distributional test not evaluated", ev)
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

    if declared is None:
        return CheckResult(gate, UNDECLARED,
                           "paired outputs observed; no requirement of identity or distinction was supplied", ev)

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

    if all(delta <= tol for delta in deltas) and distributional:
        return CheckResult(gate, SKIP,
                           "equal outputs on this finite probe do not contradict a population distinction", ev)

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
            gate, SKIP,
            "DISTRIBUTIONAL SEPARABILITY NOT EVALUABLE: delivered tensor size "
            "varies across seeds, contradicting the fixed statistical design",
            ev)

    if not exchangeability_justification or not exchangeability_justification.strip():
        return CheckResult(gate, SKIP,
                           "paired exchangeability is not justified; common seeds alone are insufficient", ev)

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
        gate, SKIP,
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

    if not declared:
        return CheckResult(
            "target_transforms", UNDECLARED,
            f"observed {transform_name} {per_frame:g}x per target frame; no expected count was declared in the inspected contract",
            ev)

    if abs(per_frame - expected) < 1e-9:
        return CheckResult("target_transforms", PASS,
                           f"{transform_name} applied {expected}x per frame, as declared", ev)
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
            # On the first draw an empty loader is a clip too short for the
            # subject, which the clip-length escalation must see; reporting it
            # as "no operator calls observed" hid an undeclared minimum length.
            if i == 0:
                raise
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
          "orders_observed": sorted("->".join(o) for o in distinct),
          "decision_scope": "fixed-probe order variation, not distributional randomness",
          "minimum_distinct_orders": 2 if declared_policy == "random_permutation" else 1}

    if declared_policy == "random_permutation":
        if len(distinct) == 1:
            return CheckResult(
                "operator_trace", FAIL,
                f"policy declared '{declared_policy}' but all {len(orders)} draws "
                f"executed the same order: {'->'.join(next(iter(distinct)))}. "
                "This establishes no observed order variation; inspect source to determine the cause.", ev)
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

    Unlike the eight treatment-delivery gates, this check does not drive a
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
          "aspect_error": round(rel, 5), "shape_source": shape_source,
          "scale_ratio_y": dh / eh, "scale_ratio_x": dw / ew}

    if (dh, dw) == (eh, ew):
        return CheckResult("geometry", PASS, f"delivered {dh}x{dw}, as declared", ev)
    if rel <= aspect_tol:
        return CheckResult("geometry", PASS,
                           f"delivered {dh}x{dw} preserves the declared aspect ratio", ev)
    return CheckResult(
        "geometry", FAIL,
        f"delivered {dh}x{dw} distorts the declared {eh}x{ew} aspect ratio by "
        f"{rel:.2%}; metrics computed here are not on the declared geometry", ev)


# --------------------------------------------------------------------------
# Gate 9 — was the content resized, or was it cropped?
# --------------------------------------------------------------------------

#: A period is only believable if the delivered tile is several periods wide
#: and each period spans enough pixels to survive sampling. Without both, a
#: small crop of a coarse pattern returns a confident number that means nothing,
#: which is how this gate first reported four subjects as distorted when they
#: had merely cropped.
MIN_PERIOD_PX = 6.0
MIN_CYCLES = 4.0


def _dominant_period(profile) -> float | None:
    """Spatial period of the strongest non-constant component of a profile.

    Returns ``None`` when the profile cannot support a reading, rather than the
    best available guess.
    """
    x = np.asarray(profile, dtype=np.float64)
    x = x - x.mean()
    if not np.any(np.abs(x) > 1e-9):
        return None
    spectrum = np.abs(np.fft.rfft(x))
    spectrum[0] = 0.0
    k = int(np.argmax(spectrum))
    if k == 0:
        return None
    # A 32-pixel window quantises the bin index coarsely, and the rounding
    # differs between two axes of different period, which manufactures an
    # anisotropy that is not there. Interpolating the peak parabolically
    # recovers a sub-bin estimate and removes that artefact.
    if 0 < k < len(spectrum) - 1:
        a, b, c = spectrum[k - 1], spectrum[k], spectrum[k + 1]
        denom = a - 2.0 * b + c
        if abs(denom) > 1e-12:
            k = k + 0.5 * (a - c) / denom
    if k <= 0:
        return None
    period = len(x) / k
    if period < MIN_PERIOD_PX or (len(x) / period) < MIN_CYCLES:
        return None
    return period


def _grey_tile(delivered) -> np.ndarray:
    a = np.asarray(delivered, dtype=np.float64)
    while a.ndim > 3:
        a = a[0]
    if a.ndim == 3:
        if a.shape[0] in (1, 2, 3) and a.shape[2] not in (1, 2, 3):
            a = np.transpose(a, (1, 2, 0))
        a = a.mean(axis=2)
    return a


def measure_content_scale(delivered) -> tuple[float | None, float | None]:
    """Per-axis spatial period of a delivered tile, in delivered pixels."""
    a = _grey_tile(delivered)
    return _dominant_period(a.mean(axis=0)), _dominant_period(a.mean(axis=1))


#: A rotation smaller than this is not reported; it is within what the
#: frequency peaks of a small tile can resolve.
ROTATION_TOL_DEG = 5.0


def measure_content_orientation(delivered, period_x: float, period_y: float
                                ) -> dict | None:
    """Rotation and per-axis scale of the scale-grid pattern, at any angle.

    The axis profiles of :func:`measure_content_scale` assume the pattern
    arrives upright. A loader that rotates the picture breaks that: a quarter
    turn swaps the two periods and reads as a reshape, and any other angle
    smears both profiles. The two-dimensional spectrum does not assume an
    orientation. The fixture has one frequency along each source axis, so the
    delivered spectrum has two peaks; their directions give the rotation and
    their lengths the scale along each rotated source axis.

    Peaks are assigned to the source axes so that the implied scaling is the
    most even, so a rotation combined with an anisotropy larger than the ratio
    of the two periods cannot be resolved. Flips do not show: the pattern is
    mirror-symmetric. The angle, clockwise in image coordinates, is folded into
    (-90, 90]. Returns ``None``
    when two non-collinear peaks cannot be read.
    """
    a = _grey_tile(delivered)
    if a.ndim != 2 or min(a.shape) < 2 * MIN_PERIOD_PX:
        return None
    h, w = a.shape
    a = a - a.mean()
    spectrum = np.abs(np.fft.fft2(a * np.outer(np.hanning(h), np.hanning(w))))
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    radius = np.hypot(fx, fy)
    usable = (radius >= MIN_CYCLES / min(h, w)) & (radius <= 1.0 / MIN_PERIOD_PX)
    work = np.where(usable, spectrum, 0.0)

    def refine(iy: int, ix: int) -> np.ndarray:
        def offset(m, c, p):
            d = m - 2.0 * c + p
            return 0.5 * (m - p) / d if abs(d) > 1e-12 else 0.0
        c = spectrum[iy, ix]
        dy = offset(spectrum[(iy - 1) % h, ix], c, spectrum[(iy + 1) % h, ix])
        dx = offset(spectrum[iy, (ix - 1) % w], c, spectrum[iy, (ix + 1) % w])
        return np.array([np.fft.fftfreq(w)[ix] + dx / w,
                         np.fft.fftfreq(h)[iy] + dy / h])

    i1 = np.unravel_index(int(np.argmax(work)), work.shape)
    if work[i1] <= 0:
        return None
    f1 = refine(*i1)
    # Drop everything collinear with the first peak: its mirror and its
    # harmonics, of which a square-wave pattern has plenty.
    cosine = np.abs(fx * f1[0] + fy * f1[1]) / (np.maximum(radius, 1e-12)
                                                 * np.hypot(*f1))
    work = np.where(cosine < np.cos(np.radians(20.0)), work, 0.0)
    i2 = np.unravel_index(int(np.argmax(work)), work.shape)
    if work[i2] < 0.2 * spectrum[i1]:
        return None
    f2 = refine(*i2)

    best = None
    for fa, fb in ((f1, f2), (f2, f1)):
        sx = 1.0 / (np.hypot(*fa) * float(period_x))
        sy = 1.0 / (np.hypot(*fb) * float(period_y))
        spread = abs(np.log(sx / sy))
        if best is None or spread < best[0]:
            best = (spread, fa, sx, sy)
    _, fa, sx, sy = best
    # Image rows grow downwards, so a positive angle is a clockwise turn.
    angle = round(float(np.degrees(np.arctan2(fa[1], fa[0]))), 2)
    if angle > 90.0:
        angle -= 180.0
    elif angle <= -90.0:
        angle += 180.0
    return {"rotation_deg": angle, "scale_x": float(sx), "scale_y": float(sy)}


def check_content_scale(probes, *, declared: str | None = None,
                        tol: float = 0.06) -> CheckResult:
    r"""Report what the loader did to the picture, not to the tensor.

    ``geometry`` asks whether the delivered tensor has the declared shape. A
    loader can satisfy that and still have squashed a wide picture into a
    square, because a crop to 256x256 and a resize to 256x256 give the same
    shape and a different picture.

    Cropping and rescaling are both ordinary things for a loader to do, so this
    gate classifies rather than judges. It reports which of three happened ---

    * ``crop``: the content arrives at its original scale;
    * ``rescale``: the content is scaled and keeps its proportions;
    * ``reshape``: the two axes are scaled differently, so the proportions of
      the picture come from the loader rather than from the data ---

    and turns that into a verdict only against something the subject actually
    declared. With no declaration the finding is \texttt{UNDECL}: the code does
    it and nothing says so.

    ``probes`` is a sequence of ``(delivered, period_x, period_y, source_aspect)``
    taken at *different* source aspects, because the size of a reshaping is not
    a property of the loader: forcing every input to a square changes a 16:9
    picture by 43.8\% and a square one by nothing. What belongs to the loader is
    whether the delivered proportions follow the source or ignore it, and that
    only shows when the source aspect changes.
    """
    readings = []
    for delivered, px, py, src_aspect in probes:
        # Orientation first: a rotated pattern gives confident and wrong axis
        # periods (a quarter turn swaps them), so when the spectrum shows a
        # rotation the scales come from it. An upright delivery keeps the axis
        # reading, so nothing measured on an upright loader changes.
        orient = measure_content_orientation(delivered, px, py)
        rotation = None
        if orient is not None and abs(orient["rotation_deg"]) > ROTATION_TOL_DEG:
            rotation = orient["rotation_deg"]
            sx, sy = orient["scale_x"], orient["scale_y"]
        else:
            mx, my = measure_content_scale(delivered)
            if mx is None or my is None:
                continue
            sx, sy = mx / float(px), my / float(py)
        reading = {"source_aspect": round(float(src_aspect), 4),
                   "scale_x": round(sx, 4), "scale_y": round(sy, 4),
                   "delivered_aspect": round(float(src_aspect) * sx / sy, 4),
                   "anisotropy": round(abs(sx - sy) / max(sx, sy), 4)}
        if rotation is not None:
            reading["rotation_deg"] = rotation
        readings.append(reading)

    if not readings:
        return CheckResult(
            "content_scale", SKIP,
            "no probe delivered a tile large enough, or a pattern fine enough, "
            "for a period to be read on both axes; nothing is claimed rather "
            "than something the measurement cannot support", {"probes": 0})

    worst = max(readings, key=lambda r: r["anisotropy"])
    scale = float(np.mean([r["scale_x"] for r in readings]))
    if worst["anisotropy"] > tol:
        behaviour = "reshape"
    elif abs(scale - 1.0) > tol:
        behaviour = "rescale"
    else:
        behaviour = "crop"

    ev = {"behaviour": behaviour, "readings": readings,
          "max_anisotropy": worst["anisotropy"],
          "at_source_aspect": worst["source_aspect"],
          "mean_scale": round(scale, 4), "declared": declared,
          "tolerance": tol, "probes": len(readings)}

    if behaviour == "reshape" and len(readings) >= 2:
        aspects = [r["delivered_aspect"] for r in readings]
        spread = (max(aspects) - min(aspects)) / max(abs(max(aspects)), 1e-9)
        ev["delivered_aspect_spread"] = round(spread, 4)
        ev["fixed_output_aspect"] = bool(spread <= tol)

    described = {
        "crop": "the content arrives at its original scale: the loader crops it or passes it through whole",
        "rescale": (f"the content is rescaled by {scale:.3f} and keeps its "
                    "proportions"),
        "reshape": (f"the loader reshapes the picture: its two axes are scaled "
                    f"differently, by {worst['anisotropy']:.1%} at source aspect "
                    f"{worst['source_aspect']:.3f}"
                    + (", and the delivered proportions are the same whatever "
                       "the source is, so they come from the loader and not "
                       "from the data"
                       if ev.get("fixed_output_aspect") else "")),
    }[behaviour]

    # Rotation is reported beside the scale class, not instead of it: a loader
    # that crops and turns the picture still crops.
    rotated = [r["rotation_deg"] for r in readings if "rotation_deg" in r]
    if rotated:
        ev["rotated_probes"] = len(rotated)
        ev["rotation_deg"] = rotated
        angles = ", ".join(f"{a:g}" for a in rotated)
        described += (f"; {len(rotated)} of {len(readings)} probes arrive "
                      f"rotated (by {angles} degrees)")

    if declared is None:
        return CheckResult("content_scale", UNDECLARED,
                           described + "; no declaration of this was found", ev)

    if declared == behaviour:
        return CheckResult("content_scale", PASS,
                           described + f", as declared ({declared})", ev)
    return CheckResult("content_scale", FAIL,
                       described + f", and {declared} was declared", ev)


def check_content_scale_by_item(probes_by_item, *, declared: str | None = None,
                                tol: float = 0.06) -> CheckResult:
    """Whether the loader does the same thing to every item it delivers.

    ``check_content_scale`` reads one delivery per source shape and names what
    the loader did to the picture. That is the right question only while the
    loader does one thing to every item. A loader that crops what is already
    large enough and resizes what is not does two, and a single reading reports
    whichever of them the observed item happened to receive.

    ``probes_by_item`` maps an item key --- whatever the caller used to ask for
    it, an index or a name --- to that item's own sequence of
    ``(delivered, period_x, period_y, source_aspect)`` probes. Every item is
    classified on its own terms by the same rule, and this gate reports the set
    of behaviours rather than one of them. A single behaviour across items is
    that behaviour; more than one is ``mixed``. The evidence keeps which item
    fell where, so a split can be inspected instead of trusted.

    The reach of the result is the items actually asked for: this says that the
    observed items were treated alike, not that every item the dataset can
    produce would be. Items whose pattern could not be read are counted and
    excluded rather than assumed to agree.
    """
    per_item: dict = {}
    unreadable: list = []
    for key, probes in dict(probes_by_item).items():
        res = check_content_scale(probes, declared=None, tol=tol)
        behaviour = res.evidence.get("behaviour") if res.evidence else None
        if behaviour is None:
            unreadable.append(key)
            continue
        per_item[key] = behaviour

    if not per_item:
        return CheckResult(
            "content_scale_by_item", SKIP,
            "no item delivered a tile large enough, or a pattern fine enough, "
            "for a period to be read on both axes; nothing is claimed rather "
            "than something the measurement cannot support",
            {"items_read": 0, "items_unreadable": len(unreadable)})

    counts: dict = {}
    for behaviour in per_item.values():
        counts[behaviour] = counts.get(behaviour, 0) + 1
    distinct = sorted(counts)
    behaviour = distinct[0] if len(distinct) == 1 else "mixed"

    ev = {"behaviour": behaviour, "per_item": per_item, "counts": counts,
          "items_read": len(per_item), "items_unreadable": len(unreadable),
          "unreadable_items": unreadable, "declared": declared,
          "tolerance": tol}

    if behaviour == "mixed":
        tally = ", ".join(f"{counts[b]} {b}" for b in distinct)
        described = (f"the loader does not treat every item alike: {tally} "
                     f"over {len(per_item)} items read")
    else:
        described = (f"every one of the {len(per_item)} items read was "
                     f"delivered the same way: {behaviour}")

    if declared is None:
        return CheckResult("content_scale_by_item", UNDECLARED,
                           described + "; no declaration of this was found", ev)
    if declared == behaviour:
        return CheckResult("content_scale_by_item", PASS,
                           described + f", as declared ({declared})", ev)
    return CheckResult("content_scale_by_item", FAIL,
                       described + f", and {declared} was declared", ev)


ALL_CHECKS = (
    "precomputed_input",
    "temporal_window",
    "separability",
    "target_transforms",
    "operator_trace",
    "value_range",
    "channel_content",
    "content_scale",
    "geometry",
)
