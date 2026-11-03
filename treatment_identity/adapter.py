"""The interface a pipeline must expose to be audited.

The protocol is deliberately small: everything the checks need is a way to
build a loader over a fixture and a way to pull one sample out of it. Nothing
in this module knows about film restoration, or about the study that produced
it.
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

__all__ = [
    "Sample", "StreamContract", "ContentContract", "LoaderSpec",
    "LoaderAdapter", "AdapterError", "GeneratorReached",
]


class AdapterError(RuntimeError):
    """Raised when an adapter cannot honour a requested configuration."""


class GeneratorReached(RuntimeError):
    """Raised by the sentinel generator installed by the pre-computed-input gate.

    The gate replaces the online generator with a function that raises this.
    A loader that honours its pre-computed-input flag never calls it and returns
    the fixture untouched; a loader that ignores the flag reaches it, and the
    exception --- rather than a plausible tensor --- is the finding.
    """


@dataclass
class Sample:
    """One item observed at a declared loader or training-step boundary.

    ``lq`` and ``gt`` are sequences of frames, each ``(H, W, C)`` or ``(C, H, W)``;
    the checks normalise orientation themselves. ``frame_ids`` is the loader's
    own claim about which frames it selected, when it exposes one --- the
    temporal check compares that claim against the delivered content. This
    object does not attest that an optimiser subsequently consumed the sample.

    ``gt`` may be ``None``, and the reason is a defect this type used to cause.
    A zero-reference loader --- one that trains on inputs with no ground truth
    at all --- has no target to return, and with ``gt`` mandatory the only way
    to adapt one was to report its single delivered tensor as both input and
    target. The target-dependent gates then compared an observation against a
    declaration and passed, on a subject that has nothing for them to assert
    over: a pass produced by the absence of the thing being checked. A type
    that cannot say "there is no target" makes an adapter say something else
    instead, and what it says looks like agreement. ``None`` is that sentence,
    and the gates that need a target answer ``N/A`` when they see it.
    """

    lq: np.ndarray | list[np.ndarray]
    gt: np.ndarray | list[np.ndarray] | None
    frame_ids: list[int] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def has_target(self) -> bool:
        """Whether this sample carries a target at all."""
        return self.gt is not None

    def as_arrays(self) -> tuple[np.ndarray, np.ndarray | None]:
        gt = None if self.gt is None else np.asarray(self.gt)
        return np.asarray(self.lq), gt


@dataclass(frozen=True)
class StreamContract:
    """Content properties one delivered tensor stream must satisfy.

    ``value_range`` declares the numerical convention at the observation
    boundary, for example ``(0.0, 1.0)`` or ``(-1.0, 1.0)``.  Bounds are always
    asserted.  ``require_range_extrema`` additionally requires a purpose-built
    fixture containing both anchors to arrive with those extrema; this is what
    distinguishes ``[0, 1]`` from a silently rescaled ``[0, 255]`` stream even
    though the former is contained in the latter.

    ``channels`` asserts the channel count.  ``require_distinct_channels`` is
    for colour contracts: a fixture with a different signature in every
    channel must retain at least ``channel_atol`` mean absolute separation for
    every pair.  It detects replication and colour-to-luminance collapse; it is
    not a general colour-science oracle.
    """

    value_range: tuple[float, float] | None = None
    range_atol: float = 1e-5
    require_range_extrema: bool = False
    require_finite: bool = True
    channels: int | None = None
    require_distinct_channels: bool = False
    channel_atol: float = 1e-5

    def __post_init__(self) -> None:
        if self.value_range is not None:
            lo, hi = self.value_range
            if not np.isfinite([lo, hi]).all() or lo >= hi:
                raise ValueError("value_range must contain two finite increasing bounds")
        if self.range_atol < 0 or self.channel_atol < 0:
            raise ValueError("content-contract tolerances must be non-negative")
        if self.channels is not None and self.channels < 1:
            raise ValueError("channels must be positive when declared")
        if self.require_distinct_channels and (self.channels or 0) < 2:
            raise ValueError("distinct-channel content requires at least two channels")


@dataclass(frozen=True)
class ContentContract:
    """Declared content semantics for the input and optional target streams."""

    lq: StreamContract
    gt: StreamContract | None = None


@dataclass
class LoaderSpec:
    """What the caller declares the loader should do.

    The point of the protocol is that every field here is a *claim*, and each
    check tries to falsify one of them against the delivered tensor.
    """

    gt_root: Path
    lq_root: Path | None = None
    use_precomputed_lq: bool = False
    num_frames: int = 5
    seed: int = 0
    train: bool = True
    #: Declared operator sampling policy: "fixed" or "random_permutation".
    operator_order: str = "fixed"
    #: How many times the target is expected to be transformed (e.g. sharpened).
    target_transforms: int = 0
    #: Whether this arm has a learning target at all. ``False`` declares a
    #: zero-reference task -- one trained on inputs with no ground truth -- and
    #: it is an APPLICABILITY claim, not a value: the target-dependent gates
    #: answer ``N/A`` rather than comparing a count of zero against a declared
    #: zero and calling the agreement a pass. It is the same distinction
    #: ``num_frames=1`` draws for the temporal gate.
    has_target: bool = True
    #: Declared output geometry as (H, W); ``None`` means "same as input".
    output_shape: tuple[int, int] | None = None
    #: Declared length of a SOURCE CLIP the loader requires, in frames. Not the
    #: same claim as ``num_frames``, which is the window one sample contains:
    #: this is what the loader needs to be given. ``None`` means the arm
    #: declares no requirement, which is the common and correct case -- most
    #: loaders work with whatever clip length they are handed. When it is
    #: ``None`` and the loader turns out to need a particular length anyway,
    #: that is a requirement its interface does not express, and the gates
    #: report UNDECLARED rather than crashing.
    clip_length: int | None = None
    #: Actual source-clip length manufactured for this probe.  Keeping this
    #: separate from ``clip_length`` prevents the fixture implementation from
    #: silently becoming the subject's declaration.  Escalation changes this
    #: value while leaving an absent declaration absent.
    fixture_clip_length: int | None = None
    #: Optional, explicit content contract evaluated at a named boundary.
    content: ContentContract | None = None
    options: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class LoaderAdapter(Protocol):
    """Minimal surface an auditable pipeline must expose."""

    #: Human-readable identifier recorded in the certificate.
    name: str

    def build(self, spec: LoaderSpec) -> Any:
        """Instantiate the pipeline's dataset object for ``spec``."""

    def sample(self, loader: Any, index: int) -> Sample:
        """Return item ``index`` at the loader-output observation boundary."""

    def sample_training_step(self, loader: Any, index: int) -> Sample:  # pragma: no cover - optional
        """Return the post-collate item immediately before model invocation.

        Optional.  Adapters that implement this method let the content gates
        cover collate functions, normalisation and device transfer as well as
        the dataset item.  The protocol reports ``SKIP`` at this boundary when
        an adapter cannot expose it; it never relabels loader output as a
        training-step observation.
        """
        ...

    def disable_generator(self) -> Any:  # pragma: no cover - optional
        """Context manager installing a generator that raises ``GeneratorReached``.

        Optional. When an adapter provides it, the pre-computed-input gate has a
        positive discriminator: reaching the generator is an exception, not a
        number. Adapters that cannot patch their generator may omit it, and the
        gate falls back to inspecting the delivered tensor alone --- a weaker
        test, and the certificate records which of the two was used.
        """
        ...

    def __len__(self) -> int:  # pragma: no cover - optional
        ...


@dataclass
class CallRecorder:
    """Records which functions a pipeline actually reached.

    An adapter names the callables it wants observed and :meth:`wrap` replaces
    each by a wrapper that records the call and then runs the original with the
    same arguments, so the loader behaves as before; the checks read the log.
    A divergence between the declared call graph and this log is the finding.
    """

    calls: list[str] = field(default_factory=list)
    #: Folders of the project's own code. While :attr:`watching`, :meth:`traced`
    #: records, without anything being named, every call to a function defined
    #: there and every call such a function makes into other code (OpenCV,
    #: NumPy, ...), but nothing inside that other code.
    roots: list[str] = field(default_factory=list)
    watching: bool = False
    #: While traced, each call as (number of the calling call, its function,
    #: the function called), the calling call numbered in order of execution.
    nested: list[tuple] = field(default_factory=list)

    def record(self, name: str) -> None:
        self.calls.append(name)

    @contextmanager
    def traced(self):
        if not (self.watching and self.roots):
            yield
            return
        roots = [os.path.join(os.path.realpath(r), "") for r in self.roots]
        own = os.path.join(os.path.dirname(os.path.realpath(__file__)), "")
        known: dict[str, str | None] = {}

        def project(filename: str) -> str | None:
            if filename not in known:
                p, rel = os.path.realpath(filename), None
                if (not filename.startswith("<") and not p.startswith(own)
                        and "site-packages" not in p):
                    for r in roots:
                        if p.startswith(r):
                            rel = p[len(r):]
                            break
                known[filename] = rel
            return known[filename]

        running: dict[int, tuple[int, str]] = {}    # frame of the project's code -> its call

        def note(name: str, caller) -> None:
            self.record(name)
            parent = running.get(id(caller))
            if parent is not None:
                self.nested.append((*parent, name))

        def hook(frame, event, arg):
            if event == "call":
                code = frame.f_code
                rel = project(code.co_filename)
                if rel is not None:
                    owner = frame.f_locals.get("self")
                    cls = f"{type(owner).__name__}." if owner is not None else ""
                    name = f"{rel}:{cls}{code.co_name}"
                    note(name, frame.f_back)
                    running[id(frame)] = (len(self.calls), name)
                elif frame.f_back is not None and project(frame.f_back.f_code.co_filename):
                    name = _public(_offered_by(frame.f_globals.get("__name__", ""), code),
                                   code.co_name)
                    if name and not name.startswith("treatment_identity"):
                        note(name, frame.f_back)
            elif event == "return":
                running.pop(id(frame), None)
            elif event == "c_call" and project(frame.f_code.co_filename):
                name = _library_name(arg)
                if name:
                    note(name, frame)

        previous = sys.getprofile()
        sys.setprofile(hook)
        try:
            yield
        finally:
            sys.setprofile(previous)

    def wrap(self, owner, attr: str, label: str | None = None) -> bool:
        """Record every call to ``owner.attr`` under ``label`` (default: ``attr``).

        Wrapping is idempotent, so building a loader twice does not record a
        call twice. Returns False when ``owner`` has no such attribute.
        """
        original = getattr(owner, attr, None)
        if original is None:
            return False
        if getattr(original, "_ti_wrapped", False):
            return True
        name, recorder = label or attr, self

        def wrapper(*args, **kwargs):
            recorder.record(name)
            return original(*args, **kwargs)

        wrapper._ti_wrapped = True                    # type: ignore[attr-defined]
        wrapper.__wrapped__ = original                # type: ignore[attr-defined]
        setattr(owner, attr, wrapper)
        return True

    def count(self, name: str) -> int:
        return sum(1 for c in self.calls if c == name)

    def order(self, among: set[str]) -> list[str]:
        return [c for c in self.calls if c in among]

    def reset(self) -> None:
        self.calls.clear()
        self.nested.clear()


_PLAIN_TYPES = (str, bytes, list, tuple, dict, set, frozenset, int, float, bool,
                type(None), range, slice)


def _library_name(fn) -> str | None:
    """The name of a compiled function or method called from the project's code,
    or None for the language's own (``len``, ``list.append``, ...)."""
    owner = getattr(fn, "__self__", None)
    module = getattr(fn, "__module__", None)
    name = getattr(fn, "__name__", "?")
    if module and module != "builtins":
        return _public(module, name)
    if owner is None or isinstance(owner, _PLAIN_TYPES) or type(owner).__module__ == "builtins":
        return None
    if isinstance(owner, type):                       # a class method
        return _public(owner.__module__, f"{owner.__name__}.{name}")
    return _public(type(owner).__module__, f"{type(owner).__name__}.{name}")


def _offered_by(module: str, code) -> str:
    """The package that offers a function at its top (``numpy`` for
    ``numpy.lib.function_base.flip``), or the module it is defined in."""
    top = sys.modules.get(module.split(".", 1)[0])
    fn = getattr(top, code.co_name, None)
    fn = getattr(fn, "__wrapped__", fn)               # NumPy's dispatchers
    return top.__name__ if getattr(fn, "__code__", None) is code else module


def _public(module: str, name: str) -> str | None:
    """``module.name`` for a function a library offers its users; None for the
    machinery underneath (``_thread``, NumPy's dispatchers, ``__getattr__``)."""
    last = name.rsplit(".", 1)[-1]
    if (not module or any(p.startswith("_") for p in module.split("."))
            or (last.startswith("_") and last != "__call__")):
        return None
    return f"{module}.{name}"
