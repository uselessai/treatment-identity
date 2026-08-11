"""The interface a pipeline must expose to be audited.

The protocol is deliberately small: everything the checks need is a way to
build a loader over a fixture and a way to pull one sample out of it. Nothing
in this module knows about film restoration, or about the study that produced
it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

__all__ = ["Sample", "LoaderSpec", "LoaderAdapter", "AdapterError", "GeneratorReached"]


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
    """One item returned by the loader at the training-step-input boundary.

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
    options: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class LoaderAdapter(Protocol):
    """Minimal surface an auditable pipeline must expose."""

    #: Human-readable identifier recorded in the certificate.
    name: str

    def build(self, spec: LoaderSpec) -> Any:
        """Instantiate the pipeline's dataset object for ``spec``."""

    def sample(self, loader: Any, index: int) -> Sample:
        """Return item ``index`` exactly as the training loop would receive it."""

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

    Adapters patch the callables they want observed; the checks read the log.
    A divergence between the declared call graph and this log is the finding.
    """

    calls: list[str] = field(default_factory=list)

    def record(self, name: str) -> None:
        self.calls.append(name)

    def count(self, name: str) -> int:
        return sum(1 for c in self.calls if c == name)

    def order(self, among: set[str]) -> list[str]:
        return [c for c in self.calls if c in among]

    def reset(self) -> None:
        self.calls.clear()
