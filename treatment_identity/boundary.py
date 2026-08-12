"""Executable content contracts at the last boundary before model invocation.

Loader audits establish what a dataset object returns.  Training code may still
collate, normalise, cast or move that value before it reaches the model.  This
module wraps the real training-step/model callable and checks that final value;
it never substitutes a loader observation for a pre-model observation.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import wraps
from typing import Any, TypeVar

from .adapter import ContentContract, Sample
from .certificate import Certificate
from .checks import (CheckResult, FAIL, UNDECLARED,
                     check_sample_channel_content,
                     check_sample_value_range)

__all__ = [
    "TrainingStepGuard",
    "TreatmentContractViolation",
    "guard_training_step",
]

F = TypeVar("F", bound=Callable[..., Any])
SampleExtractor = Callable[[Any], Sample]


def _default_extractor(batch: Any) -> Sample:
    """Extract the explicit protocol shape without guessing framework aliases."""
    if isinstance(batch, Sample):
        return batch
    if isinstance(batch, Mapping) and "lq" in batch:
        return Sample(
            lq=batch["lq"],
            gt=batch.get("gt"),
            frame_ids=batch.get("frame_ids"),
            extra={"mapping_keys": sorted(str(key) for key in batch)},
        )
    raise TypeError(
        "training-step batch must be a Sample or a mapping containing 'lq'; "
        "provide extractor=... for any other framework batch shape"
    )


class TreatmentContractViolation(RuntimeError):
    """Raised before model execution when the delivered content contradicts it."""

    def __init__(self, results: tuple[CheckResult, ...]):
        self.results = results
        findings = "; ".join(
            f"{result.name}: {result.message}"
            for result in results if result.status in {FAIL, UNDECLARED}
        )
        super().__init__(f"training-step treatment contract blocked execution: {findings}")


@dataclass
class TrainingStepGuard:
    """Validate the post-collate batch immediately before calling the model.

    ``batch_arg`` names the positional index or keyword used by the wrapped
    callable.  Methods commonly use ``batch_arg=1`` when wrapping an unbound
    method (``self`` occupies position zero); a bound method usually uses zero.
    Every observation can be appended to the same treatment certificate used
    for the loader gates.
    """

    contract: ContentContract
    extractor: SampleExtractor = _default_extractor
    certificate: Certificate | None = None
    batch_arg: int | str = 0
    block_statuses: tuple[str, ...] = (FAIL, UNDECLARED)

    def observe(self, batch: Any) -> tuple[CheckResult, CheckResult]:
        sample = self.extractor(batch)
        results = (
            check_sample_value_range(
                sample, self.contract, boundary="training_step_input"),
            check_sample_channel_content(
                sample, self.contract, boundary="training_step_input"),
        )
        if self.certificate is not None:
            for result in results:
                self.certificate.add(result)
        return results

    def assert_batch(self, batch: Any) -> tuple[CheckResult, CheckResult]:
        results = self.observe(batch)
        if any(result.status in self.block_statuses for result in results):
            raise TreatmentContractViolation(results)
        return results

    def _batch_from_call(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        if isinstance(self.batch_arg, str):
            if self.batch_arg not in kwargs:
                raise TypeError(
                    f"wrapped call did not receive keyword batch {self.batch_arg!r}")
            return kwargs[self.batch_arg]
        try:
            return args[self.batch_arg]
        except IndexError as exc:
            raise TypeError(
                f"wrapped call has no positional batch argument {self.batch_arg}") from exc

    def wrap(self, step: F) -> F:
        """Return a callable that blocks a corrupt batch before ``step`` runs."""
        @wraps(step)
        def checked(*args: Any, **kwargs: Any) -> Any:
            self.assert_batch(self._batch_from_call(args, kwargs))
            return step(*args, **kwargs)

        return checked  # type: ignore[return-value]


def guard_training_step(*, contract: ContentContract,
                        extractor: SampleExtractor = _default_extractor,
                        certificate: Certificate | None = None,
                        batch_arg: int | str = 0,
                        block_statuses: tuple[str, ...] = (FAIL, UNDECLARED)
                        ) -> Callable[[F], F]:
    """Decorator form of :class:`TrainingStepGuard`."""
    guard = TrainingStepGuard(
        contract=contract,
        extractor=extractor,
        certificate=certificate,
        batch_arg=batch_arg,
        block_statuses=block_statuses,
    )
    return guard.wrap
