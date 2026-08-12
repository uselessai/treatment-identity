"""Verify that the treatment an experiment declares is the one its loader delivers."""
from ._version import __version__
from .adapter import (AdapterError, CallRecorder, ContentContract,
                      LoaderAdapter, LoaderSpec, Sample, StreamContract)
from .boundary import (TrainingStepGuard, TreatmentContractViolation,
                       guard_training_step)
from .certificate import Certificate, SCHEMA_VERSION, environment_fingerprint, file_sha256
from .checks import (CLIP_LENGTH_LADDER, CheckResult, with_clip_escalation,
                     check_channel_content,
                     check_geometry, check_operator_trace,
                     check_precomputed_input, check_separability,
                     check_sample_channel_content, check_sample_value_range,
                     check_target_transforms, check_temporal_window)
from .checks import check_value_range
from .schema import load_certificate_schema, validate_certificate
from .seeding import UNCONTROLLED, seed_all
from . import fixtures

__all__ = [
    "AdapterError", "CallRecorder", "ContentContract", "LoaderAdapter",
    "LoaderSpec", "Sample", "StreamContract",
    "Certificate", "SCHEMA_VERSION", "environment_fingerprint", "file_sha256",
    "CheckResult", "CLIP_LENGTH_LADDER", "with_clip_escalation",
    "check_channel_content", "check_geometry", "check_operator_trace",
    "check_precomputed_input", "check_separability", "check_target_transforms",
    "check_temporal_window", "check_value_range",
    "check_sample_channel_content", "check_sample_value_range",
    "TrainingStepGuard", "TreatmentContractViolation", "guard_training_step",
    "fixtures", "load_certificate_schema",
    "validate_certificate", "seed_all", "UNCONTROLLED", "__version__",
]
