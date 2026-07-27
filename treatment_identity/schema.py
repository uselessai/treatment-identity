"""Load and apply the distributed treatment-certificate JSON Schema."""

from __future__ import annotations

import copy
import json
from functools import lru_cache
from importlib import resources
from typing import Any, Mapping

SCHEMA_FILE = "schemas/treatment-certificate-1.0.schema.json"

__all__ = ["SCHEMA_FILE", "load_certificate_schema", "validate_certificate"]


@lru_cache(maxsize=1)
def _schema_document() -> dict[str, Any]:
    resource = (resources.files("treatment_identity")
                .joinpath("schemas")
                .joinpath("treatment-certificate-1.0.schema.json"))
    schema = json.loads(resource.read_text(encoding="utf-8"))
    return schema


@lru_cache(maxsize=1)
def _validator() -> Any:
    # Keep ordinary gate execution importable in audited legacy environments
    # that predate the schema tooling. A correctly installed wheel
    # obtains jsonschema from project dependencies; source-tree callers only
    # need it when they explicitly validate or write a certificate.
    try:
        from jsonschema import Draft202012Validator, FormatChecker
    except ImportError as exc:
        raise RuntimeError(
            "Certificate validation requires jsonschema>=4.10. Install the "
            "treatment-identity package with its declared dependencies."
        ) from exc
    Draft202012Validator.check_schema(_schema_document())
    return Draft202012Validator(
        _schema_document(),
        format_checker=FormatChecker(),
    )


def load_certificate_schema() -> dict[str, Any]:
    """Return a copy of the JSON Schema shipped inside the installed package."""
    return copy.deepcopy(_schema_document())


def validate_certificate(document: Mapping[str, Any]) -> None:
    """Validate one serialised certificate.

    Raises :class:`jsonschema.exceptions.ValidationError` when the document does
    not satisfy the distributed schema. The function has no filesystem or
    environment side effects.
    """
    _validator().validate(dict(document))
