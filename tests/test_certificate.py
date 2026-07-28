"""Unit tests for certificate status, types, schema, and versioning."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import get_type_hints

from jsonschema import ValidationError

from treatment_identity import (
    Certificate,
    CheckResult,
    __version__,
    load_certificate_schema,
    validate_certificate,
)
from treatment_identity.certificate import GeometryEvidence, ValueRangeEvidence


def certificate_with(*statuses: str) -> Certificate:
    cert = Certificate("fixture", "one bounded fixture treatment")
    for index, status in enumerate(statuses):
        cert.add(CheckResult(f"gate_{index}", status, "test result", {}))
    return cert


class CertificateStatusTests(unittest.TestCase):
    def test_no_executed_gate_is_inconclusive(self) -> None:
        for statuses in ((), ("N/A",), ("SKIP",), ("N/A", "SKIP")):
            with self.subTest(statuses=statuses):
                self.assertEqual(certificate_with(*statuses).status, "INCONCLUSIVE")

    def test_all_applicable_gates_pass(self) -> None:
        self.assertEqual(certificate_with("PASS", "PASS").status, "PASS")
        self.assertEqual(certificate_with("PASS", "N/A").status, "PASS")

    def test_skipped_applicable_gate_makes_result_partial(self) -> None:
        self.assertEqual(certificate_with("PASS", "SKIP").status, "PARTIAL")
        self.assertEqual(certificate_with("PASS", "N/A", "SKIP").status, "PARTIAL")

    def test_failure_dominates(self) -> None:
        self.assertEqual(
            certificate_with("PASS", "SKIP", "FAIL", "N/A").status,
            "FAIL",
        )

    def test_undeclared_blocks_identity_without_being_a_failure(self) -> None:
        cert = certificate_with("PASS", "UNDECL", "N/A")
        self.assertEqual(cert.status, "UNDECLARED")
        # Nothing was contradicted, so nothing is a failure ...
        self.assertEqual(cert.failures, [])
        # ... but identity is blocked all the same.
        self.assertEqual(cert.undeclared, ["gate_1"])
        self.assertEqual(cert.divergences, ["gate_1"])

    def test_a_contradicted_claim_outranks_an_absent_one(self) -> None:
        self.assertEqual(certificate_with("UNDECL", "FAIL").status, "FAIL")


class SeedingTests(unittest.TestCase):
    """The protocol's own reproducibility, which an earlier version lacked."""

    def test_seed_all_reports_what_it_reached_and_what_it_did_not(self) -> None:
        from treatment_identity import seed_all

        record = seed_all(1234)
        self.assertEqual(record["seed"], 1234)
        self.assertIn("python.random", record["seeded"])
        self.assertIn("numpy.random", record["seeded"])
        # Naming the generator we cannot govern is part of the contract.
        self.assertIn("albumentations", record["uncontrolled"])

    def test_the_same_seed_gives_the_same_draws(self) -> None:
        import random

        from treatment_identity import seed_all

        seed_all(7)
        first = [random.random() for _ in range(5)]
        seed_all(7)
        self.assertEqual(first, [random.random() for _ in range(5)])


class EnvironmentAssumptionTests(unittest.TestCase):
    """Claims the manuscript makes about the environment, asserted not assumed."""

    def test_scipy_finfo_is_the_numpy_object(self) -> None:
        # The audit calls the one differing token in fspecial_gaussian
        # semantically equivalent. If a future SciPy drops the re-export, this
        # fails and the claim is revisited rather than silently inherited.
        import numpy as np
        import scipy

        self.assertTrue(hasattr(scipy, "finfo"))
        self.assertIs(scipy.finfo, np.finfo)


class CertificateSchemaTests(unittest.TestCase):
    def test_schema_is_distributed_and_validates_a_typed_certificate(self) -> None:
        schema = load_certificate_schema()
        self.assertEqual(
            schema["$id"],
            "urn:treatment-identity:treatment-certificate:1.1",
        )

        cert = certificate_with("PASS", "SKIP")
        cert.value_range = {
            "gt": [0.0, 1.0],
            "lq": [-1.0, 1.0],
            "note": "fixture range",
        }
        cert.geometry = {
            "declared": [180, 320],
            "delivered": [368, 640],
            "source": "recorded adapter evidence",
        }
        cert.data_order_seed = "framework sampler does not expose this seed"
        validate_certificate(cert.to_dict())

    def test_schema_rejects_unknown_gate_status(self) -> None:
        document = certificate_with("PASS").to_dict()
        document["gates"][0]["status"] = "MAYBE"
        with self.assertRaises(ValidationError):
            validate_certificate(document)

    def test_write_validates_by_default(self) -> None:
        cert = certificate_with("PASS")
        cert.geometry = {
            "declared": [180, 320],
            "delivered": [368, 640],
            "source": "fixture",
        }
        with tempfile.TemporaryDirectory(prefix="ti_unit_") as directory:
            path = cert.write(Path(directory) / "certificate.json")
            document = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(document["status"], "PASS")
        validate_certificate(document)

    def test_write_refuses_invalid_document(self) -> None:
        cert = certificate_with("PASS")
        cert.geometry = {  # type: ignore[typeddict-item]
            "declared": [180, 320],
            "delivered": [368, 640],
        }
        with tempfile.TemporaryDirectory(prefix="ti_unit_") as directory:
            path = Path(directory) / "invalid.json"
            with self.assertRaises(ValidationError):
                cert.write(path)
            self.assertFalse(path.exists())


class CertificateContractTests(unittest.TestCase):
    def test_version_is_the_contract_release(self) -> None:
        self.assertEqual(__version__, "1.1.0")

    def test_runtime_type_hints_match_serialised_evidence(self) -> None:
        hints = get_type_hints(Certificate)
        self.assertIn(ValueRangeEvidence, hints["value_range"].__args__)
        self.assertIn(GeometryEvidence, hints["geometry"].__args__)
        self.assertEqual(set(hints["data_order_seed"].__args__), {int, str, type(None)})


if __name__ == "__main__":
    unittest.main()
