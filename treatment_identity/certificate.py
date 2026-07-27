"""The treatment certificate.

A configuration file records what a run was *asked* to do. The certificate
records what it *did*: which branch executed, which bytes arrived, how many
frames were reachable, which operators ran and in what order, and whether every
gate passed. It is the artefact we propose reporting in place of the config.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypedDict

from .checks import CheckResult
from .schema import validate_certificate

SCHEMA_VERSION = "treatment-certificate/1.0"

__all__ = [
    "Certificate",
    "GateRecord",
    "GeometryEvidence",
    "SCHEMA_VERSION",
    "ValueRangeEvidence",
    "environment_fingerprint",
    "file_sha256",
]


class ValueRangeEvidence(TypedDict, total=False):
    """Observed numeric ranges, or an explanation when probing failed."""

    gt: list[float]
    lq: list[float]
    note: str
    error: str


class GeometryEvidence(TypedDict):
    """Declared and delivered ``[height, width]`` plus evidence provenance."""

    declared: list[int]
    delivered: list[int]
    source: str | None


class GateRecord(TypedDict):
    """Serialised result of one treatment-identity gate."""

    gate: str
    status: str
    message: str
    evidence: dict[str, Any]


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit(repo: str | Path) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:
        return None


def _git_dirty(repo: str | Path) -> bool | None:
    """Is the checkout modified, ignoring bytecode the audit itself produced?

    Importing a repository's modules writes ``__pycache__``, which would mark
    every audited clone dirty and make the flag useless exactly where it is
    supposed to be informative.
    """
    try:
        out = subprocess.run(["git", "-C", str(repo), "status", "--porcelain"],
                             capture_output=True, text=True, timeout=10)
    except Exception:
        return None
    for line in out.stdout.splitlines():
        path = line[3:].strip().strip('"')
        if path.endswith((".pyc", ".pyo")) or "__pycache__" in path:
            continue
        if path:
            return True
    return False


def environment_fingerprint(packages: tuple[str, ...] = (
        "numpy", "torch", "torchvision", "opencv-python", "albumentations",
        "scipy", "Pillow", "scikit-image")) -> dict[str, Any]:
    """Record the versions that change semantics, not just the ones that exist."""
    import importlib.metadata as md
    versions: dict[str, str | None] = {}
    for p in packages:
        try:
            versions[p] = md.version(p)
        except Exception:
            versions[p] = None
    # cv2/albumentations often import under a different name than they ship as
    for mod, key in (("cv2", "opencv-python"), ("albumentations", "albumentations")):
        if versions.get(key) is None:
            try:
                versions[key] = __import__(mod).__version__
            except Exception:
                pass
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": versions,
    }


@dataclass
class Certificate:
    """What a run actually delivered."""

    pipeline: str
    expected_treatment: str
    schema: str = SCHEMA_VERSION
    created_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # provenance
    repository: str | None = None
    commit: str | None = None
    working_tree_dirty: bool | None = None

    # what was reached and delivered
    branch_reached: str | None = None
    delivery_modality: str | None = None          # "online" | "pre-rendered"
    gt_source: str | None = None
    lq_source: str | None = None
    gt_stream_sha256: str | None = None
    lq_stream_sha256: str | None = None
    channels: int | None = None
    value_range: ValueRangeEvidence | None = None
    geometry: GeometryEvidence | None = None
    unique_frames_reachable: int | None = None
    clip_length: int | None = None
    observed_operator_order: list[str] | None = None
    target_transform_count: float | None = None

    # randomness, kept apart on purpose
    training_seed: int | None = None
    render_seed: int | None = None
    data_order_seed: int | str | None = None
    uncontrolled_rng: list[str] = field(default_factory=list)

    environment: dict[str, Any] = field(default_factory=environment_fingerprint)
    gates: list[GateRecord] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    def add(self, result: CheckResult) -> "Certificate":
        self.gates.append({
            "gate": result.name,
            "status": result.status,
            "message": result.message,
            "evidence": result.evidence,
        })
        return self

    def with_repository(self, path: str | Path) -> "Certificate":
        self.repository = str(path)
        self.commit = _git_commit(path)
        self.working_tree_dirty = _git_dirty(path)
        return self

    @property
    def status(self) -> str:
        statuses = [g.get("status") for g in self.gates]
        if "FAIL" in statuses:
            return "FAIL"
        if "PASS" not in statuses:
            return "INCONCLUSIVE"
        if "SKIP" in statuses or any(
                status not in {"PASS", "N/A"} for status in statuses):
            return "PARTIAL"
        return "PASS"

    @property
    def failures(self) -> list[str]:
        return [g["gate"] for g in self.gates if g["status"] == "FAIL"]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status
        d["failed_gates"] = self.failures
        return d

    def validate(self) -> "Certificate":
        """Validate the serialised certificate against the bundled JSON Schema."""
        validate_certificate(self.to_dict())
        return self

    def write(self, path: str | Path, *, validate: bool = True) -> Path:
        """Serialise the certificate, validating it by default."""
        path = Path(path)
        document = self.to_dict()
        if validate:
            validate_certificate(document)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(document, indent=2, sort_keys=False) + "\n")
        return path

    def summary(self) -> str:
        lines = [f"treatment certificate — {self.pipeline}",
                 f"  expected treatment : {self.expected_treatment}",
                 f"  commit             : {(self.commit or '?')[:12]}"
                 + ("  (working tree dirty)" if self.working_tree_dirty else ""),
                 f"  overall            : {self.status}"]
        if self.unique_frames_reachable is not None and self.clip_length:
            pct = 100.0 * self.unique_frames_reachable / self.clip_length
            lines.append(f"  frames reachable   : {self.unique_frames_reachable}"
                         f"/{self.clip_length}  ({pct:.1f}%)")
        for g in self.gates:
            lines.append(f"  [{g['status']:4}] {g['gate']}")
        return "\n".join(lines)
