"""Where the results of a release are kept, one folder per loader.

    data_<release>/loaders/<loader>/profile.adapter.json    through its adapter (Studies 3, 4)
                                    profile.wizard.json     through its training command (Study 5)
                                    certificate.wizard.json requirements, followed values, switches
                                    answers.wizard.json     the answers the wizard used
                                    questions.wizard.json   every question, and how it was answered
    data_<release>/loaders/comparison.csv                   the two profiles, field by field

Each result is stored once; this module reads them back.
"""
from __future__ import annotations

import json
from pathlib import Path

ADAPTER, WIZARD = "profile.adapter.json", "profile.wizard.json"
CERTIFICATE, ANSWERS, QUESTIONS = ("certificate.wizard.json", "answers.wizard.json",
                                   "questions.wizard.json")


def loader_dir(data: Path, key: str) -> Path:
    return Path(data) / "loaders" / key


def write(data: Path, key: str, name: str, document) -> Path:
    d = loader_dir(data, key)
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text(json.dumps(document, indent=2, default=str) + "\n")
    return d / name


def read(data: Path, key: str, name: str):
    f = loader_dir(data, key) / name
    return json.loads(f.read_text()) if f.exists() else None


def adapter_profiles(data: Path) -> list[dict]:
    """Every profile taken through an adapter."""
    root = Path(data) / "loaders"
    return [json.loads(f.read_text()) for f in sorted(root.glob(f"*/{ADAPTER}"))]
