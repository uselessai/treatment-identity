#!/usr/bin/env python3
"""Regenerate the two matched deliveries consumed by campaign C.

The script executes the released RTN and RRTN degradation paths through the
same adapter and controlled fixture used by the field audit.  It refuses dirty
or incorrectly pinned clones by default and records the exact inputs in a
manifest beside the resulting ``.npy`` files.

Example::

    python3 campaigns/gen_deliveries.py \
        --repos /path/to/pinned/clones \
        --bank /path/to/noise_data \
        --out campaigns/data_1.7.1/deliveries
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from adapters.vp_code import VPCodeAdapter  # noqa: E402


SUBJECTS = {
    "RTN": {
        "folders": ("Bringing-Old-Films-Back-to-Life",
                    "zBringing-Old-Films-Back-to-Life"),
        "commit": "54df59417c9c",
        "channels": 3,
    },
    "RRTN": {
        "folders": ("RRTN-old-film-restoration",
                    "zRRTN-old-film-restoration"),
        "commit": "832495f24555",
        "channels": 1,
    },
}


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True,
        capture_output=True, text=True).stdout.strip()


def _resolve(roots: list[Path], folders: tuple[str, ...]) -> Path:
    for root in roots:
        for folder in folders:
            candidate = root / folder
            if candidate.is_dir():
                return candidate
    raise FileNotFoundError(f"none of {folders} found below {roots}")


def _fixture(seed: int, frames: int = 3,
             shape: tuple[int, int] = (64, 64)) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    height, width = shape
    return [rng.random((height, width, 3)).astype(np.float32)
            for _ in range(frames)]


def _digest(array: np.ndarray) -> str:
    data = np.ascontiguousarray(array, dtype=np.float32).tobytes()
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repos", type=Path, required=True, action="append",
                        help="root containing pinned public clones; repeatable")
    parser.add_argument("--bank", type=Path, required=True,
                        help="texture-template bank referenced by the releases")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fixture-seed", type=int, default=7)
    parser.add_argument("--render-seed", type=int, default=1234)
    parser.add_argument("--allow-dirty", action="store_true",
                        help="permit modified subject trees (not for release evidence)")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    fixture = _fixture(args.fixture_seed)
    manifest = {
        "generator": "campaigns/gen_deliveries.py",
        "fixture_seed": args.fixture_seed,
        "render_seed": args.render_seed,
        "neutralise_colour_jitter": True,
        "subjects": {},
    }
    digests = {}

    for name, metadata in SUBJECTS.items():
        repo = _resolve(args.repos, metadata["folders"])
        commit = _git(repo, "rev-parse", "--short=12", "HEAD")
        dirty = bool(_git(repo, "status", "--porcelain"))
        if commit != metadata["commit"]:
            parser.error(f"{name}: expected {metadata['commit']}, found {commit}")
        if dirty and not args.allow_dirty:
            parser.error(f"{name}: clone is dirty: {repo}")

        adapter = VPCodeAdapter(name, repo, texture_bank=args.bank,
                                channels=metadata["channels"])
        delivery = adapter.render(
            [frame.copy() for frame in fixture], seed=args.render_seed,
            neutralise_colour_jitter=True)
        delivery = np.asarray(delivery, dtype=np.float32)
        np.save(args.out / f"{name}.npy", delivery, allow_pickle=False)
        digest = _digest(delivery)
        digests[name] = digest
        manifest["subjects"][name] = {
            "repository_path": str(repo.resolve()),
            "commit": commit,
            "working_tree_dirty": dirty,
            "shape": list(delivery.shape),
            "dtype": str(delivery.dtype),
            "sha256_float32": digest,
        }

    manifest["deliveries_identical"] = len(set(digests.values())) == 1
    if not manifest["deliveries_identical"]:
        parser.error("the regenerated RTN and RRTN deliveries are not identical")
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"wrote two matched deliveries and {args.out / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
