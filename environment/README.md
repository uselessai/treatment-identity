# Reproducing the retained audit environment

The environment used to rerun the package tests and the M/P campaigns on
2026-08-11 is recorded in three complementary forms:

- `rrtn-linux-64-conda.lock`: explicit Linux/x86-64 Conda artifact URLs,
  including exact builds; this bypasses dependency solving.
- `rrtn-pip.lock`: exact versions of packages installed through pip (Conda
  packages and an obsolete installed copy of `treatment-identity` are not
  duplicated here).
- `rrtn-linux-64.yml`: human-readable exact-version export with channels and
  both package families.
- `usercustomize.py`: the compatibility layer loaded by the retained Python
  interpreter; it restores the removed top-level `scipy.finfo` alias used by
  the audited KAIR BlindSR loader.

The machine-specific `prefix` was deliberately removed. CUDA packages are
retained even though the treatment-identity gates run on CPU, because this is
an environment record rather than a minimal dependency claim.

Create and validate a clean environment at an explicit temporary prefix with:

```bash
bash environment/create_and_validate.sh /absolute/path/to/new/environment
```

To run M and both P modes as part of the same validation, provide a second
absolute path:

```bash
bash environment/create_and_validate.sh \
  /absolute/path/to/new/environment /absolute/path/to/results
```

The script refuses a non-empty environment prefix. It installs the exact Conda
artifacts, bootstraps the recorded `setuptools==60.2.0`, installs the pip lock
with `--no-deps --no-build-isolation`, installs the current source tree, and
installs the recorded compatibility layer before it runs the unit and isolated
wheel tests. With the second argument it then runs M, P and P with clip
escalation and compares their logical cells with the retained release-candidate
matrices. It leaves the environment and results in place for inspection.

The `--no-deps` choice is intentional. This lock records the complete observed
environment, which contains pre-existing metadata inconsistencies (notably
`pyiqa==0.1.14.1` alongside `accelerate==1.13.0` and optional packages omitted
from the environment). Asking a current resolver to reinterpret that state
fails; installing all recorded distributions directly recreates it and is the
procedure validated here. This is an exact-state reconstruction, not a claim
that the historical environment is dependency-solver clean.

The portability campaign also needs the eight public subject repositories at
the commits recorded by the manuscript. Their paths are parameters of the
campaign adapters and they are not bundled in this repository.

## Validation host

- OS/kernel: Ubuntu 24.04, Linux 6.11.0-29-generic, x86-64.
- CPU: Intel Core i9-12900K, 16 physical cores / 24 logical CPUs.
- RAM: 125 GiB.
- Workspace storage: XPG GAMMIX S11 Pro NVMe, 953.9 GiB.
- Conda: 24.11.3.
- Retained environment: Python 3.10.14, NumPy 1.26.4, SciPy 1.14.1,
  OpenCV 4.10.0.84, PyTorch 2.1.2, torchvision 0.16.2, MMCV 2.2.0,
  jsonschema 4.26.0 and Albumentations 2.0.8.

These host details describe the 2026-08-11 validation run. They do not
retroactively identify the unlogged hardware used for the original timing
campaign; the manuscript therefore continues to treat those timings as
descriptive observations rather than a hardware-normalised benchmark.

## Lock integrity

```text
3f0f9b619efe76d9ae8e8ff1190242e34edee814e4021d03ce672ff3f0cf1423  rrtn-linux-64.yml
8b96384fbc0c57c33733f9e05a5898684a2a1431090fc4ee78588dd080b11906  rrtn-linux-64-conda.lock
9f7c72f97738365138e0eccfc756fcc4b09edae18f357fab879e4b64ccc47f1f  rrtn-pip.lock
14b2f442812bd086dad05054d6f29571fcf8311302b0f9226fce26ee5e249b23  usercustomize.py
```
