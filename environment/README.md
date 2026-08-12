# Exact environment reconstruction

This directory reconstructs the environment used for the current 1.6.0
evidence. It describes the final state, not a release-by-release history.

- `rrtn-linux-64-conda.lock`: explicit Linux/x86-64 Conda artifact URLs.
- `rrtn-pip.lock`: exact pip-installed state.
- `rrtn-linux-64.yml`: human-readable exact-version export.
- `usercustomize.py`: compatibility aliases loaded by the retained interpreter.
- `create_and_validate.sh`: creates an empty prefix, installs the package,
  runs unit and isolated-wheel tests, regenerates Campaigns M/P and verifies
  their logical matrices.
- `verify_clean_campaigns.py`: compares logical outcomes with
  `campaigns/data_1.6.0` while ignoring host-dependent timings.
- `VALIDATION_2026-08-12.md`: final empty-prefix validation record.

The pip lock is an observed executable state, not a newly solvable dependency
specification. Installation uses `--no-deps` because retained package metadata
contains declared conflicts.

```bash
environment/create_and_validate.sh \
  /absolute/path/to/empty/prefix \
  /absolute/path/to/results
```

The script refuses a non-empty target prefix.
