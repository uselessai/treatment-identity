# Clean-environment validation, 2026-08-12

The current 1.6.0 source was validated from an empty Linux/x86-64 prefix using
the explicit Conda artifact lock, the exact pip lock and the retained
`usercustomize.py` compatibility layer. The temporary paths were:

- environment: `/tmp/treatment-identity-1.6-env-fZLrpf`;
- regenerated evidence: `/tmp/treatment-identity-1.6-results-mR38Qz`.

The final source, including the anchored temporal fixture, produced these
results:

- all 42 unit tests passed;
- the wheel built as `treatment_identity-1.6.0-py3-none-any.whl`, installed in
  an isolated prefix and ran its CLI with the source tree unreachable;
- the CLI self-test made each of the six delivery defects fail its targeted
  gate and passed its positive separability and geometry controls;
- Campaign M detected 10/10 targeted pairs, with zero collateral failures and
  zero non-`PASS` outcomes in 120 conforming-loader gate executions;
- robust-default Campaign P ran all eight registered subjects, produced 48
  logical gate cells, and reported no `FAIL`, `UNDECL` or execution error; and
- `verify_clean_campaigns.py` confirmed that both clean-run logical matrices
  match the retained `campaigns/data_1.6.0` evidence.

The validation host ran Ubuntu 24.04, Linux 6.11.0-29, on an Intel Core
i9-12900K with 24 logical CPUs and 125 GiB RAM. Timing values remain
descriptive because process, cache and host load were not controlled.

The pip lock is an observed executable state rather than a newly solvable
dependency specification. The validation installs it with `--no-deps` because
retained package metadata contains declared conflicts. This limitation does
not alter the successful unit, wheel, mutation or portability checks above.

The public reconstruction entry point is:

```bash
bash environment/create_and_validate.sh \
  /absolute/path/to/empty/prefix \
  /absolute/path/to/results
```
