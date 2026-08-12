# Clean-environment validation, 2026-08-11

Validation was performed from the exact Linux/x86-64 Conda artifact lock, the
154-entry pip lock and the archived `usercustomize.py` compatibility layer.  The
temporary environment prefix was
`/tmp/treatment-identity-clean-20260811`; campaign output was written outside
the repository to `/tmp/treatment-identity-clean-validation-20260811`.

The source tree validated here identified itself as `1.5.0rc1` and distributed
schema `treatment-certificate/1.2`.  That tree was tagged `v1.5.0` without
further change: the released 1.5.0 differs from the validated candidate only in
the version string carried by `_version.py`, `CITATION.cff`, the assertion in
`tests/test_certificate.py` that pins the two together, and the directory names
holding the regenerated evidence.  No gate, schema or campaign input differs.
This paragraph records what the run actually observed rather than restating it
under the later name.  In the clean environment:

- all 23 unit tests passed;
- the isolated wheel-installation and CLI self-test passed;
- Campaign M detected 8/8 targeted cases, reported the two designed misses,
  produced zero collateral failures and zero false alarms in 80 gate
  executions;
- default Campaign P ran all eight subjects, produced seven complete profiles
  and one partial profile, and recorded the same 32 logical cells as the
  retained release-candidate output;
- clip-escalated Campaign P ran all eight subjects to completion and recorded
  the same 32 logical cells as the retained release-candidate output; and
- `verify_clean_campaigns.py` reported all three clean-run logical matrices
  identical to the retained candidate, whose own verifier reported 12 valid
  file hashes and logical identity with the archived 1.4.2 matrices.

The first reconstruction attempt also exposed a previously implicit
environment input.  The retained interpreter loaded a `usercustomize.py` that
restored the removed top-level `scipy.finfo` alias used by KAIR BlindSR.  Without
that file the clean campaign correctly produced four `ERROR` cells for that
subject.  The exact compatibility layer is now archived and installed by
`create_and_validate.sh`; after it was installed, the clean logical matrices
matched exactly.

The pip state is intentionally reconstructed with `--no-deps`: it is a record
of an observed installed environment, and its package metadata is not
solver-consistent (`pyiqa==0.1.14.1` declares optional packages absent from the
environment and an upper bound on `accelerate` that the retained state exceeds).
This limitation does not affect the unit, wheel, M or P validations above, but
it is recorded so the lock is not misrepresented as a newly resolved
environment.

After integrating those two reconstruction requirements, the complete public
procedure was itself rerun from a second empty prefix with one command:

```bash
bash environment/create_and_validate.sh \
  /tmp/treatment-identity-one-command-20260811 \
  /tmp/treatment-identity-one-command-results-20260811
```

It completed without intervention: 23 unit tests passed, the isolated wheel and
CLI self-test passed, M detected 8/8 targeted cases with zero collateral or
false alarms, default and escalated P ran all eight registered subjects, and
the final comparison reported all three clean-run logical matrices identical to
the retained candidate.
