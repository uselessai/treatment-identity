# Evidence for treatment-identity 1.6.0

This directory is the complete logical evidence used by the SQJ manuscript.
It describes the current mechanism only.

- Campaign M: 10/10 targeted defect--gate pairs detected, zero collateral
  failures, and 0/120 non-`PASS` outcomes across 20 conforming-loader runs.
- Campaign P: eight third-party loaders, six loader-driven gates per subject,
  48 logical cells, eight complete profiles, and no `FAIL`, `UNDECL` or
  `ERROR` outcome.
- `clip_escalation=1` records the robust default. The BasicSR REDS contract
  explicitly declares its documented 100-frame source-clip requirement.

Verify hashes and logical invariants from the repository root:

```bash
python3 campaigns/data_1.6.0/verify_evidence.py
```
