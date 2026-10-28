# Regeneration under treatment-identity 1.5.0

These files were regenerated on 2026-08-11 after correcting the
`has_target=true`/missing-target branch and moving new certificates to schema
1.2. They are intentionally separate from `campaigns/data/`, the archived
1.4.2 results, so a later run cannot silently replace deposited evidence.
The generating tree identified itself as `1.5.0rc1` and was tagged `v1.5.0`
without further change; see `environment/VALIDATION_2026-08-11.md`.

- `default/`: Campaign M (20 fixture seeds) and Campaign P with default fixture
  lengths and five timing repetitions.
- `escalation/`: Campaign P with clip-length escalation and five timing
  repetitions.

`verify_regeneration.py` compares the logical fields with the archived 1.4.2
tables. All three comparisons are identical:

- Campaign M: 40 `(defect, target gate, gate, status, target?, detected?)`
  records.
- Campaign P default: 32 `(subject, gate, status)` records.
- Campaign P escalation: 32 `(subject, gate, status)` records.

Timing columns are not compared. They changed substantially with cache and
machine load, as the manuscript already warns. The corrected missing-target
branch is covered by its dedicated regression test but is not exercised by any
M/P table cell, so unchanged campaign statuses are the expected result.

Run:

```bash
python campaigns/data_1.5.0/verify_regeneration.py
```

This directory is the evidence deposited with release 1.5.0. The frozen tables
of the manuscript were produced by 1.4.2 and are cited at that version's DOI;
nothing here supersedes them.
