# Studies

The studies of the article *The data-delivery checker: testing what data
loaders deliver in deep-learning experiments*, with
the scripts that ran them and their results. Every study was run with DDC
release 1.12.0, on one performance core of an Intel Core i9-12900K, with one
thread and no GPU (`taskset -c 4`, `OMP_NUM_THREADS=1`,
`CUDA_VISIBLE_DEVICES=`). External projects are not included; they are cloned
at the commits listed in the certificates and in `data_1.12.0/P_effort.csv`.

| Study | Question | Script | Results |
|---|---|---|---|
| 1, 2 | Does DDC detect introduced defects, without false alarms, and at what cost? | `campaign_M_seeded_defects.py`, `campaign_M2_multi_defect.py` | `data_1.12.0/M_*` |
| 3 | What does applying DDC to sixteen external loaders cost, and what does it report about them? | `campaign_P_portability.py`, `profile_loader.py`, `sweep_settings.py`, `adapters/` | `data_1.12.0/P_*` |
| 4 | Which defects appear in four released old-film restoration systems, and in DART? | `old_film/audit_vp_lineage.py`, `old_film/repeat_draws.py`, `old_film/audit_dart.py` | `old_film/certificates/` |
| 4, follow-ups | DDColor image mixing | `campaign_S_ddcolor_cutmix.py`, `probe_ddcolor_cutmix.py` | printed |

`old_film/documents/` holds the documents inspected for Study 4 (publications,
README files at the audited commits, search log and decisions) and a verifier
that checks them against the certificates. `figures/` holds the scripts that
draw the article's figures from the released loaders. `history/` keeps the
results of earlier releases; the article does not use them.

## Commands

```bash
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1
taskset -c 4 python studies/campaign_M_seeded_defects.py --seeds 20 --out studies/data_1.12.0
taskset -c 4 python studies/campaign_M2_multi_defect.py --seeds 5 --out studies/data_1.12.0
taskset -c 4 python studies/campaign_P_portability.py --out studies/data_1.12.0
taskset -c 4 python studies/profile_loader.py --json studies/data_1.12.0/P_profiles.json
taskset -c 4 python studies/sweep_settings.py --out studies/data_1.12.0
taskset -c 4 python studies/old_film/audit_vp_lineage.py --repos /path/to/clones \
    --bank /path/to/noise_data --out studies/old_film/certificates/released
taskset -c 4 python studies/old_film/repeat_draws.py --repos /path/to/clones \
    --bank /path/to/noise_data --draws 100 --seeds 5
python studies/old_film/documents/verify_documentary_audit.py
# DART runs in its own environment, with AbsoluteDegradation v1.0.0 on the path:
<DART>/.venv/bin/python studies/old_film/audit_dart.py
<DART>/.venv/bin/python studies/profile_loader.py --subject dart
```

The roots of the external projects are set in `campaign_P_portability.py`
(`SUBJECTS`); `audit_vp_lineage.py` expects clean clones of RTN, RRTN,
MambaOFR and MgfrOFR under `--repos`.
