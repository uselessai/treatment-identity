# Studies

The studies of the article *The data-delivery checker: testing what data
loaders deliver in deep-learning experiments*, with
the scripts that ran them and their results. Every study was run with DDC
release 1.14.0, on one performance core of an Intel Core i9-12900K, with one
thread and no GPU (`taskset -c 4`, `OMP_NUM_THREADS=1`,
`CUDA_VISIBLE_DEVICES=`). External projects are not included; they are cloned
at the commits listed in the certificates and in `data_1.14.0/P_effort.csv`.

| Study | Question | Script | Results |
|---|---|---|---|
| 1, 2 | Does DDC detect introduced defects, without false alarms, and at what cost? | `campaign_M_seeded_defects.py`, `campaign_M2_multi_defect.py` | `data_1.14.0/M_*` |
| 3 | What does applying DDC to sixteen external loaders cost, and what does it report about them? | `campaign_P_portability.py`, `profile_loader.py`, `sweep_settings.py`, `adapters/` | `data_1.14.0/P_*` |
| 4 | Which defects appear in four released old-film restoration systems, and in DART? | `old_film/audit_vp_lineage.py`, `old_film/repeat_draws.py`, `old_film/audit_dart.py` | `old_film/certificates/` |
| 5 | How much of connecting DDC, and of its requirements, comes from each project's training command and configuration? | `wizard/run_trials.sh`, `wizard/trial.py`, `wizard/summarize.py` | `data_1.14.0/W_wizard.csv`, `data_1.14.0/W_heldout.csv`, `data_1.14.0/loaders/` |
| 4, follow-ups | DDColor image mixing | `campaign_S_ddcolor_cutmix.py`, `probe_ddcolor_cutmix.py` | printed |

`old_film/documents/` holds the documents inspected for Study 4 (publications,
README files at the audited commits, search log and decisions) and a verifier
that checks them against the certificates. `figures/` holds the scripts that
draw the article's figures from the released loaders. `history/` keeps the
results of earlier releases; the article does not use them.

## Commands

```bash
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1
taskset -c 4 python studies/campaign_M_seeded_defects.py --seeds 20 --out studies/data_1.14.0
taskset -c 4 python studies/campaign_M2_multi_defect.py --seeds 5 --out studies/data_1.14.0
taskset -c 4 python studies/campaign_P_portability.py --out studies/data_1.14.0
taskset -c 4 python studies/profile_loader.py --data studies/data_1.14.0
taskset -c 4 python studies/sweep_settings.py --out studies/data_1.14.0
taskset -c 4 python studies/old_film/audit_vp_lineage.py --repos /path/to/clones \
    --bank /path/to/noise_data --out studies/old_film/certificates/released
taskset -c 4 python studies/old_film/repeat_draws.py --repos /path/to/clones \
    --bank /path/to/noise_data --draws 100 --seeds 5
python studies/old_film/documents/verify_documentary_audit.py
# DART runs in its own environment, with AbsoluteDegradation v1.0.0 on the path:
<DART>/.venv/bin/python studies/old_film/audit_dart.py
<DART>/.venv/bin/python studies/profile_loader.py --subject dart --data studies/data_1.14.0
# Study 5: the wizard on the twenty-one loaders, each in its own environment, then
# each trial into its loader's folder, the summary, and the article's two tables
bash studies/wizard/run_trials.sh
python studies/wizard/summarize.py --data studies/data_1.14.0 \
    --cost-tex tab_cost17.tex --tex tab_wizard.tex
# Study 5, held out: the wizard, unchanged (wizard/heldout_frozen.sha256), on eight
# projects not used to develop it; their trials go to W_heldout.csv, apart from the
# twenty-one loaders
sha256sum -c studies/wizard/heldout_frozen.sha256
bash studies/wizard/run_heldout.sh
python studies/wizard/summarize.py --data studies/data_1.14.0 --heldout-tex tab_heldout.tex
```

The roots of the external projects are set in `campaign_P_portability.py`
(`SUBJECTS`); `audit_vp_lineage.py` expects clean clones of RTN, RRTN,
MambaOFR and MgfrOFR under `--repos`.

Results about one loader are kept once, in its folder of
`data_1.14.0/loaders/` (see `data_layout.py`): `profile.adapter.json`, taken
through its adapter (Studies 3 and 4); and from Study 5 `profile.wizard.json`,
`certificate.wizard.json` (the requirements the configuration states, the
configured values changed and the true/false settings switched),
`answers.wizard.json` and `questions.wizard.json` (every question and how it
was answered). `loaders/comparison.csv` sets the two profiles of each loader
side by side, field by field. `wizard/oracle/` holds the answers a person gave
in Study 5. `run_trials.sh` gives each project's training command and the
environment it runs in; packages an environment lacked were installed beside
it, in the folder that `ENVS` names.
