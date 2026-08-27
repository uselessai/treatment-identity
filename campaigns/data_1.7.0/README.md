# Evidence for treatment-identity 1.7.0

This directory adds two campaigns to the evidence of 1.6.0, which remains valid
and unchanged. Both are reported in the SQJ manuscript.

## Campaign C --- what established tooling reports

Five established tools and one hand-written baseline, against three defects the
mechanism reports on the audited codebases:

- **D1** a declared random operator order is never delivered
- **D2** a computed sampling window is discarded and a fixed position is opened
- **D3** two conditions declared distinct deliver identical data

Findings, in `C_static_analysis.csv`, `C_tool_comparison.json` and
`C_experiment_tracking.json`:

- `ruff` in its default configuration returns **447** diagnostics over the four
  codebases and none is one of these defects; the rule that reaches D1 and D2
  (`B007`) is not enabled by default.
- `pylint` reports D1 and D2 without configuration, inside **2,297**
  diagnostics of which **139** carry the same `unused-variable` message. About
  ten of those 139 are the defect.
- Deepchecks reports no failed condition on either delivery alone, and none when
  both are supplied to its comparison suite.
- Evidently reports drift in **0 of 7** columns.
- MLflow records two runs whose every field differs --- including a declared
  channel count of three against one --- while the delivered digests are equal.
- **A hand-written five-line assertion does detect D3.** This is the strictest
  baseline available and it is recorded as passing. The mechanism's contribution
  is not that the check is impossible otherwise, but that writing it requires
  knowing it is worth making and which two conditions to compare, and that in
  the four audited codebases nobody wrote it.

Static analysis needs pinned clones, which are not bundled. Recorded deliveries
are in `deliveries/`.

## Campaign T --- reach beyond image and video

A synthetic multivariate time-series loader (32 steps, 6 sensors) carrying the
same two execution-path defects. In `T_modality.json`:

- `separability` and the sample-level forms of `value_range` and
  `channel_content` ran **unmodified**; both seeded defects were reported.
- `operator_trace` transferred in logic but not in its fixture driver, which
  writes image files.
- `temporal_window` and `geometry` remain modality-bound.

The loader and the defects are ours, as in campaign M. This establishes that the
predicates can be evaluated on a non-image modality, not detection over defects
written by other people.

## Verify

```bash
python3 campaigns/data_1.7.0/verify_evidence.py
```

## Reproduce

```bash
python3 campaigns/campaign_T_modality.py --out campaigns/data_1.7.0
python3 campaigns/campaign_C_tool_comparison.py \
    --repos /path/to/pinned/clones \
    --tensors campaigns/data_1.7.0/deliveries \
    --out campaigns/data_1.7.0
```

Tool versions recorded in `C_tool_comparison.json`: `ruff` 0.15.1,
`pylint` 4.0.7, Deepchecks 0.19.1, Evidently 0.7.21, MLflow 3.15.2.
