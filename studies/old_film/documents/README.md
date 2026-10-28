# Documentary audit for `UNDECL`

This directory makes the manuscript's documentary result reproducible. It
freezes the publication and repository README inspected for each audited
system, records their hashes and versions, defines the coding rule, and keeps
the search and adjudication log separate from the executable certificate.

The question is deliberately narrow:

> Does the recorded publication/README corpus explicitly state that the
> training loader sharpens the learning target once per returned target frame?

The executable certificates observe one target-side `sharpen` call per target
frame. A documentary decision of `UNDECL` means that the recorded corpus does
not explicitly declare that operation. It does not mean that no document
anywhere mentions sharpness, nor that the authors denied the operation.

## Files

- `corpus/`: exact PDFs, README snapshots from the audited commits, and
  deterministic `pdftotext -layout` extractions.
- `SHA256SUMS`: integrity manifest for the frozen corpus.
- `documents.csv`: document identity, version, source and inspected sections.
- `search_log.csv`: literal/contextual search counts and manual disposition.
- `decisions.csv`: one decision per system tied to executable evidence.
- `verify_documentary_audit.py`: checks all hashes and the coded search counts.

Run:

```bash
python documentary_audit/verify_documentary_audit.py
```

## Review method and remaining limit

Pass 1 used case-insensitive literal searches for `sharpen*`, `unsharp*`,
`target`, `ground truth`, `GT`, `training data`, `data generation`,
`degradation pipeline` and `augmentation`. Pass 2 manually inspected every hit
for `sharpen*`/`unsharp*` and the publication sections describing training-data
generation, plus the training/data portions of each README.

The two passes use different evidence paths (automated term/context search and
manual section reading), but they were not coded by two independent human
readers. `decisions.csv` records that limitation explicitly. A later human
replication can add `coder_2` without changing the frozen corpus.

The RTN PDF is the Computer Vision Foundation open-access version. The other
three PDFs were already retained locally from the official paper/repository
materials. README files were exported with `git show HEAD:README.md` at the
commits recorded in `documents.csv`, so uncommitted working-tree changes are
not part of this corpus.
