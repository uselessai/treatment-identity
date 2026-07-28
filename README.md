# treatment-identity

Verify that the treatment an experiment *declares* is the one its data loader
*delivers*.

Pinning a commit, publishing a configuration and fixing a seed constrain what a
pipeline **is**. None of them constrain what its loader **delivers to the
training step**. A
configuration can set a flag no loader reads; a loader can compute a sampling
window and discard it; a generator can be present, correct, and never reached.
None of these raise an error. The experiment runs to completion and returns a
full table of plausible numbers.

This package asserts the missing layer. Five gates at the loader boundary plus
one complementary evaluation-integrity gate, on CPU, in seconds, against
synthetic fixtures, before any GPU budget is committed.

```
declared  →  released  →  configured  →  delivered
  paper       commit        YAML         ← only this one trains the model
```

## Install

Python 3.10 or newer. Runtime dependencies are declared by the package
(`numpy`, `opencv-python`, and `jsonschema`).

```bash
python -m pip install .
```

## Run

```bash
python selftest.py                       # the suite must discriminate before you trust it
python -m unittest discover -s tests     # certificate and schema contract
python audit_vp_lineage.py \
    --repos /path/to/clones \
    --bank  /path/to/noise_data \
    --out   certificates/
bash tests/test_wheel_install.sh         # the wheel must work with this tree unreachable
```

The last one is not ceremony. A wheel can build cleanly and still declare a
console script for a module it never packaged; the only way to find out is to
install it under an isolated prefix and run the command from somewhere else.

`--working` audits `z`-prefixed working copies instead of pristine clones, so a
study can separate defects it inherited from defects it introduced.

## The gates

Five assert treatment delivery at the loader boundary. The sixth, `geometry`,
checks that prediction and reference entered a metric at the same shape, which
is evaluation integrity rather than treatment identity; it is kept here because
an experiment can be invalidated at either boundary, and the certificate records
which family a result belongs to.

| Gate | Question | Fixture |
|---|---|---|
| `precomputed_input` | Does the pre-computed input reach the model, or is it silently regenerated? | flat target + checkerboard input |
| `temporal_window` | Do the frames the sampler computed reach the file system? | clip where frame *i* has every pixel equal to *i* |
| `separability` | Are two treatments declared distinct distinguishable in a matched-seed realisation? | matched seeds, direct generator calls |
| `target_transforms` | How many times is the target transformed, against how many the **paper** declares? | instrumented call counting |
| `operator_trace` | Is the declared operator sampling policy the one that runs? | repeated draws, realised order |
| `geometry` | Does the spatial contract survive, or is a resize absorbed? | declared vs delivered shape |

The gates exercise bounded adapter, loader, or generator contracts.
`geometry` either executes a resize rule supplied by the adapter or asserts
against a shape observed elsewhere. The certificate records which source was
used, so an assertion over a recorded shape is not reported as an independent
measurement.

`expected` in `target_transforms` is read off the **publication**, not the code.
Passing the observed count would make the gate assert the implementation against
itself, and it could never fail.

For this lineage no publication mentions ground-truth sharpening at all, so the
gate is called with `expected=0, declared=False`. Those two arguments say
different things and the distinction is the point: `expected` is the count to
compare against, and `declared` records whether the publication addresses the
operation. Silence is *not* a declared count of zero — it is the absence of a
declaration — so one observed application is reported as `UNDECL` rather than
`FAIL`. The paper has not stated anything untrue; it has left the target
unspecified, and identity is blocked because there is nothing to check the
delivered tensor against.

Every gate returns `PASS`, `FAIL`, `UNDECL`, `SKIP` or `N/A`.

| Status | Meaning | Blocks identity |
|---|---|---|
| `PASS` | the delivered tensor matches the declaration | no |
| `FAIL` | the delivered tensor contradicts the declaration | yes |
| `UNDECL` | the code does something no publication declares | yes |
| `SKIP` | an applicable check did not execute | inconclusive |
| `N/A` | the property is not part of the declared contract | no |

The certificate summary is `INCONCLUSIVE` when no gate passes, `PARTIAL` when at
least one gate passes but another applicable gate is skipped, `PASS` when all
executed applicable gates pass, `UNDECLARED` when nothing is contradicted but
something is undeclared, and `FAIL` when any gate fails. A mixture of `PASS` and
`N/A` remains `PASS` because the `N/A` gates are not applicable.

## Reproducibility of the gates themselves

`LoaderSpec.seed` is applied, not merely recorded. `treatment_identity.seeding`
seeds Python, NumPy and — when present — PyTorch before a loader is built and
again before a probe series, and the certificate lists both the generators that
were seeded and the known ones that ignore those seeds (`albumentations`).

This exists because version 1.0 accepted the seed and never applied it. Loaders
with deterministic temporal selection were unaffected, but a loader that samples
its window reported a different frame coverage on every run of the same command
— 14 to 19 of 32 — and whichever draw was current is the number that reached a
figure. A certificate whose value depends on when it was generated is not a
certificate. Version 1.1 fixes it and reports coverage as what it is: exact when
every probe returns one window, a seeded lower bound otherwise.

### The gate that matters most

`separability` fails on **unexpected equality**. Conventional testing checks
that things which should be equal are; nothing checks that things which should
*differ* do. That asymmetry is why an experiment with an inert factor is
indistinguishable from a successful one:

```
[FAIL] separability[A_vs_B]: UNEXPECTED EQUALITY: two treatments declared
distinct produced tensors identical to within 1e-06 (max delta 0.000e+00).
The declared distinction is not observable under this probe; verify treatment
identity before interpreting a comparison.
```

## Auditing your own pipeline

Implement two methods:

```python
class MyAdapter:
    name = "my-pipeline"

    def build(self, spec: LoaderSpec):
        return MyDataset(gt=spec.gt_root, lq=spec.lq_root,
                         use_precomputed=spec.use_precomputed_lq,
                         num_frames=spec.num_frames, seed=spec.seed)

    def sample(self, loader, index) -> Sample:
        item = loader[index]
        return Sample(lq=item["lq"], gt=item["gt"],
                      frame_ids=item.get("frame_list"))
```

`Sample.frame_ids` is optional but worth providing: when present, the temporal
gate compares what the loader *claims* it selected against what it *delivered*,
and a mismatch is itself a finding.

`selftest.py` contains a 60-line reference loader that passes every gate, plus
one deliberately defective variant per divergence class. It is the executable
form of the contract.

## The certificate

Each run emits a JSON record of what was actually delivered — the branch
reached, the provenance and hashes of both streams, the modality, channels and
range, the observed operator order, the number of target transformations, the
count of unique frames reachable, training and render seeds kept apart, the
uncontrolled RNG sources, the environment, and the status of every gate.

The JSON Schema is distributed inside the wheel at
`treatment_identity/schemas/treatment-certificate-1.1.schema.json`.
`Certificate.write()` validates by default, and external documents can be
validated without writing files:

```python
import json
from pathlib import Path

from treatment_identity import load_certificate_schema, validate_certificate

document = json.loads(Path("certificate.json").read_text(encoding="utf-8"))
validate_certificate(document)  # raises jsonschema.ValidationError on failure
schema = load_certificate_schema()
```

The configuration and certificate serve different purposes: the former records
what was requested, while the latter records bounded observations about what
was delivered. Both should accompany a run.

## Reference results

Run against the RTN → RRTN → MambaOFR → MgfrOFR lineage of old-film restoration
(commits `54df5941`, `832495f2`, `0a53f6dc`, `0302b901`). Certificates in
`examples/certificates_released/` (published RTN, RRTN, MambaOFR, and MgfrOFR
code) and
`examples/certificates_working/` (the working copies of a study that trained on
them).

**Published code:**

| Gate | RTN | RRTN | MambaOFR | MgfrOFR |
|---|---|---|---|---|
| `precomputed_input` | N/A | N/A | N/A | N/A |
| `temporal_window` | **FAIL** 5/32 frames | **PASS** 17/32 | **FAIL** 5/32 | **FAIL** 5/32 |
| `target_transforms` | **UNDECL** 1x, declared 0 | **UNDECL** 1x | **UNDECL** 1x | **UNDECL** 1x |
| `operator_trace` | FAIL | FAIL | FAIL | FAIL |
| `separability` (each pair) | FAIL | FAIL | FAIL | PASS |
| `geometry` | FAIL | FAIL | FAIL | FAIL |

Reproduce with:

```bash
python audit_vp_lineage.py \
  --repos <clones> --bank <noise_data> --out examples/certificates_released
```

Read out:

- **`temporal_window`** — RTN and MambaOFR compute a window and then open the
  clip prefix, so a 32-frame clip yields 5 reachable frames. That count is
  exact: every probe returns the same window, so no further probe can widen it.
  RRTN passes — its source keeps the inherited line commented out immediately
  above the replacement — and its 17/32 is a different kind of number: the
  coverage eight probes reached **at seed 0**, a reproducible lower bound, not
  a guarantee that half of every training clip is sampled. Earlier releases of
  this protocol did not seed the loader, so that figure moved between 14 and 19
  across runs of the same command; `treatment_identity.seeding` exists because
  of that defect and the certificate now records the seed and the RNGs it could
  not reach. In the inspected 100-frame training clips, the inherited prefix
  path reaches 7 frames for RTN and 5 for MambaOFR.
- **`target_transforms`** — reported as `UNDECL`, not `FAIL`. The code sharpens
  the target once per frame and no publication in the lineage mentions it. A
  paper that is silent has not made a false statement, so the gate does not
  report one; what it reports is that the target cannot be reconstructed from
  the publication. Identity is blocked either way, which is why `UNDECL` counts
  as a divergence.
- **`separability`** — every pair among RTN, RRTN and MambaOFR reports a maximum
  delta of exactly `0.0` under the controlled matched-seed probe with the
  uncontrolled colour operator neutralised. This is evidence about the shared
  deterministic core and the tested realisation, not a distributional
  equivalence test. MgfrOFR passes because its refactor changes the order in
  which operators consume the shared random stream, yielding a different
  matched-seed realisation. That pass alone does not establish a different
  distribution.
- **`operator_trace`** — all three draw a random permutation of the operator
  menu, bind it, and index the unpermuted list. Twelve draws, one order:
  `blur→downsample→noise→jpeg`.
- **`precomputed_input`** — no published repository implements a pre-computed
  input branch at all, so the released trees report **N/A**, not FAIL: a feature
  nobody claimed cannot diverge from its claim. In the working copies, where the
  study's configurations do set the flag, two of the three honour it and pass;
  the third ignores it and its gate is red. That divergence belongs to the study,
  not upstream — which is the point of auditing both trees.
- **`geometry`** — the evaluation path raises a `320×180` clip to `640×368`
  rather than the aspect-preserving `640×360`, a 2.17% distortion, and the
  evaluator resizes the reference to match instead of failing.

## Scope

The gates provide bounded evidence about **reachability**, matched-seed
realisation separability, target transformations, operator order, temporal
delivery, and geometry. They do not by themselves validate dependency
probabilities, bank provenance, physical realism, optimisation correctness, or
distributional difference. Complementary checks are required for those claims.

## Licence

MIT.
