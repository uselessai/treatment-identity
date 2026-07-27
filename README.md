# treatment-identity

Verify that the treatment an experiment *declares* is the one its data loader
*delivers*.

Pinning a commit, publishing a configuration and fixing a seed constrain what a
pipeline **is**. None of them constrain what it **hands to the optimiser**. A
configuration can set a flag no loader reads; a loader can compute a sampling
window and discard it; a generator can be present, correct, and never reached.
None of these raise an error. The experiment runs to completion and returns a
full table of plausible numbers.

This package asserts the missing layer. Six gates, on CPU, in seconds, against
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

## The six gates

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
itself, and it could never fail. For this lineage the declared count of
ground-truth sharpening is zero, because no paper mentions it, so one observed
application is the finding.

Every gate returns `PASS`, `FAIL`, `SKIP` or `N/A`. Only `FAIL` records an
observed divergence. `N/A` means the property is not part of the declared
contract; `SKIP` means an applicable check was not executed. The certificate
summary is `INCONCLUSIVE` when no gate passes, `PARTIAL` when at least one gate
passes but another applicable gate is skipped, `PASS` when all executed
applicable gates pass, and `FAIL` when any gate fails. A mixture of `PASS` and
`N/A` remains `PASS` because the `N/A` gates are not applicable.

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
`treatment_identity/schemas/treatment-certificate-1.0.schema.json`.
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
| `temporal_window` | **FAIL** 5/32 frames | **PASS** 16/32 | **FAIL** 5/32 | **FAIL** 5/32 |
| `target_transforms` | **FAIL** 1x, declared 0 | **FAIL** 1x | **FAIL** 1x | **FAIL** 1x |
| `operator_trace` | FAIL | FAIL | FAIL | FAIL |
| `separability` (each pair) | FAIL | FAIL | FAIL | PASS |
| `geometry` | FAIL | FAIL | FAIL | FAIL |

Read out:

- **`temporal_window`** — RTN and MambaOFR compute a window and then open the
  clip prefix, so a 32-frame clip yields 5 reachable frames. RRTN passes: its
  source keeps the inherited line commented out immediately above the
  replacement. Its 16/32 value is coverage from eight fixture probes, not a
  guarantee that half of every training clip is sampled. In the inspected
  100-frame training clips, the inherited prefix path reaches 7 frames for RTN
  and 5 for MambaOFR.
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

## Relation to the study that produced it

This package was written to audit a specific lineage of old-film restoration
models, and the divergences it reports there are documented in the article that
introduces the protocol. The experiments, per-clip metrics, training
configurations and analysis scripts of that study live in a separate
repository — [old-film-degradation-audit](https://github.com/uselessai/old-film-degradation-audit)
— deliberately, so that this one stays small enough to install and read.

Nothing here is specific to film restoration. The adapter in `adapters/` is a
reference implementation for one repository family; the gates themselves know
only about loaders, tensors and fixtures.

## Citing

See `CITATION.cff`. Please cite the article as well as the software: the
software is the executable form of an argument the article makes.
