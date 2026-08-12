# treatment-identity

Verify that the treatment an experiment *declares* is the one its data loader
*delivers*.

Pinning a commit, publishing a configuration and fixing a seed constrain what a
pipeline **is**. None of them constrain what its loader **returns at the
boundary where the training step receives a sample**. A configuration can set
a flag no loader reads; a loader can compute a sampling window and discard it;
a generator can be present, correct, and never reached. None of these raise an
error. The experiment runs to completion and returns a full table of plausible
numbers.

This package probes that missing layer. Five treatment-delivery gates run at
the loader/training-step-input boundary. One separate geometry gate checks
evaluation integrity. All six checks run on CPU, in seconds, against synthetic
fixtures, before any GPU budget is committed.

```
declared  →  released  →  configured  →  loader output  →  training step
  paper       commit        YAML              ↑
                                      delivery is attested here
```

The certificate therefore describes bounded observations of the loader output
and evaluation geometry. It does **not** prove that an optimiser consumed a
particular tensor or that the optimisation procedure was correct.

## Release status

This tree is release **1.5.0**. Version 1.4.2 remains the archived release
carrying the version DOI associated with the previously frozen tables; those
tables are not superseded by anything here. The notes below are cumulative, newest
last, and every entry after 1.2.0 came out of using the mechanism on a subject
it was not written for.

**1.5.0 closes the remaining target-presence branch and tightens the
certificate contract.** A contract with `has_target=false` still makes
`target_transforms` inapplicable and returns `N/A`. A contract with
`has_target=true` whose loader delivers `gt=None` now returns `FAIL`, because
the delivered sample contradicts an explicit precondition. The three-case
regression matrix pins target-free, required-but-missing, and required-present
behaviour.

Certificate schema 1.2 also makes `undeclared_gates` required. Consumers can
therefore distinguish an empty list from an older producer that did not emit
the policy-relevant field. Schema 1.1 certificates remain archived evidence;
new certificates use `treatment-certificate/1.2`.

**Clip length is now part of the contract, and an undeclared one is a finding.**
`LoaderSpec.clip_length` declares the length of a *source clip* the loader
requires --- a different claim from `num_frames`, which is the window one sample
contains. It is optional, and `None` is the ordinary case: most loaders accept
whatever length they are handed.

What matters is what happens when it is `None` and the loader needs a length
anyway. Before 1.3.0 the gate raised out of the loader, the harness recorded
that the subject could not be audited, and nobody learned anything. Now
`with_clip_escalation` retries along a ladder (16, 32, 64, 100, 200) and, if a
longer clip makes the gate run, reports **UNDECL** with the shortest length that
worked --- because a loader that requires 100-frame clips and never says so has
a requirement its interface does not express, which is the property this package
exists to detect.

It does what it was built to do. On BasicSR's REDS loader, which
`basicsr/data/reds_dataset.py` writes against a hardcoded clip length of 100,
three gates used to abort with `FileNotFoundError`. They now report `UNDECL`,
and two of them recover **exactly 100** by probing, without reading the source.

**1.3.1 fixes the defect 1.3.0 reported.** Three of the four adapter-driven
gates never seeded their probe series --- only `check_temporal_window` did ---
so on a subject that consumes global randomness they were not reproducible:
`check_precomputed_input` returned `PASS`, 32 and 64 across five runs of the
same command. All four now seed from the contract's seed, and the consequences
are worth stating because they are not cosmetic:

* The escalation became reproducible. All four gates now recover **exactly
  100** on BasicSR's REDS loader, five runs out of five, which is the constant
  its source hardcodes. Before the fix they recovered 32, 64, 100 and 100.
* Without escalation, that subject's fourth gate now **aborts like the other
  three**. It used to pass. That pass was a coin flip on an unseeded probe, not
  a property of the loader, and any study that reported it reported a cell a
  reader would not reproduce.

**1.4.1 adds the subject that was chosen for a gate rather than for a count.**
Study 3 now runs seven loaders from six projects. The seventh is MMagic
(OpenMMLab): a different ecosystem again, with its own registry and its own
transform pipeline, and the only external subject besides KAIR's recurrent
loader with a temporal dimension. `temporal_window` had been exercised
outside the audited family on one subject; it is now exercised on two.

It also samples its window in a *pipeline stage* rather than inside the dataset,
which tests an assumption the gate makes -- that the loader decides the window --
against a design that puts the decision somewhere else. It still delivers
something the gate can read.

`operator_trace` did not move, and adding four subjects since has not moved it.
The pattern it targets has two well-known origins and one of them is already a
subject here, so we say the gate rests on one external example rather than
filling the column.

Three of the seven do not run as published, by three unrelated mechanisms: a
deleted dependency function, an untracked generated file, and a version
assertion that excludes the installed version exactly. Each is reconstructed in
its adapter and recorded in the certificate.

**1.4.2 repairs the second pass this package is on record as granting wrongly,
and a test that was checking its own staleness.**

`Sample.gt` may now be `None`, and `LoaderSpec.has_target` declares a
zero-reference arm. Before that there was no way for an adapter to say that a
subject has no target, so the Zero-DCE adapter --- the one subject in Study 3
chosen precisely *because* it has no ground truth --- reported its single
delivered tensor as both input and target. `check_target_transforms` then
compared zero observed transformations of that fabricated target against a
declared zero, agreed, and returned `PASS`. The gate was not wrong about what it
observed. It was wrong about what the observation was worth, which is what
happens when a gate asks whether a property was *declared* before asking whether
it *applies*.

The gate now asks applicability first and answers `N/A` for a zero-reference
arm. Release 1.4.2 incorrectly answered `N/A` again when a contract claimed a
target and the loader delivered none; 1.5.0 corrects that contradiction to
`FAIL`. Four regression tests in `tests/test_absent_target.py` pin all of it, including
the case that matters most: a real target is still asserted over, so the repair
did not buy silence in place of an unearned pass. `campaigns/data/` is
regenerated under 1.4.2 and Zero-DCE's `target_transforms` cell moves from
`PASS` to `N/A`.

`tests/test_wheel_install.sh` asserted `Version: 1.1.1` in three places while
the package was 1.4.1, so it failed against every version it shipped in and what
it reported was its own staleness. It now reads the version from
`treatment_identity/_version.py`. A check that pins a constant beside a thing
that moves is this package's own subject, one layer down.

**1.4.0 doubles the portability base, and says what that did not buy.** Study 3
now runs six loaders from five projects outside the audited family --- KAIR's
recurrent video and blind super-resolution loaders, Uformer, MPRNet, Zero-DCE
and BasicSR's REDS loader --- all third-party, public and pinned by commit.

The subject that used to sit there was withdrawn. It was written by one of us
for an unrelated project of our own, in a private repository with no
accompanying publication, inside a study that asks whether the mechanism travels
to code its authors did not write. It could not answer that question.

Choosing replacements had one hard filter: a candidate that bundles BasicSR is
not independent of the BasicSR subject already present. That ruled out two
otherwise ideal transformers. Zero-DCE was chosen for the opposite reason --- it
is zero-reference, so it has **no target at all**, and every other subject here
manufactures an input from one. It is the subject the contract vocabulary might
not fit.

Three findings came out of adapting them, none of them line counts:

* MPRNet reads `target/` beside `input/`; Uformer reads `groundtruth/`. The
  conventions look interchangeable and neither repository states which it uses.
* Uformer and KAIR both ship a module called `utils`. The campaign imports every
  adapter into one process, so the first shadows the second, which then fails on
  a name that exists in its own tree. No single-subject run produces this.
* Uformer imports a sorting package absent from our environment --- but it
  declares that package in its `requirements.txt`. Nothing is missing from the
  repository; what was missing was our having installed what it asks for. The
  adapter reconstructs it so the audited environment stays fixed across
  subjects. Two subjects of six, not three, fail to run as published.

What breadth did not buy: `operator_trace` is still exercised outside the
audited family on one subject, because none of the three new ones composes an
operator sequence.

**1.3.3 makes the fixture seed reach the subject, and repairs a mutant that did
not do what it said.** Two defects, both found by an external audit of the
campaign scripts rather than by the gates:

* Three of the four delivery gates built their `LoaderSpec` with a hardcoded
  `seed=0`. A campaign that set the subject's generator directly had that
  assignment destroyed on the next line, because a subject re-seeds itself from
  the spec it is handed. Twenty labelled fixture seeds were twenty repetitions
  of seed zero; the CSV changed the label and not the experiment. Every gate now
  takes a seed and passes it on, and two regression tests assert both that the
  seed arrives and that two different seeds produce different series.
* The `value_range` mutant cast `uint8` to `float32` and left the values alone,
  so it changed the dtype and not the range. It nevertheless produced an alarm,
  from a heuristic in the fixture decoder rather than from the defect. The
  mutant now returns `[0,1]` where the contract is in level units, which is what
  it was always described as doing, and it is a plain miss: no gate asserts
  value range. Collateral firing over the twelve variants is zero.

Rebuilding that mutant exposed something worth knowing about the fixtures: the
temporal decoder rescales a floating-point tensor by 255 when its maximum is at
most one, on the assumption that such a tensor is normalised. A legitimate
frame 1 in level units is indistinguishable from a normalised frame 255 under
that assumption, and nothing raises.

`CITATION.cff` also declared `cff-version: 1.3.2`. That field is the version of
the Citation File Format standard, not of this software; it is `1.2.0`. It was
broken here on 10 August 2026 by a substring replacement that matched inside
`cff-version`, and three releases shipped it.

**1.3.2 repairs the pass this package is on record as granting wrongly.**
`check_temporal_window` asserted *which* frames arrived and never *how many*, so
a single-image loader returning one frame where the contract asked for five was
accepted: eight one-frame samples are eight distinct windows, and the coverage
figure came out fine. The gate now checks the delivered window length against
the declared one, and the two outcomes are deliberately different statuses --- a
contradicted declaration is `FAIL`, while a contract that declares no window at
all is `N/A`, because the loader never claimed the branch under test. Two
regression tests pin both.

This is reported in the article as a limitation found by the portability study
and left unrepaired at the time of measurement. The measurement stands; the
software does not ship the defect.

The gates are one harness with one switch, not two harnesses:
`campaign_P_portability.py --clip-escalation` turns the ladder on, it is off by
default, and the mode is recorded in `P_effort.csv`. A number whose meaning
depends on a flag has to carry the flag.

`campaigns/data/` was originally frozen under 1.4.2. The release-candidate
campaigns are regenerated separately so their comparison with the archived
tables is explicit; a new archival DOI is required before calling 1.5.0 final.

### A note on tag names, because this package is about exactly this

Tags up to now did not carry the package version: `v0.1.0` contains package
1.0.2, and `v0.1.1` contains package **1.1.0** --- the version the currently
cited [doi:10.5281/zenodo.21647598](https://doi.org/10.5281/zenodo.21647598)
resolves to. Somebody looking for a tag named after the version they were told
to cite would not have found one. From this release the tag is the version:
`v1.2.0` contains package 1.2.0, and it will stay that way. The earlier tags
are left as they are, since rewriting a published tag is worse than documenting
it.

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

## The protocol checks

Five assert treatment delivery at the loader/training-step-input boundary. The
sixth, `geometry`, checks that prediction and reference entered a metric at the
same shape. It is an evaluation-integrity check rather than a treatment-identity
gate; it is kept here because an experiment can be invalidated at either
boundary. It remains explicitly identifiable in a certificate by the
`geometry` gate name and its shape evidence.

| Gate | Question | Fixture |
|---|---|---|
| `precomputed_input` | Does the loader return the pre-computed input, or silently regenerate it? | flat target + checkerboard input |
| `temporal_window` | Do the frames the sampler computed reach the file system? | clip where frame *i* has every pixel equal to *i* |
| `separability` | Are two treatments declared distinct distinguishable in a matched-seed realisation? | matched seeds, direct generator calls |
| `target_transforms` | How many times is the target transformed, against how many the **paper** declares? | instrumented call counting |
| `operator_trace` | Is the declared operator sampling policy the one that runs? | repeated draws, realised order |
| `geometry` | Does the spatial contract survive, or is a resize absorbed? | declared vs delivered shape |

The five treatment-delivery gates exercise bounded adapter, loader, or generator
contracts. `geometry` either executes a resize rule supplied by the adapter or
asserts against a shape observed elsewhere. The certificate records which
source was used, so an assertion over a recorded shape is not reported as an
independent measurement.

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

Every protocol check returns `PASS`, `FAIL`, `UNDECL`, `SKIP` or `N/A`.

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

`selftest.py` contains a 60-line reference loader that passes the delivery
checks, one deliberately defective variant per delivery-divergence class, and
the complementary geometry check. It is the executable form of the contract.

## The certificate

Each run emits a JSON record of bounded observations at the loader output — the
branch reached, the provenance and hashes of both streams, the modality,
channels and range, the observed operator order, the number of target
transformations, the count of unique frames reachable, training and render
seeds kept apart, the uncontrolled RNG sources, the environment, and the
status of every check. The separate `geometry` record describes the evaluation
boundary.

The current JSON Schema is distributed inside the wheel at
`treatment_identity/schemas/treatment-certificate-1.2.schema.json`.
The historical 1.1 schema is retained beside it so deposited 1.1 certificates
remain independently validatable; new writes use 1.2.
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
(commits `54df5941`, `832495f2`, `0a53f6dc`, `0302b901`). The schema 1.1
certificates that accompanied archived release 1.4.2 remain in
`examples/certificates_released/` and `examples/certificates_working/`.
Release 1.5.0 writes its separately regenerated schema 1.2 evidence under
`examples/certificates_1.5.0/{released,working}/`; its verifier checks eight
hashes, schema validity and logical continuity with the historical documents.

**Published code:**

| Gate | RTN | RRTN | MambaOFR | MgfrOFR |
|---|---|---|---|---|
| `precomputed_input` | N/A | N/A | N/A | N/A |
| `temporal_window` | **FAIL** 5/32 frames | **PASS** 17/32 | **FAIL** 5/32 | **FAIL** 5/32 |
| `target_transforms` | **UNDECL** 1x observed, none declared | **UNDECL** 1x observed | **UNDECL** 1x observed | **UNDECL** 1x observed |
| `operator_trace` | FAIL | FAIL | FAIL | FAIL |
| `separability` (each pair) | FAIL | FAIL | FAIL | PASS |
| `geometry` | FAIL | FAIL | FAIL | FAIL |

Reproduce with:

```bash
python audit_vp_lineage.py \
  --repos <clones> --bank <noise_data> \
  --out examples/certificates_1.5.0/released
python examples/certificates_1.5.0/verify_certificates.py
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

The five treatment-delivery gates provide bounded evidence about
**reachability**, matched-seed realisation separability, target transformations,
operator order, and temporal delivery. The separate geometry check provides
bounded evidence about the evaluation shape contract. These checks do not by
themselves validate dependency probabilities, bank provenance, physical
realism, optimiser input or behaviour, optimisation correctness, or
distributional difference. Complementary checks are required for those claims.

## Licence

MIT.
