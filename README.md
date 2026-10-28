> **Release 1.11.0 (2026-10-27).** Every run now describes what the loader
> delivered (`treatment_identity.profile`), and the certificate carries that
> profile next to the verdicts (schema 1.3; schema-1.2 certificates still
> validate). `content_scale` probes five source shapes (16:9, 4:3, 1:1, 3:4,
> 9:16) and tells the two axes apart by waveform after any colour conversion.
> All studies of the accompanying article were rerun with this release; their
> scripts, adapters and results are in [`studies/`](studies/README.md).

# treatment-identity

Verify that the treatment an experiment declares is the treatment its data
path delivers.

DDC runs a training loader on small synthetic fixtures through a short
adapter, without changing the loader. It always describes what the loader
delivered: sample shapes, values and channels, whether the input was read or
built, the frame window, the order of operations and the scale of the content.
When a contract declares what the experiment requires, it also compares the
delivery with each requirement and issues a verdict. Results and their
evidence are written to a treatment certificate.

## What it checks

| Gate | Executable assertion |
|---|---|
| `precomputed_input` | A configured pre-rendered input reaches loader output; an online generator sentinel fails if reached. |
| `temporal_window` | Delivered frame identities, window length and repeated sampling agree with the declared temporal contract. |
| `separability` | Exact unexpected equality fails; otherwise 16 matched seeds and a paired energy randomisation test evaluate the declared distributional threshold. |
| `target_transforms` | Observed target-side operations agree with the declared count and target applicability. |
| `operator_trace` | Realised operator order agrees with the fixed or random-permutation policy. |
| `value_range` | Values are finite, lie inside declared bounds and preserve required fixture anchors. |
| `channel_content` | Channel count and required per-channel signatures survive delivery. |
| `content_scale` | The content is cropped, rescaled evenly or reshaped, read from known spatial periods at five source shapes; rotations are reported. |
| `geometry` | Prediction/reference geometry satisfies the evaluation contract. |

These are bounded, contract-relative observations. Passing does not prove that
real data have correct labels, that normalisation statistics came from the
right population, or that optimisation is correct.

## Observation boundaries

Loader-driven gates observe `LoaderAdapter.sample()` and record
`boundary=loader_output`. Content assertions can also be installed immediately
after collation and before model invocation:

```python
from treatment_identity import (
    ContentContract,
    StreamContract,
    guard_training_step,
)

rgb01 = ContentContract(
    lq=StreamContract(
        value_range=(0.0, 1.0),
        channels=3,
        require_distinct_channels=True,
    )
)

@guard_training_step(contract=rgb01)
def model_step(batch):
    return model(batch.lq)
```

The guard checks the actual batch before delegating. A loader observation is
never relabelled as a pre-model observation. For non-standard batches, pass an
`extractor` that returns `Sample`.

## Content contracts

`StreamContract` supports:

- `value_range=(lo, hi)` and `require_finite=True`;
- `require_range_extrema=True` for purpose-built anchor fixtures;
- `channels=n`;
- `require_distinct_channels=True` for declared colour content;
- explicit numerical tolerances.

The tiled content fixture includes crop-stable all-zero/all-maximum pixels and
different patterns in all three channels. A silent `/255` conversion fails the
range anchors. Luminance replication fails pairwise channel separation.
Intended greyscale remains valid when one channel, or replicated luminance, is
what the contract actually declares.

## Source-clip escalation

The contract keeps three quantities separate:

- `num_frames`: frames one delivered sample must contain;
- `clip_length`: the source-clip length the subject declares it requires;
- `fixture_clip_length`: the length manufactured for one probe.

With no declared source requirement, a short-fixture abort triggers the ladder
`16, 32, 64, 100, 200`. If a longer fixture works, the gate reports `UNDECL`
with the shortest observed requirement. Escalation never writes the discovered
value back as a subject declaration. Campaign P enables this robust behaviour
by default; `--no-clip-escalation` is only a sensitivity mode.

## Separability design

The exact-equality discriminator is primary: arms declared distinct that agree
within tolerance in every matched draw fail without a p-value. When draws
differ, `distributional=True` computes a full-tensor energy statistic and a
paired randomisation p-value. A pass requires both the declared alpha and
minimum effect. A non-significant result means that the fixed design did not
establish separability; it is not proof of equal populations.

## Adapter surface

An adapter must build a subject loader and return one `Sample`:

```python
class MyAdapter:
    name = "project.Dataset"

    def build(self, spec):
        return Dataset(...)

    def sample(self, loader, index):
        item = loader[index]
        return Sample(lq=item["lq"], gt=item.get("gt"))
```

Optional methods expose the post-collate boundary or install a sentinel over
the online generator:

- `sample_training_step(loader, index)`;
- `disable_generator()`;
- `__len__()`.

## Run it

```bash
python3 -m unittest discover -s tests -v
python3 selftest.py
```

The studies of the article, with their commands and results, are listed in
[`studies/README.md`](studies/README.md).

## Environment reconstruction

`environment/` contains the exact Linux/x86-64 Conda artifact lock, exact pip
state, human-readable export, `usercustomize.py`, clean-build script and
logical-outcome verifier. The pip lock is an observed historical state and is
installed without dependency solving because the retained metadata is not
solver-consistent.

```bash
environment/create_and_validate.sh \
  /absolute/path/to/new/environment \
  /absolute/path/to/results
```

## Certificate status

- `PASS`: the applicable declaration matched delivery.
- `FAIL`: delivery contradicted a declaration.
- `UNDECL`: executable behaviour has no declaration to compare against.
- `N/A`: the contract explicitly makes the property inapplicable.
- `SKIP`: the adapter cannot expose the observation.

The profile is recorded whatever the statuses are: it says what was
delivered, and the statuses say whether that met the contract.

`ERROR` is intentionally not a gate status; it means no observation was
obtained. Certificates keep failed and undeclared gates separate.

## Citation

Use [CITATION.cff](CITATION.cff). The concept DOI
[10.5281/zenodo.21633585](https://doi.org/10.5281/zenodo.21633585) resolves to
the latest archived release.
