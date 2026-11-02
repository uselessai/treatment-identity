> **Release 1.13.0 (2026-11-01).** DDC connects to a loader through the
> project's own training command, without an adapter
> (`python -m treatment_identity.wizard`): it asks only what it cannot read from
> the configuration or the samples, judges the requirements the configuration
> states, and with `--follow-config` checks that the delivery follows each
> configured value and reports which true/false settings change it. A contract
> file adds, replaces or sets aside requirements. DDC runs on Python 3.8 and
> later. All studies of the accompanying article were rerun with this release,
> with a fifth study on connecting through the training command.
>
> **Release 1.12.0 (2026-10-29).** The profile, written on every run, also
> gives how many times each recorded operation runs per delivered frame and
> whether a stream's channels are identical. All studies of the accompanying
> article were rerun with this release; their scripts, adapters and results
> are in [`studies/`](studies/README.md).
>
> **Release 1.11.0 (2026-10-28).** Every run now describes what the loader
> delivered (`treatment_identity.profile`), and the certificate carries that
> profile next to the verdicts (schema 1.3; schema-1.2 certificates still
> validate). `content_scale` probes five source shapes (16:9, 4:3, 1:1, 3:4,
> 9:16) and tells the two axes apart by waveform after any colour conversion.

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

## Install

DDC needs Python 3.8 or later, and installs NumPy, OpenCV and jsonschema. It
runs on a CPU and trains nothing. Install it in the environment where the
project you want to check already runs, since DDC builds that project's loader:

```bash
pip install "git+https://github.com/uselessai/treatment-identity.git@v1.13.0"
treatment-identity-selftest        # should end with "SELF-TEST PASSED"
```

To work on DDC, or to rerun the studies of the article, install it from a
clone:

```bash
git clone https://github.com/uselessai/treatment-identity.git
cd treatment-identity
pip install -e .
python -m unittest discover -s tests
```

## Quick start

Write an adapter: two methods that build the project's loader on the fixtures
DDC supplies and return one sample from it.

```python
from pathlib import Path
from treatment_identity import Certificate, Sample, check_temporal_window
from my_project.data import TrainDataset          # the loader being checked

class MyAdapter:
    name = "my_project.TrainDataset"

    def build(self, spec):                         # spec.gt_root, spec.lq_root: fixture folders
        return TrainDataset(gt_root=spec.gt_root, lq_root=spec.lq_root,
                            num_frames=spec.num_frames)

    def sample(self, loader, index):
        item = loader[index]
        return Sample(lq=item["lq"], gt=item["gt"])

adapter = MyAdapter()
cert = Certificate("my_project", "windows of 5 frames sampled over the clip")
cert.add(check_temporal_window(adapter, Path("ddc_work")))   # a check against a requirement
cert.describe(adapter)                             # the profile: what the loader delivered
cert.write("certificate.json")
print(cert.summary())
```

The profile alone, with no requirement, is
`treatment_identity.profile.describe_loader(adapter)`. The adapters of the
article's sixteen loaders, in `studies/adapters/`, are worked examples.

## Ways to use DDC

| | What you write | What you get | Machine time per run |
|---|---|---|---|
| Adapter | a Python class with `build` and `sample` (40 to 195 lines for the article's loaders) | the profile; verdicts for the requirements you pass to each check | seconds: only the loader is built |
| Wizard | nothing but the training command; answers to the questions it cannot settle (0 to 4 for the article's loaders) | the profile; verdicts for the requirements the configuration states | tens of seconds to minutes: everything the command does before its loader runs |
| + contract | a YAML file of requirements (`--contract`) | verdicts for your requirements as well, replacing or setting aside configured ones | the same |
| + `--follow-config` | the flag | for every configured value: whether the delivery follows a changed value; for every true/false setting of the loader: whether switching it changes the delivery | one more run per setting |

The adapter is the fastest and the most complete: every check, including
separability and the evaluation geometry, takes requirements through it. The
wizard needs no code, and is slower because it runs the project's command up
to the loader, building the model, the trainer and whatever else comes first.

### The wizard

```bash
python -m treatment_identity.wizard --cwd /path/to/project --answers ddc.json \
    --profile profile.json --certificate certificate.json \
    -- python train.py -opt options/train.yml
```

It runs the command in-process until the training `DataLoader` is about to be
created. At that moment it keeps the dataset the project built from its own
configuration and stops the run, so training never starts; loaders of
validation sets are built as usual and the run continues to the training one.
Every image, video or ZIP archive the dataset then reads under its data
folders is served from the DDC fixture at read time, so no path, option name
or framework has to be translated, and none of the project's own pixels are
read. A dataset that loads all its data when it is built reads nothing later,
and the wizard reports that the fixtures cannot reach it.

Data folders, their layout, the input scale, the patch size and which entries
of an item are the input and the target are read from the configuration and
the item itself. What cannot be read there is asked in the terminal, with any
value the wizard proposes (Enter accepts it); answers are kept in the answers
file and the run continues. The next run asks nothing. Without a terminal the
run stops with exit status 2 at the first question, which is printed with its
proposal, until the answer is added to the file (`"detected"` accepts the
proposal).

| Answer | Meaning |
|---|---|
| `data_roots` | `{folder: "gt" or "lq"}`: the training data folders and the stream each holds |
| `layout:<folder>` | stand-in layout for a folder absent on this machine: `{"images": n}`, `{"clips": n, "frames": n}`, `{"zips": [...]}` or `{"videos": [...]}` |
| `lq_scale`, `min_frame_size` | how much smaller a stored input is; the patch the loader crops |
| `target_key`, `input_key` | entries of an item; for inpainting the input can be `{"masked_by": <mask entry>}` |
| `set` | values of the configuration named on the command line to change, in a copy |
| `set_in` | `{"<file>:<key>": value}`: a value of any configuration file, supplied when the run reads it |
| `files` | small files, such as a clip index, the run should read |
| `setup` | Python lines run before the command, for what the environment lacks |
| `observe` | `"module:function"` names whose calls the profile counts |

The wizard never deletes anything. Stand-in files for absent data folders,
configuration copies and whatever the training command creates as it starts
are listed in `<answers>.created.txt`, to be removed by hand.

### Requirements: the configuration and a contract

A training configuration states part of what the loader must deliver. The
wizard reads these values and judges each against the profile, naming the file
and key it comes from:

| Requirement | Read from keys such as | Compared with |
|---|---|---|
| `patch` | `gt_size`, `patch_size`, `crop_size`, `train_ps`, `out_size`, `H_size` | every delivered size, for six stored sizes |
| `short_side` | `resolution` | the shorter delivered side |
| `frames` | `num_frame`, `nframes` | frames per item |
| `scale` | `scale` | target size over input size |
| `rotation` | `use_rot` | rotated probes |
| `input` | `dataroot_lq` and similar | whether the stored input is read |

A contract adds what no configuration states, replaces a configured value, or
sets it aside:

```yaml
# contract.yml, passed with --contract contract.yml
from_config: true          # false: only the requirements below
ignore: [patch]            # configured requirements not to judge
requirements:
  frames: 5                # replaces the configured value
  window: moves            # 'moves' or 'fixed'
  input: stored            # 'stored' or 'made'
  content: crop            # 'crop', 'rescale' or 'reshape'
  value_range: [0, 1]
  channels: 3
  rotation: none           # 'none', 'present' or 'allowed'
  operator_order: random   # with functions named under `observe`
  operations: {sharpen: 1} # times per delivered frame
vary:                      # run again with each value, and compare
  --crop_size: [256, 128]  # an option of the command
  datasets.train.use_hflip: [true, false]
```

The same requirements can be judged on a profile from an adapter:
`treatment_identity.requirements.judge(profile, requirements)`.

### Does the code follow the configuration?

With `--follow-config` the wizard changes every configured value (halving the
patch and the number of frames, changing the scale, switching rotations), runs
again and checks that the delivery follows the new value. It also switches
every true/false setting of the loader's part of the configuration and reports
whether the delivery changes. A first run with nothing changed tells whether
the loader delivers the same values twice under the fixed seed; only then is
"changes nothing in the delivery" reported, and otherwise "changes nothing the
profile measures". Changed configurations are supplied to the project when it
reads them; no file on disk is modified.

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

## Studies of the article

The studies of the article, with their scripts, results and the command that
reruns all of them, are listed in [`studies/README.md`](studies/README.md).

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
the latest archived release; release 1.12.0 is archived at
[10.5281/zenodo.23770357](https://doi.org/10.5281/zenodo.23770357).
