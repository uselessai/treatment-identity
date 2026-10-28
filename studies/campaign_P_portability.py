#!/usr/bin/env python
"""Campaign P --- portability of the treatment-identity gates outside the lineage.

Study 3 of the Software Quality Journal manuscript. The four codebases audited
in the article share a common ancestor: they are one family, not four samples.
This campaign reports eight measurements from one training loader against
*independently developed* projects: six contract-relative assertion gates plus
content-scale and sample-shape diagnostics. Separability has a paired campaign;
evaluation geometry is not inferred from a training sample.

Reporting rule, stated up front because it decides how the result reads: a run
in which every external loader PASSES is a result, not a failure of the study.
It bounds the population in which the audited failures are concentrated, and it
is a stronger statement than "we only looked at one family".

Response variables per subject:
    gate outcomes            what each gate returns
    adapter_loc              non-blank, non-comment lines of the adapter
    adapter_seconds          one sequential observation of module import and
                             adapter construction. It includes cache and order
                             effects and is not human adaptation time.
    gates_seconds            median wall clock of the gate suite over
                             --repeats timings (min and max also recorded)
    gates_applicable         gates that ran, versus those the subject cannot support

Outputs (under ``revistas/claude/data/``):
    P_portability_matrix.csv     subject x gate
    P_effort.csv                 one row per subject
    P_summary.json

The default outputs retain all eight reported columns. The last column is
called ``sample_shape``: an earlier result file called it ``geometry``, but it
observes a training sample rather than a prediction/reference pair at the
evaluation boundary. Options to omit either diagnostic write suffixed files so
they cannot overwrite the complete profile.

Usage:
    python campaign_P_portability.py [--subject NAME] [--out DIR]
    python campaign_P_portability.py --list

CPU only. Requires the subject projects to be present locally; subjects that
are absent are reported as ``UNAVAILABLE`` rather than silently skipped.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import tempfile
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _locate_package() -> None:
    """Put ``treatment_identity`` on the path from wherever this script lives.

    The previous version resolved it by counting parent directories, which
    works from exactly one location. This script is deposited alongside the
    package as well as run from the manuscript tree, and a deposited script
    that cannot import its own package is the defect this campaign is written
    to detect -- lesson L5, in the artefact that reports it.
    """
    try:
        import treatment_identity  # noqa: F401
        return
    except ModuleNotFoundError:
        pass
    for base in (HERE, *HERE.parents):
        for cand in (base, base / "treatment_identity"):
            if (cand / "treatment_identity" / "__init__.py").exists():
                sys.path.insert(0, str(cand))
                return
    raise SystemExit(
        "treatment_identity not found: install it (pip install -e .) or run "
        "this script from a tree that contains the package")


_locate_package()


def _report_resolved_package() -> None:
    """Say which copy of the mechanism is about to run, and its version.

    A working tree can hold more than one copy of this package --- a deposit,
    a checkout, a snapshot beside a manuscript --- and the search above returns
    the first one it finds, which depends on where the script was invoked from.
    Running the same command from two directories can therefore produce two
    different tables with nothing to say so, which is this campaign's own
    subject matter applied to the campaign. One printed line makes the
    resolution observable instead of implicit.
    """
    import treatment_identity as _ti
    print(f"mechanism: {_ti.__version__} from {Path(_ti.__file__).parent}")


# Called from main(), not at import time: this module is also imported by the
# manuscript's number checker for its LOC metric, and a campaign banner printed
# in the middle of a verification report is noise.

# The adapters sit beside this script in the manuscript tree and under
# ``adapters/`` in the deposit. Both are offered; neither is required to exist.
sys.path.insert(0, str(HERE))
if (HERE / "adapters").is_dir():
    sys.path.insert(0, str(HERE / "adapters"))


def _default_out() -> Path:
    return HERE / "data" if (HERE / "data").is_dir() else HERE.parent / "data"

from treatment_identity import (ContentContract, StreamContract,    # noqa: E402
                                check_channel_content,
                                check_operator_trace,
                                check_precomputed_input,
                                check_target_transforms,
                                check_temporal_window,
                                check_content_scale,
                                measure_content_scale,
                                check_value_range,
                                with_clip_escalation)
import numpy as np                                                 # noqa: E402
from treatment_identity.checks import CheckResult, SKIP        # noqa: E402
from treatment_identity.adapter import CallRecorder, LoaderSpec                # noqa: E402


def _rgb_contract(value_range: tuple[float, float], *,
                  lq_extrema: bool = True,
                  gt_extrema: bool = True,
                  has_target: bool = True) -> ContentContract:
    return ContentContract(
        lq=StreamContract(value_range=value_range,
                          require_range_extrema=lq_extrema,
                          channels=3, require_distinct_channels=True),
        gt=(StreamContract(value_range=value_range,
                           require_range_extrema=gt_extrema,
                           channels=3, require_distinct_channels=True)
            if has_target else None),
    )


RGB_255 = _rgb_contract((0.0, 255.0))
RGB_01 = _rgb_contract((0.0, 1.0))

# ---------------------------------------------------------------------------
# Subjects. Each entry: (module, class, root that must exist, one-line note)
# ---------------------------------------------------------------------------
SUBJECTS = {
    "kair_video": dict(
        # KAIR options/rvrt/001_train_rvrt_videosr_bi_reds_30frames.json gt_size
        declared_shape=(256, 256),
        module="adapters_kair_video",
        cls="KairVideoTrainAdapter",
        root=Path("/home/laura/02ImproveData/zKAIR"),
        note="KAIR VideoRecurrentTrainDataset; recurrent video training loader, "
             "unrelated project, own meta-info convention",
        # Stated explicitly: the historical campaign evaluated this subject
        # against the harness default of zero, which is the same contract.
        expected_transforms=0,
        content=RGB_255,
    ),
    "kair_blindsr": dict(
        # KAIR options/train_bsrgan_x4_psnr.json H_size
        declared_shape=(320, 320),
        module="adapters_kair_blindsr",
        cls="KairBlindSRAdapter",
        root=Path("/home/laura/02ImproveData/zKAIR"),
        note="KAIR DatasetBlindSR; the BSRGAN degradation pipeline, which "
             "composes a randomised operator sequence -- the subject that "
             "exercises operator_trace",
        declared_policy="random_permutation",
        expected_transforms=0,
        precomputed=False,
        # Single-image pipeline: it has no temporal window, and saying so
        # in the contract is what lets temporal_window report N/A instead of
        # being handed a five-frame declaration it cannot meet.
        num_frames=1,
        content=_rgb_contract((0.0, 255.0), lq_extrema=False),
    ),
    # "llie_singleframe" was here and has been removed. It was a loader the
    # first author wrote for an unrelated project of her own, held in a private
    # repository with no accompanying publication. It sat in a study whose
    # question is whether the mechanism travels to code its authors did not
    # write, and it was neither third-party code nor reproducible by a reader.
    # A portability subject has to be somebody else's, public, and pinned.
    "uformer": dict(
        # Uformer options.py --train_ps
        declared_shape=(128, 128),
        module="adapters_uformer",
        cls="UformerAdapter",
        root=Path("/home/laura/02ImproveData/zUformer"),
        note="Uformer DataLoaderTrain; paired pre-rendered restoration loader "
             "from an unrelated group, with no BasicSR dependency",
        declared_policy="fixed",
        expected_transforms=0,
        num_frames=1,
        content=RGB_01,
    ),
    "mprnet": dict(
        # MPRNet Denoising/training.yml TRAIN_PS
        declared_shape=(128, 128),
        module="adapters_mprnet",
        cls="MprnetAdapter",
        root=Path("/home/laura/02ImproveData/zMPRNet"),
        note="MPRNet DataLoaderTrain (denoising); a second, independent "
             "codebase that arrived at the same paired input convention",
        declared_policy="fixed",
        expected_transforms=0,
        num_frames=1,
        content=RGB_01,
    ),
    "zerodce": dict(
        # The loader hardcodes self.size = 256 and resizes to a square.
        declared_shape=(256, 256),
        module="adapters_zerodce",
        cls="ZeroDceAdapter",
        root=Path("/home/laura/02ImproveData/zZero-DCE"),
        note="Zero-DCE lowlight_loader; zero-reference, so it has NO target at "
             "all -- the subject the contract vocabulary may not fit",
        declared_policy="fixed",
        expected_transforms=0,
        num_frames=1,
        # The vocabulary did not fit, and this is the field that admits it.
        # Without it the adapter had to fabricate a target, and the
        # target-dependent gate passed on the absence of what it checks.
        has_target=False,
        content=_rgb_contract((0.0, 1.0), lq_extrema=False,
                              has_target=False),
    ),
    "mmagic": dict(
        module="adapters_mmagic",
        cls="MmagicFramesAdapter",
        root=Path("/home/laura/02ImproveData/zmmagic"),
        note="MMagic BasicFramesDataset; OpenMMLab, a different ecosystem with "
             "its own registry and pipeline. Added to exercise temporal_window "
             "outside the audited family on more than one subject",
        declared_policy="fixed",
        expected_transforms=0,
        num_frames=5,
        content=RGB_255,
    ),
    "pix2pix_colorization": dict(
        # pix2pix options/base_options.py --crop_size
        declared_shape=(256, 256),
        module="adapters_pix2pix_colorization",
        cls="Pix2pixColorizationAdapter",
        root=Path("/home/laura/02ImproveData/zpix2pix"),
        note="pix2pix ColorizationDataset; the second external subject that "
             "MANUFACTURES its input online from the target rather than "
             "reading a pre-rendered pair, and the only one outside the "
             "BasicSR/KAIR lineages. Also a different task: colourisation",
        declared_policy="fixed",
        expected_transforms=0,
        num_frames=1,
        precomputed=False,
        content=ContentContract(
            lq=StreamContract(value_range=(-1.0, 1.0), channels=1),
            gt=StreamContract(value_range=(-1.0, 1.0), channels=2,
                              require_distinct_channels=True)),
    ),
    "restormer": dict(
        # Restormer Deraining/Options/Deraining_Restormer.yml gt_size
        declared_shape=(384, 384),
        module="adapters_restormer",
        cls="RestormerAdapter",
        root=Path("/home/laura/02ImproveData/zRestormer"),
        note="Restormer Dataset_PairedImage; a vendored BasicSR whose package "
             "__init__ the repository does not track, so a clean clone imports "
             "a different installed basicsr instead",
        declared_policy="fixed",
        expected_transforms=0,
        num_frames=1,
        content=RGB_255,
    ),
    "ddcolor_lab": dict(
        declared_shape=(256, 256),
        module="adapters_ddcolor_lab",
        cls="DDColorLabAdapter",
        root=Path("/home/laura/02ImproveData/zDDColor"),
        note="DDColor LabDataset; a colourisation loader that manufactures its "
             "input from the target and exposes CutMix and FMix as treatment "
             "switches it ships turned off",
        # Like pix2pix, it manufactures its input from the target rather than
        # reading a pre-rendered pair, so the pre-computed branch is one this
        # subject never declares.
        precomputed=False,
        declared_policy="fixed",
        expected_transforms=0,
        num_frames=1,
        # Lab, not RGB: one luminance channel in and a two-channel ab target
        # out. Handing this subject an RGB contract would fail it for not being
        # something it never claimed to be.
        content=ContentContract(
            lq=StreamContract(value_range=(0.0, 100.0),
                              require_range_extrema=False, channels=1,
                              require_distinct_channels=False),
            gt=StreamContract(value_range=(-128.0, 128.0),
                              require_range_extrema=False, channels=2,
                              require_distinct_channels=True)),
    ),
    "basicsr_reds": dict(
        # BasicSR options/train/BasicVSR/train_BasicVSR_REDS.yml gt_size
        declared_shape=(256, 256),
        module="adapters_basicsr_reds",
        cls="BasicSRRedsAdapter",
        root=Path("/home/laura/02ImproveData/zBasicSR"),
        note="BasicSR REDSDataset; the most widely reused video training loader "
             "in restoration. Included for reach, and as a concordance check "
             "against subject 1, which descends from it",
        declared_policy="fixed",
        expected_transforms=0,
        # REDSDataset documents and hard-codes clips 0..99.
        clip_length=100,
        content=RGB_255,
    ),
}


def _register_extension_subjects() -> None:
    """Add the subjects registered beside their adapter.

    A subject added after the original ten ships as ``adapters/_subject_<key>.py``
    next to its adapter, defining ``KEY`` and ``entry(h)``. ``h`` carries the
    contract helpers of this module, so an entry is written in the same
    vocabulary as the ten above without importing this script a second time.
    Keeping the contract in a file of its own means that adding a subject never
    edits the registry of the others.
    """
    import importlib.util
    import types

    helpers = types.SimpleNamespace(
        Path=Path, ContentContract=ContentContract, StreamContract=StreamContract,
        rgb_contract=_rgb_contract, RGB_255=RGB_255, RGB_01=RGB_01)
    folder = HERE / "adapters"
    for path in sorted(folder.glob("_subject_*.py")) if folder.is_dir() else ():
        spec = importlib.util.spec_from_file_location(path.stem, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if mod.KEY in SUBJECTS:
            raise SystemExit(f"{path.name}: subject {mod.KEY!r} is already registered")
        SUBJECTS[mod.KEY] = mod.entry(helpers)


_register_extension_subjects()

#: The primary Study 3 profile. Six of the seven delivery gates can run from one
#: loader; ``separability`` needs two declared conditions and has its own paired
#: campaign. Evaluation geometry is deliberately absent: a training sample is
#: not a prediction/reference pair at the scoring boundary.
PRIMARY_GATES = ("precomputed_input", "temporal_window",
                 "target_transforms", "operator_trace",
                 "value_range", "channel_content")


def adapter_loc(module_name: str) -> int:
    """Non-blank, non-comment, non-docstring lines of the adapter module."""
    path = HERE / f"{module_name}.py"
    if not path.exists():
        path = HERE / "adapters" / f"{module_name}.py"
    if not path.exists():
        return -1
    # Counted from the parse rather than by matching quote characters: the
    # line-based version toggled its docstring state twice on an opening
    # line, so every multi-line docstring was counted as code and the
    # reported sizes were inflated by different amounts per adapter.
    import ast
    import io
    import tokenize

    source = path.read_text()
    docstring_lines: set[int] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                    and isinstance(first.value.value, str)):
                docstring_lines.update(range(first.lineno, first.end_lineno + 1))
    skip = {tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT,
            tokenize.DEDENT, tokenize.ENCODING, tokenize.ENDMARKER}
    code_lines: set[int] = set()
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if tok.type not in skip:
            code_lines.update(range(tok.start[0], tok.end[0] + 1))
    return len(code_lines - docstring_lines)


def _pin(root: Path) -> str:
    """The subject's commit, so a portability result is as pinned as an audit one.

    The article pins the four audited repositories by commit; a portability
    subject that is not pinned is a weaker artefact than the thing it is meant
    to strengthen. Subjects outside version control report ``not-versioned``.
    """
    import subprocess
    try:
        out = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        if out.returncode == 0:
            return out.stdout.strip()[:12]
    except Exception:
        pass
    return "not-versioned"


def _observed_ops(recorder) -> set:
    """Operators the adapter's instrumentation actually saw.

    A subject that exposes no operator sequence yields an empty set, and the
    gate reports SKIP rather than a pass -- which is the honest outcome and one
    of this study's findings about scope.
    """
    return {c for c in recorder.calls if c != "sharpen"}


#: Package names more than one subject ships a copy of. Three subjects here
#: carry a ``basicsr``: BasicSR itself and two projects that vendored it. In one
#: process the last import owns the name, so a later subject can be audited
#: through an earlier subject's code --- and it does not raise, because the
#: copies define most of the same names. The harness clears them between
#: subjects; doing it here rather than in the adapters keeps the adapter line
#: counts, which are a response variable of this study, measuring adaptation
#: and not isolation.
SHARED_PACKAGE_NAMES = ("basicsr",)


def _isolate_shared_package_names(root: Path) -> None:
    """Drop cached copies of packages several subjects share, and put this
    subject's own root first on the path.

    The roots of the other subjects leave the path too. Clearing the cached
    module was not enough on its own: with an earlier subject's root still on
    the path, ``import basicsr`` from a subject that uses the installed package
    resolved to that subject's vendored copy instead.
    """
    for name in [m for m in list(sys.modules)
                 if m in SHARED_PACKAGE_NAMES
                 or any(m.startswith(f"{p}.") for p in SHARED_PACKAGE_NAMES)]:
        del sys.modules[name]
    # Directories inside another subject go too (MPRNet's adapter adds its
    # Denoising/ folder): a regular package there, such as its ``utils``,
    # outranks a namespace directory of the same name anywhere on the path.
    others = [Path(c["root"]) for c in SUBJECTS.values() if Path(c["root"]) != Path(root)]

    def _inside_other(entry: str) -> bool:
        p = Path(entry)
        return any(p == o or o in p.parents for o in others)

    sys.path[:] = [p for p in sys.path if p != str(root) and not _inside_other(p)]
    sys.path.insert(0, str(root))


def _content_scale_on_delivered(adapter, workdir: Path, cfg: dict, *,
                                clip_length: int, declared_clip_length):
    """Ask what the loader did to the picture, not to the tensor.

    Five sources are probed, 16:9, 4:3, 1:1, 3:4 and 9:16, with a ladder of
    pattern periods so a subject that delivers a small patch still gets a
    reading. Several aspects are the point: the size of a distortion depends on
    the input, and only its presence across different inputs is a property of
    the loader.
    """
    from treatment_identity.fixtures import SCALE_PROBE_SOURCES, make_scale_grid

    probes = []
    sources = SCALE_PROBE_SOURCES
    ladder = ((6, 10), (12, 20), (24, 40), (40, 60))
    for tag, (sh, sw) in sources:
        for px, py in ladder:
            d = workdir / f"{tag}_{px}"
            d.mkdir(parents=True, exist_ok=True)
            for side in ("GT", "LQ"):
                make_scale_grid(d / side, n_frames=max(clip_length, 8),
                                shape=(sh, sw), period_x=px, period_y=py)
            spec = LoaderSpec(gt_root=d / "GT", lq_root=d / "LQ",
                              num_frames=cfg.get("num_frames", 5), seed=0)
            loader = adapter.build(spec)
            sample = adapter.sample(loader, 0)
            stream = sample.gt if sample.gt is not None else sample.lq
            if stream is None:
                continue
            mx, my = measure_content_scale(stream)
            if mx is not None and my is not None:
                probes.append((stream, px, py, sw / sh))
                break            # this source is read; move to the next aspect
    return check_content_scale(probes,
                               declared=cfg.get("declared_content"))


def _sample_shape_on_delivered(adapter, workdir: Path, cfg: dict, *,
                               clip_length: int, declared_clip_length):
    """Loader-sample shape check, not evaluation geometry.

    This compares a delivered training sample with the patch size requested by
    the subject configuration. It never observes prediction/reference geometry
    at a scoring boundary and must therefore retain a distinct gate name.
    """
    from treatment_identity.fixtures import make_pair

    workdir.mkdir(parents=True, exist_ok=True)
    # The fixture is half as large again as the configured sample, so a loader
    # that crops or resizes to the configured size shows it, and one that
    # passes the stored image through shows that instead. A fixture of exactly
    # the configured size would hide the second case.
    declared = cfg.get("declared_shape")
    side = max(128, -(-int(1.5 * max(declared)) // 16) * 16) if declared else 128
    make_pair(workdir, n_frames=clip_length, shape=(side, side))
    spec = LoaderSpec(gt_root=workdir / "GT", lq_root=workdir / "LQ",
                      num_frames=cfg.get("num_frames", 5), seed=0)
    loader = adapter.build(spec)
    sample = adapter.sample(loader, 0)
    # A zero-reference subject delivers no target, so the gate reads the
    # stream that subject does deliver rather than reporting an error.
    stream = sample.gt if sample.gt is not None else sample.lq
    if stream is None:
        return CheckResult("sample_shape", SKIP,
                           "the subject delivered no stream to measure", {})
    arr = np.asarray(stream)
    while arr.ndim > 3:
        arr = arr[0]
    # Subjects deliver both layouts. Reading the channel axis as a spatial one
    # produces a shape like 3x32 and a confident, wrong verdict, so the axis
    # order is resolved before anything is asserted.
    if arr.ndim == 3 and arr.shape[0] in (1, 2, 3) and arr.shape[2] not in (1, 2, 3):
        arr = np.transpose(arr, (1, 2, 0))
    delivered = (int(arr.shape[0]), int(arr.shape[1]))
    if declared is None:
        # Comparing the delivered shape with itself would pass every time and
        # attest nothing. A subject whose configuration names no output size
        # gets no verdict here.
        return CheckResult("sample_shape", SKIP,
                           "the subject's configuration declares no output "
                           "size for this gate to assert", {})
    declared = tuple(int(v) for v in declared)
    evidence = {"declared_shape": list(declared),
                "delivered_shape": list(delivered),
                "fixture_shape": [side, side],
                "shape_source": "measured from the delivered training sample"}
    if delivered == declared:
        return CheckResult("sample_shape", "PASS",
                           f"delivered {delivered[0]}x{delivered[1]}, as declared",
                           evidence)
    return CheckResult(
        "sample_shape", "FAIL",
        f"delivered {delivered[0]}x{delivered[1]} differs from the requested "
        f"sample shape {declared[0]}x{declared[1]}", evidence)


def run_subject(key: str, cfg: dict, workdir: Path,
                repeats: int = 5, escalate: bool = True, *,
                experimental_content_scale: bool = True,
                sample_aspect_diagnostic: bool = True) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    effort = dict(subject=key, project="", url="", root=str(cfg["root"]),
                  commit=_pin(cfg["root"]), note=cfg["note"],
                  status="", adapter_loc=adapter_loc(cfg["module"]),
                  adapter_seconds="", gates_seconds="",
                  gates_seconds_min="", gates_seconds_max="",
                  gates_seconds_repeats=0, clip_escalation=int(escalate),
                  experimental_content_scale=int(experimental_content_scale),
                  sample_aspect_diagnostic=int(sample_aspect_diagnostic),
                  gates_applicable=0, gates_failed=0, gates_undecl=0,
                  error="")

    if not cfg["root"].exists():
        effort["status"] = "UNAVAILABLE"
        effort["error"] = f"{cfg['root']} not present"
        return rows, effort

    _isolate_shared_package_names(cfg["root"])

    # Timed on purpose: for two of the four subjects the import is where the
    # reconstructed dependency is installed, so this is the only place the
    # environment-reconstruction cost shows up as a number.
    #
    # Caveat, recorded so the column is not misread: within one process the
    # first subject absorbs the one-time import of the deep-learning framework
    # the later subjects then find cached. The figures are therefore a cost of
    # the run, not a per-subject comparison, and the manuscript does not rank
    # subjects by them.
    t_adapter = time.perf_counter()
    try:
        mod = __import__(cfg["module"])
        adapter = getattr(mod, cfg["cls"])()
        effort["project"] = getattr(adapter, "project", "")
        effort["url"] = getattr(adapter, "url", "")
    except Exception as exc:                                  # pragma: no cover
        effort["adapter_seconds"] = round(time.perf_counter() - t_adapter, 4)
        effort["status"] = "ADAPTER_ERROR"
        effort["error"] = f"{type(exc).__name__}: {exc}"[:300]
        return rows, effort
    effort["adapter_seconds"] = round(time.perf_counter() - t_adapter, 4)

    recorder = getattr(adapter, "recorder", CallRecorder())
    results: dict = {}
    errors: dict[str, str] = {}
    suite_times: list[float] = []

    # Each gate is isolated so one subject error cannot mask the others.
    # The suite is timed over several repeats and reported as a median. A
    # single wall clock on a shared machine is not a reproducible number: the
    # same suite measured here with a training job on the box ran up to an
    # order of magnitude slower than on an idle one, while the median across
    # repeats barely moved. Statuses are deterministic, so they are taken from
    # the first repeat and only the timing is repeated.
    # Escalation is the robust default.  It keeps the declared requirement
    # separate from the manufactured fixture length and reports an undeclared
    # minimum rather than converting a short-fixture abort into missing data.
    declared_clip = cfg.get("clip_length")

    def _gate(fn, gate, default):
        if not escalate:
            return fn(default)
        return with_clip_escalation(fn, gate=gate, declared=declared_clip,
                                    default=default)

    def plan_for(rep: int):
        base = workdir / f"r{rep}"
        plan = [
            ("precomputed_input",
             lambda: _gate(lambda n: check_precomputed_input(
                 adapter, base / "g1",
                 declared=cfg.get("precomputed", True), clip_length=n,
                 declared_clip_length=declared_clip,
                 value_range=cfg["content"].lq.value_range),
                 "precomputed_input", 16)),
            ("temporal_window",
             lambda: _gate(lambda n: check_temporal_window(
                 adapter, base / "g2", clip_length=n,
                 declared_clip_length=declared_clip,
                 num_frames=cfg.get("num_frames", 5)),
                 "temporal_window", 32)),
            ("target_transforms",
             lambda: _gate(lambda n: check_target_transforms(
                 adapter, base / "g4", recorder, "sharpen",
                 expected=cfg.get("expected_transforms", 0),
                 # A contract that states zero has declared a count. Reading
                 # the value as a truth test made "0" and "not stated" the same
                 # thing, which the corrected gate now reports as UNDECL.
                 declared="expected_transforms" in cfg,
                 has_target=cfg.get("has_target", True),
                 clip_length=n, declared_clip_length=declared_clip),
                 "target_transforms", 8)),
            ("operator_trace",
             lambda: _gate(lambda n: check_operator_trace(
                 adapter, base / "g5", recorder,
                 set(cfg.get("ops", set())) or _observed_ops(recorder),
                 declared_policy=cfg.get("declared_policy", "fixed"),
                 clip_length=n, declared_clip_length=declared_clip),
                 "operator_trace", 8)),
            ("value_range",
             lambda: _gate(lambda n: check_value_range(
                 adapter, base / "g6", cfg["content"], clip_length=n,
                 declared_clip_length=declared_clip),
                 "value_range", 16)),
            ("channel_content",
             lambda: _gate(lambda n: check_channel_content(
                 adapter, base / "g7", cfg["content"], clip_length=n,
                 declared_clip_length=declared_clip),
                 "channel_content", 16)),
        ]
        if experimental_content_scale:
            plan.append(("content_scale",
             lambda: _gate(lambda n: _content_scale_on_delivered(
                 adapter, base / "g9", cfg, clip_length=n,
                 declared_clip_length=declared_clip),
                 "content_scale", 8)))
        if sample_aspect_diagnostic:
            plan.append(("sample_shape",
             lambda: _gate(lambda n: _sample_shape_on_delivered(
                 adapter, base / "g8", cfg, clip_length=n,
                 declared_clip_length=declared_clip),
                 "sample_shape", 8)))
        return plan

    for rep in range(max(1, repeats)):
        t0 = time.perf_counter()
        for gate, fn in plan_for(rep):
            try:
                out = fn()
                if rep == 0:
                    results[gate] = out
            except Exception as exc:
                if rep == 0:
                    errors[gate] = f"{type(exc).__name__}: {exc}"[:200]
                    effort["status"] = "PARTIAL"
        suite_times.append(time.perf_counter() - t0)

    suite_times.sort()
    effort["gates_seconds"] = round(statistics.median(suite_times), 4)
    effort["gates_seconds_min"] = round(suite_times[0], 4)
    effort["gates_seconds_max"] = round(suite_times[-1], 4)
    effort["gates_seconds_repeats"] = len(suite_times)
    if errors:
        effort["error"] = "; ".join(f"{g}: {m[:60]}" for g, m in errors.items())[:300]

    reported_gates = list(PRIMARY_GATES)
    if experimental_content_scale:
        reported_gates.append("content_scale")
    if sample_aspect_diagnostic:
        reported_gates.append("sample_shape")
    for gate in reported_gates:
        res = results.get(gate)
        rows.append(dict(subject=key, gate=gate,
                         status=res.status if res else "ERROR",
                         message=(res.message[:200] if res else errors.get(gate, ""))))
    effort["gates_applicable"] = sum(
        1 for r in rows if r["status"] not in ("SKIP", "ERROR"))
    effort["gates_failed"] = sum(1 for r in rows if r["status"] == "FAIL")
    effort["gates_undecl"] = sum(1 for r in rows if r["status"] == "UNDECL")
    effort["status"] = effort["status"] or "RUN"
    return rows, effort


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject", action="append",
                    help="run only these subjects (default: all registered)")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--out", type=Path, default=_default_out())
    clip_group = ap.add_mutually_exclusive_group()
    clip_group.add_argument("--clip-escalation", dest="clip_escalation",
                            action="store_true", help=argparse.SUPPRESS)
    clip_group.add_argument(
        "--no-clip-escalation", dest="clip_escalation", action="store_false",
        help="sensitivity run with a fixed short fixture; the robust default "
             "escalates and reports undeclared minimum clip requirements")
    ap.set_defaults(clip_escalation=True)
    ap.add_argument("--repeats", type=int, default=5,
                    help="times the gate suite is timed per subject; the "
                         "reported wall clock is the median (default: 5)")
    ap.add_argument("--experimental-content-scale",
                    dest="experimental_content_scale", action="store_true",
                    default=True, help=argparse.SUPPRESS)
    ap.add_argument(
        "--without-content-scale", dest="experimental_content_scale",
        action="store_false",
        help="omit the content-scale diagnostic and write a suffixed output")
    ap.add_argument("--sample-aspect-diagnostic",
                    dest="sample_aspect_diagnostic", action="store_true",
                    default=True, help=argparse.SUPPRESS)
    ap.add_argument(
        "--without-sample-shape", dest="sample_aspect_diagnostic",
        action="store_false",
        help="omit the training-sample shape diagnostic and write a suffixed "
             "output; this diagnostic is not evaluation geometry")
    args = ap.parse_args()

    _report_resolved_package()

    if args.list:
        for k, c in SUBJECTS.items():
            mark = "present" if c["root"].exists() else "ABSENT"
            print(f"{k:16s} [{mark}]  {c['note']}")
        return 0

    keys = args.subject or list(SUBJECTS)
    args.out.mkdir(parents=True, exist_ok=True)
    wd = Path(tempfile.mkdtemp(prefix="campaign_P_"))

    all_rows: list[dict] = []
    all_effort: list[dict] = []
    for key in keys:
        if key not in SUBJECTS:
            print(f"unknown subject: {key}", file=sys.stderr)
            return 64
        print(f"== {key} ==")
        rows, effort = run_subject(
            key, SUBJECTS[key], wd / key, args.repeats, args.clip_escalation,
            experimental_content_scale=args.experimental_content_scale,
            sample_aspect_diagnostic=args.sample_aspect_diagnostic)
        all_rows += rows
        all_effort.append(effort)
        print(f"   status={effort['status']} adapter_loc={effort['adapter_loc']} "
              f"adapter_seconds={effort['adapter_seconds']} "
              f"gates_seconds={effort['gates_seconds']} {effort['error']}")
        for r in rows:
            print(f"   {r['gate']:20s} {r['status']:7s} {r['message'][:70]}")

    def write(path: Path, data: list[dict]) -> None:
        if not data:
            return
        with path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(data[0]), lineterminator="\n")
            w.writeheader()
            w.writerows(data)
        print(f"wrote {path} ({len(data)} rows)")

    # Mode and scope are encoded in filenames. Partial or sensitivity runs
    # cannot overwrite the complete robust matrix used by the manuscript.
    partial = bool(args.subject) and set(args.subject) != set(SUBJECTS)
    omitted = []
    if not args.experimental_content_scale:
        omitted.append("content_scale")
    if not args.sample_aspect_diagnostic:
        omitted.append("sample_shape")
    mode = ("_without_" + "_and_".join(omitted)) if omitted else ""
    scope = f"{mode}{'_partial' if partial else ''}"
    if args.clip_escalation:
        matrix_path = args.out / f"P_portability_matrix{scope}.csv"
        effort_path = args.out / f"P_effort{scope}.csv"
        summary_path = args.out / f"P_summary{scope}.json"
    else:
        matrix_path = args.out / f"P_portability_fixed_fixture{scope}.csv"
        effort_path = args.out / f"P_effort_fixed_fixture{scope}.csv"
        summary_path = args.out / f"P_summary_fixed_fixture{scope}.json"
    if partial:
        print(f"NOTE: partial run ({', '.join(sorted(args.subject))}); writing "
              f"*{scope} files and leaving the full-matrix outputs alone")
    write(matrix_path, all_rows)
    write(effort_path, all_effort)

    ran = [e for e in all_effort if e["status"] in ("RUN", "PARTIAL")]
    summary = dict(
        subjects_registered=len(SUBJECTS),
        subjects_run=len(ran),
        subjects_unavailable=[e["subject"] for e in all_effort
                              if e["status"] == "UNAVAILABLE"],
        subjects_partial=[e["subject"] for e in all_effort
                          if e["status"] == "PARTIAL"],
        subjects_adapter_error=[e["subject"] for e in all_effort
                                if e["status"] in ("ADAPTER_ERROR", "GATE_ERROR")],
        median_adapter_loc=(sorted(e["adapter_loc"] for e in ran)[len(ran) // 2]
                            if ran else None),
        min_adapter_loc=(min(e["adapter_loc"] for e in ran) if ran else None),
        max_adapter_loc=(max(e["adapter_loc"] for e in ran) if ran else None),
        max_adapter_seconds=(max(e["adapter_seconds"] for e in ran)
                             if ran else None),
        total_failures=sum(e["gates_failed"] for e in ran),
        total_undeclared=sum(e["gates_undecl"] for e in ran),
        gate_rows=len(all_rows),
    )
    summary["clip_escalation"] = int(args.clip_escalation)
    summary["assertion_gates"] = list(PRIMARY_GATES)
    summary["reported_columns"] = list(PRIMARY_GATES)
    if args.experimental_content_scale:
        summary["reported_columns"].append("content_scale")
    if args.sample_aspect_diagnostic:
        summary["reported_columns"].append("sample_shape")
    summary["experimental_content_scale"] = int(args.experimental_content_scale)
    summary["sample_aspect_diagnostic"] = int(args.sample_aspect_diagnostic)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print("\n--- Campaign P summary ---")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print("\nA subject that PASSES every gate is a reportable result, not a null.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
