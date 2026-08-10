#!/usr/bin/env python
"""Campaign P --- portability of the treatment-identity gates outside the lineage.

Study 3 of the Software Quality Journal manuscript. The four codebases audited
in the article share a common ancestor: they are one family, not four samples.
This campaign runs the same gates against training loaders from *independently
developed* projects and records what it cost to get there.

Reporting rule, stated up front because it decides how the result reads: a run
in which every external loader PASSES is a result, not a failure of the study.
It bounds the population in which the audited failures are concentrated, and it
is a stronger statement than "we only looked at one family".

Response variables per subject:
    gate outcomes            what each gate returns
    adapter_loc              non-blank, non-comment lines of the adapter
    adapter_seconds          import of the adapter module plus construction of
                             the adapter object. This is where any environment
                             reconstruction a subject needs is actually paid,
                             so it is the machine-time component of adaptation
                             effort and it is reported as such.
    gates_seconds            median wall clock of the gate suite over
                             --repeats timings (min and max also recorded)
    gates_applicable         gates that ran, versus those the subject cannot support

Measurement note, 2026-08-10. An earlier version of this harness declared a
``build_seconds`` column -- "time to construct the loader over a fixture" --
initialised it to the empty string and never wrote to it. Four subjects were
reported with that column blank and nobody noticed, because every consumer read
the columns that were filled. That is the defect class this campaign exists to
detect, occurring in the harness that measures it, and it is the second time
this harness has reproduced the article's own thesis (the first is the shared
exception handler recorded as lesson L7). The column is now ``adapter_seconds``,
it is measured, and what it measures is stated above rather than implied by its
name.

Outputs (under ``revistas/claude/data/``):
    P_portability_matrix.csv     subject x gate
    P_effort.csv                 one row per subject
    P_summary.json

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

# The adapters sit beside this script in the manuscript tree and under
# ``adapters/`` in the deposit. Both are offered; neither is required to exist.
sys.path.insert(0, str(HERE))
if (HERE / "adapters").is_dir():
    sys.path.insert(0, str(HERE / "adapters"))


def _default_out() -> Path:
    return HERE / "data" if (HERE / "data").is_dir() else HERE.parent / "data"

from treatment_identity import (check_operator_trace,               # noqa: E402
                                check_precomputed_input,
                                check_target_transforms,
                                check_temporal_window,
                                with_clip_escalation)
from treatment_identity.adapter import CallRecorder                # noqa: E402

# ---------------------------------------------------------------------------
# Subjects. Each entry: (module, class, root that must exist, one-line note)
# ---------------------------------------------------------------------------
SUBJECTS = {
    "kair_video": dict(
        module="adapters_kair_video",
        cls="KairVideoTrainAdapter",
        root=Path("/home/laura/02ImproveData/zKAIR"),
        note="KAIR VideoRecurrentTrainDataset; recurrent video training loader, "
             "unrelated project, own meta-info convention",
    ),
    "kair_blindsr": dict(
        module="adapters_kair_blindsr",
        cls="KairBlindSRAdapter",
        root=Path("/home/laura/02ImproveData/zKAIR"),
        note="KAIR DatasetBlindSR; the BSRGAN degradation pipeline, which "
             "composes a randomised operator sequence -- the subject that "
             "exercises operator_trace",
        declared_policy="random_permutation",
        expected_transforms=0,
        # Single-image pipeline: it has no temporal window, and saying so
        # in the contract is what lets temporal_window report N/A instead of
        # being handed a five-frame declaration it cannot meet.
        num_frames=1,
    ),
    "llie_singleframe": dict(
        module="adapters_llie_singleframe",
        cls="LlieSingleFrameAdapter",
        root=Path("/home/laura/02ImproveData/zLLIE-arch415"),
        note="single-frame low-light loader; fixed augmentation order and an "
             "explicit target-sharpening probability. Included to show which "
             "gates are inapplicable to a non-temporal pipeline (RQ4)",
        declared_policy="fixed",
        expected_transforms=1,
        # Single-image pipeline: it has no temporal window, and saying so
        # in the contract is what lets temporal_window report N/A instead of
        # being handed a five-frame declaration it cannot meet.
        num_frames=1,
    ),
    "basicsr_reds": dict(
        module="adapters_basicsr_reds",
        cls="BasicSRRedsAdapter",
        root=Path("/home/laura/02ImproveData/zBasicSR"),
        note="BasicSR REDSDataset; the most widely reused video training loader "
             "in restoration. Included for reach, and as a concordance check "
             "against subject 1, which descends from it",
        declared_policy="fixed",
        expected_transforms=0,
    ),
}

DELIVERY_GATES = ("precomputed_input", "temporal_window",
                  "target_transforms", "operator_trace")


def adapter_loc(module_name: str) -> int:
    """Non-blank, non-comment, non-docstring lines of the adapter module."""
    path = HERE / f"{module_name}.py"
    if not path.exists():
        path = HERE / "adapters" / f"{module_name}.py"
    if not path.exists():
        return -1
    loc, in_doc = 0, False
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if line.startswith(('"""', "'''")):
            in_doc = not in_doc or line.count('"""') == 2 and False
            if line.count('"""') >= 2:
                in_doc = False
            else:
                in_doc = not in_doc
            continue
        if in_doc or not line or line.startswith("#"):
            continue
        loc += 1
    return loc


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


def run_subject(key: str, cfg: dict, workdir: Path,
                repeats: int = 5, escalate: bool = False) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    effort = dict(subject=key, project="", url="", root=str(cfg["root"]),
                  commit=_pin(cfg["root"]), note=cfg["note"],
                  status="", adapter_loc=adapter_loc(cfg["module"]),
                  adapter_seconds="", gates_seconds="",
                  gates_seconds_min="", gates_seconds_max="",
                  gates_seconds_repeats=0, clip_escalation=int(escalate),
                  gates_applicable=0, gates_failed=0, gates_undecl=0,
                  error="")

    if not cfg["root"].exists():
        effort["status"] = "UNAVAILABLE"
        effort["error"] = f"{cfg['root']} not present"
        return rows, effort

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

    # Each gate is isolated. An earlier version ran them in one try block, so
    # the first exception masked three gates that would have reported. That is
    # the same failure mode the article is about, in the harness that measures
    # it.
    #
    # The suite is timed over several repeats and reported as a median. A
    # single wall clock on a shared machine is not a reproducible number: the
    # same suite measured here with a training job on the box ran up to an
    # order of magnitude slower than on an idle one, while the median across
    # repeats barely moved. Statuses are deterministic, so they are taken from
    # the first repeat and only the timing is repeated.
    # Clip-length escalation: OFF by default, and that is a setting, not a fork.
    #
    # There is one harness. With --clip-escalation the gates retry along the
    # ladder and an undeclared clip-length requirement comes back as UNDECL
    # with the shortest length that worked; without it they behave exactly as
    # they did before the option existed. The mode is recorded in P_effort.csv,
    # because a number whose meaning depends on a flag has to carry the flag.
    #
    # The manuscript's table is the escalation-OFF run. Reporting a table
    # produced one way while shipping a harness that defaults to the other
    # would be the defect this study is about.
    declared_clip = cfg.get("clip_length")

    def _gate(fn, gate, default, base):
        if not escalate:
            return fn(default)
        return with_clip_escalation(fn, gate=gate, declared=declared_clip,
                                    default=default)

    def plan_for(rep: int):
        base = workdir / f"r{rep}"
        return [
            ("precomputed_input",
             lambda: _gate(lambda n: check_precomputed_input(
                 adapter, base / "g1", clip_length=n),
                 "precomputed_input", 16, base)),
            ("temporal_window",
             lambda: _gate(lambda n: check_temporal_window(
                 adapter, base / "g2", clip_length=n,
                 num_frames=cfg.get("num_frames", 5)),
                 "temporal_window", 32, base)),
            ("target_transforms",
             lambda: _gate(lambda n: check_target_transforms(
                 adapter, base / "g4", recorder, "sharpen",
                 expected=cfg.get("expected_transforms", 0),
                 declared=bool(cfg.get("expected_transforms", 0)),
                 clip_length=n),
                 "target_transforms", 8, base)),
            ("operator_trace",
             lambda: _gate(lambda n: check_operator_trace(
                 adapter, base / "g5", recorder,
                 set(cfg.get("ops", set())) or _observed_ops(recorder),
                 declared_policy=cfg.get("declared_policy", "fixed"),
                 clip_length=n),
                 "operator_trace", 8, base)),
        ]

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

    for gate in DELIVERY_GATES:
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
    ap.add_argument("--clip-escalation", action="store_true",
                    help="on an undeclared clip-length requirement, retry along "
                         "the ladder and report UNDECL with the shortest length "
                         "that works, instead of letting the gate abort")
    ap.add_argument("--repeats", type=int, default=5,
                    help="times the gate suite is timed per subject; the "
                         "reported wall clock is the median (default: 5)")
    args = ap.parse_args()

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
        rows, effort = run_subject(key, SUBJECTS[key], wd / key, args.repeats,
                                   args.clip_escalation)
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
            w = csv.DictWriter(fh, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
        print(f"wrote {path} ({len(data)} rows)")

    write(args.out / "P_portability_matrix.csv", all_rows)
    write(args.out / "P_effort.csv", all_effort)

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
    )
    (args.out / "P_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("\n--- Campaign P summary ---")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print("\nA subject that PASSES every gate is a reportable result, not a null.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
