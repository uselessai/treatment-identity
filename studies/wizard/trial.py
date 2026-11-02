#!/usr/bin/env python
"""Run the wizard on one project as a person would: accept every answer it
proposes, and answer the others. Every question and how it was answered is
logged next to the answers file.

Every run judges the requirements the project's configuration states, and
those of ``contracts/<key>.yml`` when it exists, and changes each configured
value to check that the delivery follows it (``--follow-config``).

The answers a person gives are taken from ``oracle/<key>.json``. The run starts
from those of them that are written up front (``setup``, ``set``, ``files``);
the others are given only when the wizard asks. Without an oracle the run
stops at the first question the wizard cannot propose an answer for.

    python studies/wizard/trial.py <key> <project dir> [VAR=value ...] -- <training command>

VAR=value sets an environment variable the project's instructions ask for.
EXTRA_PATH, in the environment, names folders of packages the project needs
that its environment lacks; they are put ahead of it.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
UP_FRONT = ("setup", "set", "files")
REPO = HERE.parents[1]


def main() -> int:
    key, cwd = sys.argv[1], Path(sys.argv[2])
    split = sys.argv.index("--")
    command = sys.argv[split + 1:]
    variables = dict(a.split("=", 1) for a in sys.argv[3:split])
    answers = HERE / f"{key}.json"
    log = HERE / f"{key}.log.json"
    history = json.loads(log.read_text()) if log.exists() else []
    contract = HERE / "contracts" / f"{key}.yml"
    oracle_file = HERE / "oracle" / f"{key}.json"
    oracle = json.loads(oracle_file.read_text()) if oracle_file.exists() else None
    if oracle is not None:
        up_front = {k: oracle[k] for k in UP_FRONT if k in oracle}
        answers.write_text(json.dumps(up_front, indent=2) + "\n")
        history = [{"given up front": sorted(up_front)}]
    path = [os.environ["EXTRA_PATH"]] if os.environ.get("EXTRA_PATH") else []
    env = {"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1",
           "PYTHONPATH": os.pathsep.join(path + [str(REPO)]),
           "PATH": "/usr/bin:/bin", "HOME": str(Path.home()),
           "USER": os.environ.get("USER", ""), **variables}
    python = sys.executable
    for _ in range(12):
        run = subprocess.run(
            ["taskset", "-c", "4", python, "-m", "treatment_identity.wizard", "--cwd", str(cwd),
             "--answers", str(answers), "--profile", str(HERE / f"{key}.profile.json"),
             "--certificate", str(HERE / f"{key}.certificate.json"), "--follow-config",
             *(["--contract", str(contract)] if contract.exists() else []), "--", *command],
            cwd=cwd, env=env, capture_output=True, text=True)
        (HERE / f"{key}.out.txt").write_text(run.stdout[-20000:] + "\n--- stderr ---\n" + run.stderr[-20000:])
        if run.returncode == 0:
            detected = re.search(r"used without asking .*?: (.*)", run.stderr)
            # one more run with every answer in place and nothing changed: the
            # machine time of a run of the wizard, start to profile
            start = time.monotonic()
            timed = subprocess.run(
                ["taskset", "-c", "4", python, "-m", "treatment_identity.wizard", "--cwd", str(cwd),
                 "--answers", str(answers), "--", *command],
                cwd=cwd, env=env, capture_output=True, text=True)
            seconds = round(time.monotonic() - start, 1) if timed.returncode == 0 else None
            history.append({"result": "profile written",
                            "used_without_asking": detected.group(1).split(", ") if detected else [],
                            "seconds_per_run": seconds})
            log.write_text(json.dumps(history, indent=2) + "\n")
            asked = [h for h in history if "question" in h]
            given = sum(h["answered"] == "given by a person" for h in asked)
            print(f"{key}: profile written; {len(asked)} question(s), {given} answered by a person")
            return 0
        q = re.search(r"QUESTION \[(.+?)\]: (.*)", run.stdout)
        if not q:
            history.append({"result": "error", "tail": run.stderr.strip().splitlines()[-3:]})
            log.write_text(json.dumps(history, indent=2) + "\n")
            print(f"{key}: error\n" + "\n".join(run.stderr.strip().splitlines()[-6:]))
            return 1
        proposal = re.search(r"^  detected: (.*)$", run.stdout, re.M)
        has_suggestion = proposal is not None
        data = json.loads(answers.read_text()) if answers.exists() else {}
        same = has_suggestion and q.group(1) in (oracle or {}) and \
            (oracle[q.group(1)] == "detected" or oracle[q.group(1)] == json.loads(proposal.group(1)))
        if q.group(1) not in data and oracle is not None and q.group(1) in oracle and not same:
            data[q.group(1)] = oracle[q.group(1)]
            answers.write_text(json.dumps(data, indent=2) + "\n")
            history.append({"question": q.group(1), "answered": "given by a person"})
            continue
        if not has_suggestion or q.group(1) in data:
            history.append({"question": q.group(1), "text": q.group(2), "answered": "needs a person"})
            log.write_text(json.dumps(history, indent=2) + "\n")
            print(f"{key}: needs an answer to [{q.group(1)}]: {q.group(2)}")
            return 2
        data[q.group(1)] = "detected"
        answers.write_text(json.dumps(data, indent=2) + "\n")
        history.append({"question": q.group(1), "answered": "proposal accepted"})
    log.write_text(json.dumps(history, indent=2) + "\n")
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
