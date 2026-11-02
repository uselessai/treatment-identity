"""Requirements: what a run should deliver, and whether the profile shows it.

A requirement can come from two places. The project's own configuration states
some of them (a patch size, a number of frames, a scale, whether rotations are
used, whether a stored input is read). A contract file states the rest, or
replaces what the configuration says, or sets aside what should not be
required. Every requirement keeps its source, and each is judged against the
profile, which is written in every run whatever the requirements are.

A contract file (YAML or JSON):

    from_config: true            # take the requirements the configuration states
    ignore: [patch]              # ...except these
    requirements:                # add, or replace what the configuration says
      frames: 5
      window: moves
      input: stored
    vary:                        # run again with each value; does the delivery change?
      datasets.train.use_rot: [true, false]

The names a requirement can have are those of ``VOCABULARY``.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .checks import CheckResult

PASS, FAIL, NA, SKIP = "PASS", "FAIL", "N/A", "SKIP"

VOCABULARY = {
    "frames": "frames per delivered item",
    "window": "'moves' or 'fixed': whether the frame window changes between draws",
    "patch": "side of the delivered target in pixels, or [height, width]",
    "short_side": "shorter side of the delivered target in pixels",
    "scale": "how many times smaller the delivered input is than the target",
    "input": "'stored' (read from disk) or 'made' (built by the loader)",
    "rotation": "'none', 'present' (some probes arrive rotated) or 'allowed'",
    "content": "'crop', 'rescale' or 'reshape'",
    "value_range": "[low, high] that every delivered value lies within",
    "channels": "channels of the delivered target",
    "operator_order": "'fixed' or 'random'",
    "operations": "{function: times per delivered frame}",
}

# Configuration keys that state a requirement, and how their value reads as one.
CONFIG_KEYS = {
    "frames": (r"num_frames?|nframes", lambda v: int(v)),
    "patch": (r"gt_size|gt_patch_size|patch_size|crop_size|train_ps|image_size|img_size"
              r"|out_size|h_size", lambda v: [int(x) for x in v] if isinstance(v, list)
              else int(v)),
    # a resolution sets the shorter side; a square crop meets it too
    "short_side": (r"resolution", lambda v: int(v)),
    "scale": (r"scale|sf", lambda v: int(v)),
    "rotation": (r"use_rot", lambda v: "allowed" if _true(v) else "none"),
    "input": (r"dataroot_lq|dataroot_l|lq_root|input_dir|lq_path",
              lambda v: "stored" if isinstance(v, str) and v.strip() else None),
}


def _true(v) -> bool:
    return v is True or str(v).strip().lower() in ("true", "1", "yes")


# ------------------------------------------------------------ from the config
def _entries(cwd: Path, command: list[str]):
    """(rank, source, key path, value) for every value of the configuration and
    of the command line. Rank 0 is the command line, 1 a file it names, 2 a
    file found otherwise (named in the script, or composed by Hydra)."""
    from .wizard import _config_files, _load_config, _walk

    args = list(command)
    for i, a in enumerate(args):
        if a.startswith("--") and "=" in a:
            k, v = a[2:].split("=", 1)
            yield 0, "command line", k, v
        elif a.startswith("--") and i + 1 < len(args) and not args[i + 1].startswith("-"):
            yield 0, "command line", a[2:], args[i + 1]
        elif not a.startswith("-") and "=" in a:
            k, v = a.split("=", 1)
            yield 0, "command line", k.lstrip("+~"), v
    named = {(cwd / a).resolve() for a in args}
    for f in _config_files(cwd, command):
        cfg = _load_config(f)
        try:
            name = str(f.relative_to(cwd))
        except ValueError:
            name = str(f)
        rank = 1 if f.resolve() in named else 2
        for key, value in _walk(cfg) if cfg is not None else []:
            yield rank, name, key, value
    # a Python configuration named on the command line: its plain assignments
    for a in command[2:]:
        p = cwd / a
        if a.endswith(".py") and p.is_file():
            for m in re.finditer(r"\b(\w+)\s*=\s*(-?\d+|True|False|'[^']*'|\"[^\"]*\")",
                                 p.read_text(errors="ignore")):
                yield 1, a, m.group(1), m.group(2).strip("'\"")


def from_configuration(cwd: Path, command: list[str]) -> tuple[dict, dict]:
    """The requirements the configuration states for the training set, and
    those it states more than once with different values (left out)."""
    found: dict[str, list] = {}
    entries, pairs = [], {}
    for rank, source, key, value in _entries(Path(cwd), command):
        # a list of two numbers, such as "gt_size: [256, 256]", is one value
        head, _, index = key.rpartition(".")
        if index.isdigit() and isinstance(value, (int, float)) and not isinstance(value, bool):
            pairs.setdefault((rank, source, head), []).append(value)
            continue
        entries.append((rank, source, key, value))
    entries += [(r, s, k, v) for (r, s, k), v in pairs.items() if len(v) == 2]
    for rank, source, key, value in entries:
        parts = re.split(r"[.\[\]]", key.lower())
        if any(p.startswith(("val", "test")) for p in parts):
            continue
        last = parts[-1] if parts else ""
        for name, (pattern, read) in CONFIG_KEYS.items():
            if re.fullmatch(pattern, last):
                try:
                    declared = read(value)
                except (TypeError, ValueError):
                    continue
                if declared is not None:
                    found.setdefault(name, []).append((rank, declared, f"{source}: {key}"))
    # an input folder that is the target's own folder states no stored input:
    # the loader makes its input from the target
    targets = {str(v).rstrip("/") for _, _, k, v in entries
               if re.split(r"[.\[\]]", k.lower())[-1] in ("dataroot_gt", "dataroot_h", "gt_root")}
    if "input" in found:
        lq = {str(v).rstrip("/") for _, _, k, v in entries
              if re.fullmatch(CONFIG_KEYS["input"][0], re.split(r"[.\[\]]", k.lower())[-1])}
        if lq & targets:
            found.pop("input")
    chosen, conflicting = {}, {}
    for name, ranked in found.items():
        # what the command line states, else a file it names, else the rest
        top = min(r for r, _, _ in ranked)
        values = [(v, s) for r, v, s in ranked if r == top]
        distinct = {json.dumps(v) for v, _ in values}
        if len(distinct) == 1:
            chosen[name] = {"declared": values[0][0], "source": values[0][1]}
        else:
            conflicting[name] = [{"declared": v, "source": s} for v, s in values]
    return chosen, conflicting


# -------------------------------------------------------------- the contract
def load_contract(path: Path | None) -> dict:
    if path is None:
        return {}
    text = Path(path).read_text()
    if Path(path).suffix.lower() == ".json":
        return json.loads(text) or {}
    import yaml
    return yaml.safe_load(text) or {}


def combine(configured: dict, contract: dict, contract_name: str = "contract") -> dict:
    """The requirements of a run: those of the configuration unless the
    contract sets them aside, then the contract's, which replace any of the
    configuration with the same name."""
    unknown = sorted(set(contract.get("requirements") or {}) - set(VOCABULARY))
    if unknown:
        raise ValueError(f"unknown requirement(s) {unknown}; known: {sorted(VOCABULARY)}")
    out = {}
    if contract.get("from_config", True):
        ignore = set(contract.get("ignore") or [])
        out = {k: dict(v) for k, v in configured.items() if k not in ignore}
    for name, declared in (contract.get("requirements") or {}).items():
        entry = {"declared": declared, "source": contract_name}
        if name in out:
            entry["replaces"] = out[name]
        out[name] = entry
    return out


# ------------------------------------------------------------------ judging
def _get(d, *keys):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def _observe(name: str, declared, profile: dict):
    """(observed, status, note) for one requirement."""
    gt, lq = _get(profile, "streams", "gt"), _get(profile, "streams", "lq")
    if name == "frames":
        seen = _get(profile, "window", "frames")
        return seen, (PASS if seen == int(declared) else FAIL), f"{seen} frame(s) per item"
    if name == "window":
        frames, distinct = _get(profile, "window", "frames"), _get(profile, "window", "distinct_windows")
        if frames == 1:
            return "single image", NA, "one frame per item: there is no window"
        seen = "moves" if (distinct or 0) > 1 else "fixed"
        return seen, (PASS if seen == declared else FAIL), _get(profile, "window", "note")
    if name == "patch":
        # every stored size the probes used: the stream fixture and the five
        # content fixtures of different proportions
        # two sides, in whichever order the configuration writes them
        want = sorted(declared) if isinstance(declared, (list, tuple)) else [int(declared)] * 2
        shape = (gt or lq or {}).get("shape")
        sizes = ([{"stored": "stream fixture", "delivered": shape[:2]}] if shape else []) \
            + (_get(profile, "content", "sizes") or [])
        if not sizes:
            return None, SKIP, "no delivered size was read"
        wrong = [s for s in sizes if sorted(s["delivered"]) != want]
        note = (f"all {len(sizes)} stored sizes delivered as {want}" if not wrong else
                "; ".join(f"stored {s['stored']} delivered {s['delivered']}" for s in wrong))
        return sizes, (PASS if not wrong else FAIL), note
    if name == "short_side":
        shape = (gt or lq or {}).get("shape")
        sizes = ([{"stored": "stream fixture", "delivered": shape[:2]}] if shape else []) \
            + (_get(profile, "content", "sizes") or [])
        if not sizes:
            return None, SKIP, "no delivered size was read"
        wrong = [s for s in sizes if min(s["delivered"]) != int(declared)]
        note = (f"all {len(sizes)} stored sizes delivered with shorter side {declared}"
                if not wrong else
                "; ".join(f"stored {s['stored']} delivered {s['delivered']}" for s in wrong))
        return sizes, (PASS if not wrong else FAIL), note
    if name == "scale":
        if not gt or not lq:
            return None, NA, "no stored input and target pair is delivered"
        ratio = gt["shape"][0] / lq["shape"][0]
        return round(ratio, 3), (PASS if abs(ratio - int(declared)) < 1e-6 else FAIL), \
            f"target {gt['shape'][:2]}, input {lq['shape'][:2]}"
    if name == "input":
        source = _get(profile, "input_source", "source") or ""
        seen = "stored" if "pre-rendered" in source else "made" if "manufactures" in source else None
        if seen is None:
            return None, SKIP, _get(profile, "input_source", "note") or "not readable"
        return seen, (PASS if seen == declared else FAIL), _get(profile, "input_source", "note")
    if name == "rotation":
        rotated = _get(profile, "content", "evidence", "rotated_probes")
        readings = _get(profile, "content", "evidence", "readings")
        if rotated is None and readings:
            rotated = sum("rotation_deg" in r for r in readings)
        if rotated is None:
            return None, SKIP, _get(profile, "content", "note") or "not readable"
        probes = _get(profile, "content", "evidence", "probes") or len(readings or [])
        seen = f"{rotated} of {probes} probes rotated"
        ok = declared == "allowed" or (rotated == 0) == (declared == "none")
        return seen, (PASS if ok else FAIL), seen
    if name == "content":
        seen = _get(profile, "content", "behaviour")
        return seen, (SKIP if seen is None else PASS if seen == declared else FAIL), \
            _get(profile, "content", "note")
    if name == "value_range":
        lo, hi = declared
        ranges = {s: v["range"] for s, v in (profile.get("streams") or {}).items() if v}
        ok = all(lo <= r[0] and r[1] <= hi for r in ranges.values())
        return ranges, (PASS if ok else FAIL), f"delivered {ranges}"
    if name == "channels":
        seen = (gt or lq or {}).get("channels")
        return seen, (PASS if seen == int(declared) else FAIL), f"{seen} channel(s)"
    if name == "operator_order":
        distinct = _get(profile, "operators", "distinct_orders")
        if not _get(profile, "operators", "observed"):
            return None, SKIP, "no operation was observed; name the functions to observe"
        seen = "random" if (distinct or 0) > 1 else "fixed"
        return seen, (PASS if seen == declared else FAIL), _get(profile, "operators", "note")
    if name == "operations":
        counts = _get(profile, "operators", "operations") or {}
        seen = {k: (counts.get(k) or {}).get("per_frame", 0.0) for k in declared}
        ok = all(abs(seen[k] - float(v)) < 1e-6 for k, v in declared.items())
        return seen, (PASS if ok else FAIL), f"per delivered frame: {seen}"
    raise ValueError(f"unknown requirement {name!r}")


def judge(profile: dict, requirements: dict) -> list[CheckResult]:
    """One result per requirement, read from the profile."""
    out = []
    for name, req in requirements.items():
        observed, status, note = _observe(name, req["declared"], profile)
        evidence = {"declared": req["declared"], "observed": observed, "source": req["source"]}
        if "replaces" in req:
            evidence["replaces"] = req["replaces"]
        out.append(CheckResult(f"requirement:{name}", status,
                               f"{name}: declared {req['declared']!r} ({req['source']}); {note}",
                               evidence))
    return out


# ------------------------------------------------------------------ varying
#: For a requirement read from the configuration: another value of the setting
#: (as written in the configuration) and the requirement it should then meet.
#: Several are tried in order when the loader rejects one (a patch too small
#: for its degradations, for example). Scale is changed only where it is
#: above one: a task without a stored input of another size has none to change.
def _smaller(v) -> list:
    if isinstance(v, list):                          # [a, b]: both sides
        return [([x // 2 for x in v],) * 2, ([x * 3 // 4 // 16 * 16 for x in v],) * 2]
    v = int(v)
    return [(v // 2, v // 2), (v * 3 // 4 // 16 * 16, v * 3 // 4 // 16 * 16)]


ALTERNATIVE = {
    "patch": lambda r: _smaller(r["declared"]),
    "short_side": lambda r: _smaller(r["declared"]),
    "frames": lambda r: ([(int(r["declared"]) // 2, int(r["declared"]) // 2)]
                         if int(r["declared"]) >= 4 else
                         [(int(r["declared"]) + 2, int(r["declared"]) + 2)]),
    "scale": lambda r: ([(2, 2)] if int(r["declared"]) > 2 else
                        [(4, 4)] if int(r["declared"]) == 2 else []),
    "rotation": lambda r: [(False, "none")] if r["declared"] == "allowed" else [(True, "present")],
}


def training_switches(cwd: Path, command: list[str], roots: list[str]) -> list[dict]:
    """The true/false settings of the loader's part of the configuration: the
    section, in the same file, of a key that names one of its data folders."""
    from .wizard import _resolved_entries

    cwd = Path(cwd)
    roots = [Path(r).resolve() for r in roots]
    sections = set()
    for f, key, value in _resolved_entries(cwd, command):
        if f is None or "${" in value:
            continue
        path = ((cwd / value) if not value.startswith("/") else Path(value)).resolve()
        if path == cwd.resolve() or path in cwd.resolve().parents:
            continue
        if any(p.startswith(("val", "test")) for p in re.split(r"[.\[\]]", key.lower())):
            continue
        if any(path == r or r in path.parents or path in r.parents for r in roots):
            parts = key.split(".")
            while parts and parts[-1].isdigit():        # an item of a list
                parts.pop()
            parts = parts[:-1]
            sections.add((Path(f).resolve(), ".".join(parts)))
    # only the files the run reads first: one named on the command line wins
    # over one the script names as a default
    named = {(cwd / a).resolve() for a in command}
    if any(sf in named for sf, _ in sections):
        sections = {(sf, sec) for sf, sec in sections if sf in named}
    found = []
    for rank, source, key, value in _entries(cwd, command):
        if source == "command line" or not (isinstance(value, bool) or value in ("True", "False")):
            continue
        f = ((cwd / source) if not source.startswith("/") else Path(source)).resolve()
        if any(f == sf and (not sec or key.startswith(sec + ".")) for sf, sec in sections):
            found.append({"rank": rank, "source": source, "key": key,
                          "value": value if isinstance(value, bool) else value == "True"})
    return found


def compare_profiles(base: dict, other: dict) -> dict[str, Any]:
    """The parts of the delivery that differ between two profiles."""
    fields = {"frames": ("window", "frames"), "window": ("window", "distinct_windows"),
              "target shape": ("streams", "gt", "shape"), "input shape": ("streams", "lq", "shape"),
              "content": ("content", "behaviour"),
              "rotated probes": ("content", "rotated"),
              "input": ("input_source", "source"),
              "operations": ("operators", "operations"),
              "delivered values": ("window", "digest")}
    def read(profile, path):
        if path == ("content", "rotated"):
            readings = _get(profile, "content", "evidence", "readings") or []
            return sum("rotation_deg" in r for r in readings)
        return _get(profile, *path)
    return {n: {"before": read(base, p), "after": read(other, p)}
            for n, p in fields.items() if read(base, p) != read(other, p)}
