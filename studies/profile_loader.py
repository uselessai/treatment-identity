#!/usr/bin/env python3
"""Profile the registered external loaders (Study 3).

DDC writes a profile on every run: what the loader delivers, observed under
recorded probe settings (adapter, fixtures, requested sample length, seed and
number of draws), with no expected value. The probes
live in ``treatment_identity.profile``; this script runs them on every subject
of ``campaign_P_portability.SUBJECTS`` and writes the profiles as JSON.

Run:  python studies/profile_loader.py --subject ddcolor_lab
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE / "adapters"))

from treatment_identity.profile import describe_loader          # noqa: E402

import campaign_P_portability as camp                             # noqa: E402


def profile(key: str, cfg: dict) -> dict:
    module = __import__(cfg["module"])
    adapter = getattr(module, cfg["cls"])()   # its own root by default
    out = {"subject": key, "project": getattr(adapter, "project", ""),
           "subject_commit": camp._pin(cfg["root"])}
    out.update(describe_loader(adapter))
    return out


def render(p: dict) -> str:
    lines = [f"{p['loader']}  ({p['project']})", "-" * 66]
    for name in ("lq", "gt"):
        s = p["streams"].get(name)
        if s is None:
            lines.append(f"  {name:9} not delivered")
        else:
            lines.append(f"  {name:9} shape {s['shape']}, "
                         f"range [{s['range'][0]}, {s['range'][1]}], "
                         f"{s['channels']} channel(s)")
    op = p["operators"]
    lines.append(f"  operators {op.get('note', '')}"
                 + (f" ({', '.join(op['example'])})" if op.get("example") else ""))
    lines.append(f"  window    {p.get('window', {}).get('note', '')}")
    lines.append(f"  input     {p.get('input_source', {}).get('note', '')}")
    c = p["content"]
    lines.append(f"  content   {c['behaviour'] or 'not measurable'}: {c['note']}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subject", action="append",
                    help="profile only these (default: every registered subject)")
    ap.add_argument("--json", type=Path, help="also write the profiles here")
    args = ap.parse_args()

    keys = args.subject or camp.default_subjects()
    results, failures = [], []
    for key in keys:
        cfg = camp.SUBJECTS[key]
        camp._isolate_shared_package_names(cfg["root"])
        try:
            p = profile(key, cfg)
            results.append(p)
            print(render(p))
        except Exception as exc:                                  # noqa: BLE001
            failures.append(key)
            print(f"{key}: could not be profiled: "
                  f"{type(exc).__name__}: {str(exc)[:120]}")
        print()
    if args.json:
        args.json.write_text(json.dumps(results, indent=2) + "\n")
        print(f"wrote {args.json}")
    if failures:
        print("incomplete profile; failed subjects: " + ", ".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
