#!/usr/bin/env python
"""Collect the wizard's trials into the release's data, and summarise them.

Each trial's answers, questions, profile and certificate are written to the
loader's folder of the release (see ``studies/data_layout.py``), beside the
profile taken through its adapter. Then, for every loader:

* the questions a person answered, split into those about the loader and those
  about the layout of data folders absent on the study machine (not asked
  where the data are present), the proposals accepted and the values used
  without asking;
* the profile fields that differ between the adapter and the wizard
  (``loaders/comparison.csv``);
* from the certificate: the requirements read from the configuration and their
  verdicts, the configured values changed and whether the delivery followed
  them, and the true/false settings switched.

    python studies/wizard/summarize.py --data studies/data_1.13.0 \
        [--cost-tex cost.tex] [--tex wizard.tex]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from campaign_P_portability import adapter_loc          # noqa: E402
from data_layout import (ADAPTER, ANSWERS, CERTIFICATE, QUESTIONS, WIZARD,  # noqa: E402
                         read, write)

FIELDS = [("target shape", ("streams", "gt", "shape")), ("input shape", ("streams", "lq", "shape")),
          ("target range", ("streams", "gt", "range")), ("input range", ("streams", "lq", "range")),
          ("channels", ("streams", "gt", "channels")),
          ("identical channels", ("streams", "gt", "identical_channels")),
          ("content", ("content", "behaviour")), ("frames", ("window", "frames")),
          ("distinct windows", ("window", "distinct_windows")),
          ("input source", ("input_source", "source")),
          ("operations observed", ("operators", "observed"))]

# How each project states its configuration, in the order the article groups them.
GROUPS = [("A file named on the command line",
           ["basicsr_reds", "restormer", "gfpgan", "ddcolor_lab", "kair_video", "kair_blindsr",
            "vrt", "e2fgvi", "propainter", "mgfrofr"]),
          ("A file named in the training script", ["mprnet"]),
          ("A file named by the value of a command-line option", ["rtn", "rrtn", "mambaofr"]),
          ("Files composed by Hydra, with command-line overrides", ["lama", "dart"]),
          ("A Python configuration file", ["mmagic"]),
          ("Command-line options only, defaults in the code",
           ["zerodce", "uformer", "pix2pix_colorization", "diffueraser"])]
KIND = {k: g for g, keys in GROUPS for k in keys}
NAMES = {"kair_video": "KAIR video", "kair_blindsr": "KAIR BlindSR", "uformer": "Uformer",
         "mprnet": "MPRNet", "zerodce": "Zero-DCE", "mmagic": "MMagic",
         "pix2pix_colorization": "pix2pix colour", "restormer": "Restormer",
         "ddcolor_lab": "DDColor", "basicsr_reds": "BasicSR REDS", "diffueraser": "DiffuEraser",
         "e2fgvi": "E2FGVI", "gfpgan": "GFPGAN", "lama": "LaMa", "propainter": "ProPainter",
         "vrt": "VRT", "dart": "DART", "rtn": "RTN", "rrtn": "RRTN", "mambaofr": "MambaOFR",
         "mgfrofr": "MgfrOFR"}
#: The four old-film systems of Study 4 share one adapter.
OLD_FILM = ("rtn", "rrtn", "mambaofr", "mgfrofr")
TASKS = {"kair_video": "Video super-resolution", "kair_blindsr": "Blind image super-resolution",
         "uformer": "Image restoration", "mprnet": "Image restoration",
         "zerodce": "Low-light enhancement", "mmagic": "Video super-resolution",
         "pix2pix_colorization": "Colourisation", "restormer": "Image restoration",
         "ddcolor_lab": "Colourisation", "basicsr_reds": "Video super-resolution",
         "diffueraser": "Video inpainting", "e2fgvi": "Video inpainting",
         "gfpgan": "Face restoration", "lama": "Image inpainting",
         "propainter": "Video inpainting", "vrt": "Video super-resolution",
         "dart": "Old-film restoration", **dict.fromkeys(OLD_FILM, "Old-film restoration")}


#: Projects held out from the wizard's development (run_heldout.sh): their
#: configuration, task, and what a person did beyond answering its questions.
HELDOUT = {
    "nafnet": ("NAFNet", "File", "Image deblurring", ""),
    "realesrgan": ("Real-ESRGAN", "File", "Blind super-resolution",
                   "gave the target as the input: the loader delivers no input"),
    "codeformer": ("CodeFormer", "File", "Face restoration", ""),
    "retinexformer": ("Retinexformer", "File", "Low-light enhancement", ""),
    "hat": ("HAT", "File", "Super-resolution", ""),
    "edsr": ("EDSR", "Options", "Super-resolution", ""),
    "edsr_img": ("EDSR, \\texttt{-{}-ext img}", "Options", "Super-resolution",
                 "chose an option of the project that reads each image when it is used"),
    "mimounet": ("MIMO-UNet", "Options", "Image deblurring",
                 "corrected the data folders and their layout"),
    "srcnn": ("SRCNN", "Options", "Super-resolution",
              "the wizard stands in a folder for its HDF5 file of training data"),
}


def _get(d, path):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def collect(data: Path) -> list[str]:
    """Write each trial of studies/wizard/ into its loader's folder."""
    keys = []
    for log in sorted(HERE.glob("*.log.json")):
        key = log.name[:-len(".log.json")]
        for src, name in ((log, QUESTIONS), (HERE / f"{key}.json", ANSWERS),
                          (HERE / f"{key}.profile.json", WIZARD),
                          (HERE / f"{key}.certificate.json", CERTIFICATE)):
            if src.exists():
                write(data, key, name, json.loads(src.read_text()))
        keys.append(key)
    # releases before 1.13.0 kept the adapter profiles in one file
    for old in ("P_profiles.json", "P_profile_dart.json"):
        f = data / old
        if f.exists():
            for p in (json.loads(f.read_text()) if f.read_text().lstrip().startswith("[")
                      else [json.loads(f.read_text())]):
                if read(data, p["subject"], ADAPTER) is None:
                    write(data, p["subject"], ADAPTER, p)
    return keys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=HERE.parent / "data_1.13.0")
    ap.add_argument("--out", type=Path, help="summary CSV (default <data>/W_wizard.csv)")
    ap.add_argument("--cost-tex", type=Path, help="write the article's table of cost here")
    ap.add_argument("--tex", type=Path, help="write the article's table of Study 5 here")
    ap.add_argument("--heldout-tex", type=Path,
                    help="write the article's table of the held-out projects here")
    args = ap.parse_args()
    keys = collect(args.data)
    effort = {r["subject"]: r for r in csv.DictReader(open(args.data / "P_effort.csv"))}
    rows, comparison = [], []
    for key in keys:
        history = read(args.data, key, QUESTIONS) or []
        answers = read(args.data, key, ANSWERS) or {}
        wizard = read(args.data, key, WIZARD)
        adapter = read(args.data, key, ADAPTER)
        cert = read(args.data, key, CERTIFICATE) or {"gates": []}
        up_front = history[0].get("given up front", []) if history else []
        person = [h["question"] for h in history if h.get("answered") == "given by a person"]
        accepted = [h["question"] for h in history if h.get("answered") == "proposal accepted"]
        written = person + accepted + up_front
        done = next((h for h in history if h.get("result") == "profile written"), None)
        diffs = []
        if wizard and adapter:
            for name, path in FIELDS:
                a, w = _get(adapter, path), _get(wizard, path)
                comparison.append({"loader": key, "field": name, "adapter": json.dumps(a),
                                   "wizard": json.dumps(w), "same": a == w})
                if a != w:
                    diffs.append(name)
        gates = cert["gates"]
        from_contract = lambda g: "contracts/" in str(g["evidence"].get("source", ""))  # noqa: E731
        con = [g for g in gates if g["gate"].startswith("requirement:") and from_contract(g)]
        req = [g for g in gates if g["gate"].startswith("requirement:") and not from_contract(g)]
        fol = [g for g in gates if g["gate"].startswith("follows:")]
        sw = (wizard or {}).get("switched_settings") or []
        repeats = next((x.get("delivery repeats under the fixed seed") for x in sw
                        if "delivery repeats under the fixed seed" in x), None)
        sw = [x for x in sw if "setting" in x]
        last = lambda x: x["setting"].split(": ")[-1].split(".")[-1]      # noqa: E731
        e = effort.get(key, {})
        rows.append({
            "subject": key,
            "configuration": KIND.get(key) or (HELDOUT[key][1] if key in HELDOUT else ""),
            "profile": "written" if done else "not written",
            "profile_readable": bool(wizard) and "note" not in (wizard.get("streams") or {}),
            "profile_note": (wizard or {}).get("streams", {}).get("note", ""),
            "why_not": "" if done else " ".join(
                (next((h for h in reversed(history) if h.get("result") == "error"), {})
                 .get("tail") or [""])[-1:]),
            "adapter_loc": int(e["adapter_loc"]) if e else adapter_loc(
                "old_film/vp_code" if key in OLD_FILM else "adapters_" + key),
            "adapter_profile_seconds": (adapter or {}).get("seconds"),
            "adapter_checks_seconds": float(e["gates_seconds"]) if e else None,
            "wizard_seconds_per_run": (done or {}).get("seconds_per_run"),
            "person_loader": len([q for q in person if not q.startswith("layout:")]),
            "person_layout_absent_data": len([q for q in person if q.startswith("layout:")]),
            "proposals_accepted": len(accepted),
            "used_without_asking": len([k for k in answers if k not in written]),
            "entries_up_front": sum(len(answers[k]) for k in ("set", "files") if k in answers),
            "setup_lines": len(answers.get("setup", [])),
            "person_questions": " ".join(q.split(":")[0] for q in person),
            "differs_from_adapter_profile": "; ".join(diffs),
            "contract_requirements": len(con),
            "contract_met": sum(g["status"] == "PASS" for g in con),
            "contract_not_met": " ".join(g["gate"].split(":")[1] for g in con
                                         if g["status"] == "FAIL"),
            "configured_requirements": len(req),
            "configured_met": sum(g["status"] == "PASS" for g in req),
            "configured_not_met": " ".join(g["gate"].split(":")[1] for g in req
                                           if g["status"] == "FAIL"),
            "changed_values": len(fol),
            "changed_followed": sum(g["status"] == "PASS" for g in fol),
            "changed_not_followed": " ".join(g["gate"].split(":")[1] for g in fol
                                             if g["status"] == "FAIL"),
            "changed_not_run": " ".join(g["gate"].split(":")[1] for g in fol
                                        if g["status"] == "SKIP"),
            "delivery_repeats": repeats,
            "switches": len([x for x in sw if x["effect"] != "not switched"]),
            "switches_changing_delivery": sum(x["effect"] == "changes the delivery" for x in sw),
            "switches_changing_nothing": sum(x["effect"].startswith("changes nothing") for x in sw),
            "switches_not_run": sum(x["effect"] == "run failed" for x in sw),
            "switches_changing_nothing_names": " ".join(last(x) for x in sw
                                                        if x["effect"].startswith("changes nothing")),
            "switches_not_observable": " ".join(last(x) for x in sw
                                                if x["effect"].startswith("not observable")),
            "switches_not_switched": " ".join(last(x) for x in sw
                                              if x["effect"] == "not switched"),
        })
    heldout = [r for r in rows if r["subject"] in HELDOUT]
    rows = [r for r in rows if r["subject"] in KIND]
    comparison = [c for c in comparison if c["loader"] in KIND]
    out = args.out or args.data / "W_wizard.csv"
    for path, table in ((out, rows), (args.data / "loaders" / "comparison.csv", comparison),
                        (args.data / "W_heldout.csv", heldout)):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(table[0]))
            w.writeheader()
            w.writerows(table)
        print(f"wrote {path}")
    if args.cost_tex:
        write_cost_table(rows, args.cost_tex)
    if args.tex:
        write_table(rows, args.tex)
    if args.heldout_tex:
        write_heldout_table(heldout, args.heldout_tex)
    return 0


def _ordered(rows):
    by = {r["subject"]: r for r in rows}
    return [(g, [by[k] for k in keys if k in by]) for g, keys in GROUPS]


def _s(v, digits=0):
    return "--" if v in (None, "") else f"{float(v):.{digits}f}"


def write_cost_table(rows: list[dict], path: Path) -> None:
    order = list(NAMES)
    rows = sorted(rows, key=lambda r: order.index(r["subject"]) if r["subject"] in order else 99)
    lines = [r"%% GENERATED by studies/wizard/summarize.py -- do not edit by hand.",
             r"\begin{table}[tbp]",
             r"\caption{The twenty-one loaders and what connecting DDC to each costs: the sixteen "
             r"of Study~3 and the five old-film systems of Study~4. \emph{Adapter}: lines of the "
             r"hand-written adapter (one adapter serves RTN, RRTN, MambaOFR and MgfrOFR); "
             r"seconds to import the project, build the loader and write its profile; seconds of "
             r"one run of all checks on the built loader. \emph{Wizard} (Study~5): answers a person "
             r"gave about the loader, plus those about the layout of data absent from the machine; "
             r"seconds of one run, from the training command to the profile.}\label{tab:projects}",
             r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{@{}llrrrcr@{}}", r"\toprule",
             r" & & \multicolumn{3}{c}{Adapter} & \multicolumn{2}{c}{Wizard} \\",
             r"\cmidrule(lr){3-5}\cmidrule(l){6-7}",
             r"Loader & Task & lines & profile (s) & checks (s) & answers & run (s) \\",
             r"\midrule"]
    for r in rows:
        lines.append(f"{NAMES.get(r['subject'], r['subject'])} & {TASKS.get(r['subject'], '')} & "
                     f"{r['adapter_loc']} & {_s(r['adapter_profile_seconds'], 1)} & "
                     f"{_s(r['adapter_checks_seconds'], 1)} & "
                     f"{r['person_loader']} + {r['person_layout_absent_data']} & "
                     f"{_s(r['wizard_seconds_per_run'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(lines) + "\n")


def write_table(rows: list[dict], path: Path) -> None:
    def cell(r):
        req = (f"{r['configured_met']}/{r['configured_requirements']}"
               if r["configured_requirements"] else "--")
        if r["configured_not_met"]:
            req += f" ({r['configured_not_met'].replace('_', ' ')})"
        fol = f"{r['changed_followed']}/{r['changed_values']}" if r["changed_values"] else "--"
        if r["changed_not_followed"]:
            fol += f" ({r['changed_not_followed'].replace('_', ' ')})"
        sw = (f"{r['switches_changing_delivery']}/{r['switches_changing_nothing']}"
              if r["switches"] else "--")
        con = (f"{r['contract_met']}/{r['contract_requirements']}"
               if r["contract_requirements"] else "--")
        if r["contract_not_met"]:
            con += f" ({r['contract_not_met']})"
        name = NAMES.get(r["subject"], r["subject"])
        if r["profile"] != "written":
            return (f"{name} & \\multicolumn{{4}}{{l}}{{no profile: the training command "
                    f"fails before it creates its loader}} \\\\")
        return f"{name} & {req} & {fol} & {sw} & {con} \\\\"
    lines = [r"%% GENERATED by studies/wizard/summarize.py -- do not edit by hand.",
             r"\begin{table}[tbp]",
             r"\caption{The twenty-one loaders connected through their training commands "
             r"(Study~5), grouped by how each project states its configuration. "
             r"\emph{Configured}: requirements read from the configuration that the delivery "
             r"meets, of those read, with those not met. \emph{Followed}: configured values "
             r"that, once changed, the delivery follows, of those changed, with those it does "
             r"not follow. \emph{Switches}: true/false settings of the loader whose switch "
             r"changes the delivery / changes nothing in it. \emph{Contract}: requirements of "
             r"a one-line contract, \texttt{window: moves}, that the delivery meets.}"
             r"\label{tab:wizard}",
             r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{@{}lllll@{}}", r"\toprule",
             r"Loader & Configured & Followed & Switches & Contract \\"]
    for group, members in _ordered(rows):
        lines += [r"\midrule", rf"\multicolumn{{5}}{{@{{}}l}}{{\emph{{{group}}}}} \\"]
        lines += [cell(r) for r in members]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(lines) + "\n")

def write_heldout_table(rows: list[dict], path: Path) -> None:
    def frac(met, n, failed, not_run=""):
        notes = [x for x in (failed.replace("_", " "),
                             not_run and f"{not_run.replace('_', ' ')}: run fails") if x]
        return (f"{met}/{n}" + (f" ({'; '.join(notes)})" if notes else "")) if n else "--"

    lines = [r"%% GENERATED by studies/wizard/summarize.py -- do not edit by hand.",
             r"\begin{table}[tbp]",
             r"\caption{The wizard on eight projects held out from its development, run with its "
             r"code unchanged. \emph{Configuration}: a file named on the command line, or "
             r"command-line options only. \emph{Answers}: given by a person about the loader, "
             r"plus those about the layout of absent data. \emph{Configured} and "
             r"\emph{Followed} as in Table~\ref{tab:wizard}.}\label{tab:heldout}",
             r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{0.14\textwidth}lcr"
             r">{\raggedright\arraybackslash}p{0.11\textwidth}"
             r">{\raggedright\arraybackslash}p{0.13\textwidth}"
             r">{\raggedright\arraybackslash}p{0.24\textwidth}@{}}",
             r"\toprule",
             r"Project & Configuration & Answers & Run (s) & Configured & Followed & "
             r"Profile \\", r"\midrule"]
    by = {r["subject"]: r for r in rows}
    for key, (name, kind, _task, done_by_person) in HELDOUT.items():
        r = by.get(key)
        if r is None:
            continue
        readable = r["profile"] == "written" and r["profile_readable"]
        if r["profile"] != "written":
            profile = "not written: " + (done_by_person or r["why_not"].split(":")[0])
        elif not readable:
            profile = "written, but states that the dataset read its data when it was built"
        else:
            profile = "written" + ("; a person " + done_by_person if done_by_person else "")
        configured = frac(r["configured_met"], r["configured_requirements"],
                          r["configured_not_met"]) if readable else "--"
        followed = frac(r["changed_followed"], r["changed_values"], r["changed_not_followed"],
                        r["changed_not_run"]) if readable else "--"
        lines.append(
            f"{name} & {kind} & {r['person_loader']} + {r['person_layout_absent_data']} & "
            f"{_s(r['wizard_seconds_per_run'], 1)} & {configured} & {followed} & {profile} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
