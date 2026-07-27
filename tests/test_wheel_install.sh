#!/usr/bin/env bash
# Build the wheel, install it under an isolated temporary prefix, and run the
# console script from a directory that contains none of the source.
#
# This is the test that a green `python -m build` does not give you: a wheel can
# build cleanly and still declare an entry point for a module it never packaged.
# The command below is run with the project tree unreachable on purpose.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-python3}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
SRC="$TMP/source"

# The test is worthless if the source tree can be reached. A PYTHONPATH with an
# empty entry puts the working directory on sys.path, and a stale *.egg-info in
# the tree then makes pip treat the package as already installed: it reports
# success, skips the install, and never writes the console scripts. Unset it and
# build an isolated copy, so this test never removes or rewrites source-tree
# build artefacts.
unset PYTHONPATH
mkdir -p "$SRC"
cp -a "$HERE/." "$SRC/"
rm -rf "$SRC/build" "$SRC/treatment_identity.egg-info"
cd "$TMP"

echo "== building wheel from isolated copy of $HERE"
"$PY" -m pip wheel --no-deps -q -w "$TMP/dist" "$SRC"
WHEEL="$(ls "$TMP"/dist/treatment_identity-*.whl)"
echo "   $(basename "$WHEEL")"

echo "== contents"
"$PY" - "$WHEEL" <<'EOF'
import sys, zipfile
with zipfile.ZipFile(sys.argv[1]) as wheel:
    names = sorted(wheel.namelist())
    metadata_name = next(n for n in names if n.endswith(".dist-info/METADATA"))
    metadata = wheel.read(metadata_name).decode()
    for n in names:
        if n.endswith((".py", ".json")):
            print("   ", n)
required = {"selftest.py", "audit_vp_lineage.py",
            "treatment_identity/checks.py", "treatment_identity/_version.py",
            "treatment_identity/schemas/treatment-certificate-1.0.schema.json",
            "adapters/vp_code.py"}
missing = {r for r in required if not any(n.endswith(r) for n in names)}
if missing:
    sys.exit(f"MISSING FROM WHEEL: {sorted(missing)}")
for expected in (
        "Version: 1.0.2",
        "Requires-Python: >=3.10",
        "Requires-Dist: jsonschema>=4.10",
):
    if expected not in metadata:
        sys.exit(f"MISSING FROM WHEEL METADATA: {expected}")
EOF

echo "== installing under an isolated temporary prefix"
# Dependencies come from the host: this test is about what the wheel itself
# ships and whether its console scripts resolve, and it must run without
# network access. --no-deps keeps the install offline; --prefix avoids mutating
# the active interpreter even on hosts without the optional python3-venv
# package.
PREFIX="$TMP/prefix"
"$PY" -m pip install -q --no-deps --ignore-installed --prefix "$PREFIX" "$WHEEL"
SITE="$(find "$PREFIX" -type d \( -path '*/site-packages' -o -path '*/dist-packages' \) -print -quit)"
if [[ -z "$SITE" ]]; then
    echo "temporary site-packages directory not found" >&2
    exit 1
fi
CLI="$(find "$PREFIX" -type f -name 'treatment-identity-selftest' -print -quit)"
if [[ -z "$CLI" ]]; then
    echo "temporary console script not found" >&2
    exit 1
fi

echo "== running the console script from outside the source tree"
cd "$TMP"
PYTHONPATH="$SITE" "$CLI"
rc=$?

echo "== exit code check: a failing gate must be non-zero"
PYTHONPATH="$SITE" "$PY" - <<'EOF'
from importlib.metadata import version
from treatment_identity import (
    Certificate, CheckResult, __version__, check_geometry, validate_certificate,
)

assert version("treatment-identity") == __version__ == "1.0.2"

r = check_geometry((368, 640), (180, 320))
assert r.status == "FAIL", r
assert r.is_divergence

cert = Certificate("wheel-fixture", "bounded installation test")
cert.add(CheckResult("applicable", "PASS", "executed", {}))
cert.add(CheckResult("deferred", "SKIP", "not executed", {}))
assert cert.status == "PARTIAL", cert.status
validate_certificate(cert.to_dict())

print("   distorted geometry -> FAIL, as required")
print("   package/runtime version -> 1.0.2")
print("   PASS + SKIP -> PARTIAL; bundled schema validates")
EOF

echo
echo "OK: wheel installs under a temporary prefix and the CLI runs with the project tree unreachable (rc=$rc)"
