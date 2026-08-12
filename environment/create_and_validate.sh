#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
CONDA_BIN="${CONDA_BIN:-/home/laura/miniconda3/bin/conda}"

if [[ $# -lt 1 || $# -gt 2 || "$1" != /* ]]; then
  echo "usage: $0 /absolute/path/to/new/environment [/absolute/path/to/results]" >&2
  exit 2
fi

PREFIX="$1"
if [[ -e "$PREFIX" ]] && [[ -n "$(find "$PREFIX" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "refusing non-empty prefix: $PREFIX" >&2
  exit 2
fi

"$CONDA_BIN" create -y -p "$PREFIX" --file "$HERE/rrtn-linux-64-conda.lock"
# The retained environment is an observed installed state, not a newly solved
# dependency set.  Bootstrap its recorded setuptools before building legacy
# packages (notably mmcv), then install every recorded distribution exactly as
# observed without asking a current resolver to reinterpret old metadata.
"$PREFIX/bin/python" -m pip install setuptools==60.2.0
"$PREFIX/bin/python" -m pip install --no-deps --no-build-isolation \
  -r "$HERE/rrtn-pip.lock"
SITE_PACKAGES="$("$PREFIX/bin/python" -c \
  'import sysconfig; print(sysconfig.get_path("purelib"))')"
cp "$HERE/usercustomize.py" "$SITE_PACKAGES/usercustomize.py"
"$PREFIX/bin/python" -m pip install --no-deps "$ROOT"

cd "$ROOT"
"$PREFIX/bin/python" -m unittest discover -v -s tests
PY="$PREFIX/bin/python" bash tests/test_wheel_install.sh

if [[ $# -eq 2 ]]; then
  RESULTS="$2"
  if [[ "$RESULTS" != /* ]]; then
    echo "results path must be absolute: $RESULTS" >&2
    exit 2
  fi
  mkdir -p "$RESULTS"
  "$PREFIX/bin/python" campaigns/campaign_M_seeded_defects.py \
    --seeds 20 --out "$RESULTS"
  "$PREFIX/bin/python" campaigns/campaign_P_portability.py \
    --out "$RESULTS"
  "$PREFIX/bin/python" environment/verify_clean_campaigns.py "$RESULTS"
fi

echo "validated clean environment: $PREFIX"
