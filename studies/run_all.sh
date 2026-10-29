#!/usr/bin/env bash
# Rerun every study of the article with the installed DDC release, on one
# performance core, one thread, no GPU. DART needs its own environment.
#   studies/run_all.sh <ddc python> <dart python> <old-film clones> <noise_data>
set -euo pipefail
PY=$1; DART_PY=$2; CLONES=$3; BANK=$4
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
V=$($PY -c "import treatment_identity as t; print(t.__version__)")
D=studies/data_$V; C=studies/old_film/certificates; mkdir -p "$D" "$C/released"
run() { echo "== $*"; taskset -c 4 "$@"; }
run $PY studies/campaign_M_seeded_defects.py --seeds 20 --out "$D" > "$D/M.log" 2>&1
run $PY studies/campaign_M2_multi_defect.py --seeds 5 --out "$D" > "$D/M2.log" 2>&1
run $PY studies/campaign_P_portability.py --out "$D" > "$D/P.log" 2>&1 || true
run $PY studies/profile_loader.py --json "$D/P_profiles.json" > "$D/profile.log" 2>&1
run $PY studies/sweep_settings.py --out "$D" > "$D/sweep.log" 2>&1
run $PY studies/old_film/audit_vp_lineage.py --repos "$CLONES" --bank "$BANK" --out "$C/released" > "$C/audit.log" 2>&1 || true
run $PY studies/old_film/repeat_draws.py --repos "$CLONES" --bank "$BANK" --draws 100 --seeds 5 > "$C/repeat_draws.log" 2>&1
run $DART_PY studies/old_film/audit_dart.py --out "$C/released" > "$C/audit_dart.log" 2>&1
run $DART_PY studies/profile_loader.py --subject dart --json "$D/P_profile_dart.json" > "$D/profile_dart.log" 2>&1
run $PY studies/old_film/documents/verify_documentary_audit.py
echo "done: $D"
