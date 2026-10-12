#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$ROOT"
N_ENVS=${N_ENVS:-64}
case "$N_ENVS" in 32|64) ;; *) echo "N_ENVS must be 32 or 64" >&2; exit 2 ;; esac
PYTHON=${PYTHON:-python}
PLUGIN=${MCR_RHS_PLUGIN:-$SCRIPT_DIR/native/_build/lib/libMCRContactReferenceConsistency.so}
BEAM_PLUGIN=$ROOT/mcr_sim/native/beam_unilateral/_build/lib/libMCRBeamLinearizedUnilateral.so
test -f "$PLUGIN"
test -f "$BEAM_PLUGIN"
# The candidate changes native C++ contact instrumentation.  Refuse to
# silently load a stale previously-built .so after git pull.
for source in "$SCRIPT_DIR/native/ReferenceContact.h" "$SCRIPT_DIR/native/TestOnlyRelinearizingGenericConstraintSolver.cpp"; do
  if [[ "$PLUGIN" -ot "$source" ]]; then
    echo "Candidate native plugin is older than $source; rebuild with: bash $SCRIPT_DIR/build.sh" >&2
    exit 3
  fi
done
export MCR_RHS_PLUGIN="$PLUGIN"
STAMP=$(date +%Y%m%d_%H%M%S)
NAME=${EXP_NAME:-ppo_beam_rhs_consistent_coarse_v2}_${N_ENVS}env_${STAMP}
RUN=$ROOT/../training_runs/$NAME
mkdir -p "$RUN/models" "$RUN/logs"
export MCR_CANDIDATE_RUN_DIR="$RUN"
nohup "$PYTHON" "$SCRIPT_DIR/train_candidate.py" \
 --device npu --epochs 100 --episodes-per-epoch 100 --n-envs "$N_ENVS" --n-steps 256 --batch-size 1024 --n-epochs 5 \
 --physics-substeps 2 --frame-skip 1 --time-step 0.005 --training-curriculum \
 --beam-safety-validation audit --beam-safety-audit-interval-substeps 16 --beam-safety-audit-near-clearance-mm 0.300 \
 --beam-safety-plugin "$BEAM_PLUGIN" --exp-name "$NAME" > "$RUN/startup.out" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$RUN/pid"
printf 'PID=%s\nLOG=%s/startup.out\n' "$PID" "$RUN"
