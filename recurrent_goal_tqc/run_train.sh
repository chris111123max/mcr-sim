#!/usr/bin/env bash
# Standalone launcher for the recurrent Goal-TQC experiment.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_ROOT="$(cd "$HERE/.." && pwd)"
PROJECT_ROOT="$(cd "$PY_ROOT/.." && pwd)"
WORKSPACE_ROOT="$(cd "$PROJECT_ROOT/../.." && pwd)"
ENTRY="$HERE/train.py"

if [[ "${1:-}" == "--eval" ]]; then
  shift
  ENTRY="$HERE/evaluate.py"
fi

if [[ "${1:-}" == "--nohup" ]]; then
  shift
  EXP="rgtqc_pilot_2npu_32env"
  ARGS=("$@")
  for ((i=0; i<${#ARGS[@]}; i++)); do
    if [[ "${ARGS[$i]}" == "--exp-name" && $((i+1)) -lt ${#ARGS[@]} ]]; then
      EXP="${ARGS[$((i+1))]}"
    fi
  done
  RUN="$PROJECT_ROOT/training_runs/${EXP}_$(date +%Y%m%d_%H%M%S)"
  mkdir -p "$RUN/logs"
  nohup bash "$HERE/run_train.sh" --run-dir "$RUN" "${ARGS[@]}" >"$RUN/logs/launcher.log" 2>&1 &
  echo "[STARTED] algorithm=recurrent_goal_tqc pid=$!"
  echo "[STARTED] run_dir=$RUN"
  echo "[STARTED] launcher_log=$RUN/logs/launcher.log"
  exit 0
fi

SETUP_MCR_SOFA="${SETUP_MCR_SOFA:-$WORKSPACE_ROOT/setup_mcr_sofa.sh}"
if [[ ! -f "$SETUP_MCR_SOFA" ]]; then
  echo "[ERROR] Cannot find SOFA setup script: $SETUP_MCR_SOFA"; exit 1
fi
# shellcheck disable=SC1090
source "$SETUP_MCR_SOFA"
SOFA_BUILD_DEFAULT="$WORKSPACE_ROOT/mcr_env/sofa/build_plugins"
SOFA_BUILD="${SOFA_BUILD:-${SOFA_ROOT:-$SOFA_BUILD_DEFAULT}}"
SOFA_BUILD_PLUGINS="${SOFA_BUILD_PLUGINS:-${SOFAPYTHON3_ROOT:-$SOFA_BUILD_DEFAULT}}"
STLIB_ROOT="${STLIB_ROOT:-$WORKSPACE_ROOT/mcr_env/sofa/src/STLIB}"
SOFTROBOTS_LIB="$SOFA_BUILD_PLUGINS/external_directories/SoftRobots/lib"
BEAMADAPTER_LIB="$SOFA_BUILD_PLUGINS/external_directories/BeamAdapter/lib"
STLIB_LIB="$SOFA_BUILD_PLUGINS/external_directories/STLIB/lib"
export SOFA_ROOT="$SOFA_BUILD" SOFAPYTHON3_ROOT="$SOFA_BUILD_PLUGINS"
export PYTHONPATH="$SOFA_BUILD/lib/python3/site-packages:$SOFA_BUILD_PLUGINS/lib/python3/site-packages:$STLIB_ROOT:$STLIB_ROOT/python:$STLIB_ROOT/python3/src:$PY_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$SOFA_BUILD/lib:$SOFA_BUILD_PLUGINS/lib:$STLIB_LIB:$SOFTROBOTS_LIB:$BEAMADAPTER_LIB${CONDA_PREFIX:+:$CONDA_PREFIX/lib}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export SOFA_PLUGIN_PATH="$SOFA_BUILD/lib:$SOFA_BUILD_PLUGINS/lib:$STLIB_LIB:$SOFTROBOTS_LIB:$BEAMADAPTER_LIB${SOFA_PLUGIN_PATH:+:$SOFA_PLUGIN_PATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
cd "$PY_ROOT"

WORLD=2
USE_DDP=0
for ((i=1; i<=$#; i++)); do
  if [[ "${!i}" == "--world-size" ]]; then j=$((i+1)); WORLD="${!j}"; fi
  if [[ "${!i}" == "--distributed" ]]; then USE_DDP=1; fi
done
if [[ "$USE_DDP" == "1" ]]; then
  if [[ "$ENTRY" != "$HERE/train.py" ]]; then
    echo "[ERROR] Evaluation is single-process; omit --distributed"; exit 1
  fi
  exec torchrun --standalone --nnodes=1 --nproc_per_node="$WORLD" "$ENTRY" --distributed --world-size "$WORLD" "$@"
fi
exec python "$ENTRY" "$@"
