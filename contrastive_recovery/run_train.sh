#!/usr/bin/env bash
# Isolated launcher.  Existing PPO/SAC launchers are intentionally untouched.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_ROOT="$(cd "$HERE/.." && pwd)"
PROJECT_ROOT="$(cd "$PY_ROOT/.." && pwd)"
WORKSPACE_ROOT="$(cd "$PROJECT_ROOT/.." && pwd)"
TRAIN="$HERE/train.py"
DEFAULT_EXP="contrastive_recovery_2npu_32env"

if [[ "${1:-}" == "--nohup" ]]; then
  shift
  EXP="$DEFAULT_EXP"
  ARGS=("$@")
  for ((i=0; i<${#ARGS[@]}; i++)); do
    if [[ "${ARGS[$i]}" == "--exp-name" && $((i+1)) -lt ${#ARGS[@]} ]]; then EXP="${ARGS[$((i+1))]}"; fi
  done
  STAMP="$(date +%Y%m%d_%H%M%S)"
  ROOT_RUN="$PROJECT_ROOT/training_runs/${EXP}_${STAMP}"
  mkdir -p "$ROOT_RUN/logs"
  nohup bash "$HERE/run_train.sh" --run-dir "$ROOT_RUN" "${ARGS[@]}" >"$ROOT_RUN/logs/launcher.log" 2>&1 &
  PID=$!
  echo "[STARTED] algorithm=contrastive_recovery pid=$PID"
  echo "[STARTED] launcher_log=$ROOT_RUN/logs/launcher.log"
  echo "[STARTED] trainer will create its synchronized run_dir under training_runs/"
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
USE_DDP=1
for ((i=1; i<=$#; i++)); do
  if [[ "${!i}" == "--world-size" ]]; then j=$((i+1)); WORLD="${!j}"; fi
  if [[ "${!i}" == "--distributed" ]]; then USE_DDP=1; fi
done

BASE=(--device npu --n-envs 32 --sequence-batch-size 256 --sequence-length 32 --updates-per-rollout 2 --npu-fast-execution)
if [[ "$USE_DDP" == "1" ]]; then
  exec torchrun --standalone --nnodes=1 --nproc_per_node="$WORLD" "$TRAIN" --distributed --world-size "$WORLD" "${BASE[@]}" "$@"
else
  exec python "$TRAIN" "${BASE[@]}" "$@"
fi
