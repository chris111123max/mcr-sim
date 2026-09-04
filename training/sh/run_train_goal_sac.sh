#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
PROJECT_ROOT="$(cd -- "$PYTHON_ROOT/.." && pwd)"
WORKSPACE_ROOT="$(cd -- "$PROJECT_ROOT/../.." && pwd)"

MANAGED_NOHUP=0
FORWARD_ARGS=()
for arg in "$@"; do
    if [[ "$arg" == "--nohup" ]]; then MANAGED_NOHUP=1; else FORWARD_ARGS+=("$arg"); fi
done
set -- "${FORWARD_ARGS[@]}"

if (( MANAGED_NOHUP == 1 )) && [[ "${MCR_MANAGED_NOHUP_CHILD:-0}" != "1" ]]; then
    EXP_NAME=""
    ARGS=("$@")
    for ((i=0; i<${#ARGS[@]}; i++)); do
        case "${ARGS[$i]}" in
            --exp-name) EXP_NAME="${ARGS[$((i+1))]}";;
            --exp-name=*) EXP_NAME="${ARGS[$i]#*=}";;
        esac
    done
    if [[ -z "$EXP_NAME" || ! "$EXP_NAME" =~ ^[A-Za-z0-9._-]+$ ]]; then
        echo "[ERROR] managed --nohup requires a safe explicit --exp-name"; exit 2
    fi
    RUN_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
    RUN_DIR="$PROJECT_ROOT/training_runs/${EXP_NAME}_${RUN_TIMESTAMP}"
    mkdir -p "$RUN_DIR/logs"
    nohup env MCR_MANAGED_NOHUP_CHILD=1 MCR_RUN_TIMESTAMP="$RUN_TIMESTAMP" \
        bash "$SCRIPT_DIR/run_train_goal_sac.sh" "$@" \
        >"$RUN_DIR/logs/launcher.log" 2>&1 </dev/null &
    echo "[STARTED] algorithm=goal_sac pid=$!"
    echo "[STARTED] run_dir=$RUN_DIR"
    echo "[STARTED] launcher_log=$RUN_DIR/logs/launcher.log"
    exit 0
fi

SETUP_MCR_SOFA="${SETUP_MCR_SOFA:-$WORKSPACE_ROOT/setup_mcr_sofa.sh}"
if [[ ! -f "$SETUP_MCR_SOFA" ]]; then
    echo "[ERROR] Cannot find SOFA setup script: $SETUP_MCR_SOFA"; exit 1
fi
source "$SETUP_MCR_SOFA"
SOFA_BUILD_DEFAULT="$WORKSPACE_ROOT/mcr_env/sofa/build_plugins"
SOFA_BUILD="${SOFA_BUILD:-${SOFA_ROOT:-$SOFA_BUILD_DEFAULT}}"
SOFA_BUILD_PLUGINS="${SOFA_BUILD_PLUGINS:-${SOFAPYTHON3_ROOT:-$SOFA_BUILD_DEFAULT}}"
STLIB_ROOT="${STLIB_ROOT:-$WORKSPACE_ROOT/mcr_env/sofa/src/STLIB}"
SOFTROBOTS_LIB="$SOFA_BUILD_PLUGINS/external_directories/SoftRobots/lib"
BEAMADAPTER_LIB="$SOFA_BUILD_PLUGINS/external_directories/BeamAdapter/lib"
STLIB_LIB="$SOFA_BUILD_PLUGINS/external_directories/STLIB/lib"
export SOFA_ROOT="$SOFA_BUILD" SOFAPYTHON3_ROOT="$SOFA_BUILD_PLUGINS"
export PYTHONPATH="$SOFA_BUILD/lib/python3/site-packages:$SOFA_BUILD_PLUGINS/lib/python3/site-packages:$STLIB_ROOT:$STLIB_ROOT/python:$STLIB_ROOT/python3/src:$PYTHON_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$SOFA_BUILD/lib:$SOFA_BUILD_PLUGINS/lib:$STLIB_LIB:$SOFTROBOTS_LIB:$BEAMADAPTER_LIB${CONDA_PREFIX:+:$CONDA_PREFIX/lib}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export SOFA_PLUGIN_PATH="$SOFA_BUILD/lib:$SOFA_BUILD_PLUGINS/lib:$STLIB_LIB:$SOFTROBOTS_LIB:$BEAMADAPTER_LIB${SOFA_PLUGIN_PATH:+:$SOFA_PLUGIN_PATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
cd "$PYTHON_ROOT"

DISTRIBUTED=0; WORLD_SIZE_ARG=4
ARGS=("$@")
for ((i=0; i<${#ARGS[@]}; i++)); do
    case "${ARGS[$i]}" in
        --distributed) DISTRIBUTED=1;;
        --world-size) WORLD_SIZE_ARG="${ARGS[$((i+1))]}";;
        --world-size=*) WORLD_SIZE_ARG="${ARGS[$i]#*=}";;
    esac
done

SCRIPT="$PYTHON_ROOT/training/py/train_goal_sac.py"
if (( DISTRIBUTED == 1 )) && [[ "${WORLD_SIZE:-1}" -le 1 ]]; then
    torchrun --standalone --nnodes=1 --nproc_per_node="$WORLD_SIZE_ARG" "$SCRIPT" "$@"
else
    python "$SCRIPT" "$@"
fi
