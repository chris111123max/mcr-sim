#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
PROJECT_ROOT="$(cd -- "$PYTHON_ROOT/.." && pwd)"
WORKSPACE_ROOT="$(cd -- "$PROJECT_ROOT/../.." && pwd)"

SETUP_MCR_SOFA="${SETUP_MCR_SOFA:-$WORKSPACE_ROOT/setup_mcr_sofa.sh}"
if [[ ! -f "$SETUP_MCR_SOFA" ]]; then
    echo "[ERROR] Cannot find SOFA setup script: $SETUP_MCR_SOFA"
    echo "Set SETUP_MCR_SOFA to the correct external setup script."
    exit 1
fi
source "$SETUP_MCR_SOFA"

SOFA_BUILD_DEFAULT="$WORKSPACE_ROOT/mcr_env/sofa/build_plugins"
SOFA_BUILD="${SOFA_BUILD:-${SOFA_ROOT:-$SOFA_BUILD_DEFAULT}}"
SOFA_BUILD_PLUGINS="${SOFA_BUILD_PLUGINS:-${SOFAPYTHON3_ROOT:-$SOFA_BUILD_DEFAULT}}"
STLIB_ROOT="${STLIB_ROOT:-$WORKSPACE_ROOT/mcr_env/sofa/src/STLIB}"
SOFTROBOTS_LIB="$SOFA_BUILD_PLUGINS/external_directories/SoftRobots/lib"
BEAMADAPTER_LIB="$SOFA_BUILD_PLUGINS/external_directories/BeamAdapter/lib"
STLIB_LIB="$SOFA_BUILD_PLUGINS/external_directories/STLIB/lib"
PYTHON_BIN="${PYTHON_BIN:-python}"

export SOFA_ROOT="$SOFA_BUILD"
export SOFAPYTHON3_ROOT="$SOFA_BUILD_PLUGINS"
export PYTHONPATH="$SOFA_BUILD/lib/python3/site-packages:$SOFA_BUILD_PLUGINS/lib/python3/site-packages:$STLIB_ROOT:$STLIB_ROOT/python:$STLIB_ROOT/python3/src:$PYTHON_ROOT${PYTHONPATH:+:$PYTHONPATH}"
CONDA_LIB_SUFFIX=""
if [[ -n "${CONDA_PREFIX:-}" ]]; then CONDA_LIB_SUFFIX=":$CONDA_PREFIX/lib"; fi
export LD_LIBRARY_PATH="$SOFA_BUILD/lib:$SOFA_BUILD_PLUGINS/lib:$STLIB_LIB:$SOFTROBOTS_LIB:$BEAMADAPTER_LIB$CONDA_LIB_SUFFIX${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export SOFA_PLUGIN_PATH="$SOFA_BUILD/lib:$SOFA_BUILD_PLUGINS/lib:$STLIB_LIB:$SOFTROBOTS_LIB:$BEAMADAPTER_LIB${SOFA_PLUGIN_PATH:+:$SOFA_PLUGIN_PATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

cd "$PYTHON_ROOT"
DISTRIBUTED=0
WORLD_SIZE_ARG=4
ARGS=("$@")
for ((i = 0; i < ${#ARGS[@]}; i++)); do
    case "${ARGS[$i]}" in
        --distributed) DISTRIBUTED=1 ;;
        --world-size)
            if ((i + 1 >= ${#ARGS[@]})); then echo "[ERROR] --world-size requires a value"; exit 2; fi
            WORLD_SIZE_ARG="${ARGS[$((i + 1))]}" ;;
        --world-size=*) WORLD_SIZE_ARG="${ARGS[$i]#*=}" ;;
    esac
done

export MCR_RUN_TIMESTAMP="${MCR_RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
TRAIN_SCRIPT="$PYTHON_ROOT/training/py/train_ppo.py"
WARNING_FILTER='!/\[WARNING\] \[LocalMinDistance\(localmindistance\)\] Determinant is null/'
if ((DISTRIBUTED == 1)) && [[ "${WORLD_SIZE:-1}" -le 1 ]]; then
    TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"
    echo "[LAUNCH] PPO distributed world_size=$WORLD_SIZE_ARG command=$TORCHRUN_BIN"
    "$TORCHRUN_BIN" --standalone --nnodes=1 --nproc_per_node="$WORLD_SIZE_ARG" \
        "$TRAIN_SCRIPT" "$@" 2>&1 | awk "$WARNING_FILTER"
else
    "$PYTHON_BIN" "$TRAIN_SCRIPT" "$@" 2>&1 | awk "$WARNING_FILTER"
fi
