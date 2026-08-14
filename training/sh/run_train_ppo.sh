#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
PROJECT_ROOT="$(cd -- "$PYTHON_ROOT/.." && pwd)"
WORKSPACE_ROOT="$(cd -- "$PROJECT_ROOT/../.." && pwd)"

# ``--nohup`` is owned by this launcher, not train_ppo.py. It creates the
# exact run directory up front and captures torchrun/native output alongside
# the per-rank Python logs written later by the training process.
MANAGED_NOHUP=0
FORWARD_ARGS=()
for arg in "$@"; do
    if [[ "$arg" == "--nohup" ]]; then
        MANAGED_NOHUP=1
    else
        FORWARD_ARGS+=("$arg")
    fi
done
set -- "${FORWARD_ARGS[@]}"

if ((MANAGED_NOHUP == 1)) && [[ "${MCR_MANAGED_NOHUP_CHILD:-0}" != "1" ]]; then
    EXP_NAME_ARG=""
    LOG_ROOT_ARG="$PROJECT_ROOT/training_runs"
    FORCE_MODEL_ARG=""
    ARGS=("$@")
    for ((i = 0; i < ${#ARGS[@]}; i++)); do
        case "${ARGS[$i]}" in
            --exp-name)
                if ((i + 1 >= ${#ARGS[@]})); then echo "[ERROR] --exp-name requires a value"; exit 2; fi
                EXP_NAME_ARG="${ARGS[$((i + 1))]}" ;;
            --exp-name=*) EXP_NAME_ARG="${ARGS[$i]#*=}" ;;
            --log-root)
                if ((i + 1 >= ${#ARGS[@]})); then echo "[ERROR] --log-root requires a value"; exit 2; fi
                LOG_ROOT_ARG="${ARGS[$((i + 1))]}" ;;
            --log-root=*) LOG_ROOT_ARG="${ARGS[$i]#*=}" ;;
            --force-model)
                if ((i + 1 >= ${#ARGS[@]})); then echo "[ERROR] --force-model requires a value"; exit 2; fi
                FORCE_MODEL_ARG="${ARGS[$((i + 1))]}" ;;
            --force-model=*) FORCE_MODEL_ARG="${ARGS[$i]#*=}" ;;
        esac
    done
    if [[ -z "$EXP_NAME_ARG" ]]; then
        echo "[ERROR] Managed --nohup requires an explicit --exp-name."
        exit 2
    fi
    if [[ ! "$EXP_NAME_ARG" =~ ^[A-Za-z0-9._-]+$ ]]; then
        echo "[ERROR] --exp-name may contain only letters, digits, dot, underscore, and hyphen."
        exit 2
    fi
    if [[ "$LOG_ROOT_ARG" != /* ]]; then
        LOG_ROOT_ARG="$PROJECT_ROOT/$LOG_ROOT_ARG"
    fi
    RUN_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
    FORCED_TAG=""
    if [[ -n "$FORCE_MODEL_ARG" ]]; then FORCED_TAG="_${FORCE_MODEL_ARG}_only"; fi
    RUN_DIR="$LOG_ROOT_ARG/${EXP_NAME_ARG}${FORCED_TAG}_${RUN_TIMESTAMP}"
    mkdir -p "$RUN_DIR/logs"
    LAUNCHER_LOG="$RUN_DIR/logs/launcher.log"
    nohup env \
        MCR_MANAGED_NOHUP_CHILD=1 \
        MCR_RUN_TIMESTAMP="$RUN_TIMESTAMP" \
        bash "$SCRIPT_DIR/run_train_ppo.sh" "$@" \
        >"$LAUNCHER_LOG" 2>&1 </dev/null &
    TRAIN_PID=$!
    echo "[STARTED] algorithm=ppo pid=$TRAIN_PID"
    echo "[STARTED] run_dir=$RUN_DIR"
    echo "[STARTED] launcher_log=$LAUNCHER_LOG"
    exit 0
fi

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
