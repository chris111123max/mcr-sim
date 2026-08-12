#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
PROJECT_ROOT="$(cd -- "$PYTHON_ROOT/.." && pwd)"
WORKSPACE_ROOT="$(cd -- "$PROJECT_ROOT/../.." && pwd)"

SETUP_MCR_SOFA="${SETUP_MCR_SOFA:-$WORKSPACE_ROOT/setup_mcr_sofa.sh}"
ENSURE_MCR_GUI="${ENSURE_MCR_GUI:-$WORKSPACE_ROOT/ensure_mcr_gui.sh}"
for required_script in "$SETUP_MCR_SOFA" "$ENSURE_MCR_GUI"; do
    if [[ ! -f "$required_script" ]]; then
        echo "[ERROR] Cannot find required environment script: $required_script"
        exit 1
    fi
done
source "$SETUP_MCR_SOFA"
source "$ENSURE_MCR_GUI"

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
if [[ -n "${CONDA_PREFIX:-}" ]]; then
    CONDA_LIB_SUFFIX=":$CONDA_PREFIX/lib"
fi
export LD_LIBRARY_PATH="$SOFA_BUILD/lib:$SOFA_BUILD_PLUGINS/lib:$STLIB_LIB:$SOFTROBOTS_LIB:$BEAMADAPTER_LIB$CONDA_LIB_SUFFIX${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

export SOFA_PLUGIN_PATH="$SOFA_BUILD/lib:$SOFA_BUILD_PLUGINS/lib:$STLIB_LIB:$SOFTROBOTS_LIB:$BEAMADAPTER_LIB${SOFA_PLUGIN_PATH:+:$SOFA_PLUGIN_PATH}"

cd "$PYTHON_ROOT"

MODEL_PATH="${MODEL_PATH:-$PYTHON_ROOT/run_mul/sac_10mm_V1_gate_fsm_trainfreq2_buf200k_human_retrain_V1_only_aortic_10mm_20260606_111201/models/sac_mcr_10mm_all_vessels_privileged_V1_only_ckpt_800000_steps.zip}"

TIME_STEP="${TIME_STEP:-0.01}"
FRAME_SKIP="${FRAME_SKIP:-1}"
MAX_STEPS="${MAX_STEPS:-4096}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-4096}"
TARGET_THRESHOLD="${TARGET_THRESHOLD:-0.003}"
START_WINDOW_MM="${START_WINDOW_MM:-10.0}"
TARGET_WINDOW_MM="${TARGET_WINDOW_MM:-10.0}"
INITIAL_ORIENTATION_MAX_ANGLE_DEG="${INITIAL_ORIENTATION_MAX_ANGLE_DEG:-10.0}"
ENTRY_TANGENT_POINTS="${ENTRY_TANGENT_POINTS:-5}"

normalized_args=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --force-model)
            [[ $# -ge 2 ]] || {
                echo "[ERROR] --force-model requires a value"
                exit 2
            }

            normalized_args+=("--force-model" "$2")
            shift 2
            ;;

        --force)
            [[ $# -ge 2 ]] || {
                echo "[ERROR] --force requires a value"
                exit 2
            }

            if [[ "$2" == "model" ]]; then
                [[ $# -ge 3 ]] || {
                    echo "[ERROR] --force model requires a value"
                    exit 2
                }

                normalized_args+=("--force-model" "$3")
                shift 3
            else
                normalized_args+=("--force-model" "$2")
                shift 2
            fi
            ;;

        *)
            normalized_args+=("$1")
            shift
            ;;
    esac
done

echo "=============================================================="
echo " Starting MCR SOFA GUI inference: $MAX_STEPS steps, no no-progress termination "
echo "=============================================================="
echo "MODEL_PATH=$MODEL_PATH"
echo "Python=$(which python)"
echo "SOFA_ROOT=$SOFA_ROOT"
echo "Timing: dt=$TIME_STEP frame_skip=$FRAME_SKIP max_steps=$MAX_STEPS max_episode_steps=$MAX_EPISODE_STEPS"
echo "Target threshold: $TARGET_THRESHOLD m"
echo "Randomization: start_window=${START_WINDOW_MM}mm target_window=${TARGET_WINDOW_MM}mm, initial angle=${INITIAL_ORIENTATION_MAX_ANGLE_DEG}deg, entry_tangent_points=${ENTRY_TANGENT_POINTS}"
echo "No-progress gate termination: disabled by default. Add --enable-no-progress-termination to restore it."

"$PYTHON_BIN" "$PYTHON_ROOT/testing/py/run_trained_mcr_sofa_gui.py" \
    --model "$MODEL_PATH" \
    --target-threshold "$TARGET_THRESHOLD" \
    --vessel-alpha 0.35 \
    --sleep 0.05 \
    --time-step "$TIME_STEP" \
    --frame-skip "$FRAME_SKIP" \
    --max-steps "$MAX_STEPS" \
    --max-episode-steps "$MAX_EPISODE_STEPS" \
    --start-window-mm "$START_WINDOW_MM" \
    --target-window-mm "$TARGET_WINDOW_MM" \
    --initial-orientation-max-angle-deg "$INITIAL_ORIENTATION_MAX_ANGLE_DEG" \
    --entry-tangent-points "$ENTRY_TANGENT_POINTS" \
    "${normalized_args[@]}" \
    2>&1 | grep --line-buffered -v "Determinant is null"
