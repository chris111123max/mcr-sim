#!/usr/bin/env bash
set -euo pipefail

# 激活mcr_sofa虚拟环境并设置SOFA运行路径
source /data/home/3220251075/mcr_sim/setup_mcr_sofa.sh
source /data/home/3220251075/mcr_sim/ensure_mcr_gui.sh

SOFA_BUILD=/data/home/3220251075/mcr_sim/mcr_env/sofa/build_plugins
SOFA_BUILD_PLUGINS=/data/home/3220251075/mcr_sim/mcr_env/sofa/build_plugins

STLIB_ROOT=/data/home/3220251075/mcr_sim/mcr_env/sofa/src/STLIB
SOFTROBOTS_LIB="$SOFA_BUILD_PLUGINS/external_directories/SoftRobots/lib"
BEAMADAPTER_LIB="$SOFA_BUILD_PLUGINS/external_directories/BeamAdapter/lib"
STLIB_LIB="$SOFA_BUILD_PLUGINS/external_directories/STLIB/lib"

PROJECT_PY=/data/home/3220251075/mcr_sim/mcr_project/mCR_simulator-master/python

export SOFA_ROOT="$SOFA_BUILD"
export SOFAPYTHON3_ROOT="$SOFA_BUILD_PLUGINS"

export PYTHONPATH="$SOFA_BUILD/lib/python3/site-packages:$SOFA_BUILD_PLUGINS/lib/python3/site-packages:$STLIB_ROOT:$STLIB_ROOT/python:$STLIB_ROOT/python3/src:$PROJECT_PY:${PYTHONPATH:-}"

export LD_LIBRARY_PATH="$SOFA_BUILD/lib:$SOFA_BUILD_PLUGINS/lib:$STLIB_LIB:$SOFTROBOTS_LIB:$BEAMADAPTER_LIB:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

export SOFA_PLUGIN_PATH="$SOFA_BUILD/lib:$SOFA_BUILD_PLUGINS/lib:$STLIB_LIB:$SOFTROBOTS_LIB:$BEAMADAPTER_LIB:${SOFA_PLUGIN_PATH:-}"

cd "$PROJECT_PY" || exit 1

MODEL_PATH="${MODEL_PATH:-/data/home/3220251075/mcr_sim/mcr_project/mCR_simulator-master/python/run_mul/sac_10mm_V1_gate_fsm_trainfreq2_buf200k_human_retrain_V1_only_aortic_10mm_20260606_111201/models/sac_mcr_10mm_all_vessels_privileged_V1_only_ckpt_800000_steps.zip}"

TIME_STEP="${TIME_STEP:-0.1}"
FRAME_SKIP="${FRAME_SKIP:-1}"
MAX_STEPS="${MAX_STEPS:-2000}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-2000}"
TARGET_THRESHOLD="${TARGET_THRESHOLD:-0.010}"
START_TARGET_RANDOM_RADIUS="${START_TARGET_RANDOM_RADIUS:-0.002}"
INITIAL_ORIENTATION_MAX_ANGLE_DEG="${INITIAL_ORIENTATION_MAX_ANGLE_DEG:-30.0}"
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
echo " Starting MCR SOFA GUI inference: 2000 steps, no no-progress termination "
echo "=============================================================="
echo "MODEL_PATH=$MODEL_PATH"
echo "Python=$(which python)"
echo "SOFA_ROOT=$SOFA_ROOT"
echo "Timing: dt=$TIME_STEP frame_skip=$FRAME_SKIP max_steps=$MAX_STEPS max_episode_steps=$MAX_EPISODE_STEPS"
echo "Target threshold: $TARGET_THRESHOLD m"
echo "Randomization: start/target radius=${START_TARGET_RANDOM_RADIUS}m, initial angle=${INITIAL_ORIENTATION_MAX_ANGLE_DEG}deg, entry_tangent_points=${ENTRY_TANGENT_POINTS}"
echo "No-progress gate termination: disabled by default. Add --enable-no-progress-termination to restore it."

python testing/py/run_trained_mcr_sofa_gui.py \
    --model "$MODEL_PATH" \
    --target-threshold "$TARGET_THRESHOLD" \
    --vessel-alpha 0.35 \
    --sleep 0.05 \
    --time-step "$TIME_STEP" \
    --frame-skip "$FRAME_SKIP" \
    --max-steps "$MAX_STEPS" \
    --max-episode-steps "$MAX_EPISODE_STEPS" \
    --start-target-random-radius "$START_TARGET_RANDOM_RADIUS" \
    --initial-orientation-max-angle-deg "$INITIAL_ORIENTATION_MAX_ANGLE_DEG" \
    --entry-tangent-points "$ENTRY_TANGENT_POINTS" \
    "${normalized_args[@]}" \
    2>&1 | grep --line-buffered -v "Determinant is null"
