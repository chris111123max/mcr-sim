#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_ROOT="$SCRIPT_DIR"
PROJECT_ROOT="$(cd -- "$PYTHON_ROOT/.." && pwd)"
WORKSPACE_ROOT="$(cd -- "$PROJECT_ROOT/../.." && pwd)"

SETUP_MCR_SOFA="${SETUP_MCR_SOFA:-$WORKSPACE_ROOT/setup_mcr_sofa.sh}"
if [[ ! -f "$SETUP_MCR_SOFA" ]]; then
    echo "[ERROR] Cannot find SOFA setup script: $SETUP_MCR_SOFA"
    exit 1
fi
source "$SETUP_MCR_SOFA"

SOFA_BUILD_DEFAULT="$WORKSPACE_ROOT/mcr_env/sofa/build_plugins"
SOFA_BUILD="${SOFA_BUILD:-${SOFA_ROOT:-$SOFA_BUILD_DEFAULT}}"
SOFA_BUILD_PLUGINS="${SOFA_BUILD_PLUGINS:-${SOFAPYTHON3_ROOT:-$SOFA_BUILD_DEFAULT}}"
STLIB_ROOT="${STLIB_ROOT:-$WORKSPACE_ROOT/mcr_env/sofa/src/STLIB}"
PYTHON_BIN="${PYTHON_BIN:-python}"

export SOFA_ROOT="$SOFA_BUILD"
export SOFAPYTHON3_ROOT="$SOFA_BUILD_PLUGINS"
export PYTHONPATH="$SOFA_BUILD/lib/python3/site-packages:$SOFA_BUILD_PLUGINS/lib/python3/site-packages:$STLIB_ROOT:$STLIB_ROOT/python:$STLIB_ROOT/python3/src:$PYTHON_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# Keep the historical checkpoint as a project-relative default while allowing
# each deployment to select a model through MODEL_PATH.
MODEL_PATH="${MODEL_PATH:-$PYTHON_ROOT/runs_tri/centerline_light_2mm_from_3mm_aortic_2mm_20260517_184506/models/sac_mcr_2mm_V1_y_noS_ckpt_3800000_steps.zip}"

cd "$PYTHON_ROOT"

# 3. 运行模型，并把外部命令行参数继续传给 run_trained_mcr_ros.py
# 例如：
# ./run_inference_closed_loop.sh --force-model V1
# ./run_inference_closed_loop.sh --force-model 0207 --max-episodes 3
PYTHONUNBUFFERED=1 "$PYTHON_BIN" "$PYTHON_ROOT/testing/py/run_trained_mcr_ros.py" \
  --model "$MODEL_PATH" \
  "$@" \
  2>&1 | grep --line-buffered -v "Determinant is null"
