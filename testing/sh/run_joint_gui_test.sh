#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# SOFA 官方 GUI + SAC 模型 + PyBullet/MoveIt 闭环联合测试
# ============================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
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
SOFTROBOTS_LIB="$SOFA_BUILD_PLUGINS/external_directories/SoftRobots/lib"
BEAMADAPTER_LIB="$SOFA_BUILD_PLUGINS/external_directories/BeamAdapter/lib"
STLIB_LIB="$SOFA_BUILD_PLUGINS/external_directories/STLIB/lib"

export SOFA_ROOT="$SOFA_BUILD"
export SOFAPYTHON3_ROOT="$SOFA_BUILD_PLUGINS"

# Python module paths
export PYTHONPATH="$SOFA_BUILD/lib/python3/site-packages:$SOFA_BUILD_PLUGINS/lib/python3/site-packages:$STLIB_ROOT:$STLIB_ROOT/python:$STLIB_ROOT/python3/src:$PYTHON_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# Native shared libraries required by SOFA plugins
CONDA_LIB_SUFFIX=""
if [[ -n "${CONDA_PREFIX:-}" ]]; then
    CONDA_LIB_SUFFIX=":$CONDA_PREFIX/lib"
fi
export LD_LIBRARY_PATH="$SOFA_BUILD/lib:$SOFA_BUILD_PLUGINS/lib:$STLIB_LIB:$SOFTROBOTS_LIB:$BEAMADAPTER_LIB$CONDA_LIB_SUFFIX${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Let SOFA discover both core and external plugins at runtime
export SOFA_PLUGIN_PATH="$SOFA_BUILD/lib:$SOFA_BUILD_PLUGINS/lib:$STLIB_LIB:$SOFTROBOTS_LIB:$BEAMADAPTER_LIB${SOFA_PLUGIN_PATH:+:$SOFA_PLUGIN_PATH}"

cd "$PYTHON_ROOT"

SCENE_FILE="$PYTHON_ROOT/testing/py/test_ros_gui.py"

if [ ! -f "$SCENE_FILE" ]; then
    echo "[ERROR] Cannot find scene file: $SCENE_FILE"
    exit 1
fi

if [[ -n "${RUNSOFA_BIN:-}" ]]; then
    :
elif [[ -x "$SOFA_BUILD/bin/runSofa-21.12.00" ]]; then
    RUNSOFA_BIN="$SOFA_BUILD/bin/runSofa-21.12.00"
else
    RUNSOFA_BIN="$SOFA_BUILD/bin/runSofa"
fi

if [ ! -x "$RUNSOFA_BIN" ]; then
    echo "[ERROR] Cannot find executable runSofa: $RUNSOFA_BIN"
    echo "Set RUNSOFA_BIN or SOFA_BUILD to the correct external SOFA installation."
    exit 1
fi

echo "=========================================================="
echo " Starting MCR joint test in SOFA official GUI "
echo "=========================================================="
echo "[INFO] SOFA_ROOT=$SOFA_ROOT"
echo "[INFO] SOFAPYTHON3_ROOT=$SOFAPYTHON3_ROOT"
echo "[INFO] RUNSOFA_BIN=$RUNSOFA_BIN"
echo "[INFO] SCENE_FILE=$SCENE_FILE"
echo "=========================================================="

PYTHONUNBUFFERED=1 "$RUNSOFA_BIN" "$SCENE_FILE" "$@" \
  2>&1 | grep --line-buffered -v "Determinant is null"
