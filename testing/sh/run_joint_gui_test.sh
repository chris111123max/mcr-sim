#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# SOFA 官方 GUI + SAC 模型 + PyBullet/MoveIt 闭环联合测试
# ============================================================

# SOFA runtime roots
SOFA_BUILD=/home/chen/SOFAA/sofa_ws/sofa/build
SOFA_BUILD_PLUGINS=/home/chen/SOFAA/sofa_ws/sofa/build_plugins
BEAMADAPTER_LIB=/home/chen/SOFAA/sofa_ws/sofa/build_plugins/external_directories/BeamAdapter/lib

export SOFA_ROOT="$SOFA_BUILD"
export SOFAPYTHON3_ROOT="$SOFA_BUILD_PLUGINS"

# Python module paths
export PYTHONPATH="$SOFA_BUILD/lib/python3/site-packages:$SOFA_BUILD_PLUGINS/lib/python3/site-packages:/home/chen/SOFAA/sofa_ws/STLIB:/home/chen/SOFAA/sofa_ws/STLIB/python:/home/chen/SOFAA/sofa_ws/STLIB/python3/src:/home/chen/SOFAA/projects/mCR_simulator-master/python:${PYTHONPATH:-}"

# Native shared libraries required by SOFA plugins
export LD_LIBRARY_PATH="$SOFA_BUILD/lib:$SOFA_BUILD_PLUGINS/lib:$BEAMADAPTER_LIB:${LD_LIBRARY_PATH:-}"

# Let SOFA discover both core and external plugins at runtime
export SOFA_PLUGIN_PATH="$SOFA_BUILD/lib:$SOFA_BUILD_PLUGINS/lib:$BEAMADAPTER_LIB:${SOFA_PLUGIN_PATH:-}"

cd /home/chen/SOFAA/projects/mCR_simulator-master/python || exit 1

SCENE_FILE="/home/chen/SOFAA/projects/mCR_simulator-master/python/testing/py/test_ros_gui.py"

if [ ! -f "$SCENE_FILE" ]; then
    echo "[ERROR] Cannot find scene file: $SCENE_FILE"
    exit 1
fi

RUNSOFA_BIN="$SOFA_BUILD/bin/runSofa"

if [ ! -x "$RUNSOFA_BIN" ]; then
    echo "[ERROR] Cannot find executable runSofa: $RUNSOFA_BIN"
    echo "Try:"
    echo "  find /home/chen/SOFAA -name runSofa -type f 2>/dev/null"
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
