#!/usr/bin/env bash
set -euo pipefail

source /data/home/3220251075/mcr_sim/setup_mcr_sofa.sh

SOFA_BUILD=/data/home/3220251075/mcr_sim/mcr_env/sofa/build_plugins
PROJECT_PY=/data/home/3220251075/mcr_sim/mcr_project/mCR_simulator-master/python
STLIB_ROOT=/data/home/3220251075/mcr_sim/mcr_env/sofa/src/STLIB

export SOFA_ROOT="$SOFA_BUILD"
export SOFAPYTHON3_ROOT="$SOFA_BUILD"
export MCR_GUI_SOFA_DT="${MCR_GUI_SOFA_DT:-0.001}"

export PYTHONPATH="$SOFA_BUILD/lib/python3/site-packages:$STLIB_ROOT:$STLIB_ROOT/python:$STLIB_ROOT/python3/src:$PROJECT_PY:${PYTHONPATH:-}"

export LD_LIBRARY_PATH="$SOFA_BUILD/lib:$SOFA_BUILD/external_directories/STLIB/lib:$SOFA_BUILD/external_directories/SoftRobots/lib:$SOFA_BUILD/external_directories/BeamAdapter/lib:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

export SOFA_PLUGIN_PATH="$SOFA_BUILD/lib:$SOFA_BUILD/external_directories/STLIB/lib:$SOFA_BUILD/external_directories/SoftRobots/lib:$SOFA_BUILD/external_directories/BeamAdapter/lib:${SOFA_PLUGIN_PATH:-}"

SOFAPYTHON3_LIB="$SOFA_BUILD/lib/libSofaPython3.so"
SOFTROBOTS_LIB="$SOFA_BUILD/external_directories/SoftRobots/lib/libSoftRobots.so"
BEAMADAPTER_LIB="$SOFA_BUILD/external_directories/BeamAdapter/lib/libBeamAdapter.so"
SCENE_FILE="$PROJECT_PY/scene/example_aortic_arch.py"

if [[ -x "$SOFA_BUILD/bin/runSofa-21.12.00" ]]; then
    RUNSOFA="$SOFA_BUILD/bin/runSofa-21.12.00"
elif [[ -x "$SOFA_BUILD/bin/runSofa" ]]; then
    RUNSOFA="$SOFA_BUILD/bin/runSofa"
else
    echo "[ERROR] 找不到 runSofa 可执行文件"
    find "$SOFA_BUILD/bin" -maxdepth 1 -name 'runSofa*' -ls
    exit 1
fi

for path in \
    "$SOFAPYTHON3_LIB" \
    "$SOFTROBOTS_LIB" \
    "$BEAMADAPTER_LIB" \
    "$SCENE_FILE"
do
    if [[ ! -e "$path" ]]; then
        echo "[ERROR] 文件不存在：$path"
        exit 1
    fi
done

echo "=============================================================="
echo " Starting SOFA example scene"
echo "=============================================================="
echo "runSofa    : $RUNSOFA"
echo "SOFA_ROOT  : $SOFA_ROOT"
echo "Scene      : $SCENE_FILE"
echo "SOFA dt    : $MCR_GUI_SOFA_DT"
echo "=============================================================="

cd "$SOFA_BUILD/bin"

"$RUNSOFA" \
    -l "$SOFAPYTHON3_LIB" \
    -l "$SOFTROBOTS_LIB" \
    -l "$BEAMADAPTER_LIB" \
    "$SCENE_FILE" \
    2>&1 | grep --line-buffered -v "Determinant is null"
