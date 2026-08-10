#!/usr/bin/env bash
set -euo pipefail

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
STLIB_ROOT="${STLIB_ROOT:-$WORKSPACE_ROOT/mcr_env/sofa/src/STLIB}"

export SOFA_ROOT="$SOFA_BUILD"
export SOFAPYTHON3_ROOT="$SOFA_BUILD"
export MCR_GUI_SOFA_DT="${MCR_GUI_SOFA_DT:-0.001}"

export PYTHONPATH="$SOFA_BUILD/lib/python3/site-packages:$STLIB_ROOT:$STLIB_ROOT/python:$STLIB_ROOT/python3/src:$PYTHON_ROOT${PYTHONPATH:+:$PYTHONPATH}"

CONDA_LIB_SUFFIX=""
if [[ -n "${CONDA_PREFIX:-}" ]]; then
    CONDA_LIB_SUFFIX=":$CONDA_PREFIX/lib"
fi
export LD_LIBRARY_PATH="$SOFA_BUILD/lib:$SOFA_BUILD/external_directories/STLIB/lib:$SOFA_BUILD/external_directories/SoftRobots/lib:$SOFA_BUILD/external_directories/BeamAdapter/lib$CONDA_LIB_SUFFIX${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

export SOFA_PLUGIN_PATH="$SOFA_BUILD/lib:$SOFA_BUILD/external_directories/STLIB/lib:$SOFA_BUILD/external_directories/SoftRobots/lib:$SOFA_BUILD/external_directories/BeamAdapter/lib${SOFA_PLUGIN_PATH:+:$SOFA_PLUGIN_PATH}"

SOFAPYTHON3_LIB="${SOFAPYTHON3_LIB:-$SOFA_BUILD/lib/libSofaPython3.so}"
SOFTROBOTS_LIB="${SOFTROBOTS_LIB:-$SOFA_BUILD/external_directories/SoftRobots/lib/libSoftRobots.so}"
BEAMADAPTER_LIB="${BEAMADAPTER_LIB:-$SOFA_BUILD/external_directories/BeamAdapter/lib/libBeamAdapter.so}"
SCENE_FILE="$PYTHON_ROOT/scene/example_aortic_arch.py"

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
