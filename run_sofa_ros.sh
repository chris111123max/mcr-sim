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

export SOFA_ROOT="$SOFA_BUILD"
export SOFAPYTHON3_ROOT="$SOFA_BUILD_PLUGINS"
export PYTHONPATH="$SOFA_BUILD/lib/python3/site-packages:$SOFA_BUILD_PLUGINS/lib/python3/site-packages:$STLIB_ROOT:$STLIB_ROOT/python:$STLIB_ROOT/python3/src:$PYTHON_ROOT${PYTHONPATH:+:$PYTHONPATH}"

SOFAPYTHON3_LIB="${SOFAPYTHON3_LIB:-$SOFA_BUILD_PLUGINS/lib/libSofaPython3.so}"
SOFTROBOTS_LIB="${SOFTROBOTS_LIB:-$SOFA_BUILD_PLUGINS/external_directories/SoftRobots/lib/libSoftRobots.so}"
BEAMADAPTER_LIB="${BEAMADAPTER_LIB:-$SOFA_BUILD_PLUGINS/external_directories/BeamAdapter/lib/libBeamAdapter.so}"
SCENE_FILE="$PYTHON_ROOT/scene/example_aortic_arch_ros.py"

if [[ -n "${RUNSOFA_BIN:-}" ]]; then
    :
elif [[ -x "$SOFA_BUILD/bin/runSofa-21.12.00" ]]; then
    RUNSOFA_BIN="$SOFA_BUILD/bin/runSofa-21.12.00"
else
    RUNSOFA_BIN="$SOFA_BUILD/bin/runSofa"
fi

for path in "$RUNSOFA_BIN" "$SOFAPYTHON3_LIB" "$SOFTROBOTS_LIB" "$BEAMADAPTER_LIB" "$SCENE_FILE"; do
    if [[ ! -e "$path" ]]; then
        echo "[ERROR] Required runtime path does not exist: $path"
        exit 1
    fi
done

cd "$SOFA_BUILD/bin"

"$RUNSOFA_BIN" \
    -l "$SOFAPYTHON3_LIB" \
    -l "$SOFTROBOTS_LIB" \
    -l "$BEAMADAPTER_LIB" \
    "$SCENE_FILE" \
    2>&1 | grep --line-buffered -v "Determinant is null"
