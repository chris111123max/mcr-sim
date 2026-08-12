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
PYTHON_BIN="${PYTHON_BIN:-python}"

# Keep GUI runtime libraries in the persistent mcr_sofa Conda environment.
# Worker-local apt/system packages disappear when a SCOW job is recreated.
PYTHON_PREFIX="$($PYTHON_BIN -c 'import sys; print(sys.prefix)')"
PERSISTENT_GUI_LIB="$PYTHON_PREFIX/lib"

export SOFA_ROOT="$SOFA_BUILD"
export SOFAPYTHON3_ROOT="$SOFA_BUILD"
export PYTHONPATH="$SOFA_BUILD/lib/python3/site-packages:$STLIB_ROOT:$STLIB_ROOT/python:$STLIB_ROOT/python3/src:$PYTHON_ROOT${PYTHONPATH:+:$PYTHONPATH}"

export LD_LIBRARY_PATH="$PERSISTENT_GUI_LIB:$SOFA_BUILD/lib:$SOFA_BUILD/external_directories/STLIB/lib:$SOFA_BUILD/external_directories/SoftRobots/lib:$SOFA_BUILD/external_directories/BeamAdapter/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export SOFA_PLUGIN_PATH="$SOFA_BUILD/lib:$SOFA_BUILD/external_directories/STLIB/lib:$SOFA_BUILD/external_directories/SoftRobots/lib:$SOFA_BUILD/external_directories/BeamAdapter/lib${SOFA_PLUGIN_PATH:+:$SOFA_PLUGIN_PATH}"

if ! find "$PERSISTENT_GUI_LIB" -maxdepth 1 \
    \( -name 'libGLEW.so.2.2' -o -name 'libGLEW.so.2.2.*' \) \
    -print -quit 2>/dev/null | grep -q .; then
    echo "[ERROR] Persistent GUI library is missing: libGLEW.so.2.2"
    echo "[ERROR] Expected under: $PERSISTENT_GUI_LIB"
    echo "[HINT]  Install once into the persistent mcr_sofa Conda environment:"
    echo "        conda install -n mcr_sofa -c conda-forge 'glew=2.2.*'"
    exit 1
fi

# Ascend NPUs are not graphics devices. Mesa software EGL is the portable
# rendering backend on this display-less worker.
export LIBGL_ALWAYS_SOFTWARE="${LIBGL_ALWAYS_SOFTWARE:-1}"

exec "$PYTHON_BIN" "$PYTHON_ROOT/testing/test_gui/run_mcr_web_viewer.py" "$@"
