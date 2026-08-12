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

export SOFA_ROOT="$SOFA_BUILD"
export SOFAPYTHON3_ROOT="$SOFA_BUILD"
export PYTHONPATH="$SOFA_BUILD/lib/python3/site-packages:$STLIB_ROOT:$STLIB_ROOT/python:$STLIB_ROOT/python3/src:$PYTHON_ROOT${PYTHONPATH:+:$PYTHONPATH}"

CONDA_LIB_SUFFIX=""
if [[ -n "${CONDA_PREFIX:-}" ]]; then
    CONDA_LIB_SUFFIX=":$CONDA_PREFIX/lib"
fi
export LD_LIBRARY_PATH="$SOFA_BUILD/lib:$SOFA_BUILD/external_directories/STLIB/lib:$SOFA_BUILD/external_directories/SoftRobots/lib:$SOFA_BUILD/external_directories/BeamAdapter/lib$CONDA_LIB_SUFFIX${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export SOFA_PLUGIN_PATH="$SOFA_BUILD/lib:$SOFA_BUILD/external_directories/STLIB/lib:$SOFA_BUILD/external_directories/SoftRobots/lib:$SOFA_BUILD/external_directories/BeamAdapter/lib${SOFA_PLUGIN_PATH:+:$SOFA_PLUGIN_PATH}"

# Ascend NPUs are not graphics devices. Mesa software EGL is the portable
# rendering backend on this display-less worker.
export LIBGL_ALWAYS_SOFTWARE="${LIBGL_ALWAYS_SOFTWARE:-1}"

exec "$PYTHON_BIN" "$PYTHON_ROOT/testing/test_gui/run_mcr_web_viewer.py" "$@"
