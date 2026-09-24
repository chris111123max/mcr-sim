#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME="${SOFA26_RUNTIME:-$HERE/_runtime}"
ENV_PREFIX="${SOFA26_ENV_PREFIX:-$RUNTIME/conda}"
BUILD_ROOT="$RUNTIME/build"
INSTALL_ROOT="$RUNTIME/install"

DEFAULT_CONDA="/data/home/3220251075/mcr_sim/mcr_env/miniforge3/bin/conda"
CONDA_BIN="${CONDA_BIN:-$DEFAULT_CONDA}"

if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
    echo "[ERROR] SOFA26 env missing. Run install_and_test_v26_06.sh first."
    return 2 2>/dev/null || exit 2
fi

# shellcheck disable=SC1091
source "$("$CONDA_BIN" info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"

export SOFA26_RUNTIME="$RUNTIME"
export SOFA_ROOT="$BUILD_ROOT"
export SOFAPYTHON3_ROOT="$BUILD_ROOT/external_directories/SofaPython3"

export SOFA_PLUGIN_PATH="$BUILD_ROOT:$BUILD_ROOT/external_directories/SofaPython3:$BUILD_ROOT/external_directories/BeamAdapter:$INSTALL_ROOT/plugins${SOFA_PLUGIN_PATH:+:$SOFA_PLUGIN_PATH}"
export LD_LIBRARY_PATH="$BUILD_ROOT/lib:$BUILD_ROOT/external_directories/SofaPython3/lib:$BUILD_ROOT/external_directories/BeamAdapter/lib:$INSTALL_ROOT/lib:$ENV_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

PY_PATHS=(
    "$BUILD_ROOT/lib/python3/site-packages"
    "$BUILD_ROOT/external_directories/SofaPython3/lib/python3/site-packages"
    "$INSTALL_ROOT/lib/python3/site-packages"
)

for p in "${PY_PATHS[@]}"; do
    if [[ -d "$p" ]]; then
        export PYTHONPATH="$p${PYTHONPATH:+:$PYTHONPATH}"
    fi
done
