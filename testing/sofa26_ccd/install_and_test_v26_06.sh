#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME="${SOFA26_RUNTIME:-$HERE/_runtime}"
ENV_PREFIX="${SOFA26_ENV_PREFIX:-$RUNTIME/conda}"
SRC_ROOT="$RUNTIME/src"
BUILD_ROOT="$RUNTIME/build"
INSTALL_ROOT="$RUNTIME/install"
LOG_ROOT="$RUNTIME/logs"
JOBS="${SOFA26_JOBS:-10}"

DEFAULT_CONDA="/data/home/3220251075/mcr_sim/mcr_env/miniforge3/bin/conda"
CONDA_BIN="${CONDA_BIN:-$DEFAULT_CONDA}"

mkdir -p "$RUNTIME" "$SRC_ROOT" "$LOG_ROOT"

if [[ "$(uname -m)" != "aarch64" ]]; then
    echo "[WARN] Expected aarch64 server, got $(uname -m)"
fi

if [[ ! -x "$CONDA_BIN" ]]; then
    echo "[ERROR] conda not found: $CONDA_BIN"
    exit 2
fi

echo "[SOFA26] runtime=$RUNTIME"
echo "[SOFA26] env=$ENV_PREFIX"
echo "[SOFA26] jobs=$JOBS"

if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
    packages=(
        "python=3.12"
        "cmake>=3.22"
        ninja
        git
        c-compiler
        cxx-compiler
        ccache
        "pybind11=2.12"
        numpy
        scipy
        eigen
        boost-cpp
        tinyxml2
        cxxopts
        nlohmann_json
        zlib
        metis
    )
    "$CONDA_BIN" create -y -p "$ENV_PREFIX" -c conda-forge "${packages[@]}"
fi

# shellcheck disable=SC1091
source "$("$CONDA_BIN" info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"

CC_BIN="${CC:-}"
CXX_BIN="${CXX:-}"
if [[ -z "$CC_BIN" ]]; then
    CC_BIN="$(find "$ENV_PREFIX/bin" -maxdepth 1 -type f -name '*-cc' | head -n1 || true)"
fi
if [[ -z "$CXX_BIN" ]]; then
    CXX_BIN="$(find "$ENV_PREFIX/bin" -maxdepth 1 -type f \( -name '*-c++' -o -name '*-g++' \) | head -n1 || true)"
fi
if [[ -z "$CC_BIN" || -z "$CXX_BIN" || ! -x "$CC_BIN" || ! -x "$CXX_BIN" ]]; then
    echo "[ERROR] conda C/C++ compiler not found"
    exit 3
fi

clone_tagged() {
    local url="$1"
    local dir="$2"
    local tag="$3"
    if [[ -d "$dir/.git" ]]; then
        echo "[SOFA26] source exists: $dir"
        git -C "$dir" fetch --tags --depth 1 origin "$tag"
        git -C "$dir" checkout -f "$tag"
    else
        git clone --depth 1 --branch "$tag" "$url" "$dir"
    fi
}

clone_tagged https://github.com/sofa-framework/sofa.git "$SRC_ROOT/sofa" v26.06
clone_tagged https://github.com/sofa-framework/SofaPython3.git "$SRC_ROOT/SofaPython3" v26.06
clone_tagged https://github.com/sofa-framework/BeamAdapter.git "$SRC_ROOT/BeamAdapter" v26.06

rm -rf "$BUILD_ROOT"
mkdir -p "$BUILD_ROOT" "$INSTALL_ROOT"

EXTERNAL_DIRS="$SRC_ROOT/SofaPython3;$SRC_ROOT/BeamAdapter"

echo "[SOFA26] configure"
cmake -S "$SRC_ROOT/sofa" -B "$BUILD_ROOT" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_ROOT" \
    -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" \
    -DCMAKE_C_COMPILER="$CC_BIN" \
    -DCMAKE_CXX_COMPILER="$CXX_BIN" \
    -DSOFA_ALLOW_FETCH_DEPENDENCIES=ON \
    -DSOFA_BUILD_TESTS=OFF \
    -DSP3_BUILD_TEST=OFF \
    -DBEAMADAPTER_BUILD_TESTS=OFF \
    -DAPPLICATION_RUNSOFA=OFF \
    -DLIBRARY_SOFA_GUI=OFF \
    -DSOFA_BUILD_RELEASE_PACKAGE=OFF \
    -DSOFA_INSTALL_RESOURCES_FILES=OFF \
    -DSOFA_USE_CCACHE=ON \
    -DSP3_LINK_TO_USER_SITE=OFF \
    -DPython_EXECUTABLE="$ENV_PREFIX/bin/python" \
    -DPython_ROOT_DIR="$ENV_PREFIX" \
    -DSOFA_EXTERNAL_DIRECTORIES="$EXTERNAL_DIRS" \
    2>&1 | tee "$LOG_ROOT/configure.log"

echo "[SOFA26] build"
cmake --build "$BUILD_ROOT" --parallel "$JOBS" \
    2>&1 | tee "$LOG_ROOT/build.log"

echo "[SOFA26] install into test runtime only"
cmake --install "$BUILD_ROOT" \
    2>&1 | tee "$LOG_ROOT/install.log"

cat > "$RUNTIME/build_manifest.txt" <<EOF
date=$(date -Iseconds)
arch=$(uname -m)
sofa=$(git -C "$SRC_ROOT/sofa" rev-parse HEAD)
sofapython3=$(git -C "$SRC_ROOT/SofaPython3" rev-parse HEAD)
beamadapter=$(git -C "$SRC_ROOT/BeamAdapter" rev-parse HEAD)
python=$("$ENV_PREFIX/bin/python" -V 2>&1)
cmake=$(cmake --version | head -n1)
compiler=$("$CXX_BIN" --version | head -n1)
EOF

echo "[SOFA26] run CCD preflight"
"$HERE/run_preflight.sh"

echo "[SOFA26] PASS: install/build/preflight completed"
echo "[SOFA26] manifest: $RUNTIME/build_manifest.txt"
