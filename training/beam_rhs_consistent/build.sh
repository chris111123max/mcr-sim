#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
: "${SOFA_ROOT:?Set SOFA_ROOT to the SOFA 21.12 build/install prefix}"
: "${SOFA_SRC:?Set SOFA_SRC to the SOFA 21.12 source tree}"
BUILD_DIR=${MCR_RHS_BUILD_DIR:-$SCRIPT_DIR/native/_build_cpp_v3}
cmake -S "$SCRIPT_DIR/native" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build "$BUILD_DIR" -j "${BUILD_JOBS:-2}"
