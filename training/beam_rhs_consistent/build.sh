#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
: "${SOFA_ROOT:?Set SOFA_ROOT to the SOFA 21.12 build/install prefix}"
: "${SOFA_SRC:?Set SOFA_SRC to the SOFA 21.12 source tree}"
cmake -S "$SCRIPT_DIR/native" -B "$SCRIPT_DIR/native/_build" -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build "$SCRIPT_DIR/native/_build" -j "${BUILD_JOBS:-2}"
