#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "\${BASH_SOURCE[0]}")" && pwd)"
BUILD="\${MCR_BEAM_UNILATERAL_BUILD_DIR:-$HERE/_build}"
JOBS="\${MCR_BEAM_UNILATERAL_BUILD_JOBS:-2}"

cmake -S "$HERE" -B "$BUILD" -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD" -j"$JOBS"
find "$BUILD" -name 'libMCRBeamLinearizedUnilateral.so' -print
