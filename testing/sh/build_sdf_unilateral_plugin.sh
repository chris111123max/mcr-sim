#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_ROOT="$(cd "$HERE/../.." && pwd)"
PROJECT_ROOT="$(cd "$PY_ROOT/.." && pwd)"
WORKSPACE_ROOT="$(cd "$PROJECT_ROOT/../.." && pwd)"

SETUP_MCR_SOFA="${SETUP_MCR_SOFA:-$WORKSPACE_ROOT/setup_mcr_sofa.sh}"
if [[ -f "$SETUP_MCR_SOFA" ]]; then
    # shellcheck disable=SC1090
    source "$SETUP_MCR_SOFA"
fi

SOFA_BUILD_DEFAULT="$WORKSPACE_ROOT/mcr_env/sofa/build_plugins"
SOFA_BUILD="${SOFA_BUILD:-${SOFA_ROOT:-$SOFA_BUILD_DEFAULT}}"
PLUGIN_ROOT="$PY_ROOT/cpp/SDFUnilateralConstraint"
BUILD_DIR="$PLUGIN_ROOT/build"

if [[ ! -d "$SOFA_BUILD" ]]; then
    echo "[ERROR] SOFA build directory not found: $SOFA_BUILD" >&2
    exit 2
fi

CONFIG_PATH="$(find "$SOFA_BUILD" -type f \( -name 'SofaConstraintConfig.cmake' -o -name 'sofaconstraint-config.cmake' \) -print -quit 2>/dev/null || true)"
if [[ -z "$CONFIG_PATH" ]]; then
    echo "[ERROR] SofaConstraintConfig.cmake not found under: $SOFA_BUILD" >&2
    echo "[INFO] Inspect with: find '$SOFA_BUILD' -iname '*SofaConstraint*Config*.cmake'" >&2
    exit 3
fi

echo "[SDF_UNILATERAL_BUILD] workspace=$WORKSPACE_ROOT"
echo "[SDF_UNILATERAL_BUILD] sofa_build=$SOFA_BUILD"
echo "[SDF_UNILATERAL_BUILD] SofaConstraintConfig=$CONFIG_PATH"
echo "[SDF_UNILATERAL_BUILD] compiler=$(command -v c++ || true)"
echo "[SDF_UNILATERAL_BUILD] cmake=$(command -v cmake || true)"

rm -rf "$BUILD_DIR"

cmake -S "$PLUGIN_ROOT" -B "$BUILD_DIR" \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="$SOFA_BUILD"

cmake --build "$BUILD_DIR" -j2

echo "[SDF_UNILATERAL_BUILD] built libraries:"
find "$BUILD_DIR" -type f \( -name '*.so' -o -name '*SDFUnilateralConstraint*' \) -print
