#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
: "${SOFA_ROOT:?Set SOFA_ROOT to the SOFA 21.12 build/install prefix}"
: "${SOFA_SRC:?Set SOFA_SRC to the SOFA 21.12 source tree}"
BUILD_DIR=${MCR_RHS_BUILD_DIR:-$SCRIPT_DIR/native/_build_cpp_v3}

# Reuse the exact Eigen header folder that built the old V2 native plugin.
# Never update conda, download dependencies or rebuild the in-use V2 plugin.
eigen_dir=""
check_eigen() {
  local candidate="${1:-}"
  if [[ -n "$candidate" && -f "$candidate/Eigen/Core" ]]; then
    eigen_dir="$candidate"
    return 0
  fi
  return 1
}

if [[ -n "${EIGEN3_INCLUDE_DIR:-}" ]]; then
  if ! check_eigen "$EIGEN3_INCLUDE_DIR"; then
    printf 'Invalid EIGEN3_INCLUDE_DIR=%s (expected Eigen/Core)\n' "$EIGEN3_INCLUDE_DIR" >&2
    exit 2
  fi
fi

if [[ -z "$eigen_dir" ]]; then
  for cache in \
    "$SCRIPT_DIR/native/_build/CMakeCache.txt" \
    "$SOFA_ROOT/CMakeCache.txt" \
    "$SOFA_SRC/CMakeCache.txt" \
    "$SOFA_SRC/sofa/CMakeCache.txt"; do
    [[ -f "$cache" ]] || continue
    cached=$(sed -nE 's/^EIGEN3_INCLUDE_DIR:(PATH|UNINITIALIZED|STRING)=(.*)$/\2/p' "$cache" | head -n 1)
    if check_eigen "$cached"; then
      printf '[V3 BUILD] Reusing Eigen headers from %s\n' "$cache"
      break
    fi
  done
fi

if [[ -z "$eigen_dir" ]]; then
  for candidate in \
    "${CONDA_PREFIX:-}/include/eigen3" \
    "${CONDA_PREFIX:-}/include" \
    "$SOFA_ROOT/include/eigen3" \
    "$SOFA_ROOT/include" \
    "$SOFA_SRC/sofa/extlibs/eigen3" \
    "$SOFA_SRC/sofa/extlibs/eigen" \
    "$SOFA_SRC/sofa/extlibs/eigen-3.3.7" \
    "$SOFA_SRC/extlibs/eigen3" \
    "/usr/include/eigen3" \
    "/usr/local/include/eigen3"; do
    if check_eigen "$candidate"; then break; fi
  done
fi

# Some torch installations bundle Eigen even when conda has no eigen package.
if [[ -z "$eigen_dir" ]]; then
  PYTHON_FOR_EIGEN=${PYTHON:-python}
  torch_eigen=$("$PYTHON_FOR_EIGEN" - <<'PY'
import importlib.util
from pathlib import Path
spec = importlib.util.find_spec("torch")
if spec and spec.origin:
    base = Path(spec.origin).resolve().parent
    for directory in (
        base / "include" / "third_party" / "eigen3",
        base / "include" / "third_party",
    ):
        if (directory / "Eigen" / "Core").is_file():
            print(directory)
            break
PY
  )
  check_eigen "$torch_eigen" || true
fi

if [[ -z "$eigen_dir" ]]; then
  echo "[V3 BUILD] No Eigen/Core in V2 cache, SOFA, conda, system or PyTorch." >&2
  echo "[V3 BUILD] Set EIGEN3_INCLUDE_DIR to the directory containing Eigen/Core." >&2
  exit 2
fi
printf '[V3 BUILD] Eigen3 include directory: %s\n' "$eigen_dir"

# -D corrects the previous failed V3 CMake cache; V2 remains untouched.
cmake -S "$SCRIPT_DIR/native" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  "-DEIGEN3_INCLUDE_DIR:PATH=$eigen_dir"
cmake --build "$BUILD_DIR" -j "${BUILD_JOBS:-2}"
