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
MESA_RUNTIME_ROOT="${MCR_MESA_ROOT:-$WORKSPACE_ROOT/mcr_env/mesa_llvmpipe}"

export SOFA_ROOT="$SOFA_BUILD"
export SOFAPYTHON3_ROOT="$SOFA_BUILD"
export PYTHONPATH="$SOFA_BUILD/lib/python3/site-packages:$STLIB_ROOT:$STLIB_ROOT/python:$STLIB_ROOT/python3/src:$PYTHON_ROOT${PYTHONPATH:+:$PYTHONPATH}"

if [[ -d "$MESA_RUNTIME_ROOT/lib" ]]; then
    export LD_LIBRARY_PATH="$MESA_RUNTIME_ROOT/lib:$PERSISTENT_GUI_LIB:$SOFA_BUILD/lib:$SOFA_BUILD/external_directories/STLIB/lib:$SOFA_BUILD/external_directories/SoftRobots/lib:$SOFA_BUILD/external_directories/BeamAdapter/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
else
    export LD_LIBRARY_PATH="$PERSISTENT_GUI_LIB:$SOFA_BUILD/lib:$SOFA_BUILD/external_directories/STLIB/lib:$SOFA_BUILD/external_directories/SoftRobots/lib:$SOFA_BUILD/external_directories/BeamAdapter/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
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
export EGL_PLATFORM="${EGL_PLATFORM:-surfaceless}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export GALLIUM_DRIVER="${GALLIUM_DRIVER:-llvmpipe}"

# Keep Mesa independent from the old SOFA Python environment. New conda-forge
# packages split llvmpipe into mesa-llvmpipe, which can live in MESA_RUNTIME_ROOT.
MESA_SEARCH_ROOTS=("$MESA_RUNTIME_ROOT" "$PYTHON_PREFIX")
MESA_EGL_PROVIDER="$(find "${MESA_SEARCH_ROOTS[@]}" \( -type f -o -type l \) \
    \( -name 'libEGL_mesa.so' -o -name 'libEGL_mesa.so.*' \) \
    -print -quit 2>/dev/null || true)"
MESA_EGL_VENDOR_JSON="$(find "${MESA_SEARCH_ROOTS[@]}" \( -type f -o -type l \) \
    -path '*/glvnd/egl_vendor.d/*.json' -print -quit 2>/dev/null || true)"
MESA_DRI_DRIVER="$(find "${MESA_SEARCH_ROOTS[@]}" \( -type f -o -type l \) \
    \( -name 'swrast_dri.so' -o -name 'kms_swrast_dri.so' \
       -o -name 'libgallium_dri.so' \) \
    -print -quit 2>/dev/null || true)"

# Some mesa-llvmpipe builds ship the provider but omit GLVND's tiny manifest.
# Generate it in the persistent runtime instead of depending on worker /usr.
if [[ -n "$MESA_EGL_PROVIDER" && -z "$MESA_EGL_VENDOR_JSON" ]]; then
    MESA_EGL_VENDOR_JSON="$MESA_RUNTIME_ROOT/share/glvnd/egl_vendor.d/50_mesa.json"
    mkdir -p "$(dirname -- "$MESA_EGL_VENDOR_JSON")"
    printf '%s\n' \
        '{' \
        '    "file_format_version": "1.0.0",' \
        '    "ICD": {' \
        '        "library_path": "libEGL_mesa.so.0"' \
        '    }' \
        '}' > "$MESA_EGL_VENDOR_JSON"
fi

if [[ -n "$MESA_EGL_VENDOR_JSON" ]]; then
    export __EGL_VENDOR_LIBRARY_FILENAMES="$MESA_EGL_VENDOR_JSON"
fi
if [[ -n "$MESA_DRI_DRIVER" ]]; then
    export LIBGL_DRIVERS_PATH="$(dirname -- "$MESA_DRI_DRIVER")"
fi

if [[ -z "$MESA_EGL_PROVIDER" || -z "$MESA_EGL_VENDOR_JSON" || -z "$MESA_DRI_DRIVER" ]]; then
    echo "[ERROR] Persistent Mesa EGL software renderer is incomplete."
    echo "[ERROR] EGL Mesa provider: ${MESA_EGL_PROVIDER:-missing}"
    echo "[ERROR] EGL vendor JSON: ${MESA_EGL_VENDOR_JSON:-missing}"
    echo "[ERROR] DRI software driver: ${MESA_DRI_DRIVER:-missing}"
    echo "[HINT]  Create the persistent runtime once:"
    echo "        conda create -p '$MESA_RUNTIME_ROOT' -c conda-forge 'mesa-llvmpipe=26.1.*' -y"
    exit 1
fi

exec "$PYTHON_BIN" "$PYTHON_ROOT/testing/test_gui/run_mcr_web_viewer.py" "$@"
