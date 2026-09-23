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

SOFA_HOME="${SOFA_HOME:-$WORKSPACE_ROOT/mcr_env/sofa}"
SOFA_BUILD_DEFAULT="$SOFA_HOME/build_plugins"
SOFA_BUILD="${SOFA_BUILD:-${SOFA_ROOT:-$SOFA_BUILD_DEFAULT}}"
SOFA_SRC="${SOFA_SRC:-$SOFA_HOME/src/sofa}"

PLUGIN_ROOT="$PY_ROOT/cpp/SDFUnilateralConstraint"
BUILD_DIR="$PLUGIN_ROOT/build"

if [[ ! -d "$SOFA_BUILD" ]]; then
    echo "[ERROR] SOFA build directory not found: $SOFA_BUILD" >&2
    exit 2
fi
if [[ ! -d "$SOFA_SRC" ]]; then
    echo "[ERROR] SOFA source directory not found: $SOFA_SRC" >&2
    exit 3
fi

echo "[SDF_UNILATERAL_BUILD] workspace=$WORKSPACE_ROOT"
echo "[SDF_UNILATERAL_BUILD] sofa_build=$SOFA_BUILD"
echo "[SDF_UNILATERAL_BUILD] sofa_src=$SOFA_SRC"
echo "[SDF_UNILATERAL_BUILD] compiler=$(command -v c++ || true)"
echo "[SDF_UNILATERAL_BUILD] cmake=$(command -v cmake || true)"

# The server's SOFA tree is a runnable build tree, not a complete installed
# development package: Config.cmake files exist, while exported *Targets.cmake
# files remain under CMakeFiles/Export.  Do not modify or install into SOFA.
# Instead, compile this plugin read-only against the existing source/build tree.

SOFA_CONSTRAINT_LIBRARY=""
if [[ -e "$SOFA_BUILD/lib/libSofaConstraint.so" ]]; then
    SOFA_CONSTRAINT_LIBRARY="$SOFA_BUILD/lib/libSofaConstraint.so"
else
    SOFA_CONSTRAINT_LIBRARY="$(find "$SOFA_BUILD" \( -type f -o -type l \) -name 'libSofaConstraint.so*' -print -quit 2>/dev/null || true)"
fi
if [[ -z "$SOFA_CONSTRAINT_LIBRARY" ]]; then
    echo "[ERROR] libSofaConstraint.so was not found under: $SOFA_BUILD" >&2
    exit 4
fi

echo "[SDF_UNILATERAL_BUILD] SofaConstraint library=$SOFA_CONSTRAINT_LIBRARY"

BOOST_INCLUDE_HINT=""
for candidate in \
    "$CONDA_PREFIX/include" \
    "/usr/include" \
    "/usr/local/include"; do
    if [[ -d "$candidate/boost" ]]; then
        BOOST_INCLUDE_HINT="$candidate"
        break
    fi
done
if [[ -z "$BOOST_INCLUDE_HINT" ]]; then
    echo "[ERROR] Boost headers were not found." >&2
    exit 5
fi
echo "[SDF_UNILATERAL_BUILD] Boost_INCLUDE_DIR=$BOOST_INCLUDE_HINT"

# Recover the include search path from the already-built SOFA tree whenever
# possible.  This is more reliable than guessing individual framework modules.
declare -a SOFA_INCLUDE_ARRAY=()

add_include_dir() {
    local d="$1"
    [[ -n "$d" && -d "$d" ]] || return 0
    local existing
    for existing in "${SOFA_INCLUDE_ARRAY[@]:-}"; do
        [[ "$existing" == "$d" ]] && return 0
    done
    SOFA_INCLUDE_ARRAY+=("$d")
}

parse_compile_command() {
    python -c '
import shlex, sys
text = sys.stdin.read().strip()
if not text:
    raise SystemExit(0)
tokens = shlex.split(text)
out = []
i = 0
while i < len(tokens):
    tok = tokens[i]
    val = None
    if tok in ("-I", "-isystem", "-iquote") and i + 1 < len(tokens):
        val = tokens[i + 1]
        i += 2
    elif tok.startswith("-I") and len(tok) > 2:
        val = tok[2:]
        i += 1
    elif tok.startswith("-isystem") and len(tok) > len("-isystem"):
        val = tok[len("-isystem"):]
        i += 1
    else:
        i += 1
    if val and val not in out:
        out.append(val)
print("\n".join(out))
'
}

SAMPLE_COMMAND=""

COMPILE_DB=""
for candidate in "$SOFA_BUILD/compile_commands.json" "$SOFA_HOME/build/compile_commands.json"; do
    if [[ -f "$candidate" ]]; then
        COMPILE_DB="$candidate"
        break
    fi
done

if [[ -n "$COMPILE_DB" ]]; then
    SAMPLE_COMMAND="$(python - "$COMPILE_DB" <<'PY'
import json, sys
p = sys.argv[1]
with open(p, "r", encoding="utf-8") as f:
    data = json.load(f)
for e in data:
    fn = str(e.get("file", ""))
    if "SofaConstraint" in fn:
        print(e.get("command") or " ".join(e.get("arguments", [])))
        break
PY
)"
    [[ -n "$SAMPLE_COMMAND" ]] && echo "[SDF_UNILATERAL_BUILD] include source=compile_commands.json"
fi

if [[ -z "$SAMPLE_COMMAND" && -f "$SOFA_BUILD/build.ninja" ]] && command -v ninja >/dev/null 2>&1; then
    SAMPLE_COMMAND="$(ninja -C "$SOFA_BUILD" -t commands SofaConstraint 2>/dev/null | grep -m1 ' -c ' || true)"
    [[ -n "$SAMPLE_COMMAND" ]] && echo "[SDF_UNILATERAL_BUILD] include source=ninja command database"
fi

if [[ -n "$SAMPLE_COMMAND" ]]; then
    while IFS= read -r d; do
        add_include_dir "$d"
    done < <(printf '%s\n' "$SAMPLE_COMMAND" | parse_compile_command)
fi

# Make the fallback independent of whether compile_commands.json was enabled.
# Every parent of a "sofa" include tree is a valid include root.
while IFS= read -r sofa_dir; do
    add_include_dir "${sofa_dir%/sofa}"
done < <(find "$SOFA_SRC" "$SOFA_BUILD" -type d -name sofa -print 2>/dev/null)

# Generated package headers can live under build subdirectories whose include
# root does not otherwise contain a lowercase sofa directory.
while IFS= read -r config_header; do
    root="${config_header%/sofa/config.h}"
    add_include_dir "$root"
done < <(find "$SOFA_BUILD" -type f -path '*/sofa/config.h' -print 2>/dev/null)

# Common third-party header roots, if already present.  No packages are
# installed by this script.
add_include_dir "$BOOST_INCLUDE_HINT"
add_include_dir "$CONDA_PREFIX/include"
add_include_dir "$CONDA_PREFIX/include/eigen3"
add_include_dir "/usr/include/eigen3"

if [[ ${#SOFA_INCLUDE_ARRAY[@]} -eq 0 ]]; then
    echo "[ERROR] Could not recover any SOFA include directories." >&2
    exit 6
fi

SOFA_INCLUDE_DIRS="$(IFS=';'; echo "${SOFA_INCLUDE_ARRAY[*]}")"
SOFA_RUNTIME_LIBRARY_DIRS="$(dirname "$SOFA_CONSTRAINT_LIBRARY");$SOFA_BUILD/lib"

echo "[SDF_UNILATERAL_BUILD] include roots=${#SOFA_INCLUDE_ARRAY[@]}"
printf '  [include] %s\n' "${SOFA_INCLUDE_ARRAY[@]}"

rm -rf "$BUILD_DIR"

cmake -S "$PLUGIN_ROOT" -B "$BUILD_DIR" \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DMCR_SOFA_BUILD_TREE_FALLBACK=ON \
    -DMCR_SOFA_INCLUDE_DIRS="$SOFA_INCLUDE_DIRS" \
    -DMCR_SOFA_CONSTRAINT_LIBRARY="$SOFA_CONSTRAINT_LIBRARY" \
    -DMCR_SOFA_RUNTIME_LIBRARY_DIRS="$SOFA_RUNTIME_LIBRARY_DIRS" \
    -DBoost_INCLUDE_DIR="$BOOST_INCLUDE_HINT"

cmake --build "$BUILD_DIR" -j2

echo "[SDF_UNILATERAL_BUILD] built libraries:"
find "$BUILD_DIR" -type f \( -name '*.so' -o -name '*SDFUnilateralConstraint*' \) -print
