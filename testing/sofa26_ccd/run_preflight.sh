#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/env.sh"

mkdir -p "$SOFA26_RUNTIME/logs"

python "$HERE/preflight_ccd.py"     --output "$SOFA26_RUNTIME/ccd_preflight.json"     2>&1 | tee "$SOFA26_RUNTIME/logs/ccd_preflight.log"
