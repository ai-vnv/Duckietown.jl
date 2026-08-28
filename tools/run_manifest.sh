#!/bin/bash
# FJ9.9f — isolation matrix + reproducibility manifest.
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
export PATH="$HOME/.juliaup/bin:$PATH"
cd "$REPO"
LOG="${1:-/tmp/fj99_manifest.log}"
julia --project=experiments tools/reproducibility_manifest.jl > "$LOG" 2>&1
status=$?
echo "MANIFEST_EXIT=$status" | tee -a "$LOG"
exit $status
