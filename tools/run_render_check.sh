#!/bin/bash
# FJ9 — render figures in a headless process with no Python and no solver.
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
export PATH="$HOME/.juliaup/bin:$PATH"
cd "$REPO"
LOG="${1:-/tmp/fj9_render.log}"
julia --project=experiments tools/fj9_render_check.jl > "$LOG" 2>&1
status=$?
echo "RENDER_EXIT=$status" | tee -a "$LOG"
exit $status
