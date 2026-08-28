#!/bin/bash
# FJ9.8 — build the publication composites in a headless process with no
# Python, no planning library and no environment step.
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
export PATH="$HOME/.juliaup/bin:$PATH"
cd "$REPO"
LOG="${1:-/tmp/fj98_publication.log}"
julia --project=experiments tools/publication_figures.jl > "$LOG" 2>&1
status=$?
echo "PUBLICATION_EXIT=$status" | tee -a "$LOG"
exit $status
