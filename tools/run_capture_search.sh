#!/bin/bash
# FJ9.5b — capture MCTS and DPW search snapshots from one frozen state.
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
export PATH="$HOME/.juliaup/bin:$PATH"
cd "$REPO"
LOG="${1:-/tmp/fj95_capture.log}"
julia --project=experiments tools/capture_search.jl > "$LOG" 2>&1
status=$?
echo "CAPTURE_EXIT=$status" | tee -a "$LOG"
exit $status
