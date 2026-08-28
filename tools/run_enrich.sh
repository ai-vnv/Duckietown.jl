#!/bin/bash
# FJ8.4c — reproduce the frozen FJ8.4b protocol with per-decision logging.
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
export PATH="$HOME/.juliaup/bin:$PATH"
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate ddm-torch 2>/dev/null || true
cd "$REPO"
LOG="${1:-/tmp/fj84c_enrich.log}"
julia --project=experiments tools/enrich_decision_log.jl > "$LOG" 2>&1
status=$?
echo "ENRICH_EXIT=$status" | tee -a "$LOG"
exit $status
