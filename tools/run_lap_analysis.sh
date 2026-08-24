#!/bin/bash
# Measure lap completion on small_loop for every solver.
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
export PATH="$HOME/.juliaup/bin:$PATH"
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate ddm-torch 2>/dev/null || true
cd "$REPO"
LOG="${1:-/tmp/fj8_laps.log}"
julia --project=experiments tools/fj8_lap_analysis.jl > "$LOG" 2>&1
echo "LAP_EXIT=$?" | tee -a "$LOG"
