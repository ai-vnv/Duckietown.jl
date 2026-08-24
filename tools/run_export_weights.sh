#!/bin/bash
# FJ9.8 provisioning — export frozen actor parameters to .npy (needs ddm-torch).
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
export PATH="$HOME/.juliaup/bin:$PATH"
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate ddm-torch 2>/dev/null || true
cd "$REPO"
LOG="${1:-/tmp/fj98_weights.log}"
julia --project=experiments tools/fj98_export_weights.jl > "$LOG" 2>&1
status=$?
echo "EXPORT_EXIT=$status" | tee -a "$LOG"
exit $status
