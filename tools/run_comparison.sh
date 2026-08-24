#!/bin/bash
# FJ8.4b — run the frozen six-solver comparison.
#
# MCTS.jl is a weak dependency, so it is deliberately absent from the package
# manifest. The experiment therefore runs in its own environment under
# `experiments/`, which devs the package and adds the solver. Running from the
# repository directory keeps the juliaup override (1.11.3) in force.
set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
export PATH="$HOME/.juliaup/bin:$PATH"
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate ddm-torch 2>/dev/null || true   # SAC/TD3 weight export oracle
cd "$REPO"

LOG="${1:-/tmp/fj84b_comparison.log}"
mkdir -p experiments

julia --project=experiments -e '
using Pkg
Pkg.develop(path = ".")
have = Pkg.project().dependencies
for p in ("MCTS", "POMDPs", "YAML", "Random")
    haskey(have, p) || Pkg.add(p)
end
Pkg.instantiate()' > "$LOG" 2>&1

echo "JULIA: $(julia --version)" | tee -a "$LOG"
julia --project=experiments tools/fj8_comparison.jl >> "$LOG" 2>&1
status=$?
echo "COMPARISON_EXIT=$status" | tee -a "$LOG"
exit $status
