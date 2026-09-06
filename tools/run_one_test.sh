#!/usr/bin/env bash
# Run one test file standalone in a persistent dev environment that has the
# package (dev'd) plus the test extras (MCTS, POMDPs). Iteration tool — the
# real verdict is still Pkg.test.
#   tools/run_one_test.sh test_evaluation_native [more files...]
set -u
cd "$(dirname "$0")/.."
export PATH="$HOME/.juliaup/bin:$PATH"
ENVDIR="$HOME/.julia/environments/duckietown-dev"

if [ ! -f "$ENVDIR/Project.toml" ]; then
  julia -e "
    using Pkg
    Pkg.activate(\"$ENVDIR\")
    Pkg.develop(path = \"$PWD\")
    Pkg.add([\"POMDPs\", \"MCTS\", \"Test\", \"Random\", \"JSON3\", \"YAML\"])
  "
fi

for f in "$@"; do
  echo "=== $f ==="
  julia --project="$ENVDIR" -e "
    using Duckietown, Test
    cd(\"test\")
    include(\"reporter.jl\")
    include(\"reference_guard.jl\")
    @testset \"probe\" begin
        include(\"$f.jl\")
    end" 2>&1 | tail -25
done
