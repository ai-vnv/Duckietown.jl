#!/usr/bin/env bash
# List the exact "Error During Test at" sites in the given test files when the
# reference checkout is absent. Companion to probe_gated_tests.sh.
set -u
cd "$(dirname "$0")/.."
export PATH="$HOME/.juliaup/bin:$PATH"

if [ -d ../duckduck ]; then
  mv ../duckduck ../duckduck_hidden
  trap 'mv ../duckduck_hidden ../duckduck' EXIT
fi

for f in "$@"; do
  echo "=== $f ==="
  timeout 420 julia --project=. -e "
    using DuckietownDecisionModels, Test
    cd(\"test\")
    include(\"reporter.jl\")
    include(\"reference_guard.jl\")
    @testset \"probe\" begin
        include(\"$f.jl\")
    end" 2>&1 | grep -E "Error During Test at|LoadError" | sort -u
done
