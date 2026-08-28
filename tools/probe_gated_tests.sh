#!/usr/bin/env bash
# Empirically probe which reference-gated test files actually pass WITHOUT the
# reference checkout present. Used to decide what can be un-gated for CI
# coverage. Hides ../duckduck for the duration, restores it on exit.
set -u
cd "$(dirname "$0")/.."
export PATH="$HOME/.juliaup/bin:$PATH"

if [ -d ../duckduck ]; then
  mv ../duckduck ../duckduck_hidden
  trap 'mv ../duckduck_hidden ../duckduck' EXIT
fi

for f in "$@"; do
  out=$(timeout 420 julia --project=. -e "
    using DuckietownDecisionModels, Test
    cd(\"test\")
    include(\"reporter.jl\")
    include(\"reference_guard.jl\")
    @testset \"probe\" begin
        include(\"$f.jl\")
    end" 2>&1)
  status=$?
  tail1=$(printf '%s\n' "$out" | grep -E "Test Summary|ERROR|LoadError" | head -2 | tr '\n' ' ')
  echo "RESULT $f exit=$status :: $tail1"
done
