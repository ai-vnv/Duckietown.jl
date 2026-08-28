#!/usr/bin/env bash
# Run the suite twice: contributor mode (reference hidden) and release-grade
# (reference present, via the canonical runner). Prints both summaries and the
# structured-report counts for each mode.
set -u
cd "$(dirname "$0")/.."
export PATH="$HOME/.juliaup/bin:$PATH"

echo "=== MODE 1: contributor (no reference checkout) ==="
if [ -d ../duckduck ]; then mv ../duckduck ../duckduck_hidden; fi
julia --project=. -e 'using Pkg; Pkg.test()' 2>&1 | grep -E "REPORT_|tests passed|Test Summary|ERROR"
echo "MODE1_EXIT=$?"
if [ -d ../duckduck_hidden ]; then mv ../duckduck_hidden ../duckduck; fi

echo "=== MODE 2: release-grade (reference present) ==="
bash tools/run_full_suite.sh 2>&1 | grep -E "REPORT_|PKGTEST_EXIT|SUITE EXIT|Test Summary" | tail -8
echo "MODE2_EXIT=$?"
