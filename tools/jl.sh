#!/bin/bash
# Run a Julia snippet or file in the package project.
#
# Exists because quoting a `julia -e` one-liner through
# `wsl.exe -- bash -c '...'` from the Windows host mangles reliably: `$PATH`
# gets expanded on the wrong side, and parentheses in the snippet become shell
# syntax errors. Passing the code as a file, or as "$@" to this script,
# avoids the whole class of problem.
#
#   tools/jl.sh -e 'println(1+1)'
#   tools/jl.sh path/to/script.jl
set -uo pipefail
export PATH="$HOME/.juliaup/bin:$PATH"
cd "$(dirname "$0")/.."
exec julia --project=. "$@"
