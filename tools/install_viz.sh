#!/bin/bash
# Install the visualisation backend into the experiments environment.
#
# CairoMakie is a CPU/static backend, which is what a headless CI and paper
# figures need; it is deliberately NOT a dependency of the package.
set -uo pipefail
export PATH="$HOME/.juliaup/bin:$PATH"
cd "$(dirname "$0")/.."
julia --project=experiments -e '
using Pkg
have = Pkg.project().dependencies
for p in ("CairoMakie",)
    haskey(have, p) || Pkg.add(p)
end
Pkg.instantiate()'
echo "VIZ_INSTALL_EXIT=$?"
