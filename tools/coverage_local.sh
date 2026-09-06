#!/usr/bin/env bash
# Measure line coverage locally in CONTRIBUTOR mode (reference hidden) — the
# same mode CI runs — and print a per-file summary with the uncovered line
# ranges that matter. Iteration tool for the 95% coverage push.
set -u
cd "$(dirname "$0")/.."
export PATH="$HOME/.juliaup/bin:$PATH"

find src -name "*.cov" -delete

if [ -d ../duckduck ]; then
  mv ../duckduck ../duckduck_hidden
  trap 'mv ../duckduck_hidden ../duckduck' EXIT
fi

julia --project=. -e 'using Pkg; Pkg.test(coverage=true)' 2>&1 | tail -3

julia --startup-file=no -e '
using Pkg
Pkg.activate(mktempdir(); io=devnull)
Pkg.add("Coverage"; io=devnull)
using Coverage
fcs = process_folder("src")
tot_h = tot_l = 0
rows = []
for f in fcs
    h = count(x -> x !== nothing && x > 0, f.coverage)
    l = count(x -> x !== nothing, f.coverage)
    tot_h += h; tot_l += l
    push!(rows, (f.filename, h, l))
end
sort!(rows; by = r -> r[3] - r[2], rev = true)
println("== per file (worst absolute gap first) ==")
for (fn, h, l) in rows
    l == 0 && continue
    println(rpad(fn, 46), lpad(h, 5), "/", rpad(l, 5), "  ",
        round(100h / l; digits=1), "%")
end
println("TOTAL ", tot_h, "/", tot_l, " = ", round(100tot_h / tot_l; digits=2), "%")
' 2>&1 | grep -v "Precompiling\|Resolving\|Installed\|Updating"
