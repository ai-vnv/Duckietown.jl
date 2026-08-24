# FJ9.9b — the structured test reporter.
#
# Standing technical debt since FJ9.0, and it earned its place: the terminal
# parsers dropped rows three separate times — on minute-format durations
# (`3m17.5s`), on negative durations when the WSL clock stepped backwards, and
# on failing rows, which carry three count columns instead of two. Each was
# found only because two independently written parsers disagreed.
#
# The fix is to stop parsing rendered output. `Test` already builds a result
# tree; this walks it and writes the counts as JSON.
#
#     structured JSON  -> authoritative
#     tools/suite_summary.sh / .py -> sanity checks that must agree
#
# The parsers are kept precisely because agreement between three independent
# implementations is evidence, and a silent regression in the reporter itself
# would otherwise be invisible.

module SuiteReporter

using Test
using JSON3

"""
    ResultCounts

Leaf counts of one node of the test tree.
"""
Base.@kwdef mutable struct ResultCounts
    passes::Int = 0
    fails::Int = 0
    errors::Int = 0
    broken::Int = 0
end

Base.:+(a::ResultCounts, b::ResultCounts) = ResultCounts(
    a.passes + b.passes, a.fails + b.fails, a.errors + b.errors,
    a.broken + b.broken)

"""
    count_results(ts) -> ResultCounts

Walk a testset tree and total its leaves. Nested testsets are recursed into,
so an assertion is counted once no matter how deeply it is nested — the
terminal parsers could only ever see top-level rows.

`n_passed` must be read at EVERY level, not only the top. A finished
`DefaultTestSet` discards its individual `Pass` objects and keeps only the
counter, so recursing while looking for `Pass` results finds nothing and the
total silently collapses. That is exactly what the first version did: it
reported 88 213 against the parsers' 148 758, which is why two independent
cross-checks are worth keeping.
"""
function count_results(ts::Test.DefaultTestSet)
    c = ResultCounts(passes = ts.n_passed)
    for r in ts.results
        if r isa Test.DefaultTestSet
            c = c + count_results(r)
        elseif r isa Test.Pass
            c.passes += 1
        elseif r isa Test.Fail
            c.fails += 1
        elseif r isa Test.Error
            c.errors += 1
        elseif r isa Test.Broken
            c.broken += 1
        end
    end
    return c
end

"""
    testset_rows(ts) -> Vector{NamedTuple}

One row per top-level testset, which is the unit the legacy parsers count.
"""
function testset_rows(ts::Test.DefaultTestSet)
    rows = NamedTuple[]
    for r in ts.results
        r isa Test.DefaultTestSet || continue
        c = count_results(r)
        push!(rows, (name = r.description, passes = c.passes,
            fails = c.fails, errors = c.errors, broken = c.broken))
    end
    return rows
end

"""
    write_report(ts, path; extra) -> Dict

Emit the structured summary. This is the authoritative count.
"""
function write_report(ts::Test.DefaultTestSet, path::AbstractString;
        extra = Dict{String,Any}())
    rows = testset_rows(ts)
    report = Dict{String,Any}(
        "schema" => "fj99.reporter.1",
        "testsets" => length(rows),
        "assertions" => sum(r -> r.passes, rows; init = 0),
        "failures" => sum(r -> r.fails, rows; init = 0),
        "errors" => sum(r -> r.errors, rows; init = 0),
        "broken" => sum(r -> r.broken, rows; init = 0),
        "julia_version" => string(VERSION),
        "testsets_detail" => [Dict("name" => r.name, "passes" => r.passes,
            "fails" => r.fails, "errors" => r.errors, "broken" => r.broken)
            for r in rows])
    merge!(report, extra)
    mkpath(dirname(path))
    open(path, "w") do io
        JSON3.pretty(io, report)
    end
    return report
end

end # module
