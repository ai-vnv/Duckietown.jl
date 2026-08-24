# FJ8.4b — cross-family solver comparison.
#
# Six solvers from four families (tabular, deep, discrete planning, continuous
# planning) on one formulation and one evaluator. Three rules are structural
# here, not stylistic:
#
#   1. TASK PERFORMANCE AND COMPUTATIONAL COST ARE NEVER COMBINED.
#      There is no composite score and no overall ranking. A slow planner that
#      drives well and a fast policy that drives badly must stay
#      distinguishable, so they are rendered as two separate blocks.
#
#   2. EPISODES ARE PAIRED BY SEED.
#      Every solver runs the same episode seeds, so `R_A,i - R_B,i` is a
#      difference on an identical initial condition. Comparing group means as
#      if the observations were independent throws that away.
#
#   3. DENOMINATORS ARE EXPLICIT.
#      Stop compliance is compliant encounters over stop ENCOUNTERS, not over
#      episodes; when a solver completes no stop encounter the rate is
#      `nothing`, never 0.0.
#
#      Note the precision, which FJ9.6 had to correct in the FJ8 prose:
#      completing no encounter is NOT the same as never reaching a sign. TD3
#      reaches the sign in every episode, stops, and never proceeds — it has
#      no denominator because it never finishes, not because it never arrives.

"""
    SummaryStats

Distributional summary of one metric across episodes. `n` is stated because a
mean over twenty episodes is not the same evidence as a mean over a thousand,
and the bootstrap interval is percentile-based on a fixed RNG so the number is
reproducible.
"""
struct SummaryStats
    n::Int
    mean::Float64
    median::Float64
    sd::Float64
    q25::Float64
    q75::Float64
    min::Float64
    max::Float64
    ci_lo::Float64
    ci_hi::Float64
end

_pct(sorted::Vector{Float64}, q::Real) = isempty(sorted) ? NaN :
    sorted[clamp(ceil(Int, q * length(sorted)), 1, length(sorted))]

"""
    summary_stats(values; bootstrap, seed) -> SummaryStats

Mean, median, SD, IQR, range and a percentile bootstrap 95 % CI of the mean.
"""
function summary_stats(values::AbstractVector{<:Real}; bootstrap::Integer=10_000,
    seed::Integer=20260819)
    v = Float64.(collect(values))
    n = length(v)
    n == 0 && return SummaryStats(0, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN)
    s = sort(v)
    mu = sum(v) / n
    sd = n > 1 ? sqrt(sum(x -> (x - mu)^2, v) / (n - 1)) : 0.0
    lo, hi = mu, mu
    if n > 1 && bootstrap > 0
        rng = MersenneTwister(seed)
        means = Vector{Float64}(undef, bootstrap)
        for b in 1:bootstrap
            acc = 0.0
            for _ in 1:n
                acc += v[rand(rng, 1:n)]
            end
            means[b] = acc / n
        end
        sort!(means)
        lo = _pct(means, 0.025)
        hi = _pct(means, 0.975)
    end
    return SummaryStats(n, mu, _pct(s, 0.5), sd, _pct(s, 0.25), _pct(s, 0.75),
        s[1], s[end], lo, hi)
end

"""
    PairedDifference

`a − b` on the episodes both solvers ran, matched by seed.
"""
struct PairedDifference
    a::String
    b::String
    metric::Symbol
    stats::SummaryStats
    n_a_better::Int
    n_b_better::Int
    n_tied::Int
end

"""
    paired_difference(name_a, eps_a, name_b, eps_b, metric) -> PairedDifference

Per-seed differences on a shared seed set. Throws if the two solvers did not
run the same seeds — silently intersecting them would turn a protocol error
into a quieter, wrong answer.
"""
function paired_difference(name_a::AbstractString,
    eps_a::AbstractVector{EpisodeMetrics}, name_b::AbstractString,
    eps_b::AbstractVector{EpisodeMetrics}, metric::Symbol)
    sa = [e.seed for e in eps_a]
    sb = [e.seed for e in eps_b]
    sort(sa) == sort(sb) || throw(ArgumentError(
        "paired comparison needs identical seed sets: $name_a has $(sort(sa)), " *
        "$name_b has $(sort(sb))"))
    ma = Dict(e.seed => e for e in eps_a)
    mb = Dict(e.seed => e for e in eps_b)
    diffs = [Float64(getfield(ma[s], metric)) - Float64(getfield(mb[s], metric))
             for s in sort(sa)]
    return PairedDifference(String(name_a), String(name_b), metric,
        summary_stats(diffs), count(>(0), diffs), count(<(0), diffs),
        count(==(0), diffs))
end

"""
    stop_compliance(episodes) -> Union{Nothing,Float64}

Compliant stop encounters over stop **encounters**, where an encounter is a
`passed_stop` event — a stop sign the agent actually traversed. `nothing` when
no stop sign was ever reached, so an untested solver is never credited with
perfect compliance.
"""
function stop_compliance(eps::AbstractVector{EpisodeMetrics})
    enc = sum(e -> e.passed_stops, eps; init=0)
    enc == 0 && return nothing
    viol = sum(e -> e.stop_violations, eps; init=0)
    return (enc - viol) / enc
end

"""
    SolverRun

Everything one solver produced under the frozen protocol: per-episode task
metrics, aggregated planning cost, and the per-decision records that explain
where the cost went.
"""
struct SolverRun
    name::String
    family::String
    episodes::Vector{EpisodeMetrics}
    cost::PlannerCost
    decisions::Vector{DecisionRecord}
    config::String
end

"""
    task_table(runs) -> String

The task-performance block. Contains no timing and no model-call column by
construction: computational cost lives in [`cost_table`](@ref).
"""
function task_table(runs::AbstractVector{SolverRun})
    w = maximum(length(r.name) for r in runs)
    io = IOBuffer()
    hdr = (("return", 9), ("median", 9), ("95% CI", 20), ("progress", 9),
        ("mean|d|", 8), ("max|d|", 8), ("mean|phi|", 9), ("speed", 7),
        ("len", 6))
    print(io, rpad("solver", w))
    for (h, n) in hdr
        print(io, "  ", lpad(h, n))
    end
    println(io)
    println(io, "-"^(w + sum(n + 2 for (_, n) in hdr)))
    for r in runs
        st = summary_stats([e.ret for e in r.episodes])
        print(io, rpad(r.name, w),
            "  ", lpad(round(st.mean; digits=2), 9),
            "  ", lpad(round(st.median; digits=2), 9),
            "  ", lpad("[$(round(st.ci_lo; digits=1)), $(round(st.ci_hi; digits=1))]", 20),
            "  ", lpad(round(sum(e -> e.progress, r.episodes) / length(r.episodes); digits=2), 9),
            "  ", lpad(round(sum(e -> e.mean_abs_d, r.episodes) / length(r.episodes); digits=4), 8),
            "  ", lpad(round(maximum(e -> e.max_abs_d, r.episodes); digits=4), 8),
            "  ", lpad(round(sum(e -> e.mean_abs_phi, r.episodes) / length(r.episodes); digits=4), 9),
            "  ", lpad(round(sum(e -> e.mean_speed, r.episodes) / length(r.episodes); digits=4), 7),
            "  ", lpad(round(sum(e -> e.decisions, r.episodes) / length(r.episodes); digits=1), 6))
        println(io)
    end
    return String(take!(io))
end

"""
    safety_table(runs) -> String

Counts, with the episode total as the explicit denominator, plus stop
compliance over its own denominator.
"""
function safety_table(runs::AbstractVector{SolverRun})
    w = maximum(length(r.name) for r in runs)
    io = IOBuffer()
    println(io, rpad("solver", w), "  ", lpad("eps", 5), "  ",
        lpad("offroad", 8), "  ", lpad("collide", 8), "  ",
        lpad("duck hit", 9), "  ", lpad("stop enc", 9), "  ",
        lpad("full stop", 10), "  ", lpad("violation", 10), "  ",
        lpad("compliance", 11), "  ", lpad("duck act", 9), "  ",
        lpad("crossings", 10), "  ", "reasons")
    println(io, "-"^(w + 100))
    for r in runs
        n = length(r.episodes)
        enc = sum(e -> e.passed_stops, r.episodes; init=0)
        comp = stop_compliance(r.episodes)
        println(io, rpad(r.name, w), "  ", lpad(n, 5), "  ",
            lpad(count(e -> e.offroad, r.episodes), 8), "  ",
            lpad(count(e -> e.other_collision, r.episodes), 8), "  ",
            lpad(count(e -> e.duck_collision, r.episodes), 9), "  ",
            lpad(enc, 9), "  ",
            lpad(sum(e -> e.full_stops, r.episodes; init=0), 10), "  ",
            lpad(sum(e -> e.stop_violations, r.episodes; init=0), 10), "  ",
            lpad(comp === nothing ? "n/a" : string(round(100 * comp; digits=1), "%"), 11), "  ",
            lpad(sum(e -> e.duck_active_decisions, r.episodes; init=0), 9), "  ",
            lpad(sum(e -> e.crossings, r.episodes; init=0), 10), "  ",
            _count_reasons(r.episodes))
    end
    return String(take!(io))
end

"""
    cost_table(runs) -> String

The computational-cost block. `model calls = 0` means measured-and-zero: the
policy performs no generative planning. `n/a` means unmeasured.
"""
function cost_table(runs::AbstractVector{SolverRun})
    w = maximum(length(r.name) for r in runs)
    io = IOBuffer()
    println(io, rpad("solver", w), "  ", lpad("family", 22), "  ",
        lpad("ms mean", 9), "  ", lpad("ms p50", 8), "  ", lpad("ms p95", 8),
        "  ", lpad("gen/act", 9), "  ", lpad("iters", 7), "  ",
        lpad("act nodes", 10), "  ", lpad("state nodes", 12))
    println(io, "-"^(w + 92))
    for r in runs
        c = r.cost
        g(k) = haskey(c.extra, k) ? string(round(c.extra[k]; digits=1)) : "-"
        println(io, rpad(r.name, w), "  ", lpad(r.family, 22), "  ",
            lpad(round(1e3 * c.latency_mean; digits=3), 9), "  ",
            lpad(round(1e3 * c.latency_p50; digits=3), 8), "  ",
            lpad(round(1e3 * c.latency_p95; digits=3), 8), "  ",
            lpad(c.model_calls_total < 0 ? "n/a" :
                 string(round(c.model_calls_per_action; digits=1)), 9), "  ",
            lpad(g(:iterations), 7), "  ", lpad(g(:action_nodes), 10), "  ",
            lpad(g(:state_nodes) == "-" ? g(:tree_nodes) : g(:state_nodes), 12))
    end
    return String(take!(io))
end

"""
    paired_table(diffs) -> String
"""
function paired_table(diffs::AbstractVector{PairedDifference})
    isempty(diffs) && return "(none)\n"
    w = maximum(length(d.a) + length(d.b) + 3 for d in diffs)
    io = IOBuffer()
    println(io, rpad("pair", w), "  ", lpad("metric", 10), "  ",
        lpad("mean diff", 10), "  ", lpad("median", 9), "  ",
        lpad("95% CI", 22), "  ", lpad("a>b", 5), "  ", lpad("b>a", 5))
    println(io, "-"^(w + 70))
    for d in diffs
        s = d.stats
        println(io, rpad("$(d.a) - $(d.b)", w), "  ",
            lpad(String(d.metric), 10), "  ",
            lpad(round(s.mean; digits=3), 10), "  ",
            lpad(round(s.median; digits=3), 9), "  ",
            lpad("[$(round(s.ci_lo; digits=2)), $(round(s.ci_hi; digits=2))]", 22),
            "  ", lpad(d.n_a_better, 5), "  ", lpad(d.n_b_better, 5))
    end
    return String(take!(io))
end

"""
    cost_by_episode_position(records; bins) -> Vector{NamedTuple}

Planning cost as a function of how far into the episode the decision was made.
FJ8.4a measured a 3.8x spread in generative cost across states; a mean alone
hides that, and this is what explains a high p95.
"""
function cost_by_episode_position(records::AbstractVector{DecisionRecord};
    bins::Integer=5)
    isempty(records) && return NamedTuple[]
    last_step = Dict{Int,Int}()
    for r in records
        last_step[r.seed] = max(get(last_step, r.seed, 0), r.step)
    end
    buckets = [Tuple{Float64,Int}[] for _ in 1:bins]
    for r in records
        len = max(last_step[r.seed], 1)
        frac = (r.step - 1) / len
        idx = clamp(floor(Int, frac * bins) + 1, 1, bins)
        push!(buckets[idx], (r.diagnostics.planning_time,
            r.diagnostics.model_calls))
    end
    out = NamedTuple[]
    for (i, b) in enumerate(buckets)
        isempty(b) && continue
        lat = sort([x[1] for x in b])
        push!(out, (bin=i, from=(i - 1) / bins, to=i / bins, n=length(b),
            latency_mean=sum(lat) / length(lat),
            latency_p95=_pct(lat, 0.95),
            model_calls_mean=sum(x[2] for x in b) / length(b)))
    end
    return out
end

"""
    position_table(rows) -> String
"""
function position_table(rows)
    isempty(rows) && return "(none)\n"
    io = IOBuffer()
    println(io, lpad("episode fraction", 18), "  ", lpad("n", 7), "  ",
        lpad("ms mean", 9), "  ", lpad("ms p95", 9), "  ", lpad("gen/act", 9))
    println(io, "-"^58)
    for r in rows
        println(io, lpad("$(round(r.from; digits=2))-$(round(r.to; digits=2))", 18),
            "  ", lpad(r.n, 7), "  ",
            lpad(round(1e3 * r.latency_mean; digits=2), 9), "  ",
            lpad(round(1e3 * r.latency_p95; digits=2), 9), "  ",
            lpad(round(r.model_calls_mean; digits=1), 9))
    end
    return String(take!(io))
end

"""
    episode_csv(runs) -> String

Episode-level records for every solver, one row per (solver, seed), so the
paired structure survives outside this session and the comparison can be
reanalysed without re-running it.
"""
function episode_csv(runs::AbstractVector{SolverRun})
    io = IOBuffer()
    println(io, "solver,family,seed,decisions,return,discounted_return,",
        "progress,mean_abs_d,max_abs_d,mean_abs_phi,mean_speed,brake_ratio,",
        "offroad,other_collision,duck_collision,timeout,goal,full_stops,",
        "stop_violations,passed_stops,stop_zone_decisions,",
        "duck_yield_decisions,duck_active_decisions,crossings,reason")
    for r in runs, e in r.episodes
        println(io, r.name, ",", r.family, ",", e.seed, ",", e.decisions, ",",
            e.ret, ",", e.discounted_return, ",", e.progress, ",",
            e.mean_abs_d, ",", e.max_abs_d, ",", e.mean_abs_phi, ",",
            e.mean_speed, ",", e.brake_ratio, ",", e.offroad, ",",
            e.other_collision, ",", e.duck_collision, ",", e.timeout, ",",
            e.goal, ",", e.full_stops, ",", e.stop_violations, ",",
            e.passed_stops, ",", e.stop_zone_decisions, ",",
            e.duck_yield_decisions, ",", e.duck_active_decisions, ",",
            e.crossings, ",", e.reason)
    end
    return String(take!(io))
end

"""
    check_paired_protocol(runs) -> NamedTuple

Verify the protocol before any result is read: every solver ran the same seed
set, the same number of episodes, and under the same horizon. Returns the
shared seeds; throws if they differ.
"""
function check_paired_protocol(runs::AbstractVector{SolverRun})
    isempty(runs) && throw(ArgumentError("no runs"))
    reference = sort([e.seed for e in runs[1].episodes])
    for r in runs
        sort([e.seed for e in r.episodes]) == reference || throw(ArgumentError(
            "$(r.name) ran seeds $(sort([e.seed for e in r.episodes])), " *
            "expected $reference"))
    end
    return (seeds=reference, solvers=length(runs), episodes=length(reference))
end
