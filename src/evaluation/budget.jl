# FJ8.4a — the cost–search curve, measured solver-agnostically.
#
# The question is NOT "what is the best budget". It is: what does a millisecond
# of planning, and an iteration of a solver, actually buy in generative-model
# work? Two solvers quoting "500 iterations" are not on the same computational
# budget if one consumes 15 generative calls per iteration and the other 23, so
# every study here reports **two budget axes**:
#
#   solver-native      iterations, the number the solver was configured with
#   model-equivalent   generative calls actually consumed per decision
#
# Only the second is comparable across solvers, and it is the one
# [`compute_matched_budget`](@ref) selects on.
#
# Nothing here knows what a tree is. A planner is supplied as a closure
# `budget -> planner`, so the same study runs for any solver — or for a learned
# policy, whose curve is flat by construction.

"""
    BudgetPoint

One (solver, budget) cell of the cost–search curve, aggregated over repeated
measurements at a fixed set of states.

Latency is summarised by mean/p50/p95/max because FJ8.0 already showed wall
time varies ~±20 % run to run; `model_calls` and the node counts are exactly
reproducible, since the search is deterministic per planner seed and the
repeats therefore measure timing noise only.
"""
struct BudgetPoint
    label::String
    budget::Int                     # solver-native (iterations)
    states::Int
    repeats::Int
    decisions::Int
    latency_mean::Float64
    latency_p50::Float64
    latency_p95::Float64
    latency_max::Float64
    model_calls_mean::Float64
    model_calls_per_iteration::Float64
    bytes_per_decision::Float64
    extra::Dict{Symbol,Float64}
end

"""
    budget_study(mdp, make_planner, budgets; states, repeats, warmup) -> Vector{BudgetPoint}

Measure one planner family across budgets.

- `make_planner(budget, seed)` returns a planner. Passing a closure is what
  keeps this solver-agnostic: the study never names a solver type.
- `states` is the SAME state set at every budget, so the curve isolates the
  budget rather than mixing in state difficulty.
- `repeats` re-runs each (state, budget) with the identical planner seed. The
  search is therefore identical and only wall time varies, which is exactly
  the quantity that needs repeating.
- `warmup` runs one untimed decision per budget first.

`mdp` should be an [`InstrumentedMDP`](@ref); otherwise `model_calls` is
reported as `-1` (unmeasured) rather than a misleading zero.
"""
function budget_study(mdp::AnyMDPLike, make_planner, budgets;
    states::AbstractVector{DuckieWorldState}, repeats::Integer=3,
    warmup::Bool=true, label::AbstractString="planner",
    planner_seed::Integer=201)
    points = BudgetPoint[]
    for b in budgets
        if warmup && !isempty(states)
            p = make_planner(b, planner_seed)
            plan_action(p, mdp, states[1])
        end
        lat = Float64[]
        calls = Int[]
        bytes = Float64[]
        extras = Dict{Symbol,Vector{Float64}}()
        for (i, s) in enumerate(states)
            seed = planner_seed + i - 1
            for _ in 1:repeats
                planner = make_planner(b, seed)
                before = Base.gc_num()
                a, d = plan_action(planner, mdp, s)
                diff = Base.GC_Diff(Base.gc_num(), before)
                a === nothing && error("planner returned no action")
                push!(lat, d.planning_time)
                push!(calls, d.model_calls)
                push!(bytes, Float64(diff.allocd))
                for k in keys(d.extra)
                    v = d.extra[k]
                    v isa Real || continue
                    push!(get!(extras, k, Float64[]), Float64(v))
                end
            end
        end
        sort!(lat)
        n = length(lat)
        measured = all(>=(0), calls)
        mean_calls = measured ? sum(calls) / n : -1.0
        push!(points, BudgetPoint(String(label), Int(b), length(states),
            Int(repeats), n,
            sum(lat) / n, _quantile_sorted(lat, 0.50),
            _quantile_sorted(lat, 0.95), lat[end],
            mean_calls,
            measured && b > 0 ? mean_calls / b : NaN,
            sum(bytes) / n,
            Dict(k => sum(v) / length(v) for (k, v) in extras)))
    end
    return points
end

"""
    budget_table(points; extra_fields) -> String

The cost–search curve as a fixed-width table, with both budget axes side by
side so an iteration count is never mistaken for a computational budget.
"""
function budget_table(points::AbstractVector{BudgetPoint};
    extra_fields=Symbol[])
    isempty(points) && return "(no measurements)\n"
    w = maximum(length(p.label) for p in points)
    io = IOBuffer()
    println(io, rpad("solver", w), "  ", lpad("iters", 6), "  ",
        lpad("gen/act", 9), "  ", lpad("gen/iter", 9), "  ",
        lpad("ms mean", 9), "  ", lpad("ms p50", 8), "  ", lpad("ms p95", 8),
        "  ", lpad("MiB/act", 8),
        join([lpad(String(f), 14) for f in extra_fields]))
    println(io, "-"^(w + 66 + 14 * length(extra_fields)))
    for p in points
        print(io, rpad(p.label, w), "  ", lpad(p.budget, 6), "  ",
            lpad(p.model_calls_mean < 0 ? "n/a" :
                 string(round(p.model_calls_mean; digits=1)), 9), "  ",
            lpad(isnan(p.model_calls_per_iteration) ? "n/a" :
                 string(round(p.model_calls_per_iteration; digits=2)), 9), "  ",
            lpad(round(1e3 * p.latency_mean; digits=2), 9), "  ",
            lpad(round(1e3 * p.latency_p50; digits=2), 8), "  ",
            lpad(round(1e3 * p.latency_p95; digits=2), 8), "  ",
            lpad(round(p.bytes_per_decision / 1024^2; digits=2), 8))
        for f in extra_fields
            print(io, lpad(haskey(p.extra, f) ?
                string(round(p.extra[f]; digits=1)) : "-", 14))
        end
        println(io)
    end
    return String(take!(io))
end

"""
    compute_matched_budget(points, target_calls) -> NamedTuple

The measured point whose generative-model consumption is closest to
`target_calls` per decision, with the **actual** value reported rather than the
target. Comparing two planners at equal iteration counts compares nothing; this
is what makes an online-planner comparison fair.

Returns `(; budget, actual_calls, error, latency_mean, point)`.
"""
function compute_matched_budget(points::AbstractVector{BudgetPoint},
    target_calls::Real)
    usable = filter(p -> p.model_calls_mean >= 0, points)
    isempty(usable) &&
        throw(ArgumentError("no point has measured model calls; wrap the " *
                            "model in an InstrumentedMDP"))
    best = argmin(p -> abs(p.model_calls_mean - target_calls), usable)
    return (budget=best.budget, actual_calls=best.model_calls_mean,
        error=best.model_calls_mean - target_calls,
        relative_error=(best.model_calls_mean - target_calls) / target_calls,
        latency_mean=best.latency_mean, point=best)
end

"""
    estimate_budget_for_calls(points, target_calls) -> Int

The solver-native budget predicted to consume `target_calls` generative calls
per decision, from the measured calls-per-iteration.

Selecting the nearest point of a coarse grid can be off by tens of percent;
this instead inverts the measured rate, so the caller can *run* the estimate
and report what it realised. It is an estimate from measurement, never a
substitute for measuring the result.
"""
function estimate_budget_for_calls(points::AbstractVector{BudgetPoint},
    target_calls::Real)
    usable = filter(p -> p.model_calls_mean >= 0 &&
        isfinite(p.model_calls_per_iteration) &&
        p.model_calls_per_iteration > 0, points)
    isempty(usable) &&
        throw(ArgumentError("no point has a measured calls-per-iteration rate"))
    rate = sum(p -> p.model_calls_per_iteration, usable) / length(usable)
    return max(1, round(Int, target_calls / rate))
end

"""
    matched_operating_points(curves, targets) -> Vector{NamedTuple}

Apply [`compute_matched_budget`](@ref) to several planner curves at several
generative-call targets. `curves` maps a label to its `BudgetPoint` vector.
The result is the operating-point table an equal-compute comparison is run at,
with each solver's realised call count recorded next to the target it was
matched to.
"""
function matched_operating_points(curves::AbstractDict, targets)
    out = NamedTuple[]
    for target in targets, label in sort(collect(keys(curves)))
        m = compute_matched_budget(curves[label], target)
        push!(out, (label=String(label), target=Float64(target),
            budget=m.budget, actual_calls=m.actual_calls,
            relative_error=m.relative_error, latency_mean=m.latency_mean))
    end
    return out
end

"""
    operating_point_table(rows) -> String
"""
function operating_point_table(rows)
    isempty(rows) && return "(none)\n"
    w = maximum(length(r.label) for r in rows)
    io = IOBuffer()
    println(io, rpad("solver", w), "  ", lpad("target gen", 11), "  ",
        lpad("iters", 6), "  ", lpad("actual gen", 11), "  ",
        lpad("rel err", 9), "  ", lpad("ms mean", 9))
    println(io, "-"^(w + 54))
    for r in rows
        println(io, rpad(r.label, w), "  ",
            lpad(round(Int, r.target), 11), "  ", lpad(r.budget, 6), "  ",
            lpad(round(r.actual_calls; digits=1), 11), "  ",
            lpad(string(round(100 * r.relative_error; digits=1), "%"), 9), "  ",
            lpad(round(1e3 * r.latency_mean; digits=2), 9))
    end
    return String(take!(io))
end

"""
    planning_seed_config(path=default) -> NamedTuple

Load the frozen FJ8 seed split (`configs/planning/fj8_seeds.yaml`). The
development and evaluation sets are disjoint on purpose: configuration choices
are made on the former, and the six-solver comparison runs on the latter, so a
planner can never be tuned against the seeds it will be reported on.
"""
function planning_seed_config(path::AbstractString=joinpath(
    pkgdir(@__MODULE__), "configs", "planning", "fj8_seeds.yaml"))
    cfg = YAML.load_file(path)
    dev = cfg["development"]
    ev = cfg["evaluation"]
    d_states = Int.(dev["states"])
    e_episodes = Int.(ev["episodes"])
    isempty(intersect(d_states, e_episodes)) ||
        throw(ArgumentError("development and evaluation seeds overlap: " *
                            string(intersect(d_states, e_episodes))))
    return (development=(states=d_states, planner=Int.(dev["planner"])),
        evaluation=(episodes=e_episodes, planner=Int(ev["planner"]),
            max_steps=Int(ev["max_steps"])),
        path=String(path))
end
