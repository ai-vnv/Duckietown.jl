# Does Duckietown satisfy the condition DORA's exactness actually rests on?
#
# From the docs: Dijkstra returns the exact optimal value function and policy
# when the REDUCED COSTS are nonnegative,
#
#     w*(s,a) = Q*(s,a) - V*(sigma(s,a))  >= 0
#
# where sigma(s,a) is the determinized (most likely) successor. The package
# ships the diagnostic for this: causality_margin / reduced_costs. This is the
# principled test of whether the problem is DORA-shaped, and it is what should
# be run before any transfer experiment.

using DuckietownDecisionModels
using POMDPs, POMDPTools, DORASolvers
using Random, Statistics, Printf

const MDP_ = DuckietownMDP(scenario_config(:stop_and_duck); action_space=:discrete)
const ACTS = collect(POMDPs.actions(MDP_))
const SCFG = MDP_.transition.state_cfg
disc(s) = discretize(first(get_raw_state(s, SCFG)))

classify_result(res) =
    res.events.passed_stop ? :goal :
    (res.events.offroad || res.events.other_collision ||
     res.events.collision_duck) ? :crash : :normal
step_cost(res) = 1.0 + 4.0abs(res.raw_state.d) + 0.5abs(res.raw_state.phi)

function behaviour_action(s, rng, eps)
    rand(rng) < eps && return rand(rng, 1:length(ACTS))
    raw, _ = get_raw_state(s, SCFG)
    err = raw.d + 0.35raw.phi
    a = if err < -0.02
            raw.phi > 0.25 ? SLOW_STRAIGHT : SLOW_LEFT
        elseif err > 0.02
            raw.phi < -0.25 ? SLOW_STRAIGHT : SLOW_RIGHT
        elseif abs(raw.phi) < 0.10 && abs(raw.d) < 0.03
            FAST_STRAIGHT
        else
            SLOW_STRAIGHT
        end
    return findfirst(==(a), ACTS)
end

const GOAL_S  = ntuple(_ -> -1, 7)
const CRASH_S = ntuple(_ -> -2, 7)
struct DuckieSSP <: MDP{NTuple{7,Int},Int}
    P::Dict{Tuple{NTuple{7,Int},Int},Vector{Tuple{NTuple{7,Int},Float64}}}
    C::Dict{Tuple{NTuple{7,Int},Int,NTuple{7,Int}},Float64}
end
POMDPs.actions(::DuckieSSP) = collect(1:7)
POMDPs.discount(::DuckieSSP) = 1.0
POMDPs.isterminal(::DuckieSSP, s) = s == GOAL_S || s == CRASH_S
function POMDPs.transition(m::DuckieSSP, s, a)
    POMDPs.isterminal(m, s) && return Deterministic(s)
    e = get(m.P, (s, a), nothing)
    e === nothing && return Deterministic(s)
    SparseCat(first.(e), last.(e))
end

function build(; episodes=1500, horizon=150, seed=20260826, eps=0.15,
                 min_samples=1)
    counts = Dict{Tuple{NTuple{7,Int},Int},Dict{Any,Int}}()
    costs  = Dict{Tuple{NTuple{7,Int},Int},Dict{Any,Float64}}()
    starts = Dict{NTuple{7,Int},Int}()
    rng = MersenneTwister(seed)
    for _ in 1:episodes
        s = rand(rng, initialstate(MDP_))
        starts[disc(s)] = get(starts, disc(s), 0) + 1
        for _ in 1:horizon
            ds = disc(s); ai = behaviour_action(s, rng, eps)
            res = simulate_decision(MDP_.transition, s, ACTS[ai], rng)
            cls = classify_result(res)
            tgt = cls === :normal ? disc(res.sp) : cls
            c = get!(counts, (ds, ai), Dict{Any,Int}())
            k = get!(costs,  (ds, ai), Dict{Any,Float64}())
            c[tgt] = get(c, tgt, 0) + 1
            k[tgt] = get(k, tgt, 0.0) + step_cost(res)
            (cls !== :normal || res.terminated || res.truncated) && break
            s = res.sp
        end
    end
    P = Dict{Tuple{NTuple{7,Int},Int},Vector{Tuple{NTuple{7,Int},Float64}}}()
    C = Dict{Tuple{NTuple{7,Int},Int,NTuple{7,Int}},Float64}()
    for ((ds, ai), tg) in counts
        n = sum(values(tg)); n < min_samples && continue
        v = Tuple{NTuple{7,Int},Float64}[]
        for (t, k) in tg
            st = t === :goal ? GOAL_S : t === :crash ? CRASH_S : t::NTuple{7,Int}
            push!(v, (st, k / n)); C[(ds, ai, st)] = costs[(ds, ai)][t] / k
        end
        P[(ds, ai)] = v
    end
    return DuckieSSP(P, C), argmax(starts)
end

function diagnose(min_samples)
    ssp, start = build(; min_samples=min_samples)
    p = solve(DORASolver(start = start,
        classify = sp -> sp == GOAL_S ? :goal : sp == CRASH_S ? :crash : :normal,
        cost = (s, a, sp) -> get(ssp.C, (s, a, sp), 1.0),
        c_min = 0.5, c_to = 200.0, c_crash = 100.0, horizon = 150), ssp)
    tab = p.tab
    V, pistar = optimal_value(tab)
    worst, frac = causality_margin(tab, V, pistar)
    w = reduced_costs(tab, V)
    finite = filter(isfinite, vec(w))
    neg = count(<(0), finite)
    sr, cr, _ = outcome_rates(tab, pistar)
    return (; tab, V, worst, frac, neg, nfin=length(finite), sr, cr,
              minw = isempty(finite) ? NaN : minimum(finite))
end

println("The condition DORA's exactness rests on: reduced costs w*(s,a) >= 0\n")
@printf("%12s %7s %10s %12s %14s %10s\n",
        "min_samples", "states", "frac ok", "worst margin", "negative w",
        "in-model")
for ms in (1, 10, 30)
    d = diagnose(ms)
    @printf("%12d %7d %9.3f %12.3f %8d/%-5d %9.1f%%\n",
            ms, d.tab.S, d.frac, d.worst, d.neg, d.nfin, 100d.sr)
end

println("""

`frac` is the share of state-action pairs whose reduced cost is nonnegative.
Where it is below 1 the Dijkstra oracle is no longer exact on this problem,
and the gap is not an implementation detail — it is the assumption failing.""")

# ---------------------------------------------------------------------------
# Why is it violated? Determinization only means something when the intended
# successor is what usually happens. Measure the probability mass the most
# likely outcome actually carries, at two levels of abstraction.
# ---------------------------------------------------------------------------
function concentration(; level::Symbol, episodes=1200, horizon=150, seed=5, eps=0.15)
    ring = let m = initial_map(scenario_config(:stop_and_duck))
        t = collect(drivable_tiles(m))
        cx = sum(first.(t))/length(t); cy = sum(last.(t))/length(t)
        sort(t; by = p -> atan(p[2]-cy, p[1]-cx))
    end
    rix = Dict(t => k for (k,t) in enumerate(ring)); nr = length(ring)
    tile_of(s) = get_grid_coords(s.map, collect(s.ego.pos))
    key(prog, s) = level === :lane ? disc(s) : (prog, 0, 0)

    counts = Dict{Tuple{Any,Int},Dict{Any,Int}}()
    rng = MersenneTwister(seed)
    for _ in 1:episodes
        s = rand(rng, initialstate(MDP_)); prog = 0; tile = tile_of(s)
        for _ in 1:horizon
            k0 = key(prog, s); ai = behaviour_action(s, rng, eps)
            res = simulate_decision(MDP_.transition, s, ACTS[ai], rng)
            nt = tile_of(res.sp)
            if nt != tile
                a = get(rix, tile, 0); b = get(rix, nt, 0)
                (a != 0 && b != 0 && mod(b-a, nr) == 1) && (prog += 1)
            end
            cls = classify_result(res)
            tgt = cls === :normal ? key(prog, res.sp) : cls
            c = get!(counts, (k0, ai), Dict{Any,Int}())
            c[tgt] = get(c, tgt, 0) + 1
            (cls !== :normal || res.terminated || res.truncated) && break
            s = res.sp; tile = nt
        end
    end
    # share of mass on the single most likely outcome, for well-sampled pairs
    conc = Float64[]
    for (_, tg) in counts
        n = sum(values(tg)); n < 20 && continue
        push!(conc, maximum(values(tg)) / n)
    end
    return conc
end

println("\nProbability mass carried by the determinized (most likely) successor,")
println("over state-action pairs with at least 20 samples:\n")
@printf("%-28s %6s %9s %9s %9s\n", "state abstraction", "pairs", "median", "mean", "<0.5")
for (nm, lv) in (("lane 7-tuple (used above)", :lane),
                 ("ring progress (tile level)", :tile))
    c = concentration(level = lv)
    @printf("%-28s %6d %9.3f %9.3f %8.1f%%\n", nm, length(c), median(c),
            mean(c), 100 * count(<(0.5), c) / max(length(c), 1))
end

# ---------------------------------------------------------------------------
# CORRECTION. The number above counts self-loops, and `tabularize` explicitly
# discards them: succ[s,a] is "the most likely outcome that is neither a self
# loop nor the crash sink". At tile level most decisions do not change tile,
# so a high concentration there is concentration on a NON-EDGE and is no
# evidence at all. Re-measure over the outcomes that can actually be edges.
# ---------------------------------------------------------------------------
function concentration2(; level::Symbol, episodes=1200, horizon=150, seed=5,
                          eps=0.15, minn=20)
    ring = let m = initial_map(scenario_config(:stop_and_duck))
        t = collect(drivable_tiles(m))
        cx = sum(first.(t))/length(t); cy = sum(last.(t))/length(t)
        sort(t; by = p -> atan(p[2]-cy, p[1]-cx))
    end
    rix = Dict(t => k for (k,t) in enumerate(ring)); nr = length(ring)
    tile_of(s) = get_grid_coords(s.map, collect(s.ego.pos))
    key(prog, s) = level === :lane ? disc(s) : (prog, 0, 0)

    counts = Dict{Tuple{Any,Int},Dict{Any,Int}}()
    rng = MersenneTwister(seed)
    for _ in 1:episodes
        s = rand(rng, initialstate(MDP_)); prog = 0; tile = tile_of(s)
        for _ in 1:horizon
            k0 = key(prog, s); ai = behaviour_action(s, rng, eps)
            res = simulate_decision(MDP_.transition, s, ACTS[ai], rng)
            nt = tile_of(res.sp)
            if nt != tile
                a = get(rix, tile, 0); b = get(rix, nt, 0)
                (a != 0 && b != 0 && mod(b-a, nr) == 1) && (prog += 1)
            end
            cls = classify_result(res)
            tgt = cls === :normal ? key(prog, res.sp) : cls
            c = get!(counts, (k0, ai), Dict{Any,Int}())
            c[tgt] = get(c, tgt, 0) + 1
            (cls !== :normal || res.terminated || res.truncated) && break
            s = res.sp; tile = nt
        end
    end
    selfshare = Float64[]; edgeconc = Float64[]; noedge = 0; npairs = 0
    for ((k0, _), tg) in counts
        n = sum(values(tg)); n < minn && continue
        npairs += 1
        push!(selfshare, get(tg, k0, 0) / n)
        # outcomes that can be an edge: not the self loop, not crash
        others = Dict(t => v for (t, v) in tg if t != k0 && t !== :crash)
        if isempty(others)
            noedge += 1
        else
            push!(edgeconc, maximum(values(others)) / sum(values(others)))
        end
    end
    return (; npairs, selfshare, edgeconc, noedge)
end

println("\nCorrected: self-loops are not edges, so measure them separately.\n")
@printf("%-28s %6s %12s %14s %12s\n", "state abstraction", "pairs",
        "self-loop", "edge conc.", "no edge")
for (nm, lv) in (("lane 7-tuple", :lane), ("ring progress (tile)", :tile))
    c = concentration2(level = lv)
    @printf("%-28s %6d %11.3f %13.3f %10d\n", nm, c.npairs,
            median(c.selfshare),
            isempty(c.edgeconc) ? NaN : median(c.edgeconc), c.noedge)
end
