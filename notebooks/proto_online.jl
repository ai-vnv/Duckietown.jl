# DORA used ONLINE, which is what it is for.
#
# The earlier attempt pre-baked a batch transition model and handed it over
# with known_costs=true. That line in solve() does:
#
#     L.st.N .= 1.0                                # one pseudo-observation each
#     optimistic = sol.optimistic && !known_costs  # confidence bounds OFF
#
# so DORA trusted a mean built from 5 samples exactly as much as one built
# from 780. The optimizer's curse measured earlier was a consequence of that
# choice, not of the solver.
#
# Online: known_costs=false, and every real decision is reported back with
# observe!, so the cost statistics come from the environment itself.

using DuckietownDecisionModels
using POMDPs, POMDPTools, DORASolvers
using DORASolvers: observe!
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

# --- SSP structure (the graph DORA searches) --------------------------------
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

function sample_structure(; episodes=1200, horizon=150, seed=20260826, eps=0.15)
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
        n = sum(values(tg)); v = Tuple{NTuple{7,Int},Float64}[]
        for (t, k) in tg
            st = t === :goal ? GOAL_S : t === :crash ? CRASH_S : t::NTuple{7,Int}
            push!(v, (st, k / n)); C[(ds, ai, st)] = costs[(ds, ai)][t] / k
        end
        P[(ds, ai)] = v
    end
    return DuckieSSP(P, C), argmax(starts)
end

println("sampling the SSP structure...")
ssp, start_state = sample_structure()

make_planner(; known_costs) = solve(DORASolver(
    start = start_state,
    classify = sp -> sp == GOAL_S ? :goal : sp == CRASH_S ? :crash : :normal,
    cost = (s, a, sp) -> get(ssp.C, (s, a, sp), 1.0),
    c_min = 0.5, c_to = 200.0, c_crash = 100.0, horizon = 150,
    known_costs = known_costs, optimistic = true, episodes_budget = 500), ssp)

# --- deploy online in the REAL environment ----------------------------------
"""Run real episodes. When `learn`, every decision's observed cost is reported
back with `observe!`, which is the interface the solver documents for
deployment-time cost learning."""
function deploy!(planner; episodes, horizon=150, seed=99, learn::Bool)
    known = Set(planner.tab.states)
    rng = MersenneTwister(seed)
    outcomes = Symbol[]
    for _ in 1:episodes
        s = rand(rng, initialstate(MDP_))
        out = :timeout
        for t in 1:horizon
            ds = disc(s)
            inmodel = ds in known
            ai = inmodel ? action(planner, ds) : behaviour_action(s, rng, 0.0)
            res = simulate_decision(MDP_.transition, s, ACTS[ai], rng)
            if learn && inmodel
                observe!(planner, ds, ai, step_cost(res))
            end
            cls = classify_result(res)
            if cls === :goal
                out = :goal; break
            elseif cls === :crash || res.terminated || res.truncated
                out = :crash; break
            end
            s = res.sp
        end
        push!(outcomes, out)
    end
    return outcomes
end

rate(o, sym) = count(==(sym), o) / length(o)

println("\n=== mode comparison, 400 real episodes each ===")
for (label, kc, learn) in (("offline  (known_costs=true)",  true,  false),
                           ("online   (known_costs=false)", false, true))
    p = make_planner(known_costs = kc)
    o = deploy!(p; episodes = 400, learn = learn)
    @printf("%-30s goal %5.1f%%   crash %5.1f%%   timeout %5.1f%%\n",
            label, 100rate(o, :goal), 100rate(o, :crash), 100rate(o, :timeout))
end

# does it improve as it observes? split the same online run into blocks
println("\n=== online learning curve (blocks of 100 real episodes) ===")
p = make_planner(known_costs = false)
for blk in 1:6
    o = deploy!(p; episodes = 100, seed = 1000 + blk, learn = true)
    @printf("  block %d  goal %5.1f%%   crash %5.1f%%\n",
            blk, 100rate(o, :goal), 100rate(o, :crash))
end

# control: the same schedule without reporting anything back
println("\n=== control: identical schedule, observe! never called ===")
p2 = make_planner(known_costs = false)
for blk in 1:6
    o = deploy!(p2; episodes = 100, seed = 1000 + blk, learn = false)
    @printf("  block %d  goal %5.1f%%   crash %5.1f%%\n",
            blk, 100rate(o, :goal), 100rate(o, :crash))
end
