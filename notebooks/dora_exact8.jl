# DORA on Duckietown, exact and closed.
#
# Three facts, each measured earlier in this session:
#   1. the transition is deterministic given (s,a)      2940/2940 identical
#   2. `key` is the package's documented aggregation     04_custom_mdp pattern
#   3. one action must span ~8 decisions or BFS collapses into a single
#      forward orbit (k=1 -> 4 states/1 tile; k=8 -> 182 states/8 tiles)
#
# So: exact deterministic edges, the FJ2 discretization as the key, and a
# macro action of 8 decisions. Nothing is sampled and no kernel is estimated,
# which is what DORA's premise requires.

using DuckietownDecisionModels
using POMDPs, POMDPTools, DORASolvers
using Random, Printf, Statistics

const BASE = DuckietownMDP(scenario_config(:stop_and_duck); action_space = :discrete)
const TR = BASE.transition
const SCFG = TR.state_cfg
const ACTS = collect(POMDPs.actions(BASE))
const RNG = MersenneTwister(1)          # the chain never draws from it
const K = 8                             # decisions per macro action

raw_of(s) = first(get_raw_state(s, SCFG))
tile_of(s) = get_grid_coords(s.map, collect(s.ego.pos))
dkey(s) = discretize(raw_of(s))

"""Hold one command for K decisions. Returns (successor, accumulated cost,
died). Exact: the chain consumes no randomness."""
function macro_step(s, a)
    c = 0.0
    for _ in 1:K
        r = simulate_decision(TR, s, a, RNG)
        c += 1.0 + 4.0abs(r.raw_state.d) + 0.5abs(r.raw_state.phi)
        s = r.sp
        (r.terminated || r.truncated) && return (s, c, true)
    end
    return (s, c, false)
end

struct DuckieMacro <: MDP{DuckieWorldState,MacroAction} end
POMDPs.actions(::DuckieMacro) = ACTS
POMDPs.discount(::DuckieMacro) = 1.0
POMDPs.isterminal(::DuckieMacro, s) = POMDPs.isterminal(BASE, s)
POMDPs.transition(::DuckieMacro, s, a) = Deterministic(first(macro_step(s, a)))

s0 = rand(MersenneTwister(1001), initialstate(BASE))
const START_TILE = tile_of(s0)

# goal: get three tiles around the ring — a property of the state, so
# `classify` can decide it without the transition's event flags
const RING = let m = s0.map
    t = collect(drivable_tiles(m))
    cx = sum(first.(t)) / length(t); cy = sum(last.(t)) / length(t)
    sort(t; by = p -> atan(p[2] - cy, p[1] - cx))
end
const RIX = Dict(t => i for (i, t) in enumerate(RING))
const GOAL_TILE = RING[mod1(RIX[START_TILE] + 3, length(RING))]

@printf("start tile %s   goal tile %s   (3 steps around an %d-tile ring)\n",
        START_TILE, GOAL_TILE, length(RING))

classify(sp) = POMDPs.isterminal(BASE, sp) ? :crash :
               tile_of(sp) == GOAL_TILE ? :goal : :normal
cost(s, a, sp) = max(0.5, last(macro_step(s, a)) === true ? 1.0 :
                          macro_step(s, a)[2])

println("\ntabularizing...")
t0 = time()
planner = solve(DORASolver(
        start = s0, classify = classify,
        cost = (s, a, sp) -> macro_step(s, a)[2],
        key = dkey,
        c_min = 0.5, c_to = 400.0, c_crash = 200.0, horizon = 60,
    ), DuckieMacro())
tab = planner.tab
@printf("  %d states, %d actions, MAXOUT %d  (%.1f s)\n",
        tab.S, tab.NA, tab.MAXOUT, time() - t0)

V, pistar = optimal_value(tab)
sr, cr, tr_ = outcome_rates(tab, pistar)
@printf("\nin-model: optimal cost %.2f   success %.3f   crash %.3f   timeout %.3f\n",
        V[tab.start], sr, cr, tr_)
worst, frac = causality_margin(tab, V, pistar)
@printf("causality margin worst %.3f, states ok %.3f\n", worst, frac)
gap = (eval_policy(tab, planner.pi)[tab.start] - V[tab.start]) / V[tab.start]
@printf("DORA policy vs SSP optimum: gap %.4f\n", gap)

# --- execute on the real dynamics -------------------------------------------
# The model is exact and deterministic, so this should agree with the model,
# not merely be close to it.
function execute(; horizon = 60)
    s = s0
    total = 0.0
    path = [tile_of(s)]
    for t in 1:horizon
        POMDPs.isterminal(BASE, s) && return (:crash, t, total, path)
        tile_of(s) == GOAL_TILE && return (:goal, t, total, path)
        a = action(planner, s)
        sp, c, died = macro_step(s, a)
        total += c
        s = sp
        push!(path, tile_of(s))
        died && return (:crash, t, total, path)
    end
    return (:timeout, horizon, total, path)
end

out, steps, total, path = execute()
@printf("\nexecuted: %s after %d macro actions (%d decisions), cost %.2f\n",
        out, steps, steps * K, total)
println("tiles visited: ", unique(path))
@printf("\nin-model optimal cost %.2f   executed cost %.2f   difference %.4f\n",
        V[tab.start], total, abs(V[tab.start] - total))
