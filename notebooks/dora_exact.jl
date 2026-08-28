# DORA on Duckietown with an EXACT transition — no sampling anywhere.
#
# The transition is deterministic given (s, a): 2940/2940 identical successors
# across rng streams, 0 divergence over episodes. So
#
#     transition(m, s, a) = Deterministic(simulate_decision(tr, s, a, rng).sp)
#
# is exact, and `key = discretize` is the package's own documented mechanism
# for a state type that should be aggregated. Nothing here estimates a kernel,
# which is what DORA's premise requires.
#
# Written to match examples/04_custom_mdp.jl: only actions, discount,
# isterminal, initialstate and transition — no `states`, no `stateindex`,
# no `reward`. tabularize discovers the state set itself.

using DuckietownDecisionModels
using POMDPs, POMDPTools, DORASolvers
using Random, Printf, Statistics

const BASE = DuckietownMDP(scenario_config(:stop_and_duck); action_space = :discrete)
const TR = BASE.transition
const SCFG = TR.state_cfg
const ACTS = collect(POMDPs.actions(BASE))
const RNG = MersenneTwister(1)          # unused by the chain; kept for the API

tile_of(s) = get_grid_coords(s.map, collect(s.ego.pos))
dkey(s) = discretize(first(get_raw_state(s, SCFG)))

"""The deterministic Duckietown model, exposed through the EXPLICIT interface."""
struct DuckieExact <: MDP{DuckieWorldState,MacroAction} end

POMDPs.actions(::DuckieExact) = ACTS
POMDPs.discount(::DuckieExact) = 1.0
POMDPs.isterminal(::DuckieExact, s) = POMDPs.isterminal(BASE, s)
POMDPs.transition(::DuckieExact, s, a) =
    Deterministic(simulate_decision(TR, s, a, RNG).sp)

model = DuckieExact()
s0 = rand(MersenneTwister(1001), initialstate(BASE))
@printf("start tile %s   d %.4f\n", tile_of(s0),
        first(get_raw_state(s0, SCFG)).d)

# Goal: reach a target tile. That is a property of the STATE, so classify can
# decide it without needing the transition's event flags.
const GOAL_TILE = (0, 1)     # two tiles around the ring from the spawn tile

classify(sp) = POMDPs.isterminal(BASE, sp) ? :crash :
               tile_of(sp) == GOAL_TILE ? :goal : :normal

# positive, undiscounted: one unit per decision plus lane deviation
function cost(s, a, sp)
    raw, _ = get_raw_state(sp, SCFG)
    return 1.0 + 4.0abs(raw.d) + 0.5abs(raw.phi)
end

println("\ntabularizing (BFS over discrete keys, exact deterministic edges)...")
t0 = time()
planner = solve(DORASolver(
        start = s0,
        classify = classify,
        cost = cost,
        key = dkey,                 # the documented aggregation mechanism
        c_min = 0.5, c_to = 400.0, c_crash = 200.0, horizon = 200,
    ), model)
tab = planner.tab
@printf("  %d ordinary states, %d actions, MAXOUT %d   (%.1f s)\n",
        tab.S, tab.NA, tab.MAXOUT, time() - t0)
@printf("  MAXOUT == 1 means every edge is deterministic: %s\n",
        tab.MAXOUT == 1 ? "yes" : "NO — some (s,a) has several outcomes")

V, pistar = optimal_value(tab)
sr, cr, tr_ = outcome_rates(tab, pistar)
@printf("\noptimal cost from start %.2f   success %.3f   crash %.3f   timeout %.3f\n",
        V[tab.start], sr, cr, tr_)

worst, frac = causality_margin(tab, V, pistar)
@printf("causality margin: worst %.3f, fraction of states ok %.3f\n", worst, frac)

gap = (eval_policy(tab, planner.pi)[tab.start] - V[tab.start]) / V[tab.start]
@printf("DORA policy vs SSP optimum: relative gap %.4f\n", gap)

# ---------------------------------------------------------------------------
# Execute DORA's policy on the model it was built from. Because the model is
# exact and deterministic, in-model and executed should AGREE exactly — that
# is the check the sampled version could never pass.
# ---------------------------------------------------------------------------
function run_policy(; horizon = 200)
    s = s0
    total = 0.0
    for t in 1:horizon
        POMDPs.isterminal(BASE, s) && return (:crash, t, total)
        tile_of(s) == GOAL_TILE && return (:goal, t, total)
        a = action(planner, s)
        sp = simulate_decision(TR, s, a, RNG).sp
        total += cost(s, a, sp)
        s = sp
    end
    return (:timeout, horizon, total)
end

out, steps, total = run_policy()
@printf("\nexecuted on the real dynamics: %s after %d decisions, cost %.2f\n",
        out, steps, total)
@printf("in-model optimal cost %.2f  vs  executed cost %.2f   difference %.2f\n",
        V[tab.start], total, abs(V[tab.start] - total))
