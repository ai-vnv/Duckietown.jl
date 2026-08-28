# DORA completing a full lap of small_loop, with the ducks and the stop sign.
#
# Built on the three measured facts from this session:
#   - the transition is deterministic given (s,a)   (2940/2940 identical)
#   - `key` is the package's documented aggregation  (04_custom_mdp pattern)
#   - one action must span ~8 decisions, or determinism makes BFS explore a
#     single forward orbit (k=1 -> 4 states; k=8 -> 172 states, whole ring)
#
# A lap needs progress in the state: the 7-tuple is purely local lane geometry
# and cannot tell "one tile from home" from "just started". So the MDP state
# is (world state, ring progress) and the key carries the progress.

using DuckietownDecisionModels
using POMDPs, POMDPTools, DORASolvers
using Random, Printf, Statistics, Serialization

const CFG = scenario_config(:stop_and_duck)
const BASE = DuckietownMDP(CFG; action_space = :discrete)
const TR = BASE.transition
const SCFG = TR.state_cfg
const ACTS = collect(POMDPs.actions(BASE))
const RNG = MersenneTwister(1)
const K = 8

raw_of(s) = first(get_raw_state(s, SCFG))
tile_of(s) = get_grid_coords(s.map, collect(s.ego.pos))

const RING = let m = initial_map(CFG)
    t = collect(drivable_tiles(m))
    cx = sum(first.(t)) / length(t); cy = sum(last.(t)) / length(t)
    sort(t; by = p -> atan(p[2] - cy, p[1] - cx))
end
const RIX = Dict(t => i for (i, t) in enumerate(RING))
const NRING = length(RING)

"""Monotone ring counter: only a forward step to the next ring tile advances."""
function advance(prog, prev, new)
    new == prev && return prog
    a = get(RIX, prev, 0); b = get(RIX, new, 0)
    (a == 0 || b == 0) && return prog
    return mod(b - a, NRING) == 1 ? prog + 1 : prog
end

"""MDP state: the world plus how far around the ring we have come."""
struct LapState
    s::DuckieWorldState
    prog::Int
end

"""One macro action: hold a command for K decisions, tracking ring progress
and accumulating the real cost. Exact — the chain consumes no randomness."""
function macro_step(ls::LapState, a)
    s = ls.s; prog = ls.prog; tile = tile_of(s); c = 0.0
    for _ in 1:K
        r = simulate_decision(TR, s, a, RNG)
        c += 1.0 + 4.0abs(r.raw_state.d) + 0.5abs(r.raw_state.phi)
        nt = tile_of(r.sp)
        prog = advance(prog, tile, nt)
        tile = nt; s = r.sp
        (r.terminated || r.truncated) && return (LapState(s, prog), c, true)
    end
    return (LapState(s, prog), c, false)
end

struct LapMDP <: MDP{LapState,MacroAction} end
POMDPs.actions(::LapMDP) = ACTS
POMDPs.discount(::LapMDP) = 1.0
POMDPs.isterminal(::LapMDP, ls) = POMDPs.isterminal(BASE, ls.s)
POMDPs.transition(::LapMDP, ls, a) = Deterministic(first(macro_step(ls, a)))

lapkey(ls) = (min(ls.prog, NRING), discretize(raw_of(ls.s)))
classify(ls) = POMDPs.isterminal(BASE, ls.s) ? :crash :
               ls.prog >= NRING ? :goal : :normal

s0 = LapState(rand(MersenneTwister(1001), initialstate(BASE)), 0)
@printf("ring of %d tiles, start %s, goal = %d forward steps (one lap)\n",
        NRING, tile_of(s0.s), NRING)

println("\ntabularizing...")
t0 = time()
planner = solve(DORASolver(
        start = s0, classify = classify,
        cost = (ls, a, lsp) -> macro_step(ls, a)[2],
        key = lapkey,
        c_min = 0.5, c_to = 2000.0, c_crash = 1000.0, horizon = 120,
    ), LapMDP())
tab = planner.tab
@printf("  %d states, MAXOUT %d   (%.1f s)\n", tab.S, tab.MAXOUT, time() - t0)

V, pistar = optimal_value(tab)
sr, cr, tr_ = outcome_rates(tab, pistar)
@printf("\nin-model: optimal cost %.2f   success %.3f   crash %.3f   timeout %.3f\n",
        V[tab.start], sr, cr, tr_)
worst, frac = causality_margin(tab, V, pistar)
@printf("causality margin worst %.3f, states ok %.3f\n", worst, frac)

# --- execute, recording EVERY underlying decision for the video -------------
function execute(; horizon = 120)
    ls = s0
    frames = [(ls.s, 0)]        # (world state, ring progress)
    total = 0.0
    for t in 1:horizon
        POMDPs.isterminal(BASE, ls.s) && return (:crash, total, frames, "isterminal")
        ls.prog >= NRING && return (:lap, total, frames, "lap")
        a = action(planner, ls)
        # replay the macro action decision by decision so the video is smooth
        s = ls.s; prog = ls.prog; tile = tile_of(s)
        for _ in 1:K
            r = simulate_decision(TR, s, a, RNG)
            total += 1.0 + 4.0abs(r.raw_state.d) + 0.5abs(r.raw_state.phi)
            nt = tile_of(r.sp); prog = advance(prog, tile, nt)
            tile = nt; s = r.sp
            push!(frames, (s, prog))
            # completing the lap wins over a same-step termination: check the
            # goal first, and report the reason either way rather than
            # collapsing both into "crash"
            prog >= NRING && return (:lap, total, frames, string(r.reason))
            if r.terminated || r.truncated
                return (:crash, total, frames, string(r.reason))
            end
        end
        ls = LapState(s, prog)
    end
    return (:timeout, total, frames, "horizon")
end

outcome, total, frames, reason = execute()
@printf("\nexecuted: %s after %d decisions, cost %.2f\n",
        outcome, length(frames) - 1, total)
@printf("ring progress reached: %d of %d\n", last(frames)[2], NRING)
println("tiles visited in order: ",
        unique([tile_of(f[1]) for f in frames]))
@printf("in-model cost %.2f  vs  executed %.2f   difference %.4f\n",
        V[tab.start], total, abs(V[tab.start] - total))

serialize(joinpath(@__DIR__, "lap_frames.jls"),
          (frames = frames, outcome = outcome, total = total,
           cost_model = V[tab.start]))
println("\nwrote lap_frames.jls (", length(frames), " frames)")
