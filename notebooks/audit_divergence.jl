# Where does execution stop matching the determinized model?
#
# The table's successor for (key, action) is computed once, from the BFS
# representative of that key. Execution visits other concrete states with the
# same key; if the same macro action from those states lands in a DIFFERENT
# key, the plan and reality fork. Find the first fork instead of theorising.

using DuckietownDecisionModels
using POMDPs, POMDPTools, DORASolvers
using Random, Printf

const CFG = scenario_config(:stop_and_duck_safe)
const BASE = DuckietownMDP(CFG; action_space = :discrete)
const TR = BASE.transition
const SCFG = TR.state_cfg
const ACTS = collect(POMDPs.actions(BASE))
const RNG = MersenneTwister(1)
const K = 8
const STEP_COST = 1.0
const C_MIN = 0.05

raw_of(s) = first(get_raw_state(s, SCFG))
tile_of(s) = get_grid_coords(s.map, collect(s.ego.pos))
step_cost(r) = max(C_MIN, STEP_COST - r.reward.total)

const RING = let m = initial_map(CFG)
    t = collect(drivable_tiles(m))
    cx = sum(first.(t)) / length(t); cy = sum(last.(t)) / length(t)
    sort(t; by = p -> atan(p[2] - cy, p[1] - cx))
end
const RIX = Dict(t => i for (i, t) in enumerate(RING))
const NRING = length(RING)

function advance(prog, prev, new)
    new == prev && return prog
    a = get(RIX, prev, 0); b = get(RIX, new, 0)
    (a == 0 || b == 0) && return prog
    return mod(b - a, NRING) == 1 ? prog + 1 : prog
end

struct LapState
    s::DuckieWorldState
    prog::Int
end

function macro_step(ls::LapState, a)
    s = ls.s; prog = ls.prog; tile = tile_of(s); c = 0.0
    for _ in 1:K
        r = simulate_decision(TR, s, a, RNG)
        c += step_cost(r)
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
raw0 = raw_of(s0.s)
@printf("start pose (%.3f, %.3f) angle %.1f deg   d %.3f  phi %.3f  d_stop %s\n",
        s0.s.ego.pos[1], s0.s.ego.pos[3], rad2deg(s0.s.ego.angle),
        raw0.d, raw0.phi,
        raw0.d_stop === nothing ? "none" : @sprintf("%.3f", raw0.d_stop))

planner = solve(DORASolver(
        start = s0, classify = classify,
        cost = (ls, a, lsp) -> macro_step(ls, a)[2],
        key = lapkey,
        c_min = K * C_MIN, c_to = 2000.0, c_crash = 1000.0, horizon = 120,
    ), LapMDP())
tab = planner.tab
V, pistar = optimal_value(tab)
@printf("%d states   in-model cost %.2f\n\n", tab.S, V[tab.start])

# the table keeps states but not the key->index map; rebuild it
const INDEX = Dict(lapkey(tab.states[i]) => i for i in 1:tab.S)

describe(i) = i == tab.goal ? "GOAL" : i == tab.crash ? "CRASH" :
              i == 0 ? "unexplored" : string(lapkey(tab.states[i]))

# trace: at each step compare the table's predicted successor key with the key
# the executed successor actually has
function trace(; horizon = 120)
    ls = s0
    for t in 1:horizon
        POMDPs.isterminal(BASE, ls.s) && (println("terminal at step $t"); return)
        ls.prog >= NRING && (println("LAP at step $t"); return)
        a = action(planner, ls)
        si = INDEX[lapkey(ls)]
        ai = findfirst(==(a), ACTS)
        pred = tab.succ[si, ai]                     # predicted successor index
        nls = first(macro_step(ls, a))
        nk = lapkey(nls)
        actual = POMDPs.isterminal(BASE, nls.s) ? tab.crash :
                 nls.prog >= NRING ? tab.goal : get(INDEX, nk, 0)
        raw = raw_of(ls.s)
        mark = pred == actual ? "" : "   <-- FORK"
        @printf("%3d prog %d  a %-13s  d %+.3f phi %+.3f v %.3f  pred %4d actual %4d%s\n",
                t, ls.prog, string(a), raw.d, raw.phi, raw.v, pred, actual, mark)
        if pred != actual
            @printf("     predicted: %s\n", describe(pred))
            @printf("     actual   : %s\n", describe(actual))
            # how far apart are the two concrete worlds that share this key?
            rep = tab.states[si]
            @printf("     key rep pose (%.3f, %.3f) angle %.1f vs executed (%.3f, %.3f) angle %.1f\n",
                    rep.s.ego.pos[1], rep.s.ego.pos[3], rad2deg(rep.s.ego.angle),
                    ls.s.ego.pos[1], ls.s.ego.pos[3], rad2deg(ls.s.ego.angle))
            return
        end
        ls = nls
    end
end

trace()
