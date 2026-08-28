# How should Duckietown.jl discretize state so that an explicit-interface
# solver gets a usable graph?
#
# The transition is deterministic, so `transition` can be exact and `key` is
# the documented aggregation mechanism. The open question is the GRANULARITY
# of that key: too coarse and BFS collapses (the 7-tuple gives 4 states), too
# fine and the graph never closes.
#
# This measures the trade-off directly, which is the engineering input the
# proposed discrete formulation needs.

using DuckietownDecisionModels
using POMDPs, POMDPTools, DORASolvers
using Random, Printf

const BASE = DuckietownMDP(scenario_config(:stop_and_duck); action_space = :discrete)
const TR = BASE.transition
const SCFG = TR.state_cfg
const ACTS = collect(POMDPs.actions(BASE))
const RNG = MersenneTwister(1)

tile_of(s) = get_grid_coords(s.map, collect(s.ego.pos))
raw_of(s) = first(get_raw_state(s, SCFG))

struct DuckieExact <: MDP{DuckieWorldState,MacroAction} end
POMDPs.actions(::DuckieExact) = ACTS
POMDPs.discount(::DuckieExact) = 1.0
POMDPs.isterminal(::DuckieExact, s) = POMDPs.isterminal(BASE, s)
POMDPs.transition(::DuckieExact, s, a) =
    Deterministic(simulate_decision(TR, s, a, RNG).sp)

bin(x, w) = round(Int, x / w)

# candidate keys, coarse to fine
const KEYS = (
    ("tabular 7-tuple (FJ2)", s -> discretize(raw_of(s))),
    ("tile only", s -> tile_of(s)),
    ("tile + d/0.05 + phi/0.15", s -> (tile_of(s)..., bin(raw_of(s).d, 0.05),
                                       bin(raw_of(s).phi, 0.15))),
    ("tile + d/0.02 + phi/0.05", s -> (tile_of(s)..., bin(raw_of(s).d, 0.02),
                                       bin(raw_of(s).phi, 0.05))),
    ("+ speed/0.05", s -> (tile_of(s)..., bin(raw_of(s).d, 0.02),
                           bin(raw_of(s).phi, 0.05), bin(raw_of(s).v, 0.05))),
    ("+ speed/0.02", s -> (tile_of(s)..., bin(raw_of(s).d, 0.01),
                           bin(raw_of(s).phi, 0.03), bin(raw_of(s).v, 0.02))),
)

"""BFS over keys with exact deterministic edges, exactly as tabularize does,
but instrumented: how many keys, how many tiles reached, how deep."""
function explore(key; start, maxstates = 20_000)
    idx = Dict{Any,Int}(key(start) => 1)
    slist = Any[start]
    tiles = Set{NTuple{2,Int}}([tile_of(start)])
    head = 1
    edges = 0
    terminal = 0
    while head <= length(slist) && length(slist) < maxstates
        s = slist[head]
        if !POMDPs.isterminal(BASE, s)
            for a in ACTS
                sp = simulate_decision(TR, s, a, RNG).sp
                edges += 1
                POMDPs.isterminal(BASE, sp) && (terminal += 1)
                k = key(sp)
                if !haskey(idx, k)
                    idx[k] = length(slist) + 1
                    push!(slist, sp)
                    push!(tiles, tile_of(sp))
                end
            end
        end
        head += 1
    end
    return (states = length(slist), tiles = length(tiles), edges = edges,
            terminal = terminal, exhausted = head > length(slist))
end

s0 = rand(MersenneTwister(1001), initialstate(BASE))
@printf("start tile %s\n\n", tile_of(s0))
@printf("%-28s %8s %7s %8s %10s %s\n",
        "key", "states", "tiles", "edges", "terminal", "closed")
for (name, k) in KEYS
    t0 = time()
    r = explore(k; start = s0)
    @printf("%-28s %8d %7d %8d %10d %s   (%.1f s)\n",
            name, r.states, r.tiles, r.edges, r.terminal,
            r.exhausted ? "yes" : "NO (hit cap)", time() - t0)
end

println("""

'closed' means BFS exhausted the reachable set. 'tiles' is how much of the
8-tile ring the abstraction can actually represent reaching — a key that
reaches one tile cannot express a navigation problem at all.""")

# ---------------------------------------------------------------------------
# Determinism means BFS explores a single forward ORBIT per representative,
# not the set of states the system can occupy. Does making each action k
# decisions long change that, or is the collapse structural?
# ---------------------------------------------------------------------------
function explore_k(key, k; start, maxstates = 20_000)
    idx = Dict{Any,Int}(key(start) => 1)
    slist = Any[start]
    tiles = Set{NTuple{2,Int}}([tile_of(start)])
    head = 1
    while head <= length(slist) && length(slist) < maxstates
        s = slist[head]
        if !POMDPs.isterminal(BASE, s)
            for a in ACTS
                sp = s
                dead = false
                for _ in 1:k
                    r = simulate_decision(TR, sp, a, RNG)
                    sp = r.sp
                    (r.terminated || r.truncated) && (dead = true; break)
                end
                dead && continue
                kk = key(sp)
                if !haskey(idx, kk)
                    idx[kk] = length(slist) + 1
                    push!(slist, sp)
                    push!(tiles, tile_of(sp))
                end
            end
        end
        head += 1
    end
    return (states = length(slist), tiles = length(tiles),
            closed = head > length(slist))
end

println("\none action = k decisions of the same command, key = tabular 7-tuple:")
@printf("%4s %8s %7s %s\n", "k", "states", "tiles", "closed")
for k in (1, 2, 4, 8, 16, 32)
    r = explore_k(s -> discretize(raw_of(s)), k; start = s0)
    @printf("%4d %8d %7d %s\n", k, r.states, r.tiles,
            r.closed ? "yes" : "NO (hit cap)")
end
