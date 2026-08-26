# Why does the K = 8 model say a zigzag lap is impossible (crash 1.000)?
#
# Hypothesis: the macro timescale. On small_loop every curve bends the same
# way and spans several tiles, so holding one command for 1.6 s works. The
# zigzag alternates left and right turns within a couple of tiles; if the
# road changes direction INSIDE one held command, no action can track it.
# If that is the cause, shortening K should turn in-model success from 0 to 1
# without touching anything else. Measure, don't assume.

using DuckietownDecisionModels
using POMDPs, POMDPTools, DORASolvers
using Random, Printf

# max_steps is 1500 PHYSICS FRAMES = 250 decisions; a 26-tile zigzag lap
# needs ~330 decisions, so within the stock budget every path ends in TIMEOUT
# — which classifies as terminal, which the tabular model reports as crash
# 1.000 at every K. The experiment raises the budget; nothing shipped changes.
const CFG = let base = scenario_config(:lane_following)
    env = DuckietownDecisionModels._with(base.environment, max_steps = 9000)
    DuckietownConfig(base.algorithm, base.stage, base.seed, env, base.state,
        base.continuous_state, base.actions, base.duck_controller,
        base.reward, base.solver, base.lane_teacher, base.transition_model,
        base.training, base.evaluation, base.wandb)
end
const BASE = DuckietownMDP(CFG; action_space = :discrete)
const TR = BASE.transition
const SCFG = TR.state_cfg
const ACTS = collect(POMDPs.actions(BASE))
const RNG = MersenneTwister(1)
const C_MIN = 0.05

step_cost(r) = max(C_MIN, 1.0 - r.reward.total)
raw_of(s) = first(get_raw_state(s, SCFG))
tile_of(s) = get_grid_coords(s.map, collect(s.ego.pos))

zigzag_tiles() = [
    "asphalt" "asphalt"      "asphalt"       "asphalt"       "asphalt"       "asphalt"    "asphalt"       "asphalt"      "asphalt"
    "asphalt" "curve_left/W" "curve_left/N"  "asphalt"       "curve_left/W"  "straight/W" "straight/W"    "curve_left/N" "asphalt"
    "asphalt" "straight/S"   "curve_right/W" "straight/W"    "curve_right/S" "asphalt"    "curve_right/N" "curve_left/E" "asphalt"
    "asphalt" "straight/S"   "asphalt"       "asphalt"       "asphalt"       "asphalt"    "straight/N"    "asphalt"      "asphalt"
    "asphalt" "straight/S"   "asphalt"       "asphalt"       "curve_right/N" "straight/E" "curve_left/E"  "asphalt"      "asphalt"
    "asphalt" "straight/S"   "asphalt"       "curve_right/N" "curve_left/E"  "asphalt"    "asphalt"       "asphalt"      "asphalt"
    "asphalt" "straight/S"   "asphalt"       "straight/N"    "asphalt"       "asphalt"    "asphalt"       "asphalt"      "asphalt"
    "asphalt" "curve_left/S" "straight/E"    "curve_left/E"  "asphalt"       "asphalt"    "asphalt"       "asphalt"      "asphalt"
    "asphalt" "asphalt"      "asphalt"       "asphalt"       "asphalt"       "asphalt"    "asphalt"       "asphalt"      "asphalt"
]

const ZMAP = RoadMap("zigzag_dists", 0.585,
    parse_map_tiles(zigzag_tiles(), 0.585), MapObjectData[])

function spawn_zigzag(rng; attempts = 500)
    tiles = collect(drivable_tiles(ZMAP))
    for _ in 1:attempts
        t = tiles[rand(rng, 1:length(tiles))]
        pos, angle = try
            sample_spawn_pose(ZMAP, rng, t[1], t[2], 10.0)
        catch
            continue
        end
        ego = initial_ego(Tuple(pos), angle, size(ZMAP.grid, 1), ZMAP.tile_size)
        w = DuckieWorldState(ego, DuckieState[], StopSignState[], ZMAP,
            StopMemory(false, 0, nothing, nothing), (0.0, 0.0),
            Int[], Bool[], MersenneTwister(7))
        raw, fb = try
            get_raw_state(w, SCFG)
        catch
            continue
        end
        (abs(raw.d) <= 0.08 && abs(raw.phi) <= 0.175) || continue
        w.lane_fallback = fb
        return w
    end
    error("no acceptable spawn")
end

function tile_connections(t)
    i0, j0 = t
    ts = ZMAP.tile_size
    spec = ZMAP.grid[j0 + 1, i0 + 1]
    sides = Set{NTuple{2,Int}}()
    for curve in spec.curves, p in (curve[1], curve[4])
        fx = p[1] / ts - i0
        fz = p[3] / ts - j0
        fx < 0.02 && push!(sides, (i0 - 1, j0))
        fx > 0.98 && push!(sides, (i0 + 1, j0))
        fz < 0.02 && push!(sides, (i0, j0 - 1))
        fz > 0.98 && push!(sides, (i0, j0 + 1))
    end
    return sides
end

function ring_order(w0)
    ts = Set(drivable_tiles(ZMAP))
    conn = Dict(t => [x for x in tile_connections(t) if x in ts] for t in ts)
    start = tile_of(w0)
    ring = [start]
    prev = start
    cur = first(conn[start])
    while cur != start
        push!(ring, cur)
        nxt = first(x for x in conn[cur] if x != prev)
        prev = cur
        cur = nxt
    end
    fwd, _ = lane_frame_tabular(w0)
    c(t) = ((t[1] + 0.5) * ZMAP.tile_size, (t[2] + 0.5) * ZMAP.tile_size)
    d1 = (c(ring[2])[1] - c(ring[1])[1], c(ring[2])[2] - c(ring[1])[2])
    if d1[1] * fwd[1] + d1[2] * fwd[3] < 0
        ring = vcat(ring[1:1], reverse(ring[2:end]))
    end
    return ring
end

const w0 = spawn_zigzag(MersenneTwister(2001))
const RING = ring_order(w0)
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

# K lives in a Ref so one script probes several macro lengths
const KREF = Ref(8)

function macro_step(ls::LapState, a)
    s = ls.s; prog = ls.prog; tile = tile_of(s); c = 0.0
    for _ in 1:KREF[]
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

const s0 = LapState(w0, 0)

@printf("%-4s %8s %10s %10s %8s %8s %8s\n",
        "K", "states", "cost", "time s", "success", "crash", "timeout")
for k in (8, 6, 4, 2)
    KREF[] = k
    t0 = time()
    planner = solve(DORASolver(
            start = s0, classify = classify,
            cost = (ls, a, lsp) -> macro_step(ls, a)[2],
            key = lapkey,
            c_min = k * C_MIN, c_to = 5000.0, c_crash = 1000.0,
            horizon = ceil(Int, 8 * 200 / k),
        ), LapMDP())
    tab = planner.tab
    V, pistar = optimal_value(tab)
    sr, cr, to = outcome_rates(tab, pistar)
    @printf("%-4d %8d %10.2f %10.1f %8.3f %8.3f %8.3f\n",
            k, tab.S, V[tab.start], time() - t0, sr, cr, to)
end
