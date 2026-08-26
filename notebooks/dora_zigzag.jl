# DORA on zigzag_dists: a 26-tile non-convex loop with left AND right turns.
#
# What transfers from the small_loop experiment unchanged: the transition
# model (the map lives in the state, the model is map-agnostic), determinism,
# macro actions (K = 8), the documented cost form, substep recording with
# bit-identity assertions, and full receding horizon.
#
# What has to change, and why:
#   - the map is built here with the package's own exported FJ3.1 parser
#     (`parse_map_tiles`) from the YAML tile matrix, transcribed verbatim;
#   - the ring order can NOT be an angular sort around the centroid — the
#     loop is non-convex, so tiles are ordered by walking the adjacency
#     graph (every drivable tile in an intersection-free loop has exactly
#     two drivable neighbours; asserted, not assumed);
#   - travel direction is read off the lane frame at the spawn, not guessed;
#   - no ducks and no stop sign exist on this map: the objects in the YAML
#     are optional visual distractors on border tiles, aimed at camera-based
#     lane followers; this model reads state, not pixels, so the test here
#     is purely the lane-following torture course.

using DuckietownDecisionModels
using POMDPs, POMDPTools, DORASolvers
using Random, Printf, Statistics, Serialization, JSON3

# max_steps is 1500 PHYSICS FRAMES = 250 decisions; a 26-tile zigzag lap
# needs ~330 decisions, so within the stock budget every path ends in TIMEOUT
# — which classifies as terminal, which the tabular model reports as crash
# 1.000 at every K (measured before this fix). The experiment raises the
# budget; nothing shipped changes.
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
const K = 8
const STEP_COST = 1.0
const C_MIN = 0.05

step_cost(r) = max(C_MIN, STEP_COST - r.reward.total)
raw_of(s) = first(get_raw_state(s, SCFG))
tile_of(s) = get_grid_coords(s.map, collect(s.ego.pos))

# --- the map, transcribed verbatim from zigzag_dists.yaml ------------------
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

# --- spawn: the package's own sampler + the scenario's sanity bounds --------
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
    error("no acceptable spawn in $attempts attempts")
end

# --- ring order by ROAD connectivity, oriented by the spawn's lane frame ----
#
# Plain 4-adjacency is wrong on this map: two parallel road segments touch
# without being connected (the left column brushes the top curves), so a
# tile can have 3 drivable neighbours while the road itself never branches.
# The truth is in the tile's own lane curves: each straight/curve tile's
# Bezier endpoints sit exactly on the two borders the road crosses, so the
# connected neighbours are the cells across those two borders.
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
    for (t, ns) in conn
        length(ns) == 2 || error("tile $t connects to $(length(ns)) road tiles")
    end
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
    # orient so that step +1 is the direction the lane actually travels
    fwd, _ = lane_frame_tabular(w0)
    c(t) = ((t[1] + 0.5) * ZMAP.tile_size, (t[2] + 0.5) * ZMAP.tile_size)
    d1 = (c(ring[2])[1] - c(ring[1])[1], c(ring[2])[2] - c(ring[1])[2])
    if d1[1] * fwd[1] + d1[2] * fwd[3] < 0
        ring = vcat(ring[1:1], reverse(ring[2:end]))
    end
    return ring
end

w0 = spawn_zigzag(MersenneTwister(2001))
const RING = ring_order(w0)
const RIX = Dict(t => i for (i, t) in enumerate(RING))
const NRING = length(RING)
@printf("zigzag ring of %d tiles, spawn %s pose (%.3f, %.3f) angle %.1f deg\n",
        NRING, tile_of(w0), w0.ego.pos[1], w0.ego.pos[3],
        rad2deg(w0.ego.angle))

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

const s0 = LapState(w0, 0)

function plan_from(from::LapState)
    t0 = time()
    planner = solve(DORASolver(
            start = from, classify = classify,
            cost = (ls, a, lsp) -> macro_step(ls, a)[2],
            key = lapkey,
            c_min = K * C_MIN, c_to = 2000.0, c_crash = 1000.0, horizon = 200,
        ), LapMDP())
    @printf("  plan: %d states (%.1f s)\n", planner.tab.S, time() - t0)
    return planner
end

println("\ninitial plan...")
planner0 = plan_from(s0)
V, pistar = optimal_value(planner0.tab)
sr, cr, tr_ = outcome_rates(planner0.tab, pistar)
@printf("in-model: optimal cost %.2f   success %.3f   crash %.3f   timeout %.3f\n",
        V[planner0.tab.start], sr, cr, tr_)

# --- receding-horizon execution with exact substep recording ----------------
function replay_substeps!(subs, s, r, rng, prog, tile, dec)
    w = before_step(s, TR.duck_cfg, rng)
    wheels64 = (Float64(r.wheel_commands[1]), Float64(r.wheel_commands[2]))
    for _ in 1:TR.frame_skip
        w = ego_tick(w, wheels64)
        nt = tile_of(w)
        prog = advance(prog, tile, nt); tile = nt
        push!(subs, (w = w, prog = prog, dec = dec, v = w.ego.speed))
    end
    @assert w.ego.pos == r.sp.ego.pos && w.ego.angle == r.sp.ego.angle
    return prog, tile
end

function execute(; horizon = 200)
    ls = s0
    subs = NamedTuple[(w = w0, prog = 0, dec = 0, v = 0.0)]
    total = 0.0
    dec = 0
    plans = 0
    for _ in 1:horizon
        POMDPs.isterminal(BASE, ls.s) && return (:crash, total, subs, "isterminal", plans)
        ls.prog >= NRING && return (:lap, total, subs, "lap", plans)
        pl = plan_from(ls)
        plans += 1
        a = action(pl, ls)
        s = ls.s; prog = ls.prog; tile = tile_of(s)
        for _ in 1:K
            dec += 1
            rngc = copy(RNG)
            r = simulate_decision(TR, s, a, RNG)
            prog, tile = replay_substeps!(subs, s, r, rngc, prog, tile, dec)
            total += step_cost(r)
            s = r.sp
            prog >= NRING && return (:lap, total, subs, string(r.reason), plans)
            if r.terminated || r.truncated
                return (:crash, total, subs, string(r.reason), plans)
            end
        end
        ls = LapState(s, prog)
    end
    return (:timeout, total, subs, "horizon", plans)
end

println("\nexecuting (receding horizon)...")
outcome, total, subs, reason, plans = execute()
@printf("\nexecuted: %s after %d decisions (%d substeps), cost %.2f  (reason %s)\n",
        outcome, last(subs).dec, length(subs) - 1, total, reason)
@printf("ring progress: %d of %d   plans %d\n", last(subs).prog, NRING, plans)
@printf("substep replay matched simulate_decision exactly on all %d decisions\n",
        last(subs).dec)
@printf("first-plan cost %.2f  vs  executed %.2f\n", V[planner0.tab.start], total)

d_abs = [abs(raw_of(f.w).d) for f in subs]
@printf("lane keeping: |d| mean %.3f  max %.3f m\n", mean(d_abs), maximum(d_abs))

# --- export for the real renderer (poses only: no ducks, no signs) ----------
payload = Dict(
    "outcome" => string(outcome), "cost" => total,
    "cost_model" => V[planner0.tab.start],
    "map" => "zigzag_dists", "nring" => NRING,
    "states" => [Dict("i" => i, "dec" => f.dec, "progress" => f.prog,
                      "v" => f.v,
                      "pos" => collect(f.w.ego.pos),
                      "angle" => f.w.ego.angle)
                 for (i, f) in enumerate(subs)])
out = joinpath(@__DIR__, "zigzag_lap.json")
open(io -> JSON3.write(io, payload), out, "w")
@printf("\nwrote %s (%.1f MB)\n", basename(out), filesize(out) / 1e6)
