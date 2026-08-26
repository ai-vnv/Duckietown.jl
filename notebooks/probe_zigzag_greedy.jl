# Is the zigzag lane even FOLLOWABLE with the 5 discrete actions?
#
# The map curves are bit-identical to the reference (checked), and the macro
# length is not the cause (K = 8/6/4 all say crash 1.000). So test the layer
# below the planner: a 1-step greedy lane keeper — at every decision, branch
# all 5 actions from the true state (the transition is branch-pure) and take
# whichever minimises |d| + 0.3|phi| next step. Purely diagnostic.
#
#   greedy completes the lap  -> the lane is followable; the fault is in the
#                                planning/aggregation layer
#   greedy dies somewhere     -> the fault is in dynamics/observer/actions,
#                                localised at the tile where it dies

using DuckietownDecisionModels
using POMDPs, Random, Printf

const CFG = scenario_config(:lane_following)
const BASE = DuckietownMDP(CFG; action_space = :discrete)
const TR = BASE.transition
const SCFG = TR.state_cfg
const ACTS = collect(POMDPs.actions(BASE))
const RNG = MersenneTwister(1)

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

println("actions and their wheel commands:")
for a in ACTS
    wl = action_to_wheels(Int(a), TR.action_cfg)
    @printf("  %-14s wheels (%.3f, %.3f)\n", string(a), wl[1], wl[2])
end

const w0 = spawn_zigzag(MersenneTwister(2001))
r0 = raw_of(w0)
@printf("\nspawn tile %s pose (%.3f, %.3f) angle %.1f  d %+.3f phi %+.3f\n",
        tile_of(w0), w0.ego.pos[1], w0.ego.pos[3], rad2deg(w0.ego.angle),
        r0.d, r0.phi)

function greedy(; horizon = 800)
    s = w0
    visited = [tile_of(s)]
    for t in 1:horizon
        best = nothing
        for a in ACTS
            r = simulate_decision(TR, s, a, RNG)
            (r.terminated || r.truncated) && continue
            score = abs(r.raw_state.d) + 0.3 * abs(r.raw_state.phi)
            if best === nothing || score < best[1]
                best = (score, a, r)
            end
        end
        raw = raw_of(s)
        if best === nothing
            @printf("DEAD at decision %d: every action terminates\n", t)
            @printf("  tile %s  pose (%.3f, %.3f) angle %.1f  d %+.3f phi %+.3f v %.3f\n",
                    tile_of(s), s.ego.pos[1], s.ego.pos[3],
                    rad2deg(s.ego.angle), raw.d, raw.phi, raw.v)
            for a in ACTS
                r = simulate_decision(TR, s, a, RNG)
                @printf("    %-14s -> %s\n", string(a), string(r.reason))
            end
            return visited
        end
        s = best[3].sp
        nt = tile_of(s)
        nt != last(visited) && push!(visited, nt)
        if t % 100 == 0
            @printf("  t %3d  tile %s  d %+.3f  tiles visited %d\n",
                    t, nt, best[3].raw_state.d, length(visited))
        end
        length(visited) > 27 && break
    end
    return visited
end

visited = greedy()
println("\ntiles visited in order (", length(visited), "):")
println(visited)
