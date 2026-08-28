# Feasibility: can the simulator actually be driven on a branching map?
#
# Building the map is not enough. Spawning, lane projection and the transition
# chain were only ever validated on small_loop, so before designing anything
# around a junction map this has to be checked rather than assumed.

using DuckietownDecisionModels
using POMDPs, Random, Printf, Statistics

const TILES = ["curve_left/W" "straight/W"  "3way_left/W" "straight/W"  "curve_left/N";
               "straight/S"   "asphalt"     "straight/N"  "asphalt"     "straight/N";
               "curve_left/S" "straight/E"  "3way_left/E" "straight/E"  "curve_left/E"]

cfg = scenario_config(:stop_and_duck)
branch_map = RoadMap("double_loop", 0.585, parse_map_tiles(TILES, 0.585),
                     MapObjectData[])

mdp = DuckietownMDP(cfg; action_space = :discrete, map = branch_map)
ACTS = collect(POMDPs.actions(mdp))
SCFG = mdp.transition.state_cfg

println("map: ", branch_map.name, "  drivable tiles: ",
        length(drivable_tiles(branch_map)))

# 1. does the initial-state distribution work at all?
# Accumulate inside a function: a bare top-level loop makes the counter a
# fresh local. This is the trap recorded in CONTRIBUTING, and it was hit again
# here — with a `try` around the body it also disguised itself as a spawn
# failure, which is worse than crashing.
function try_spawn(m, n)
    ok = 0
    err = ""
    for sd in 1:n
        try
            rand(MersenneTwister(sd), initialstate(m))
            ok += 1
        catch e
            isempty(err) && (err = first(sprint(showerror, e), 160))
        end
    end
    return ok, err
end

ok_spawn, spawn_err = try_spawn(mdp, 40)
@printf("spawned %d / 40 seeds\n", ok_spawn)
isempty(spawn_err) || println("first error: ", spawn_err)
ok_spawn == 0 && exit(1)

# 2. is the lane projection sane at spawn? (|d| small, kappa finite)
ds = Float64[]; phis = Float64[]
for sd in 1:40
    s = rand(MersenneTwister(sd), initialstate(mdp))
    raw, _ = get_raw_state(s, SCFG)
    push!(ds, raw.d); push!(phis, raw.phi)
end
@printf("spawn |d| max %.4f   |phi| max %.4f   (bounds asked for 0.08 / 0.175)\n",
        maximum(abs, ds), maximum(abs, phis))

# 3. can a lane follower survive? compare with small_loop.
function follower(s, rng, cfgstate)
    raw, _ = get_raw_state(s, cfgstate)
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
    return a
end

function survival(m, scfg; episodes = 120, horizon = 250)
    rng = MersenneTwister(7)
    lens = Int[]; tiles = Int[]
    for _ in 1:episodes
        s = rand(rng, initialstate(m)); seen = Set{NTuple{2,Int}}()
        n = 0
        for t in 1:horizon
            push!(seen, get_grid_coords(s.map, collect(s.ego.pos)))
            res = simulate_decision(m.transition, s, follower(s, rng, scfg), rng)
            n = t
            (res.terminated || res.truncated) && break
            s = res.sp
        end
        push!(lens, n); push!(tiles, length(seen))
    end
    return lens, tiles
end

ref = DuckietownMDP(cfg; action_space = :discrete)
lr, tr = survival(ref, ref.transition.state_cfg)
lb, tb = survival(mdp, SCFG)
@printf("\n%-14s median episode %5.1f decisions   distinct tiles visited %.1f\n",
        "small_loop", median(lr), mean(tr))
@printf("%-14s median episode %5.1f decisions   distinct tiles visited %.1f\n",
        "double_loop", median(lb), mean(tb))

# 4. The follower visits 1.5 tiles before dying, so there is no route-level
# problem for it to have. Does the TRAINED controller traverse tiles?
qpath = joinpath(pkgdir(DuckietownDecisionModels), "..", "duckduck",
                 "policies", "q_learning", "policy.npy")
if !isfile(qpath)
    println("\nno trained checkpoint; cannot test the controller")
else
    qpol = QTablePolicy(qpath; solver = :q_learning)
    qact(s, scfg) = decide(qpol,
        discretize(first(get_raw_state(s, scfg)))).action

    function survival_q(m, scfg; episodes = 120, horizon = 250)
        rng = MersenneTwister(7)
        lens = Int[]; tiles = Int[]; crashed = 0
        for _ in 1:episodes
            s = rand(rng, initialstate(m)); seen = Set{NTuple{2,Int}}(); n = 0
            for t in 1:horizon
                push!(seen, get_grid_coords(s.map, collect(s.ego.pos)))
                res = simulate_decision(m.transition, s, qact(s, scfg), rng)
                n = t
                if res.terminated || res.truncated
                    crashed += 1; break
                end
                s = res.sp
            end
            push!(lens, n); push!(tiles, length(seen))
        end
        return lens, tiles, crashed
    end

    lqr, tqr, cqr = survival_q(ref, ref.transition.state_cfg)
    lqb, tqb, cqb = survival_q(mdp, SCFG)
    println("\ntrained q_learning controller (horizon 250):")
    @printf("%-14s median episode %5.1f   tiles visited %.1f / %d   ended early %d/120\n",
            "small_loop", median(lqr), mean(tqr),
            length(drivable_tiles(ref.map)), cqr)
    @printf("%-14s median episode %5.1f   tiles visited %.1f / %d   ended early %d/120\n",
            "double_loop", median(lqb), mean(tqb),
            length(drivable_tiles(branch_map)), cqb)
end
