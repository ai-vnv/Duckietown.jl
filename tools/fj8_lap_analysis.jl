# "Which solver can complete small_loop?" — measured, not inferred.
#
# The FJ8.4b table cannot answer this. Two reasons:
#
#   1. `goal_tile: null` in all four shipped configs, so the GOAL terminal can
#      never fire. "Completing the loop" is not a terminal condition of the
#      baseline task at all — it has to be defined and measured separately.
#   2. The FJ8.4b horizon of 150 decisions is 30 s of simulated time. At the
#      fastest observed speed (~0.14 m/s) that is ~4.2 m, just under one lap of
#      the ~4.7 m loop. The horizon was too short for a lap by construction.
#
# So: define completion as the winding number of the ego around the loop
# centre reaching 2*pi, and run each solver to the ENVIRONMENT's own step limit
# rather than to the comparison horizon.
#
#     tools/run_lap_analysis.sh

using DuckietownDecisionModels
using POMDPs
using Random
using MCTS

const ROOT = pkgdir(DuckietownDecisionModels)
const DUCK = joinpath(ROOT, "..", "duckduck")
const DT_DECISION = 6 * EGO_DT          # frame_skip x physics dt = 0.2 s

qcfg = joinpath(DUCK, "policies", "q_learning", "training_config.yaml")
ccfg = joinpath(DUCK, "policies", "sac", "training_config.yaml")
tcfg = joinpath(DUCK, "policies", "td3", "training_config.yaml")

"""Loop geometry: the centre the ego winds around, and the lane-centreline
length of one lap, both taken from the map rather than assumed."""
function loop_geometry(mdp)
    m = mdp.map
    h, w = size(m.grid)
    centre = (w * m.tile_size / 2, h * m.tile_size / 2)
    # Lane-centreline lap length: each drivable tile carries one lane curve per
    # travel direction, so the per-tile contribution is the mean over that
    # tile's curves — the two directions differ on a corner tile.
    arclen(c) = begin
        s = 0.0
        prev = bezier_point(c, 0.0)
        for t in range(0.02, 1.0; length=60)
            p = bezier_point(c, t)
            s += sqrt(sum(abs2, p .- prev))
            prev = p
        end
        s
    end
    total = 0.0
    for (i, j) in drivable_tiles(m)
        tile = _get_tile(m, i, j)
        (tile === nothing || isempty(tile.curves)) && continue
        total += sum(arclen(curve_matrix(c)) for c in tile.curves) /
                 length(tile.curves)
    end
    return (centre=centre, grid=(h, w), tile_size=m.tile_size,
        drivable=length(drivable_tiles(m)), lap_length=total,
        tile_centre_ring=8 * m.tile_size)
end

"""One episode's lap trace."""
function lap_episode(mdp, policy, seed, max_decisions, centre)
    s = rand(MersenneTwister(seed), initialstate(mdp))
    rng = MersenneTwister(seed)
    prev = (s.ego.pos[1], s.ego.pos[3])
    theta = atan(prev[2] - centre[2], prev[1] - centre[1])
    winding = 0.0
    path = 0.0
    tiles = Set{Tuple{Int,Int}}()
    reason = "horizon"
    k = 0
    while k < max_decisions
        a = policy_action(policy, mdp, s)
        r = simulate_decision(mdp.transition, s, a, rng)
        s = r.sp
        k += 1
        p = (s.ego.pos[1], s.ego.pos[3])
        path += sqrt((p[1] - prev[1])^2 + (p[2] - prev[2])^2)
        th = atan(p[2] - centre[2], p[1] - centre[1])
        d = th - theta
        d > pi && (d -= 2pi)
        d < -pi && (d += 2pi)
        winding += d
        theta = th
        prev = p
        push!(tiles, get_grid_coords(mdp.map, collect(s.ego.pos)))
        if r.terminated || r.truncated
            reason = lowercase(string(r.reason))
            break
        end
    end
    return (seed=seed, decisions=k, path=path, laps=abs(winding) / 2pi,
        tiles=length(tiles), reason=reason, sim_seconds=k * DT_DECISION)
end

r2(x) = lpad(round(x; digits=2), 6)

function report(name, mdp, policy, seeds, max_decisions, geo)
    eps = [lap_episode(mdp, policy, sd, max_decisions, geo.centre) for sd in seeds]
    laps = [e.laps for e in eps]
    n = length(eps)
    crashed = count(e -> e.reason in ("offroad", "other_collision",
        "duck_collision"), eps)
    println(rpad(name, 12), lpad(n, 4), " seeds",
        "   laps mean ", r2(sum(laps) / n), "  max ", r2(maximum(laps)),
        "   >=1 lap ", lpad(count(>=(1.0), laps), 2), "/", lpad(n, 2),
        "   path ", r2(sum(e -> e.path, eps) / n), " m",
        "   ", lpad(round(sum(e -> e.sim_seconds, eps) / n; digits=1), 6), " s",
        "   tiles ", r2(sum(e -> e.tiles, eps) / n),
        "   crashed ", lpad(crashed, 2),
        "   ", join(sort(unique(e.reason for e in eps)), "/"))
    return eps
end

seeds_cfg = planning_seed_config()
EVAL = seeds_cfg.evaluation.episodes

dmdp = DuckietownMDP(qcfg; action_space=:discrete)
geo = loop_geometry(dmdp)
tab_limit = dmdp.transition.max_steps ÷ 6

println("small_loop: ", geo.grid[1], "x", geo.grid[2], " tiles of ",
    geo.tile_size, " m, ", geo.drivable, " drivable tiles, loop centre ",
    round.(geo.centre; digits=4))
println("lap length: lane centreline ", round(geo.lap_length; digits=3),
    " m, tile-centre ring ", round(geo.tile_centre_ring; digits=3), " m")
println("environment step limit (tabular config): ",
    dmdp.transition.max_steps, " physics ticks = ", tab_limit, " decisions")
println("one lap at the fastest observed speed (0.14 m/s): ",
    round(geo.lap_length / 0.14; digits=1), " s = ",
    round(Int, geo.lap_length / 0.14 / DT_DECISION), " decisions")
println("the FJ8.4b comparison horizon was 150 decisions = ",
    round(150 * DT_DECISION; digits=1), " s\n")

println("== learned policies, run to the environment's own limit ==")
for (name, dir) in ("q_learning" => "q_learning", "sarsa" => "sarsa")
    f = joinpath(DUCK, "policies", dir, "policy.npy")
    isfile(f) || continue
    report(name, dmdp, QTablePolicy(f; solver=Symbol(dir)), EVAL, tab_limit, geo)
end

if torch_policy_available()
    b = TorchPolicyReferenceBackend()
    try
        for (name, cfgp, ctor) in (("sac", ccfg, SACActorPolicy),
            ("td3", tcfg, TD3ActorPolicy))
            meta = torch_policy_init!(b, name)
            m = DuckietownMDP(cfgp; action_space=:continuous)
            report(name, m, ctor(String(meta["weights_dir"])), EVAL, 600, geo)
        end
    finally
        close(b)
    end
else
    println("(ddm-torch unavailable: SAC and TD3 skipped)")
end

println("\n== online planners (5 seeds: ~0.1 s of wall time per decision) ==")
plan_seeds = EVAL[1:5]
mcts_mdp = DuckietownMDP(qcfg; action_space=:discrete)
dpw_mdp = DuckietownMDP(ccfg; action_space=:continuous)
report("mcts@1k", mcts_mdp,
    solve(MCTSSolver(n_iterations=36, depth=10, exploration_constant=5.0,
        rng=MersenneTwister(2026), reuse_tree=false), mcts_mdp),
    plan_seeds, tab_limit, geo)
report("dpw@1k", dpw_mdp,
    solve(DPWSolver(n_iterations=35, depth=10, exploration_constant=5.0,
        enable_action_pw=true, enable_state_pw=false, k_action=4.0,
        alpha_action=0.5, rng=MersenneTwister(2026), keep_tree=false), dpw_mdp),
    plan_seeds, tab_limit, geo)

println("\nLAP_ANALYSIS_OK=true")
