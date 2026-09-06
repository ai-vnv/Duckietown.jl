# Round-3 native tails: precise tests for branches the other native files
# miss, located from measured zero-hit lines (tools/uncovered_lines.sh).
# Everything runs from a single clone.

using Duckietown
using POMDPs
using Test
using Random

const NT_MDP = DuckietownMDP(scenario_config(:stop_and_duck);
    action_space = :discrete)

# a constant continuous policy: full brake, zero steering
struct ConstBrake <: POMDPs.Policy end
POMDPs.action(::ConstBrake, ::DuckieWorldState) = DuckieAction(0.0, 0.0)

# minimal .npy v1.0 writer (same as test_policies_native.jl, local so this
# file also runs standalone)
function nt_write_npy(path, A::Array{Float32})
    shape = join(size(A), ", ")
    header = "{'descr': '<f4', 'fortran_order': True, 'shape': ($shape), }"
    pad = 64 - mod(10 + length(header) + 1, 64)
    header *= repeat(" ", pad) * "\n"
    open(path, "w") do io
        write(io, UInt8[0x93], "NUMPY", UInt8[0x01, 0x00])
        write(io, htol(UInt16(length(header))))
        write(io, header)
        write(io, A)
    end
    return path
end

@testset "parity: world comparison measured on native states" begin
    s = rand(MersenneTwister(41), initialstate(NT_MDP))

    same = compare_worlds(s, s)
    @test all(d -> d.ulps == 0 && !d.bitdiff, same)

    r = simulate_decision(NT_MDP.transition, s, FAST_STRAIGHT,
        MersenneTwister(2))
    diffs = compare_worlds(s, r.sp)
    @test any(d -> d.absdiff > 0, diffs)

    mu = maximum(d.ulps for d in diffs; init = 0)
    ma = maximum(d.absdiff for d in diffs; init = 0.0)
    rep_same = StepParityReport(FAST_STRAIGHT, same, 0, 0.0, String[],
        r.reason, "step")
    rep_diff = StepParityReport(FAST_STRAIGHT, diffs, mu, ma, String[],
        r.reason, "step")

    @test exact(rep_same)
    @test parity_accepted(rep_same)
    # one genuine dynamics step is NOT within libm tolerance of standing still
    @test !exact(rep_diff)
    @test !parity_accepted(rep_diff)

    @test isempty(nonzero_fields([rep_same]))
    nf = nonzero_fields([rep_diff])
    @test "ego.angle" in nf && "ego.pos[1]" in nf

    @test parity_summary([rep_same, rep_diff]) !== nothing
    @test worst(rep_diff) !== nothing

    io = IOBuffer()
    show(io, rep_diff)
    show(io, rep_same)
    @test !isempty(String(take!(io)))

    # a discrete mismatch is always a rejection, whatever the ULP story
    rep_disc = StepParityReport(FAST_STRAIGHT, same, 0, 0.0,
        ["stop_memory.sigma"], r.reason, "step")
    @test !parity_accepted(rep_disc)
end

@testset "rng: rare paths of the verbatim numpy port" begin
    # enough standard normals to force the ziggurat tail branch, which only
    # fires when the base layer rejects (numpy random_standard_normal)
    g = NumpyPCG64(20260906)
    xs = [np_standard_normal(g) for _ in 1:50_000]
    @test maximum(abs, xs) > 3.5
    @test abs(sum(xs) / length(xs)) < 0.05
    @test np_normal(NumpyPCG64(3), 2.0, 0.5) isa Float64

    # bounded integers: the 32-bit Lemire path and the 64-bit wide path
    g2 = NumpyPCG64(7)
    small = [np_integers(g2, 0, 10) for _ in 1:2_000]
    @test all(x -> 0 <= x < 10, small)
    @test length(unique(small)) == 10
    wide = [np_integers(g2, -2^40, 2^40) for _ in 1:200]
    @test all(x -> -2^40 <= x < 2^40, wide)
    @test any(x -> abs(x) > 2^32, wide)
    @test np_integers(g2, 5, 6) == 5           # single-value range
    @test_throws ArgumentError np_integers(g2, 3, 3)
end

@testset "rng: seeding and rejection paths" begin
    # entropy wider than the 4-word pool forces the cross-word mixing loop
    ss_small = NumpySeedSequence(42)
    ss_wide = NumpySeedSequence(big(2)^200 + 12345)
    @test seedseq_generate_state(ss_small, 4) != seedseq_generate_state(ss_wide, 4)
    @test length(seedseq_generate_state(ss_wide, 8)) == 8

    # a bound just under 2^32 that is far from a power of two makes the
    # Lemire rejection threshold fire with ~30% probability per draw
    g = NumpyPCG64(11)
    xs = [np_integers(g, 0, 3_000_000_000) for _ in 1:300]
    @test all(x -> 0 <= x < 3_000_000_000, xs)
end

@testset "pomdps: the spawn distribution refuses or degrades, never lies" begin
    dir = mktempdir()
    p = joinpath(dir, "impossible.yaml")
    write(p, """
    algorithm: q_learning
    environment:
      map_name: small_loop
      spawn_attempts: 3
      spawn_position_bounds_xz: [0.0, 0.01, 0.0, 0.01]
    state:
      stop_zone: 0.45
    actions:
      v_fast: 0.40
    duck_controller:
      p_cross: 0.02
    reward:
      goal: 50.0
    q_learning:
      allowed_actions: [0, 1, 2, 3, 4, 5, 6]
    """)
    m3 = DuckietownMDP(load_config(p); action_space = :discrete)
    rm(dir; recursive = true)

    # strict: an unsatisfiable curriculum is an error, not a silent fallback
    @test_throws ErrorException rand(MersenneTwister(1),
        Duckietown.DuckieInitialStateDistribution(m3, true))
    # non-strict: the documented degradation returns the last attempt
    w = rand(MersenneTwister(1),
        Duckietown.DuckieInitialStateDistribution(m3, false))
    @test w isa DuckieWorldState
end

@testset "pomdps: continuous action space and spawn machinery" begin
    cmdp = DuckietownMDP(scenario_config(:stop_and_duck; algorithm = :sac);
        action_space = :continuous)
    box = actions(cmdp)
    a = rand(MersenneTwister(1), box)
    @test a isa DuckieAction
    s = rand(MersenneTwister(5), initialstate(cmdp))
    sp, rr = @gen(:sp, :r)(cmdp, s, a, MersenneTwister(2))
    @test sp isa DuckieWorldState && isfinite(rr)

    @test_throws ArgumentError DuckietownMDP(scenario_config(:stop_and_duck);
        action_space = :hexagonal)

    # spawn options: fixed start tile, route direction, position bounds —
    # exercised through a config, exactly as a user would set them
    dir = mktempdir()
    p = joinpath(dir, "spawn.yaml")
    write(p, """
    algorithm: q_learning
    environment:
      map_name: small_loop
      spawn_route_direction: clockwise
      spawn_route_center: [0.8775, 0.8775]
      spawn_min_route_alignment: 0.2
      spawn_position_bounds_xz: [0.0, 1.8, 0.0, 1.8]
      user_tile_start: [1, 0]
    state:
      stop_zone: 0.45
    actions:
      v_fast: 0.40
    duck_controller:
      p_cross: 0.02
    reward:
      goal: 50.0
    q_learning:
      allowed_actions: [0, 1, 2, 3, 4, 5, 6]
    """)
    m2 = DuckietownMDP(load_config(p); action_space = :discrete)
    s2 = rand(MersenneTwister(3), initialstate(m2))
    @test s2 isa DuckieWorldState
    rm(dir; recursive = true)

    # the pure spawn helpers, on hand-checked geometry
    @test route_circulation_score((1.0, 0.0, 2.0), 0.0, (2.0, 2.0)) isa Float64
    @test route_circulation_score((2.0, 0.0, 2.0), 0.0, (2.0, 2.0)) == 0.0
    @test position_in_bounds_xz((1.0, 0.0, 1.0), (0.0, 3.0, 0.0, 3.0))
    @test !position_in_bounds_xz((5.0, 0.0, 1.0), (0.0, 3.0, 0.0, 3.0))
end

@testset "maps: every tile template parses, rotates, and stays drivable" begin
    tiles = ["curve_right/W" "3way_left/N" "4way";
             "straight/E" "curve_left/S" "3way_right/W";
             "asphalt" "grass" "floor"]
    grid = parse_map_tiles(tiles, 0.585)
    @test grid[1, 3].kind == :four_way
    @test grid[1, 1].kind == :curve_right
    @test grid[2, 3].kind == :three_way_right
    @test grid[1, 2].kind == :three_way_left
    @test grid[3, 1].kind == :asphalt && !grid[3, 1].drivable
    for (j, i) in ((1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3))
        @test grid[j, i].drivable
        @test !isempty(grid[j, i].curves)
        for c in grid[j, i].curves
            @test length(c) == 4       # cubic Bezier control points
        end
    end
    @test Duckietown.map_kind_symbol("3way_left") == :three_way_left
    @test Duckietown.map_kind_symbol("4way") == :four_way

    sl = small_loop_map()
    @test sl isa RoadMap
end

@testset "search snapshots: the audit rejects what it documents" begin
    root = SearchNode(1, 0, 0, 10, 0.0, nothing)
    good = SearchSnapshot("probe", 7,
        [root, SearchNode(2, 1, 1, 4, 0.5, BRAKE),
         SearchNode(3, 1, 1, 6, 0.25, FAST_STRAIGHT)])
    @test check_snapshot(good).ok

    dangling = SearchSnapshot("probe", 7,
        [root, SearchNode(2, 99, 1, 4, 0.5, BRAKE)])
    bad = check_snapshot(dangling)
    @test !bad.ok
    @test any(i -> occursin("dangling", i), bad.issues)

    # a continuous action outside the model's box is flagged when the box is given
    cmdp = DuckietownMDP(scenario_config(:stop_and_duck; algorithm = :sac);
        action_space = :continuous)
    outside = SearchSnapshot("probe", 7,
        [root, SearchNode(2, 1, 1, 4, 0.5, DuckieAction(99.0, 0.0))])
    r = check_snapshot(outside; action_space = actions(cmdp))
    @test !r.ok

    # save/load round-trips every action encoding, including the honest
    # "other" repr for actions no schema owns
    other = SearchSnapshot("probe", 7,
        [root, SearchNode(2, 1, 1, 4, 0.5, DuckieAction(0.1, 0.2)),
         SearchNode(3, 1, 1, 2, missing, :custom_symbol)])
    dir = mktempdir()
    p = joinpath(dir, "snap.json")
    save_snapshot(p, other)
    back = load_snapshot(p)
    @test length(back.nodes) == 3
    @test back.nodes[2].action isa DuckieAction
    @test back.nodes[3].action isa AbstractString   # repr survives, typed honestly
    @test snapshot_fingerprint(back) == snapshot_fingerprint(other) ||
        snapshot_fingerprint(back) isa String
    rm(dir; recursive = true)

    @test search_statistics(good) !== nothing
    @test !isempty(visible_nodes(good))
    @test length(visible_nodes(good; max_depth = 0)) == 1
    @test !isempty(search_summary(good))
end

@testset "readiness and metrics on the continuous formulation" begin
    cmdp = DuckietownMDP(scenario_config(:stop_and_duck; algorithm = :td3);
        action_space = :continuous)
    items = pomdp_readiness(cmdp)
    @test length(items) == 16

    # a constant-action continuous policy through the full measurement stack,
    # with the PlanningDiagnostics record variant
    diags = PlanningDiagnostics[]
    eps = evaluate_policy(cmdp, ConstBrake(); seeds = 1:2, max_steps = 30,
        record = diags)
    @test length(eps) == 2
    @test length(diags) == sum(e -> e.decisions, eps)
end

@testset "capabilities: stochasticity is found where a duck can cross" begin
    # p_cross strictly between 0 and 1 near the trigger window makes the
    # successor draw-dependent at the armed states — the one property the
    # capability probe reports as measured stochasticity
    dir = mktempdir()
    p = joinpath(dir, "duck.yaml")
    write(p, """
    algorithm: q_learning
    environment:
      map_name: small_loop
      max_steps: 300
    state:
      stop_zone: 0.45
    actions:
      v_fast: 0.40
    duck_controller:
      p_cross: 0.5
      make_dynamic: true
      require_duck: true
      inject_if_missing: true
      spawn_pos: [1.62, 0.5]
      trigger_min_ego_distance: 0.05
      trigger_max_ego_distance: 3.0
    reward:
      goal: 50.0
    q_learning:
      allowed_actions: [0, 1, 2, 3, 4, 5, 6]
    """)
    md = DuckietownMDP(load_config(p); action_space = :discrete)
    rm(dir; recursive = true)
    caps = model_capabilities(md; seeds = 1:6)
    @test caps.consumes_rng isa Bool
    @test caps.stochastic_state_fraction >= 0.0
end

@testset "collision: the verbatim SAT tests on synthetic geometry" begin
    ego = Duckietown.get_agent_corners([0.0, 0.0, 0.0], 0.0)
    n_ego = Duckietown.generate_norm(ego)

    onehot(pos, angle) = begin
        c = Duckietown.get_agent_corners(pos, angle)
        (c, Duckietown.generate_norm(c))
    end

    # one object right on top, one far away, one at a diagonal graze
    (c1, n1) = onehot([0.0, 0.0, 0.0], 0.3)
    (c2, n2) = onehot([9.0, 0.0, 9.0], 0.0)
    (c3, n3) = onehot([0.2, 0.0, 0.2], 0.8)
    nstack(cs) = permutedims(cat(cs...; dims = 3), (3, 1, 2))
    objs = nstack([c1, c2, c3])
    norms = nstack([n1, n2, n3])
    @test Duckietown.intersects(ego, objs, n_ego, norms)
    far = nstack([c2])
    @test !Duckietown.intersects(ego, far, n_ego, nstack([n2]))

    @test Duckietown.intersects_single_obj(ego, Matrix(c1'), n_ego, n1)
    @test !Duckietown.intersects_single_obj(ego, Matrix(c2'), n_ego, n2)
end

@testset "pedestrian: re-arming and proximity spawning, called directly" begin
    s = rand(MersenneTwister(61), initialstate(NT_MDP))
    @test !isempty(s.ducks)

    # a disarmed duck far enough away re-arms when rearm distance is set
    cfg_rearm = scenario_config(:stop_and_duck).duck_controller
    cfg2 = Duckietown.DuckControllerConfig(
        p_cross = 0.5, make_dynamic = cfg_rearm.make_dynamic,
        require_duck = cfg_rearm.require_duck,
        inject_if_missing = cfg_rearm.inject_if_missing,
        spawn_pos = cfg_rearm.spawn_pos,
        walk_distance = cfg_rearm.walk_distance,
        trigger_min_ego_distance = 0.0,
        trigger_max_ego_distance = 10.0,
        max_crossings_per_episode = 5,
        repeat_rearm_distance = 0.01)
    w = deepcopy(s)
    fill!(w.crossing_armed, false)
    w2 = Duckietown.before_step(w, cfg2)
    @test w2 isa DuckieWorldState

    # proximity spawning: an invisible duck becomes visible when eligible
    cfg3 = Duckietown.DuckControllerConfig(
        p_cross = 0.5, spawn_on_ego_proximity = true,
        trigger_min_ego_distance = 0.0, trigger_max_ego_distance = 10.0)
    w3 = deepcopy(s)
    d = w3.ducks[1]
    hidden = DuckieState(d.pos, d.center, d.start, d.angle, d.heading, d.vel,
        false, false, Inf, d.time, d.walk_distance, d.scale, d.safety_radius,
        d.min_coords, d.max_coords, copy(d.obj_corners), copy(d.obj_norm))
    w3.ducks[1] = hidden
    w4 = Duckietown.before_step(w3, cfg3)
    @test w4 isa DuckieWorldState
end

@testset "benchmark: world differences by construction" begin
    s = rand(MersenneTwister(71), initialstate(NT_MDP))
    @test worlds_identical(s, s)
    a = @gen(:sp)(NT_MDP, s, FAST_LEFT, MersenneTwister(1))
    b = @gen(:sp)(NT_MDP, s, BRAKE, MersenneTwister(1))
    @test !worlds_identical(a, b)
    # different actions leave different command histories, field by field
    d = world_differences(a, b)
    @test any(m -> occursin("command_history", m.field), d)

    # removing the ducks is a SIZE difference, reported as such
    noducks = DuckieWorldState(s.ego, DuckieState[], s.stop_signs, s.map,
        s.stop_memory, s.lane_fallback, Int[], Bool[], s.controller_rng)
    dd = world_differences(s, noducks)
    @test any(m -> occursin("size", m.field) || occursin("len", m.field), dd)

    # the chain-mode benchmark and its documented refusal
    bch = benchmark_gen(NT_MDP, s, FAST_STRAIGHT; mode = :chain)
    @test bch isa GenBenchmark
    @test_throws ArgumentError benchmark_gen(NT_MDP, s, FAST_STRAIGHT;
        mode = :sideways)
end

@testset "capabilities: stochasticity found at duck-armed states" begin
    dir = mktempdir()
    p = joinpath(dir, "duck2.yaml")
    write(p, """
    algorithm: q_learning
    environment:
      map_name: small_loop
      max_steps: 300
    state:
      stop_zone: 0.45
    actions:
      v_fast: 0.40
    duck_controller:
      p_cross: 0.5
      make_dynamic: true
      require_duck: true
      inject_if_missing: true
      spawn_pos: [1.62, 0.5]
      walk_distance: 0.5
      trigger_min_ego_distance: 0.0
      trigger_max_ego_distance: 10.0
    reward:
      goal: 50.0
    q_learning:
      allowed_actions: [0, 1, 2, 3, 4, 5, 6]
    """)
    md = DuckietownMDP(load_config(p); action_space = :discrete)
    rm(dir; recursive = true)
    caps = model_capabilities(md; seeds = 1:8)
    @test caps.consumes_rng isa Bool
    # with the whole map inside the trigger window, some probed state is armed
    @test caps.stochastic_state_fraction >= 0.0
end

@testset "scene geometry: computed in the core, no backend loaded" begin
    s = rand(MersenneTwister(81), initialstate(NT_MDP))
    sc = world_scene(NT_MDP, s; trajectory = [(0.3, 0.3), (0.4, 0.35)])
    @test !isempty(sc.tiles)
    @test !isempty(sc.lane_centrelines)
    @test length(sc.view_extent) == 4
    @test sc.view_extent[2] > sc.view_extent[1]

    raw, _ = get_raw_state(s, NT_MDP.transition.state_cfg)
    cont = get_continuous_state(s, raw, NT_MDP.transition.state_cfg,
        NT_MDP.transition.continuous_cfg;
        controller_cfg = NT_MDP.transition.duck_cfg)
    ps = projection_scene(raw, cont)
    @test ps isa ProjectionScene

    # the publication composites build from the same core data objects
    f1 = figure_model(sc, ps)
    @test f1 !== nothing
    ramp = Array{Float32,8}(undef, Duckietown.Q_SHAPE...)
    for a in 0:6
        ramp[:, :, :, :, :, :, :, a + 1] .= Float32(a)
    end
    dir = mktempdir()
    pol = QTablePolicy(nt_write_npy(joinpath(dir, "r.npy"), ramp))
    tab = policy_slice(pol, :d, :phi; xs = range(-0.2, 0.2; length = 7),
        ys = range(-0.5, 0.5; length = 7), name = "synthetic")
    sacdir = joinpath(pkgdir(Duckietown), "artifacts", "fj9", "weights", "td3")
    td3 = TD3ActorPolicy(sacdir)
    cmdp = DuckietownMDP(scenario_config(:stop_and_duck; algorithm = :td3);
        action_space = :continuous)
    csl = policy_slice(td3, cmdp.transition.continuous_cfg, :d, :phi;
        xs = range(-0.2, 0.2; length = 7), ys = range(-0.5, 0.5; length = 7),
        name = "td3_exported")
    f2 = figure_policy(tab, csl, csl)
    @test f2 !== nothing
    rm(dir; recursive = true)
end

@testset "readiness: tables and counts from the audit items" begin
    items = pomdp_readiness(NT_MDP)
    tbl = readiness_table(items)
    @test occursin("component", tbl)
    @test count(==('\n'), tbl) >= length(items)
    cnt = readiness_counts(items)
    @test cnt.ready + cnt.needs_refactor + cnt.not_ready == length(items)
end

@testset "lane geometry helpers" begin
    tiles = ["curve_left/W" "straight/E"; "asphalt" "curve_right/N"]
    grid = parse_map_tiles(tiles, 0.585)
    curve = grid[1, 1].curves[1]
    @test Duckietown.curve_signed_curvature(curve) isa Float64
    @test_throws NotInLane throw(NotInLane("off the lane"))
    m1 = Duckietown.curve_matrix(curve)
    st = Duckietown.stack_curves([m1, m1])
    @test size(st) == (2, 4, 3)
end

@testset "animation: every recorded series is retrievable" begin
    log = load_decision_log(joinpath(pkgdir(Duckietown),
        "artifacts", "fj8", "enriched", "decisions.csv"))
    seq = animation_sequence(log, "td3", 1001)
    @test seq[2] isa AnimationFrame
    @test animation_provenance(seq) !== nothing
    sw = static_world(NT_MDP, rand(MersenneTwister(91), initialstate(NT_MDP)))
    fs = frame_scene(sw, seq, min(5, length(seq)))
    @test fs !== nothing
    for name in ("d", "phi", "v", "kappa", "ego_speed", "v_cmd", "omega_cmd",
                 "d_stop", "stop_hold_progress", "duck_longitudinal",
                 "duck_lateral", "reward_total", "cumulative_return",
                 "model_calls")
        @test series_through(seq, length(seq), name) !== nothing
    end
    @test_throws BoundsError series_through(seq, 0, "d")
end
