# FJ4: POMDPs.jl model interface.
#
# Scope is deliberately narrow: make the FJ3 generative core usable as a
# POMDPs.jl benchmark problem (gen / initialstate / isterminal / discount /
# actions), with the discrete and continuous action variants explicit. No
# solver work here.
#
# Parity leverage: no new fixture is needed. The reference objects and reset
# poses already recorded in the FJ3 fixtures are reused —
# - `fj3_duck.json::duck_init` is the reference duckie right after the real
#   `env.reset()`, so `initial_duckie` is checked field by field against it;
# - every `fj37_transition.json` scenario `init` is a real reference reset
#   pose, so our `spawn_accepted` predicate must accept all of them;
# - replaying an FJ3.7 scenario to its end pins `isterminal`/`is_truncated`
#   against the recorded termination reason.

using DuckietownDecisionModels
using POMDPs
using Test
using JSON3
using Random

const DUCKDUCK_POLICIES_FJ4 = joinpath(pkgdir(DuckietownDecisionModels),
    "..", "duckduck", "policies")

qlearning_cfg_path() =
    joinpath(DUCKDUCK_POLICIES_FJ4, "q_learning", "training_config.yaml")
sac_cfg_path() = joinpath(DUCKDUCK_POLICIES_FJ4, "sac", "training_config.yaml")

@testset "FJ4 model construction + POMDPs interface" begin
    mdp = DuckietownMDP(qlearning_cfg_path())
    @test mdp isa MDP{DuckieWorldState,MacroAction}
    @test statetype(mdp) == DuckieWorldState
    @test actiontype(mdp) == MacroAction
    @test discount(mdp) == mdp.config.solver.gamma == 0.99
    @test actions(mdp) == ALL_MACRO_ACTIONS
    @test length(actions(mdp)) == 7
    for (i, a) in enumerate(actions(mdp))
        @test actionindex(mdp, a) == i
    end
    # the transition model carries the YAML parameters, not source defaults
    @test mdp.transition.frame_skip == mdp.config.environment.frame_skip == 6
    @test mdp.transition.max_steps == mdp.config.environment.max_steps
    @test mdp.transition.reward_cfg === mdp.config.reward

    mdpc = DuckietownMDP(sac_cfg_path(); action_space=:continuous)
    @test mdpc isa MDP{DuckieWorldState,DuckieAction}
    @test actiontype(mdpc) == DuckieAction
    space = actions(mdpc)
    @test space isa DuckieActionSpace
    @test space.v_min == 0.0 && space.v_max == mdpc.config.actions.v_fast
    @test space.omega_min == -mdpc.config.actions.w0
    @test space.omega_max == mdpc.config.actions.w0
    rng = MersenneTwister(11)
    for _ in 1:200
        a = rand(rng, space)
        @test a isa DuckieAction && a in space
    end
    @test !(DuckieAction(-0.1, 0.0) in space)
    @test !(DuckieAction(0.1, 99.0) in space)
    @test_throws ArgumentError DuckietownMDP(qlearning_cfg_path();
        action_space=:bogus)
end

@testset "FJ4 injected objects match the reference reset" begin
    mdp = DuckietownMDP(qlearning_cfg_path())
    # the map's stop sign, derived from the tile-frame descriptor, must equal
    # the FJ3.1-verified hardcoded world anchor
    ref_map = small_loop_map()
    @test length(mdp.map.static_objects) == 1
    got = mdp.map.static_objects[1]
    want = ref_map.static_objects[1]
    @test got.kind == want.kind === :sign_stop
    @test got.pos == want.pos
    @test got.angle == want.angle
    @test got.corners == want.corners
    @test got.norm == want.norm
    @test got.safety_radius == want.safety_radius

    # object_world_pose is the verbatim tile->world rule
    @test object_world_pose(mdp.map, (1.20, 2.10), 180.0)[1] ==
        (0.702, 0.0, 1.8135)
    @test object_world_pose(mdp.map, (1.62, 0.50), 0.0)[1] ==
        (0.9477, 0.0, 0.8775)

    # the duckie, against the reference `duck_init` recorded after a real
    # `env.reset()` in the FJ3.4/3.5 fixture
    fx = JSON3.read(joinpath(pkgdir(DuckietownDecisionModels), "test",
        "fixtures", "fj3_duck.json"))
    d0 = fx.duck_init
    unf(x) = x isa AbstractDict ?
        (x["nonfinite"] == "nan" ? NaN : x["nonfinite"] == "inf" ? Inf :
         x["nonfinite"] == "-inf" ? -Inf : -0.0) : Float64(x)
    duck = initial_duckie(mdp.map, mdp.transition.duck_cfg)
    for k in 1:3
        @test duck.pos[k] == unf(d0.pos[k])
        @test duck.center[k] == unf(d0.center[k])
        @test duck.start[k] == unf(d0.start[k])
        @test duck.heading[k] == unf(d0.heading[k])
    end
    @test duck.angle == unf(d0.angle)
    @test duck.vel == unf(d0.vel) == 0.02
    @test duck.pedestrian_wait_time == unf(d0.wait) == 8.0
    @test duck.time == unf(d0.time) == 0.0
    @test duck.pedestrian_active == Bool(d0.active) == false
    @test duck.visible == Bool(d0.visible)
    for i in 1:4, j in 1:2
        @test duck.obj_corners[i][j] == unf(d0.corners[i][j])
    end
    for i in 1:2, j in 1:2
        @test duck.obj_norm[i, j] == unf(d0.norm[i][j])
    end
    # `duck_init` predates DuckController.__init__, which stamps
    # cfg.walk_distance over the simulator default (FJ3.5 finding)
    @test unf(d0.walk_distance) == mdp.map.tile_size == 0.585
    @test duck.walk_distance == mdp.transition.duck_cfg.walk_distance == 0.90
end

@testset "FJ4 initialstate distribution" begin
    mdp = DuckietownMDP(qlearning_cfg_path())
    d = initialstate(mdp)
    @test d isa DuckieInitialStateDistribution
    @test eltype(typeof(d)) == DuckieWorldState

    env = mdp.config.environment
    for seed in (11, 53, 73, 101)
        s = rand(MersenneTwister(seed), d)
        @test s isa DuckieWorldState
        # reproducible by seed
        s2 = rand(MersenneTwister(seed), d)
        @test s.ego.pos == s2.ego.pos && s.ego.angle == s2.ego.angle
        # a fresh episode: ego at rest, empty delay window, cleared memory
        @test s.ego.step_count == 0
        @test s.ego.timestamp == 0.0
        @test isempty(s.ego.command_history)
        @test s.ego.v_long == 0.0 && s.ego.omega == 0.0 && s.ego.speed == 0.0
        @test !s.stop_memory.sigma_stop && s.stop_memory.hold_steps == 0
        @test s.crossings_started == [0] && s.crossing_armed == [true]
        @test !s.ducks[1].pedestrian_active
        # the curriculum predicate actually holds for the sampled state
        raw, _ = get_raw_state(s, mdp.transition.state_cfg; sigma_stop=false)
        @test abs(raw.d) <= env.spawn_max_abs_d
        @test abs(raw.phi) <= env.spawn_max_abs_phi
        @test spawn_accepted(mdp, s, raw)
        # a fresh state is never terminal
        @test !isterminal(mdp, s)
        @test termination_reason(mdp.transition, s) == IN_PROGRESS
        # the reset stop memory equals the state's own first candidate
        dist, id = next_stop_candidate(s, mdp.transition.state_cfg)
        @test s.stop_memory.last_stop_id == id
        @test s.stop_memory.last_d_stop == dist
    end

    # the SAC config additionally constrains the route direction (clockwise)
    mdpc = DuckietownMDP(sac_cfg_path(); action_space=:continuous)
    @test mdpc.config.environment.spawn_route_direction === :clockwise
    for seed in (5, 17)
        s = rand(MersenneTwister(seed), initialstate(mdpc))
        center = (size(mdpc.map.grid, 2) * mdpc.map.tile_size / 2.0,
            size(mdpc.map.grid, 1) * mdpc.map.tile_size / 2.0)
        score = route_circulation_score(s.ego.pos, s.ego.angle, center)
        @test score >= mdpc.config.environment.spawn_min_route_alignment
    end

    # a NumpyPCG64 drives the same sampler through the reference draw stream
    s_np = rand(NumpyPCG64(53), initialstate(mdp))
    @test s_np isa DuckieWorldState
    @test rand(NumpyPCG64(53), initialstate(mdp)).ego.pos == s_np.ego.pos
end

@testset "FJ4 spawn_accepted agrees with the reference resets" begin
    # every scenario init in the FJ3.7 fixture is a pose the reference
    # wrapper's own `_spawn_is_accepted` accepted
    mdp = DuckietownMDP(qlearning_cfg_path())
    fx = JSON3.read(joinpath(pkgdir(DuckietownDecisionModels), "test",
        "fixtures", "fj37_transition.json"))
    unf(x) = x isa AbstractDict ? NaN : Float64(x)
    n = 0
    for s in fx.scenarios
        String(s.config) == "q_learning" || continue
        init = s.init
        pos = (unf(init.pos[1]), unf(init.pos[2]), unf(init.pos[3]))
        world = build_world(mdp, pos, unf(init.angle))
        raw, fallback = get_raw_state(world, mdp.transition.state_cfg;
            sigma_stop=false)
        world.lane_fallback = fallback
        @test spawn_accepted(mdp, world, raw)
        n += 1
    end
    @test n >= 10
end

@testset "FJ4 gen is a thin adapter over simulate_decision" begin
    mdp = DuckietownMDP(qlearning_cfg_path())
    s0 = rand(MersenneTwister(53), initialstate(mdp))

    for a in actions(mdp)
        ref = simulate_decision(mdp.transition, s0, a, MersenneTwister(7))
        x = gen(mdp, s0, a, MersenneTwister(7))
        @test keys(x) == (:sp, :r)
        @test x.r === ref.reward.total
        @test x.sp.ego.pos == ref.sp.ego.pos
        @test x.sp.ego.angle == ref.sp.ego.angle
        @test x.sp.ego.step_count == ref.sp.ego.step_count == mdp.transition.frame_skip
        @test x.sp.ducks[1].center == ref.sp.ducks[1].center
        @test x.sp.stop_memory.sigma_stop == ref.sp.stop_memory.sigma_stop
    end

    # branch purity through the POMDPs layer
    snap = deepcopy(s0)
    sp1 = gen(mdp, s0, FAST_LEFT, MersenneTwister(3)).sp
    sp2 = gen(mdp, s0, FAST_RIGHT, MersenneTwister(3)).sp
    @test s0.ego.pos == snap.ego.pos
    @test s0.ego.command_history == snap.ego.command_history
    @test s0.ducks[1].center == snap.ducks[1].center
    @test s0.crossings_started == snap.crossings_started
    @test sp1.ego.pos != sp2.ego.pos
    @test sp1.ego.command_history !== s0.ego.command_history
    @test sp1.ducks !== s0.ducks

    # continuous variant: same problem definition, different action type
    mdpc = DuckietownMDP(sac_cfg_path(); action_space=:continuous)
    s0c = rand(MersenneTwister(73), initialstate(mdpc))
    for a in (DuckieAction(0.0, 0.0), DuckieAction(0.17, 0.0),
        DuckieAction(0.41, 1.5), DuckieAction(0.25, -0.8))
        ref = simulate_decision(mdpc.transition, s0c, a, MersenneTwister(9))
        x = gen(mdpc, s0c, a, MersenneTwister(9))
        @test x.r === ref.reward.total
        @test x.sp.ego.pos == ref.sp.ego.pos
    end
end

@testset "FJ4 isterminal / truncation semantics" begin
    mdp = DuckietownMDP(qlearning_cfg_path())
    fx = JSON3.read(joinpath(pkgdir(DuckietownDecisionModels), "test",
        "fixtures", "fj37_transition.json"))
    unf(x) = x isa AbstractDict ? NaN : Float64(x)
    tuple3(v) = (unf(v[1]), unf(v[2]), unf(v[3]))

    reasons_seen = Set{String}()
    for s in fx.scenarios
        String(s.config) == "q_learning" || continue
        String(s.kind) == "discrete" || continue
        # baseline-config scenarios only: the variant scenarios move objects
        haskey(s, :meta) && continue
        init = s.init
        world = build_world(mdp, tuple3(init.pos), unf(init.angle))
        raw, fallback = get_raw_state(world, mdp.transition.state_cfg;
            sigma_stop=false)
        world.lane_fallback = fallback
        _, sid = next_stop_candidate(world, mdp.transition.state_cfg)
        world.stop_memory = StopMemory(false, 0, sid, raw.d_stop)
        rng = MersenneTwister(2024)
        final_reason = IN_PROGRESS
        for dd in s.decisions
            r = simulate_decision(mdp.transition, world, Int(dd.action), rng)
            world = r.sp
            final_reason = r.reason
            # the state-only reason equals the chain's reason
            @test termination_reason(mdp.transition, world) == r.reason
            # POMDPs.isterminal is the genuine-terminal half of it
            @test isterminal(mdp, world) == r.terminated
            @test is_truncated(mdp, world) == r.truncated
            r.terminated && break
        end
        push!(reasons_seen, lowercase(string(final_reason)))
    end
    # the baseline scenarios reach both an absorbing terminal and truncation
    @test "offroad" in reasons_seen
    @test "timeout" in reasons_seen

    # explicit invariant: a timeout is NOT absorbing (TD bootstrap preserved)
    s = rand(MersenneTwister(53), initialstate(mdp))
    e = s.ego
    timed_out = DuckieWorldState(
        DuckieEgoState(e.pos, e.angle, e.v_long, e.omega, e.speed,
            mdp.transition.max_steps, e.timestamp, copy(e.command_history),
            copy(e.q0), copy(e.v0), e.axis_left_rad, e.axis_right_rad),
        s.ducks, s.stop_signs, s.map, s.stop_memory, s.lane_fallback,
        s.crossings_started, s.crossing_armed, s.controller_rng)
    @test termination_reason(mdp.transition, timed_out) == TIMEOUT
    @test !isterminal(mdp, timed_out)
    @test is_truncated(mdp, timed_out)
end

@testset "FJ4 end-user rollout works like a benchmark MDP" begin
    mdp = DuckietownMDP(qlearning_cfg_path())
    rng = MersenneTwister(101)
    s = rand(rng, initialstate(mdp))
    total = 0.0
    discountf = 1.0
    steps = 0
    while !isterminal(mdp, s) && !is_truncated(mdp, s) && steps < 40
        a = rand(rng, actions(mdp))
        x = gen(mdp, s, a, rng)
        total += discountf * x.r
        discountf *= discount(mdp)
        s = x.sp
        steps += 1
    end
    @test steps > 0
    @test isfinite(total)
    @test s.ego.step_count == steps * mdp.transition.frame_skip
end
