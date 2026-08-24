# FJ3.4/FJ3.5: duckie walk dynamics and the crossing-controller trigger pass,
# parity against test/fixtures/fj3_duck.json (reference run in ddm-ref).
#
# Part A pins `DuckieObj.step` (objects.py) tick by tick for 600 ticks in two
# standalone scenarios: auto-activation after the 8 s wait, and an active duck
# walking a full cycle (forward 0.9 m, finish_walk, 8 s wait, walk back).
#
# Part B replays a 137-decision rollout against the reference DuckController:
# `before_step` observes the previous decision's ego pose and activates the
# crossing exactly when the ego enters the trigger window, then the duck walks
# across the road, finishes, and waits — while `ego_tick` chain advances.
# Fields compared per decision: ego pos/angle/step_count, crossings_started,
# crossing_armed, and the full duck state.

using DuckietownDecisionModels
using Test
using JSON3
using Random

const FIXTURES_DUCK = joinpath(pkgdir(DuckietownDecisionModels),
    "test", "fixtures", "fj3_duck.json")

fixtures_duck = JSON3.read(FIXTURES_DUCK)

function unf_duck(x)
    if x isa AbstractDict
        tag = x["nonfinite"]
        return tag == "nan" ? NaN :
            tag == "inf" ? Inf :
            tag == "-inf" ? -Inf : -0.0
    end
    return Float64(x)
end

ulps_duck(a::Float64, b::Float64) =
    abs(reinterpret(Int64, a) - reinterpret(Int64, b))

sc_close_duck(a, b; n=2) = a == b || ulps_duck(unf_duck(a), unf_duck(b)) <= n

mat_close_duck(a, b) = all(sc_close_duck(x, y) for (x, y) in zip(a, b))

function duck_from_json(d; visible=true, walk_distance=nothing)
    corners = [(unf_duck(d.corners[i][1]), unf_duck(d.corners[i][2]))
        for i in 1:4]
    norm_m = Float64[unf_duck(d.norm[i][j]) for i in 1:2, j in 1:2]
    wd = walk_distance === nothing ? unf_duck(d.walk_distance) :
        Float64(walk_distance)
    return DuckieState(
        (unf_duck(d.pos[1]), unf_duck(d.pos[2]), unf_duck(d.pos[3])),
        (unf_duck(d.center[1]), unf_duck(d.center[2]), unf_duck(d.center[3])),
        (unf_duck(d.start[1]), unf_duck(d.start[2]), unf_duck(d.start[3])),
        unf_duck(d.angle),
        (unf_duck(d.heading[1]), unf_duck(d.heading[2]), unf_duck(d.heading[3])),
        unf_duck(d.vel), visible, Bool(d.active), unf_duck(d.wait),
        unf_duck(d.time), wd, 1.0, 0.1,
        (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), corners, norm_m)
end

function duck_close(d::DuckieState, f; n=2)
    ok = sc_close_duck(d.pos[1], f.pos[1]; n) && sc_close_duck(d.pos[2], f.pos[2]; n) &&
        sc_close_duck(d.pos[3], f.pos[3]; n) &&
        sc_close_duck(d.center[1], f.center[1]; n) &&
        sc_close_duck(d.center[2], f.center[2]; n) &&
        sc_close_duck(d.center[3], f.center[3]; n) &&
        sc_close_duck(d.start[1], f.start[1]; n) && sc_close_duck(d.start[2], f.start[2]; n) &&
        sc_close_duck(d.start[3], f.start[3]; n) &&
        sc_close_duck(d.angle, f.angle; n) &&
        sc_close_duck(d.heading[1], f.heading[1]; n) &&
        sc_close_duck(d.heading[2], f.heading[2]; n) &&
        sc_close_duck(d.heading[3], f.heading[3]; n) &&
        sc_close_duck(d.vel, f.vel; n) &&
        d.pedestrian_active == Bool(f.active) &&
        sc_close_duck(d.pedestrian_wait_time, f.wait; n) &&
        sc_close_duck(d.time, f.time; n) &&
        sc_close_duck(d.walk_distance, f.walk_distance; n) &&
        d.visible == Bool(f.visible) &&
        mat_close_duck([c for c in d.obj_corners],
            [(unf_duck(f.corners[i][1]), unf_duck(f.corners[i][2]))
                for i in 1:4]) &&
        mat_close_duck(mat_rows(d.obj_norm),
            [unf_duck(f.norm[i][j]) for i in 1:2 for j in 1:2])
    return ok
end

function mat_rows(m::AbstractMatrix)
    return [m[i, j] for i in 1:size(m, 1) for j in 1:size(m, 2)]
end

function fresh_duck_world(; ego_at_rollout_spawn::Bool=false)
    d0 = fixtures_duck.duck_init
    # duck_init was recorded BEFORE DuckController construction, so it still
    # carries the map-default walk_distance (0.585); the rollout runs after
    # `DuckController.__init__` stamped cfg.walk_distance (0.90) onto the
    # duck. The walk scenarios (Part A) use the map default as recorded.
    pos, ang, wd = if ego_at_rollout_spawn
        # the rollout records the reset ego pose as decision 0's pos_pre
        r0 = fixtures_duck.rollout[1]
        ((unf_duck(r0.pos_pre[1]), unf_duck(r0.pos_pre[2]),
            unf_duck(r0.pos_pre[3])), unf_duck(r0.angle_pre),
            unf_duck(r0.duck.walk_distance))
    else
        ((unf_duck(d0.pos[1]), 0.0, unf_duck(d0.pos[3])), unf_duck(d0.angle),
            nothing)
    end
    return DuckieWorldState(
        initial_ego(pos, ang, 3, 0.585),
        [duck_from_json(d0; walk_distance=wd)], StopSignState[],
        small_loop_map(),
        StopMemory(false, 0, nothing, nothing), (1.0, 1.0), Int[0], Bool[true],
        MersenneTwister(53))
end

duck_cfg() = DuckControllerConfig(p_cross=1.0, make_dynamic=true,
    require_duck=true, inject_if_missing=false, spawn_pos=(1.62, 0.50),
    spawn_rotate=0.0, spawn_height=0.08, walk_distance=0.90,
    trigger_min_ego_distance=0.35, trigger_max_ego_distance=0.45,
    spawn_on_ego_proximity=false, max_crossings_per_episode=1,
    repeat_rearm_distance=0.0, inject_stop_if_missing=false,
    require_stop=false)

@testset "FJ3.4 duck walk parity (standalone DuckieObj.step)" begin
    for s in fixtures_duck.walk_scenarios
        world = fresh_duck_world()
        if s.name == "active"
            d0 = world.ducks[1]
            world = replace_duck(world, 1,
                DuckieState(d0.pos, d0.center, d0.start, d0.angle, d0.heading,
                    d0.vel, d0.visible, true, 0.0, 0.0, d0.walk_distance,
                    d0.scale, d0.safety_radius, d0.min_coords, d0.max_coords,
                    d0.obj_corners, d0.obj_norm))
        end
        for (i, f) in enumerate(s.ticks)
            world = duck_step(world, 1)
            @test duck_close(world.ducks[1], f)
            @test sc_close_duck(world.ducks[1].time, f.time; n=1)
            # fixture-consistency invariant: the center may only change on a
            # tick where the duck walked, i.e. it was active going in (f) or
            # became active/finished during it (next tick)
            @test i == 600 || Bool(f.active) || Bool(s.ticks[i + 1].active) ||
                sc_close_duck(world.ducks[1].center[1],
                    unf_duck(s.ticks[i + 1].center[1]); n=1)
        end
    end
end

@testset "FJ3.5 before_step trigger + FJ3.4 tick chain parity (rollout)" begin
    world = fresh_duck_world(ego_at_rollout_spawn=true)
    cfg = duck_cfg()
    activated = false
    for f in fixtures_duck.rollout
        # before_step observes the previous decision's ego pose
        @test sc_close_duck(world.ego.pos[1], f.pos_pre[1])
        @test sc_close_duck(world.ego.pos[2], f.pos_pre[2])
        @test sc_close_duck(world.ego.pos[3], f.pos_pre[3])
        @test sc_close_duck(world.ego.angle, f.angle_pre)
        world = before_step(world, cfg)
        @test world.crossings_started == [Int(f.crossings_started[1])]
        @test world.crossing_armed == [Bool(f.crossing_armed[1])]
        # ticks: ego + duck, frame_skip = 6
        wheels = (unf_duck(f.wheels[1]), unf_duck(f.wheels[2]))
        for _ in 1:6
            world = ego_tick(world, wheels)
            world = duck_step(world, 1)
        end
        @test sc_close_duck(world.ego.pos[1], f.pos[1])
        @test sc_close_duck(world.ego.pos[2], f.pos[2])
        @test sc_close_duck(world.ego.pos[3], f.pos[3])
        @test sc_close_duck(world.ego.angle, f.angle)
        @test world.ego.step_count == Int(f.step_count)
        @test duck_close(world.ducks[1], f.duck)
        if !activated && Bool(f.duck.active)
            activated = true
            # activation decision: exactly one crossing started, ego inside
            # the trigger window, crossing ahead of the ego
            @test world.crossings_started == [1]
            @test world.ducks[1].pedestrian_wait_time == Inf
        end
    end
    @test activated
    @test world.crossings_started == [1]
    # the duck walked across the road and finished before the record ended
    @test !world.ducks[1].pedestrian_active
    @test world.ducks[1].vel < 0.0
end

@testset "FJ3.5 before_step edge cases" begin
    world = fresh_duck_world()
    cfg = duck_cfg()
    # wait pinned to Inf for the inactive duck
    w1 = before_step(world, cfg)
    @test w1.ducks[1].pedestrian_wait_time == Inf
    @test w1.ducks[1].pedestrian_active == false
    @test w1.crossings_started == [0]
    # ego far from the crossing: no activation
    far = DuckieWorldState(initial_ego((0.1, 0.0, 0.1), 1.0, 3, 0.585),
        [duck_from_json(fixtures_duck.duck_init)], StopSignState[],
        small_loop_map(), StopMemory(false, 0, nothing, nothing), (1.0, 1.0),
        Int[0], Bool[true], MersenneTwister(53))
    w2 = before_step(far, cfg)
    @test w2.crossings_started == [0]
    @test !w2.ducks[1].pedestrian_active
    # over-limit duck never re-activates
    over = DuckieWorldState(far.ego, [duck_from_json(fixtures_duck.duck_init)],
        StopSignState[], far.map, StopMemory(false, 0, nothing, nothing),
        (1.0, 1.0), Int[1], Bool[true], MersenneTwister(53))
    w3 = before_step(over, cfg)
    @test w3.crossings_started == [1]
    @test !w3.ducks[1].pedestrian_active
    # RNG draws do not leak into the caller's world state
    rng_before = deepcopy(world.controller_rng)
    for _ in 1:3
        world = before_step(world, cfg)
    end
    @test world.controller_rng.state == rng_before.state
end