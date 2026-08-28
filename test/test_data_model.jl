using DuckietownDecisionModels
using Test
using Random: MersenneTwister

@testset "enums match Python integer values" begin
    @test Int(STRAIGHT) == 0
    @test Int(CURVE_LEFT) == 1
    @test Int(CURVE_RIGHT) == 2
    @test Int(NONE) == 0
    @test Int(SIDE_FAR) == 1
    @test Int(SIDE_NEAR) == 2
    @test Int(CROSSING_FAR) == 3
    @test Int(CROSSING_NEAR) == 4
end

@testset "OBSERVATION_NAMES is 15 features, hold progress last" begin
    @test length(OBSERVATION_NAMES) == 15
    @test OBSERVATION_NAMES[end] == "stop_hold_progress"
end

@testset "RawState / ContinuousState construction" begin
    raw = RawState(0.02, 0.05, 0.30, STRAIGHT, 0.8, false, NONE)
    @test raw.d == 0.02
    @test raw.d_stop == 0.8
    @test raw.sigma_stop == false
    raw_nostop = RawState(-0.1, -0.2, 0.17, CURVE_LEFT, nothing, true, CROSSING_NEAR)
    @test raw_nostop.d_stop === nothing

    cont = ContinuousState(0.02, 0.05, 0.30, 0.0, true, 0.8, false,
        true, 0.5, 0.1, -0.3, 0.02, true, true)
    @test cont.stop_hold_progress == 0.0
    @test cont.d_stop == 0.8
end

@testset "StopTracker state semantics" begin
    t = StopTracker()
    @test t.zone == 0.45
    @test t.speed == 0.02
    @test t.pass_distance == 0.55
    @test t.hold_steps_required == 1
    @test t.hold_steps == 0
    @test !t.sigma_stop
    @test hold_progress(t) == 0.0

    t2 = StopTracker(0.45, 0.02, 0.30, 1)
    @test t2.pass_distance == 0.45
    t3 = StopTracker(0.45, 0.02, 0.55, 0)
    @test t3.hold_steps_required == 1
    t4 = StopTracker(0.45, 0.02, 0.55, 3)
    @test t4.hold_steps_required == 3
    @test hold_progress(t4) == 0.0

    r = reset_tracker(t4)
    @test r.hold_steps == 0
    @test !r.sigma_stop
    @test r.hold_steps_required == 3
end

@testset "DuckieWorldState is branchable" begin
    tile = TileSpec(:straight, 0.0, true, Vector{Matrix{Float64}}())
    map = RoadMap("synthetic", 0.585, fill(tile, 3, 3))
    duck = DuckieState((0.9, 0.0, 0.9), (0.9, 0.0, 0.9), (0.9, 0.0, 0.9), 0.0,
        (1.0, 0.0, 0.0), 0.02, true, false, Inf, 0.0, 0.90,
        0.0527, 0.0965, (-0.82, 0.0, -0.56), (0.83, 1.52, 0.59),
        [(0.8, 0.8), (1.0, 0.8), (1.0, 1.0), (0.8, 1.0)], [1.0 0.0; 0.0 1.0])
    world = DuckieWorldState(
        DuckieEgoState((0.0, 0.0, 0.0), 0.0, 0.0, 0.0, 0.0, 0, 0.0,
            [(0.0, 0.0, 0.0)], zeros(3, 3), zeros(3, 3), 0.0, 0.0),
        [duck], [StopSignState((0.702, 0.0, 1.8135), π)], map,
        StopMemory(false, 0, nothing, nothing), (1.0, 1.0), [0], [true],
        MersenneTwister(73),
    )

    @test world.ego.step_count == 0
    @test world.stop_memory.last_stop_id === nothing
    @test world.crossings_started == [0]

    b = branch(world)
    push!(b.ego.command_history, (0.1, 0.2, 0.3))
    push!(b.ducks[1].obj_corners, (2.0, 2.0))
    b.crossings_started[1] = 1
    b.stop_memory = StopMemory(true, 1, 0, 0.3)
    rand(b.controller_rng)

    @test length(world.ego.command_history) == 1
    @test length(world.ducks[1].obj_corners) == 4
    @test world.crossings_started == [0]
    @test !world.stop_memory.sigma_stop
    @test world.ego.step_count == b.ego.step_count
    @test world.map === b.map
end

@testset "EventFlags defaults" begin
    e = EventFlags()
    @test !e.collision_duck && !e.other_collision && !e.offroad
    @test !e.timeout && !e.stop_violation && !e.full_stop
    @test !e.passed_stop && !e.goal
    e2 = EventFlags(collision_duck=true, full_stop=true)
    @test e2.collision_duck && e2.full_stop
end