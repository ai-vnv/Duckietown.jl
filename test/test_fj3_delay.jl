# FJ3.3: delay buffer — the 0.15 s trimmed command window carried in
# `DuckieEgoState.command_history`, parity against the reference
# `DelayedDynamics` window content recorded in test/fixtures/fj3_ego.json
# (buf_ts / buf_cmds per tick) and its cross-decision continuity under
# frame_skip = 6 decisions.
#
# The reference trims `commands`/`timestamps` to the delay window on every
# physics tick (`DelayedDynamics.__init__` slices at the selection index for
# `timestamps[-1] - delay`), so the buffer never exceeds delay/dt + 1 entries.
# `ego_tick` replicates this; the FJ3.3 gate pins the window *content*
# (timestamps and commands, bit-exact) and proves a decision-step API that
# runs ticks in chunks of 6 produces the same world state as one continuous
# run (no reset at decision boundaries).

using DuckietownDecisionModels
using Test
using JSON3

const FIXTURES_DELAY = joinpath(pkgdir(DuckietownDecisionModels),
    "test", "fixtures", "fj3_ego.json")

fixtures_delay = JSON3.read(FIXTURES_DELAY)

function unf_delay(x)
    if x isa AbstractDict
        tag = x["nonfinite"]
        return tag == "nan" ? NaN :
            tag == "inf" ? Inf :
            tag == "-inf" ? -Inf : -0.0
    end
    return Float64(x)
end

ulps_delay(a::Float64, b::Float64) =
    abs(reinterpret(Int64, a) - reinterpret(Int64, b))

mat_close_delay(a, b) = all(ulps_delay(unf_delay(x), unf_delay(y)) <= 1 ||
    isapprox(unf_delay(x), unf_delay(y); rtol=4eps()) for (x, y) in zip(a, b))

mat_rows_delay(m::AbstractMatrix) =
    [m[i, j] for i in 1:size(m, 1) for j in 1:size(m, 2)]

function fresh_world_delay()
    gh = Int(fixtures_delay.grid_height)
    ts = unf_delay(fixtures_delay.tile_size)
    pos = (unf_delay(fixtures_delay.init_pos[1]), 0.0,
        unf_delay(fixtures_delay.init_pos[3]))
    return DuckieWorldState(
        initial_ego(pos, unf_delay(fixtures_delay.init_angle), gh, ts),
        DuckieState[], [], small_loop_map(),
        StopMemory(false, 0, nothing, nothing), (1.0, 1.0), Int[], Bool[],
        MersenneTwister(53))
end

ego_fields_equal(a::DuckieEgoState, b::DuckieEgoState) =
    ulps_delay(a.pos[1], b.pos[1]) <= 2 && ulps_delay(a.pos[2], b.pos[2]) <= 2 &&
    ulps_delay(a.pos[3], b.pos[3]) <= 2 && ulps_delay(a.angle, b.angle) <= 2 &&
    ulps_delay(a.v_long, b.v_long) <= 2 && ulps_delay(a.omega, b.omega) <= 2 &&
    a.step_count == b.step_count && ulps_delay(a.timestamp, b.timestamp) <= 2 &&
    a.command_history == b.command_history &&
    mat_close_delay(mat_rows_delay(a.q0), mat_rows_delay(b.q0)) &&
    mat_close_delay(mat_rows_delay(a.v0), mat_rows_delay(b.v0)) &&
    ulps_delay(a.axis_left_rad, b.axis_left_rad) <= 2 &&
    ulps_delay(a.axis_right_rad, b.axis_right_rad) <= 2

@testset "FJ3.3 delay buffer window parity (18 ticks)" begin
    world = fresh_world_delay()
    @test world.ego.command_history == Tuple{Float64,Float64,Float64}[]
    for f in fixtures_delay.ticks
        world = ego_tick(world, (unf_delay(f.cmd[1]), unf_delay(f.cmd[2])))
        hist = world.ego.command_history
        # window timestamps bit-exact against the reference trimmed window
        @test [h[1] for h in hist] == [unf_delay(t) for t in f.buf_ts]
        # window commands bit-exact (reference keeps the same PWMCommands)
        @test [h[2] for h in hist] == [unf_delay(c[1]) for c in f.buf_cmds]
        @test [h[3] for h in hist] == [unf_delay(c[2]) for c in f.buf_cmds]
        # buffer bound: at most delay/dt + 1 = 5.5 -> 6 entries
        @test length(hist) <= 6
        # strictly increasing timestamps
        @test all(hist[i][1] < hist[i + 1][1] for i in 1:(length(hist) - 1))
    end
end

@testset "FJ3.3 cross-decision memory (frame_skip=6 chunks)" begin
    world_chunked = fresh_world_delay()
    world_cont = fresh_world_delay()
    ticks = collect(fixtures_delay.ticks)
    @assert length(ticks) == 18
    for (decision, chunk) in enumerate(Iterators.partition(ticks, 6))
        for f in chunk
            world_cont = ego_tick(world_cont,
                (unf_delay(f.cmd[1]), unf_delay(f.cmd[2])))
        end
        # decision step runs its 6 ticks against the SAME world state
        for f in chunk
            world_chunked = ego_tick(world_chunked,
                (unf_delay(f.cmd[1]), unf_delay(f.cmd[2])))
        end
        # chunked == continuous at every decision boundary (buffer carries over)
        @test ego_fields_equal(world_chunked.ego, world_cont.ego)
        @test world_chunked.ego.command_history == world_cont.ego.command_history
    end
    # sanity: after 3 decisions the ego has advanced
    @test world_chunked.ego.step_count == 18
    @test world_chunked.ego.pos != fresh_world_delay().ego.pos
end

@testset "FJ3.3 u0 fallback before the window" begin
    world = fresh_world_delay()
    sel = get_commands_at(world.ego.command_history, -0.01)
    @test sel == DelayedCommand(0, 0.0, (0.0, 0.0))
    # first 4 ticks apply u0 = (0, 0) while the buffer fills
    used = NTuple{2,Float64}[]
    for _ in 1:4
        world = ego_tick(world, (0.7, 0.4))
        push!(used, get_commands_at(world.ego.command_history,
            world.ego.timestamp - EGO_DELAY).used)
    end
    @test used == fill((0.0, 0.0), 4)
    @test length(world.ego.command_history) == 4
    @test [h[1] for h in world.ego.command_history] ==
        [unf_delay(t) for t in fixtures_delay.ticks[4].buf_ts]
    # from tick 5 on the first commanded action is applied
    world = ego_tick(world, (0.7, 0.4))
    sel = get_commands_at(world.ego.command_history,
        world.ego.timestamp - EGO_DELAY)
    @test sel.used == (0.7, 0.4)
    @test sel.idx == 1
end