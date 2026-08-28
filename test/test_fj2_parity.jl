# FJ2: PYTHON ↔ JULIA semantic parity against the fixture file produced by
# tools/parity/gen_fj2_fixtures.py (real duckduck/src functions, numpy 1.26.4).
# Floats are compared bit-for-bit (Float64: reinterpret UInt64; Float32 values
# stored as their exact float64 image, then re-rounded to Float32).

using DuckietownDecisionModels
using Test
using JSON3

const FIXTURES = joinpath(pkgdir(DuckietownDecisionModels),
    "test", "fixtures", "fj2_parity.json")

fixtures = JSON3.read(FIXTURES)

asfloat(x) = Float64(x)
asint(x) = Int(x)
asbool(x) = Bool(x)

function unf(x)
    # JSON markers for non-finite / signed-zero floats:
    # {"nonfinite": "nan"|"inf"|"-inf"|"-0.0"}
    if x isa AbstractDict
        tag = x["nonfinite"]
        return tag == "nan" ? NaN :
            tag == "inf" ? Inf :
            tag == "-inf" ? -Inf : -0.0
    end
    return Float64(x)
end

bits(x::Float64) = reinterpret(UInt64, x)
bits32(x::Float32) = reinterpret(UInt32, x)

@testset "FJ2.1 actions parity" begin
    @testset "build_action_table" begin
        names = [spec.name for spec in build_action_table()]
        @test names == ["fast_left", "fast_straight", "fast_right",
            "slow_left", "slow_straight", "slow_right", "brake"]
        for (name, v, omega) in fixtures.actions.table
            spec = build_action_table()[findfirst(n -> n == name, names)]
            @test spec.v == Float64(v) && spec.omega == Float64(omega)
        end
    end

    @testset "vw_to_wheels ($(length(fixtures.actions.vw_to_wheels)) cases)" begin
        for (v, omega, wb, expected) in fixtures.actions.vw_to_wheels
            l, r = vw_to_wheels(asfloat(v), asfloat(omega), asfloat(wb))
            @test bits32(l) == bits32(Float32(asfloat(expected[1])))
            @test bits32(r) == bits32(Float32(asfloat(expected[2])))
        end
    end

    @testset "action_to_wheels" begin
        for (aid, expected) in fixtures.actions.action_to_wheels
            l, r = action_to_wheels(asint(aid))
            @test bits32(l) == bits32(Float32(asfloat(expected[1])))
            @test bits32(r) == bits32(Float32(asfloat(expected[2])))
        end
        for (aid, vf, vs, w0, wb, expected) in fixtures.actions.action_to_wheels_alt
            cfg = ActionConfig(v_fast=vf, v_slow=vs, w0=w0, wheel_base=wb)
            l, r = action_to_wheels(asint(aid), cfg)
            @test bits32(l) == bits32(Float32(asfloat(expected[1])))
            @test bits32(r) == bits32(Float32(asfloat(expected[2])))
        end
        for bad in fixtures.actions.bad_ids
            @test_throws ArgumentError action_to_wheels(asint(bad))
        end
    end
end

@testset "FJ2.2 discretizer parity ($(length(fixtures.discretize.cases)) cases)" begin
    @test STATE_SHAPE == (asint.(fixtures.discretize.state_shape)...,)
    @test Q_SHAPE == (STATE_SHAPE..., 7)
    for (d, phi, v, tile, d_stop, sigma, duck, idx) in fixtures.discretize.cases
        st = RawState(asfloat(d), asfloat(phi), asfloat(v),
            TileType(asint(tile)),
            d_stop === nothing ? nothing : asfloat(d_stop),
            asbool(sigma), DuckThreat(asint(duck)))
        @test collect(discretize(st)) == asint.(idx)
    end
    @test fixtures.discretize.errors == []  # guard unreachable, as in Python
end

@testset "FJ2.3 encoding parity ($(length(fixtures.encoding.cases)) cases)" begin
    cfgs = (ContinuousStateConfig(),
        ContinuousStateConfig(max_speed=0.5, max_abs_curvature=4.0,
            max_stop_distance=1.5, max_duck_distance=1.5, max_relative_speed=1.0))
    for c in fixtures.encoding.cases
        (d, phi, v, kappa, stop_present, d_stop, sigma, duck_present,
            dlong, dlat, vlong, vlat, active, crossing, hold, cfg_id, obs) = c
        st = ContinuousState(asfloat(d), asfloat(phi), asfloat(v), asfloat(kappa),
            asbool(stop_present),
            d_stop === nothing ? nothing : asfloat(d_stop),
            asbool(sigma), asbool(duck_present), asfloat(dlong), asfloat(dlat),
            asfloat(vlong), asfloat(vlat), asbool(active), asbool(crossing),
            asfloat(hold))
        got = encode_continuous_state(st, cfgs[asint(cfg_id) + 1])
        @test length(got) == 15
        for i in 1:15
            @test bits32(got[i]) == bits32(Float32(unf(obs[i])))
        end
    end

    low, high = continuous_observation_space()
    @test low == Float32[unf(x) for x in fixtures.encoding.low]
    @test high == Float32[unf(x) for x in fixtures.encoding.high]

    @test_throws ArgumentError encode_continuous_state(
        ContinuousState(0.0, NaN, 0.0, 0.0, false, nothing, false,
            false, 0.0, 0.0, 0.0, 0.0, false, false, 0.0),
        ContinuousStateConfig())
end

@testset "FJ2.3 gate_duck_visibility parity ($(length(fixtures.gate.cases)) cases)" begin
    cfgs = (ContinuousStateConfig(),
        ContinuousStateConfig(duck_detection_range=1.2,
            duck_detection_corridor_width=0.6, duck_detection_forward_only=true),
        ContinuousStateConfig(duck_detection_range=1.2),
        ContinuousStateConfig(duck_detection_corridor_width=0.6),
        ContinuousStateConfig(duck_detection_forward_only=true))
    for (present, long, lat, vlong, vlat, active, crossing, cfg_id, out) in fixtures.gate.cases
        duck = DuckRelativeState(asbool(present), asfloat(long), asfloat(lat),
            asfloat(vlong), asfloat(vlat), asbool(active), asbool(crossing))
        g = gate_duck_visibility(duck, cfgs[asint(cfg_id) + 1])
        @test (g.present, g.longitudinal, g.lateral,
            g.v_longitudinal_relative, g.v_lateral_relative,
            g.active, g.crossing_available) ==
            (asbool(out[1]), unf(out[2]), unf(out[3]), unf(out[4]),
                unf(out[5]), asbool(out[6]), asbool(out[7]))
    end
end

@testset "FJ2.6 classify_tile parity" begin
    for (drivable, kind, expected) in fixtures.classify_tile.cases
        @test Int(classify_tile(asbool(drivable), String(kind))) == asint(expected)
    end
    for (drivable, kind) in fixtures.classify_tile.errors
        if kind === nothing
            @test_throws ArgumentError classify_tile(nothing)
        else
            @test_throws ArgumentError classify_tile(asbool(drivable), String(kind))
        end
    end
end

@testset "FJ2.4 StopTracker parity ($(length(fixtures.stop_tracker.cases)) sequences)" begin
    raw_from(c) = RawState(asfloat(c[1]), asfloat(c[2]), asfloat(c[3]),
        TileType(asint(c[4])),
        c[5] === nothing ? nothing : asfloat(c[5]),
        asbool(c[6]), DuckThreat(asint(c[7])))
    events_of(ev) = EventFlags(collision_duck=ev[1], other_collision=ev[2],
        offroad=ev[3], timeout=ev[4], stop_violation=ev[5], full_stop=ev[6],
        passed_stop=ev[7], goal=ev[8])
    for (name, cfg, steps) in fixtures.stop_tracker.cases
        tracker = StopTracker(asfloat(cfg[1]), asfloat(cfg[2]), asfloat(cfg[3]),
            asint(cfg[4]))
        for (prev_c, curr_c, prev_id, curr_id, sigma, ev, tsigma, thold) in steps
            prev = raw_from(prev_c)
            curr = raw_from(curr_c)
            prev_id === nothing || (prev_id = asint(prev_id))
            curr_id === nothing || (curr_id = asint(curr_id))
            got_sigma, got_events = update!(tracker, prev, curr, prev_id, curr_id)
            @test got_sigma == asbool(sigma)
            @test got_events == events_of(ev)
            @test tracker.sigma_stop == asbool(tsigma)
            @test tracker.hold_steps == asint(thold)
        end
        @test tracker.sigma_stop == asbool(steps[end][7])
        @test tracker.hold_steps == asint(steps[end][8])
    end
end

@testset "FJ2.5 reward parity ($(length(fixtures.reward.cases)) cases)" begin
    cfgs = (RewardConfig(),
        RewardConfig(duck_yield=1.0, duck_unsafe=-5.0, unnecessary_stop=-2.0),
        RewardConfig(duck_unsafe=-5.0, unnecessary_stop=-2.0,
            stop_approach_distance=0.60, stop_approach_speed=0.02,
            stop_approach_yield=1.0, stop_approach_unsafe=-5.0,
            straight_steer_penalty=0.5),
        RewardConfig(alpha_progress=1.0, alpha_lateral=2.0, alpha_heading=0.5,
            step_cost=0.01, collision_duck=-200.0, other_collision=-50.0,
            offroad=-200.0, stop_violation=-40.0, full_stop=15.0, goal=50.0))
    events_of(ev) = EventFlags(collision_duck=ev[1], other_collision=ev[2],
        offroad=ev[3], timeout=ev[4], stop_violation=ev[5], full_stop=ev[6],
        passed_stop=ev[7], goal=ev[8])
    evs = (events_of([false, false, false, false, false, false, false, false]),
        events_of([true, false, false, false, false, false, false, false]),
        events_of([false, false, true, false, false, false, false, false]),
        events_of([true, false, true, false, false, false, false, false]),
        events_of([false, false, false, false, true, false, false, false]),
        events_of([false, false, false, false, false, true, false, false]),
        events_of([false, false, false, false, true, true, false, false]),
        events_of([false, false, false, false, false, false, false, true]),
        events_of([true, true, true, true, true, true, true, true]))
    for (d, phi, v, tile, d_stop, sigma, duck, omega, curvature, cfg_id, ev_id, out) in fixtures.reward.cases
        st = RawState(asfloat(d), asfloat(phi), asfloat(v), TileType(asint(tile)),
            d_stop === nothing ? nothing : asfloat(d_stop),
            asbool(sigma), DuckThreat(asint(duck)))
        curv = curvature === nothing ? nothing : unf(curvature)
        bd = compute_reward(st, evs[asint(ev_id) + 1], cfgs[asint(cfg_id) + 1];
            action_omega=asfloat(omega), curvature=curv)
        fields = (bd.progress, bd.lateral, bd.heading, bd.time, bd.pedestrian,
            bd.stagnation, bd.stop_approach, bd.steering, bd.events, bd.total)
        for i in 1:10
            @test bits(fields[i]) == bits(unf(out[i]))
        end
    end
end

@testset "FJ2.6 bezier + curvature parity" begin
    # NaN is a legitimate bezier_tangent output on zero-length segments; the
    # two runtimes canonically differ in the NaN sign bit (Julia NaN = -NaN),
    # so compare NaN as NaN and everything else bit-for-bit.
    bits_eq(a, b) = bits(a) == bits(b) || (isnan(a) && isnan(b))
    curves = Dict{String,Matrix{Float64}}(
        String(name) => permutedims(reshape(map(unf, collect(Iterators.flatten(v))), 3, 4)) for
        (name, v) in pairs(fixtures.geometry.curves))
    for (name, t, kind, expected) in fixtures.geometry.bezier
        curve = curves[name]
        got = kind == "point" ? bezier_point(curve, asfloat(t)) :
            bezier_tangent(curve, asfloat(t))
        @test length(got) == 3
        for i in 1:3
            @test bits_eq(got[i], unf(expected[i]))
        end
    end
    for (name, samples, threshold, value) in fixtures.geometry.curvature
        got = curve_signed_curvature(curves[name]; samples=asint(samples),
            straight_angle_threshold=asfloat(threshold))
        # Bit-exact, except for atan2: the Windows OpenLibm and the WSL glibc
        # implementations differ by 1 ULP in the last bit for some arguments.
        # Every other intermediate (tangents, cross, dot, arc length, division)
        # is bit-identical; this tolerance covers only that one libm call.
        exp = unf(value)
        @test bits(got) == bits(exp) ||
            abs(got - exp) <= eps(max(abs(got), abs(exp)))
    end
    for (name, samples, threshold) in fixtures.geometry.curvature_errors
        @test_throws ArgumentError curve_signed_curvature(curves[name];
            samples=asint(samples), straight_angle_threshold=asfloat(threshold))
    end
end

@testset "FJ2.6 terminal_lane_fallback parity" begin
    for (last_d, last_phi, expected) in fixtures.terminal_fallback.cases
        d, phi = last_d === nothing ? terminal_lane_fallback(1.0, 1.0) :
            terminal_lane_fallback(asfloat(last_d), asfloat(last_phi))
        @test bits(d) == bits(asfloat(expected[1]))
        @test bits(phi) == bits(asfloat(expected[2]))
    end
end