# FJ3.6: state-extraction observer parity against test/fixtures/fj3_obs.json
# (reference run in ddm-ref: state.py get_raw_state / next_stop_candidate /
# classify_duck / tile_ahead + continuous_state.py duck_relative_state /
# signed_curvature_ahead / build_continuous_state / encode).
#
# Rows are latent-in -> outputs-out: the world state is rebuilt from each
# row's recorded latent (ego pose/speed, duck center/heading/vel/active,
# controller counters, sigma/fallback memory), so this pins the pure
# extraction functions without accumulating dynamics drift. 300 rows come
# from a driven 3-lap rollout (curves, the crossing, both duck sides), 29
# from synthetic sweeps covering every stop-candidate filter branch, all 5
# DuckThreat classes, the detection gate, controller-counter availability,
# the NotInLane fallback, and the off-map STRAIGHT fallback.
#
# Known scenario observation (recorded, not a defect of the port): with the
# reference small_loop + injected sign at (0.702, 1.8135) rot 180, the legal
# lane direction never passes the sign within the 0.40 m lateral filter, so
# d_stop fires only in the synthetic rows here. Whether the reference
# training wrapper ever observes d_stop is an FJ4/FJ6 scenario question.

using DuckietownDecisionModels
using Test
using JSON3
using Random
using LinearAlgebra: I

const FIXTURES_OBS = joinpath(pkgdir(DuckietownDecisionModels),
    "test", "fixtures", "fj3_obs.json")

fixtures_obs = JSON3.read(FIXTURES_OBS)

function unf_obs(x)
    x === nothing && return nothing
    if x isa AbstractDict
        tag = x["nonfinite"]
        return tag == "nan" ? NaN :
            tag == "inf" ? Inf :
            tag == "-inf" ? -Inf : -0.0
    end
    return Float64(x)
end

ulps_obs(a::Float64, b::Float64) =
    abs(reinterpret(Int64, a) - reinterpret(Int64, b))

sc_obs(a, b; n=2) = a == b ||
    (a isa Float64 && ulps_obs(a, unf_obs(b)) <= n)

ulps32(a::Float32, b::Float32) =
    abs(reinterpret(Int32, a) - reinterpret(Int32, b))

# EXPECTED NUMERICAL DIFFERENCE (same class as the FJ2 atan2 deviation):
# `angle_rad = acos(dotDir)` where `dotDir = dot(get_dir_vec(angle), tangent)`
# goes through libm sin/cos, which differ by <= 1 ULP between glibc (fixture,
# WSL) and OpenLibm (Windows Julia). acos is ill-conditioned near |dot| = 1
# (derivative 1/sin(angle)), so a 1-ULP input difference amplifies to
# ~eps/|sin(angle)| absolute error in the tiny angle (measured worst case
# 2.1e-14 rad at angle 5.4e-3; dist is bit-exact on every such row). The
# comparator therefore allows <= 2 ULP or the conditioning-scaled bound.
sc_angle(a::Float64, b; n=2) = sc_obs(a, b; n) ||
    abs(a - unf_obs(b)) <= 4 * eps(1.0) / max(abs(sin(a)), 1e-6)

function world_from_row(row)
    inp = row.in
    pos = (unf_obs(inp.pos[1]), unf_obs(inp.pos[2]), unf_obs(inp.pos[3]))
    ego = DuckieEgoState(pos, unf_obs(inp.angle), 0.0, 0.0,
        unf_obs(inp.speed), 0, 0.0, Tuple{Float64,Float64,Float64}[],
        zeros(3, 3), zeros(3, 3), 0.0, 0.0)
    dj = inp.duck
    center = (unf_obs(dj.center[1]), unf_obs(dj.center[2]), unf_obs(dj.center[3]))
    ds = fixtures_obs.duck_static
    duck = DuckieState(center, center,
        (unf_obs(ds.start[1]), unf_obs(ds.start[2]), unf_obs(ds.start[3])),
        0.0,
        (unf_obs(dj.heading[1]), unf_obs(dj.heading[2]), unf_obs(dj.heading[3])),
        unf_obs(dj.vel), Bool(dj.visible), Bool(dj.active), Inf, 0.0,
        unf_obs(ds.walk_distance), 1.0, 0.1, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0),
        [(0.0, 0.0), (0.1, 0.0), (0.1, 0.1), (0.0, 0.1)],
        Matrix{Float64}(I, 2, 2))
    signs = [StopSignState(
            (unf_obs(s.pos[1]), unf_obs(s.pos[2]), unf_obs(s.pos[3])),
            unf_obs(s.angle))
        for s in fixtures_obs.stop_signs]
    return DuckieWorldState(ego, [duck], signs, small_loop_map(),
        StopMemory(Bool(inp.sigma_stop), 0, nothing, nothing),
        (unf_obs(inp.fallback[1]), unf_obs(inp.fallback[2])),
        Int[Int(v) for v in inp.crossings_started],
        Bool[Bool(v) for v in inp.crossing_armed],
        MersenneTwister(53))
end

obs_ctrl_cfg() = DuckControllerConfig(
    max_crossings_per_episode=Int(fixtures_obs.cfg.max_crossings_per_episode))

@testset "FJ3.6 lane frames + raw-state extraction parity" begin
    s_cfg = StateConfig()
    for row in fixtures_obs.rows
        world = world_from_row(row)
        out = row.out

        ft, rt = lane_frame_tabular(world)
        fc, rc = lane_frame_continuous(world)
        for k in 1:3
            @test sc_obs(ft[k], out.fwd_tab[k])
            @test sc_obs(rt[k], out.right_tab[k])
            @test sc_obs(fc[k], out.fwd_cont[k])
            @test sc_obs(rc[k], out.right_cont[k])
        end

        if Bool(out.lane_ok)
            lp = get_lane_pos2(world.map, collect(world.ego.pos),
                world.ego.angle)
            @test sc_obs(lp.dist, out.lane_dist)
            @test sc_angle(lp.angle_rad, out.lane_angle)
        else
            @test_throws NotInLane get_lane_pos2(world.map,
                collect(world.ego.pos), world.ego.angle)
        end

        raw, fallback = get_raw_state(world, s_cfg;
            sigma_stop=Bool(row.in.sigma_stop))
        @test sc_obs(raw.d, out.raw.d)
        @test sc_angle(raw.phi, out.raw.phi)
        @test sc_obs(raw.v, out.raw.v)
        @test Int(raw.tile) == Int(out.raw.tile)
        if out.raw.d_stop === nothing
            @test raw.d_stop === nothing
        else
            @test raw.d_stop !== nothing && sc_obs(raw.d_stop, out.raw.d_stop)
        end
        @test raw.sigma_stop == Bool(out.raw.sigma_stop)
        @test Int(raw.duck) == Int(out.raw.duck)
        # fallback memory: updated on a successful lane query, else unchanged
        if Bool(out.lane_ok)
            @test sc_obs(fallback[1], out.lane_dist)
            @test sc_angle(fallback[2], out.lane_angle)
        else
            @test fallback == world.lane_fallback
        end

        dist, index = next_stop_candidate(world, s_cfg)
        if out.stop_index === nothing
            @test dist === nothing && index === nothing
        else
            @test index == Int(out.stop_index)
            @test dist !== nothing && sc_obs(dist, out.raw.d_stop)
        end
    end
end

@testset "FJ3.6 continuous-state extraction parity" begin
    s_cfg = StateConfig()
    plain = ContinuousStateConfig()
    gated = ContinuousStateConfig(duck_detection_range=2.0,
        duck_detection_corridor_width=0.35, duck_detection_forward_only=true)
    ctrl = obs_ctrl_cfg()
    for row in fixtures_obs.rows
        world = world_from_row(row)
        out = row.out
        ccfg = Bool(row.in.gated) ? gated : plain

        raw, _ = get_raw_state(world, s_cfg;
            sigma_stop=Bool(row.in.sigma_stop))

        drel = duck_relative_state(world, raw.v, ctrl)
        @test drel.present == Bool(out.duck_rel.present)
        @test sc_obs(drel.longitudinal, out.duck_rel.longitudinal)
        @test sc_obs(drel.lateral, out.duck_rel.lateral)
        @test sc_obs(drel.v_longitudinal_relative, out.duck_rel.v_long)
        @test sc_obs(drel.v_lateral_relative, out.duck_rel.v_lat)
        @test drel.active == Bool(out.duck_rel.active)
        @test drel.crossing_available == Bool(out.duck_rel.crossing_available)

        kappa = signed_curvature_ahead(world, s_cfg, ccfg)
        @test sc_obs(kappa, out.kappa)

        cont = get_continuous_state(world, raw, s_cfg, ccfg;
            controller_cfg=ctrl,
            stop_hold_progress=unf_obs(row.in.stop_hold_progress))
        @test sc_obs(cont.kappa, out.cont.kappa)
        @test cont.stop_present == Bool(out.cont.stop_present)
        @test cont.duck_present == Bool(out.cont.duck_present)
        @test sc_obs(cont.duck_longitudinal, out.cont.duck_longitudinal)
        @test sc_obs(cont.duck_lateral, out.cont.duck_lateral)
        @test sc_obs(cont.duck_v_longitudinal_relative, out.cont.duck_v_long)
        @test sc_obs(cont.duck_v_lateral_relative, out.cont.duck_v_lat)
        @test cont.duck_active == Bool(out.cont.duck_active)
        @test cont.duck_crossing_available ==
            Bool(out.cont.duck_crossing_available)
        @test sc_obs(cont.stop_hold_progress, out.cont.stop_hold_progress)

        encoded = encode_continuous_state(cont, ccfg)
        for k in 1:15
            expected = Float32(unf_obs(out.encoded[k]))
            @test ulps32(encoded[k], expected) <= 1
        end
    end
end

@testset "FJ3.6 coverage sanity" begin
    rows = fixtures_obs.rows
    @test length(rows) >= 320
    threats = Set(Int(r.out.raw.duck) for r in rows)
    @test threats == Set(0:4)
    @test any(r.out.raw.d_stop !== nothing for r in rows)
    @test any(r.out.raw.d_stop === nothing for r in rows)
    @test any(!Bool(r.out.lane_ok) for r in rows)
    @test any(!Bool(r.out.duck_rel.crossing_available) for r in rows)
    @test any(!Bool(r.out.duck_rel.present) for r in rows)
    tiles = Set(Int(r.out.raw.tile) for r in rows)
    @test tiles == Set(0:2)
end
