# FJ3.1: spawn-pose sampling (raw simulator reset).
#
# Verbatim port of the spawn loop in `Simulator.reset` (pinned
# gym-duckietown 6.1.34): sample `x`, `z`, `angle` on the start tile, reject
# inconvenient / invalid poses, require `|angle_deg| < accept_start_angle_deg`
# from the lane, else fall back to `(1, 0, 1)` at angle 1 after
# MAX_SPAWN_ATTEMPTS.
#
# The RNG argument is any `AbstractRNG`, and the draw order (3 draws per
# attempt: x, z, angle) is the reference one — so passing a [`NumpyPCG64`](@ref)
# (FJ3.8) reproduces the reference `np_random` draw stream exactly, while any
# native Julia RNG gives an equally valid initial-state distribution.
#
# FJ4 adds the wrapper-level reset on top (`DuckieMDPEnv.reset`): the duckie /
# stop-sign objects the DuckController injects, the curriculum acceptance loop
# over `spawn_attempts`, and the stop/lane memory reset.

"""
    sample_spawn_pose(map, rng, i, j, accept_start_angle_deg) -> (pos, angle)

Simulator `reset` spawn sampling on tile `(i, j)` (world coordinates, x/z
within `[i·ts, (i+1)·ts)`). Returns `(pos3, angle)`; on exhaustion of
`MAX_SPAWN_ATTEMPTS` the reference fallback `([1.0, 0.0, 1.0], 1.0)`.
"""
function sample_spawn_pose(map::RoadMap, rng::AbstractRNG,
    i::Int, j::Int, accept_start_angle_deg::Float64)
    ts = map.tile_size
    for _ in 1:MAX_SPAWN_ATTEMPTS
        x = (Float64(i) + rand(rng)) * ts
        z = (Float64(j) + rand(rng)) * ts
        propose_pos = [x, 0.0, z]
        propose_angle = rand(rng) * 2pi

        _inconvenient_spawn(map, propose_pos) && continue

        _valid_pose(map, propose_pos, propose_angle, 1.3) || continue

        lp = try
            get_lane_pos2(map, propose_pos, propose_angle)
        catch e
            e isa NotInLane || rethrow()
            continue
        end
        M = accept_start_angle_deg
        (-M < lp.angle_deg < M) || continue

        return propose_pos, propose_angle
    end
    return [1.0, 0.0, 1.0], 1.0
end

# ---------------------------------------------------------------------------
# FJ4: wrapper-level reset (`DuckieMDPEnv.reset` / `make_ducks_dynamic`)
# ---------------------------------------------------------------------------

"""
    initial_duckie(map, cfg) -> DuckieState

The dynamic duckie the controller injects (`duck_controller._inject_duck` +
`Simulator.interpret_object` + `DuckieObj.__init__` with `domain_rand=false`):
mesh geometry from the tile-frame descriptor, `vel = 0.02`,
`pedestrian_wait_time = 8`, `pedestrian_active = false`, `time = 0`,
`start = center = pos`, `heading = heading_vec(angle)`.

`walk_distance` is `cfg.walk_distance` because `DuckController.__init__`
stamps it over the simulator's default (`road_tile_size`) — the FJ3.5 finding.
"""
function initial_duckie(map::RoadMap, cfg::DuckControllerConfig)
    pos, angle = object_world_pose(map, cfg.spawn_pos, cfg.spawn_rotate)
    obj = interpret_object_desc(kind=:duckie, pos=pos,
        rotate_deg=cfg.spawn_rotate, height=cfg.spawn_height,
        static=false, optional=false)
    corners = [(obj.corners[k, 1], obj.corners[k, 2]) for k in 1:4]
    return DuckieState(pos, pos, pos, angle, heading_vec(angle), 0.02,
        true, false, 8.0, 0.0, cfg.walk_distance, obj.scale,
        obj.safety_radius, obj.min_coords, obj.max_coords, corners,
        copy(obj.norm))
end

"""
    initial_map(cfg) -> RoadMap

The experiment map with the controller's injected static stop sign
(`duck_controller._inject_stop`, tile-frame descriptor), when the config asks
for one. Duckies are dynamic state, not map objects.
"""
function initial_map(cfg::DuckietownConfig)
    base = small_loop_map(with_stop_sign=false)
    dcfg = cfg.duck_controller
    (dcfg.inject_stop_if_missing || dcfg.require_stop) || return base
    pos, _ = object_world_pose(base, dcfg.stop_spawn_pos, dcfg.stop_spawn_rotate)
    sign = interpret_object_desc(kind=:sign_stop, pos=pos,
        rotate_deg=dcfg.stop_spawn_rotate, height=dcfg.stop_spawn_height,
        static=true)
    return RoadMap(base.name, base.tile_size, base.grid, [sign])
end

"""
    route_circulation_score(pos, angle, center) -> Float64

`env_wrapper.route_circulation_score`: alignment of the ego heading with the
clockwise tangent around `center` (x-z plane). Positive = clockwise. Used
only to constrain the initial-state distribution; never part of the state.
"""
function route_circulation_score(pos::NTuple{3,Float64}, angle::Float64,
    center::NTuple{2,Float64})
    radial = (pos[1] - center[1], pos[3] - center[2])
    radial_norm = hypot(radial[1], radial[2])
    radial_norm <= 1e-9 && return 0.0
    tangent = (radial[2] / radial_norm, -radial[1] / radial_norm)
    forward = (cos(angle), -sin(angle))
    return forward[1] * tangent[1] + forward[2] * tangent[2]
end

"""
    position_in_bounds_xz(pos, bounds) -> Bool

`env_wrapper.position_in_bounds_xz` with `bounds = (xmin, xmax, zmin, zmax)`.
"""
function position_in_bounds_xz(pos::NTuple{3,Float64}, bounds::NTuple{4,Float64})
    xmin, xmax, zmin, zmax = bounds
    (xmin > xmax || zmin > zmax) &&
        throw(ArgumentError("spawn position bounds must satisfy min <= max"))
    return xmin <= pos[1] <= xmax && zmin <= pos[3] <= zmax
end

"""
    _uniform_tile_index(rng, n) -> Int

`np_random.integers(0, n)` for the random start-tile choice. A
[`NumpyPCG64`](@ref) takes the reference (buffered-Lemire) path; any other
RNG uses the native uniform draw.
"""
_uniform_tile_index(rng::NumpyPCG64, n::Int) = np_integers(rng, 0, n)
_uniform_tile_index(rng::AbstractRNG, n::Int) = rand(rng, 0:(n - 1))

"""
    drivable_tiles(map) -> Vector{NTuple{2,Int}}

Zero-based `(i, j)` coordinates of the drivable tiles, in the reference
row-major insertion order of `Simulator._interpret_map`.
"""
function drivable_tiles(map::RoadMap)
    out = NTuple{2,Int}[]
    for j in 1:size(map.grid, 1), i in 1:size(map.grid, 2)
        map.grid[j, i].drivable && push!(out, (i - 1, j - 1))
    end
    return out
end