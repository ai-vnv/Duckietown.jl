# FJ3.6: env-dependent state extraction — the observers that project the
# latent `DuckieWorldState` onto `RawState` (f_tab) and `ContinuousState`
# (f_cont), verbatim from `duckduck/src/state.py` and
# `duckduck/src/continuous_state.py`.
#
# The two Python modules carry two *different* `_lane_frame`/`_normalize`
# implementations; both are preserved:
# - `state.py`: normalize iff `norm > 0`, `right = cross(forward, ŷ)` NOT
#   normalized, fallback when `point is None or tangent is None` or the
#   normalized forward is all-zero;
# - `continuous_state.py`: normalize iff `norm > 1e-12` (else zeros), `right`
#   IS normalized, fallback only when `tangent is None`.

# np.cross(f, [0, 1, 0]) = (-f3, 0, f1)
_cross_y(forward::AbstractVector{Float64}) =
    [-forward[3], 0.0, forward[1]]

"""
    lane_frame_tabular(world) -> (forward, right)

`state.py::_lane_frame`: forward/right axes of the ego's lane frame used by
the tabular extraction. `right` is NOT normalized (`np.cross` of a unit
forward with ŷ is unit anyway, but the bits are the raw cross product).
"""
function lane_frame_tabular(world::DuckieWorldState)
    pos = collect(world.ego.pos)
    point, tangent = closest_curve_point(world.map, pos, world.ego.angle)
    if point === nothing || tangent === nothing
        forward = collect(heading_vec(world.ego.angle))
    else
        forward = _normalize_gt0(tangent)
        if !any(x -> x != 0.0, forward)
            forward = collect(heading_vec(world.ego.angle))
        end
    end
    return forward, _cross_y(forward)
end

"""
    lane_frame_continuous(world) -> (forward, right)

`continuous_state.py::_lane_frame`: the SAC/TD3 variant — forward normalized
with the `1e-12` threshold (zeroed below it) and `right` normalized too.
"""
function lane_frame_continuous(world::DuckieWorldState)
    pos = collect(world.ego.pos)
    _, tangent = closest_curve_point(world.map, pos, world.ego.angle)
    if tangent === nothing
        forward = collect(heading_vec(world.ego.angle))
    else
        forward = _normalize_1e12(tangent)
    end
    return forward, _normalize_1e12(_cross_y(forward))
end

"""
    tile_ahead(world, cfg) -> TileType

`state.py::tile_ahead`: curvature class at the look-ahead probe point
(`pos + tile_lookahead * forward`), falling back to the current tile when the
probe is off-map/non-drivable. Raises `ArgumentError` (Python `ValueError`)
when no tile exists at all — `get_raw_state` catches it as `STRAIGHT`.
"""
function tile_ahead(world::DuckieWorldState, cfg::StateConfig)
    forward, _ = lane_frame_tabular(world)
    pos = collect(world.ego.pos)
    probe = pos .+ cfg.tile_lookahead .* forward
    tile = _get_tile(world.map, get_grid_coords(world.map, probe)...)
    if tile === nothing || !tile.drivable
        tile = _get_tile(world.map, get_grid_coords(world.map, pos)...)
    end
    tile === nothing &&
        throw(ArgumentError("No drivable tile for curvature lookup"))
    # state.py::_ego_relative_curve: no curves -> classify by tile kind
    # (raises on a non-drivable tile, caught by get_raw_state)
    isempty(tile.curves) && return classify_tile(tile)
    return ego_relative_curve(tile.curves, forward, cfg.curvature_threshold)
end

"""
    next_stop_candidate(world, cfg) -> (distance, index)

`state.py::next_stop_candidate`: nearest stop sign ahead of and facing the
ego, filtered by lateral offset and orientation. Returns
`(nothing, nothing)` when no candidate passes; `index` is the **0-based**
position within `world.stop_signs` (the reference index runs over
`env.objects`, but only its identity across decisions matters — the
StopTracker uses it solely for change detection).
"""
function next_stop_candidate(world::DuckieWorldState, cfg::StateConfig)
    forward, right = lane_frame_tabular(world)
    pos = collect(world.ego.pos)
    best_distance = nothing
    best_index = nothing
    for (index, s) in enumerate(world.stop_signs)
        sign_facing = collect(heading_vec(s.angle))
        dot(sign_facing, forward) > -cfg.stop_orientation_cos && continue
        rel = collect(s.pos) .- pos
        ahead = dot(rel, forward)
        lateral = abs(dot(rel, right))
        (ahead <= 0.0 || lateral > cfg.stop_lateral_limit) && continue
        distance = max(0.0, ahead - cfg.sign_to_line_offset)
        distance <= cfg.stop_max_distance || continue
        if best_distance === nothing || distance < best_distance
            best_distance = distance
            best_index = index - 1
        end
    end
    return best_distance, best_index
end

"""
    distance_to_next_stop(world, cfg) -> Union{Nothing,Float64}

`state.py::distance_to_next_stop` — the distance half of
[`next_stop_candidate`](@ref).
"""
distance_to_next_stop(world::DuckieWorldState, cfg::StateConfig) =
    next_stop_candidate(world, cfg)[1]

"""
    classify_duck(world, cfg) -> DuckThreat

`state.py::classify_duck`: maximum pedestrian threat over the visible duckies
inside the ego's forward corridor (`ahead >= 0`, `|lateral| <= corridor`,
planar distance `<= duck_max_distance`); crossing vs side by
`pedestrian_active`, near vs far by `duck_near_distance`.
"""
function classify_duck(world::DuckieWorldState, cfg::StateConfig)
    forward, right = lane_frame_tabular(world)
    pos = collect(world.ego.pos)
    result = NONE
    for d in world.ducks
        d.visible || continue
        rel = collect(d.center) .- pos
        ahead = dot(rel, forward)
        lateral = abs(dot(rel, right))
        distance = sqrt(rel[1]^2 + rel[3]^2)
        in_forward_corridor = ahead >= 0.0 && lateral <= cfg.duck_corridor_width
        (!in_forward_corridor || distance > cfg.duck_max_distance) && continue
        threat = if d.pedestrian_active
            distance <= cfg.duck_near_distance ? CROSSING_NEAR : CROSSING_FAR
        else
            distance <= cfg.duck_near_distance ? SIDE_NEAR : SIDE_FAR
        end
        result = max(result, threat)
    end
    return result
end

"""
    get_raw_state(world, cfg=StateConfig(); sigma_stop=nothing)
        -> (RawState, lane_fallback)

`state.py::get_raw_state`: extract the 7-component tabular state from the
latent world. Returns the state together with the updated lane-fallback
memory (`env._mdp_last_lane_position` in Python — updated on a successful
lane query, left unchanged on `NotInLane`); the caller threads it back into
the next `DuckieWorldState`. `sigma_stop === nothing` reads
`world.stop_memory.sigma_stop` (Python `env._mdp_sigma_stop`).
"""
function get_raw_state(world::DuckieWorldState,
    cfg::StateConfig=StateConfig(); sigma_stop::Union{Nothing,Bool}=nothing)
    pos = collect(world.ego.pos)
    lane_fallback = world.lane_fallback
    local d, phi
    try
        lane = get_lane_pos2(world.map, pos, world.ego.angle)
        lane_fallback = (lane.dist, lane.angle_rad)
        d = clamp(lane.dist, -0.25, 0.25)
        phi = clamp(lane.angle_rad, -pi / 2, pi / 2)
    catch e
        e isa NotInLane || rethrow()
        d, phi = terminal_lane_fallback(world.lane_fallback...)
    end
    speed = max(0.0, world.ego.speed)
    tile = try
        tile_ahead(world, cfg)
    catch e
        e isa ArgumentError || rethrow()
        STRAIGHT
    end
    all(isfinite, (d, phi, speed)) ||
        throw(ArgumentError("Non-finite raw state"))
    sigma = sigma_stop === nothing ? world.stop_memory.sigma_stop : sigma_stop
    raw = RawState(d, phi, speed, tile, distance_to_next_stop(world, cfg),
        sigma, classify_duck(world, cfg))
    return raw, lane_fallback
end

"""
    signed_curvature_ahead(world, state_cfg, continuous_cfg) -> Float64

`continuous_state.py::signed_curvature_ahead`: signed curvature of the
directed lane at the look-ahead tile, clipped to `±max_abs_curvature`;
`0.0` when no drivable tile or no curves exist (unlike the tabular
`tile_ahead`, this never raises).
"""
function signed_curvature_ahead(world::DuckieWorldState,
    state_cfg::StateConfig, continuous_cfg::ContinuousStateConfig)
    forward, _ = lane_frame_continuous(world)
    pos = collect(world.ego.pos)
    probe = pos .+ state_cfg.tile_lookahead .* forward
    tile = _get_tile(world.map, get_grid_coords(world.map, probe)...)
    if tile === nothing || !tile.drivable
        tile = _get_tile(world.map, get_grid_coords(world.map, pos)...)
    end
    (tile === nothing || !tile.drivable) && return 0.0
    curve = _directed_curve(tile.curves, forward)
    curve === nothing && return 0.0
    value = curve_signed_curvature(curve;
        samples=continuous_cfg.curvature_samples,
        straight_angle_threshold=state_cfg.curvature_threshold)
    return clamp(value, -continuous_cfg.max_abs_curvature,
        continuous_cfg.max_abs_curvature)
end

"""
    _directed_curve(curves, forward) -> Union{Nothing,Matrix{Float64}}

`continuous_state.py::_directed_curve`: the tile curve whose end-to-end
heading best aligns with `forward` (`1e-12` normalization variant); `nothing`
when the tile carries no curves.
"""
function _directed_curve(curves::AbstractVector, forward::AbstractVector{Float64})
    isempty(curves) && return nothing
    mats = [curve_matrix(c) for c in curves]
    headings = [_normalize_1e12(m[end, :] .- m[1, :]) for m in mats]
    return mats[argmax([dot(h, forward) for h in headings])]
end

"""
    duck_relative_state(world, ego_speed, controller_cfg) -> DuckRelativeState

`continuous_state.py::duck_relative_state`: geometry of the nearest visible
duckie in the ego's continuous lane frame. `ego_speed` is `raw.v`.
`controller_cfg === nothing` reproduces `controller=None`
(`crossing_available = true`); otherwise availability follows
`DuckController.crossing_available(i)`:
`(limit <= 0 || crossings_started[i] < limit) && crossing_armed[i]` with `i`
the duckie's index in `world.ducks` (Python: identity lookup in
`controller.ducks`).
"""
function duck_relative_state(world::DuckieWorldState, ego_speed::Float64,
    controller_cfg::Union{Nothing,DuckControllerConfig}=nothing)
    ego_position = collect(world.ego.pos)
    candidate_indices = [i for (i, d) in enumerate(world.ducks) if d.visible]
    isempty(candidate_indices) && return DuckRelativeState()

    dists = [sqrt(sum(abs2, collect(world.ducks[i].center) .- ego_position))
        for i in candidate_indices]
    index = candidate_indices[argmin(dists)]
    duck = world.ducks[index]

    forward, right = lane_frame_continuous(world)
    relative_position = collect(duck.center) .- ego_position
    active = duck.pedestrian_active

    duck_velocity = active ? collect(duck.heading) .* duck.vel : zeros(3)
    ego_velocity = forward .* ego_speed
    relative_velocity = duck_velocity .- ego_velocity

    crossing_available = if controller_cfg === nothing
        true
    else
        limit = controller_cfg.max_crossings_per_episode
        (limit <= 0 || world.crossings_started[index] < limit) &&
            world.crossing_armed[index]
    end

    return DuckRelativeState(true,
        dot(relative_position, forward),
        dot(relative_position, right),
        dot(relative_velocity, forward),
        dot(relative_velocity, right),
        active, crossing_available)
end

"""
    get_continuous_state(world, raw, state_cfg, continuous_cfg;
        controller_cfg=nothing, stop_hold_progress=0.0) -> ContinuousState

`continuous_state.py::build_continuous_state` applied to the latent world:
gated duckie geometry + signed look-ahead curvature assembled onto the
tabular `raw` projection by the FJ2-pinned pure `build_continuous_state`.
"""
function get_continuous_state(world::DuckieWorldState, raw::RawState,
    state_cfg::StateConfig, continuous_cfg::ContinuousStateConfig;
    controller_cfg::Union{Nothing,DuckControllerConfig}=nothing,
    stop_hold_progress::Float64=0.0)
    duck = gate_duck_visibility(
        duck_relative_state(world, raw.v, controller_cfg), continuous_cfg)
    kappa = signed_curvature_ahead(world, state_cfg, continuous_cfg)
    return build_continuous_state(raw, kappa, duck, stop_hold_progress)
end
