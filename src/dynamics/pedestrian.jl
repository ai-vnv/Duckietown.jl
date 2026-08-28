# FJ3.4: DuckieObj walk/finish_walk/step parity (gym_duckietown 6.1.34
# `objects.py::DuckieObj`), FJ3.5: DuckController before_step trigger semantics
# (duckduck `src/duck_controller.py`).
#
# Per physics tick the reference `Simulator.update_physics` steps every
# non-duckiebot object: `obj.step(delta_time)`. `DuckieObj.step`:
#   1. `time += dt`;
#   2. inactive: `wait_time -= dt`; if `wait_time <= 0` -> active (the
#      DuckController pins `wait_time` to `Inf` before every decision, so this
#      auto-activation only fires without the controller);
#   3. active: `center += heading * vel`, corners shift by the (x, z) part;
#      if `norm(center - start) > walk_distance` -> `finish_walk`
#      (`start = center`, `angle += pi`, inactive, `vel *= -1`,
#      `wait_time = 8`); `pos = center`, `obj_norm = generate_norm(corners)`.
# The heading is NOT recomputed by finish_walk; the reversed walk comes from
# the flipped `vel` sign.
#
# `DuckController.before_step` (per decision, before the frame_skip ticks,
# observing the previous decision's ego pose) may activate each inactive,
# under-limit duck whose crossing midpoint is inside the trigger window and
# ahead of the ego (lane tangent from `closest_curve_point`); activation draws
# `rng < p_cross` (semantic parity; the exact NumPy MT19937 stream is FJ3.8).

"""
    replace_duck(world, i, duck) -> DuckieWorldState

Fresh world with `ducks[i]` replaced; every other field is shared (the
transition chain copies `ducks` before mutating, keeping branches pure).
"""
function replace_duck(world::DuckieWorldState, i::Int, duck::DuckieState)
    ducks = copy(world.ducks)
    ducks[i] = duck
    return DuckieWorldState(world.ego, ducks, world.stop_signs, world.map,
        world.stop_memory, world.lane_fallback, world.crossings_started,
        world.crossing_armed, world.controller_rng)
end

"""
    duck_step(world, i, dt=EGO_DT) -> DuckieWorldState

One physics tick of duck `i` (`DuckieObj.step`, verbatim). Returns a fresh
world state; inactive ducks only count down `wait_time`.
"""
function duck_step(world::DuckieWorldState, i::Int, dt::Float64=EGO_DT)
    d = world.ducks[i]
    time = d.time + dt
    if !d.pedestrian_active
        wait = d.pedestrian_wait_time - dt
        nd = DuckieState(d.pos, d.center, d.start, d.angle, d.heading, d.vel,
            d.visible, wait <= 0.0, wait, time, d.walk_distance, d.scale,
            d.safety_radius, d.min_coords, d.max_coords,
            copy(d.obj_corners), copy(d.obj_norm))
        return replace_duck(world, i, nd)
    end

    vx = d.heading[1] * d.vel
    vz = d.heading[3] * d.vel
    center = (d.center[1] + vx, d.center[2] + d.heading[2] * d.vel,
        d.center[3] + vz)
    corners = [(c[1] + vx, c[2] + vz) for c in d.obj_corners]
    distance = sqrt(sum(abs2, center .- d.start))

    if distance > d.walk_distance
        # finish_walk: reset origin, flip heading (angle += pi) and velocity
        nd = DuckieState(center, center, center, d.angle + pi, d.heading,
            -d.vel, d.visible, false, 8.0, time, d.walk_distance, d.scale,
            d.safety_radius, d.min_coords, d.max_coords, corners,
            generate_norm(corners_matrix(corners)))
    else
        nd = DuckieState(center, center, d.start, d.angle, d.heading, d.vel,
            d.visible, true, d.pedestrian_wait_time, time, d.walk_distance,
            d.scale, d.safety_radius, d.min_coords, d.max_coords, corners,
            generate_norm(corners_matrix(corners)))
    end
    return replace_duck(world, i, nd)
end

corners_matrix(corners::Vector{NTuple{2,Float64}}) =
    Matrix{Float64}(hcat([collect(c) for c in corners]...)')::Matrix{Float64}

"""
    activate_duck(world, i) -> DuckieWorldState

Controller activation: the duck starts walking (`time = 0`) and
`crossings_started[i]` increments (`DuckController.before_step` core).
"""
function activate_duck(world::DuckieWorldState, i::Int)
    d = world.ducks[i]
    nd = DuckieState(d.pos, d.center, d.start, d.angle, d.heading, d.vel,
        d.visible, true, d.pedestrian_wait_time, 0.0, d.walk_distance,
        d.scale, d.safety_radius, d.min_coords, d.max_coords,
        copy(d.obj_corners), copy(d.obj_norm))
    world2 = replace_duck(world, i, nd)
    cs = copy(world2.crossings_started)
    cs[i] += 1
    return DuckieWorldState(world2.ego, world2.ducks, world2.stop_signs,
        world2.map, world2.stop_memory, world2.lane_fallback, cs,
        world2.crossing_armed, world2.controller_rng)
end

"""
    before_step(world, cfg, rng) -> DuckieWorldState
    before_step(world, cfg)      -> DuckieWorldState

Decision-start trigger pass (`DuckController.before_step`, verbatim):
pins `wait_time` to `Inf` for every inactive duck, then for each inactive,
under-limit duck computes the crossing midpoint
`start + 0.5 * walk_distance * heading * sign(vel)` and activates it when its
distance to the ego is inside `[trigger_min, trigger_max]`, the lane tangent
at the ego pose says the crossing lies ahead, and the `p_cross` draw passes.
Re-arming follows `repeat_rearm_distance` (0 disarms the limit test only via
`max_crossings_per_episode`).

The 3-argument form draws from the **external** `rng` (mutating it) and never
touches `world.controller_rng` — this is the generative-MDP semantics used by
[`simulate_decision`](@ref): stochasticity is supplied by the caller, not
stored in the state. The RNG is consumed **only** on a fully eligible duck
(inactive, under limit, armed, in the trigger window, crossing ahead),
matching the reference call semantics draw-for-draw. The 2-argument form
keeps the FJ3.5 stream-in-state behaviour: it draws from a copy of
`world.controller_rng` and returns the advanced copy in the new world.
"""
function before_step(world::DuckieWorldState, cfg::DuckControllerConfig)
    rng = copy(world.controller_rng)
    w = before_step(world, cfg, rng)
    return DuckieWorldState(w.ego, w.ducks, w.stop_signs, w.map,
        w.stop_memory, w.lane_fallback, w.crossings_started, w.crossing_armed,
        rng)
end

function before_step(world::DuckieWorldState, cfg::DuckControllerConfig,
    rng::AbstractRNG)
    pos = world.ego.pos
    angle = world.ego.angle
    limit = cfg.max_crossings_per_episode
    rearm = cfg.repeat_rearm_distance
    ducks = copy(world.ducks)
    cs = copy(world.crossings_started)
    ca = copy(world.crossing_armed)

    for (i, d) in enumerate(ducks)
        if d.pedestrian_active
            continue
        end
        pinned = DuckieState(d.pos, d.center, d.start, d.angle, d.heading,
            d.vel, d.visible, false, Inf, d.time, d.walk_distance, d.scale,
            d.safety_radius, d.min_coords, d.max_coords,
            copy(d.obj_corners), copy(d.obj_norm))
        ducks[i] = pinned
        if limit > 0 && cs[i] >= limit
            continue
        end
        s = pinned.vel == 0.0 ? 1.0 : sign(pinned.vel)
        motion = pinned.heading .* s
        crossing = pinned.start .+ 0.5 .* pinned.walk_distance .* motion
        rel = crossing .- pos
        distance = hypot(rel[1], rel[3])
        if !ca[i]
            if rearm > 0.0 && distance >= rearm
                ca[i] = true
            else
                continue
            end
        end
        _, tangent = closest_curve_point(world.map, collect(pos), angle)
        ahead = tangent !== nothing && dot(collect(rel), tangent) > 0.0
        eligible = ahead && cfg.trigger_min_ego_distance <= distance <=
            cfg.trigger_max_ego_distance
        if cfg.spawn_on_ego_proximity && !pinned.visible
            if !eligible
                continue
            end
            visible_duck = DuckieState(pinned.pos, pinned.center, pinned.start,
                pinned.angle, pinned.heading, pinned.vel, true, false, Inf,
                pinned.time, pinned.walk_distance, pinned.scale,
                pinned.safety_radius, pinned.min_coords, pinned.max_coords,
                pinned.obj_corners, pinned.obj_norm)
            ducks[i] = visible_duck
            continue
        end
        if eligible && rand(rng) < cfg.p_cross
            activated = DuckieState(pinned.pos, pinned.center, pinned.start,
                pinned.angle, pinned.heading, pinned.vel, pinned.visible,
                true, Inf, 0.0, pinned.walk_distance, pinned.scale,
                pinned.safety_radius, pinned.min_coords, pinned.max_coords,
                pinned.obj_corners, pinned.obj_norm)
            ducks[i] = activated
            cs[i] += 1
            if rearm > 0.0
                ca[i] = false
            end
        end
    end
    return DuckieWorldState(world.ego, ducks, world.stop_signs, world.map,
        world.stop_memory, world.lane_fallback, cs, ca, world.controller_rng)
end