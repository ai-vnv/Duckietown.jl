# FJ3.2: DB18 nominal motor model + 0.15 s delayed dynamics
# (DuckietownWorld 6.4.3 `pwm_dynamics`, `dynamics_delay`, `generic_kinematics`,
# PyGeometry-z6 2.1.5 `poses.py` parity).
#
# The reference chain per physics tick (gym_duckietown `update_physics` ->
# `_update_pos` -> `DelayedDynamics.integrate` -> `DynamicModel.integrate`):
#   1. append (timestamp, action) to the delayed-command history;
#   2. `get_commands_at(t + dt - 0.15)` selects the applied command
#      (`bisect_left` + nearest-neighbour tie-break, `u0 = (0, 0)` before the
#      window);
#   3. trim the history to the delay window (`DelayedDynamics.__init__` slice
#      at `get_commands_at(timestamps[-1] - delay)`);
#   3. `model(commands, u_prev, w_prev)` -> body acceleration `x_dot_dot`;
#   4. forward-Euler body velocity `(v_long, omega)` (lateral pinned to 0),
#      wrapped as se(2);
#   5. SE(2): `q1 = q0 * exp(dt * se2)` (`SE2.group_from_algebra`, `multiply`);
#   6. wheel angular velocities from the differential-drive inverse map
#      `MInv @ [angular, longitudinal]`, integrated into axis radians;
#   7. world pose `(pos, angle)` via `weird_from_cartesian` (grid_height-scale).

const EGO_DELAY = 0.15
const EGO_DT = 1.0 / 30.0

# ---------------------------------------------------------------------------
# SE(2) primitives (PyGeometry-z6 2.1.5, `geometry/poses.py`, verbatim)
# ---------------------------------------------------------------------------

hat_map_2d(omega::Float64) = [0.0 -omega; omega 0.0]

"""
    se2_from_linear_angular(linear, angular) -> 3×3 Matrix

Body velocity as se(2) (`poses.se2_from_linear_angular`).
"""
function se2_from_linear_angular(linear::NTuple{2,Float64}, angular::Float64)
    M = hat_map_2d(angular)
    return [M [linear[1], linear[2]]; [0.0 0.0] 0.0]
end

"""
    SE2_from_translation_angle(t, theta) -> 3×3 Matrix

Pose from translation and rotation (`poses.SE2_from_translation_angle`).
"""
function SE2_from_translation_angle(t::NTuple{2,Float64}, theta::Float64)
    C = cos(theta)
    S = sin(theta)
    return [C -S t[1]; S C t[2]; 0.0 0.0 1.0]
end

"""
    SE2_from_se2(vel) -> 3×3 Matrix

Exponential map (`poses.SE2_from_se2`, Bullo/Murray; `|w| < 1e-8` branch).
"""
function SE2_from_se2(vel::AbstractMatrix{Float64})
    w = vel[2, 1]
    v = vel[1:2, 3]
    if abs(w) < 1e-8
        R = Matrix{Float64}(I, 2, 2)
        t = v
    else
        R = [cos(w) -sin(w); sin(w) cos(w)]
        A = [sin(w) cos(w) - 1.0; 1.0 - cos(w) sin(w)] ./ w
        t = A * v
    end
    return [R t; [0.0 0.0] 1.0]
end

"""
    se2_multiply(g, h) -> 3×3 Matrix

Matrix-group product (`SE2.multiply` = `np.dot(g, h)`).
"""
se2_multiply(g::AbstractMatrix{Float64}, h::AbstractMatrix{Float64}) = g * h

"""
    translation_angle_from_SE2(q) -> (t, angle)

(`poses.translation_angle_from_SE2`; `angle == pi` maps to `-pi` exactly.)
"""
function translation_angle_from_SE2(q::AbstractMatrix{Float64})
    t = (q[1, 3], q[2, 3])
    angle = atan(q[2, 1], q[1, 1])
    angle == pi && (angle = -pi)
    return t, angle
end

linear_angular_from_se2(vel::AbstractMatrix{Float64}) =
    (vel[1, 3], vel[2, 3]), vel[2, 1]

# ---------------------------------------------------------------------------
# DB18 nominal parameters (`pwm_dynamics.DynamicModelParameters` /
# `get_DB18_nominal`, verbatim)
# ---------------------------------------------------------------------------

"""
    DB18Parameters

Nominal DB18 platform model: autonomous response (u1..w3), forced response
gains (uar..wal), wheel radii R = 0.067/2, wheel distance D = 0.1, encoder
resolution 2π/135.
"""
struct DB18Parameters
    u1::Float64
    u2::Float64
    u3::Float64
    w1::Float64
    w2::Float64
    w3::Float64
    u_alpha_r::Float64
    u_alpha_l::Float64
    w_alpha_r::Float64
    w_alpha_l::Float64
    wheel_radius_left::Float64
    wheel_radius_right::Float64
    wheel_distance::Float64
    encoder_resolution_rad::Float64
end

function DB18Parameters(u1, u2, u3, w1, w2, w3, uar, ual, war, wal)
    R = 0.067 / 2
    D = 0.1
    ticks = 135
    res = (2 * pi) / ticks
    DB18Parameters(Float64(u1), Float64(u2), Float64(u3),
        Float64(w1), Float64(w2), Float64(w3),
        Float64(uar), Float64(ual), Float64(war), Float64(wal),
        R, R, D, res)
end

db18_nominal() = DB18Parameters(5, 0, 0, 4, 0, 0, 1.5, 1.5, 15, 15)

"""
    db18_model(p, commands, u, w) -> (x_ddot_long, x_ddot_ang)

Body acceleration from motor commands and previous (u = longitudinal,
w = angular) velocities (`DynamicModel.model`, verbatim). Commands are
`(motor_left, motor_right)`, each clipped to [-1, 1].
"""
function db18_model(p::DB18Parameters, commands::NTuple{2,Float64},
    u::Float64, w::Float64)
    U = (commands[2], commands[1])
    V = (clamp(U[1], -1.0, 1.0), clamp(U[2], -1.0, 1.0))
    f_dynamic = (-p.u1 * u - p.u2 * w + p.u3 * w^2,
        -p.w1 * w - p.w2 * u - p.w3 * u * w)
    B = [p.u_alpha_r p.u_alpha_l; p.w_alpha_r -p.w_alpha_l]
    f_forced = B * collect(V)
    return (f_dynamic[1] + f_forced[1], f_dynamic[2] + f_forced[2])
end

# ---------------------------------------------------------------------------
# Delayed command selection (`dynamics_delay.get_commands_at`, verbatim)
# ---------------------------------------------------------------------------

"""
    DelayedCommand(idx, told, used)

Selection result: 0-based `idx`, the timestamp at the decision boundary
(`told`), and the command `(motor_left, motor_right)` that gets applied.
"""
struct DelayedCommand
    idx::Int
    told::Float64
    used::NTuple{2,Float64}
end

"""
    get_commands_at(history, t) -> DelayedCommand

`bisect_left` on the command timestamps with the reference nearest-neighbour
tie-break: when the previous timestamp is strictly closer than the next, the
previous command is applied. Before the first timestamp, `u0 = (0, 0)`.
`history` entries are `(t, u_L, u_R)` in increasing `t` order.
"""
function get_commands_at(history::AbstractVector{Tuple{Float64,Float64,Float64}},
    t::Float64)
    if isempty(history) || t < history[1][1]
        return DelayedCommand(0, 0.0, (0.0, 0.0))
    end
    a = [h[1] for h in history]
    idx = searchsortedfirst(a, t) - 1
    idxp = idx + 1
    if idx > 0 && (idx == length(a) || abs(t - a[idx]) < abs(t - a[idxp]))
        return DelayedCommand(idx, a[idxp], (history[idx][2], history[idx][3]))
    else
        return DelayedCommand(idx, a[idxp], (history[idxp][2], history[idxp][3]))
    end
end

# ---------------------------------------------------------------------------
# World-frame wrapping (`simulator.cartesian_from_weird` /
# `weird_from_cartesian`, verbatim)
# ---------------------------------------------------------------------------

"""
    cartesian_from_weird(pos, angle, grid_height, tile_size) -> 3×3 Matrix

`pos = (x, y, z)` -> SE(2) pose with `cp = (x, grid_height * tile_size - z)`.
"""
function cartesian_from_weird(pos::NTuple{3,Float64}, angle::Float64,
    grid_height::Int, tile_size::Float64)
    cp = (pos[1], grid_height * tile_size - pos[3])
    return SE2_from_translation_angle(cp, angle)
end

"""
    weird_from_cartesian(q, grid_height, tile_size) -> ((x, 0, z), angle)

Inverse mapping; `angle == pi` is normalised to `-pi` inside
`translation_angle_from_SE2`.
"""
function weird_from_cartesian(q::AbstractMatrix{Float64},
    grid_height::Int, tile_size::Float64)
    t, angle = translation_angle_from_SE2(q)
    return (t[1], 0.0, grid_height * tile_size - t[2]), angle
end

# ---------------------------------------------------------------------------
# One physics tick (`DelayedDynamics.integrate` + `DynamicModel.integrate`,
# composed in reference order)
# ---------------------------------------------------------------------------

"""
    initial_ego(pos, angle, grid_height, tile_size) -> DuckieEgoState

Fresh ego at rest: zero se(2) velocity, zero axis radians, empty command
history, t0 = 0 (`Simulator.reset` state construction, verbatim).
"""
function initial_ego(pos::NTuple{3,Float64}, angle::Float64,
    grid_height::Int, tile_size::Float64)
    q0 = cartesian_from_weird(pos, angle, grid_height, tile_size)
    v0 = zeros(Float64, 3, 3)
    DuckieEgoState(pos, angle, 0.0, 0.0, 0.0, 0, 0.0,
        Tuple{Float64,Float64,Float64}[], q0, v0, 0.0, 0.0)
end

"""
    ego_tick(world, action) -> DuckieWorldState

One 1/30 s physics tick under `action = (motor_left, motor_right)` (clipped to
[-1, 1]); returns a fresh world state (no shared mutation, branch-pure).
"""
function ego_tick(world::DuckieWorldState, action::NTuple{2,Float64})
    e = world.ego
    map = world.map
    dt = EGO_DT
    p = db18_nominal()

    cmd = (clamp(action[1], -1.0, 1.0), clamp(action[2], -1.0, 1.0))
    history = vcat(e.command_history, [(e.timestamp, cmd[1], cmd[2])])
    t_new = e.timestamp + dt

    sel = get_commands_at(history, t_new - EGO_DELAY)

    # Reference `DelayedDynamics.__init__` trims the command window after each
    # tick: keep `history[it+1:end]` where `it` is the selection index for
    # `timestamps[-1] - delay` (the just-appended timestamp minus the delay).
    it = get_commands_at(history, e.timestamp - EGO_DELAY).idx
    history = history[it + 1:end]

    (longit_prev, _), angular_prev = linear_angular_from_se2(e.v0)
    xdd = db18_model(p, sel.used, longit_prev, angular_prev)
    longitudinal = longit_prev + dt * xdd[1]
    angular = angular_prev + dt * xdd[2]
    commands_se2 = se2_from_linear_angular((longitudinal, 0.0), angular)

    q1 = se2_multiply(e.q0, SE2_from_se2(dt .* commands_se2))

    d = p.wheel_distance
    Rr = p.wheel_radius_right
    Rl = p.wheel_radius_left
    M = [Rr / d -Rl / d; Rr / 2 Rl / 2]
    MInv = inv(M)
    wRL = MInv * [angular, longitudinal]
    wR = wRL[1]
    wL = wRL[2]

    axis_left_rad = e.axis_left_rad + wL * dt
    axis_right_rad = e.axis_right_rad + wR * dt

    pos, angle = weird_from_cartesian(q1, size(map.grid, 1), map.tile_size)

    # simulator.update_physics: self.speed = norm(cur_pos - prev_pos) / dt
    speed = sqrt(sum(abs2, pos .- e.pos)) / dt

    return DuckieWorldState(
        DuckieEgoState(pos, angle, longitudinal, angular, speed,
            e.step_count + 1, t_new, history,
            q1, commands_se2, axis_left_rad, axis_right_rad),
        world.ducks, world.stop_signs, map, world.stop_memory,
        world.lane_fallback, world.crossings_started, world.crossing_armed,
        world.controller_rng,
    )
end

"""
    axis_observed_ticks(axis_rad, p) -> (left, right)

Encoder ticks `int(round(axis_rad / resolution))` (Python `round` =
round-half-to-even), used for wheel-speed observations.
"""
function axis_observed_ticks(axis_rad::Float64, p::DB18Parameters)
    return Int(round(axis_rad / p.encoder_resolution_rad))
end
