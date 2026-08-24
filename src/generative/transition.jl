# FJ3.7: the full one-decision generative transition, verbatim from
# `duckduck/src/env_wrapper.py::DuckieMDPEnv.step` (discrete) and
# `duckduck/src/continuous_env.py::ContinuousDuckieMDPEnv.step` (continuous),
# with `render_observations = false` (`_simulator_step` non-render path,
# matching every reference training config).
#
# Locked order (README design constraint #4; do not reorder):
#   x_t
#     -> previous RawState.d_stop / stop id      (StopMemory)
#     -> DuckController.before_step              (external rng)
#     -> action -> (v_cmd, omega_cmd) -> wheels
#     -> frame_skip x [ego physics tick; every duck object tick]
#     -> simulator done (_valid_pose / max_steps, `_compute_done_reward`)
#     -> RawState extraction (pre-update sigma)
#     -> next stop candidate
#     -> StopTracker.update -> sigma
#     -> duck collision / any collision / goal -> reason -> events
#     -> terminated / truncated
#     -> reward (continuous: pre-action kappa + clipped omega_cmd)
#     -> x_{t+1}
#
# This is the canonical native transition; `POMDPs.gen` (FJ5) is a thin
# adapter `gen(...) -> (sp = result.sp, r = result.reward.total)` on top.

"""
    TerminationReason

Explicit single termination reason, in the reference resolution order
(`duck_collision > other_collision > timeout > offroad > goal >
in_progress`); an object collision is never conflated with going off-road.
"""
@enum TerminationReason begin
    DUCK_COLLISION
    OTHER_COLLISION
    TIMEOUT
    OFFROAD
    GOAL
    IN_PROGRESS
end

"""
    DuckieTransitionModel

Static parameters of the one-decision transition — everything
`DuckieMDPEnv`/`ContinuousDuckieMDPEnv` reads besides the world state:
action/state/reward/duck-controller/continuous configs, `frame_skip`,
`max_steps` (physics ticks, compared against `ego.step_count`), and the
optional `goal_tile`. Built from a [`DuckietownConfig`](@ref) so every
parameter traces back to a single experiment YAML.
"""
struct DuckieTransitionModel
    action_cfg::ActionConfig
    state_cfg::StateConfig
    reward_cfg::RewardConfig
    duck_cfg::DuckControllerConfig
    continuous_cfg::ContinuousStateConfig
    frame_skip::Int
    max_steps::Int
    goal_tile::Union{Nothing,NTuple{2,Int}}
end

function DuckieTransitionModel(cfg::DuckietownConfig)
    ccfg = cfg.continuous_state === nothing ? ContinuousStateConfig() :
        cfg.continuous_state
    return DuckieTransitionModel(cfg.actions, cfg.state, cfg.reward,
        cfg.duck_controller, ccfg, cfg.environment.frame_skip,
        cfg.environment.max_steps, cfg.environment.goal_tile)
end

"""
    TransitionResult

Rich output of one generative decision. `sp` is the successor world state;
everything else is the audit trail the wrapper's `info` dict carries:
solver-facing projections, the component-level reward, the event flags, the
`terminated`/`truncated` split (timeout is truncation, not an absorbing
state), the explicit reason, and the wheel commands actually applied
(Float32, post-clip).
"""
struct TransitionResult
    sp::DuckieWorldState
    raw_state::RawState
    continuous_state::ContinuousState
    reward::RewardBreakdown
    events::EventFlags
    terminated::Bool
    truncated::Bool
    reason::TerminationReason
    wheel_commands::NTuple{2,Float32}
end

"""
    _duck_collision(world) -> Bool

`env_wrapper._duck_collision`: SAT of the agent bounding box against every
visible duckie's shifted corners.
"""
function _duck_collision(world::DuckieWorldState)
    corners = get_agent_corners(collect(world.ego.pos), world.ego.angle)
    norms = generate_norm(corners)
    for d in world.ducks
        d.visible || continue
        if intersects_single_obj(corners, corner_matrix(d.obj_corners)',
            norms, d.obj_norm)
            return true
        end
    end
    return false
end

"""
    termination_reason(model, world) -> TerminationReason

The single explicit reason for `world`, in the reference resolution order
(`DuckieMDPEnv.step`). Every input is a function of the post-transition world
state alone — `simulator_done` (`Simulator._compute_done_reward`: invalid
pose, which already covers collisions and off-road, or the physics-tick
horizon), the duckie SAT test, the full collision test, and the goal tile —
so the same function serves the transition chain and
[`POMDPs.isterminal`](@ref); there is no second copy of this logic.
"""
function termination_reason(m::DuckieTransitionModel, w::DuckieWorldState)
    pos = collect(w.ego.pos)
    max_steps_flag = w.ego.step_count >= m.max_steps
    simulator_done = !_valid_pose(w.map, pos, w.ego.angle, 1.0, w.ducks) ||
        max_steps_flag
    if simulator_done
        _duck_collision(w) && return DUCK_COLLISION
        _collision(w.map, get_agent_corners(pos, w.ego.angle), w.ducks) &&
            return OTHER_COLLISION
        max_steps_flag && return TIMEOUT
        return OFFROAD
    end
    if m.goal_tile !== nothing && get_grid_coords(w.map, pos) == m.goal_tile
        return GOAL
    end
    return IN_PROGRESS
end

"""
    is_terminated(reason) -> Bool
    is_truncated(reason) -> Bool

The TD-bootstrapping split: a genuine terminal (`duck_collision`,
`other_collision`, `offroad`, `goal`) breaks the bootstrap; `timeout` is only
the experiment horizon (truncation), not an absorbing physical state.
"""
is_terminated(reason::TerminationReason) =
    reason in (DUCK_COLLISION, OTHER_COLLISION, OFFROAD, GOAL)
is_truncated(reason::TerminationReason) = reason == TIMEOUT

"""
    simulate_decision(model, s, action_id::Integer, rng) -> TransitionResult
    simulate_decision(model, s, action::DuckieAction, rng) -> TransitionResult

One full macro-decision of the Duckietown MDP from world state `s`
(`DuckieMDPEnv.step` for a discrete `action_id` in `0:6`,
`ContinuousDuckieMDPEnv.step` for a continuous `[v_cmd, omega_cmd]`).

`s` is never mutated: every step of the chain builds fresh state
(branch-pure), so a planner may call this repeatedly from the same `s` with
different actions. Stochasticity (the `p_cross` trigger draw) comes from the
external `rng` — the MDP semantics are `x' ~ T(.|x, a)` with the noise
supplied by the caller, not stored in the state.

Continuous actions are clipped to the reference `Box`
(`[0, v_fast] x [-w0, w0]`, Float32 like `np.clip` on the float32 command)
and the steering penalty uses the **pre-action** curvature of `s` with the
clipped `omega_cmd` (README design constraint #3).
"""
function simulate_decision(m::DuckieTransitionModel, s::DuckieWorldState,
    action_id::Integer, rng::AbstractRNG)
    wheels = action_to_wheels(action_id, m.action_cfg)
    return _decision_chain(m, s, wheels, rng, 0.0, nothing)
end

simulate_decision(m::DuckieTransitionModel, s::DuckieWorldState,
    action::MacroAction, rng::AbstractRNG) =
    simulate_decision(m, s, Int(action), rng)

function simulate_decision(m::DuckieTransitionModel, s::DuckieWorldState,
    action::DuckieAction, rng::AbstractRNG)
    # ContinuousDuckieMDPEnv.step: float32 command, np.clip to the Box
    v_cmd = Float64(clamp(Float32(action.v), 0.0f0, Float32(m.action_cfg.v_fast)))
    omega_cmd = Float64(clamp(Float32(action.omega),
        -Float32(m.action_cfg.w0), Float32(m.action_cfg.w0)))
    wheels = vw_to_wheels(v_cmd, omega_cmd, m.action_cfg.wheel_base)
    # pre-action kappa: the wrapper reads `self.current_state.kappa`, which was
    # extracted from this same latent state at the end of the previous decision
    kappa_pre = signed_curvature_ahead(s, m.state_cfg, m.continuous_cfg)
    return _decision_chain(m, s, wheels, rng, omega_cmd, kappa_pre)
end

function _decision_chain(m::DuckieTransitionModel, s::DuckieWorldState,
    wheels::NTuple{2,Float32}, rng::AbstractRNG,
    action_omega::Float64, kappa_pre::Union{Nothing,Float64})
    # previous decision's stop memory (wrapper `_last_state` / `_last_stop_id`;
    # StopTracker.update reads only previous.d_stop from the previous state)
    mem = s.stop_memory
    prev_raw = RawState(0.0, 0.0, 0.0, STRAIGHT, mem.last_d_stop,
        mem.sigma_stop, NONE)
    tracker = StopTracker(m.state_cfg.stop_zone, m.state_cfg.stop_speed,
        m.state_cfg.stop_pass_distance, m.state_cfg.stop_hold_steps)
    tracker.sigma_stop = mem.sigma_stop
    tracker.hold_steps = mem.hold_steps

    w = before_step(s, m.duck_cfg, rng)

    # `_simulator_step` non-render path: frame_skip x update_physics
    # (ego pose first, then every non-duckiebot object steps)
    wheels64 = (Float64(wheels[1]), Float64(wheels[2]))
    for _ in 1:m.frame_skip
        w = ego_tick(w, wheels64)
        for i in eachindex(w.ducks)
            w = duck_step(w, i)
        end
    end

    current, lane_fb = get_raw_state(w, m.state_cfg; sigma_stop=mem.sigma_stop)
    w.lane_fallback = lane_fb
    _, current_stop_id = next_stop_candidate(w, m.state_cfg)
    sigma, stop_events = update!(tracker, prev_raw, current,
        mem.last_stop_id, current_stop_id)
    current = RawState(current.d, current.phi, current.v, current.tile,
        current.d_stop, sigma, current.duck)

    reason = termination_reason(m, w)

    events = EventFlags(
        collision_duck=reason == DUCK_COLLISION,
        other_collision=reason == OTHER_COLLISION,
        offroad=reason == OFFROAD,
        timeout=reason == TIMEOUT,
        stop_violation=stop_events.stop_violation,
        full_stop=stop_events.full_stop,
        passed_stop=stop_events.passed_stop,
        goal=reason == GOAL,
    )
    terminated = is_terminated(reason)
    truncated = is_truncated(reason)

    reward = compute_reward(current, events, m.reward_cfg;
        action_omega=action_omega, curvature=kappa_pre)

    w.stop_memory = StopMemory(sigma, tracker.hold_steps, current_stop_id,
        current.d_stop)

    cont = get_continuous_state(w, current, m.state_cfg, m.continuous_cfg;
        controller_cfg=m.duck_cfg, stop_hold_progress=hold_progress(tracker))

    return TransitionResult(w, current, cont, reward, events,
        terminated, truncated, reason, wheels)
end
