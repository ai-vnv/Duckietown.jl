"""
    RewardConfig

All reward coefficients and thresholds (`src/reward.py::RewardConfig`).
Defaults mirror Python exactly. The four experiment YAMLs override a subset;
anything absent from the YAML falls back to these Python defaults (this is the
authoritative config hierarchy: YAML > source defaults, with defaults applied
per missing key, exactly as `RewardConfig(**config["reward"])` behaves).

Terms (evaluated on the *post-transition* state, except the continuous
`steering` term which uses the *pre-action* `kappa` and the clipped `ω_cmd`;
see `docs/validation/FJ1_STATUS.md`):
- dense: `progress = α_p·v·cos(phi)`, `lateral = -α_d·d²`,
  `heading = -α_φ·phi²`, `time = -c_step`
- pedestrian: `duck_yield` if `v < duck_yield_speed` during a crossing else
  `duck_unsafe`
- stagnation: `unnecessary_stop` if `v < idle_speed` and not crossing and not
  must-stop
- stop approach: `stop_approach_yield`/`stop_approach_unsafe` within
  `stop_approach_distance` of an unmet stop (disabled at distance 0)
- steering: `-|straight_steer_penalty|·min(1,|ω|/max_steer_command)²` on
  straight segments
- events: collision/offroad/stop_violation/full_stop/goal coefficients
"""
struct RewardConfig
    alpha_progress::Float64
    alpha_lateral::Float64
    alpha_heading::Float64
    step_cost::Float64
    collision_duck::Float64
    other_collision::Float64
    offroad::Float64
    stop_violation::Float64
    full_stop::Float64
    duck_yield::Float64
    duck_unsafe::Float64
    duck_yield_speed::Float64
    unnecessary_stop::Float64
    idle_speed::Float64
    stop_exemption_distance::Float64
    stop_approach_distance::Float64
    stop_approach_speed::Float64
    stop_approach_yield::Float64
    stop_approach_unsafe::Float64
    straight_steer_penalty::Float64
    straight_curvature_threshold::Float64
    max_steer_command::Float64
    goal::Float64
end

function RewardConfig(;
    alpha_progress=1.0,
    alpha_lateral=10.0,
    alpha_heading=2.0,
    step_cost=0.01,
    collision_duck=-100.0,
    other_collision=-50.0,
    offroad=-50.0,
    stop_violation=-20.0,
    full_stop=10.0,
    duck_yield=0.0,
    duck_unsafe=0.0,
    duck_yield_speed=0.04,
    unnecessary_stop=0.0,
    idle_speed=0.04,
    stop_exemption_distance=0.45,
    stop_approach_distance=0.0,
    stop_approach_speed=0.02,
    stop_approach_yield=0.0,
    stop_approach_unsafe=0.0,
    straight_steer_penalty=0.0,
    straight_curvature_threshold=0.05,
    max_steer_command=1.5,
    goal=50.0,
)
    RewardConfig(
        Float64(alpha_progress), Float64(alpha_lateral), Float64(alpha_heading),
        Float64(step_cost), Float64(collision_duck), Float64(other_collision),
        Float64(offroad), Float64(stop_violation), Float64(full_stop),
        Float64(duck_yield), Float64(duck_unsafe), Float64(duck_yield_speed),
        Float64(unnecessary_stop), Float64(idle_speed),
        Float64(stop_exemption_distance), Float64(stop_approach_distance),
        Float64(stop_approach_speed), Float64(stop_approach_yield),
        Float64(stop_approach_unsafe), Float64(straight_steer_penalty),
        Float64(straight_curvature_threshold), Float64(max_steer_command),
        Float64(goal),
    )
end

"""
    RewardBreakdown

Per-component reward decomposition (`src/reward.py::RewardBreakdown`) in the
same field order as the Python dataclass: progress, lateral, heading, time,
pedestrian, stagnation, stop_approach, steering, events, total.
"""
struct RewardBreakdown
    progress::Float64
    lateral::Float64
    heading::Float64
    time::Float64
    pedestrian::Float64
    stagnation::Float64
    stop_approach::Float64
    steering::Float64
    events::Float64
    total::Float64
end

"""
    compute_reward(state, events, cfg=RewardConfig(); action_omega=0.0, curvature=nothing) -> RewardBreakdown

Dense per-decision reward `R(s, a, s')` with per-component decomposition
(`src/reward.py::compute_reward`, exact term order and arithmetic):

    r = α_p·v·cos(phi) - α_d·d² - α_φ·phi² - c_step + r_event

- `pedestrian`: during a crossing (`duck ∈ {CROSSING_FAR, CROSSING_NEAR}`),
  `duck_yield` if `v < duck_yield_speed` else `duck_unsafe`.
- `stagnation`: `unnecessary_stop` if `v < idle_speed` and not crossing and
  not within the stop-hold zone of an unmet stop.
- `stop_approach`: within `stop_approach_distance` of an unmet stop,
  `stop_approach_yield` if `v < stop_approach_speed` else
  `stop_approach_unsafe` (disabled at distance 0, preserving every older
  baseline).
- `steering`: on straight geometry (`|curvature| ≤ straight_curvature_threshold`,
  `curvature` supplied), `-|penalty|·min(1, |ω_cmd|/max_steer_command)²`. The
  curvature is the *pre-action* `kappa` from `s_t` and `action_omega` the
  clipped command — this keeps the term in `R(s, a, s')` form and must never
  read `sp.kappa` (locked FJ1 constraint 3).
- `events`: linear combination of the discrete event flags with their
  coefficients, in Python's addition order.

All arithmetic is `Float64`, matching NumPy.
"""
function compute_reward(state::RawState, events::EventFlags,
    cfg::RewardConfig=RewardConfig();
    action_omega::Float64=0.0, curvature::Union{Nothing,Float64}=nothing)
    progress = cfg.alpha_progress * state.v * cos(state.phi)
    lateral = -cfg.alpha_lateral * state.d ^ 2
    heading = -cfg.alpha_heading * state.phi ^ 2
    time = -cfg.step_cost
    crossing = state.duck === CROSSING_FAR ||
        state.duck === CROSSING_NEAR
    pedestrian = crossing ?
        (state.v < cfg.duck_yield_speed ? cfg.duck_yield : cfg.duck_unsafe) : 0.0
    stop_hold_zone = max(cfg.stop_exemption_distance, cfg.stop_approach_distance)
    must_stop = state.d_stop !== nothing &&
        state.d_stop <= stop_hold_zone && !state.sigma_stop
    unnecessary_idle = state.v < cfg.idle_speed && !crossing && !must_stop
    stagnation = unnecessary_idle ? cfg.unnecessary_stop : 0.0
    stop_approach = 0.0
    if cfg.stop_approach_distance > 0.0 && state.d_stop !== nothing &&
        state.d_stop <= cfg.stop_approach_distance && !state.sigma_stop
        stop_approach = state.v < cfg.stop_approach_speed ?
            cfg.stop_approach_yield : cfg.stop_approach_unsafe
    end
    steering = 0.0
    if curvature !== nothing && abs(curvature) <= cfg.straight_curvature_threshold &&
        cfg.straight_steer_penalty != 0.0
        steer_scale = max(abs(cfg.max_steer_command), 1e-9)
        normalized_steer = min(1.0, abs(action_omega) / steer_scale)
        steering = -abs(cfg.straight_steer_penalty) * normalized_steer ^ 2
    end
    event = cfg.collision_duck * Float64(events.collision_duck) +
        cfg.other_collision * Float64(events.other_collision) +
        cfg.offroad * Float64(events.offroad) +
        cfg.stop_violation * Float64(events.stop_violation) +
        cfg.full_stop * Float64(events.full_stop) +
        cfg.goal * Float64(events.goal)
    total = progress + lateral + heading + time + pedestrian + stagnation +
        stop_approach + steering + event
    return RewardBreakdown(progress, lateral, heading, time, pedestrian,
        stagnation, stop_approach, steering, event, total)
end