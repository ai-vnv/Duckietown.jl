"""
    gate_duck_visibility(duck, cfg) -> DuckRelativeState

Apply the optional duckie detection gate (`src/continuous_state.py`): with the
gate disabled the nearest duckie is always reported at any distance; with it
enabled, a duckie outside the forward corridor, out of range, or behind the
ego is reported as absent, giving information parity with the tabular
`classify_duck`.
"""
function gate_duck_visibility(duck::DuckRelativeState, cfg::ContinuousStateConfig)
    duck.present || return duck
    if cfg.duck_detection_forward_only && duck.longitudinal < 0.0
        return DuckRelativeState()
    end
    if cfg.duck_detection_corridor_width !== nothing &&
        abs(duck.lateral) > cfg.duck_detection_corridor_width
        return DuckRelativeState()
    end
    if cfg.duck_detection_range !== nothing
        distance = hypot(duck.longitudinal, duck.lateral)
        if distance > cfg.duck_detection_range
            return DuckRelativeState()
        end
    end
    return duck
end

"""
    build_continuous_state(raw, kappa, duck, stop_hold_progress=0.0) -> ContinuousState

Assemble the 15-component continuous state from the tabular [`RawState`](@ref)
projection, the signed look-ahead curvature and the (already gated) duckie
relative state (`src/continuous_state.py::build_continuous_state`).

The two env-dependent inputs — `duck` (from `duck_relative_state`) and
`kappa` (from `signed_curvature_ahead`) — are produced by the FJ3 dynamics;
this pure assembly is what FJ2 pins.
"""
function build_continuous_state(raw::RawState, kappa::Float64,
    duck::DuckRelativeState, stop_hold_progress::Float64=0.0)
    return ContinuousState(
        raw.d,
        raw.phi,
        raw.v,
        kappa,
        raw.d_stop !== nothing,
        raw.d_stop,
        raw.sigma_stop,
        duck.present,
        duck.longitudinal,
        duck.lateral,
        duck.v_longitudinal_relative,
        duck.v_lateral_relative,
        duck.active,
        duck.crossing_available,
        clamp(Float64(stop_hold_progress), 0.0, 1.0),
    )
end

"""
    encode_continuous_state(state, cfg) -> Vector{Float32}

Normalize a [`ContinuousState`](@ref) into the 15-D SAC/TD3 observation vector
(`src/continuous_state.py::encode_continuous_state`). Arithmetic is performed
in `Float64` and rounded to `Float32` exactly as NumPy's
`np.array(..., dtype=np.float32)`. Non-finite observations raise
`ArgumentError` (Python: `ValueError`).
"""
function encode_continuous_state(state::ContinuousState, cfg::ContinuousStateConfig)
    stop_distance = !state.stop_present || state.d_stop === nothing ? 1.0 :
        clamp(state.d_stop / cfg.max_stop_distance, 0.0, 1.0)
    if state.duck_present
        duck_longitudinal = clamp(state.duck_longitudinal / cfg.max_duck_distance, -1.0, 1.0)
        duck_lateral = clamp(state.duck_lateral / cfg.max_duck_distance, -1.0, 1.0)
        duck_v_longitudinal = clamp(state.duck_v_longitudinal_relative / cfg.max_relative_speed, -1.0, 1.0)
        duck_v_lateral = clamp(state.duck_v_lateral_relative / cfg.max_relative_speed, -1.0, 1.0)
    else
        duck_longitudinal, duck_lateral = 1.0, 0.0
        duck_v_longitudinal, duck_v_lateral = 0.0, 0.0
    end
    values = Float32[
        clamp(state.d / 0.25, -1.0, 1.0),
        clamp(state.phi / (pi / 2.0), -1.0, 1.0),
        clamp(state.v / cfg.max_speed, 0.0, 1.0),
        clamp(state.kappa / cfg.max_abs_curvature, -1.0, 1.0),
        Float32(state.stop_present),
        Float32(stop_distance),
        Float32(state.sigma_stop),
        Float32(state.duck_present),
        Float32(duck_longitudinal),
        Float32(duck_lateral),
        Float32(duck_v_longitudinal),
        Float32(duck_v_lateral),
        Float32(state.duck_active),
        Float32(state.duck_crossing_available),
        Float32(clamp(state.stop_hold_progress, 0.0, 1.0)),
    ]
    if !all(isfinite, values)
        throw(ArgumentError("Continuous observation contains non-finite values"))
    end
    return values
end

"""
    continuous_observation_space() -> (low, high)

Normalization bounds of the 15-D observation (`src/continuous_state.py`):
`low = [-1,-1,0,-1,0,0,0,0,-1,-1,-1,-1,0,0,0]`, `high = 1` everywhere
(`Float32`). Returned as `(low, high)` — the Python wrapper is
`gym.spaces.Box(low, high, dtype=np.float32)`.
"""
function continuous_observation_space()
    low = Float32[-1, -1, 0, -1, 0, 0, 0, 0, -1, -1, -1, -1, 0, 0, 0]
    high = fill(Float32(1), 15)
    return low, high
end

# FJ3: duck_relative_state and signed_curvature_ahead (env-dependent geometry).