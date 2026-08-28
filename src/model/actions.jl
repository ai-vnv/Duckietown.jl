"""
    ActionConfig

Speed/steering limits of the action space (`src/actions.py::ActionConfig`).
Defaults mirror Python; the experiment YAMLs set `v_fast 0.41`, `v_slow 0.17`.
"""
struct ActionConfig
    v_fast::Float64
    v_slow::Float64
    w0::Float64
    wheel_base::Float64
end

function ActionConfig(; v_fast=0.40, v_slow=0.15, w0=1.50, wheel_base=0.102)
    ActionConfig(Float64(v_fast), Float64(v_slow), Float64(w0), Float64(wheel_base))
end

"""
    MacroAction

The 7 discrete macro-actions for tabular solvers (index order fixed by
`build_action_table` in `src/actions.py`; exact table values are verified in
FJ2).

The mapping is `fast_left=(v_fast,+w0)`, `fast_straight=(v_fast,0)`,
`fast_right=(v_fast,-w0)`, `slow_left=(v_slow,+w0)`, `slow_straight=(v_slow,0)`,
`slow_right=(v_slow,-w0)`, `brake=(0,0)`.
"""
@enum MacroAction begin
    FAST_LEFT = 0
    FAST_STRAIGHT = 1
    FAST_RIGHT = 2
    SLOW_LEFT = 3
    SLOW_STRAIGHT = 4
    SLOW_RIGHT = 5
    BRAKE = 6
end

"""
    DuckieAction

Canonical continuous action `(v_cmd, ω_cmd)` in simulator command magnitudes
(m/s, rad/s). Discrete macro-actions project onto it; SAC/TD3 act in the Box
`[0, -w0] × [v_fast, w0]`.
"""
struct DuckieAction
    v::Float64
    omega::Float64
end

"""
    ActionSpec

One row of the discrete action table: `name`, commanded `v`, commanded `omega`
(`src/actions.py::ActionSpec`).
"""
struct ActionSpec
    name::String
    v::Float64
    omega::Float64
end

"""
    build_action_table(cfg=ActionConfig()) -> NTuple{7,ActionSpec}

`A = {fast/slow × left/straight/right, brake}` with the Python order
(`src/actions.py::build_action_table`): `fast_left, fast_straight, fast_right,
slow_left, slow_straight, slow_right, brake`. `left` means `+w0` (this is the
sign convention the reference policies were trained with).
"""
function build_action_table(cfg::ActionConfig=ActionConfig())
    return (
        ActionSpec("fast_left", cfg.v_fast, cfg.w0),
        ActionSpec("fast_straight", cfg.v_fast, 0.0),
        ActionSpec("fast_right", cfg.v_fast, -cfg.w0),
        ActionSpec("slow_left", cfg.v_slow, cfg.w0),
        ActionSpec("slow_straight", cfg.v_slow, 0.0),
        ActionSpec("slow_right", cfg.v_slow, -cfg.w0),
        ActionSpec("brake", 0.0, 0.0),
    )
end

"""
    vw_to_wheels(v, omega, wheel_base) -> NTuple{2,Float32}

Inverse differential-drive kinematics: `u_L = v - L·ω/2`, `u_R = v + L·ω/2`,
clipped to `±1` (`src/actions.py::vw_to_wheels`). Arithmetic is `Float64`,
rounded to `Float32` before clipping, exactly like
`np.clip(np.array([l, r], dtype=np.float32), -1, 1)`.
"""
function vw_to_wheels(v::Real, omega::Real, wheel_base::Real)
    left = Float32(v - 0.5 * Float64(wheel_base) * Float64(omega))
    right = Float32(v + 0.5 * Float64(wheel_base) * Float64(omega))
    return (clamp(left, -1.0f0, 1.0f0), clamp(right, -1.0f0, 1.0f0))
end

"""
    action_to_wheels(action_id::Integer, cfg=ActionConfig()) -> NTuple{2,Float32}

Look up macro-action `action_id ∈ 0:6` and convert it to clipped wheel
commands (`src/actions.py::action_to_wheels`). Out-of-range ids raise
`ArgumentError` (Python: `ValueError`).
"""
function action_to_wheels(action_id::Integer, cfg::ActionConfig=ActionConfig())
    if !(0 <= Int(action_id) < 7)
        throw(ArgumentError("action_id must be 0..6"))
    end
    action = build_action_table(cfg)[Int(action_id) + 1]
    return vw_to_wheels(action.v, action.omega, cfg.wheel_base)
end