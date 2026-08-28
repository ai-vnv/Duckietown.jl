"""
    OBSERVATION_NAMES

The 15 privileged continuous-state features, in order. The 15th feature
(`stop_hold_progress`) was appended later and is the append-only element of
the feature list (`src/continuous_state.py::OBSERVATION_NAMES`).
"""
const OBSERVATION_NAMES = (
    "d",
    "phi",
    "v",
    "kappa",
    "stop_present",
    "d_stop",
    "sigma_stop",
    "duck_present",
    "duck_longitudinal",
    "duck_lateral",
    "duck_v_longitudinal_relative",
    "duck_v_lateral_relative",
    "duck_active",
    "duck_crossing_available",
    "stop_hold_progress",
)

"""
    ContinuousStateConfig

Normalization and detection-gate parameters for the continuous state
(`src/continuous_state.py::ContinuousStateConfig`). The three
`duck_detection_*` fields default to disabled (older SAC behaviour); the
SAC/TD3 experiment YAMLs enable the gate (`range 1.20`, `corridor 0.60`,
`forward_only true`).
"""
struct ContinuousStateConfig
    max_speed::Float64
    max_abs_curvature::Float64
    max_stop_distance::Float64
    max_duck_distance::Float64
    max_relative_speed::Float64
    curvature_samples::Int
    duck_detection_range::Union{Nothing,Float64}
    duck_detection_corridor_width::Union{Nothing,Float64}
    duck_detection_forward_only::Bool
end

function ContinuousStateConfig(;
    max_speed=0.41,
    max_abs_curvature=8.0,
    max_stop_distance=3.0,
    max_duck_distance=2.0,
    max_relative_speed=0.50,
    curvature_samples=33,
    duck_detection_range=nothing,
    duck_detection_corridor_width=nothing,
    duck_detection_forward_only=false,
)
    ContinuousStateConfig(
        Float64(max_speed),
        Float64(max_abs_curvature),
        Float64(max_stop_distance),
        Float64(max_duck_distance),
        Float64(max_relative_speed),
        Int(curvature_samples),
        duck_detection_range === nothing ? nothing : Float64(duck_detection_range),
        duck_detection_corridor_width === nothing ? nothing : Float64(duck_detection_corridor_width),
        Bool(duck_detection_forward_only),
    )
end

"""
    DuckRelativeState

Geometry of the nearest duckie in the ego lane frame
(`src/continuous_state.py::DuckRelativeState`).
"""
struct DuckRelativeState
    present::Bool
    longitudinal::Float64
    lateral::Float64
    v_longitudinal_relative::Float64
    v_lateral_relative::Float64
    active::Bool
    crossing_available::Bool
end

DuckRelativeState() = DuckRelativeState(false, 0.0, 0.0, 0.0, 0.0, false, false)

"""
    ContinuousState

15-component privileged state `(d, phi, v, kappa, stop_present, d_stop,
sigma_stop, duck_present, duck_longitudinal, duck_lateral,
duck_v_longitudinal_relative, duck_v_lateral_relative, duck_active,
duck_crossing_available, stop_hold_progress)` used by SAC/TD3
(`src/continuous_state.py::ContinuousState`).

`stop_hold_progress` is a *feature* of the stop tracker's dwell counter; the
canonical dwell memory lives in [`DuckieWorldState`](@ref).
"""
struct ContinuousState
    d::Float64
    phi::Float64
    v::Float64
    kappa::Float64
    stop_present::Bool
    d_stop::Union{Nothing,Float64}
    sigma_stop::Bool
    duck_present::Bool
    duck_longitudinal::Float64
    duck_lateral::Float64
    duck_v_longitudinal_relative::Float64
    duck_v_lateral_relative::Float64
    duck_active::Bool
    duck_crossing_available::Bool
    stop_hold_progress::Float64
end

ContinuousState(d, phi, v, kappa, stop_present, d_stop, sigma_stop, duck_present,
    duck_longitudinal, duck_lateral, duck_v_longitudinal_relative,
    duck_v_lateral_relative, duck_active, duck_crossing_available) =
    ContinuousState(d, phi, v, kappa, stop_present, d_stop, sigma_stop,
        duck_present, duck_longitudinal, duck_lateral,
        duck_v_longitudinal_relative, duck_v_lateral_relative, duck_active,
        duck_crossing_available, 0.0)