"""
    TileType

Curvature class of the directed lane ahead of the ego, as computed by
`_ego_relative_curve` in the Python `src/state.py`. Enum values match the
Python integer values used in `raw_state_to_dict`.
"""
@enum TileType begin
    STRAIGHT = 0
    CURVE_LEFT = 1
    CURVE_RIGHT = 2
end

"""
    DuckThreat

Pedestrian threat class reported by `classify_duck` (Python `src/state.py`).
Values match the Python enum integers.
"""
@enum DuckThreat begin
    NONE = 0
    SIDE_FAR = 1
    SIDE_NEAR = 2
    CROSSING_FAR = 3
    CROSSING_NEAR = 4
end

"""
    StateConfig

Parameters of the 7-component raw-state extraction. Defaults mirror
`src/state.py::StateConfig` exactly (including `duck_max_distance = 2.0` and
`duck_corridor_width = 0.35`, which the four experiment YAMLs override to
`1.20`/`0.60`).
"""
struct StateConfig
    stop_lateral_limit::Float64
    stop_orientation_cos::Float64
    sign_to_line_offset::Float64
    stop_max_distance::Float64
    stop_zone::Float64
    stop_pass_distance::Float64
    stop_speed::Float64
    stop_hold_steps::Int
    tile_lookahead::Float64
    curvature_threshold::Float64
    duck_max_distance::Float64
    duck_near_distance::Float64
    duck_corridor_width::Float64
end

function StateConfig(;
    stop_lateral_limit=0.40,
    stop_orientation_cos=0.70710678,
    sign_to_line_offset=0.20,
    stop_max_distance=3.0,
    stop_zone=0.45,
    stop_pass_distance=0.55,
    stop_speed=0.02,
    stop_hold_steps=1,
    tile_lookahead=0.30,
    curvature_threshold=0.05,
    duck_max_distance=2.0,
    duck_near_distance=0.60,
    duck_corridor_width=0.35,
)
    StateConfig(
        Float64(stop_lateral_limit),
        Float64(stop_orientation_cos),
        Float64(sign_to_line_offset),
        Float64(stop_max_distance),
        Float64(stop_zone),
        Float64(stop_pass_distance),
        Float64(stop_speed),
        Int(stop_hold_steps),
        Float64(tile_lookahead),
        Float64(curvature_threshold),
        Float64(duck_max_distance),
        Float64(duck_near_distance),
        Float64(duck_corridor_width),
    )
end

"""
    RawState

7-component lane-relative state `(d, phi, v, tile, d_stop, sigma_stop, duck)`
extracted from the latent simulator state (`src/state.py::get_raw_state`).

- `d`: signed lane offset, clipped to ±0.25 m (right of lane negative).
- `phi`: signed heading error, clipped to ±π/2 (right of tangent negative).
- `v`: max(0, ego speed) m/s.
- `tile`: curvature class of the directed look-ahead tile.
- `d_stop`: distance to the nearest valid stop line (`ahead - sign_to_line_offset`,
  clipped at 0), or `nothing` when no sign passes the orientation/lateral filters.
- `sigma_stop`: stop-compliance memory bit.
- `duck`: maximum threat over visible duckies in the forward corridor.
"""
struct RawState
    d::Float64
    phi::Float64
    v::Float64
    tile::TileType
    d_stop::Union{Nothing,Float64}
    sigma_stop::Bool
    duck::DuckThreat
end