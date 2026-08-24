"""
    classify_tile(drivable, kind) -> TileType

Classify a tile by its map `kind` (`src/state.py::classify_tile`): any
`straight`/`3way*`/`4way` tile is `STRAIGHT`; `curve_left`/`curve_right` map
directly; anything else (or a non-drivable tile) raises `ArgumentError`
(Python: `ValueError`). `kind` is compared case-insensitively.
"""
function classify_tile(drivable::Bool, kind::AbstractString)
    drivable || throw(ArgumentError("Not on a drivable tile"))
    k = lowercase(String(kind))
    if k == "curve_left"
        return CURVE_LEFT
    elseif k == "curve_right"
        return CURVE_RIGHT
    elseif k == "straight" || startswith(k, "3way") || k == "4way"
        return STRAIGHT
    end
    throw(ArgumentError("Unsupported tile kind: $k"))
end

classify_tile(::Nothing) = throw(ArgumentError("Not on a drivable tile"))

"""
    ego_relative_curve(curves, forward, threshold) -> TileType

Turn a tile's directed Bezier curves into the `kappa_t` class correct for the
ego's approach (`src/state.py::_ego_relative_curve`). A `curve_left` tile can
carry two lanes turning in opposite directions; the sign of the y-component of
the cross product of the tangent at `t = 0.10` and `t = 0.90` of the directed
curve (the one best aligned with `forward`) decides left vs right.
"""
function ego_relative_curve(curves::AbstractVector{<:AbstractMatrix{Float64}},
    forward::AbstractVector{Float64}, threshold::Float64)
    headings = [_normalize_gt0(c[end, :] .- c[1, :]) for c in curves]
    curve = curves[argmax([dot(h, forward) for h in headings])]
    tangent_before = _normalize_gt0(bezier_tangent(curve, 0.10))
    tangent_after = _normalize_gt0(bezier_tangent(curve, 0.90))
    signed_turn = tangent_before[3] * tangent_after[1] -
        tangent_before[1] * tangent_after[3]
    if signed_turn > threshold
        return CURVE_LEFT
    elseif signed_turn < -threshold
        return CURVE_RIGHT
    end
    return STRAIGHT
end

ego_relative_curve(curves::AbstractVector, forward::AbstractVector{Float64},
    threshold::Float64) =
    ego_relative_curve([curve_matrix(c) for c in curves], forward, threshold)

"""
    _normalize_gt0(vector) -> Vector{Float64}

`src/state.py::_normalize`: divide by the Euclidean norm if it is `> 0`,
else return the vector unchanged (unlike the 1e-12 variant used in
`continuous_state.py`, which zeroes the vector instead).
"""
function _normalize_gt0(vector::AbstractVector{Float64})
    n = norm(vector)
    return n > 0.0 ? vector ./ n : Vector{Float64}(vector)
end

"""
    terminal_lane_fallback(last_d, last_phi) -> (d, phi)

Fallback lane position once the ego leaves the lane
(`src/state.py::_terminal_lane_fallback`): a `0.25`-offset `d` and `π/2`
heading error, with signs taken from the last valid lane position stored by
`get_raw_state` (`_mdp_last_lane_position`, default `(1.0, 1.0)`); a
zero-valued reference falls back to sign `+1` via Python's `or 1.0`.
"""
function terminal_lane_fallback(last_d::Real, last_phi::Real)
    d_reference = abs(last_d) > 1e-9 ? last_d : last_phi
    phi_reference = abs(last_phi) > 1e-9 ? last_phi : last_d
    d_sign = d_reference == 0.0 ? 1.0 : Float64(d_reference)
    phi_sign = phi_reference == 0.0 ? 1.0 : Float64(phi_reference)
    return (copysign(0.25, d_sign), copysign(pi / 2, phi_sign))
end

# FJ3: next_stop_candidate, classify_duck, tile_ahead (env-dependent).