import LinearAlgebra: norm, dot, cross

"""
    bezier_point(curve, t) -> Vector{Float64}

Cubic Bezier point at parameter `t ∈ [0, 1]` for a `4×3` control-point matrix
(rows = points, columns = x,y,z). Verbatim arithmetic port of
`gym_duckietown.graphics.bezier_point` (pinned duckietown-gym-daffy-6.1.34):
the same term order, so results are bit-identical for identical inputs
(FJ2 parity fixture).
"""
function bezier_point(curve::AbstractMatrix{Float64}, t::Real)
    p = ((1 - t) ^ 3) .* curve[1, :]
    p += 3t .* ((1 - t) ^ 2) .* curve[2, :]
    p += 3 * (t ^ 2) * (1 - t) .* curve[3, :]
    p += (t ^ 3) .* curve[4, :]
    return p
end

"""
    bezier_tangent(curve, t) -> Vector{Float64}

Unit tangent at parameter `t` (first derivative of the cubic Bezier), ported
verbatim from `gym_duckietown.graphics.bezier_tangent` — including the
unprotected division by the norm: a zero-length segment yields `NaN` entries,
exactly as NumPy's `p /= norm` does (IEEE semantics).
"""
function bezier_tangent(curve::AbstractMatrix{Float64}, t::Real)
    p = 3 .* ((1 - t) ^ 2) .* (curve[2, :] - curve[1, :])
    p += 6 .* (1 - t) .* t .* (curve[3, :] - curve[2, :])
    p += 3 .* (t ^ 2) .* (curve[4, :] - curve[3, :])
    norm_p = norm(p)
    return p ./ norm_p
end

"""
    curve_matrix(curve) -> Matrix{Float64}

Convert one `Vector{NTuple{3,Float64}}` control polygon (the form stored in
`TileSpec.curves`) into the `4×3` matrix the Bezier helpers expect.
"""
curve_matrix(curve::AbstractVector{<:NTuple{3,Float64}}) =
    Matrix{Float64}(reshape(reinterpret(Float64, collect(curve)), 3, 4)')

"""
    _normalize_1e12(vector) -> Vector{Float64}

`src/continuous_state.py::_normalize`: divide by the Euclidean norm if it is
`> 1e-12`, else return a zero vector (stricter than the `> 0` variant used in
`state.py`; both are ported faithfully).
"""
function _normalize_1e12(vector::AbstractVector{Float64})
    n = norm(vector)
    return n > 1e-12 ? vector ./ n : zeros(Float64, 3)
end

"""
    curve_signed_curvature(curve, samples=33, straight_angle_threshold=0.05) -> Float64

Average signed curvature of a directed Bezier lane: heading change between
`t = 0.05` and `t = 0.95` divided by the arc length of `samples` sampled
points (`src/continuous_state.py::curve_signed_curvature`). The sign is the
y-component of the tangent cross product (left turns positive). Headings with
`|Δ| ≤ straight_angle_threshold` are `0.0` (straight); zero arc length yields
`0.0`; `samples < 3` raises `ArgumentError` (Python: `ValueError`).
NaN propagation from degenerate control polygons is IEEE-identical.
"""
function curve_signed_curvature(curve::AbstractMatrix{Float64};
    samples::Int=33, straight_angle_threshold::Float64=0.05)
    samples < 3 && throw(ArgumentError("curvature_samples must be at least 3"))
    tangent_before = _normalize_1e12(bezier_tangent(curve, 0.05))
    tangent_after = _normalize_1e12(bezier_tangent(curve, 0.95))
    cross_y = tangent_before[3] * tangent_after[1] -
        tangent_before[1] * tangent_after[3]
    dot_value = clamp(dot(tangent_before, tangent_after), -1.0, 1.0)
    heading_change = atan(cross_y, dot_value)
    if abs(heading_change) <= straight_angle_threshold
        return 0.0
    end
    points = [bezier_point(curve, t) for t in range(0.0, 1.0; length=samples)]
    arc_length = sum(norm(points[i + 1] - points[i]) for i in 1:(samples - 1))
    return arc_length > 1e-9 ? heading_change / arc_length : 0.0
end

curve_signed_curvature(curve::AbstractVector{<:NTuple{3,Float64}};
    kwargs...) = curve_signed_curvature(curve_matrix(curve); kwargs...)

# FJ3.1: bezier_closest, closest_curve_point, get_lane_pos2 (d, phi),
# lane-frame helpers. Verbatim ports of `graphics.bezier_closest` and
# `simulator.closest_curve_point` / `get_lane_pos2`.

"""
    NotInLane

Raised by [`get_lane_pos2`](@ref) when the agent is not in a lane
(`simulator.NotInLane`). Callers that are not on a drivable tile get
`nothing` from [`closest_curve_point`](@ref) instead.
"""
struct NotInLane <: Exception
    msg::String
end

NotInLane(msg::AbstractString) = NotInLane(String(msg))

"""
    LanePosition

Lane-relative pose (`simulator.LanePosition`): `dist` signed lateral offset
(right negative), `dot_dir` clipped heading·tangent dot product,
`angle_deg`/`angle_rad` signed heading error (right negative).
"""
struct LanePosition
    dist::Float64
    dot_dir::Float64
    angle_deg::Float64
    angle_rad::Float64
end

"""
    get_dir_vec(angle) -> NTuple{3,Float64}

Forward vector `(cos, 0, -sin)` (`simulator.get_dir_vec`, verbatim).
"""
get_dir_vec(angle::Float64) = (cos(angle), 0.0, -sin(angle))

"""
    get_right_vec(angle) -> NTuple{3,Float64}

Right vector `(sin, 0, cos)` (`simulator.get_right_vec`, verbatim).
"""
get_right_vec(angle::Float64) = (sin(angle), 0.0, cos(angle))

"""
    bezier_closest(curve, p, t_bot=0.0, t_top=1.0, n=8) -> Float64

Recursive midpoint search for the parameter of the closest point on a cubic
Bezier (`graphics.bezier_closest`, verbatim: bisection depth 8, `d_bot < d_top`
keeps the lower half).
"""
function bezier_closest(curve::AbstractMatrix{Float64}, p::AbstractVector{Float64},
    t_bot::Float64=0.0, t_top::Float64=1.0, n::Int=8)
    mid = (t_bot + t_top) * 0.5
    n == 0 && return mid
    p_bot = bezier_point(curve, t_bot)
    p_top = bezier_point(curve, t_top)
    d_bot = norm(p_bot .- p)
    d_top = norm(p_top .- p)
    if d_bot < d_top
        return bezier_closest(curve, p, t_bot, mid, n - 1)
    end
    return bezier_closest(curve, p, mid, t_top, n - 1)
end

"""
    closest_curve_point(map, pos, angle) -> (Union{Nothing,Vector{Float64}}, Union{Nothing,Vector{Float64}})

`Simulator.closest_curve_point`: closest point and unit tangent on the lane
curve best aligned with the heading (largest `dot(curve_heading, dir_vec)`),
or `(nothing, nothing)` off-drivable tiles.

`curve_headings` are normalised by the single Frobenius norm of the whole
stack (NumPy semantics), exactly like the reference.
"""
function closest_curve_point(map::RoadMap, pos::AbstractVector{Float64},
    angle::Float64)
    i, j = get_grid_coords(map, pos)
    tile = _get_tile(map, i, j)
    (tile === nothing || !tile.drivable) && return nothing, nothing

    curves = stack_curves(tile.curves)
    curve_headings = curves[:, 4, :] .- curves[:, 1, :]
    fnorm = norm(curve_headings)
    curve_headings ./= fnorm
    dir_vec = get_dir_vec(angle)

    dot_prods = curve_headings * collect(dir_vec)
    best = argmax(dot_prods)
    cps = curves[best, :, :]

    t = bezier_closest(cps, pos)
    return bezier_point(cps, t), bezier_tangent(cps, t)
end

"""
    get_lane_pos2(map, pos, angle) -> LanePosition

`Simulator.get_lane_pos2`: signed lateral distance and signed heading error
relative to the closest point of the right-lane curve. Throws [`NotInLane`](@ref)
off-lane (the reference raises it for non-drivable tiles).
"""
function get_lane_pos2(map::RoadMap, pos::AbstractVector{Float64}, angle::Float64)
    point, tangent = closest_curve_point(map, pos, angle)
    if point === nothing || tangent === nothing
        throw(NotInLane("Point not in lane: $pos"))
    end
    dir_vec = get_dir_vec(angle)
    dot_dir = clamp(dot(collect(dir_vec), tangent), -1.0, 1.0)

    pos_vec = pos .- point
    up_vec = [0.0, 1.0, 0.0]
    right_vec = cross(tangent, up_vec)
    signed_dist = dot(pos_vec, right_vec)

    angle_rad = acos(dot_dir)
    if dot(collect(dir_vec), right_vec) < 0
        angle_rad *= -1
    end
    angle_deg = rad2deg(angle_rad)
    return LanePosition(signed_dist, dot_dir, angle_deg, angle_rad)
end

"""
    stack_curves(curves) -> Array{Float64,3}

Stack the per-tile curve control polygons (`Vector{Matrix{Float64}}`) into
the `(N,4,3)` array used by [`closest_curve_point`](@ref).
"""
function stack_curves(curves::AbstractVector{<:AbstractMatrix{Float64}})
    n = length(curves)
    out = Array{Float64,3}(undef, n, 4, 3)
    for k in 1:n
        out[k, :, :] = curves[k]
    end
    return out
end

function stack_curves(curves::AbstractVector{<:AbstractVector{<:NTuple{3,Float64}}})
    n = length(curves)
    out = Array{Float64,3}(undef, n, 4, 3)
    for k in 1:n
        for r in 1:4
            out[k, r, :] = collect(curves[k][r])
        end
    end
    return out
end