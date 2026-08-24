# FJ3.1: collision geometry and validity checks.
#
# Verbatim port of the pinned gym-duckietown 6.1.34 sources
# (`collision.py`, `graphics.rotate_point`, `objects.py` geometry helpers,
# `simulator.py` `_valid_pose`/`_inconvenient_spawn`/`_collision`) plus the
# constants the simulator hardcodes at module scope.

import LinearAlgebra: norm, dot, LAPACK

# --- simulator.py module constants ------------------------------------------

const ROBOT_WIDTH = 0.13 + 0.02          # WHEEL_DIST/2 margins included
const ROBOT_LENGTH = 0.18
const WHEEL_DIST = 0.102
const CAMERA_FORWARD_DIST = 0.066
const MIN_SPAWN_OBJ_DIST = 0.25
const MAX_SPAWN_ATTEMPTS = 5000
const SAFETY_RAD_MULT = 1.8
const AGENT_SAFETY_RAD = (max(ROBOT_LENGTH, ROBOT_WIDTH) / 2) * SAFETY_RAD_MULT

"""
    rotate_point(px, py, cx, cy, theta) -> (x, z)

Rotate a 2D point around a center (`graphics.rotate_point`, verbatim).
"""
function rotate_point(px::Float64, py::Float64, cx::Float64, cy::Float64,
    theta::Float64)
    dx = px - cx
    dy = py - cy
    new_dx = dx * cos(theta) + dy * sin(theta)
    new_dy = dy * cos(theta) - dx * sin(theta)
    return cx + new_dx, cy + new_dy
end

"""
    generate_corners(pos, min_coords, max_coords, theta, scale) -> Matrix

Object bounding-box corners (4×2, `x`/`z` columns) in the exact order of
`objects.generate_corners`: (min,min), (max,min), (max,max), (min,max),
rotated about the object anchor.
"""
function generate_corners(pos::AbstractVector{Float64},
    min_coords::AbstractVector{Float64}, max_coords::AbstractVector{Float64},
    theta::Float64, scale::Float64)
    px = pos[1]
    pz = pos[3]
    pts = [
        rotate_point(min_coords[1] * scale + px, min_coords[3] * scale + pz,
            px, pz, theta),
        rotate_point(max_coords[1] * scale + px, min_coords[3] * scale + pz,
            px, pz, theta),
        rotate_point(max_coords[1] * scale + px, max_coords[3] * scale + pz,
            px, pz, theta),
        rotate_point(min_coords[1] * scale + px, max_coords[3] * scale + pz,
            px, pz, theta),
    ]
    return Matrix{Float64}(reshape(reinterpret(Float64, collect(pts)), 2, 4)')
end

"""
    generate_norm(corners) -> Matrix{Float64}

Two orthogonal axis normals (2×2) of a rectangle given its corners in
`generate_corners` order.

Python computes `np.cov(corners, rowvar=false, bias=true)` followed by
`np.linalg.eig` and returns `vect.T` (eigenvectors as rows). `np.linalg.eig`
is LAPACK `dgeev`, whose eigenvector ORDER (natural, NOT sorted by
eigenvalue) and SIGN conventions are part of the recorded reference state
(`DuckieObj.obj_norm` is compared bit-level in the duckie parity fixtures),
so Julia calls the same LAPACK driver rather than an analytic formula.
An earlier analytic implementation produced axes that were equivalent for
the SAT tests but ordered/signed differently — a real FJ3.4 state-parity
bug that the fixture comparison caught.
"""
function generate_norm(corners::AbstractMatrix{Float64})
    ca = cov_bias(corners)
    _, _, _, vr = LAPACK.geev!('N', 'V', ca)
    return Matrix{Float64}(vr')
end

"""
    cov_bias(corners) -> Matrix{Float64}

Population covariance (`np.cov(rowvar=false, bias=true)`) of the 4×2 corner
matrix: 1/4 of the centred outer products, summed over corners.
"""
function cov_bias(corners::AbstractMatrix{Float64})
    n = size(corners, 1)
    m = vec(mean(corners, dims=1))
    C = zeros(Float64, 2, 2)
    for k in 1:n
        d = corners[k, :] .- m
        C .+= d * d'
    end
    return C ./ n
end

"""
    heading_vec(angle) -> NTuple{3,Float64}

Forward vector of an object at world heading `angle`: `(cos, 0, -sin)`
(`objects.heading_vec`, verbatim).
"""
heading_vec(angle::Float64) = (cos(angle), 0.0, -sin(angle))

"""
    calculate_safety_radius(min_coords, max_coords, scale) -> Float64

`objects.calculate_safety_radius`: norm of the per-axis maximum absolute
extent (x/z), scaled.

The reference mesh bounding boxes are **float32** (`duckietown-world` obj
meshes), and `np.max([abs(min), abs(max)], axis=0)` keeps that dtype, so the
whole norm is computed in float32 before promotion by the float64 `scale`.
The norm is evaluated as `sqrt(x² + z²)` (NumPy's naive reduction), not the
BLAS `hypot`-style norm.
"""
function calculate_safety_radius(min_coords::AbstractVector{Float64},
    max_coords::AbstractVector{Float64}, scale::Float64)
    x = max(abs(Float32(min_coords[1])), abs(Float32(max_coords[1])))
    z = max(abs(Float32(min_coords[3])), abs(Float32(max_coords[3])))
    return Float64(sqrt(x * x + z * z)) * scale
end

"""
    tile_corners(pos2, width) -> Matrix{Float64}

Absolute corners of a tile given grid coordinates `(i, j)` and tile width
(`simulator.tile_corners`, verbatim: `[px·w − w, px·w + w, ...]`).
"""
function tile_corners(pos2::AbstractVector{Int}, width::Float64)
    px = Float64(pos2[1])
    pz = Float64(pos2[2])
    return [
        px * width - width pz * width - width
        px * width + width pz * width - width
        px * width + width pz * width + width
        px * width - width pz * width + width
    ]
end

"""
    agent_boundbox(true_pos, width, length, f_vec, r_vec) -> Matrix{Float64}

Agent bounding box (4×2 x/z corners; `collision.agent_boundbox`, verbatim).
Order: rear-left, rear-right, front-right, front-left (front at the end).
"""
agent_boundbox(true_pos::AbstractVector{Float64}, width::Float64,
    length::Float64, f_vec::NTuple{3,Float64}, r_vec::NTuple{3,Float64}) =
    agent_boundbox(true_pos, width, length, collect(f_vec), collect(r_vec))

function agent_boundbox(true_pos::AbstractVector{Float64}, width::Float64,
    length::Float64, f_vec::AbstractVector{Float64}, r_vec::AbstractVector{Float64})
    fv = collect(f_vec)
    rv = collect(r_vec)
    hwidth = 0.5 * width
    hlength = 0.5 * length
    raw = [
        true_pos .- hwidth .* rv .- hlength .* fv,
        true_pos .+ hwidth .* rv .- hlength .* fv,
        true_pos .+ hwidth .* rv .+ hlength .* fv,
        true_pos .- hwidth .* rv .+ hlength .* fv,
    ]
    return Matrix{Float64}(reshape(
        reinterpret(Float64, collect([(p[1], p[3]) for p in raw])), 2, 4)')
end

"""
    _actual_center(pos, angle) -> Vector{Float64}

Geometric centre of the agent (camera-to-wheelbase offset):
`pos + (CAMERA_FORWARD_DIST − ROBOT_LENGTH/2)·dir_vec` (`simulator._actual_center`).
"""
function _actual_center(pos::AbstractVector{Float64}, angle::Float64)
    d = get_dir_vec(angle)
    return pos .+ (CAMERA_FORWARD_DIST - (ROBOT_LENGTH / 2)) .* d
end

"""
    get_agent_corners(pos, angle) -> Matrix{Float64}

`simulator.get_agent_corners`: bounding box of the geometric centre.
"""
function get_agent_corners(pos::AbstractVector{Float64}, angle::Float64)
    return agent_boundbox(_actual_center(pos, angle), ROBOT_WIDTH,
        ROBOT_LENGTH, get_dir_vec(angle), get_right_vec(angle))
end

"""
    tensor_sat_test(norm, corners) -> (mins, maxs)

Projection intervals of `corners` (4×2 or stacked) on each axis of `norm`
(2×2), `min`/`max` over the corner axis (`collision.tensor_sat_test`, verbatim).
"""
function tensor_sat_test(n::AbstractMatrix{Float64}, corners::AbstractMatrix{Float64})
    dotval = n * corners'
    mins = vec(minimum(dotval, dims=2))
    maxs = vec(maximum(dotval, dims=2))
    return mins, maxs
end

is_between_ordered(val::Float64, lowerbound::Float64, upperbound::Float64) =
    lowerbound <= val <= upperbound

overlaps(min1::Float64, max1::Float64, min2::Float64, max2::Float64) =
    is_between_ordered(min2, min1, max1) || is_between_ordered(min1, min2, max2)

"""
    intersects(agent_corners, objs_stacked, agent_norm, norms_stacked) -> Bool

Tensor SAT against N stacked static objects (`simulator.intersects`, verbatim:
every pair of projection intervals must overlap for a collision).
"""
function intersects(agent_corners::AbstractMatrix{Float64},
    objs_stacked::AbstractArray{Float64,3}, agent_norm::AbstractMatrix{Float64},
    norms_stacked::AbstractArray{Float64,3})
    duckduck_min, duckduck_max = tensor_sat_test(agent_norm, agent_corners)
    for idx in 1:size(objs_stacked, 1)
        objduck_min, objduck_max = tensor_sat_test(agent_norm,
            objs_stacked[idx, :, :])
        duckobj_min, duckobj_max = tensor_sat_test(norms_stacked[idx, :, :],
            agent_corners)
        objobj_min, objobj_max = tensor_sat_test(norms_stacked[idx, :, :],
            objs_stacked[idx, :, :])
        if !overlaps(duckduck_min[1], duckduck_max[1], objduck_min[1], objduck_max[1])
            continue
        end
        if !overlaps(duckduck_min[2], duckduck_max[2], objduck_min[2], objduck_max[2])
            continue
        end
        if !overlaps(duckobj_min[1], duckobj_max[1], objobj_min[1], objobj_max[1])
            continue
        end
        if !overlaps(duckobj_min[2], duckobj_max[2], objobj_min[2], objobj_max[2])
            continue
        end
        return true
    end
    return false
end

"""
    intersects_single_obj(agent_corners, obj_corners_T, agent_norm, obj_norm) -> Bool

SAT against one dynamic object (`objects.intersects_single_obj`, verbatim;
`obj_corners_T` is the object's 2×4 corner matrix).
"""
function intersects_single_obj(agent_corners::AbstractMatrix{Float64},
    obj_corners_T::AbstractMatrix{Float64}, agent_norm::AbstractMatrix{Float64},
    obj_norm::AbstractMatrix{Float64})
    duckduck_min, duckduck_max = tensor_sat_test(agent_norm, agent_corners)
    objduck_min, objduck_max = tensor_sat_test(agent_norm, obj_corners_T')
    duckobj_min, duckobj_max = tensor_sat_test(obj_norm, agent_corners)
    objobj_min, objobj_max = tensor_sat_test(obj_norm, obj_corners_T')

    if !overlaps(duckduck_min[1], duckduck_max[1], objduck_min[1], objduck_max[1])
        return false
    end
    if !overlaps(duckduck_min[2], duckduck_max[2], objduck_min[2], objduck_max[2])
        return false
    end
    if !overlaps(duckobj_min[1], duckobj_max[1], objobj_min[1], objobj_max[1])
        return false
    end
    if !overlaps(duckobj_min[2], duckobj_max[2], objobj_min[2], objobj_max[2])
        return false
    end
    return true
end

# --- validity helpers (Simulator._valid_pose / _inconvenient_spawn) ---------

"""
    _drivable_pos(map, pos) -> Bool

`Simulator._drivable_pos`: the tile at `pos` exists and is drivable.
"""
function _drivable_pos(map::RoadMap, pos::AbstractVector{Float64})
    coords = get_grid_coords(map, pos)
    tile = _get_tile(map, coords[1], coords[2])
    tile === nothing && return false
    return tile.drivable
end

"""
    _valid_pose(map, pos, angle, safety_factor=1.0) -> Bool

`Simulator._valid_pose`: geometric centre, both wheels and the front of the
agent on drivable tiles, and no collision with static or dynamic objects.
"""
function _valid_pose(map::RoadMap, pos::AbstractVector{Float64}, angle::Float64,
    safety_factor::Float64=1.0, ducks::AbstractVector{<:DuckieState}=DuckieState[])
    c = _actual_center(pos, angle)
    f_vec = get_dir_vec(angle)
    r_vec = get_right_vec(angle)

    l_pos = c .- (safety_factor * 0.5 * ROBOT_WIDTH) .* r_vec
    r_pos = c .+ (safety_factor * 0.5 * ROBOT_WIDTH) .* r_vec
    f_pos = c .+ (safety_factor * 0.5 * ROBOT_LENGTH) .* f_vec

    all_drivable = _drivable_pos(map, c) && _drivable_pos(map, l_pos) &&
        _drivable_pos(map, r_pos) && _drivable_pos(map, f_pos)

    # Reference quirk that is part of the semantics: `Simulator._valid_pose`
    # re-assigns `pos = _actual_center(pos, angle)` at the top and then calls
    # `get_agent_corners(pos, angle)` on the ALREADY-centered pos — so the
    # collision box it tests carries a DOUBLE _actual_center offset (shifted
    # ~0.024 m behind the true agent box). The wrapper's separate
    # `_duck_collision` uses the true box; the two therefore disagree for a
    # decision or two at contact, and the episode only terminates when this
    # shifted box (or drivability) fails. Preserve it exactly.
    agent_corners = get_agent_corners(c, angle)
    no_collision = !_collision(map, agent_corners, ducks)

    return no_collision && all_drivable
end

"""
    corner_matrix(corners) -> Matrix{Float64}

Convert a `Vector{NTuple{2,Float64}}` corner list into the 4×2 matrix form.
"""
corner_matrix(corners::AbstractVector{<:NTuple{2,Float64}}) =
    Matrix{Float64}(reshape(reinterpret(Float64, collect(corners)), 2, 4)')

"""
    _collision(map, agent_corners, ducks) -> Bool

`Simulator._collision`: SAT against all stacked static objects, then a
per-object SAT against every dynamic duckie.
"""
function _collision(map::RoadMap, agent_corners::AbstractMatrix{Float64},
    ducks::AbstractVector{<:DuckieState})
    agent_norm = generate_norm(agent_corners)
    statics = map.static_objects
    if !isempty(statics)
        # reference stacks the raw corner matrices into (N, 4, 2) and the
        # normals into (N, 2, 2); `tensor_sat_test` transposes corners inside
        n = length(statics)
        objs_stacked = Array{Float64,3}(undef, n, 4, 2)
        norms_stacked = Array{Float64,3}(undef, n, 2, 2)
        for k in 1:n
            objs_stacked[k, :, :] = statics[k].corners
            norms_stacked[k, :, :] = statics[k].norm
        end
        if intersects(agent_corners, objs_stacked, agent_norm, norms_stacked)
            return true
        end
    end
    for d in ducks
        if intersects_single_obj(agent_corners, corner_matrix(d.obj_corners)',
            agent_norm, d.obj_norm)
            return true
        end
    end
    return false
end

"""
    _inconvenient_spawn(map, pos, ducks) -> Bool

`Simulator._inconvenient_spawn`: any visible object is within
`max(|max_coords|) · 0.5 · scale + MIN_SPAWN_OBJ_DIST` of the candidate pose
(3D norm; `max_coords` is the mesh bounding-box corner).
"""
function _inconvenient_spawn(map::RoadMap, pos::AbstractVector{Float64},
    ducks::AbstractVector{<:DuckieState}=DuckieState[])
    objs = Iterators.flatten((map.static_objects, ducks))
    for o in objs
        o.visible || continue
        maxc = maximum(abs, o.max_coords)
        if norm(o.pos .- pos) < maxc * 0.5 * o.scale + MIN_SPAWN_OBJ_DIST
            return true
        end
    end
    return false
end