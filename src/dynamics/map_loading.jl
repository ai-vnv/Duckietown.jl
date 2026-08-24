# FJ3.1: map reconstruction — tiles, lane curves, world↔tile coordinates,
# static-object interpretation.
#
# Verbatim port of `simulator._get_curve` / `_interpret_map` / `get_grid_coords`
# / `_get_tile`, `graphics.gen_rot_matrix`, and `DuckietownEnv.interpret_object`
# (pinned gym-duckietown 6.1.34), plus the object-injection semantics of
# `duckduck/src/duck_controller.py::prepare_task_map_data` (world-frame poses
# with `rotate` in degrees, scale from mesh height).

import LinearAlgebra: norm, dot

"""
    directions_index(orient) -> Int

`["S", "E", "N", "W"].index(orient)` (0..3) — `_interpret_map`.
"""
const MAP_DIRECTIONS = ("S", "E", "N", "W")
const DEFAULT_ORIENT = "E"

direction_index(orient::AbstractString) = findfirst(==(orient), MAP_DIRECTIONS) - 1

"""
    gen_rot_matrix(axis, angle) -> Matrix{Float64}

Quaternion rotation matrix (`graphics.gen_rot_matrix`, verbatim).
"""
function gen_rot_matrix(axis::AbstractVector{Float64}, angle::Float64)
    axis0 = collect(axis)
    a0 = norm(axis0)
    ax = axis0 ./ a0
    a = cos(angle / 2)
    bc = -ax .* sin(angle / 2)
    b, c, d = bc
    return [
        a*a+b*b-c*c-d*d 2*(b*c-a*d) 2*(b*d+a*c)
        2*(b*c+a*d) a*a+c*c-b*b-d*d 2*(c*d-a*b)
        2*(b*d-a*c) 2*(c*d+a*b) a*a+d*d-b*b-c*c
    ]
end

# Curve templates (unit frame, scaled by tile_size in tile_curves). Values
# copied verbatim from `simulator._get_curve`; each 4×3 matrix is one cubic
# Bezier control polygon (rows = control points, columns = x, y, z).
const STRAIGHT_TEMPLATE = [
    -0.20 0.0 -0.50
    -0.20 0.0 -0.25
    -0.20 0.0 0.25
    -0.20 0.0 0.50

    0.20 0.0 0.50
    0.20 0.0 0.25
    0.20 0.0 -0.25
    0.20 0.0 -0.50
]

const CURVE_LEFT_TEMPLATE = [
    -0.20 0.0 -0.50
    -0.20 0.0 0.00
    0.00 0.0 0.20
    0.50 0.0 0.20

    0.50 0.0 -0.20
    0.30 0.0 -0.20
    0.20 0.0 -0.30
    0.20 0.0 -0.50
]

const CURVE_RIGHT_TEMPLATE = [
    -0.20 0.0 -0.50
    -0.20 0.0 -0.20
    -0.30 0.0 -0.20
    -0.50 0.0 -0.20

    -0.50 0.0 0.20
    -0.30 0.0 0.20
    0.30 0.0 0.00
    0.20 0.0 -0.50
]

const THREE_WAY_TEMPLATE = [
    -0.20 0.0 -0.50
    -0.20 0.0 -0.25
    -0.20 0.0 0.25
    -0.20 0.0 0.50

    -0.20 0.0 -0.50
    -0.20 0.0 0.00
    0.00 0.0 0.20
    0.50 0.0 0.20

    0.20 0.0 0.50
    0.20 0.0 0.25
    0.20 0.0 -0.25
    0.20 0.0 -0.50

    0.50 0.0 -0.20
    0.30 0.0 -0.20
    0.20 0.0 -0.20
    0.20 0.0 -0.50

    0.20 0.0 0.50
    0.20 0.0 0.20
    0.30 0.0 0.20
    0.50 0.0 0.20

    0.50 0.0 -0.20
    0.30 0.0 -0.20
    -0.20 0.0 0.00
    -0.20 0.0 0.50
]

const FOUR_WAY_TEMPLATE = [
    -0.20 0.0 -0.50
    -0.20 0.0 0.00
    0.00 0.0 0.20
    0.50 0.0 0.20

    -0.20 0.0 -0.50
    -0.20 0.0 -0.25
    -0.20 0.0 0.25
    -0.20 0.0 0.50

    -0.20 0.0 -0.50
    -0.20 0.0 -0.20
    -0.30 0.0 -0.20
    -0.50 0.0 -0.20
]

const DRIVABLE_TILES = Set(["straight", "curve_left", "curve_right",
    "3way_left", "3way_right", "4way"])

"""
    tile_curves(kind, angle_index, i, j, tile_size) -> Array{Float64,3}

`Simulator._get_curve`: the tile's directed lane curves in world coordinates.
`angle_index` is the tile orientation index (0..3, `directions.index`).
The 4way template is rotated 4 times (`rot·π/2` for `rot ∈ 0:3`); all others
once by `(angle_index·π)/2`. Returns an `(N,4,3)` stack.
"""
function tile_curves(kind::Symbol, angle_index::Int, i::Int, j::Int,
    tile_size::Float64)
    template = if kind == :straight
        STRAIGHT_TEMPLATE
    elseif kind == :curve_left
        CURVE_LEFT_TEMPLATE
    elseif kind == :curve_right
        CURVE_RIGHT_TEMPLATE
    elseif kind == :three_way_left || kind == :three_way_right
        THREE_WAY_TEMPLATE
    elseif kind == :four_way
        FOUR_WAY_TEMPLATE
    else
        error("Cannot get bezier for kind $kind")
    end
    pts = template .* tile_size
    direction = [0.0, 1.0, 0.0]
    offset = reshape([(i + 0.5) * tile_size, 0.0, (j + 0.5) * tile_size], 1, 3)
    if kind == :four_way
        out = Array{Float64,3}(undef, 12, 4, 3)
        for rot in 0:3
            mat = gen_rot_matrix(direction, (Float64(rot) * pi) / 2.0)
            for k in 1:3
                row = 3rot + k
                out[row, :, :] = (pts[((k-1)*4+1):(k*4), :] * mat) .+ offset
            end
        end
        return out
    end
    n = size(template, 1) ÷ 4
    mat = gen_rot_matrix(direction, (Float64(angle_index) * pi) / 2.0)
    out = Array{Float64,3}(undef, n, 4, 3)
    for k in 1:n
        out[k, :, :] = (pts[((k-1)*4+1):(k*4), :] * mat) .+ offset
    end
    return out
end

"""
    map_kind_symbol(kind_s) -> Symbol

Simulator tile-kind string → canonical symbol: `3way_left`/`3way_right`/
`4way` become `:three_way_left`/`:three_way_right`/`:four_way`.
(Not to be confused with `state_projection.classify_tile` → [`TileType`](@ref),
the FJ2 lane-class projection.)
"""
map_kind_symbol(kind_s::AbstractString) =
    Symbol(replace(kind_s, "3way" => "three_way", "4way" => "four_way"))

"""
    parse_map_tiles(tiles, tile_size) -> Matrix{TileSpec}

`Simulator._interpret_map` for the grid only: each tile string
(`"curve_left/W"`, `"asphalt"`, `"empty"`, ...) becomes a [`TileSpec`](@ref)
with its directed lane curves (drivable tiles only).
"""
function parse_map_tiles(tiles::AbstractMatrix{<:AbstractString},
    tile_size::Float64)
    h, w = size(tiles)
    grid = Matrix{TileSpec}(undef, h, w)
    for j in 1:h, i in 1:w
        t = tiles[j, i]
        if t == "empty"
            continue
        end
        if occursin("/", t)
            parts = split(strip(t), "/")
            kind_s = strip(parts[1])
            orient = strip(parts[2])
            angle = direction_index(orient)
        elseif occursin("4", t)
            kind_s = "4way"
            angle = direction_index(DEFAULT_ORIENT)
        else
            kind_s = t
            angle = direction_index(DEFAULT_ORIENT)
        end
        drivable = kind_s in DRIVABLE_TILES
        kind = map_kind_symbol(kind_s)
        curves = Vector{Vector{NTuple{3,Float64}}}()
        if drivable
            stacked = tile_curves(kind, angle, i - 1, j - 1, tile_size)
            for k in 1:size(stacked, 1)
                push!(curves, [(stacked[k, r, 1], stacked[k, r, 2],
                    stacked[k, r, 3]) for r in 1:4])
            end
        end
        grid[j, i] = TileSpec(kind, Float64(90 * angle), drivable, curves)
    end
    return grid
end

"""
    small_loop_tiles() -> Matrix{String}

The `small_loop` map as tile strings, transcribed verbatim from
`duckietown_world/data/gd1/maps/small_loop.yaml` (3×3, tile_size 0.585).
Rows are `j` (z) rows, matching `_interpret_map`'s `tiles` layout.
"""
function small_loop_tiles()
    return [
        "curve_left/W" "straight/W" "curve_left/N"
        "straight/S" "asphalt" "straight/N"
        "curve_left/S" "straight/E" "curve_left/E"
    ]
end

# Mesh bounding boxes (units = mesh frame) of the two injected objects,
# copied from the resolved meshes (duckie.obj, sign_generic.obj) as loaded by
# `gym_duckietown.simulator.get_mesh`/`get_duckiebot_mesh` in the reference env.
const DUCKIE_MESH_MIN = (-0.8244785070419312, 0.0, -0.562171995639801)
const DUCKIE_MESH_MAX = (0.8303055167198181, 1.5184210538864136, 0.588129997253418)
const SIGN_STOP_MESH_MIN = (-0.02540000155568123, 0.0, -0.032499998807907104)
const SIGN_STOP_MESH_MAX = (0.02540000155568123, 0.16509999334812164, 0.032499998807907104)

"""
    interpret_object_desc(; kind, pos, rotate_deg, height=nothing, scale=nothing,
        static=true, optional=false) -> MapObjectData

`DuckietownEnv.interpret_object` for the descriptors produced by
`duckduck.src.duck_controller.prepare_task_map_data`: `pos` is already the
world-frame anchor, `rotate_deg` the world yaw in degrees, and scale is
`height / mesh.max_coords[2]` when `height` is given (else `scale`).
Throws `ArgumentError` when both `height` and `scale` are given (Python:
`assert ... cannot specify both height and scale`).
"""
function interpret_object_desc(; kind::Symbol,
    pos::NTuple{3,Float64}, rotate_deg::Float64,
    height::Union{Nothing,Float64}=nothing,
    scale::Union{Nothing,Float64}=nothing,
    static::Bool=true, optional::Bool=false)
    minc, maxc = if kind == :duckie
        (DUCKIE_MESH_MIN, DUCKIE_MESH_MAX)
    elseif kind == :sign_stop
        (SIGN_STOP_MESH_MIN, SIGN_STOP_MESH_MAX)
    else
        error("object kind unknown: $kind")
    end
    sc = if height !== nothing && scale !== nothing
        error("cannot specify both height and scale")
    elseif height !== nothing
        Float64(height) / maxc[2]
    elseif scale !== nothing
        Float64(scale)
    else
        1.0
    end
    angle = -deg2rad(rotate_deg)
    posv = collect(pos)
    minv = collect(minc)
    maxv = collect(maxc)
    corners = generate_corners(posv, minv, maxv, angle, sc)
    objnorm = generate_norm(corners)
    # Only duckies carry a `heading` attribute in the reference (`DuckieObj`);
    # plain `WorldObj` instances (e.g. the stop sign) have none.
    heading = kind == :duckie ? heading_vec(angle) : (0.0, 0.0, 0.0)
    MapObjectData(kind, pos, angle, sc, static, optional, true, minc, maxc,
        SAFETY_RAD_MULT * calculate_safety_radius(minv, maxv, sc),
        corners, objnorm, heading)
end

"""
    object_world_pose(grid_width, grid_height, tile_size, tile_pos, rotate_deg)
        -> (pos3, angle)

Map-frame object descriptor (`pos` in tile units, `rotate` in degrees) to the
simulator's world anchor, verbatim per `Simulator.interpret_object` ->
`duckietown_world.map_loading.get_transform` -> `weird_from_cartesian`:

    x_cart = pos[1] * ts
    y_cart = (grid_width - 1 - pos[2]) * ts     # non-righthanded frame
    angle  = -deg2rad(rotate_deg)
    world  = (x_cart, 0, grid_height * ts - y_cart)

For the square `small_loop` (3x3) this reduces to
`(pos[1] * ts, 0, (pos[2] + 1) * ts)`, which is where the injected stop sign
`[1.20, 2.10]` -> `(0.702, 1.8135)` and duckie `[1.62, 0.50]` ->
`(0.9477, 0.8775)` come from.
"""
function object_world_pose(grid_width::Int, grid_height::Int,
    tile_size::Float64, tile_pos::NTuple{2,Float64}, rotate_deg::Float64)
    x = tile_pos[1] * tile_size
    y = (grid_width - 1 - tile_pos[2]) * tile_size
    return (x, 0.0, grid_height * tile_size - y), -deg2rad(rotate_deg)
end

object_world_pose(map::RoadMap, tile_pos::NTuple{2,Float64},
    rotate_deg::Float64) = object_world_pose(size(map.grid, 2),
    size(map.grid, 1), map.tile_size, tile_pos, rotate_deg)

"""
    small_loop_map() -> RoadMap

The canonical experiment map: `small_loop`, tile_size 0.585, plus the
injected stop sign at stop_spawn_pos [1.20, 2.10] rotated 180° (duckduck
controller defaults; the duckie itself is dynamic state created by
[`initial_state`](@ref)).
"""
function small_loop_map(; with_stop_sign::Bool=true)
    ts = 0.585
    grid = parse_map_tiles(small_loop_tiles(), ts)
    objects = MapObjectData[]
    if with_stop_sign
        push!(objects, interpret_object_desc(kind=:sign_stop,
            pos=(0.702, 0.0, 1.8135), rotate_deg=180.0, height=0.18,
            static=true))
    end
    return RoadMap("small_loop", ts, grid, objects)
end

"""
    get_grid_coords(map, pos) -> (Int, Int)

`Simulator.get_grid_coords`: `(floor(x/ts), floor(z/ts))`, potentially
outside the grid.
"""
function get_grid_coords(map::RoadMap, pos::AbstractVector{Float64})
    x, _, z = pos
    return Int(floor(x / map.tile_size)), Int(floor(z / map.tile_size))
end

"""
    _get_tile(map, i, j) -> Union{Nothing,TileSpec}

`Simulator._get_tile`: bounds-checked lookup with **0-based** grid
coordinates `(i, j)` (matching `get_grid_coords`); returns `nothing` for
tiles outside the map.
"""
function _get_tile(map::RoadMap, i::Int, j::Int)
    w = size(map.grid, 2)
    h = size(map.grid, 1)
    (0 <= i < w && 0 <= j < h) || return nothing
    return map.grid[j + 1, i + 1]
end