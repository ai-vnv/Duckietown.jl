# Native lookalike renderer — the PURE half.
#
# This file computes everything a 3D backend needs to draw a world state the
# way the reference renderer does — tile texture assignment, OBJ/MTL geometry
# with material groups, object transforms, camera poses — using only the
# standard library. The drawing itself lives in the Makie extension
# (`render_native`), so the geometry stays testable without a plotting
# library, exactly like the FJ9.0 scene contract.
#
# HONESTY CONTRACT: this is a LOOKALIKE for casual, Python-free use. It reuses
# the reference simulator's own texture and mesh assets, but it is NOT the
# reference renderer and its output is never parity evidence. For evidence,
# render through the reference backend (FJ5), as the recorded case-study laps
# do.

"""
    NATIVE_RENDER_NOTE

The disclaimer every native-render entry point carries: lookalike output for
casual use, never parity evidence. Kept as a constant so captions and tests
can assert the exact wording.
"""
const NATIVE_RENDER_NOTE = "native Julia lookalike (reference assets, non-reference renderer) — not parity evidence"

# Camera constants, copied from the reference `simulator.py` (names kept):
# CAMERA_FOV_Y = 75, CAMERA_FLOOR_DIST = 0.108, CAMERA_FORWARD_DIST = 0.066,
# CAMERA_ANGLE = 19.15 (downward pitch, degrees).
const NATIVE_CAMERA_FOV_Y = 75.0
const NATIVE_CAMERA_FLOOR_DIST = 0.108
const NATIVE_CAMERA_FORWARD_DIST = 0.066
const NATIVE_CAMERA_ANGLE = 19.15

"""
    duckietown_assets_root() -> String

Root of the reference asset tree (the `.../duckietown_world/data/gd1`
directory holding `textures/` and `meshes/`). Resolution order:

1. `ENV["DUCKIETOWN_ASSETS"]`, when set;
2. the conventional `ddm-ref` conda location under the user's home.

Throws with instructions when neither exists. The assets are the reference
implementation's own files; this package does not ship them.
"""
function duckietown_assets_root()
    candidates = String[]
    haskey(ENV, "DUCKIETOWN_ASSETS") && push!(candidates, ENV["DUCKIETOWN_ASSETS"])
    push!(candidates, joinpath(homedir(),
        "miniconda3/envs/ddm-ref/lib/python3.9/site-packages",
        "duckietown_world/data/gd1"))
    for c in candidates
        isdir(joinpath(c, "textures")) && isdir(joinpath(c, "meshes")) && return c
    end
    throw(ArgumentError(
        "reference assets not found. Set ENV[\"DUCKIETOWN_ASSETS\"] to the " *
        "duckietown_world/data/gd1 directory of a gym-duckietown install " *
        "(it must contain textures/ and meshes/). Tried: " *
        join(candidates, ", ")))
end

"""
    tile_texture_file(spec::TileSpec) -> (filename, rot)

Texture file (in the reference's default `photos` style) and the number of
90° texture rotations for one tile. The straight/curve rotations were
calibrated against the reference renderer's own top-down view of
`small_loop`; the curve textures need a half-turn relative to the straight
mapping.
"""
function tile_texture_file(spec::TileSpec)
    f = spec.kind === :straight ? "straight_1.jpg" :
        spec.kind === :curve_left ? "curve_left_1.png" :
        spec.kind === :curve_right ? "curve_right_1.png" :
        spec.kind === :asphalt ? "asphalt_1.jpg" : "grass_1.jpg"
    rot = mod(round(Int, spec.angle_deg / 90), 4)
    spec.kind in (:curve_left, :curve_right) && (rot = mod(rot + 2, 4))
    return f, rot
end

"""
    NativeMeshGroup

One material group of an OBJ mesh: corner-expanded positions, triangle faces,
uv coordinates, and either a texture path or a solid color. Plain tuples so
no geometry package is needed in the core.
"""
struct NativeMeshGroup
    points::Vector{NTuple{3,Float32}}
    faces::Vector{NTuple{3,Int}}
    uvs::Vector{NTuple{2,Float32}}
    texture::Union{Nothing,String}
    color::NTuple{3,Float32}
end

"""
    load_obj_groups(objpath) -> Vector{NativeMeshGroup}

Minimal OBJ + MTL reader. Vertices are expanded per face corner (position and
uv duplicated), which tolerates OBJs whose uv indexing is partial — the
reference's `duckie.obj` has 94 uv-carrying faces against 132 position faces,
which stricter loaders reject. Faces are grouped by `usemtl`; materials with
`map_Kd` carry the resolved texture path, others their diffuse `Kd` color.
Quads and larger polygons are fan-triangulated.

When the `mtllib` target does not exist — the reference's `sign_stop.obj`
names a `sign_001.mtl` that was never shipped — the reader falls back to the
same-basename `.mtl` next to the OBJ, which is how the reference loader
resolves it too.
"""
function load_obj_groups(objpath::AbstractString)
    dir = dirname(objpath)
    materials = Dict{String,Tuple{NTuple{3,Float32},Union{Nothing,String}}}()
    order = String[]

    function read_mtl(path)
        isfile(path) || return
        cur = ""
        for line in eachline(path)
            t = split(strip(line))
            isempty(t) && continue
            if t[1] == "newmtl"
                cur = t[2]
                materials[cur] = ((0.64f0, 0.64f0, 0.64f0), nothing)
            elseif t[1] == "Kd" && !isempty(cur)
                kd = (parse(Float32, t[2]), parse(Float32, t[3]),
                      parse(Float32, t[4]))
                materials[cur] = (kd, materials[cur][2])
            elseif t[1] == "map_Kd" && !isempty(cur)
                tex = normpath(joinpath(dir, join(t[2:end], " ")))
                materials[cur] = (materials[cur][1], tex)
            end
        end
    end

    vs = NTuple{3,Float32}[]
    vts = NTuple{2,Float32}[]
    groups = Dict{String,NativeMeshGroup}()
    cur = "__default__"

    getgroup(name) = get!(groups, name) do
        push!(order, name)
        kd, tx = get(materials, name, ((0.64f0, 0.64f0, 0.64f0), nothing))
        NativeMeshGroup(NTuple{3,Float32}[], NTuple{3,Int}[],
            NTuple{2,Float32}[], tx, kd)
    end

    sibling_mtl = joinpath(dir,
        replace(basename(objpath), r"\.obj$" => ".mtl"))
    for line in eachline(objpath)
        t = split(strip(line))
        isempty(t) && continue
        if t[1] == "mtllib"
            named = joinpath(dir, join(t[2:end], " "))
            read_mtl(isfile(named) ? named : sibling_mtl)
        elseif t[1] == "v"
            push!(vs, (parse(Float32, t[2]), parse(Float32, t[3]),
                       parse(Float32, t[4])))
        elseif t[1] == "vt"
            # kept EXACTLY as in the file: the reference meshes rely on
            # OpenGL REPEAT wrap (the sign board's v runs 1..2), so the
            # backend must sample with a repeating sampler — flipping or
            # normalising coordinates here rotates the sign face 180°
            # (measured on sign_stop.png before this was settled)
            push!(vts, (parse(Float32, t[2]), parse(Float32, t[3])))
        elseif t[1] == "usemtl"
            cur = t[2]
        elseif t[1] == "f"
            g = getgroup(cur)
            corner_ids = Int[]
            for c in t[2:end]
                parts = split(c, "/")
                vi = parse(Int, parts[1])
                ti = length(parts) >= 2 && !isempty(parts[2]) ?
                     parse(Int, parts[2]) : 0
                push!(g.points, vs[vi])
                push!(g.uvs, ti > 0 ? vts[ti] : (0.5f0, 0.5f0))
                push!(corner_ids, length(g.points))
            end
            for k in 2:(length(corner_ids) - 1)
                push!(g.faces, (corner_ids[1], corner_ids[k],
                                corner_ids[k + 1]))
            end
        end
    end
    return [groups[n] for n in order if !isempty(groups[n].faces)]
end

"""
    NativeObject

One placed OBJ instance: its material groups plus the world transform —
uniform `scale`, rotation `angle` about +y, translation `offset` (computed so
the mesh's footprint centre lands on the requested position with its base on
the floor). A backend applies `p -> R(angle) * (scale * p) + offset`.
"""
struct NativeObject
    groups::Vector{NativeMeshGroup}
    scale::Float64
    angle::Float64
    offset::NTuple{3,Float64}
end

"""
    place_object(objpath, pos, angle, height) -> NativeObject

Load an OBJ and compute the transform that scales it to `height`, rotates it
about +y by `angle`, grounds its base, and centres its footprint on `pos`
(the reference convention for map objects).
"""
function place_object(objpath::AbstractString, pos, angle::Real, height::Real;
                      default_texture::Union{Nothing,AbstractString} = nothing)
    groups = load_obj_groups(objpath)
    # meshes without an MTL (the reference duckie.obj ships bare, its texture
    # sitting next to it) get the conventional side-by-side texture
    if default_texture !== nothing
        groups = [g.texture === nothing ?
                  NativeMeshGroup(g.points, g.faces, g.uvs,
                                  String(default_texture), g.color) : g
                  for g in groups]
    end
    xs = Float32[]; ys = Float32[]; zs = Float32[]
    for g in groups, p in g.points
        push!(xs, p[1]); push!(ys, p[2]); push!(zs, p[3])
    end
    ymin, ymax = extrema(ys)
    cx = (maximum(xs) + minimum(xs)) / 2
    cz = (maximum(zs) + minimum(zs)) / 2
    s = height / (ymax - ymin)
    ca, sa = cos(angle), sin(angle)
    off = (pos[1] - (ca * s * cx + sa * s * cz),
           -s * ymin,
           pos[3] - (-sa * s * cx + ca * s * cz))
    return NativeObject(groups, s, Float64(angle), off)
end

"""
    NativeWorld

Everything a 3D backend needs to draw one `DuckieWorldState`: the textured
floor tiles, the placed objects (visible duckies, stop signs, and the ego
robot mesh for top-down views), and the two reference camera poses.
"""
struct NativeWorld
    tile_size::Float64
    extent::NTuple{2,Float64}
    tiles::Vector{Tuple{Int,Int,String,Int}}   # (i, j, texture path, rot)
    objects::Vector{NativeObject}
    ego::NativeObject
    ego_eye::NTuple{3,Float64}
    ego_lookat::NTuple{3,Float64}
    fov::Float64
end

"""
    native_world(w::DuckieWorldState; assets=duckietown_assets_root()) -> NativeWorld

Pure scene description of one world state for the native lookalike renderer
(`render_native` in the Makie extension). Duck heights use each duckie's own
reference `scale`; the stop sign and the ego robot mesh use the reference
injection heights.
"""
function native_world(w::DuckieWorldState;
                      assets::AbstractString = duckietown_assets_root())
    ts = w.map.tile_size
    texdir = joinpath(assets, "textures", "tiles", "photos")
    grid = w.map.grid
    h, wd = size(grid)
    tiles = Tuple{Int,Int,String,Int}[]
    for j in 1:h, i in 1:wd
        isassigned(grid, j, i) || continue
        f, rot = tile_texture_file(grid[j, i])
        push!(tiles, (i - 1, j - 1, joinpath(texdir, f), rot))
    end

    objects = NativeObject[]
    for d in w.ducks
        d.visible || continue
        raw_h = (DUCKIE_MESH_MAX[2] - DUCKIE_MESH_MIN[2]) * d.scale
        push!(objects, place_object(
            joinpath(assets, "meshes", "duckie", "duckie.obj"),
            d.pos, d.angle, raw_h;
            default_texture = joinpath(assets, "meshes", "duckie", "duckie.png")))
    end
    for sg in w.stop_signs
        push!(objects, place_object(
            joinpath(assets, "meshes", "signs", "sign_stop", "sign_stop.obj"),
            sg.pos, sg.angle, 0.18))
    end

    ego = place_object(joinpath(assets, "meshes", "duckiebot", "duckiebot.obj"),
        w.ego.pos, w.ego.angle, 0.09)

    hv = heading_vec(w.ego.angle)
    ex = w.ego.pos[1] + NATIVE_CAMERA_FORWARD_DIST * hv[1]
    ez = w.ego.pos[3] + NATIVE_CAMERA_FORWARD_DIST * hv[3]
    eye = (ex, NATIVE_CAMERA_FLOOR_DIST, ez)
    # the reference pitches the camera down by CAMERA_ANGLE
    cp, sp = cosd(NATIVE_CAMERA_ANGLE), sind(NATIVE_CAMERA_ANGLE)
    lookat = (ex + cp * hv[1], NATIVE_CAMERA_FLOOR_DIST - sp,
              ez + cp * hv[3])

    return NativeWorld(ts, (wd * ts, h * ts), tiles, objects, ego,
        eye, lookat, NATIVE_CAMERA_FOV_Y)
end
