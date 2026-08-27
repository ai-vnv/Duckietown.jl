# FJ9.0 — the visualisation contract. Backend-free, by construction.
#
# Every coordinate a renderer draws is computed HERE, in the core, from the
# model. The backend extension receives finished geometry and does nothing but
# put ink on it. Two consequences, both deliberate:
#
#   * the geometry is testable without a plotting library — the FJ9 tests
#     assert that the drawn ego centre IS the state's ego position, not that
#     some pixel is a particular colour;
#   * no visualisation package is a dependency of this package, exactly as no
#     solver and no Python is.
#
# The renderer must also never read the Python simulator: everything below
# derives from `DuckieWorldState`, the `RoadMap`, and the model's own
# projections. Delete the reference environment and FJ9 still works.
#
# FJ10 fixed which entry points may exist. `render_observation` and
# `render_belief` are RESERVED and must stay unimplemented until a partially
# observable formulation actually exists; a test enforces that.

# ---------------------------------------------------------------------------
# Generic functions — declared here, given methods by the backend extension
# ---------------------------------------------------------------------------

"""
    render_world(mdp, state; kwargs...)

Top-down view of the latent world. Requires a Makie backend to be loaded
(`using CairoMakie`); without one this throws a method error, which is the
intended behaviour for a package that does not depend on a plotting library.
"""
function render_world end

"""
    render_native(w::DuckieWorldState; view=:ego, size=(800, 600), kwargs...)

Native LOOKALIKE render of the latent world using the reference simulator's
own texture and mesh assets: `view = :ego` gives the robot's forward camera
(reference constants: fov 75°, height 0.108 m), `view = :bev` the top-down
view. Requires a rasterising Makie backend (`using GLMakie`) — CairoMakie
cannot texture-map meshes per pixel. See [`NATIVE_RENDER_NOTE`](@ref): this
output is for casual, Python-free use and is never parity evidence.
"""
function render_native end

"""
    render_projection(raw, cont; kwargs...)

Panel of the model's own projections. These are **privileged** quantities, not
sensor observations — FJ10 measured that only 6 of the 15 continuous
components could come from a sensor at all — and the panel must say so.
"""
function render_projection end

"""
    render_policy(policy, mdp; kwargs...)

Policy or value slice over two chosen state dimensions, with every other
dimension reported as fixed context.
"""
function render_policy end

"""
    render_rollout(aggregate_or_comparison; kwargs...)

Draw a rollout comparison built from a FROZEN experiment artefact. Takes the
loaded, validated data — never a path, never a model — so a figure cannot
be produced by re-running anything.
"""
function render_rollout end

"""
    render_search(snapshot; kwargs...)

Draw a [`SearchSnapshot`](@ref). Takes the solver-neutral snapshot, never a
solver's own tree type, so this function never imports a planning library.
"""
function render_search end

"""
    render_search_action_plane(snapshot; kwargs...)

FJ9.5d — the continuous root actions a search actually sampled. Only
sampled points are drawn; a smoothed surface would imply the planner
evaluated action combinations it never tried.
"""
function render_search_action_plane end

"""
    render_diagnostics(episode; kwargs...)

FJ9.6c — one episode's diagnostic time series, as five separate panels.
Takes an [`EpisodeDiagnostics`](@ref) built from the frozen decision log, so
the figure cannot be produced by running the environment.

"""
function render_diagnostics end

"""
    render_diagnostics_aggregate(log, solvers; kwargs...)

FJ9.6d — paired and aggregate diagnostics across every episode of each
solver, binned by normalised progress with no interpolation.
"""
function render_diagnostics_aggregate end

"""
    render_frame(static, sequence, t; kwargs...)

FJ9.7b — one animation frame: the world at decision `t`, the history up to
`t`, and nothing after it. Takes recorded evidence only.
"""
function render_frame end

"""
    render_animation(static, sequence, path; kwargs...)

FJ9.7 — write the playback of one episode to `path` (`.mp4` or `.gif`).
"""
function render_animation end

"""
    render_paired_animation(static, a, b, path; kwargs...)

FJ9.7d — two solvers on the same seed, side by side on the absolute decision
index. A panel whose episode has ended freezes on its terminal frame.
"""
function render_paired_animation end

"""
    render_composite(composite; kwargs...)

FJ9.8 — a publication composite, drawn from the same data objects the
individual renderers consume. Never an assembly of pre-rendered images.
"""
function render_composite end

# ---------------------------------------------------------------------------
# World geometry
# ---------------------------------------------------------------------------

"""
    TilePatch

One map tile as a closed polygon in world coordinates, with the classification
a renderer needs to colour it.
"""
struct TilePatch
    i::Int
    j::Int
    kind::Symbol
    drivable::Bool
    corners::Vector{NTuple{2,Float64}}   # closed ring, 5 points
end

"""
    WorldScene

Everything needed to draw one frame, in world coordinates (x, z), computed
from the model. A backend consumes this and adds nothing of its own.

`ego_footprint` is the true collision polygon (`get_agent_corners`), not a
decorative rectangle, so what is drawn is what the physics uses.
"""
struct WorldScene
    tiles::Vector{TilePatch}
    lane_centrelines::Vector{Vector{NTuple{2,Float64}}}
    extent::NTuple{4,Float64}                 # the MAP's extent
    view_extent::NTuple{4,Float64}            # extent covering everything drawn
    tile_size::Float64
    ego_position::NTuple{2,Float64}
    ego_angle::Float64
    ego_footprint::Vector{NTuple{2,Float64}}  # closed ring
    ego_heading::NTuple{2,Float64}            # unit direction
    ego_velocity::NTuple{2,Float64}           # heading scaled by speed
    ego_speed::Float64
    stop_signs::Vector{NTuple{2,Float64}}
    stop_lines::Vector{NTuple{4,Float64}}     # x1, z1, x2, z2
    ducks::Vector{NTuple{2,Float64}}
    duck_footprints::Vector{Vector{NTuple{2,Float64}}}
    duck_visible::Vector{Bool}
    duck_active::Vector{Bool}
    trajectory::Vector{NTuple{2,Float64}}
end

_close_ring(pts) = isempty(pts) ? pts : vcat(pts, [pts[1]])

"""
    _view_extent(map_extent, groups...; margin) -> (xmin, xmax, zmin, zmax)

An extent covering the map AND everything else drawn.

This matters on `small_loop`: the injected stop sign sits at z = 1.8135 while
the 3x3 grid ends at 1.755, so the sign and its stop line are OUTSIDE the tile
grid. That is the reference's own placement — validated in FJ3 map loading and
exercised live in FJ5/FJ6 — and a view clipped to the map would silently hide
it. Showing it is the honest choice.
"""
function _view_extent(map_extent, groups...)
    margin = groups[end]
    xmin, xmax, zmin, zmax = map_extent
    for g in groups[1:(end - 1)]
        for p in g
            if length(p) == 4          # a segment (x1, z1, x2, z2)
                xmin = min(xmin, p[1], p[3]); xmax = max(xmax, p[1], p[3])
                zmin = min(zmin, p[2], p[4]); zmax = max(zmax, p[2], p[4])
            else
                xmin = min(xmin, p[1]); xmax = max(xmax, p[1])
                zmin = min(zmin, p[2]); zmax = max(zmax, p[2])
            end
        end
    end
    return (xmin - margin, xmax + margin, zmin - margin, zmax + margin)
end

"""
    tile_patches(map) -> Vector{TilePatch}
"""
function tile_patches(m::RoadMap)
    out = TilePatch[]
    h, w = size(m.grid)
    ts = m.tile_size
    for j in 1:h, i in 1:w
        tile = _get_tile(m, i - 1, j - 1)
        tile === nothing && continue
        x0, z0 = (i - 1) * ts, (j - 1) * ts
        corners = [(x0, z0), (x0 + ts, z0), (x0 + ts, z0 + ts), (x0, z0 + ts)]
        push!(out, TilePatch(i - 1, j - 1, tile.kind, tile.drivable,
            _close_ring(corners)))
    end
    return out
end

"""
    lane_centrelines(map; samples) -> Vector{Vector{NTuple{2,Float64}}}

Every drivable tile's lane curves, sampled in world coordinates. These are the
same Bezier curves the observer uses for `d` and `phi`, so the drawn lane is
the lane the model measures against.
"""
function lane_centrelines(m::RoadMap; samples::Integer=24)
    out = Vector{NTuple{2,Float64}}[]
    for (i, j) in drivable_tiles(m)
        tile = _get_tile(m, i, j)
        (tile === nothing || isempty(tile.curves)) && continue
        for c in tile.curves
            cm = curve_matrix(c)
            push!(out, [(p[1], p[3]) for p in
                (bezier_point(cm, t) for t in range(0.0, 1.0; length=samples))])
        end
    end
    return out
end

"""
    stop_line_segment(sign, forward, offset, half_width) -> (x1, z1, x2, z2)

The line the model actually measures `d_stop` against.

There is **no stop-line object in the model**. `next_stop_candidate` computes

    rel      = sign.pos - ego.pos
    ahead    = dot(rel, forward)          # forward = the EGO's lane frame
    d_stop   = max(0, ahead - sign_to_line_offset)

so the "stop line" is the locus of points at along-track offset
`sign_to_line_offset` before the sign, measured along the **ego's** direction of
travel — not along the sign's own facing. Its width is the model's own
acceptance gate, `stop_lateral_limit`, since a sign only counts while
`|dot(rel, right)| <= stop_lateral_limit`.

Getting this wrong is the exact failure this gate exists to prevent: an earlier
version of this function offset along the sign's facing, which produced a
plausible red line in a plausible place that corresponded to nothing the model
computes.
"""
function stop_line_segment(sign::StopSignState, forward::AbstractVector,
    offset::Real, half_width::Real)
    fx, fz = forward[1], forward[3]
    n = hypot(fx, fz)
    n > 0 && ((fx, fz) = (fx / n, fz / n))
    cx = sign.pos[1] - offset * fx
    cz = sign.pos[3] - offset * fz
    # perpendicular to the ego's forward, in the ground plane
    px, pz = -fz, fx
    return (cx - half_width * px, cz - half_width * pz,
        cx + half_width * px, cz + half_width * pz)
end

"""
    world_scene(mdp, state; trajectory) -> WorldScene

Extract every drawable quantity from the model and the latent state. Pure: it
reads the state and returns geometry, and mutates nothing.
"""
function world_scene(m::AnyMDPLike, s::DuckieWorldState;
    trajectory::AbstractVector=NTuple{2,Float64}[],
    lane_samples::Integer=24)
    map_ = s.map
    ts = map_.tile_size
    h, w = size(map_.grid)

    pos = (s.ego.pos[1], s.ego.pos[3])
    dir = get_dir_vec(s.ego.angle)
    heading = (dir[1], dir[3])
    speed = s.ego.speed
    # `get_agent_corners` returns a 4x2 matrix: one ROW per corner, columns
    # (x, z). Iterating columns instead would silently yield two "corners".
    corner_rows = get_agent_corners(collect(s.ego.pos), s.ego.angle)
    footprint = [(corner_rows[k, 1], corner_rows[k, 2])
                 for k in axes(corner_rows, 1)]

    cfg = m.transition.state_cfg
    forward, _ = lane_frame_tabular(s)
    lines = [stop_line_segment(sg, forward, cfg.sign_to_line_offset,
                 cfg.stop_lateral_limit)
             for sg in s.stop_signs]

    duckpos = [(d.pos[1], d.pos[3]) for d in s.ducks]
    duckfp = [_close_ring([(c[1], c[2]) for c in d.obj_corners]) for d in s.ducks]

    return WorldScene(
        tile_patches(map_),
        lane_centrelines(map_; samples=lane_samples),
        (0.0, w * ts, 0.0, h * ts),
        _view_extent((0.0, w * ts, 0.0, h * ts), footprint, duckpos, lines,
            [(sg.pos[1], sg.pos[3]) for sg in s.stop_signs],
            [(p[1], p[2]) for p in trajectory], 0.05 * ts),
        ts,
        pos,
        s.ego.angle,
        _close_ring(footprint),
        heading,
        (heading[1] * speed, heading[2] * speed),
        speed,
        [(sg.pos[1], sg.pos[3]) for sg in s.stop_signs],
        lines,
        duckpos,
        duckfp,
        [d.visible for d in s.ducks],
        [d.pedestrian_active for d in s.ducks],
        [(p[1], p[2]) for p in trajectory],
    )
end

"""
    trajectory_points(states) -> Vector{NTuple{2,Float64}}

Ego ground track of a sequence of world states, for the trajectory overlay.
"""
trajectory_points(states::AbstractVector{DuckieWorldState}) =
    [(s.ego.pos[1], s.ego.pos[3]) for s in states]

# ---------------------------------------------------------------------------
# The privileged-projection panel
# ---------------------------------------------------------------------------

"""
    PROJECTION_PANEL_TITLE

The label the projection panel must carry. FJ10 established that the 15-D
vector is a privileged policy input, not an observation; a panel that omits
this quietly undoes that distinction.
"""
const PROJECTION_PANEL_TITLE = "Privileged model state / policy projection"

"""
    ProjectionCategory

Which subsystem a component belongs to. Orthogonal to its privilege class:
`sigma_stop` is part of the stop subsystem *and* is agent memory.
"""
@enum ProjectionCategory LANE_GEOMETRY EGO_MOTION STOP_SUBSYSTEM DUCK_SUBSYSTEM

"""
    ProjectionEntry

One row of the panel, fully specified by the core: display order, the field it
came from, its raw value, a rendered string, its unit, its subsystem and its
FJ10 privilege class.

The backend must not decide any of these. Labels, units and especially the
privilege classification are **package semantics**, not visual style — a
renderer that invented them could quietly disagree with FJ10.
"""
struct ProjectionEntry
    order::Int
    field::Symbol
    value::Any
    display::String
    unit::String
    category::ProjectionCategory
    privilege::ObservabilityClass
end

"""
    ProjectionScene

The panel as data. `source` records provenance: the values are read from the
`ContinuousState` object given, the same one the SAC/TD3 encoder consumes, and
are never recomputed from the world state here.
"""
struct ProjectionScene
    title::String
    entries::Vector{ProjectionEntry}
    context::Vector{Pair{String,String}}
    source::Symbol
end

const _PROJECTION_UNITS = Dict(
    :d => "m", :phi => "rad", :v => "m/s", :kappa => "1/m",
    :stop_present => "", :d_stop => "m", :sigma_stop => "",
    :duck_present => "", :duck_longitudinal => "m", :duck_lateral => "m",
    :duck_v_longitudinal_relative => "m/s",
    :duck_v_lateral_relative => "m/s", :duck_active => "",
    :duck_crossing_available => "", :stop_hold_progress => "",
)

const _PROJECTION_CATEGORY = Dict(
    :d => LANE_GEOMETRY, :phi => LANE_GEOMETRY, :kappa => LANE_GEOMETRY,
    :v => EGO_MOTION,
    :stop_present => STOP_SUBSYSTEM, :d_stop => STOP_SUBSYSTEM,
    :sigma_stop => STOP_SUBSYSTEM, :stop_hold_progress => STOP_SUBSYSTEM,
    :duck_present => DUCK_SUBSYSTEM, :duck_longitudinal => DUCK_SUBSYSTEM,
    :duck_lateral => DUCK_SUBSYSTEM,
    :duck_v_longitudinal_relative => DUCK_SUBSYSTEM,
    :duck_v_lateral_relative => DUCK_SUBSYSTEM,
    :duck_active => DUCK_SUBSYSTEM,
    :duck_crossing_available => DUCK_SUBSYSTEM,
)

_fmt(::Nothing) = "—"
_fmt(x::Bool) = x ? "true" : "false"
_fmt(x::Real) = string(round(Float64(x); digits=4))
_fmt(x) = string(x)

"""
    projection_scene(raw, cont) -> ProjectionScene

Build the panel from the projections the model already computed.

Values are read straight off `cont` — the very object
`encode_continuous_state` turns into the SAC/TD3 input vector — so the panel
cannot drift from what the policy actually saw. The display order is
`fieldnames(ContinuousState)`, which is the order of the encoded vector, so
row *k* of the panel is component *k* of the policy input.
"""
function projection_scene(raw::RawState, cont::ContinuousState)
    cls = Dict(r.name => r.class for r in continuous_state_observability())
    entries = ProjectionEntry[]
    for (k, f) in enumerate(fieldnames(ContinuousState))
        v = getfield(cont, f)
        push!(entries, ProjectionEntry(k, f, v, _fmt(v),
            get(_PROJECTION_UNITS, f, ""), _PROJECTION_CATEGORY[f], cls[f]))
    end
    # tabular-projection facts the 15-D vector does not carry, shown as
    # context rather than as if they were policy inputs
    context = ["tile" => string(raw.tile), "duck (tabular)" => string(raw.duck)]
    return ProjectionScene(PROJECTION_PANEL_TITLE, entries, context,
        :continuous_state)
end

"""
    projection_rows(raw, cont) -> Vector{NamedTuple}

Flat view of [`projection_scene`](@ref), kept for callers that only want
label/value/class triples.
"""
projection_rows(raw::RawState, cont::ContinuousState) =
    [(label=String(e.field), value=e.display, observability=e.privilege)
     for e in projection_scene(raw, cont).entries]
