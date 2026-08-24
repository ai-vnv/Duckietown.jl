"""
    DuckietownMakieExt

FJ9.1 — the drawing half of the visualisation contract, and deliberately only
that half.

Every coordinate this file plots was computed in the core by
[`world_scene`](@ref): tile polygons, lane centrelines from the same Bezier
curves the observer measures `d` and `phi` against, the ego's true collision
footprint, the stop line the model measures `d_stop` against, and the duckies'
own corner polygons. Nothing here derives geometry of its own, so what is drawn
is what the physics uses, and the geometry stays testable without a plotting
library.

The extension is triggered by `Makie`, so any backend works; `CairoMakie` is
the one the project standardises on because it renders on the CPU and needs no
display, which is what headless figure generation requires.

`render_observation` and `render_belief` are **not** defined here. FJ10
reserved those signatures for a partially observable formulation that does not
exist yet, and a test asserts they stay unimplemented.
"""
module DuckietownMakieExt

using DuckietownDecisionModels
using Makie

const DDM = DuckietownDecisionModels

# A palette that survives greyscale printing and does not rely on hue alone.
const TILE_DRIVABLE = RGBAf(0.90, 0.90, 0.90, 1.0)
const TILE_OTHER = RGBAf(0.97, 0.95, 0.86, 1.0)
const LANE_COLOR = RGBAf(0.55, 0.55, 0.55, 0.9)
const EGO_COLOR = RGBAf(0.11, 0.35, 0.72, 1.0)
const HEADING_COLOR = RGBAf(0.05, 0.20, 0.45, 1.0)
const STOP_COLOR = RGBAf(0.78, 0.13, 0.13, 1.0)
const DUCK_ACTIVE = RGBAf(0.95, 0.62, 0.07, 1.0)
const DUCK_IDLE = RGBAf(0.75, 0.70, 0.45, 1.0)
const TRACK_COLOR = RGBAf(0.11, 0.35, 0.72, 0.55)

_xy(pts) = ([p[1] for p in pts], [p[2] for p in pts])

"""
    render_world(mdp, state; ...) -> Figure

Top-down view of the latent world state.

`trajectory` overlays a ground track; pass `trajectory_points(states)`.
`velocity_scale` only scales the drawn velocity arrow — at 0.14 m/s a
true-to-scale arrow would be invisible against a 1.755 m map — and the scale
is written into the axis subtitle so the figure never implies a length it does
not have.
"""
function DDM.render_world(mdp, state::DuckieWorldState;
    trajectory=NTuple{2,Float64}[], figure_size=(620, 640),
    velocity_scale::Real=1.5, title::AbstractString="Duckietown world",
    show_lanes::Bool=true, show_footprint::Bool=true)
    scene = world_scene(mdp, state; trajectory=trajectory)
    # the VIEW extent, not the map extent: on small_loop the stop sign sits
    # outside the tile grid, and clipping to the map would hide it
    xmin, xmax, zmin, zmax = scene.view_extent

    fig = Figure(size=figure_size)
    ax = Axis(fig[1, 1]; title=title,
        subtitle="latent world state · velocity arrow x$(velocity_scale)",
        xlabel="x (m)", ylabel="z (m)", aspect=DataAspect())

    for t in scene.tiles
        xs, ys = _xy(t.corners)
        poly!(ax, Point2f.(xs, ys);
            color=t.drivable ? TILE_DRIVABLE : TILE_OTHER,
            strokecolor=RGBAf(0.75, 0.75, 0.75, 1.0), strokewidth=0.8)
    end

    if show_lanes
        for c in scene.lane_centrelines
            xs, ys = _xy(c)
            lines!(ax, xs, ys; color=LANE_COLOR, linestyle=:dash, linewidth=1.0)
        end
    end

    for (k, sg) in enumerate(scene.stop_signs)
        x1, z1, x2, z2 = scene.stop_lines[k]
        lines!(ax, [x1, x2], [z1, z2]; color=STOP_COLOR, linewidth=3.0)
        scatter!(ax, [sg[1]], [sg[2]]; color=STOP_COLOR, marker=:octagon,
            markersize=13)
    end

    for (k, d) in enumerate(scene.ducks)
        xs, ys = _xy(scene.duck_footprints[k])
        col = scene.duck_active[k] ? DUCK_ACTIVE : DUCK_IDLE
        poly!(ax, Point2f.(xs, ys); color=(col, 0.35), strokecolor=col,
            strokewidth=1.2)
        scatter!(ax, [d[1]], [d[2]]; color=col, markersize=9,
            marker=scene.duck_visible[k] ? :circle : :xcross)
    end

    if !isempty(scene.trajectory)
        xs, ys = _xy(scene.trajectory)
        lines!(ax, xs, ys; color=TRACK_COLOR, linewidth=2.0)
    end

    if show_footprint
        xs, ys = _xy(scene.ego_footprint)
        poly!(ax, Point2f.(xs, ys); color=(EGO_COLOR, 0.35),
            strokecolor=EGO_COLOR, strokewidth=1.6)
    end
    ex, ez = scene.ego_position
    scatter!(ax, [ex], [ez]; color=EGO_COLOR, markersize=7)
    arrows2d!(ax, [ex], [ez], [scene.ego_heading[1] * scene.tile_size * 0.4],
        [scene.ego_heading[2] * scene.tile_size * 0.4];
        color=HEADING_COLOR, tipwidth=7, tiplength=9)
    if scene.ego_speed > 0
        arrows2d!(ax, [ex], [ez], [scene.ego_velocity[1] * velocity_scale],
            [scene.ego_velocity[2] * velocity_scale];
            color=EGO_COLOR, tipwidth=9, tiplength=11)
    end

    limits!(ax, xmin, xmax, zmin, zmax)
    return fig
end

"""
    render_projection(raw, cont; ...) -> Figure

The privileged-projection panel. The title is
[`PROJECTION_PANEL_TITLE`](@ref) and each row carries its observability class,
because FJ10 established that most of this vector could never come from a
sensor. A panel that presented it as "the observation" would quietly undo that.
"""
DDM.render_projection(raw::RawState, cont::ContinuousState; kwargs...) =
    DDM.render_projection(projection_scene(raw, cont); kwargs...)

function DDM.render_projection(scene::ProjectionScene; figure_size=(560, 520))
    fig = Figure(size=figure_size)
    Label(fig[1, 1], scene.title; fontsize=13, font=:bold, tellwidth=false)
    ax = Axis(fig[2, 1])
    hidedecorations!(ax)
    hidespines!(ax)

    n = length(scene.entries) + length(scene.context) + 1
    row(k) = 1.0 - k / (n + 1)
    last_cat = nothing
    for e in scene.entries
        y = row(e.order)
        # a subsystem rule, drawn from the category the CORE assigned
        if e.category !== last_cat
            lines!(ax, [0.01, 0.99], [y + 0.5 / (n + 1), y + 0.5 / (n + 1)];
                color=RGBAf(0.85, 0.85, 0.85, 1), linewidth=0.8)
            last_cat = e.category
        end
        text!(ax, 0.02, y; text=String(e.field), align=(:left, :center),
            fontsize=11)
        text!(ax, 0.56, y; text=e.display, align=(:right, :center), fontsize=11)
        isempty(e.unit) || text!(ax, 0.58, y; text=e.unit,
            align=(:left, :center), fontsize=9, color=RGBAf(0.45, 0.45, 0.45, 1))
        text!(ax, 0.70, y; text=_abbrev(e.privilege), align=(:left, :center),
            fontsize=9, color=e.privilege == DDM.SENSOR_ESTIMABLE ?
                  RGBAf(0.15, 0.45, 0.15, 1) : RGBAf(0.60, 0.30, 0.10, 1))
    end
    for (k, (label, value)) in enumerate(scene.context)
        y = row(length(scene.entries) + 1 + k)
        text!(ax, 0.02, y; text=label, align=(:left, :center), fontsize=10,
            color=RGBAf(0.4, 0.4, 0.4, 1))
        text!(ax, 0.56, y; text=value, align=(:right, :center), fontsize=10,
            color=RGBAf(0.4, 0.4, 0.4, 1))
    end
    limits!(ax, 0, 1, 0, 1)
    return fig
end

_abbrev(c) = c == DDM.SENSOR_ESTIMABLE ? "sensor-estimable" :
             c == DDM.TEMPORALLY_DERIVED ? "derived over time" :
             c == DDM.MAP_PRIVILEGED ? "map-privileged" :
             c == DDM.SIMULATOR_PRIVILEGED ? "simulator-only" : "agent memory"


"""
    render_rollout(aggregate; ...) -> Figure

Draw the frozen FJ8.4b comparison: paired per-seed returns, episode length,
and how each episode ended — environment termination and horizon expiry as
**different** markers, because they are different events.

Everything drawn was computed by the core loader from the artefact. This
method reads no file, builds no model and runs nothing. The provenance block
is drawn onto the figure so a picture cannot be separated from the experiment
it came from.
"""
function DDM.render_rollout(a::RolloutAggregate; figure_size=(1000, 760),
    metric::Symbol=:ret)
    seeds, series = paired_metric(a, metric)
    solvers = a.provenance.solvers
    palette = [RGBAf(0.11, 0.35, 0.72, 1), RGBAf(0.85, 0.37, 0.01, 1),
        RGBAf(0.17, 0.55, 0.20, 1), RGBAf(0.78, 0.13, 0.13, 1),
        RGBAf(0.46, 0.30, 0.64, 1), RGBAf(0.35, 0.35, 0.35, 1)]
    colour = Dict(s => palette[mod1(k, length(palette))]
                  for (k, s) in enumerate(solvers))

    fig = Figure(size=figure_size)
    Label(fig[1, 1:2], "FJ8.4b — six solvers, $(length(seeds)) frozen seeds";
        fontsize=15, font=:bold, tellwidth=false)

    # paired per-seed metric: one line per solver over a shared seed axis
    ax1 = Axis(fig[2, 1]; title="paired per-seed $(metric)",
        subtitle="same initial condition at each x position",
        xlabel="evaluation seed", ylabel=string(metric))
    for s in solvers
        lines!(ax1, 1:length(seeds), series[s]; color=colour[s], label=s)
        scatter!(ax1, 1:length(seeds), series[s]; color=colour[s], markersize=6)
    end
    ax1.xticks = (1:length(seeds), string.(seeds))
    ax1.xticklabelrotation = pi / 2
    ax1.xticklabelsize = 8
    axislegend(ax1; position=:lb, labelsize=9)

    # how each episode ended — the two kinds kept apart
    ax2 = Axis(fig[2, 2]; title="how episodes ended",
        subtitle="environment termination vs horizon expiry",
        xlabel="episodes", yticks=(1:length(solvers), solvers))
    for (k, s) in enumerate(solvers)
        t = solver_summary(a, s)
        barplot!(ax2, [k], [t.env_terminated]; direction=:x,
            color=RGBAf(0.78, 0.13, 0.13, 0.85), width=0.6)
        barplot!(ax2, [k], [t.horizon_reached]; direction=:x,
            offset=[Float64(t.env_terminated)],
            color=RGBAf(0.55, 0.55, 0.55, 0.55), width=0.6)
    end
    text!(ax2, 0.5, length(solvers) + 0.6;
        text="red = environment terminated   grey = horizon reached",
        fontsize=9, align=(:left, :center))

    # episode length distribution, which separates "survived" from "crashed"
    ax3 = Axis(fig[3, 1]; title="episode length",
        xlabel="decisions", yticks=(1:length(solvers), solvers))
    _, lens = paired_metric(a, :decisions)
    for (k, s) in enumerate(solvers)
        scatter!(ax3, lens[s], fill(k, length(seeds)); color=colour[s],
            markersize=7)
    end
    vlines!(ax3, [Float64(a.provenance.horizon)];
        color=RGBAf(0.3, 0.3, 0.3, 0.7), linestyle=:dash)

    # provenance, drawn on the figure
    ax4 = Axis(fig[3, 2])
    hidedecorations!(ax4)
    hidespines!(ax4)
    lines_ = provenance_lines(a)
    for (k, l) in enumerate(lines_)
        text!(ax4, 0.02, 1.0 - k / (length(lines_) + 1); text=l,
            align=(:left, :center), fontsize=10, font=:regular,
            color=RGBAf(0.35, 0.35, 0.35, 1))
    end
    limits!(ax4, 0, 1, 0, 1)
    return fig
end

DDM.render_rollout(c::RolloutComparison; kwargs...) =
    throw(ArgumentError("single-seed rollout figures need per-decision data " *
        "(positions, speed and heading over time), which the FJ8.4b episode " *
        "artefact does not contain. Extend what the experiment records " *
        "first; do not infer it here."))


# ---------------------------------------------------------------------------
# FJ9.5c / FJ9.5d — drawing a captured search
# ---------------------------------------------------------------------------
#
# These methods take a `SearchSnapshot` and nothing else. No planner, no
# `solve`, no MCTS.jl on the canonical path — `tools/fj9_render_check.jl`
# proves in a fresh process that the figures redraw with MCTS_LOADED=false.
#
# Display filters change what is DRAWN, never the snapshot; `visible_nodes`
# returns ids and the snapshot stays immutable.

const SEARCH_SELECTED = RGBAf(0.85, 0.10, 0.10, 1.0)
const SEARCH_NODE = RGBAf(0.20, 0.35, 0.60, 1.0)

_val_label(x) = x === missing ? "—" : string(round(x; digits=2))

"""
    render_search(snapshot; max_depth, min_visits, top_k, ...) -> Figure

Tree view plus a root-action summary.

For a discrete search the summary answers the question the tree makes hard to
read — *which actions actually received search effort?* — with visits grouped
by action, because a vanilla MCTS tree has one root child per simulation.

`missing` values are drawn as `—`. A missing value is not zero, and the
snapshot round-trips it as `missing` precisely so a figure cannot claim
otherwise.
"""
function DDM.render_search(s::SearchSnapshot; max_depth::Integer=3,
    min_visits::Integer=0, top_k::Integer=typemax(Int),
    figure_size=(1080, 640))
    chk = check_snapshot(s)
    chk.ok || throw(ArgumentError(
        "refusing to draw an invalid snapshot: " * join(chk.issues, "; ")))

    ids = visible_nodes(s; max_depth=max_depth, min_visits=min_visits,
        top_k=top_k)
    st = search_statistics(s)

    fig = Figure(size=figure_size)
    Label(fig[1, 1:2], "$(s.solver) — search from one frozen state";
        fontsize=15, font=:bold, tellwidth=false)

    # --- tree -------------------------------------------------------------
    ax = Axis(fig[2, 1]; title="search tree",
        subtitle="showing $(length(ids)) of $(length(s.nodes)) nodes " *
                 "(max_depth=$(max_depth), min_visits=$(min_visits))",
        xlabel="node", ylabel="depth", yreversed=true)
    hidexdecorations!(ax; grid=false)

    # lay out each depth level evenly; x is presentational only
    bydepth = Dict{Int,Vector{Int}}()
    for id in ids
        push!(get!(bydepth, s.nodes[id].depth, Int[]), id)
    end
    pos = Dict{Int,Float64}()
    for (d, level) in bydepth
        n = length(level)
        for (k, id) in enumerate(sort(level))
            pos[id] = n == 1 ? 0.5 : (k - 1) / (n - 1)
        end
    end
    for id in ids
        n = s.nodes[id]
        n.parent == 0 && continue
        haskey(pos, n.parent) || continue
        lines!(ax, [pos[n.parent], pos[id]],
            [Float64(n.depth - 1), Float64(n.depth)];
            color=RGBAf(0.7, 0.7, 0.7, 0.7), linewidth=0.8)
    end
    vis = [Float64(s.nodes[id].visits) for id in ids]
    scatter!(ax, [pos[id] for id in ids], [Float64(s.nodes[id].depth) for id in ids];
        markersize=4 .+ 14 .* sqrt.(vis ./ max(maximum(vis), 1)),
        color=SEARCH_NODE)
    sel = [id for id in ids if s.nodes[id].action == s.selected_action &&
           s.nodes[id].depth == 1]
    isempty(sel) || scatter!(ax, [pos[id] for id in sel],
        [1.0 for _ in sel]; color=SEARCH_SELECTED, marker=:circle,
        markersize=13, strokewidth=1.5, strokecolor=SEARCH_SELECTED)

    # --- root action summary ---------------------------------------------
    ax2 = Axis(fig[2, 2]; title="root actions, grouped by action",
        subtitle="visits per action — what actually received search effort",
        xlabel="visits")
    rc = root_children(s)
    byaction = Dict{Any,Int}()
    for c in rc
        byaction[c.action] = max(get(byaction, c.action, 0), c.visits)
    end
    acts = sort(collect(keys(byaction)); by=a -> -byaction[a])
    labels = [string(a) for a in acts]
    if length(labels) > 12
        acts = acts[1:12]
        labels = vcat(labels[1:12])
    end
    counts = [Float64(byaction[a]) for a in acts]
    cols = [a == s.selected_action ? SEARCH_SELECTED : SEARCH_NODE for a in acts]
    barplot!(ax2, 1:length(acts), counts; direction=:x, color=cols, width=0.7)
    ax2.yticks = (1:length(acts), _short.(labels))
    ax2.yticklabelsize = 9

    # --- provenance and statistics ---------------------------------------
    ax3 = Axis(fig[3, 1:2])
    hidedecorations!(ax3)
    hidespines!(ax3)
    text!(ax3, 0.01, 0.5; text=search_summary(s), align=(:left, :center),
        fontsize=9, color=RGBAf(0.3, 0.3, 0.3, 1))
    limits!(ax3, 0, 1, 0, 1)
    rowsize!(fig.layout, 3, Relative(0.30))
    return fig
end

_short(s::AbstractString) = length(s) <= 26 ? s : s[1:23] * "..."

"""
    render_search_action_plane(snapshot; ...) -> Figure

FJ9.5d — the continuous root actions a search actually sampled.

`x = v_cmd`, `y = omega_cmd`, marker size = visits, marker colour = the
search's value estimate. **Only sampled actions are drawn.** There is no
interpolation and no heatmap: with a couple of dozen samples a smoothed
surface would imply the planner evaluated the whole action plane, which it did
not. The action box is drawn so the sampled region can be judged against it.
"""
function DDM.render_search_action_plane(s::SearchSnapshot;
    figure_size=(720, 620), v_bounds=(0.0, 0.41), omega_bounds=(-1.5, 1.5))
    rc = filter(c -> c.action isa DuckieAction, root_children(s))
    isempty(rc) && throw(ArgumentError(
        "this snapshot has no continuous root actions; the action plane is " *
        "for continuous-action searches"))
    st = search_statistics(s)

    fig = Figure(size=figure_size)
    Label(fig[1, 1], "$(s.solver) — continuous root actions actually sampled";
        fontsize=14, font=:bold, tellwidth=false)
    ax = Axis(fig[2, 1]; xlabel="v_cmd (m/s)", ylabel="omega_cmd (rad/s)",
        title="$(st.root_actions) sampled actions, " *
              "$(round(100 * st.single_visit_fraction; digits=1))% visited once",
        subtitle="marker area ~ visits; no interpolation — only sampled points")

    # the action box, so the sampled region is judged against the real bounds
    vb, wb = v_bounds, omega_bounds
    lines!(ax, [vb[1], vb[2], vb[2], vb[1], vb[1]],
        [wb[1], wb[1], wb[2], wb[2], wb[1]];
        color=RGBAf(0.4, 0.4, 0.4, 0.8), linestyle=:dash)

    xs = [c.action.v for c in rc]
    ys = [c.action.omega for c in rc]
    vis = [Float64(c.visits) for c in rc]
    vals = [c.value === missing ? NaN : c.value for c in rc]
    sc = scatter!(ax, xs, ys;
        markersize=8 .+ 18 .* (vis ./ max(maximum(vis), 1)),
        color=vals, colormap=:viridis, strokewidth=0.5,
        strokecolor=RGBAf(0.2, 0.2, 0.2, 0.8))
    Colorbar(fig[2, 2], sc; label="search value estimate")

    if s.selected_action isa DuckieAction
        scatter!(ax, [s.selected_action.v], [s.selected_action.omega];
            marker=:xcross, markersize=18, color=SEARCH_SELECTED,
            strokewidth=2, strokecolor=SEARCH_SELECTED)
    end
    limits!(ax, vb[1] - 0.02, vb[2] + 0.02, wb[1] - 0.1, wb[2] + 0.1)

    ax3 = Axis(fig[3, 1:2])
    hidedecorations!(ax3)
    hidespines!(ax3)
    text!(ax3, 0.01, 0.5; text=search_summary(s), align=(:left, :center),
        fontsize=9, color=RGBAf(0.3, 0.3, 0.3, 1))
    limits!(ax3, 0, 1, 0, 1)
    rowsize!(fig.layout, 3, Relative(0.33))
    return fig
end

# ---------------------------------------------------------------------------
# FJ9.6c/d — diagnostic time series
# ---------------------------------------------------------------------------

const DIAG_EVENT_COLOR = Dict(
    "full_stop" => RGBAf(0.13, 0.53, 0.20, 1.0),
    "passed_stop" => RGBAf(0.20, 0.45, 0.70, 1.0),
    "stop_violation" => RGBAf(0.78, 0.13, 0.13, 1.0),
    "offroad" => RGBAf(0.55, 0.10, 0.55, 1.0),
    "other_collision" => RGBAf(0.35, 0.10, 0.10, 1.0),
    "duck_collision" => RGBAf(0.95, 0.35, 0.05, 1.0),
    "goal" => RGBAf(0.10, 0.55, 0.55, 1.0),
    "timeout" => RGBAf(0.45, 0.45, 0.45, 1.0))

const DIAG_MISSING = RGBAf(0.72, 0.72, 0.72, 0.9)
const DIAG_SERIES_COLORS = (EGO_COLOR, STOP_COLOR, DUCK_ACTIVE,
    RGBAf(0.13, 0.53, 0.20, 1), RGBAf(0.45, 0.25, 0.65, 1),
    RGBAf(0.30, 0.30, 0.30, 1), RGBAf(0.70, 0.45, 0.10, 1))

_diag_line(v) = Float64[x === missing ? NaN : x for x in v]
_diag_step(xs) = length(xs) < 2 ? 1.0 : xs[2] - xs[1]

# A missing value leaves a gap in the line AND gets a mark of its own. A gap
# alone reads as "nothing happened here", which is a third meaning neither
# missing nor zero.
function _diag_plot!(ax, xs, s, color)
    ys = _diag_line(s.values)
    lines!(ax, xs, ys; color=color, linewidth=1.6, label=s.name)
    gaps = [k for k in eachindex(s.values) if s.values[k] === missing]
    if !isempty(gaps)
        finite = filter(!isnan, ys)
        lo = isempty(finite) ? 0.0 : minimum(finite)
        hi = isempty(finite) ? 1.0 : maximum(finite)
        band = lo == hi ? lo - 0.05 : lo - 0.06 * (hi - lo)
        scatter!(ax, xs[gaps], fill(band, length(gaps)); marker=:vline,
            markersize=9, color=DIAG_MISSING, label="$(s.name): MISSING")
    end
    return ax
end

function _diag_events!(ax, ep, xs)
    for (name, ks) in ep.events
        c = get(DIAG_EVENT_COLOR, name, RGBAf(0.3, 0.3, 0.3, 1.0))
        vlines!(ax, [xs[k] for k in ks]; color=c, linewidth=1.2,
            linestyle=:dash, label=name)
    end
    return ax
end

function _diag_flag!(ax, xs, s, color)
    on = [k for k in eachindex(s.values) if s.values[k] == 1.0]
    isempty(on) && return ax
    h = 0.5 * _diag_step(xs)
    vspan!(ax, [xs[k] - h for k in on], [xs[k] + h for k in on];
        color=RGBAf(color.r, color.g, color.b, 0.16), label=s.name)
    return ax
end

"""
    render_diagnostics(episode; ...) -> Figure

FJ9.6c. One episode in six sections: navigation, motion and command, stop
subsystem, duck subsystem, reward, and computational cost.

**Within a section there is one axis per unit.** The section is the reader's
grouping; the axis is the physical one, and the two are not the same. Drawing
`d` in metres, `phi` in radians and `kappa` in 1/m together looks compact and
is a lie about scale: kappa reaches 2.15 on a straight-then-curve episode and
flattens a lateral offset that never leaves ±0.15 m. The same applies to
`planning_time` in seconds beside `model_calls` in the thousands, where the
seconds axis would read as a flat line at zero.

Flags are shaded across every axis of their section, because a flag is a
state the episode was in rather than a value with a scale of its own.

Event markers come from `episode.events` — the decision indices the log
recorded — and from nothing else.
"""
function DDM.render_diagnostics(ep::DDM.EpisodeDiagnostics;
    figure_size=nothing, mode::DDM.AxisMode=DDM.ABSOLUTE_DECISION,
    title::AbstractString="")
    xs = mode === DDM.ABSOLUTE_DECISION ? Float64.(ep.decisions) : ep.progress
    xlab = mode === DDM.ABSOLUTE_DECISION ? "decision index" :
        "normalised episode progress (NOT time)"

    sections = (("Navigation", DDM.NAVIGATION),
        ("Motion and command", DDM.MOTION_COMMAND),
        ("Stop subsystem", DDM.STOP_SUBSYS),
        ("Duck subsystem", DDM.DUCK_SUBSYS),
        ("Reward", DDM.REWARD),
        ("Computational cost", DDM.COMPUTE))

    # (section title, unit, series drawn as lines, flags shaded behind them)
    groups = Tuple{String,String,Vector{DDM.DiagnosticSeries},
        Vector{DDM.DiagnosticSeries}}[]
    for (name, cat) in sections
        ss = DDM.series_in(ep, cat)
        isempty(ss) && continue
        flags = filter(s -> s.kind === DDM.FLAG, ss)
        drawn = filter(s -> s.kind !== DDM.FLAG, ss)
        for u in unique(s.unit for s in drawn)
            push!(groups, (name, u, filter(s -> s.unit == u, drawn), flags))
        end
        isempty(drawn) && push!(groups, (name, "", DDM.DiagnosticSeries[],
            flags))
    end

    # the canvas is sized from the number of axes, so adding a unit to a
    # section grows the figure instead of silently clipping the top of it
    fig = Figure(size = figure_size === nothing ?
        (1120, 150 * length(groups) + 210) : figure_size)

    Label(fig[1, 1:2], isempty(title) ?
        "$(ep.solver) · seed $(ep.seed) · $(length(ep.decisions)) decisions" :
        title; fontsize=15, font=:bold, halign=:left)
    Label(fig[2, 1:2],
        "outcome: $(ep.outcome) ($(ep.reason))   ·   drawn from the frozen " *
        "FJ8.4c decision log — no policy, planner or environment was run   ·  " *
        " one axis per unit; a shared axis across units would destroy scale";
        fontsize=10, halign=:left, color=RGBAf(0.35, 0.35, 0.35, 1))

    for (j, (name, unit, drawn, flags)) in enumerate(groups)
        row = 2 + j
        ylab = isempty(unit) ? (isempty(drawn) ? "state" : "dimensionless") :
            unit
        ax = Axis(fig[row, 1]; titlesize=11, titlealign=:left,
            title=isempty(unit) ? name : "$name · $unit", ylabel=ylab,
            xlabel=j == length(groups) ? xlab : "")
        for (k, s) in enumerate(flags)
            _diag_flag!(ax, xs, s,
                DIAG_SERIES_COLORS[mod1(k, length(DIAG_SERIES_COLORS))])
        end
        for (k, s) in enumerate(drawn)
            _diag_plot!(ax, xs, s,
                DIAG_SERIES_COLORS[mod1(k, length(DIAG_SERIES_COLORS))])
        end
        _diag_events!(ax, ep, xs)
        # the legend sits beside the axis, never over the data, and folds into
        # a second bank rather than growing its row taller than the plot
        entries = length(drawn) + length(flags) + length(ep.events)
        Legend(fig[row, 2], ax; framevisible=false, labelsize=8, merge=true,
            nbanks=entries > 7 ? 2 : 1, padding=(4, 4, 0, 0))
    end

    ax = Axis(fig[3 + length(groups), 1:2])
    hidedecorations!(ax)
    hidespines!(ax)
    text!(ax, 0.0, 0.5;
        text=join(DDM.diagnostics_provenance(ep), "\n") * "\nfigure " *
             DDM.diagnostics_fingerprint(ep;
                 fields=[s.name for s in ep.series], mode=mode),
        align=(:left, :center), fontsize=8, color=RGBAf(0.4, 0.4, 0.4, 1))
    limits!(ax, 0, 1, 0, 1)
    rowsize!(fig.layout, 3 + length(groups), Relative(0.055))
    colsize!(fig.layout, 2, Relative(0.17))
    return fig
end

"""
    render_diagnostics_aggregate(log, solvers; ...) -> Figure

FJ9.6d. One panel per solver, binned by normalised episode progress.

Each bin shows its median and interquartile range and carries its own `n`.
There is no interpolation onto a common grid: an episode that ended at
decision 42 contributes to the bins it reached and to no others, so a bin only
the surviving episodes reach is visibly thinner evidence rather than an
equally confident point.
"""
function DDM.render_diagnostics_aggregate(log::DDM.DecisionLog,
    solvers::AbstractVector{<:AbstractString};
    column::AbstractString="model_calls", bins::Integer=5,
    figure_size=(1080, 480), ylabel::AbstractString=column)
    fig = Figure(size=figure_size)
    Label(fig[1, 1:length(solvers)],
        "$column by normalised episode progress · $bins bins · " *
        "median and IQR, no interpolation"; fontsize=14, font=:bold,
        halign=:left)

    per = [DDM.progress_bins(log, s, column; bins=bins) for s in solvers]
    allv = Float64[]
    for b in per, x in b
        append!(allv, (x.q25, x.median, x.q75))
    end
    lo, hi = isempty(allv) ? (0.0, 1.0) : extrema(allv)
    pad = hi == lo ? 1.0 : 0.08 * (hi - lo)

    for (j, s) in enumerate(solvers)
        b = per[j]
        lens = DDM.episode_lengths(log, s)
        ax = Axis(fig[2, j]; titlesize=10, xlabel="progress",
            ylabel=j == 1 ? ylabel : "",
            title="$s\n$(length(lens)) episodes, length " *
                  "$(minimum(lens))–$(maximum(lens))")
        xs = [0.5 * (x.from + x.to) for x in b]
        rangebars!(ax, xs, [x.q25 for x in b], [x.q75 for x in b];
            color=EGO_COLOR, whiskerwidth=8)
        scatterlines!(ax, xs, [x.median for x in b]; color=EGO_COLOR,
            markersize=7)
        for (k, x) in enumerate(b)
            text!(ax, xs[k], hi + 0.35 * pad; text="n=$(x.n)", fontsize=7,
                align=(:center, :bottom), color=RGBAf(0.4, 0.4, 0.4, 1))
        end
        if !isempty(b) && first(b).absent > 0
            text!(ax, 0.02, lo - 0.75 * pad;
                text="$(first(b).absent) rows MISSING, excluded (not zeroed)",
                fontsize=7, align=(:left, :bottom),
                color=RGBAf(0.6, 0.3, 0.3, 1))
        end
        limits!(ax, 0.0, 1.0, lo - pad, hi + 1.5 * pad)
    end

    Label(fig[3, 1:length(solvers)],
        "source: FJ8.4c decisions.csv  fingerprint $(log.fingerprint)   ·   " *
        "$(length(log)) decisions over $(length(log.episodes)) episodes";
        fontsize=8, halign=:left, color=RGBAf(0.4, 0.4, 0.4, 1))
    return fig
end

# ---------------------------------------------------------------------------
# FJ9.7 — artifact-driven animation
# ---------------------------------------------------------------------------
#
# One layout function builds the figure with Observables and returns a
# `setframe!` closure. `render_frame` calls it once; `render_animation` drives
# it through `record`. Two code paths drawing "the same" frame differently is
# exactly the kind of divergence that makes a still disagree with its video.

const ANIM_HISTORY = RGBAf(0.11, 0.35, 0.72, 0.85)
const ANIM_NOW = RGBAf(0.78, 0.13, 0.13, 1.0)
const ANIM_FROZEN = RGBAf(0.45, 0.45, 0.45, 1.0)

# (title, unit, series names) — one axis per unit, the FJ9.6 rule
const ANIM_PANELS = (
    ("lateral offset", "m", ("d",)),
    ("heading error", "rad", ("phi",)),
    ("speed", "m/s", ("ego_speed", "v_cmd")),
    ("angular rate", "rad/s", ("omega_cmd",)),
    ("distance to stop line", "m", ("d_stop",)),
    ("cumulative return", "", ("cumulative_return",)),
    ("generative calls", "calls", ("model_calls",)),
    ("planning time", "s", ("planning_time",)),
)

_anim_pts(xs, ys) = Point2f[Point2f(x, y === missing ? NaN : y)
                            for (x, y) in zip(xs, ys)]

"""
    _animation_panel(fig, cell, seq, title, unit, names) -> setter

One history axis. Returns a closure that redraws it for frame `t` using only
`series_through`, so it is structurally incapable of showing the future.
"""
function _animation_panel(fig, cell, seq, title, unit, names; xmax=nothing)
    ax = Axis(fig[cell...]; title="$title" * (isempty(unit) ? "" : " ($unit)"),
        titlesize=9, xlabelvisible=false, ylabelvisible=false,
        xticklabelsize=8, yticklabelsize=8)
    # The x range is the episode's own length — a property of the record, not
    # a hint about where the trajectory goes. In a paired figure the caller
    # passes the SHARED span instead, because two panels on the absolute
    # decision index must put decision 40 at the same place on the page.
    xlims!(ax, 0.5, (xmax === nothing ? length(seq) : xmax) + 0.5)
    obs = [Observable(Point2f[]) for _ in names]
    for (k, o) in enumerate(obs)
        lines!(ax, o; color=DIAG_SERIES_COLORS[mod1(k, 7)], linewidth=1.4)
    end
    gap = Observable(Point2f[])
    scatter!(ax, gap; marker=:vline, markersize=7, color=DIAG_MISSING)
    now = Observable(Point2f[])
    scatter!(ax, now; markersize=7, color=ANIM_NOW)
    # event markers are line SEGMENTS spanning the current data range, not
    # `vlines!`: an empty `vlines!` makes Makie's data_limits reduce over an
    # empty collection, and a panel legitimately has no events for most of
    # the episode
    evseg = Observable(Point2f[])
    linesegments!(ax, evseg; color=RGBAf(0.55, 0.10, 0.55, 0.7),
        linewidth=1.0, linestyle=:dash)

    return function (t)
        xs = 1:t
        lo, hi = Inf, -Inf
        for (k, nm) in enumerate(names)
            ys = series_through(seq, t, nm)
            obs[k][] = _anim_pts(xs, ys)
            for y in ys
                y === missing && continue
                lo = min(lo, y); hi = max(hi, y)
            end
        end
        ys1 = series_through(seq, t, names[1])
        base = isfinite(lo) ? lo : 0.0
        gap[] = Point2f[Point2f(k, base) for k in xs if ys1[k] === missing]
        last = ys1[t]
        now[] = last === missing ? Point2f[] : Point2f[Point2f(t, last)]
        # a panel whose every value so far is MISSING still gets an explicit
        # range, so it renders as an empty axis rather than falling through to
        # autolimits — and so the reader sees "no data yet", not "zero"
        if !isfinite(lo) || !isfinite(hi)
            lo, hi = -1.0, 1.0
        end
        pad = hi == lo ? max(abs(hi), 1.0) * 0.1 : 0.12 * (hi - lo)
        ylims!(ax, lo - pad, hi + pad)
        evseg[] = reduce(vcat,
            [[Point2f(v, lo - pad), Point2f(v, hi + pad)]
             for (_, v) in events_through(seq, t)]; init=Point2f[])
        return nothing
    end
end

"""
    _animation_world(fig, cell, sw, seq; label) -> setter

The world panel: static track drawn once, everything that moves taken from
the logged pose. No duck is drawn — its world-frame position is ABSENT.
"""
function _animation_world(fig, cell, sw::StaticWorld, seq::AnimationSequence;
    label::AbstractString="")
    ax = Axis(fig[cell...]; aspect=DataAspect(), title=label, titlesize=11,
        xlabel="x (m)", ylabel="z (m)", xlabelsize=9, ylabelsize=9,
        xticklabelsize=8, yticklabelsize=8)
    for tp in sw.tiles
        xs, ys = _xy(tp.corners)
        poly!(ax, Point2f.(xs, ys);
            color=tp.drivable ? TILE_DRIVABLE : TILE_OTHER,
            strokewidth=0.4, strokecolor=RGBAf(0.75, 0.75, 0.75, 1))
    end
    for cl in sw.lane_centrelines
        xs, ys = _xy(cl)
        lines!(ax, xs, ys; color=LANE_COLOR, linewidth=0.8, linestyle=:dot)
    end
    isempty(sw.sign_positions) || scatter!(ax,
        [p[1] for p in sw.sign_positions], [p[2] for p in sw.sign_positions];
        marker=:octagon, markersize=13, color=STOP_COLOR)

    traj = Observable(Point2f[])
    lines!(ax, traj; color=TRACK_COLOR, linewidth=2)
    foot = Observable(Point2f[])
    poly!(ax, foot; color=RGBAf(0.11, 0.35, 0.72, 0.35), strokewidth=1.4,
        strokecolor=EGO_COLOR)
    stopseg = Observable(Point2f[])
    linesegments!(ax, stopseg; color=STOP_COLOR, linewidth=2.5)
    head = Observable(Point2f[])
    linesegments!(ax, head; color=HEADING_COLOR, linewidth=2)
    banner = Observable("")
    text!(ax, sw.view_extent[1] + 0.02, sw.view_extent[4] - 0.06;
        text=banner, align=(:left, :top), fontsize=10, color=ANIM_NOW)
    limits!(ax, sw.view_extent[1], sw.view_extent[2], sw.view_extent[3],
        sw.view_extent[4])

    return function (t; frozen::Bool=false)
        sc = frame_scene(sw, seq, t)
        traj[] = Point2f.(first.(sc.trajectory), last.(sc.trajectory))
        foot[] = Point2f.(first.(sc.ego_footprint), last.(sc.ego_footprint))
        stopseg[] = reduce(vcat, [[Point2f(l[1], l[2]), Point2f(l[3], l[4])]
                                  for l in sc.stop_lines]; init=Point2f[])
        p = sc.ego_position
        head[] = [Point2f(p[1], p[2]),
            Point2f(p[1] + 0.12 * sc.ego_heading[1],
                p[2] + 0.12 * sc.ego_heading[2])]
        banner[] = frozen ?
            (seq.outcome === ENV_TERMINATED ?
             "FROZEN — TERMINATED at decision $(length(seq)) ($(seq.reason))" :
             "FROZEN — HORIZON at decision $(length(seq))") : ""
        return nothing
    end
end

"""
    animation_figure(static, sequence; ...) -> (figure, setframe!)

Build the playback layout once. `setframe!(t)` shows decision `t` and the
history up to it.
"""
function animation_figure(sw::StaticWorld, seq::AnimationSequence;
    figure_size=(1500, 980), title::AbstractString="")
    fig = Figure(size=figure_size)
    head = isempty(title) ?
        "$(seq.solver) · seed $(seq.seed) · $(ANIMATION_TIMELINE_LABEL)" : title
    Label(fig[1, 1:3], head; fontsize=15, font=:bold, halign=:left)

    grid = fig[2, 1] = GridLayout()
    setworld = _animation_world(grid, (1, 1), sw, seq)
    info = Observable("")
    axi = Axis(fig[2, 2])
    hidedecorations!(axi)
    hidespines!(axi)
    text!(axi, 0.0, 1.0; text=info, align=(:left, :top), fontsize=10,
        color=RGBAf(0.2, 0.2, 0.2, 1))
    limits!(axi, 0, 1, 0, 1)

    panels = fig[2, 3] = GridLayout()
    setters = [_animation_panel(panels, (cld(k, 2), mod1(k, 2)), seq, ttl, u, n)
               for (k, (ttl, u, n)) in enumerate(ANIM_PANELS)]

    Label(fig[3, 1:3],
        join(animation_provenance(seq), "   ·   ") * "\n" *
        join(animation_absent_lines(), "   ·   "); fontsize=7, halign=:left,
        color=RGBAf(0.45, 0.45, 0.45, 1))

    colsize!(fig.layout, 1, Relative(0.40))
    colsize!(fig.layout, 2, Relative(0.16))
    rowsize!(fig.layout, 3, Relative(0.09))

    setframe! = function (t::Integer; frozen::Bool=false)
        setworld(t; frozen=frozen)
        for s in setters
            s(t)
        end
        f = seq.frames[t]
        info[] = join(frame_caption(seq, t), "\n") * "\n\n" *
            "d_stop: " * (f.d_stop === missing ? "MISSING (no candidate)" :
                string(round(f.d_stop; digits = 3)) * " m") * "\n" *
            "sigma_stop: $(f.sigma_stop)   hold: $(f.stop_hold_progress)\n" *
            "duck present: $(f.duck_present)   active: $(f.duck_active)\n" *
            "duck (lane frame): long $(round(f.duck_longitudinal; digits = 2))" *
            "  lat $(round(f.duck_lateral; digits = 2))\n" *
            "frame $(f.fingerprint)"
        return nothing
    end
    setframe!(1)
    return fig, setframe!
end

"""
    render_frame(static, sequence, t; ...) -> Figure

FJ9.7b. One frame, built by the same layout the video uses.
"""
function DDM.render_frame(sw::StaticWorld, seq::AnimationSequence,
    t::Integer; kwargs...)
    fig, setframe! = animation_figure(sw, seq; kwargs...)
    setframe!(t)
    return fig
end

"""
    render_animation(static, sequence, path; framerate) -> String

FJ9.7. Write the playback to `path`. One logged decision is one frame.

`framerate` is a *display* rate. It carries no claim about elapsed world
time — the artefact records none — and the caption says so.
"""
function DDM.render_animation(sw::StaticWorld, seq::AnimationSequence,
    path::AbstractString; framerate::Integer=10, kwargs...)
    fig, setframe! = animation_figure(sw, seq; kwargs...)
    Makie.record(fig, path, 1:length(seq); framerate=framerate) do t
        setframe!(t)
    end
    return path
end

"""
    render_paired_animation(static, a, b, path; framerate) -> String

FJ9.7d. Two solvers on the same seed, on the **absolute decision index**.

The shorter episode's panel freezes on its terminal frame with a banner. It
is not looped, restarted, or stretched: decision 40 on the left must be
decision 40 on the right, or the pairing the experiment was built on is gone.
"""
function paired_animation_figure(sw::StaticWorld, a::AnimationSequence,
    b::AnimationSequence; figure_size=(1500, 900))
    a.seed == b.seed || throw(ArgumentError(
        "paired animation requires the same seed: got $(a.seed) and $(b.seed)"))
    n = paired_frames(a, b)
    fig = Figure(size=figure_size)
    Label(fig[1, 1:2],
        "$(a.solver) vs $(b.solver) · seed $(a.seed) · " *
        "$(ANIMATION_TIMELINE_LABEL), absolute decision index";
        fontsize=15, font=:bold, halign=:left)
    cur = Observable("")
    Label(fig[2, 1:2], cur; fontsize=10, halign=:left,
        color=RGBAf(0.3, 0.3, 0.3, 1))

    setters = Function[]
    for (j, s) in enumerate((a, b))
        g = fig[3, j] = GridLayout()
        sw_ = _animation_world(g, (1, 1), sw, s;
            label="$(s.solver) — $(length(s)) decisions, $(s.outcome)")
        ps = [_animation_panel(g, (1 + cld(k, 2), mod1(k, 2)), s, ttl, u, nm;
                  xmax=n)
              for (k, (ttl, u, nm)) in enumerate(ANIM_PANELS[1:4])]
        push!(setters, function (t)
            k = frame_index(s, t)
            sw_(k; frozen=is_frozen(s, t))
            for p in ps
                p(k)
            end
        end)
        rowsize!(g, 1, Relative(0.55))
    end
    # initialise before anything renders: an empty `poly!` observable is a
    # BoundsError the first time Makie lays the figure out
    for s in setters
        s(1)
    end
    Label(fig[4, 1:2],
        "left: " * join(animation_provenance(a), " · ") * "\nright: " *
        join(animation_provenance(b), " · "); fontsize=7, halign=:left,
        color=RGBAf(0.45, 0.45, 0.45, 1))
    rowsize!(fig.layout, 4, Relative(0.08))

    setframe! = function (t::Integer)
        for s in setters
            s(t)
        end
        cur[] = "decision $t of $n   ·   " *
            "$(a.solver): " * (is_frozen(a, t) ?
                "ended at $(length(a)) ($(a.reason))" : "running") *
            "   ·   $(b.solver): " * (is_frozen(b, t) ?
                "ended at $(length(b)) ($(b.reason))" : "running")
        return nothing
    end
    setframe!(1)
    return fig, setframe!, n
end

function DDM.render_paired_animation(sw::StaticWorld, a::AnimationSequence,
    b::AnimationSequence, path::AbstractString; framerate::Integer=10,
    kwargs...)
    fig, setframe!, n = paired_animation_figure(sw, a, b; kwargs...)
    Makie.record(fig, path, 1:n; framerate=framerate) do t
        setframe!(t)
    end
    return path
end

"""
    render_frame(static, a, b, t; ...) -> Figure

The paired analogue of [`render_frame`](@ref): one still of the side-by-side
layout at absolute decision `t`, so the freeze banner can be inspected
without decoding a video.
"""
function DDM.render_frame(sw::StaticWorld, a::AnimationSequence,
    b::AnimationSequence, t::Integer; kwargs...)
    fig, setframe!, _ = paired_animation_figure(sw, a, b; kwargs...)
    setframe!(t)
    return fig
end

# ---------------------------------------------------------------------------
# FJ9.8 — publication composites
# ---------------------------------------------------------------------------
#
# Panels are drawn from the payload objects, into cells of one figure. That is
# what makes the output a real vector document: one font stack, one axis
# style, no rescaled rasters, and a caption generated from the same data the
# panels were drawn from.
#
# Every panel is also required to survive greyscale. Colour is never the only
# channel: markers, linestyles and text annotations carry the same
# distinctions, because a printed figure loses hue and keeps everything else.

const PUB_SEQ = :viridis
const PUB_GREY = (RGBAf(0.15, 0.15, 0.15, 1), RGBAf(0.45, 0.45, 0.45, 1),
    RGBAf(0.70, 0.70, 0.70, 1))
const PUB_MARKERS = (:circle, :rect, :utriangle, :diamond, :cross, :xcross,
    :star5)
const PUB_STYLES = (:solid, :dash, :dot, :dashdot)

_pub_title!(ax, id, title) = (ax.title = "($id) $title"; ax.titlesize = 10;
    ax.titlealign = :left; ax)

function _panel_world!(fig, cell, p)
    w = p.payload
    ax = Axis(fig[cell...]; aspect=DataAspect(), xlabel="x (m)",
        ylabel="z (m)", xlabelsize=8, ylabelsize=8, xticklabelsize=7,
        yticklabelsize=7)
    _pub_title!(ax, p.id, p.title)
    for tp in w.tiles
        xs, ys = _xy(tp.corners)
        poly!(ax, Point2f.(xs, ys);
            color=tp.drivable ? TILE_DRIVABLE : TILE_OTHER,
            strokewidth=0.4, strokecolor=RGBAf(0.75, 0.75, 0.75, 1))
    end
    for cl in w.lane_centrelines
        xs, ys = _xy(cl)
        lines!(ax, xs, ys; color=LANE_COLOR, linewidth=0.8, linestyle=:dot)
    end
    for l in w.stop_lines
        lines!(ax, [l[1], l[3]], [l[2], l[4]]; color=STOP_COLOR, linewidth=2)
    end
    isempty(w.stop_signs) || scatter!(ax, first.(w.stop_signs),
        last.(w.stop_signs); marker=:octagon, markersize=10, color=STOP_COLOR)
    for (k, d) in enumerate(w.ducks)
        scatter!(ax, [d[1]], [d[2]];
            marker=w.duck_active[k] ? :utriangle : :circle, markersize=9,
            color=w.duck_active[k] ? DUCK_ACTIVE : DUCK_IDLE)
    end
    xs, ys = _xy(w.ego_footprint)
    poly!(ax, Point2f.(xs, ys); color=RGBAf(0.11, 0.35, 0.72, 0.35),
        strokewidth=1.4, strokecolor=EGO_COLOR)
    p0 = w.ego_position
    lines!(ax, [p0[1], p0[1] + 0.12 * w.ego_heading[1]],
        [p0[2], p0[2] + 0.12 * w.ego_heading[2]]; color=HEADING_COLOR,
        linewidth=2)
    limits!(ax, w.view_extent[1], w.view_extent[2], w.view_extent[3],
        w.view_extent[4])
    return ax
end

function _panel_projection!(fig, cell, p)
    s = p.payload
    ax = Axis(fig[cell...])
    hidedecorations!(ax)
    hidespines!(ax)
    _pub_title!(ax, p.id, p.title)
    txt = IOBuffer()
    for e in s.entries
        println(txt, rpad(string(e.field), 22), rpad(e.display, 14),
            rpad(e.unit, 7), e.privilege)
    end
    text!(ax, 0.0, 1.0; text=String(take!(txt)), align=(:left, :top),
        fontsize=7, font=:regular, color=RGBAf(0.15, 0.15, 0.15, 1))
    limits!(ax, 0, 1, 0, 1)
    return ax
end

function _panel_observability!(fig, cell, p)
    items = p.payload
    ax = Axis(fig[cell...])
    hidedecorations!(ax)
    hidespines!(ax)
    _pub_title!(ax, p.id, p.title)
    txt = IOBuffer()
    for o in items
        println(txt, rpad(string(o.name), 22), o.class)
    end
    text!(ax, 0.0, 1.0; text=String(take!(txt)), align=(:left, :top),
        fontsize=7, color=RGBAf(0.15, 0.15, 0.15, 1))
    limits!(ax, 0, 1, 0, 1)
    return ax
end

function _panel_slice_action!(fig, cell, p)
    sl = p.payload
    ax = Axis(fig[cell...]; xlabel=string(sl.x.field), ylabel=string(sl.y.field),
        xlabelsize=8, ylabelsize=8, xticklabelsize=7, yticklabelsize=7)
    _pub_title!(ax, p.id, p.title)
    ids = Float64[c.action_id for c in sl.cells]
    heatmap!(ax, sl.x.values, sl.y.values, ids; colormap=:tab10)
    # greyscale survival: the action boundaries are also drawn as contours, so
    # the panel still separates regions without hue
    contour!(ax, sl.x.values, sl.y.values, ids; levels=8,
        color=RGBAf(0.1, 0.1, 0.1, 0.55), linewidth=0.5)
    return ax
end

function _panel_slice_value!(fig, cell, p)
    sl = p.payload
    ax = Axis(fig[cell...]; xlabel=string(sl.x.field), ylabel=string(sl.y.field),
        xlabelsize=8, ylabelsize=8, xticklabelsize=7, yticklabelsize=7)
    _pub_title!(ax, p.id, p.title)
    hm = heatmap!(ax, sl.x.values, sl.y.values, value_surface(sl);
        colormap=PUB_SEQ)
    Colorbar(fig[cell[1], cell[2] + 1], hm; label="V(s)", labelsize=8,
        ticklabelsize=7, width=10)
    # tied cells are hatched with a marker, not shaded: on a greyscale print a
    # second colour scale on top of a value map is unreadable
    ties = tie_surface(sl)
    pts = Point2f[]
    for (i, xv) in enumerate(sl.x.values), (j, yv) in enumerate(sl.y.values)
        ties[i, j] > 1 && push!(pts, Point2f(xv, yv))
    end
    isempty(pts) || scatter!(ax, pts; marker=:xcross, markersize=2.2,
        color=RGBAf(0.95, 0.95, 0.95, 0.85))
    return ax
end

function _panel_slice_continuous!(fig, cell, p)
    sl = p.payload
    ax = Axis(fig[cell...]; xlabel=string(sl.x.field), ylabel=string(sl.y.field),
        xlabelsize=8, ylabelsize=8, xticklabelsize=7, yticklabelsize=7)
    _pub_title!(ax, p.id, p.title)
    surf = occursin("omega", p.title) ? omega_surface(sl) : v_surface(sl)
    hm = heatmap!(ax, sl.x.values, sl.y.values, surf; colormap=PUB_SEQ)
    contour!(ax, sl.x.values, sl.y.values, surf; levels=6,
        color=RGBAf(0.1, 0.1, 0.1, 0.5), linewidth=0.5)
    Colorbar(fig[cell[1], cell[2] + 1], hm;
        label=occursin("omega", p.title) ? "rad/s" : "m/s", labelsize=8,
        ticklabelsize=7, width=10)
    return ax
end

function _panel_search_tree!(fig, cell, p)
    s = p.payload
    ax = Axis(fig[cell...]; xlabel="root action", ylabel="visits",
        xlabelsize=8, ylabelsize=8, xticklabelsize=7, yticklabelsize=7)
    _pub_title!(ax, p.id, p.title)
    rc = root_children(s)
    # `SearchNode.action` is deliberately `Any` — the snapshot is solver-neutral
    # and holds whatever action type the model uses, MacroAction or DuckieAction
    byact = Dict{String,Int}()
    for c in rc
        k = string(c.action)
        byact[k] = get(byact, k, 0) + c.visits
    end
    ks = sort!(collect(keys(byact)); by=k -> -byact[k])
    vs = Float64[byact[k] for k in ks]
    barplot!(ax, 1:length(vs), vs; color=PUB_GREY[2],
        strokewidth=0.6, strokecolor=PUB_GREY[1])
    ax.xticks = (1:length(ks), [length(k) > 9 ? k[1:9] : k for k in ks])
    ax.xticklabelrotation = pi / 4
    ax.xticklabelsize = 6
    return ax
end

function _panel_search_summary!(fig, cell, p)
    s = p.payload
    st = search_statistics(s)
    ax = Axis(fig[cell...])
    hidedecorations!(ax)
    hidespines!(ax)
    _pub_title!(ax, p.id, p.title)
    text!(ax, 0.0, 1.0; text=search_summary(s), align=(:left, :top),
        fontsize=7, color=RGBAf(0.15, 0.15, 0.15, 1))
    limits!(ax, 0, 1, 0, 1)
    return ax
end

function _panel_action_plane!(fig, cell, p)
    s = p.payload
    ax = Axis(fig[cell...]; xlabel="v (m/s)", ylabel="omega (rad/s)",
        xlabelsize=8, ylabelsize=8, xticklabelsize=7, yticklabelsize=7)
    _pub_title!(ax, p.id, p.title)
    # only continuous root actions have a plane to live in
    rc = filter(c -> c.action isa DuckieAction, root_children(s))
    xs = Float64[c.action.v for c in rc]
    ys = Float64[c.action.omega for c in rc]
    vis = Float64[c.visits for c in rc]
    if !isempty(xs)
        scatter!(ax, xs, ys;
            markersize=6 .+ 10 .* (vis ./ max(maximum(vis), 1)),
            color=PUB_GREY[2], strokewidth=0.6, strokecolor=PUB_GREY[1])
    end
    if s.selected_action isa DuckieAction
        scatter!(ax, [s.selected_action.v], [s.selected_action.omega];
            marker=:xcross, markersize=13, color=STOP_COLOR, strokewidth=1.5,
            strokecolor=STOP_COLOR)
    end
    return ax
end

function _panel_summary_table!(fig, cell, p)
    ax = Axis(fig[cell...])
    hidedecorations!(ax)
    hidespines!(ax)
    _pub_title!(ax, p.id, p.title)
    text!(ax, 0.0, 1.0; text=join(p.payload, "\n"), align=(:left, :top),
        fontsize=7, color=RGBAf(0.15, 0.15, 0.15, 1))
    limits!(ax, 0, 1, 0, 1)
    return ax
end

function _panel_diagnostics!(fig, cell, p)
    ep = p.payload
    xs = Float64.(ep.decisions)
    zone = [k for k in eachindex(ep.decisions)
            if series_named(ep, "sigma_stop").values[k] == 1.0]
    h = length(xs) < 2 ? 0.5 : 0.5 * (xs[2] - xs[1])

    # one axis per unit, as in FJ9.6: m/s and m do not share a scale, and the
    # stop-zone flag is shaded rather than drawn as a line at 1.0 — as a line
    # it would compress both real quantities into the bottom of the panel
    g = fig[cell...] = GridLayout()
    axes = Axis[]
    for (r, (nm, unit)) in enumerate((("v", "m/s"), ("d_stop", "m")))
        ax = Axis(g[r, 1]; ylabel=unit, ylabelsize=8, xticklabelsize=7,
            yticklabelsize=7, xlabel=r == 2 ? "decision" : "",
            xlabelsize=8, xticklabelsvisible=(r == 2))
        r == 1 && _pub_title!(ax, p.id, p.title)
        isempty(zone) || vspan!(ax, [xs[k] - h for k in zone],
            [xs[k] + h for k in zone]; color=RGBAf(0.6, 0.6, 0.6, 0.20),
            label="sigma_stop")
        sr = series_named(ep, nm)
        ys = Float64[x === missing ? NaN : x for x in sr.values]
        lines!(ax, xs, ys; color=PUB_GREY[1], linestyle=PUB_STYLES[r],
            linewidth=1.3, label=nm)
        gaps = [i for i in eachindex(sr.values) if sr.values[i] === missing]
        if !isempty(gaps)
            fin = filter(!isnan, ys)
            base = isempty(fin) ? 0.0 : minimum(fin)
            scatter!(ax, xs[gaps], fill(base, length(gaps)); marker=:vline,
                markersize=5, color=DIAG_MISSING, label="$nm MISSING")
        end
        for (name, ks) in ep.events
            vlines!(ax, Float64.(ks); color=STOP_COLOR, linewidth=1.0,
                linestyle=:dash)
            r == 1 && for kk in ks
                text!(ax, kk, 0.0; text=" " * name, fontsize=6,
                    rotation=pi / 2, align=(:left, :center), color=STOP_COLOR)
            end
        end
        axislegend(ax; position=:rt, framevisible=false, labelsize=6,
            nbanks=2, merge=true)
        push!(axes, ax)
    end
    linkxaxes!(axes...)
    rowgap!(g, 4)
    return axes[1]
end

function _panel_bins!(fig, cell, p)
    b = p.payload
    ax = Axis(fig[cell...]; xlabel="normalised episode progress",
        ylabel="generative calls", xlabelsize=8, ylabelsize=8,
        xticklabelsize=7, yticklabelsize=7)
    _pub_title!(ax, p.id, p.title)
    xs = [0.5 * (x.from + x.to) for x in b]
    rangebars!(ax, xs, [x.q25 for x in b], [x.q75 for x in b];
        color=PUB_GREY[1], whiskerwidth=7)
    scatterlines!(ax, xs, [x.median for x in b]; color=PUB_GREY[1],
        marker=:circle, markersize=6)
    hi = maximum(x.q75 for x in b)
    for (k, x) in enumerate(b)
        text!(ax, xs[k], hi; text="n=$(x.n)", fontsize=6,
            align=(:center, :bottom), color=PUB_GREY[2])
    end
    return ax
end

const _PANEL_DISPATCH = Dict(
    :world => _panel_world!, :projection => _panel_projection!,
    :observability => _panel_observability!,
    :slice_action => _panel_slice_action!, :slice_value => _panel_slice_value!,
    :slice_continuous => _panel_slice_continuous!,
    :search_tree => _panel_search_tree!,
    :search_summary => _panel_search_summary!,
    :action_plane => _panel_action_plane!,
    :summary_table => _panel_summary_table!,
    :diagnostics => _panel_diagnostics!, :bins => _panel_bins!)

"""
    render_composite(composite; ...) -> Figure

FJ9.8. Draw a `PublicationComposite` from its panel payloads.

The caption is validated before anything is drawn. A figure whose caption
fails its rule is not produced at all — FJ9.6 showed that a wrong sentence
beside right numbers survives review, so it is treated as a build failure
rather than as a copy-editing note.
"""
function DDM.render_composite(c::PublicationComposite;
    figure_size=(1500, 1150), show_caption::Bool=true, wrap::Integer=155)
    chk = check_caption(c)
    chk.ok || throw(ArgumentError(
        "caption for $(c.figure_id) fails its rule — absent: " *
        "$(join(chk.absent, ", ")); forbidden present: " *
        "$(join(chk.present, ", "))"))

    fig = Figure(size=figure_size)
    Label(fig[1, 1], "$(c.figure_id). $(c.title)"; fontsize=14, font=:bold,
        halign=:left, tellwidth=false)
    grid = fig[2, 1] = GridLayout()
    ncols = maximum(l[2] + l[4] - 1 for l in c.layout)
    for (p, (row, col, rs, cs)) in zip(c.panels, c.layout)
        # colorbars live inside each panel's own nested grid, so the outer
        # grid needs one column per column — not two
        cellgrid = grid[row:(row + rs - 1), col:(col + cs - 1)] = GridLayout()
        f = get(_PANEL_DISPATCH, p.kind, nothing)
        if f === nothing
            _panel_summary_table!(cellgrid, (1, 1),
                PanelSpec(p.id, p.title, :summary_table, [string(p.payload)],
                    p.provenance, p.semantics))
        else
            f(cellgrid, (1, 1), p)
        end
    end
    # a table panel is wide and short; a square cell for it is mostly whitespace
    for (p, (row, _, _, _)) in zip(c.panels, c.layout)
        p.kind in (:summary_table, :projection, :observability,
            :search_summary) && rowsize!(grid, row, Relative(0.22))
    end
    # without this the columns size to their content and the whole figure
    # collapses into a narrow band with margins either side
    for j in 1:ncols
        colsize!(grid, j, Relative(1 / ncols))
    end
    if show_caption
        axc = Axis(fig[3, 1]; tellwidth=false, tellheight=false)
        hidedecorations!(axc)
        hidespines!(axc)
        text!(axc, 0.0, 1.0;
            text=wrap_text(replace(c.caption, "**" => ""), wrap) * "

" *
                 provenance_block(c), align=(:left, :top), fontsize=7,
            color=RGBAf(0.3, 0.3, 0.3, 1))
        limits!(axc, 0, 1, 0, 1)
        rowsize!(fig.layout, 3, Relative(0.30))
    end
    return fig
end

end # module
