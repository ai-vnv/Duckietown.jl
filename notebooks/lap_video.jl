# Bird's-eye video of the DORA lap, drawn to look like Duckietown.
#
# Every coordinate comes from the core: tile polygons and lane curves from
# `world_scene`, the ego footprint from `get_agent_corners`, the stop line
# from the ego's lane frame, the duckies from their own corner polygons.
# Nothing here invents geometry — that is the FJ9.0 rule, and it is what makes
# the picture the same object the physics uses.
#
# Duckietown's road markings: grey asphalt, a dashed YELLOW centre line
# between the two directed lanes, solid WHITE outer edges, grass outside.
# The two directed lane curves per tile are in the map; the centre line is
# their midpoint and the white edges are offset outwards from each.

using DuckietownDecisionModels
using POMDPs, CairoMakie, Serialization, Printf, Random, Statistics
CairoMakie.activate!(type = "png")

const CFG = scenario_config(:stop_and_duck)
const BASE = DuckietownMDP(CFG; action_space = :discrete)
const SCFG = BASE.transition.state_cfg

D = deserialize(joinpath(@__DIR__, "lap_frames.jls"))
frames = D.frames
@printf("%d frames, outcome %s, cost %.2f (model %.2f)\n",
        length(frames), D.outcome, D.total, D.cost_model)

# Duckietown palette
const ASPHALT = RGBAf(0.28, 0.28, 0.30, 1)
const GRASS   = RGBAf(0.35, 0.52, 0.29, 1)
const YELLOW  = RGBAf(0.96, 0.83, 0.16, 1)
const WHITE   = RGBAf(0.96, 0.96, 0.96, 1)
const BOT     = RGBAf(0.85, 0.35, 0.05, 1)
const BOTED   = RGBAf(0.35, 0.13, 0.02, 1)
const DUCK    = RGBAf(0.98, 0.78, 0.09, 1)
const STOPRED = RGBAf(0.80, 0.12, 0.12, 1)

base = world_scene(BASE, frames[1][1])

"""Pair up the two directed lane curves of each tile and return the dashed
centre line between them plus the two outer edges."""
function markings(scene)
    curves = scene.lane_centrelines
    centres = Vector{Vector{Point2f}}()
    edges = Vector{Vector{Point2f}}()
    used = falses(length(curves))
    for i in eachindex(curves), j in eachindex(curves)
        (i >= j || used[i] || used[j]) && continue
        ci, cj = curves[i], curves[j]
        length(ci) == length(cj) || continue
        # the partner lane is the one running alongside, close and opposite
        mid = mean(hypot(ci[k][1] - cj[end + 1 - k][1],
                         ci[k][2] - cj[end + 1 - k][2]) for k in eachindex(ci))
        mid < 0.28 || continue
        used[i] = used[j] = true
        push!(centres, [Point2f((ci[k][1] + cj[end + 1 - k][1]) / 2,
                                (ci[k][2] + cj[end + 1 - k][2]) / 2)
                        for k in eachindex(ci)])
        for c in (ci, cj)
            push!(edges, [Point2f(p[1], p[2]) for p in c])
        end
    end
    return centres, edges
end

centres, lanes = markings(base)
@printf("lane curves %d -> centre lines %d\n", length(base.lane_centrelines),
        length(centres))

fig = Figure(backgroundcolor = GRASS)
ax = Axis(fig[1, 1]; aspect = DataAspect(), width = 660, height = 660,
    backgroundcolor = GRASS,
    title = "DORA completes a lap of small_loop",
    subtitle = "bird's eye · stop sign and crossing duckie active · " *
               "one macro action = 8 decisions",
    titlesize = 15, subtitlesize = 10,
    xlabel = "x (m)", ylabel = "z (m)", xlabelsize = 9, ylabelsize = 9,
    xticklabelsize = 8, yticklabelsize = 8)

# road surface
for tp in base.tiles
    tp.drivable || continue
    xs = [p[1] for p in tp.corners]; ys = [p[2] for p in tp.corners]
    poly!(ax, Point2f.(xs, ys); color = ASPHALT)
end
# white outer lane edges, then the dashed yellow centre
for c in lanes
    lines!(ax, c; color = WHITE, linewidth = 2.2)
end
for c in centres
    lines!(ax, c; color = YELLOW, linewidth = 2.4, linestyle = (:dash, :dense))
end
# stop sign post
isempty(base.stop_signs) || scatter!(ax, first.(base.stop_signs),
    last.(base.stop_signs); marker = :octagon, markersize = 15,
    color = STOPRED, strokewidth = 1.5, strokecolor = WHITE)

trail = Observable(Point2f[])
bot = Observable(Point2f[])
nose = Observable(Point2f[])
duckpts = Observable(Point2f[])
stopseg = Observable(Point2f[])
banner = Observable("")
progbar = Observable(Point2f[])

lines!(ax, trail; color = RGBAf(1, 1, 1, 0.55), linewidth = 2)
linesegments!(ax, stopseg; color = STOPRED, linewidth = 3)
poly!(ax, bot; color = BOT, strokewidth = 1.6, strokecolor = BOTED)
linesegments!(ax, nose; color = BOTED, linewidth = 2.5)
scatter!(ax, duckpts; marker = :circle, markersize = 15, color = DUCK,
    strokewidth = 1.2, strokecolor = RGBAf(0.5, 0.35, 0.02, 1))
lines!(ax, progbar; color = YELLOW, linewidth = 7)
Label(fig[2, 1], banner; fontsize = 12, halign = :left, tellwidth = false,
    color = RGBAf(0.15, 0.15, 0.15, 1))
limits!(ax, base.view_extent[1], base.view_extent[2],
    base.view_extent[3], base.view_extent[4])
resize_to_layout!(fig)

const X0, X1 = base.view_extent[1], base.view_extent[2]
const ZBAR = base.view_extent[3] + 0.04

function setframe!(i)
    s, prog = frames[i]
    sc = world_scene(BASE, s;
        trajectory = [(f[1].ego.pos[1], f[1].ego.pos[3]) for f in frames[1:i]])
    trail[] = Point2f.(first.(sc.trajectory), last.(sc.trajectory))
    bot[] = Point2f.(first.(sc.ego_footprint), last.(sc.ego_footprint))
    p = sc.ego_position
    nose[] = [Point2f(p[1], p[2]),
        Point2f(p[1] + 0.11sc.ego_heading[1], p[2] + 0.11sc.ego_heading[2])]
    duckpts[] = Point2f.(first.(sc.ducks), last.(sc.ducks))
    stopseg[] = reduce(vcat, [[Point2f(l[1], l[2]), Point2f(l[3], l[4])]
                              for l in sc.stop_lines]; init = Point2f[])
    f = prog / 8
    progbar[] = [Point2f(X0 + 0.02, ZBAR),
                 Point2f(X0 + 0.02 + f * (X1 - X0 - 0.04), ZBAR)]
    raw, _ = get_raw_state(s, SCFG)
    ds = raw.d_stop === nothing ? "—" : @sprintf("%.2f m", raw.d_stop)
    banner[] = @sprintf("decision %3d / %d      lap %d/8      d %+.3f m      d_stop %s      %s",
        i - 1, length(frames) - 1, prog, raw.d, ds,
        i == length(frames) ? "LAP COMPLETE" : "")
    return nothing
end

setframe!(1)
out = joinpath(@__DIR__, "dora_lap_bev.mp4")
idx = vcat(1:length(frames), fill(length(frames), 18))
record(fig, out, idx; framerate = 12) do i
    setframe!(i)
end
@printf("wrote %s (%d frames, %.1f kB)\n", out, length(idx),
        filesize(out) / 1024)

setframe!(length(frames))
save(joinpath(@__DIR__, "lap_bev_end.png"), fig)
setframe!(max(1, length(frames) ÷ 2))
save(joinpath(@__DIR__, "lap_bev_mid.png"), fig)
