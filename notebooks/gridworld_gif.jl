# DORA reaching the reward tile in SimpleGridWorld, as an animation.
#
# Same conventions as the Duckietown renderers in this repository: the trail
# grows only up to the current frame, so no future position is visible before
# it happens, and every episode's outcome is labelled from what actually
# occurred rather than from what was expected.

using POMDPs, POMDPTools, POMDPModels, DORASolvers
using CairoMakie, Random, Printf
CairoMakie.activate!(type = "png")

const REWARDS = Dict(GWPos(4, 3) => -10.0, GWPos(4, 6) => -10.0,
                     GWPos(9, 3) => 10.0)
gw = SimpleGridWorld(size = (10, 10), rewards = REWARDS, tprob = 0.7)
planner = solve(DORASolver(start = GWPos(1, 1)), gw)

# --- DORA's policy on every cell it planned for ----------------------------
const ARROW = Dict(:up => (0, 1), :down => (0, -1),
                   :left => (-1, 0), :right => (1, 0))

function policy_field(planner, gw)
    xs, ys, us, vs = Float64[], Float64[], Float64[], Float64[]
    for x in 1:10, y in 1:10
        s = GWPos(x, y)
        haskey(REWARDS, s) && continue
        a = try
            action(planner, s)
        catch
            continue
        end
        haskey(ARROW, a) || continue
        d = ARROW[a]
        push!(xs, x); push!(ys, y); push!(us, d[1]); push!(vs, d[2])
    end
    return xs, ys, us, vs
end

# --- episodes, recorded exactly as they happened ---------------------------
function episodes(planner, gw; n, maxsteps = 60, seed = 4)
    rng = MersenneTwister(seed)
    out = NamedTuple[]
    for _ in 1:n
        s = GWPos(1, 1)
        path = [(s[1], s[2])]
        outcome = :timeout
        for _ in 1:maxsteps
            a = action(planner, s)
            sp = rand(rng, transition(gw, s, a))
            # Stepping ONTO a reward cell is not terminal in SimpleGridWorld:
            # the next transition goes to the sentinel GWPos(-1,-1). Classify
            # from the last real cell, and never draw the sentinel.
            if isterminal(gw, sp)
                outcome = get(REWARDS, s, 0.0) > 0 ? :goal : :crash
                break
            end
            push!(path, (sp[1], sp[2]))
            s = sp
        end
        push!(out, (path = path, outcome = outcome))
    end
    return out
end

eps_ = episodes(planner, gw; n = 6)
@printf("recorded %d episodes: %d reached the reward tile, %d crashed\n",
        length(eps_), count(e -> e.outcome === :goal, eps_),
        count(e -> e.outcome === :crash, eps_))

# --- draw ------------------------------------------------------------------
px, py, pu, pv = policy_field(planner, gw)

# DataAspect on its own lets the axis shrink inside a fixed canvas and leaves
# most of the image blank; giving the axis an explicit size and resizing the
# figure to the layout is the reliable way to get a tight square plot.
fig = Figure()
ax = Axis(fig[1, 1]; aspect = DataAspect(), width = 620, height = 620,
    title = "DORA on SimpleGridWorld — reaching the +10 tile",
    subtitle = "arrows: DORA's planned action · trail grows to the current step only",
    xlabel = "x", ylabel = "y", titlesize = 15, subtitlesize = 10)

for x in 1:10, y in 1:10
    poly!(ax, Point2f[(x - .5, y - .5), (x + .5, y - .5),
                      (x + .5, y + .5), (x - .5, y + .5)];
        color = RGBAf(.96, .96, .96, 1), strokewidth = .5,
        strokecolor = RGBAf(.85, .85, .85, 1))
end
for (cell, r) in REWARDS
    poly!(ax, Point2f[(cell[1] - .5, cell[2] - .5), (cell[1] + .5, cell[2] - .5),
                      (cell[1] + .5, cell[2] + .5), (cell[1] - .5, cell[2] + .5)];
        color = r > 0 ? RGBAf(.13, .55, .23, .35) : RGBAf(.78, .13, .13, .30),
        strokewidth = 1.5,
        strokecolor = r > 0 ? RGBAf(.13, .55, .23, 1) : RGBAf(.78, .13, .13, 1))
    text!(ax, cell[1], cell[2] + 0.34; text = r > 0 ? "+10" : "-10",
        align = (:center, :center), fontsize = 12,
        color = r > 0 ? RGBAf(.08, .35, .15, 1) : RGBAf(.5, .07, .07, 1))
end
arrows2d!(ax, px, py, 0.32 .* pu, 0.32 .* pv;
    color = RGBAf(.45, .45, .45, .75), shaftwidth = 1.2, tiplength = 6)
scatter!(ax, [1], [1]; marker = :rect, markersize = 14,
    color = RGBAf(.11, .35, .72, 1))
text!(ax, 1, 0.28; text = "start", align = (:center, :center), fontsize = 9,
    color = RGBAf(.11, .35, .72, 1))

trail = Observable(Point2f[])
head = Observable(Point2f[])
banner = Observable("")
lines!(ax, trail; color = RGBAf(.11, .35, .72, .85), linewidth = 3)
scatter!(ax, head; markersize = 17, color = RGBAf(.11, .35, .72, 1),
    strokewidth = 1.5, strokecolor = :white)
Label(fig[2, 1], banner; fontsize = 13, halign = :left, tellwidth = false)
limits!(ax, 0.3, 10.7, 0.3, 10.7)
resize_to_layout!(fig)

frames = [(e, t) for (i, e) in enumerate(eps_) for t in 1:length(e.path)]
append!(frames, [(eps_[end], length(eps_[end].path)) for _ in 1:8])   # hold

out = joinpath(@__DIR__, "dora_gridworld.gif")
record(fig, out, frames; framerate = 6) do (e, t)
    trail[] = [Point2f(p...) for p in e.path[1:t]]
    head[] = [Point2f(e.path[t]...)]
    done = t == length(e.path)
    banner[] = if !done
        "step $(t-1) of $(length(e.path)-1) — running"
    elseif e.outcome === :goal
        "reached the +10 tile in $(t-1) steps"
    elseif e.outcome === :crash
        "hit a -10 tile after $(t-1) steps"
    else
        "horizon reached"
    end
end
@printf("wrote %s (%d frames, %.1f kB)\n", out, length(frames),
        filesize(out) / 1024)
