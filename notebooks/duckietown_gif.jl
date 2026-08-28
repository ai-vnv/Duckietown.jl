# DORA's policy executed in the REAL Duckietown dynamics, as an animation.
#
# The gridworld GIF shows a solver on a model that IS the environment. This
# one shows the same solver on a model that only estimates the environment,
# so it has to show the failures too: 9 % of episodes clear the stop sign and
# 86 % leave the road. Selecting only the successes would reproduce exactly
# the over-confidence this whole exercise was about.
#
# Episodes are chosen by a stated rule, not by eye: the first success, the
# first off-road and the first collision in seed order.

using DuckietownDecisionModels
using POMDPs, POMDPTools, DORASolvers
using CairoMakie, Random, Statistics, Printf
CairoMakie.activate!(type = "png")

const MDP_ = DuckietownMDP(scenario_config(:stop_and_duck); action_space = :discrete)
const ACTS = collect(POMDPs.actions(MDP_))
const SCFG = MDP_.transition.state_cfg
disc(s) = discretize(first(get_raw_state(s, SCFG)))

classify_result(res) =
    res.events.passed_stop ? :goal :
    (res.events.offroad ? :offroad :
     (res.events.other_collision || res.events.collision_duck) ? :collision :
     :normal)
step_cost(res) = 1.0 + 4.0abs(res.raw_state.d) + 0.5abs(res.raw_state.phi)

function behaviour_action(s, rng, eps)
    rand(rng) < eps && return rand(rng, 1:length(ACTS))
    raw, _ = get_raw_state(s, SCFG)
    err = raw.d + 0.35raw.phi
    a = if err < -0.02
            raw.phi > 0.25 ? SLOW_STRAIGHT : SLOW_LEFT
        elseif err > 0.02
            raw.phi < -0.25 ? SLOW_STRAIGHT : SLOW_RIGHT
        elseif abs(raw.phi) < 0.10 && abs(raw.d) < 0.03
            FAST_STRAIGHT
        else
            SLOW_STRAIGHT
        end
    return findfirst(==(a), ACTS)
end

const GOAL_S = ntuple(_ -> -1, 7)
const CRASH_S = ntuple(_ -> -2, 7)
struct DuckieSSP <: MDP{NTuple{7,Int},Int}
    P::Dict{Tuple{NTuple{7,Int},Int},Vector{Tuple{NTuple{7,Int},Float64}}}
    C::Dict{Tuple{NTuple{7,Int},Int,NTuple{7,Int}},Float64}
end
POMDPs.actions(::DuckieSSP) = collect(1:7)
POMDPs.discount(::DuckieSSP) = 1.0
POMDPs.isterminal(::DuckieSSP, s) = s == GOAL_S || s == CRASH_S
function POMDPs.transition(m::DuckieSSP, s, a)
    POMDPs.isterminal(m, s) && return Deterministic(s)
    e = get(m.P, (s, a), nothing)
    e === nothing && return Deterministic(s)
    SparseCat(first.(e), last.(e))
end

function build_ssp(; episodes = 2000, horizon = 150, seed = 20260826, eps = 0.15)
    counts = Dict{Tuple{NTuple{7,Int},Int},Dict{Any,Int}}()
    costs = Dict{Tuple{NTuple{7,Int},Int},Dict{Any,Float64}}()
    starts = Dict{NTuple{7,Int},Int}()
    rng = MersenneTwister(seed)
    for _ in 1:episodes
        s = rand(rng, initialstate(MDP_))
        starts[disc(s)] = get(starts, disc(s), 0) + 1
        for _ in 1:horizon
            ds = disc(s); ai = behaviour_action(s, rng, eps)
            res = simulate_decision(MDP_.transition, s, ACTS[ai], rng)
            cls = classify_result(res)
            tgt = cls === :normal ? disc(res.sp) :
                  cls === :goal ? :goal : :crash
            c = get!(counts, (ds, ai), Dict{Any,Int}())
            k = get!(costs, (ds, ai), Dict{Any,Float64}())
            c[tgt] = get(c, tgt, 0) + 1
            k[tgt] = get(k, tgt, 0.0) + step_cost(res)
            (cls !== :normal || res.terminated || res.truncated) && break
            s = res.sp
        end
    end
    P = Dict{Tuple{NTuple{7,Int},Int},Vector{Tuple{NTuple{7,Int},Float64}}}()
    C = Dict{Tuple{NTuple{7,Int},Int,NTuple{7,Int}},Float64}()
    for ((ds, ai), tg) in counts
        n = sum(values(tg)); v = Tuple{NTuple{7,Int},Float64}[]
        for (t, k) in tg
            st = t === :goal ? GOAL_S : t === :crash ? CRASH_S : t::NTuple{7,Int}
            push!(v, (st, k / n)); C[(ds, ai, st)] = costs[(ds, ai)][t] / k
        end
        P[(ds, ai)] = v
    end
    return DuckieSSP(P, C), argmax(starts)
end

println("building the sampled SSP...")
ssp, start_state = build_ssp()
planner = solve(DORASolver(start = start_state,
    classify = sp -> sp == GOAL_S ? :goal : sp == CRASH_S ? :crash : :normal,
    cost = (s, a, sp) -> get(ssp.C, (s, a, sp), 1.0),
    c_min = 0.5, c_to = 200.0, c_crash = 100.0, horizon = 150), ssp)
known = Set(planner.tab.states)

"""Run one seeded episode under DORA, recording the world state at every
decision so the animation replays what happened rather than re-simulating."""
function episode(seed; horizon = 150)
    rng = MersenneTwister(seed)
    s = rand(rng, initialstate(MDP_))
    states = [s]; offmodel = 0; outcome = :horizon
    for t in 1:horizon
        ds = disc(s)
        inmodel = ds in known
        inmodel || (offmodel += 1)
        ai = inmodel ? action(planner, ds) : behaviour_action(s, rng, 0.0)
        res = simulate_decision(MDP_.transition, s, ACTS[ai], rng)
        push!(states, res.sp)
        cls = classify_result(res)
        if cls !== :normal
            outcome = cls; break
        elseif res.terminated || res.truncated
            outcome = :terminated; break
        end
        s = res.sp
    end
    return (; seed, states, outcome, offmodel)
end

println("running episodes under DORA in the real dynamics...")
runs = [episode(sd) for sd in 1:400]
tally = Dict(o => count(r -> r.outcome === o, runs) for o in
             (:goal, :offroad, :collision, :horizon, :terminated))
@printf("400 episodes: goal %d, offroad %d, collision %d, horizon %d\n",
        tally[:goal], tally[:offroad], tally[:collision], tally[:horizon])

# stated selection rule: first of each outcome in seed order
pick(o) = findfirst(r -> r.outcome === o, runs)
chosen = filter(!isnothing, [pick(:goal), pick(:offroad), pick(:collision)])
shown = [runs[i] for i in chosen]
@printf("showing seeds %s (first success, first off-road, first collision)\n",
        join([r.seed for r in shown], ", "))

# --- draw -------------------------------------------------------------------
base = world_scene(MDP_, first(shown).states[1])

fig = Figure()
ax = Axis(fig[1, 1]; aspect = DataAspect(), width = 620, height = 620,
    title = "DORA's policy in Duckietown — executed on the real dynamics",
    subtitle = "trail grows to the current decision only · outcome labelled from what happened",
    xlabel = "x (m)", ylabel = "z (m)", titlesize = 14, subtitlesize = 10)

for tp in base.tiles
    xs = [p[1] for p in tp.corners]; ys = [p[2] for p in tp.corners]
    poly!(ax, Point2f.(xs, ys);
        color = tp.drivable ? RGBAf(.90, .90, .90, 1) : RGBAf(.97, .95, .86, 1),
        strokewidth = .5, strokecolor = RGBAf(.78, .78, .78, 1))
end
for cl in base.lane_centrelines
    lines!(ax, [p[1] for p in cl], [p[2] for p in cl];
        color = RGBAf(.55, .55, .55, .8), linewidth = .8, linestyle = :dot)
end
isempty(base.stop_signs) || scatter!(ax, first.(base.stop_signs),
    last.(base.stop_signs); marker = :octagon, markersize = 13,
    color = RGBAf(.78, .13, .13, 1))

trail = Observable(Point2f[])
foot = Observable(Point2f[])
head = Observable(Point2f[])
stopseg = Observable(Point2f[])
ducks = Observable(Point2f[])
banner = Observable("")

lines!(ax, trail; color = RGBAf(.11, .35, .72, .8), linewidth = 2.5)
linesegments!(ax, stopseg; color = RGBAf(.78, .13, .13, 1), linewidth = 2.5)
scatter!(ax, ducks; marker = :utriangle, markersize = 12,
    color = RGBAf(.95, .62, .07, 1))
poly!(ax, foot; color = RGBAf(.11, .35, .72, .35), strokewidth = 1.5,
    strokecolor = RGBAf(.11, .35, .72, 1))
linesegments!(ax, head; color = RGBAf(.05, .20, .45, 1), linewidth = 2)
Label(fig[2, 1], banner; fontsize = 12, halign = :left, tellwidth = false)
limits!(ax, base.view_extent[1], base.view_extent[2],
    base.view_extent[3], base.view_extent[4])
resize_to_layout!(fig)

frames = [(r, t) for r in shown for t in 1:length(r.states)]
append!(frames, [(shown[end], length(shown[end].states)) for _ in 1:10])

# An empty `poly!` observable is a BoundsError the first time Makie lays the
# figure out, so the setter runs once before recording starts.
function setframe!(r, t)
    s = r.states[t]
    sc = world_scene(MDP_, s;
        trajectory = [(x.ego.pos[1], x.ego.pos[3]) for x in r.states[1:t]])
    trail[] = Point2f.(first.(sc.trajectory), last.(sc.trajectory))
    foot[] = Point2f.(first.(sc.ego_footprint), last.(sc.ego_footprint))
    p = sc.ego_position
    head[] = [Point2f(p[1], p[2]),
        Point2f(p[1] + .12sc.ego_heading[1], p[2] + .12sc.ego_heading[2])]
    stopseg[] = reduce(vcat, [[Point2f(l[1], l[2]), Point2f(l[3], l[4])]
                              for l in sc.stop_lines]; init = Point2f[])
    ducks[] = Point2f.(first.(sc.ducks), last.(sc.ducks))
    raw, _ = get_raw_state(s, SCFG)
    ds = raw.d_stop === nothing ? "MISSING" :
         string(round(raw.d_stop; digits = 3), " m")
    last_ = t == length(r.states)
    tag = !last_ ? "running" :
          r.outcome === :goal ? "CLEARED THE STOP SIGN" :
          r.outcome === :offroad ? "OFF ROAD" :
          r.outcome === :collision ? "COLLISION" : string(r.outcome)
    banner[] = string("seed ", r.seed, "   decision ", t - 1, " of ",
        length(r.states) - 1, "   d ", round(raw.d; digits = 3), " m   d_stop ",
        ds, "   —   ", tag)
    return nothing
end

setframe!(shown[1], 1)

out = joinpath(@__DIR__, "dora_duckietown.gif")
record(fig, out, frames; framerate = 12) do (r, t)
    setframe!(r, t)
end
@printf("wrote %s (%d frames, %.1f kB)\n", out, length(frames),
        filesize(out) / 1024)
