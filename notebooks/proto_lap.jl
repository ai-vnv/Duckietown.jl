# Lap-completion SSP: the goal is to get all the way around small_loop.
#
# The 7-tuple discretization is purely LOCAL lane geometry — it cannot tell
# "one tile from home" from "just started". A lap goal therefore needs a state
# that carries progress around the ring, which is what turns this into an
# actual navigation SSP rather than a lane-keeping problem with a goal bolted on.

using DuckietownDecisionModels
using POMDPs, POMDPTools, DORASolvers
using Random, Statistics, Printf

const MDP_ = DuckietownMDP(scenario_config(:stop_and_duck); action_space=:discrete)
const ACTS = collect(POMDPs.actions(MDP_))
const CFG = MDP_.transition.state_cfg

# --- the ring of drivable tiles, taken from the map ------------------------
function ring_order(m)
    tiles = collect(drivable_tiles(m))
    cx = sum(first.(tiles)) / length(tiles)
    cy = sum(last.(tiles)) / length(tiles)
    sort!(tiles; by = t -> atan(t[2] - cy, t[1] - cx))
    return tiles
end

const RING = ring_order(initial_map(scenario_config(:stop_and_duck)))
const RING_IX = Dict(t => k for (k, t) in enumerate(RING))
const NRING = length(RING)

tile_of(s) = get_grid_coords(s.map, collect(s.ego.pos))

# --- progress-carrying discrete state --------------------------------------
dbin(d) = clamp(round(Int, d / 0.05), -3, 3)
pbin(p) = clamp(round(Int, p / 0.15), -3, 3)

"""State = (ring steps completed, lateral bin, heading bin)."""
function lap_state(prog, s)
    raw, _ = get_raw_state(s, CFG)
    return (prog, dbin(raw.d), pbin(raw.phi))
end

"""Advance the monotone ring counter. Forward one tile increments it; any
other tile change is a wrong turn and is recorded as such rather than being
silently absorbed."""
function advance(prog, prev_tile, new_tile)
    new_tile == prev_tile && return (prog, :same)
    a = get(RING_IX, prev_tile, 0); b = get(RING_IX, new_tile, 0)
    (a == 0 || b == 0) && return (prog, :off)
    step = mod(b - a, NRING)
    step == 1 && return (prog + 1, :forward)
    step == NRING - 1 && return (prog, :backward)
    return (prog, :jump)
end

classify_lap(prog, res) =
    prog >= NRING ? :goal :
    (res.events.offroad || res.events.other_collision ||
     res.events.collision_duck) ? :crash : :normal

step_cost(res) = 1.0 + 4.0 * abs(res.raw_state.d) + 0.5 * abs(res.raw_state.phi)

function behaviour_action(s, rng, eps)
    rand(rng) < eps && return rand(rng, 1:length(ACTS))
    raw, _ = get_raw_state(s, CFG)
    err = raw.d + 0.35 * raw.phi
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

function collect_lap(; episodes=1500, horizon=300, seed=1, eps=0.12)
    counts = Dict{Tuple{Any,Int},Dict{Any,Int}}()
    costs  = Dict{Tuple{Any,Int},Dict{Any,Float64}}()
    starts = Dict{Any,Int}()
    rng = MersenneTwister(seed)
    goals = crashes = steps = 0
    laps = Int[]
    for _ in 1:episodes
        s = rand(rng, initialstate(MDP_))
        prog = 0; tile = tile_of(s)
        starts[lap_state(prog, s)] = get(starts, lap_state(prog, s), 0) + 1
        for _ in 1:horizon
            ds = lap_state(prog, s)
            ai = behaviour_action(s, rng, eps)
            res = simulate_decision(MDP_.transition, s, ACTS[ai], rng)
            ntile = tile_of(res.sp)
            prog2, _ = advance(prog, tile, ntile)
            cls = classify_lap(prog2, res)
            tgt = cls === :normal ? lap_state(prog2, res.sp) : cls
            c = get!(counts, (ds, ai), Dict{Any,Int}())
            k = get!(costs,  (ds, ai), Dict{Any,Float64}())
            c[tgt] = get(c, tgt, 0) + 1
            k[tgt] = get(k, tgt, 0.0) + step_cost(res)
            steps += 1
            cls === :goal && (goals += 1)
            cls === :crash && (crashes += 1)
            if cls !== :normal || res.terminated || res.truncated
                push!(laps, prog2); break
            end
            s = res.sp; tile = ntile; prog = prog2
        end
    end
    return (; counts, costs, starts, goals, crashes, steps, laps)
end

println("ring tiles: ", NRING, "  ", RING)
println("sampling with a lap goal (this takes a while)...")
D = collect_lap()
@printf("  decisions          : %d\n", D.steps)
@printf("  distinct (s,a)     : %d\n", length(D.counts))
@printf("  distinct states    : %d\n", length(unique(first(k) for k in keys(D.counts))))
@printf("  laps completed     : %d / 1500 episodes\n", D.goals)
@printf("  crashes            : %d\n", D.crashes)
@printf("  ring progress: mean %.2f  max %d  (of %d)\n",
        mean(D.laps), maximum(D.laps), NRING)

# ---------------------------------------------------------------------------
# The hand-written follower cannot drive a lap, so the sampled graph has no
# path to the goal at all. Does a TRAINED policy populate it?
# ---------------------------------------------------------------------------
qpath = joinpath(pkgdir(DuckietownDecisionModels), "..", "duckduck",
                 "policies", "q_learning", "policy.npy")
if !isfile(qpath)
    println("\nno trained checkpoint available; stopping here")
else
    qpol = QTablePolicy(qpath; solver=:q_learning)
    qcfg = load_config(joinpath(dirname(qpath), "training_config.yaml"))
    QMDP_ = DuckietownMDP(qcfg; action_space=:discrete)
    QCFG = QMDP_.transition.state_cfg

    function q_behaviour(s, rng, eps)
        rand(rng) < eps && return rand(rng, 1:length(ACTS))
        raw, _ = get_raw_state(s, QCFG)
        return findfirst(==(decide(qpol, discretize(raw)).action), ACTS)
    end

    function collect_lap_q(; episodes=300, horizon=300, seed=3, eps=0.10)
        counts = Dict{Tuple{Any,Int},Dict{Any,Int}}()
        rng = MersenneTwister(seed)
        goals = crashes = steps = 0; laps = Int[]
        for _ in 1:episodes
            s = rand(rng, initialstate(QMDP_))
            prog = 0; tile = tile_of(s)
            for _ in 1:horizon
                ai = q_behaviour(s, rng, eps)
                res = simulate_decision(QMDP_.transition, s, ACTS[ai], rng)
                prog2, _ = advance(prog, tile, tile_of(res.sp))
                cls = classify_lap(prog2, res)
                c = get!(counts, (lap_state(prog, s), ai), Dict{Any,Int}())
                tgt = cls === :normal ? lap_state(prog2, res.sp) : cls
                c[tgt] = get(c, tgt, 0) + 1
                steps += 1
                cls === :goal && (goals += 1)
                cls === :crash && (crashes += 1)
                if cls !== :normal || res.terminated || res.truncated
                    push!(laps, prog2); break
                end
                s = res.sp; tile = tile_of(res.sp); prog = prog2
            end
        end
        return (; counts, goals, crashes, steps, laps)
    end

    println("\nsampling with the TRAINED q_learning policy as behaviour...")
    Q = collect_lap_q()
    @printf("  decisions       : %d\n", Q.steps)
    @printf("  distinct (s,a)  : %d\n", length(Q.counts))
    @printf("  laps completed  : %d / 300 episodes\n", Q.goals)
    @printf("  crashes         : %d\n", Q.crashes)
    @printf("  ring progress   : mean %.2f  max %d  (of %d)\n",
            mean(Q.laps), maximum(Q.laps), NRING)
end
