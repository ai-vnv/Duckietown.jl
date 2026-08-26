# A lap of small_loop under the SAFETY-SHAPED reward.
#
# Difference from `dora_lap.jl` in exactly two places:
#
#   1. scenario `:stop_and_duck_safe`, which switches on the yield and
#      stop-approach weights the source defaults leave at zero. Under the old
#      scenario `pedestrian` was 0.000 whether the vehicle ploughed through a
#      crossing duck or braked for it, and braking scored strictly worse
#      (0.040 against 0.102). Nothing asked the vehicle to yield, so it didn't.
#
#   2. the cost handed to DORA is now the documented default form
#
#          cost = max(c_min, step_cost - reward(s, a, s'))
#
#      instead of the hand-written `1 + 4|d| + 0.5|phi|`. The reward already
#      contains the lateral, heading, pedestrian and stop-approach terms, so
#      the safety semantics come from the model rather than from a number I
#      chose. Costs stay positive, which is what the SSP reduction needs.
#
# Everything else — determinism, the macro action, the ring counter, the key —
# is unchanged, so any behavioural difference is attributable to the reward.

using DuckietownDecisionModels
using POMDPs, POMDPTools, DORASolvers
using Random, Printf, Statistics, Serialization

const CFG = scenario_config(:stop_and_duck_safe)
const BASE = DuckietownMDP(CFG; action_space = :discrete)
const TR = BASE.transition
const SCFG = TR.state_cfg
const ACTS = collect(POMDPs.actions(BASE))
const RNG = MersenneTwister(1)
const K = 8
const STEP_COST = 1.0
const C_MIN = 0.05          # per underlying decision, so a macro step >= 0.4

raw_of(s) = first(get_raw_state(s, SCFG))
tile_of(s) = get_grid_coords(s.map, collect(s.ego.pos))

"""Cost of one underlying decision, in DORA's documented form."""
step_cost(r) = max(C_MIN, STEP_COST - r.reward.total)

const RING = let m = initial_map(CFG)
    t = collect(drivable_tiles(m))
    cx = sum(first.(t)) / length(t); cy = sum(last.(t)) / length(t)
    sort(t; by = p -> atan(p[2] - cy, p[1] - cx))
end
const RIX = Dict(t => i for (i, t) in enumerate(RING))
const NRING = length(RING)

"""Monotone ring counter: only a forward step to the next ring tile advances."""
function advance(prog, prev, new)
    new == prev && return prog
    a = get(RIX, prev, 0); b = get(RIX, new, 0)
    (a == 0 || b == 0) && return prog
    return mod(b - a, NRING) == 1 ? prog + 1 : prog
end

struct LapState
    s::DuckieWorldState
    prog::Int
end

"""Hold a command for K decisions, tracking ring progress and accumulating
the reward-derived cost. Exact — the chain consumes no randomness."""
function macro_step(ls::LapState, a)
    s = ls.s; prog = ls.prog; tile = tile_of(s); c = 0.0
    for _ in 1:K
        r = simulate_decision(TR, s, a, RNG)
        c += step_cost(r)
        nt = tile_of(r.sp)
        prog = advance(prog, tile, nt)
        tile = nt; s = r.sp
        (r.terminated || r.truncated) && return (LapState(s, prog), c, true)
    end
    return (LapState(s, prog), c, false)
end

struct LapMDP <: MDP{LapState,MacroAction} end
POMDPs.actions(::LapMDP) = ACTS
POMDPs.discount(::LapMDP) = 1.0
POMDPs.isterminal(::LapMDP, ls) = POMDPs.isterminal(BASE, ls.s)
POMDPs.transition(::LapMDP, ls, a) = Deterministic(first(macro_step(ls, a)))

lapkey(ls) = (min(ls.prog, NRING), discretize(raw_of(ls.s)))
classify(ls) = POMDPs.isterminal(BASE, ls.s) ? :crash :
               ls.prog >= NRING ? :goal : :normal

# Spawn selection: the rejection sampler's draws depend on the map objects,
# and with this map the first seeds tried landed the ego already inside the
# sign's hold zone at v = 0 — a "full stop" satisfied at birth demonstrates
# nothing. Take the first seed whose spawn is OUTSIDE the sign's detection
# corridor, so the approach, the stop, and the pass all happen on camera.
# This selects where the episode starts and nothing about how it is scored.
function pick_start()
    for seed in 1001:1100
        s = rand(MersenneTwister(seed), initialstate(BASE))
        raw = raw_of(s)
        if raw.d_stop === nothing
            @printf("spawn seed %d: pose (%.3f, %.3f) angle %.1f deg, outside the stop corridor\n",
                    seed, s.ego.pos[1], s.ego.pos[3], rad2deg(s.ego.angle))
            return LapState(s, 0)
        end
    end
    error("no seed in 1001:1100 spawns outside the stop corridor")
end

s0 = pick_start()
@printf("ring of %d tiles, start %s, goal = %d forward steps (one lap)\n",
        NRING, tile_of(s0.s), NRING)
println("scenario :stop_and_duck_safe   duck_unsafe=", CFG.reward.duck_unsafe,
        "  stop_approach_unsafe=", CFG.reward.stop_approach_unsafe,
        "  stop_approach_distance=", CFG.reward.stop_approach_distance)

"""Plan from `from`: tabularize the reachable graph and solve. Returns the
planner plus the key->index map (the table keeps states, not the map)."""
function plan_from(from::LapState)
    t0 = time()
    planner = solve(DORASolver(
            start = from, classify = classify,
            cost = (ls, a, lsp) -> macro_step(ls, a)[2],
            key = lapkey,
            c_min = K * C_MIN, c_to = 2000.0, c_crash = 1000.0, horizon = 120,
        ), LapMDP())
    tab = planner.tab
    index = Dict(lapkey(tab.states[i]) => i for i in 1:tab.S)
    @printf("  plan: %d states (%.1f s)\n", tab.S, time() - t0)
    return planner, tab, index
end

println("\ntabularizing...")
planner, tab, INDEX0 = plan_from(s0)
@printf("  MAXOUT %d\n", tab.MAXOUT)

V, pistar = optimal_value(tab)
sr, cr, tr_ = outcome_rates(tab, pistar)
@printf("\nin-model: optimal cost %.2f   success %.3f   crash %.3f   timeout %.3f\n",
        V[tab.start], sr, cr, tr_)
worst, frac = causality_margin(tab, V, pistar)
@printf("causality margin worst %.3f, states ok %.3f\n", worst, frac)

# --- execute, recording EVERY PHYSICS SUBSTEP for a smooth 30 fps video -----
#
# One decision is frame_skip = 6 physics ticks at 30 Hz, so a per-decision
# recording plays back at 5 fps and looks like a slideshow. Instead of
# interpolating (fabricated frames), replay each decision's physics with the
# package's own exported primitives — before_step, ego_tick, duck_step, the
# exact `_decision_chain` order — from a COPY of the rng, and assert the
# replayed end-of-decision world is bit-identical to `simulate_decision`'s.
# Every recorded frame is then a state the simulator actually passed through.
function replay_substeps!(subs, s, r, rng, prog, tile, dec)
    # `_decision_chain` hands the CALLER's rng to before_step (the 3-arg
    # form), so the replay must consume a copy of that same stream — taken
    # before simulate_decision advanced it — not the state's controller_rng
    w = before_step(s, TR.duck_cfg, rng)
    wheels64 = (Float64(r.wheel_commands[1]), Float64(r.wheel_commands[2]))
    for _ in 1:TR.frame_skip
        w = ego_tick(w, wheels64)
        for i in eachindex(w.ducks)
            w = duck_step(w, i)
        end
        nt = tile_of(w)
        prog = advance(prog, tile, nt); tile = nt
        push!(subs, (w = w, prog = prog))
    end
    @assert w.ego.pos == r.sp.ego.pos && w.ego.angle == r.sp.ego.angle
    @assert all(w.ducks[i].pos == r.sp.ducks[i].pos for i in eachindex(w.ducks))
    return prog, tile
end

"""One recorded substep with the decision-level annotations stamped on."""
substep_record(t, dec, a, r) = (
    w = t.w, prog = t.prog, dec = dec, a = a,
    v = t.w.ego.speed,
    d_stop = distance_to_next_stop(t.w, SCFG),
    ped = r.reward.pedestrian,
    stop_approach = r.reward.stop_approach,
    sigma = r.raw_state.sigma_stop,
    event = r.events.full_stop ? "FULL_STOP" :
            r.events.stop_violation ? "STOP_VIOLATION" :
            r.events.passed_stop ? "PASSED_STOP" : "")

# The determinized table stores, for each KEY, the successor computed from
# that key's BFS representative. Two concrete states can share a key (the
# discretization is coarse) yet behave differently under the same action —
# replanning only when a key-level fork appears at a macro boundary still
# crashed, because a fatal divergence can happen MID-macro without ever
# showing a key mismatch at a boundary. So: full receding horizon. Re-plan
# from the true current state before EVERY macro action. The first BFS
# expansion is from that state itself, which makes the first action's
# predicted outcome exact — execution can never diverge from the step it is
# about to take, only deeper plan segments can be aliased, and those are
# re-grounded before they are ever executed.
function execute(; horizon = 120)
    ls = s0
    subs = NamedTuple[]
    push!(subs, substep_record((w = s0.s, prog = 0), 0, nothing,
        (reward = (pedestrian = 0.0, stop_approach = 0.0),
         raw_state = (sigma_stop = false,),
         events = (full_stop = false, stop_violation = false,
                   passed_stop = false))))
    total = 0.0
    dec = 0
    plans = 0
    for _ in 1:horizon
        POMDPs.isterminal(BASE, ls.s) && return (:crash, total, subs, "isterminal", plans)
        ls.prog >= NRING && return (:lap, total, subs, "lap", plans)
        pl, _, _ = plan_from(ls)
        plans += 1
        a = action(pl, ls)
        s = ls.s; prog = ls.prog; tile = tile_of(s)
        for _ in 1:K
            dec += 1
            # copy the rng BEFORE the trusted step advances it, so the trusted
            # step and the replay consume the identical stream
            rngc = copy(RNG)
            raw_subs = NamedTuple[]
            r = simulate_decision(TR, s, a, RNG)
            prog2, tile2 = replay_substeps!(raw_subs, s, r, rngc, prog, tile, dec)
            for t in raw_subs
                push!(subs, substep_record(t, dec, a, r))
            end
            total += step_cost(r)
            prog = prog2; tile = tile2; s = r.sp
            prog >= NRING && return (:lap, total, subs, string(r.reason), plans)
            if r.terminated || r.truncated
                return (:crash, total, subs, string(r.reason), plans)
            end
        end
        ls = LapState(s, prog)
    end
    return (:timeout, total, subs, "horizon", plans)
end

outcome, total, subs, reason, plans = execute()
ndec = last(subs).dec
@printf("\nexecuted: %s after %d decisions (%d physics substeps), cost %.2f  (reason %s)\n",
        outcome, ndec, length(subs) - 1, total, reason)
@printf("substep replay matched simulate_decision exactly on all %d decisions\n", ndec)
@printf("ring progress reached: %d of %d   receding-horizon plans %d\n",
        last(subs).prog, NRING, plans)
@printf("first-plan cost %.2f  vs  executed %.2f\n", V[tab.start], total)

"""Did it yield, and did the stop subsystem actually fire? Report numbers per
decision, never impressions."""
function behaviour_report()
    ducks = NamedTuple[]; stops = NamedTuple[]
    for f in subs
        f.dec == 0 && continue
        raw, _ = get_raw_state(f.w, SCFG)
        raw.duck == NONE ||
            push!(ducks, (dec = f.dec, threat = raw.duck, v = f.v, ped = f.ped))
        f.d_stop === nothing ||
            push!(stops, (dec = f.dec, d = f.d_stop, v = f.v,
                          sa = f.stop_approach, sigma = f.sigma, ev = f.event))
    end
    return ducks, stops
end

ducks, stops = behaviour_report()
println("\nsubsteps with a duck threat: ", length(ducks))
if !isempty(ducks)
    vs = [r.v for r in ducks]
    @printf("  speed while threatened: min %.3f mean %.3f max %.3f   penalised substeps %d\n",
            minimum(vs), mean(vs), maximum(vs), count(r -> r.ped < 0, ducks))
end
println("substeps with d_stop available: ", length(stops),
        "   (was 0 before the sign was turned to face the traffic)")
if !isempty(stops)
    @printf("  d_stop range %.3f .. %.3f m   sigma_stop reached: %s\n",
            minimum(r.d for r in stops), maximum(r.d for r in stops),
            any(r.sigma for r in stops))
    # scan ALL substeps for events: the pass/violation fires on the decision
    # AFTER the sign leaves the corridor, when d_stop is already nothing, so
    # filtering by d_stop would hide exactly the event that matters
    evs = unique((f.dec, f.event) for f in subs if !isempty(f.event))
    println("  stop events seen: ", isempty(evs) ? "none" :
            join(("$(e[2]) at dec $(e[1])" for e in evs), ", "))
    println("  by decision (last substep of each):")
    for dec in sort(unique(r.dec for r in stops))
        r = last([r for r in stops if r.dec == dec])
        @printf("    dec %3d  d_stop %.3f  v %.3f  stop_approach %+7.3f  sigma %s %s\n",
                dec, r.d, r.v, r.sa, r.sigma, r.ev)
    end
end

serialize(joinpath(@__DIR__, "lap_frames_safe.jls"),
          (subs = subs, outcome = outcome, total = total,
           cost_model = V[tab.start], scenario = :stop_and_duck_safe,
           frame_skip = TR.frame_skip))
println("\nwrote lap_frames_safe.jls (", length(subs), " substep frames)")
