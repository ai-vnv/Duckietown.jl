# Prototype: can DORA be applied to Duckietown at all, and does the answer
# survive execution on the real dynamics?

using DuckietownDecisionModels
using POMDPs, POMDPTools, DORASolvers
using Random, Statistics, Printf

const MDP_ = DuckietownMDP(scenario_config(:stop_and_duck); action_space=:discrete)
const ACTS = collect(POMDPs.actions(MDP_))

disc(s) = discretize(first(get_raw_state(s, MDP_.transition.state_cfg)))

# ---------------------------------------------------------------------------
# 1. Sample the real dynamics into an empirical tabular model.
# ---------------------------------------------------------------------------
# class of an outcome, in SSP terms. The sub-task is "legally clear the stop
# sign": that is the goal. Off-road / collision is the crash sink.
function classify_result(res)
    res.events.passed_stop && return :goal
    (res.events.offroad || res.events.other_collision ||
     res.events.collision_duck) && return :crash
    return :normal
end

# positive traversal cost. NOT derived from the reward: the reward is a
# discounted shaped signal with negative and positive terms, and DORA needs a
# strictly positive undiscounted cost. This is a modelling CHOICE and is
# reported as such: one unit per decision, plus a lane-deviation term.
step_cost(res) = 1.0 + 4.0 * abs(res.raw_state.d) + 0.5 * abs(res.raw_state.phi)

# A random walk rarely gets to the stop sign: 8 goals in 400 episodes.
# Exploration has to be good enough to make the sub-task reachable, or DORA is
# handed a graph with no path to the goal. This is an eps-greedy lane follower.
function behaviour_action(s, rng, eps)
    rand(rng) < eps && return rand(rng, 1:length(ACTS))
    raw, _ = get_raw_state(s, MDP_.transition.state_cfg)
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

function collect_transitions(; episodes=400, horizon=120, seed=20260826,
                             eps=0.25)
    counts = Dict{Tuple{NTuple{7,Int},Int}, Dict{Any,Int}}()
    costs  = Dict{Tuple{NTuple{7,Int},Int}, Dict{Any,Float64}}()
    starts = Dict{NTuple{7,Int},Int}()
    rng = MersenneTwister(seed)
    goals = 0; crashes = 0; steps = 0
    for ep in 1:episodes
        s = rand(rng, initialstate(MDP_))
        starts[disc(s)] = get(starts, disc(s), 0) + 1
        for t in 1:horizon
            ds = disc(s)
            ai = behaviour_action(s, rng, eps)
            res = simulate_decision(MDP_.transition, s, ACTS[ai], rng)
            cls = classify_result(res)
            key = (ds, ai)
            tgt = cls === :normal ? disc(res.sp) : cls
            c = get!(counts, key, Dict{Any,Int}())
            k = get!(costs, key, Dict{Any,Float64}())
            c[tgt] = get(c, tgt, 0) + 1
            k[tgt] = get(k, tgt, 0.0) + step_cost(res)
            steps += 1
            cls === :goal && (goals += 1)
            cls === :crash && (crashes += 1)
            (cls !== :normal || res.terminated || res.truncated) && break
            s = res.sp
        end
    end
    return (counts=counts, costs=costs, starts=starts, goals=goals,
            crashes=crashes, steps=steps)
end

println("sampling the real dynamics...")
D = collect_transitions(; episodes=2000, horizon=150, eps=0.15)
@printf("  decisions sampled : %d\n", D.steps)
@printf("  distinct (s,a)    : %d\n", length(D.counts))
@printf("  distinct states   : %d\n", length(unique(first(k) for k in keys(D.counts))))
@printf("  goal events       : %d\n", D.goals)
@printf("  crash events      : %d\n", D.crashes)
@printf("  distinct starts   : %d\n", length(D.starts))

# ---------------------------------------------------------------------------
# 2. Wrap the empirical model as a POMDPs.jl MDP with an EXPLICIT transition.
# ---------------------------------------------------------------------------
const GOAL_S  = ntuple(_ -> -1, 7)
const CRASH_S = ntuple(_ -> -2, 7)

struct DuckieSSP <: MDP{NTuple{7,Int},Int}
    P::Dict{Tuple{NTuple{7,Int},Int}, Vector{Tuple{NTuple{7,Int},Float64}}}
    C::Dict{Tuple{NTuple{7,Int},Int,NTuple{7,Int}}, Float64}
end

POMDPs.actions(::DuckieSSP) = collect(1:length(ACTS))
POMDPs.discount(::DuckieSSP) = 1.0
POMDPs.isterminal(::DuckieSSP, s) = s == GOAL_S || s == CRASH_S

function POMDPs.transition(m::DuckieSSP, s, a)
    POMDPs.isterminal(m, s) && return Deterministic(s)
    e = get(m.P, (s, a), nothing)
    # an (s,a) the sampling never tried is a SELF LOOP, which tabularize reads
    # as "no determinized edge". Encoding it as anything else would invent a
    # transition the dynamics never showed us.
    e === nothing && return Deterministic(s)
    return SparseCat(first.(e), last.(e))
end

function build_ssp(D)
    P = Dict{Tuple{NTuple{7,Int},Int}, Vector{Tuple{NTuple{7,Int},Float64}}}()
    C = Dict{Tuple{NTuple{7,Int},Int,NTuple{7,Int}}, Float64}()
    for ((ds, ai), tgts) in D.counts
        n = sum(values(tgts))
        v = Tuple{NTuple{7,Int},Float64}[]
        for (t, k) in tgts
            st = t === :goal ? GOAL_S : t === :crash ? CRASH_S : t::NTuple{7,Int}
            push!(v, (st, k / n))
            C[(ds, ai, st)] = D.costs[(ds, ai)][t] / k
        end
        P[(ds, ai)] = v
    end
    return DuckieSSP(P, C)
end

ssp = build_ssp(D)
start_state = argmax(D.starts)
@printf("\nstart state %s (seen %d times)\n", start_state, D.starts[start_state])

nstates = length(unique(first(k) for k in keys(ssp.P)))
tried = length(ssp.P)
@printf("(s,a) pairs with data : %d of %d possible (%.1f%%)\n",
        tried, nstates * length(ACTS), 100 * tried / (nstates * length(ACTS)))

# ---------------------------------------------------------------------------
# 3. Run DORA.
# ---------------------------------------------------------------------------
println("\nsolving with DORA...")
planner = solve(DORASolver(
        start = start_state,
        classify = sp -> sp == GOAL_S ? :goal : sp == CRASH_S ? :crash : :normal,
        cost = (s, a, sp) -> get(ssp.C, (s, a, sp), 1.0),
        c_min = 0.5, c_to = 200.0, c_crash = 100.0, horizon = 150,
    ), ssp)

tab = planner.tab
@printf("tabularized: %d ordinary states, %d actions\n", tab.S, tab.NA)
V, pistar = optimal_value(tab)
@printf("optimal expected cost from start : %.2f\n", V[tab.start])
sr, cr, _ = outcome_rates(tab, pistar)
@printf("optimal success rate (in-model)  : %.3f\n", sr)
@printf("optimal crash rate  (in-model)   : %.3f\n", cr)

# ---------------------------------------------------------------------------
# 4. THE CHECK THAT MATTERS: run DORA's policy on the real dynamics.
# ---------------------------------------------------------------------------
# The 80.8 % above is the success rate of the ABSTRACTION against itself. The
# abstraction is an empirical model over a discretization of a continuous
# process, so it is not Markov and its transition estimates come from finite
# samples. Whether the policy survives contact with the real dynamics is a
# separate question, and it has to be measured, not assumed.

function run_real(policy_fn; episodes=300, horizon=150, seed=99)
    rng = MersenneTwister(seed)
    goals = crashes = timeouts = 0
    steps = Int[]
    offmodel = 0; total = 0
    for ep in 1:episodes
        s = rand(rng, initialstate(MDP_))
        for t in 1:horizon
            ai, known = policy_fn(s, rng)
            total += 1; known || (offmodel += 1)
            res = simulate_decision(MDP_.transition, s, ACTS[ai], rng)
            cls = classify_result(res)
            if cls === :goal
                goals += 1; push!(steps, t); break
            elseif cls === :crash || res.terminated || res.truncated
                crashes += 1; break
            elseif t == horizon
                timeouts += 1
            end
            s = res.sp
        end
    end
    return (goal=goals/episodes, crash=crashes/episodes,
            timeout=timeouts/episodes,
            median_steps=isempty(steps) ? missing : median(steps),
            offmodel=offmodel/total)
end

known_states = Set(tab.states)

function dora_policy(s, rng)
    ds = disc(s)
    if ds in known_states
        return (findfirst(==(action(planner, ds)), 1:length(ACTS)) === nothing ?
                action(planner, ds) : action(planner, ds)), true
    end
    return behaviour_action(s, rng, 0.0), false   # fall back off-model
end

baseline_policy(s, rng) = (behaviour_action(s, rng, 0.0), true)

println("\nexecuting on the REAL dynamics (300 episodes each)...")
rd = run_real(dora_policy)
rb = run_real(baseline_policy)

@printf("\n%-22s %8s %8s %8s %10s %10s\n",
        "policy", "goal", "crash", "timeout", "med steps", "off-model")
for (nm, r) in (("DORA", rd), ("lane follower", rb))
    @printf("%-22s %7.1f%% %7.1f%% %7.1f%% %10s %9.1f%%\n", nm,
            100r.goal, 100r.crash, 100r.timeout,
            r.median_steps === missing ? "-" : string(r.median_steps),
            100r.offmodel)
end
@printf("\nin-model prediction for DORA : %.1f%% success\n", 100 * sr)
@printf("measured on real dynamics    : %.1f%% success\n", 100 * rd.goal)

# ---------------------------------------------------------------------------
# 5. Why? Two hypotheses, both testable.
# ---------------------------------------------------------------------------
# H1: DORA solved the wrong problem (a solver bug or a bad SSP encoding).
# H2: DORA solved the right problem, and the problem is a bad model of reality.

# H1 is checked with the package's own exact evaluator: how far is DORA's
# learned policy from the SSP optimum it was given?
gap = (eval_policy(tab, planner.pi)[tab.start] - V[tab.start]) / V[tab.start]
@printf("\nH1  DORA policy vs SSP optimum : %.4f relative gap\n", gap)

# H2: the optimizer's curse. Transition estimates come from finite samples
# under a BEHAVIOUR policy; an optimizer prefers whichever action looks best,
# which is biased towards actions whose optimism came from having few samples.
# So: how much data backs the (s,a) pairs DORA actually chooses?
sample_count = Dict(k => sum(values(v)) for (k, v) in D.counts)

function visited_support(policy_fn; episodes=200, horizon=150, seed=7)
    rng = MersenneTwister(seed)
    counts = Int[]
    for ep in 1:episodes
        s = rand(rng, initialstate(MDP_))
        for t in 1:horizon
            ai, _ = policy_fn(s, rng)
            push!(counts, get(sample_count, (disc(s), ai), 0))
            res = simulate_decision(MDP_.transition, s, ACTS[ai], rng)
            (classify_result(res) !== :normal || res.terminated) && break
            s = res.sp
        end
    end
    return counts
end

cd_ = visited_support(dora_policy)
cb_ = visited_support(baseline_policy)
@printf("\nH2  samples backing the (s,a) each policy actually uses\n")
@printf("      %-16s median %5.1f   %%with <5 samples %5.1f\n", "DORA",
        median(cd_), 100 * count(<(5), cd_) / length(cd_))
@printf("      %-16s median %5.1f   %%with <5 samples %5.1f\n", "lane follower",
        median(cb_), 100 * count(<(5), cb_) / length(cb_))

# and the decisive comparison: what the model PREDICTS for the states DORA
# visits, against what actually happens there
@printf("\n    model crash rate  %.1f%%   real crash rate  %.1f%%   ratio %.1fx\n",
        100 * cr, 100 * rd.crash, rd.crash / cr)

# ---------------------------------------------------------------------------
# 6. Separate the two causes, then fix the one that is ours.
# ---------------------------------------------------------------------------
# The 0.476 gap says DORA had not converged on the SSP it was given. Give it
# more budget: if the gap closes but real performance does not move, then the
# abstraction — not the solver — is what is wrong.
println("\nDORA budget sweep (in-model gap vs real success):")
for b in (500, 4000, 20000)
    p = solve(DORASolver(start=start_state,
            classify = sp -> sp == GOAL_S ? :goal : sp == CRASH_S ? :crash : :normal,
            cost = (s, a, sp) -> get(ssp.C, (s, a, sp), 1.0),
            c_min=0.5, c_to=200.0, c_crash=100.0, horizon=150,
            episodes_budget=b), ssp)
    g = (eval_policy(p.tab, p.pi)[p.tab.start] - V[p.tab.start]) / V[p.tab.start]
    ks = Set(p.tab.states)
    pol(s, rng) = (disc(s) in ks ? (action(p, disc(s)), true) :
                   (behaviour_action(s, rng, 0.0), false))
    r = run_real(pol; episodes=200)
    @printf("  budget %6d   gap %.4f   real success %5.1f%%   real crash %5.1f%%\n",
            b, g, 100r.goal, 100r.crash)
end

# The fix for the optimizer's curse: an edge backed by too little data should
# not be available at all. This is the same convention the package already
# uses for a blocked move — no determinized edge — applied to "we never
# measured this".
function build_ssp_pruned(D; min_samples::Int)
    P = Dict{Tuple{NTuple{7,Int},Int}, Vector{Tuple{NTuple{7,Int},Float64}}}()
    C = Dict{Tuple{NTuple{7,Int},Int,NTuple{7,Int}}, Float64}()
    kept = dropped = 0
    for ((ds, ai), tgts) in D.counts
        n = sum(values(tgts))
        if n < min_samples
            dropped += 1; continue
        end
        kept += 1
        v = Tuple{NTuple{7,Int},Float64}[]
        for (t, k) in tgts
            st = t === :goal ? GOAL_S : t === :crash ? CRASH_S : t::NTuple{7,Int}
            push!(v, (st, k / n))
            C[(ds, ai, st)] = D.costs[(ds, ai)][t] / k
        end
        P[(ds, ai)] = v
    end
    return DuckieSSP(P, C), kept, dropped
end

println("\npruning under-sampled edges:")
for ms in (1, 10, 30, 60)
    sp_, kept, dropped = build_ssp_pruned(D; min_samples=ms)
    p = solve(DORASolver(start=start_state,
            classify = s -> s == GOAL_S ? :goal : s == CRASH_S ? :crash : :normal,
            cost = (s, a, sp) -> get(sp_.C, (s, a, sp), 1.0),
            c_min=0.5, c_to=200.0, c_crash=100.0, horizon=150,
            episodes_budget=4000), sp_)
    ks = Set(p.tab.states)
    pol(s, rng) = (disc(s) in ks ? (action(p, disc(s)), true) :
                   (behaviour_action(s, rng, 0.0), false))
    r = run_real(pol; episodes=200)
    Vp, pis = optimal_value(p.tab)
    srp, _, _ = outcome_rates(p.tab, pis)
    @printf("  min_samples %3d  edges %4d kept / %4d dropped  in-model %5.1f%%  real %5.1f%%  off-model %4.1f%%\n",
            ms, kept, dropped, 100srp, 100r.goal, 100r.offmodel)
end
