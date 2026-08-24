# FJ7.6: one solver-independent evaluation harness.
#
# Every policy family — tabular (Q-learning, SARSA), learned continuous
# (SAC, TD3) and later online planners (MCTS, DPW) — is evaluated through the
# SAME function on the SAME validated MDP. No solver gets its own environment
# semantics; the only thing that differs is how an action is chosen.
#
#     evaluate_policy(mdp, policy; seeds, max_steps)
#
# `policy` is anything `policy_action` can ask for a decision — a POMDPs.jl
# planner returned by `solve`, one of this package's own adapters, or a
# hand-written policy — so the harness never needs to know which family it
# belongs to, and no solver is privileged.

"""
    EpisodeMetrics

Outcome of one evaluated episode. Every field is derived from the rollout log,
never from a separate bookkeeping path.
"""
struct EpisodeMetrics
    seed::Int
    decisions::Int
    ret::Float64                    # undiscounted return
    discounted_return::Float64
    progress::Float64               # sum of the progress reward term
    mean_abs_d::Float64
    max_abs_d::Float64
    mean_abs_phi::Float64
    mean_speed::Float64
    brake_ratio::Float64            # fraction of decisions with zero commanded speed
    offroad::Bool
    other_collision::Bool
    duck_collision::Bool
    timeout::Bool
    goal::Bool
    full_stops::Int
    stop_violations::Int
    passed_stops::Int
    stop_zone_decisions::Int        # decisions with d_stop inside the stop zone
    min_d_stop::Union{Nothing,Float64}
    duck_yield_decisions::Int       # crossing duck ahead AND ego slow
    duck_active_decisions::Int
    crossings::Int
    reason::String
end

"""
    DecisionRecord

One decision's cost, with the episode and step it belongs to. FJ8.4a measured
a 3.8x spread in generative cost across states, so a flat list of diagnostics
throws away exactly the structure needed to explain a high p95 latency.
"""
struct DecisionRecord
    seed::Int
    step::Int
    diagnostics::PlanningDiagnostics
end

_push_record!(rec::Vector{PlanningDiagnostics}, ::Int, ::Int,
    d::PlanningDiagnostics) = push!(rec, d)
_push_record!(rec::Vector{DecisionRecord}, seed::Int, step::Int,
    d::PlanningDiagnostics) = push!(rec, DecisionRecord(seed, step, d))

# a trace vector is filled after the transition, not here
_push_record!(::Vector, ::Int, ::Int, ::PlanningDiagnostics) = nothing

summarize_planning(rs::AbstractVector{DecisionRecord}) =
    summarize_planning([r.diagnostics for r in rs])

# ---------------------------------------------------------------------------
# FJ8.4c — per-decision observational trace
# ---------------------------------------------------------------------------
#
# The logger records values the evaluator ALREADY computed. It calls no
# observer, no policy and no transition of its own: everything below is read
# off the `TransitionResult` the decision produced, the state it was taken
# from, and the `PlanningDiagnostics` the decision already carried.
#
# That restriction is the whole point. An extra `get_raw_state`, an extra
# `action`, or an extra draw from the caller's rng would make the enriched run
# a DIFFERENT experiment, and FJ9.5b already showed how easily a second call
# produces a second search.
#
# Row k means: "from this pose, the policy took this action, and this is what
# the decision produced". The projections and reward are post-transition,
# because that is what the chain computes; the pose is pre-transition, because
# that is the state the policy saw. Both are stated rather than implied.

"""
    DecisionTrace

One decision, recorded observationally. Every field is a value the evaluator
had in hand at the moment the decision completed.
"""
struct DecisionTrace
    solver::String
    seed::Int
    decision::Int
    # pose the decision was taken FROM
    ego_x::Float64
    ego_z::Float64
    ego_angle::Float64
    ego_speed::Float64
    # action
    action_kind::String
    action_id::Int          # -1 for continuous
    v_cmd::Float64
    omega_cmd::Float64
    wheel_left::Float64
    wheel_right::Float64
    # projections AFTER the transition (what the reward and events used)
    d::Float64
    phi::Float64
    v::Float64
    kappa::Float64
    tile::String
    d_stop::Union{Nothing,Float64}
    sigma_stop::Bool
    stop_hold_progress::Float64
    duck_present::Bool
    duck_longitudinal::Float64
    duck_lateral::Float64
    duck_active::Bool
    duck_class::String
    # from the successor STATE, which the projections do not carry
    duck_active_state::Bool
    crossings_started::Int
    # reward
    reward_total::Float64
    reward_progress::Float64
    reward_lateral::Float64
    reward_heading::Float64
    reward_time::Float64
    reward_pedestrian::Float64
    reward_stagnation::Float64
    reward_stop_approach::Float64
    reward_steering::Float64
    reward_events::Float64
    # events and outcome
    full_stop::Bool
    passed_stop::Bool
    stop_violation::Bool
    offroad::Bool
    other_collision::Bool
    duck_collision::Bool
    timeout::Bool
    goal::Bool
    terminated::Bool
    truncated::Bool
    reason::String
    # planning cost
    planning_time::Float64
    model_calls::Int
end

_action_kind(::MacroAction) = "macro"
_action_kind(::DuckieAction) = "continuous"
_action_kind(a) = string(typeof(a))

_action_id(a::MacroAction) = Int(a)
_action_id(a) = -1

_cmd_v(a::MacroAction, mdp) = build_action_table(mdp.transition.action_cfg)[Int(a) + 1].v
_cmd_v(a::DuckieAction, mdp) = a.v
_cmd_omega(a::MacroAction, mdp) = build_action_table(mdp.transition.action_cfg)[Int(a) + 1].omega
_cmd_omega(a::DuckieAction, mdp) = a.omega

"""
    _trace(solver, seed, k, s, a, r, diag, mdp) -> DecisionTrace

Assemble one row from values already computed. No call here recomputes
anything the decision did not already produce.
"""
function _trace(solver, seed, k, s, a, r::TransitionResult,
    diag::PlanningDiagnostics, mdp)
    c = r.continuous_state
    return DecisionTrace(String(solver), Int(seed), Int(k),
        s.ego.pos[1], s.ego.pos[3], s.ego.angle, s.ego.speed,
        _action_kind(a), _action_id(a), _cmd_v(a, mdp), _cmd_omega(a, mdp),
        Float64(r.wheel_commands[1]), Float64(r.wheel_commands[2]),
        r.raw_state.d, r.raw_state.phi, r.raw_state.v, c.kappa,
        string(r.raw_state.tile), r.raw_state.d_stop, r.raw_state.sigma_stop,
        c.stop_hold_progress, c.duck_present, c.duck_longitudinal,
        c.duck_lateral, c.duck_active, string(r.raw_state.duck),
        any(d -> d.pedestrian_active, r.sp.ducks),
        sum(r.sp.crossings_started; init=0),
        r.reward.total, r.reward.progress, r.reward.lateral, r.reward.heading,
        r.reward.time, r.reward.pedestrian, r.reward.stagnation,
        r.reward.stop_approach, r.reward.steering, r.reward.events,
        r.events.full_stop, r.events.passed_stop, r.events.stop_violation,
        r.events.offroad, r.events.other_collision, r.events.collision_duck,
        r.events.timeout, r.events.goal,
        r.terminated, r.truncated, lowercase(string(r.reason)),
        diag.planning_time, diag.model_calls)
end

const DECISION_TRACE_SCHEMA = Tuple(String.(fieldnames(DecisionTrace)))

"""
    decision_csv(traces) -> String
"""
function decision_csv(ts::AbstractVector{DecisionTrace})
    io = IOBuffer()
    println(io, join(DECISION_TRACE_SCHEMA, ","))
    for t in ts
        vals = Any[]
        for f in fieldnames(DecisionTrace)
            v = getfield(t, f)
            push!(vals, v === nothing ? "" : v)
        end
        println(io, join(vals, ","))
    end
    return String(take!(io))
end

"""
    reaggregate_episodes(traces) -> Vector{EpisodeMetrics}

Rebuild episode metrics from the per-decision trace, using the same
definitions [`evaluate_policy`](@ref) uses.

If the two ever disagree, that is a genuine finding about the enrichment run
and not something to reconcile with a tolerance — the protocol is frozen and
deterministic, so exact reproduction is the expectation.
"""
function reaggregate_episodes(ts::AbstractVector{DecisionTrace};
    stop_zone::Real=0.45, yield_speed::Real=0.04, discount::Real=0.99)
    out = EpisodeMetrics[]
    for seed in sort(unique(t.seed for t in ts))
        rows = sort(filter(t -> t.seed == seed, ts); by=t -> t.decision)
        isempty(rows) && continue
        n = length(rows)
        g = 1.0
        disc = 0.0
        for r in rows
            disc += g * r.reward_total
            g *= discount
        end
        min_stop = nothing
        for r in rows
            r.d_stop === nothing && continue
            min_stop = min_stop === nothing ? r.d_stop :
                min(min_stop, r.d_stop)
        end
        push!(out, EpisodeMetrics(seed, n,
            sum(r -> r.reward_total, rows), disc,
            sum(r -> r.reward_progress, rows),
            sum(r -> abs(r.d), rows) / n,
            maximum(r -> abs(r.d), rows),
            sum(r -> abs(r.phi), rows) / n,
            sum(r -> r.v, rows) / n,
            count(r -> r.v_cmd == 0.0, rows) / n,
            any(r -> r.offroad, rows), any(r -> r.other_collision, rows),
            any(r -> r.duck_collision, rows), any(r -> r.timeout, rows),
            any(r -> r.goal, rows),
            count(r -> r.full_stop, rows), count(r -> r.stop_violation, rows),
            count(r -> r.passed_stop, rows),
            count(r -> r.d_stop !== nothing && r.d_stop <= stop_zone, rows),
            min_stop,
            count(r -> r.duck_class in ("CROSSING_FAR", "CROSSING_NEAR") &&
                  r.v < yield_speed, rows),
            count(r -> r.duck_active_state, rows),
            rows[end].crossings_started,
            rows[end].reason))
    end
    return out
end

"""
    evaluate_policy(mdp, policy; seeds, max_steps, stop_zone, yield_speed)
        -> Vector{EpisodeMetrics}

Run one episode per seed: sample `x0` from `initialstate(mdp)`, then act
greedily with `policy` until the episode ends or `max_steps` decisions elapse.
Returns the per-episode metrics; use [`summarize_evaluation`](@ref) to
aggregate.

Pass `record` a `Vector{PlanningDiagnostics}` — or a
`Vector{DecisionRecord}` to keep the episode and step each measurement belongs
to — to also collect what each decision cost. This changes nothing about the episode — the same action is
taken either way — it only routes the call through [`plan_action`](@ref) so the
cost can be observed. Task performance and planning cost are reported
separately and never mixed into one score.
"""
function evaluate_policy(mdp::AnyMDPLike, policy;
    seeds=1:5, max_steps::Integer=250,
    stop_zone::Real=mdp.transition.state_cfg.stop_zone,
    yield_speed::Real=mdp.transition.reward_cfg.duck_yield_speed,
    record::Union{Nothing,Vector{PlanningDiagnostics},
        Vector{DecisionRecord},Vector{DecisionTrace}}=nothing,
    trace_solver::AbstractString="")
    out = EpisodeMetrics[]
    for seed in seeds
        rng = MersenneTwister(seed)
        s = rand(MersenneTwister(seed), initialstate(mdp))
        acc = 0.0
        disc = 0.0
        g = 1.0
        progress = 0.0
        sum_d = 0.0
        max_d = 0.0
        sum_phi = 0.0
        sum_v = 0.0
        n_brake = 0
        n_full = 0
        n_viol = 0
        n_passed = 0
        n_zone = 0
        n_yield = 0
        n_duck_active = 0
        min_stop = nothing
        crossings = 0
        reason = "in_progress"
        offroad = other = duck = timeout = goal = false
        k = 0
        while k < max_steps
            s_before = s
            diag = PlanningDiagnostics(0.0, -1, NamedTuple())
            a = if record === nothing
                policy_action(policy, mdp, s)
            else
                act, d = plan_action(policy, mdp, s)
                diag = d
                _push_record!(record, Int(seed), k + 1, d)
                act
            end
            r = simulate_decision(mdp.transition, s, a, rng)
            # Observational only: every value below already exists at this
            # point. No observer, policy or transition is called again, and no
            # draw is taken from `rng`.
            record isa Vector{DecisionTrace} && push!(record,
                _trace(trace_solver, seed, k + 1, s_before, a, r, diag, mdp))
            k += 1
            acc += r.reward.total
            disc += g * r.reward.total
            g *= mdp.discount
            progress += r.reward.progress
            sum_d += abs(r.raw_state.d)
            max_d = max(max_d, abs(r.raw_state.d))
            sum_phi += abs(r.raw_state.phi)
            sum_v += r.raw_state.v
            _commanded_speed(a, mdp) == 0.0 && (n_brake += 1)
            r.events.full_stop && (n_full += 1)
            r.events.stop_violation && (n_viol += 1)
            r.events.passed_stop && (n_passed += 1)
            if r.raw_state.d_stop !== nothing
                r.raw_state.d_stop <= stop_zone && (n_zone += 1)
                min_stop = min_stop === nothing ? r.raw_state.d_stop :
                    min(min_stop, r.raw_state.d_stop)
            end
            crossing = r.raw_state.duck in (CROSSING_FAR, CROSSING_NEAR)
            crossing && r.raw_state.v < yield_speed && (n_yield += 1)
            # counted over every duck, not just the first: the shipped configs
            # have one, but a silent single-duck assumption would misreport any
            # scenario that adds another
            any(d -> d.pedestrian_active, r.sp.ducks) && (n_duck_active += 1)
            crossings = sum(r.sp.crossings_started; init=0)
            reason = lowercase(string(r.reason))
            offroad |= r.events.offroad
            other |= r.events.other_collision
            duck |= r.events.collision_duck
            timeout |= r.events.timeout
            goal |= r.events.goal
            s = r.sp
            (r.terminated || r.truncated) && break
        end
        n = max(k, 1)
        push!(out, EpisodeMetrics(Int(seed), k, acc, disc, progress,
            sum_d / n, max_d, sum_phi / n, sum_v / n, n_brake / n,
            offroad, other, duck, timeout, goal,
            n_full, n_viol, n_passed, n_zone, min_stop, n_yield,
            n_duck_active, crossings, reason))
    end
    return out
end

_commanded_speed(a::MacroAction, mdp::AnyMDPLike) =
    build_action_table(mdp.transition.action_cfg)[Int(a) + 1].v
_commanded_speed(a::DuckieAction, ::AnyMDPLike) = a.v

"""
    summarize_evaluation(metrics) -> NamedTuple

Aggregate episodes into the comparison row used across solvers.
"""
function summarize_evaluation(ms::AbstractVector{EpisodeMetrics})
    isempty(ms) && return (episodes=0,)
    n = length(ms)
    mean_(f) = sum(f, ms) / n
    return (episodes=n,
        mean_return=mean_(m -> m.ret),
        mean_discounted_return=mean_(m -> m.discounted_return),
        mean_length=mean_(m -> m.decisions),
        mean_progress=mean_(m -> m.progress),
        mean_abs_d=mean_(m -> m.mean_abs_d),
        max_abs_d=maximum(m -> m.max_abs_d, ms),
        mean_abs_phi=mean_(m -> m.mean_abs_phi),
        mean_speed=mean_(m -> m.mean_speed),
        brake_ratio=mean_(m -> m.brake_ratio),
        offroad=count(m -> m.offroad, ms),
        other_collision=count(m -> m.other_collision, ms),
        duck_collision=count(m -> m.duck_collision, ms),
        timeout=count(m -> m.timeout, ms),
        goal=count(m -> m.goal, ms),
        full_stops=sum(m -> m.full_stops, ms),
        stop_violations=sum(m -> m.stop_violations, ms),
        passed_stops=sum(m -> m.passed_stops, ms),
        stop_zone_decisions=sum(m -> m.stop_zone_decisions, ms),
        duck_yield_decisions=sum(m -> m.duck_yield_decisions, ms),
        crossings=sum(m -> m.crossings, ms),
        reasons=_count_reasons(ms))
end

function _count_reasons(ms)
    d = Dict{String,Int}()
    for m in ms
        d[m.reason] = get(d, m.reason, 0) + 1
    end
    return d
end

# ---------------------------------------------------------------------------
# Planning cost — kept strictly apart from task performance
# ---------------------------------------------------------------------------

"""
    PlannerCost

What producing the decisions cost, aggregated over an evaluation. This is
never combined with return, progress or any other task metric: a slow planner
that drives well and a fast one that drives badly must stay distinguishable.

`extra` holds the mean of every numeric field the planner reported in its
`PlanningDiagnostics.extra`, so a tree search's node counts and a particle
method's belief counts both aggregate without the evaluator knowing either.
"""
struct PlannerCost
    decisions::Int
    total_time::Float64
    latency_mean::Float64
    latency_p50::Float64
    latency_p95::Float64
    latency_max::Float64
    model_calls_total::Int
    model_calls_per_action::Float64
    extra::Dict{Symbol,Float64}
end

_quantile_sorted(v::Vector{Float64}, q::Real) = isempty(v) ? NaN :
    v[clamp(ceil(Int, q * length(v)), 1, length(v))]

"""
    summarize_planning(diagnostics) -> PlannerCost

Aggregate per-decision diagnostics. `model_calls_total` is `-1` when the model
was not instrumented, rather than `0`, so "not measured" never reads as "free".
"""
function summarize_planning(ds::AbstractVector{PlanningDiagnostics})
    n = length(ds)
    n == 0 && return PlannerCost(0, 0.0, NaN, NaN, NaN, NaN, -1, NaN,
        Dict{Symbol,Float64}())
    lat = sort!([d.planning_time for d in ds])
    total = sum(lat)
    measured = all(d -> d.model_calls >= 0, ds)
    calls = measured ? sum(d -> d.model_calls, ds) : -1
    per_action = measured ? calls / n : NaN

    extra = Dict{Symbol,Float64}()
    counts = Dict{Symbol,Int}()
    for d in ds, k in keys(d.extra)
        v = d.extra[k]
        v isa Real || continue
        extra[k] = get(extra, k, 0.0) + Float64(v)
        counts[k] = get(counts, k, 0) + 1
    end
    for k in keys(extra)
        extra[k] /= counts[k]
    end

    return PlannerCost(n, total, total / n, _quantile_sorted(lat, 0.50),
        _quantile_sorted(lat, 0.95), lat[end], calls, per_action, extra)
end

"""
    evaluate_planner(mdp, planner; seeds, max_steps) -> (episodes, cost)

Evaluate any planner through the **same** [`evaluate_policy`](@ref) used for
the learned and tabular baselines, and return its planning cost alongside —
two separate values, deliberately not one score.

Wrap `mdp` in an [`InstrumentedMDP`](@ref) before `solve` to have
`model_calls` measured; without it the cost report says so rather than
claiming zero.
"""
function evaluate_planner(mdp::AnyMDPLike, planner; seeds=1:5,
    max_steps::Integer=250, kwargs...)
    record = DecisionRecord[]
    episodes = evaluate_policy(mdp, planner; seeds=seeds, max_steps=max_steps,
        record=record, kwargs...)
    return (episodes=episodes, cost=summarize_planning(record),
        decisions=record)
end

"""
    compare_policies(mdp_by_name, policies; seeds, max_steps) -> Dict

Evaluate several policies and return one summary per name. Each policy is run
on the MDP appropriate to its action space (tabular policies need the discrete
variant, actor policies the continuous one), which is why the MDPs are passed
per name — the *problem definition* is the same either way, only the action
representation differs.
"""
function compare_policies(mdp_by_name::AbstractDict, policies::AbstractDict;
    seeds=1:5, max_steps::Integer=250)
    out = Dict{String,Any}()
    for (name, pol) in policies
        mdp = mdp_by_name[name]
        out[String(name)] = summarize_evaluation(
            evaluate_policy(mdp, pol; seeds=seeds, max_steps=max_steps))
    end
    return out
end
