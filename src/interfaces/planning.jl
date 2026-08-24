# FJ8.1 — the solver-facing contract, written without reference to any solver.
#
# This package is a MODEL, not a planner host. Nothing here knows what MCTS
# is, what a tree is, or which library will be plugged in. The direction of
# dependency is fixed:
#
#     DuckietownDecisionModels.jl  --implements-->  POMDPs.jl contracts
#                                                          ^
#                                       any solver ---------+
#
# A solver is admitted by satisfying the model; the model is never reshaped to
# suit a solver. Three solver-agnostic pieces live here:
#
#   1. PlanningDiagnostics  what a decision cost, with an open `extra` slot
#   2. InstrumentedMDP      a transparent generative-call counter
#   3. model_capabilities   what the model offers, measured by exercising it

# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

"""
    PlanningDiagnostics(planning_time, model_calls, extra)

What producing one action cost. Only two fields are universal:

- `planning_time` — wall-clock seconds for the decision;
- `model_calls` — generative-model calls consumed, or `-1` when not measured.

Everything solver-specific goes in `extra`, an open `NamedTuple`. A tree
search may report `(tree_nodes = ..., max_depth = ..., iterations = ...)`; a
value-iteration style solver `(backups = ..., convergence_iterations = ...)`;
a particle method `(particles = ..., belief_nodes = ...)`. The evaluator never
looks inside, so it cannot assume a planner has a tree — or has any internal
structure at all, which is the normal case for a learned policy.
"""
struct PlanningDiagnostics
    planning_time::Float64
    model_calls::Int
    extra::NamedTuple
end

PlanningDiagnostics(; planning_time::Real=0.0, model_calls::Integer=-1,
    extra::NamedTuple=NamedTuple()) =
    PlanningDiagnostics(Float64(planning_time), Int(model_calls), extra)

function Base.show(io::IO, d::PlanningDiagnostics)
    print(io, "PlanningDiagnostics(", round(1e3 * d.planning_time; digits=3),
        " ms, ", d.model_calls < 0 ? "calls n/a" : "$(d.model_calls) calls")
    isempty(d.extra) || print(io, ", ", d.extra)
    print(io, ")")
end

# ---------------------------------------------------------------------------
# A transparent counter
# ---------------------------------------------------------------------------

"""
    InstrumentedMDP(mdp)

The same model, counting generative calls. It forwards every POMDPs.jl
function and — through `getproperty` — every field, so anything written
against `DuckietownMDP` works on it unchanged.

It is a *measuring device*, not a translation layer: it must not, and does
not, alter states, rewards, transitions, action bounds or termination. That
transparency is asserted bitwise in `test/test_fj8_planning.jl` rather than
merely claimed here.

Only calls through `POMDPs.gen` are counted, which is exactly a planner's
consumption of the model — the evaluator's own environment step goes through
`simulate_decision` and is deliberately not charged as planning cost.
"""
mutable struct InstrumentedMDP{A,M<:DuckietownMDP{A}} <: MDP{DuckieWorldState,A}
    inner::M
    calls::Int
end

InstrumentedMDP(m::DuckietownMDP{A}) where {A} =
    InstrumentedMDP{A,typeof(m)}(m, 0)

"""
    MDPLike{A}

The model or a transparent wrapper around it, for a given action type. Code
that only reads the model accepts this so a diagnostic wrapper never forces a
second implementation of anything.
"""
const MDPLike{A} = Union{DuckietownMDP{A},InstrumentedMDP{A}}

"""
    AnyMDPLike

[`MDPLike`](@ref) for either action type.
"""
const AnyMDPLike = Union{DuckietownMDP,InstrumentedMDP}

function Base.getproperty(m::InstrumentedMDP, f::Symbol)
    (f === :inner || f === :calls) && return getfield(m, f)
    return getproperty(getfield(m, :inner), f)
end

Base.propertynames(m::InstrumentedMDP) =
    (:inner, :calls, propertynames(getfield(m, :inner))...)

Base.show(io::IO, m::InstrumentedMDP) =
    print(io, "InstrumentedMDP(", getfield(m, :inner), ", ",
        getfield(m, :calls), " calls)")

"""
    model_calls(mdp) -> Int

Generative calls consumed so far; `-1` for an uninstrumented model, so callers
report "not measured" rather than a misleading zero.
"""
model_calls(m::InstrumentedMDP) = getfield(m, :calls)
model_calls(::DuckietownMDP) = -1

"""
    reset_model_calls!(mdp) -> Int

Zero the counter and return its previous value.
"""
function reset_model_calls!(m::InstrumentedMDP)
    old = getfield(m, :calls)
    setfield!(m, :calls, 0)
    return old
end
reset_model_calls!(::DuckietownMDP) = -1

function POMDPs.gen(m::InstrumentedMDP, s::DuckieWorldState, a,
    rng::AbstractRNG)
    setfield!(m, :calls, getfield(m, :calls) + 1)
    return POMDPs.gen(getfield(m, :inner), s, a, rng)
end

POMDPs.discount(m::InstrumentedMDP) = POMDPs.discount(getfield(m, :inner))
POMDPs.actions(m::InstrumentedMDP) = POMDPs.actions(getfield(m, :inner))
POMDPs.actions(m::InstrumentedMDP, s::DuckieWorldState) =
    POMDPs.actions(getfield(m, :inner), s)
POMDPs.actionindex(m::InstrumentedMDP, a) =
    POMDPs.actionindex(getfield(m, :inner), a)
POMDPs.isterminal(m::InstrumentedMDP, s::DuckieWorldState) =
    POMDPs.isterminal(getfield(m, :inner), s)
POMDPs.initialstate(m::InstrumentedMDP) =
    POMDPs.initialstate(getfield(m, :inner))
is_truncated(m::InstrumentedMDP, s::DuckieWorldState) =
    is_truncated(getfield(m, :inner), s)

# ---------------------------------------------------------------------------
# The bridge every POMDPs.jl solver gets for free
# ---------------------------------------------------------------------------

"""
    policy_action(policy, mdp, s) -> action

Ask any policy for its action, reconciling the two calling conventions in
play — **inside this package**, without adding a method to `POMDPs.action`.

POMDPs.jl's contract is `action(policy, x)`: `solve` has already bound the
model into the policy, so a planner needs no model argument. This package's
evaluator passes the model explicitly, because a *stateless* policy (a Q-table,
an actor network) is reusable across models and needs it.

Extending `POMDPs.action` with a three-argument form would change how a
generic POMDPs.jl policy behaves for everyone who loads this package — the
opposite of the goal, which is that solvers plug in without the model altering
the ecosystem around them. So the adaptation lives here, on a function this
package owns:

- a `POMDPs.Policy` (anything `solve` returns) is asked the standard way;
- an `AbstractPolicy` (this package's tabular and actor adapters) is asked with
  the model, which is how those are defined;
- anything else falls back to the three-argument form, so a hand-written
  policy that defines it keeps working.
"""
policy_action(p::POMDPs.Policy, ::AnyMDPLike, s::DuckieWorldState) =
    POMDPs.action(p, s)
policy_action(p::AbstractPolicy, m::AnyMDPLike, s::DuckieWorldState) =
    POMDPs.action(p, m, s)
policy_action(p, m, s::DuckieWorldState) = POMDPs.action(p, m, s)

"""
    plan_action(policy, mdp, s) -> (action, PlanningDiagnostics)

One decision plus what it cost. The default times [`policy_action`](@ref) and
reads the model's call counter, which already works for *any* solver without
the solver knowing this package exists.

A solver extension may add a method that fills richer `extra` fields. That is
the only thing an extension is ever expected to add, and it is optional.
"""
function plan_action(p, m, s::DuckieWorldState)
    before = model_calls(m)
    t0 = time_ns()
    a = policy_action(p, m, s)
    dt = (time_ns() - t0) / 1e9
    after = model_calls(m)
    used = (before < 0 || after < 0) ? -1 : after - before
    return a, PlanningDiagnostics(dt, used, NamedTuple())
end

# ---------------------------------------------------------------------------
# Capabilities
# ---------------------------------------------------------------------------

_action_eltype(::MDPLike{MacroAction}) = MacroAction
_action_eltype(::MDPLike{DuckieAction}) = DuckieAction

"""
    model_capabilities(mdp; seeds) -> NamedTuple

What this model actually offers, determined by **exercising the interface**
rather than by asserting it. The question a new solver should raise is "are
this solver's requirements met by the model?", not "can the model be bent to
fit the solver?" — this is the answer to the first form.

Stochasticity is reported as **three separate facts**, because this model is
*conditionally* stochastic and collapsing that into one flag would mislead a
planner:

- `consumes_rng` — the transition draws from the caller's stream at *some*
  reachable state;
- `stochastic_state_fraction` — at what fraction of the probed states it does
  so. The pedestrian trigger only fires when a duck is armed, ahead and inside
  the trigger distance window, so most states have a deterministic successor;
- `stochastic_outcomes` — whether two seeds were actually observed to produce
  different successors.

A planner that widens over *states* (DPW's `k_state`/`alpha_state`) is doing
useful work only in the fraction of states counted here; elsewhere the
successor is a function of `(s, a)` alone. A `false` is a statement about the
states and seeds probed, never a proof of determinism.

Pass `policy` to drive the probe trajectory with a competent controller. This
matters: under a constant action the vehicle leaves the road within a few
decisions and never reaches the states where a pedestrian can trigger, so the
default probe under-reports what the model can do. The policy is only a way of
reaching representative states — no property reported here depends on which
policy is used, only on which states it visits.
"""
function model_capabilities(m::AnyMDPLike; seeds=1:8, state_seed::Integer=11,
    probe_states::Integer=80, policy=nothing)
    s0 = rand(MersenneTwister(state_seed), POMDPs.initialstate(m))
    acts = POMDPs.actions(m)
    enumerable = applicable(length, acts)
    a1 = enumerable ? first(acts) : rand(MersenneTwister(1), acts)

    generative = false
    try
        x = POMDPs.gen(m, s0, a1, MersenneTwister(1))
        generative = haskey(x, :sp) && haskey(x, :r) &&
            x.sp isa DuckieWorldState && x.r isa Real
    catch
        generative = false
    end

    # Walk a trajectory so the probe is not confined to the spawn state, where
    # the duck is always far away and nothing can trigger.
    states, chosen = _probe_states(m, s0, a1, probe_states, policy)
    n_stochastic = 0
    outcomes_differ = false
    for (s, a) in zip(states, chosen)
        probe = MersenneTwister(1)
        POMDPs.gen(m, s, a, probe)
        probe == MersenneTwister(1) && continue
        n_stochastic += 1
        outcomes_differ && continue
        base = POMDPs.gen(m, s, a, MersenneTwister(first(seeds))).sp
        for k in seeds
            if !worlds_identical(base, POMDPs.gen(m, s, a, MersenneTwister(k)).sp)
                outcomes_differ = true
                break
            end
        end
    end

    g = POMDPs.discount(m)
    return (
        generative_transition=generative,
        discrete_actions=_action_eltype(m) === MacroAction,
        continuous_actions=_action_eltype(m) === DuckieAction,
        enumerable_actions=enumerable,
        continuous_state=true,
        consumes_rng=n_stochastic > 0,
        stochastic_state_fraction=isempty(states) ? 0.0 :
            n_stochastic / length(states),
        stochastic_outcomes=outcomes_differ,
        terminal_states=applicable(POMDPs.isterminal, m, s0),
        truncation_separate_from_termination=applicable(is_truncated, m, s0),
        discount=0.0 < g <= 1.0,
        initial_state_sampler=s0 isa DuckieWorldState,
        explicit_transition_distribution=false,
        observation_model=false,
        belief_updater=false,
    )
end

"""States along one driven trajectory, with the action taken at each, used so
capability probing is not confined to the spawn state. Returns
`(states, actions)`."""
function _probe_states(m::AnyMDPLike, s0::DuckieWorldState, a_default,
    n::Integer, policy)
    states = DuckieWorldState[]
    chosen = Any[]
    rng = MersenneTwister(99)
    s = s0
    while length(states) < n
        a = policy === nothing ? a_default : policy_action(policy, m, s)
        push!(states, s)
        push!(chosen, a)
        r = simulate_decision(m.transition, s, a, rng)
        s = r.sp
        (r.terminated || r.truncated) && break
    end
    return states, chosen
end

"""
    capability_report(mdp) -> String

[`model_capabilities`](@ref) rendered as the compatibility table used in the
gate documents.
"""
function capability_report(m::AnyMDPLike)
    caps = model_capabilities(m)
    width = maximum(length(String(k)) for k in keys(caps))
    io = IOBuffer()
    for k in keys(caps)
        v = caps[k]
        println(io, rpad(String(k), width), "  ",
            v === true ? "YES" : v === false ? "NO" : string(v))
    end
    return String(take!(io))
end
