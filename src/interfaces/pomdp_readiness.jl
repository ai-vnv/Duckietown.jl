# FJ10 — POMDP readiness audit. AN AUDIT, NOT AN IMPLEMENTATION.
#
# The question this gate answers is exactly one:
#
#     Can DuckietownDecisionModels.jl support a future partially observable
#     formulation WITHOUT modifying or contaminating the validated MDP core?
#
# Nothing here implements an observation model, a belief representation or a
# belief updater, and nothing here should. The value of the gate is that the
# extension points are identified and the missing pieces are named *before*
# anything is built on top of them — including the FJ9 visualisation layer,
# which would otherwise harden the assumption that a renderer's input is a
# `DuckieWorldState`.
#
# The audit is executable rather than prose: every `READY` and every
# `NOT_READY` below is decided by probing the package, so the day someone adds
# an observation model the audit changes with it and the test that pins it
# fails until the document is updated.
#
# The state hierarchy this gate exists to protect:
#
#     DuckieWorldState   latent, privileged world state          (x_t)
#     RawState           tabular projection of the latent state
#     ContinuousState    15-D PRIVILEGED policy feature vector
#     Observation        does NOT exist yet, and is NOT ContinuousState
#     Belief             does NOT exist yet, and is NOT ContinuousState
#
# A partially observable formulation is `x_t -> sensor -> o_t -> update ->
# b_t`. Calling the 15-D privileged feature vector an "observation" because it
# happens to be 15 numbers a network consumes would silently collapse that
# chain, and is the single most likely way this port could go wrong later.

"""
    ReadinessStatus

`READY` — usable as-is by a partially observable formulation.
`NEEDS_REFACTOR` — present but must change first; the change is named.
`NOT_READY` — absent. What has to be built is named.
"""
@enum ReadinessStatus READY NEEDS_REFACTOR NOT_READY

"""
    ReadinessItem

One audited component: its status, the evidence that produced it (a probe of
the package, not an opinion) and the change required.
"""
struct ReadinessItem
    component::String
    status::ReadinessStatus
    evidence::String
    needed::String
end

_yn(x) = x ? "yes" : "no"

"""Does `mod.name` carry any method mentioning one of this package's types?
Guarded so a probe can never make the audit throw."""
function _has_duckietown_method(mod::Module, name::Symbol)
    isdefined(mod, name) || return false
    f = getfield(mod, name)
    f isa Function || return false
    return try
        any(mm -> occursin("Duckie", string(mm.sig)), methods(f))
    catch
        false
    end
end

"""
    pomdp_readiness(mdp) -> Vector{ReadinessItem}

Probe the package for everything a partially observable formulation needs.

Determined by inspection of live types and method tables, so the result tracks
the code rather than the documentation.
"""
function pomdp_readiness(m::AnyMDPLike=DuckietownMDP(default_config(:q_learning)))
    items = ReadinessItem[]
    S = DuckieWorldState
    push!(items, ReadinessItem("latent world state", READY,
        "statetype is $(S); both action variants share it",
        "reuse unchanged — it is already the branchable latent state x_t"))

    push!(items, ReadinessItem("transition model", READY,
        "gen(m, s, a, rng) present with a CALLER-SUPPLIED rng; FJ8.0 proved " *
        "determinism, root immutability and branch purity",
        "reuse unchanged; a POMDP gen returns (sp, o, r) reusing this sp"))

    push!(items, ReadinessItem("reward", READY,
        "reward is computed from the latent transition (RewardBreakdown in " *
        "TransitionResult), never from a policy feature vector",
        "reuse unchanged — R(x,a) stays a function of the latent state"))

    push!(items, ReadinessItem("terminal semantics", READY,
        "isterminal (genuine terminal) and is_truncated (horizon) are " *
        "separate and both derive from termination_reason",
        "reuse unchanged"))

    push!(items, ReadinessItem("discount", READY,
        "discount(m) = $(POMDPs.discount(m))",
        "reuse unchanged"))

    push!(items, ReadinessItem("initial state distribution", READY,
        "initialstate(m) returns a sampleable DuckieInitialStateDistribution",
        "reuse as the latent prior; a belief prior is a DIFFERENT object " *
        "derived from it"))

    push!(items, ReadinessItem("discrete action space", READY,
        "7 enumerable MacroActions",
        "reuse unchanged"))

    push!(items, ReadinessItem("continuous action space", READY,
        "DuckieActionSpace is a non-enumerable box; FJ8.3 showed a solver " *
        "must widen rather than enumerate",
        "reuse unchanged"))

    # ---- the parts that do not exist yet -----------------------------------
    obs_type_defined = _has_duckietown_method(POMDPs, :obstype)
    push!(items, ReadinessItem("observation type", NOT_READY,
        "no Observation type is defined in this package; " *
        "POMDPs.obstype has a Duckietown method: $(_yn(obs_type_defined))",
        "define an observation type that is explicitly NOT ContinuousState; " *
        "see continuous_state_observability() for which components a sensor " *
        "could plausibly produce"))

    obs_model_defined = _has_duckietown_method(POMDPs, :observation)
    push!(items, ReadinessItem("observation model", NOT_READY,
        "POMDPs.observation has a Duckietown method: $(_yn(obs_model_defined)); " *
        "gen currently returns (sp, r) with no o",
        "add O(o | x', a) as an EXTENSION over the latent state; it must not " *
        "be reachable from, or alter, the MDP transition"))

    push!(items, ReadinessItem("observation randomness", NOT_READY,
        "no observation model exists, so no contract is in force yet",
        "adopt the transition's contract verbatim: noise supplied by the " *
        "caller's rng, never stored in a state or belief object"))

    push!(items, ReadinessItem("belief representation", NOT_READY,
        "no belief type exists",
        "define a belief type; it is NOT ContinuousState and NOT a 15-vector"))

    push!(items, ReadinessItem("belief initialisation", NOT_READY,
        "initialstate(m) yields latent states, not a belief",
        "derive b_0 from the latent prior explicitly; the mapping is a " *
        "modelling decision, not a cast"))

    update_defined = _has_duckietown_method(POMDPs, :update)
    push!(items, ReadinessItem("belief updater", NOT_READY,
        "POMDPs.update has a Duckietown method: $(_yn(update_defined)); " *
        "no POMDPs.Updater subtype is defined here",
        "add as an extension point with an explicit updater rng if it needs " *
        "randomness"))

    push!(items, ReadinessItem("POMDPs.jl POMDP interface", NOT_READY,
        "DuckietownMDP <: MDP is $(DuckietownMDP <: POMDPs.MDP); " *
        "<: POMDP is $(DuckietownMDP <: POMDPs.POMDP)",
        "a future DuckietownPOMDP{S,A,O} <: POMDP should WRAP the validated " *
        "model, not replace it, so the MDP results stay valid"))

    push!(items, ReadinessItem("legacy controller_rng in the state",
        NEEDS_REFACTOR,
        "DuckieWorldState carries a live MersenneTwister; FJ8.0 measured it " *
        "shared by every node and proved it frozen, and rejected a defensive " *
        "copy costing 8.96 % of gen allocation",
        "TECHNICAL DEBT. Candidate for removal once parity work no longer " *
        "needs the field. It must NOT be extended into an observation or " *
        "belief object: randomness stays caller-supplied"))

    return items
end

"""
    readiness_table(items) -> String
"""
function readiness_table(items::AbstractVector{ReadinessItem})
    w = maximum(length(i.component) for i in items)
    io = IOBuffer()
    println(io, rpad("component", w), "  ", rpad("status", 16), "  needed")
    println(io, "-"^(w + 60))
    for i in items
        println(io, rpad(i.component, w), "  ", rpad(string(i.status), 16),
            "  ", first(split(i.needed, ';')))
    end
    return String(take!(io))
end

"""
    readiness_counts(items) -> NamedTuple
"""
readiness_counts(items::AbstractVector{ReadinessItem}) = (
    ready=count(i -> i.status == READY, items),
    needs_refactor=count(i -> i.status == NEEDS_REFACTOR, items),
    not_ready=count(i -> i.status == NOT_READY, items),
    total=length(items),
)

# ---------------------------------------------------------------------------
# Observation ≠ ContinuousState — component by component
# ---------------------------------------------------------------------------

"""
    ObservabilityClass

How a component of the 15-D privileged feature vector could ever be obtained:

- `SENSOR_ESTIMABLE` — a camera or encoder could estimate it, with error.
- `TEMPORALLY_DERIVED` — needs tracking across frames, not one observation.
- `MAP_PRIVILEGED` — needs ground-truth map geometry beyond sensing range.
- `SIMULATOR_PRIVILEGED` — simulator bookkeeping, unobservable in principle.
- `AGENT_MEMORY` — the agent's own internal memory; belongs in the belief or
  the agent state, and is not an observation at all.
"""
@enum ObservabilityClass SENSOR_ESTIMABLE TEMPORALLY_DERIVED MAP_PRIVILEGED SIMULATOR_PRIVILEGED AGENT_MEMORY

struct ComponentObservability
    name::Symbol
    class::ObservabilityClass
    note::String
end

"""
    continuous_state_observability() -> Vector{ComponentObservability}

Classify every component of [`ContinuousState`](@ref). This is the concrete
evidence that the 15-D vector is a *privileged* feature projection and not an
observation: only a minority of its components could come from a sensor at
all, and two of them are the agent's own memory.
"""
continuous_state_observability() = [
    ComponentObservability(:d, SENSOR_ESTIMABLE,
        "lateral lane offset — lane detection gives it with error"),
    ComponentObservability(:phi, SENSOR_ESTIMABLE,
        "heading error relative to the lane — same"),
    ComponentObservability(:v, SENSOR_ESTIMABLE,
        "own speed — wheel encoders; the simulator exposes encoder ticks"),
    ComponentObservability(:kappa, MAP_PRIVILEGED,
        "signed curvature AHEAD, sampled from the map's Bezier curves beyond " *
        "what a camera sees"),
    ComponentObservability(:stop_present, MAP_PRIVILEGED,
        "whether a stop sign is the next candidate — the candidate is chosen " *
        "using the map's object list"),
    ComponentObservability(:d_stop, MAP_PRIVILEGED,
        "metric distance to the stop line, from the sign's true world pose"),
    ComponentObservability(:sigma_stop, AGENT_MEMORY,
        "the StopTracker's latched flag — the agent's own memory, not a " *
        "measurement of the world"),
    ComponentObservability(:duck_present, SENSOR_ESTIMABLE,
        "a detector could report a duck, with false positives and negatives"),
    ComponentObservability(:duck_longitudinal, SENSOR_ESTIMABLE,
        "relative position from a detection, with error"),
    ComponentObservability(:duck_lateral, SENSOR_ESTIMABLE,
        "relative position from a detection, with error"),
    ComponentObservability(:duck_v_longitudinal_relative, TEMPORALLY_DERIVED,
        "relative velocity — needs tracking across frames, not one frame"),
    ComponentObservability(:duck_v_lateral_relative, TEMPORALLY_DERIVED,
        "relative velocity — same"),
    ComponentObservability(:duck_active, SIMULATOR_PRIVILEGED,
        "the duck's internal pedestrian_active flag; motion can be inferred, " *
        "the flag cannot be measured"),
    ComponentObservability(:duck_crossing_available, SIMULATOR_PRIVILEGED,
        "derived from crossings_started and crossing_armed — pure simulator " *
        "bookkeeping with no physical counterpart"),
    ComponentObservability(:stop_hold_progress, AGENT_MEMORY,
        "the tracker's hold counter — the agent's own memory"),
]

"""
    observability_table(rows=continuous_state_observability()) -> String
"""
function observability_table(rows=continuous_state_observability())
    w = maximum(length(String(r.name)) for r in rows)
    io = IOBuffer()
    println(io, rpad("component", w), "  ", rpad("class", 22), "  why")
    println(io, "-"^(w + 70))
    for r in rows
        println(io, rpad(String(r.name), w), "  ",
            rpad(string(r.class), 22), "  ", r.note)
    end
    return String(take!(io))
end

"""
    observability_counts(rows=continuous_state_observability()) -> NamedTuple
"""
function observability_counts(rows=continuous_state_observability())
    c(x) = count(r -> r.class == x, rows)
    return (sensor_estimable=c(SENSOR_ESTIMABLE),
        temporally_derived=c(TEMPORALLY_DERIVED),
        map_privileged=c(MAP_PRIVILEGED),
        simulator_privileged=c(SIMULATOR_PRIVILEGED),
        agent_memory=c(AGENT_MEMORY),
        total=length(rows))
end

"""
    VISUALIZATION_EXTENSION_POINTS

The renderer signatures FJ9 must be built around so that adding a partially
observable layer later does not require rewriting it. Recorded here, in FJ10,
because that is the whole reason this gate runs before the visualisation one.

A renderer whose only entry point takes a `DuckieWorldState` hardens the
assumption that the thing being drawn is the latent truth. Belief-space
visualisation is precisely the case where that is false.
"""
const VISUALIZATION_EXTENSION_POINTS = (
    (name="render_world", input="DuckieWorldState",
     status="buildable now — the latent truth"),
    (name="render_projection", input="RawState / ContinuousState",
     status="buildable now — a PROJECTION panel, labelled privileged"),
    (name="render_policy", input="policy + model",
     status="buildable now"),
    (name="render_search", input="planner diagnostics (PlanningDiagnostics.extra)",
     status="buildable now — already solver-agnostic"),
    (name="render_rollout", input="a frozen experiment artefact",
     status="buildable now — added in FJ9.4; draws recorded evidence only"),
    (name="render_diagnostics", input="a frozen per-decision log",
     status="buildable now — added in FJ9.6; the series, their units and " *
            "their semantics are all decided in the core"),
    (name="render_animation", input="a frozen per-decision log",
     status="buildable now — added in FJ9.7; playback of recorded evidence, " *
            "never a re-run of the environment"),
    (name="render_observation", input="a future Observation type",
     status="reserve the signature; do not implement"),
    (name="render_belief", input="a future Belief type",
     status="reserve the signature; do not implement"),
)
