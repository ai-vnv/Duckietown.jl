# FJ4: the POMDPs.jl model interface.
#
# `DuckietownMDP{A} <: MDP{DuckieWorldState, A}` makes the FJ3 generative core
# usable like any POMDPs.jl benchmark problem. Everything here is a THIN
# adapter over the canonical native transition — `POMDPs.gen` returns
# `(sp = r.sp, r = r.reward.total)` from `simulate_decision`, and
# `POMDPs.isterminal` calls the same `termination_reason` the chain uses.
# There is no second copy of any dynamics, reward or termination logic.
#
# Two action variants stay EXPLICIT (never merged):
#   DuckietownMDP{MacroAction}  — the 7 macro actions (Q-learning/SARSA/MCTS)
#   DuckietownMDP{DuckieAction} — continuous [v_cmd, omega_cmd] (SAC/TD3/DPW)
# Both share one problem definition: same world state, dynamics, reward and
# initial-state distribution; only the action representation differs.
#
# API pinned against the installed POMDPs.jl v1.0.0:
#   gen(m, s, a, rng) -> NamedTuple, initialstate(m) -> sampleable,
#   isterminal(m, s), discount(m), actions(m).

import POMDPs
using POMDPs: MDP

"""
    DuckieActionSpace

Continuous action box `[0, v_fast] x [-w0, w0]` (m/s, rad/s), matching the
reference `ContinuousDuckieMDPEnv.action_space`. `rand(rng, space)` samples
uniformly — this is what a progressive-widening planner draws from.
"""
struct DuckieActionSpace
    v_min::Float64
    v_max::Float64
    omega_min::Float64
    omega_max::Float64
end

DuckieActionSpace(cfg::ActionConfig) =
    DuckieActionSpace(0.0, cfg.v_fast, -cfg.w0, cfg.w0)

Base.rand(rng::AbstractRNG, s::DuckieActionSpace) = DuckieAction(
    s.v_min + (s.v_max - s.v_min) * rand(rng),
    s.omega_min + (s.omega_max - s.omega_min) * rand(rng))

Base.in(a::DuckieAction, s::DuckieActionSpace) =
    s.v_min <= a.v <= s.v_max && s.omega_min <= a.omega <= s.omega_max

Base.eltype(::Type{DuckieActionSpace}) = DuckieAction

"""
    DuckietownMDP{A} <: MDP{DuckieWorldState, A}

The Duckietown driving task as a `POMDPs.jl` MDP over the canonical
branchable world state.

- `A = MacroAction`: discrete 7-action problem (Q-learning/SARSA/MCTS).
- `A = DuckieAction`: continuous `[v_cmd, omega_cmd]` problem (SAC/TD3/DPW).

Construct from an experiment YAML (the reference config is the single source
of every parameter):

```julia
mdp  = DuckietownMDP("../duckduck/policies/q_learning/training_config.yaml")
mdpc = DuckietownMDP("../duckduck/policies/sac/training_config.yaml";
                     action_space = :continuous)

s0 = rand(rng, initialstate(mdp))
x  = gen(mdp, s0, FAST_STRAIGHT, rng)   # x.sp, x.r
```

`simulate_decision(mdp.transition, s, a, rng)` remains available for the full
[`TransitionResult`](@ref) (reward breakdown, events, reason, projections) —
`gen` deliberately exposes only `(sp, r)`.
"""
struct DuckietownMDP{A} <: MDP{DuckieWorldState,A}
    config::DuckietownConfig
    transition::DuckieTransitionModel
    map::RoadMap
    action_space::Union{Vector{MacroAction},DuckieActionSpace}
    discount::Float64
end

const ALL_MACRO_ACTIONS = [FAST_LEFT, FAST_STRAIGHT, FAST_RIGHT,
    SLOW_LEFT, SLOW_STRAIGHT, SLOW_RIGHT, BRAKE]

"""
    DuckietownMDP(config; action_space=:discrete, map=initial_map(config),
                  discount=config.solver.gamma)

Build the MDP from a loaded [`DuckietownConfig`](@ref). The discrete variant
restricts the action set to the solver's `allowed_actions` when the config
declares them (tabular experiments use `0:6`, i.e. all seven).
"""
function DuckietownMDP(config::DuckietownConfig;
    action_space::Symbol=:discrete,
    map::RoadMap=initial_map(config),
    discount::Real=config.solver.gamma)
    transition = DuckieTransitionModel(config)
    if action_space === :discrete
        allowed = hasproperty(config.solver, :allowed_actions) ?
            config.solver.allowed_actions : collect(0:6)
        acts = [ALL_MACRO_ACTIONS[i + 1] for i in allowed]
        return DuckietownMDP{MacroAction}(config, transition, map, acts,
            Float64(discount))
    elseif action_space === :continuous
        return DuckietownMDP{DuckieAction}(config, transition, map,
            DuckieActionSpace(config.actions), Float64(discount))
    end
    throw(ArgumentError("action_space must be :discrete or :continuous"))
end

DuckietownMDP(config_path::AbstractString; kwargs...) =
    DuckietownMDP(load_config(config_path); kwargs...)

POMDPs.discount(m::DuckietownMDP) = m.discount

POMDPs.actions(m::DuckietownMDP) = m.action_space
POMDPs.actions(m::DuckietownMDP, ::DuckieWorldState) = m.action_space

POMDPs.actionindex(m::DuckietownMDP{MacroAction}, a::MacroAction) =
    findfirst(==(a), m.action_space)

"""
    POMDPs.gen(mdp, s, a, rng) -> (sp = ..., r = ...)

Thin adapter over [`simulate_decision`](@ref): one macro-decision
(`frame_skip` physics ticks under the locked transition order), with the
stochastic pedestrian trigger drawn from `rng`. `s` is never mutated, so the
same state may be branched with different actions.
"""
POMDPs.gen(m::DuckietownMDP{MacroAction}, s::DuckieWorldState, a::MacroAction,
    rng::AbstractRNG) = _gen_result(simulate_decision(m.transition, s, a, rng))

POMDPs.gen(m::DuckietownMDP{DuckieAction}, s::DuckieWorldState,
    a::DuckieAction, rng::AbstractRNG) =
    _gen_result(simulate_decision(m.transition, s, a, rng))

_gen_result(r::TransitionResult) = (sp=r.sp, r=r.reward.total)

"""
    POMDPs.isterminal(mdp, s) -> Bool

`true` only for a GENUINE terminal (`duck_collision`, `other_collision`,
`offroad`, `goal`) — the cases that break TD bootstrapping. A `timeout` is
truncation imposed by the experiment horizon, not an absorbing physical
state, so it is deliberately NOT terminal here; use [`is_truncated`](@ref)
(or [`termination_reason`](@ref)) for the horizon, exactly as the reference
wrapper separates `terminated` from `truncated`.
"""
POMDPs.isterminal(m::DuckietownMDP, s::DuckieWorldState) =
    is_terminated(termination_reason(m.transition, s))

"""
    is_truncated(mdp, s) -> Bool

Horizon truncation (`step_count >= max_steps`) for the same state.
"""
is_truncated(m::DuckietownMDP, s::DuckieWorldState) =
    is_truncated(termination_reason(m.transition, s))

"""
    DuckieInitialStateDistribution

The initial-state distribution `rho_0` (`DuckieMDPEnv.reset`). Sampling is
implicit: `rand(rng, d)` runs the reference spawn loop — up to
`spawn_attempts` curriculum attempts, each sampling a pose on the start tile
(`Simulator.reset`), rebuilding the world, and testing the wrapper's
acceptance predicate (`|d|`, `|phi|`, position bounds, route direction).
The last candidate is returned if none is accepted, matching the reference
`RuntimeError` case being unreachable in the shipped configs; pass
`strict = true` to raise instead.
"""
struct DuckieInitialStateDistribution
    mdp::DuckietownMDP
    strict::Bool
end

POMDPs.initialstate(m::DuckietownMDP) =
    DuckieInitialStateDistribution(m, false)

Base.eltype(::Type{DuckieInitialStateDistribution}) = DuckieWorldState

"""
    spawn_accepted(mdp, world, raw) -> Bool

`DuckieMDPEnv._spawn_is_accepted`: the curriculum limits on `|d|` and `|phi|`,
the optional x-z spawn rectangle, and the optional route-direction alignment.
"""
function spawn_accepted(m::DuckietownMDP, world::DuckieWorldState,
    raw::RawState)
    env = m.config.environment
    d_ok = env.spawn_max_abs_d === nothing || abs(raw.d) <= env.spawn_max_abs_d
    phi_ok = env.spawn_max_abs_phi === nothing ||
        abs(raw.phi) <= env.spawn_max_abs_phi
    position_ok = env.spawn_position_bounds_xz === nothing ||
        position_in_bounds_xz(world.ego.pos, env.spawn_position_bounds_xz)
    route_ok = true
    if env.spawn_route_direction !== nothing
        center = env.spawn_route_center === nothing ?
            (size(m.map.grid, 2) * m.map.tile_size / 2.0,
                size(m.map.grid, 1) * m.map.tile_size / 2.0) :
            env.spawn_route_center
        score = route_circulation_score(world.ego.pos, world.ego.angle, center)
        route_ok = env.spawn_route_direction === :clockwise ?
            score >= env.spawn_min_route_alignment :
            score <= -env.spawn_min_route_alignment
    end
    return d_ok && phi_ok && position_ok && route_ok
end

"""
    build_world(mdp, pos, angle) -> DuckieWorldState

A fresh world at the given ego pose: ego at rest with an empty command
window, the injected duckie in its reset condition, the map's stop signs, and
cleared stop/lane memory (`DuckieMDPEnv.reset` sets `_mdp_sigma_stop = false`
and `_mdp_last_lane_position = (1.0, 1.0)`, and `stop_tracker.reset()`).
"""
function build_world(m::DuckietownMDP, pos::NTuple{3,Float64}, angle::Float64)
    dcfg = m.transition.duck_cfg
    ducks = (dcfg.require_duck || dcfg.inject_if_missing) ?
        [initial_duckie(m.map, dcfg)] : DuckieState[]
    signs = [StopSignState(o.pos, o.angle) for o in m.map.static_objects
        if o.kind === :sign_stop]
    ego = initial_ego(pos, angle, size(m.map.grid, 1), m.map.tile_size)
    return DuckieWorldState(ego, ducks, signs, m.map,
        StopMemory(false, 0, nothing, nothing), (1.0, 1.0),
        zeros(Int, length(ducks)), trues(length(ducks)), MersenneTwister(0))
end

function Base.rand(rng::AbstractRNG, d::DuckieInitialStateDistribution)
    m = d.mdp
    env = m.config.environment
    tiles = drivable_tiles(m.map)
    world = nothing
    for _ in 1:max(1, env.spawn_attempts)
        i, j = if env.user_tile_start !== nothing
            env.user_tile_start
        else
            tiles[_uniform_tile_index(rng, length(tiles)) + 1]
        end
        pos, angle = sample_spawn_pose(m.map, rng, i, j,
            env.accept_start_angle_deg)
        world = build_world(m, (pos[1], pos[2], pos[3]), angle)
        raw, fallback = get_raw_state(world, m.transition.state_cfg;
            sigma_stop=false)
        world.lane_fallback = fallback
        if spawn_accepted(m, world, raw)
            # reset() records the first stop candidate as the previous-decision
            # memory (`_last_state` / `_last_stop_id`)
            _, stop_id = next_stop_candidate(world, m.transition.state_cfg)
            world.stop_memory = StopMemory(false, 0, stop_id, raw.d_stop)
            return world
        end
    end
    d.strict && throw(ErrorException(
        "could not sample a curriculum spawn in $(env.spawn_attempts) attempts"))
    raw, _ = get_raw_state(world, m.transition.state_cfg; sigma_stop=false)
    _, stop_id = next_stop_candidate(world, m.transition.state_cfg)
    world.stop_memory = StopMemory(false, 0, stop_id, raw.d_stop)
    return world
end

Base.rand(d::DuckieInitialStateDistribution) = rand(Random.GLOBAL_RNG, d)
