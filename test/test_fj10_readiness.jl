# FJ10 — POMDP readiness audit.
#
# An audit gate, so the tests verify the AUDIT'S CLAIMS rather than exercising
# new functionality. Two kinds of check:
#
#   * every READY row is confirmed against the live package, so the audit
#     cannot claim readiness that does not exist;
#   * every NOT_READY row is confirmed absent, so the day someone adds an
#     observation model or a belief updater these tests fail and the audit has
#     to be updated with it. The audit stays true by construction rather than
#     by remembering to edit a document.

using DuckietownDecisionModels
using POMDPs
using Test
using Random

const FJ10_QCFG = joinpath(pkgdir(DuckietownDecisionModels), "..", "duckduck",
    "policies", "q_learning", "training_config.yaml")
const FJ10_CCFG = joinpath(pkgdir(DuckietownDecisionModels), "..", "duckduck",
    "policies", "sac", "training_config.yaml")

# Three sets audit MDPs built from the reference checkout's frozen training
# configs; the other three audit the package itself. Contributors without the
# checkout still run the latter — the former skip by name, never silently.
const FJ10_HAVE_POLICIES = isfile(FJ10_QCFG)
if !FJ10_HAVE_POLICIES
    @info "FJ10: skipping 3 config-backed testsets (they need ../duckduck/policies/, the reference checkout's frozen training configs)"
end

FJ10_HAVE_POLICIES && @testset "FJ10 the audit is well formed" begin
    items = pomdp_readiness(DuckietownMDP(FJ10_QCFG; action_space=:discrete))
    @test length(items) == 16
    @test all(i -> !isempty(i.component), items)
    @test all(i -> !isempty(i.evidence), items)
    @test all(i -> !isempty(i.needed), items)
    @test all(i -> i.status in (READY, NEEDS_REFACTOR, NOT_READY), items)
    @test length(unique(i.component for i in items)) == length(items)

    c = readiness_counts(items)
    @test c.total == 16
    @test c.ready + c.needs_refactor + c.not_ready == c.total
    @test c.ready == 8
    @test c.needs_refactor == 1
    @test c.not_ready == 7
    @test occursin("component", readiness_table(items))

    # the audit must work on both action variants and on the wrapper
    @test length(pomdp_readiness(DuckietownMDP(FJ10_CCFG;
        action_space=:continuous))) == 16
    @test length(pomdp_readiness(InstrumentedMDP(
        DuckietownMDP(FJ10_QCFG; action_space=:discrete)))) == 16

    @info "FJ10 readiness\n" * readiness_table(items)
    @info "FJ10 readiness counts" c
end

FJ10_HAVE_POLICIES && @testset "FJ10 every READY claim is true of the live package" begin
    mdp = DuckietownMDP(FJ10_QCFG; action_space=:discrete)
    cmdp = DuckietownMDP(FJ10_CCFG; action_space=:continuous)
    s = rand(MersenneTwister(11), initialstate(mdp))

    # latent state shared by both variants
    @test statetype(mdp) === DuckieWorldState
    @test statetype(cmdp) === DuckieWorldState

    # transition takes a CALLER-SUPPLIED rng and returns (sp, r) only
    x = gen(mdp, s, FAST_STRAIGHT, MersenneTwister(1))
    @test Set(keys(x)) == Set((:sp, :r))
    @test x.sp isa DuckieWorldState
    @test hasmethod(POMDPs.gen,
        Tuple{typeof(mdp),DuckieWorldState,MacroAction,AbstractRNG})

    # reward is a function of the LATENT transition, not of a feature vector
    @test hasmethod(compute_reward, Tuple{RawState,EventFlags,RewardConfig})
    @test !hasmethod(compute_reward,
        Tuple{ContinuousState,EventFlags,RewardConfig})
    r = simulate_decision(mdp.transition, s, FAST_STRAIGHT, MersenneTwister(1))
    @test r.reward isa RewardBreakdown
    @test r.reward.total === x.r

    # terminal and truncation stay separate
    @test isterminal(mdp, s) isa Bool
    @test is_truncated(mdp, s) isa Bool
    @test hasmethod(termination_reason,
        Tuple{DuckieTransitionModel,DuckieWorldState})

    @test 0.0 < discount(mdp) <= 1.0
    @test rand(MersenneTwister(3), initialstate(mdp)) isa DuckieWorldState
    @test length(actions(mdp)) == 7
    @test actions(cmdp) isa DuckieActionSpace
    @test !applicable(length, actions(cmdp))
end

@testset "FJ10 every NOT_READY claim is genuinely absent" begin
    # this is the guard: if any of these starts existing, the audit is stale
    @test DuckietownMDP <: POMDPs.MDP
    @test !(DuckietownMDP <: POMDPs.POMDP)
    @test !(InstrumentedMDP <: POMDPs.POMDP)

    duckie_method(f) = any(m -> occursin("Duckie", string(m.sig)), methods(f))
    for name in (:observation, :obstype, :update, :initialobs, :initialize_belief)
        isdefined(POMDPs, name) || continue
        f = getfield(POMDPs, name)
        f isa Function || continue
        @test !duckie_method(f)
    end

    # no type in this package is an observation or a belief
    pkg = DuckietownDecisionModels
    names_ = names(pkg; all=false)
    @test !any(n -> occursin(r"Observation$|^Belief|Belief$", String(n)), names_)
    # ... and nothing subtypes POMDPs.Updater here
    if isdefined(POMDPs, :Updater)
        for n in names_
            v = try getfield(pkg, n) catch; nothing end
            v isa Type || continue
            @test !(v <: POMDPs.Updater)
        end
    end
end

@testset "FJ10 ContinuousState is a privileged projection, not an observation" begin
    rows = continuous_state_observability()
    @test length(rows) == 15
    @test length(rows) == fieldcount(ContinuousState)
    # every classified name is a real field, and every field is classified
    @test Set(r.name for r in rows) == Set(fieldnames(ContinuousState))
    @test all(r -> !isempty(r.note), rows)

    c = observability_counts(rows)
    @test c.total == 15
    @test c.sensor_estimable + c.temporally_derived + c.map_privileged +
          c.simulator_privileged + c.agent_memory == 15

    # the substantive claim: most of the vector is NOT sensor-estimable
    @test c.sensor_estimable == 6
    @test c.sensor_estimable < 15 - c.sensor_estimable

    # two components have no physical counterpart at all ...
    priv = [r.name for r in rows if r.class == SIMULATOR_PRIVILEGED]
    @test Set(priv) == Set([:duck_active, :duck_crossing_available])
    # ... and two are the agent's own memory, not a measurement
    mem = [r.name for r in rows if r.class == AGENT_MEMORY]
    @test Set(mem) == Set([:sigma_stop, :stop_hold_progress])
    # ... and three need ground-truth map geometry
    @test Set(r.name for r in rows if r.class == MAP_PRIVILEGED) ==
        Set([:kappa, :stop_present, :d_stop])

    @info "FJ10 observability of the 15-D vector\n" * observability_table(rows)
    @info "FJ10 observability counts" c
end

FJ10_HAVE_POLICIES && @testset "FJ10 the RNG contract holds and is not extended" begin
    mdp = DuckietownMDP(FJ10_QCFG; action_space=:discrete)
    s = rand(MersenneTwister(11), initialstate(mdp))

    # transition randomness is caller-supplied: there is no method that
    # invents an rng for the caller
    @test hasmethod(POMDPs.gen,
        Tuple{typeof(mdp),DuckieWorldState,MacroAction,AbstractRNG})
    @test !hasmethod(POMDPs.gen,
        Tuple{typeof(mdp),DuckieWorldState,MacroAction})
    @test hasmethod(simulate_decision,
        Tuple{DuckieTransitionModel,DuckieWorldState,MacroAction,AbstractRNG})

    # the only live RNG anywhere in the state is the legacy controller_rng,
    # which FJ8.0 proved frozen; it is audited as technical debt
    rng_fields = [f for f in fieldnames(DuckieWorldState)
                  if fieldtype(DuckieWorldState, f) <: AbstractRNG]
    @test rng_fields == [:controller_rng]
    snapshot = copy(s.controller_rng)
    for a in actions(mdp)
        gen(mdp, s, a, MersenneTwister(5))
    end
    @test rng_frozen([s], snapshot)

    items = pomdp_readiness(mdp)
    debt = only(filter(i -> occursin("controller_rng", i.component), items))
    @test debt.status == NEEDS_REFACTOR
    @test occursin("TECHNICAL DEBT", debt.needed)
    @test occursin("caller-supplied", debt.needed)
end

@testset "FJ10 the visualization extension points are recorded for FJ9" begin
    pts = VISUALIZATION_EXTENSION_POINTS
    # 9 since FJ9.7 added render_animation; the audit tracks the list, and
    # what must stay fixed is that the RESERVED points remain unbuilt.
    @test length(pts) == 9
    names_ = [p.name for p in pts]
    @test "render_world" in names_
    @test "render_observation" in names_
    @test "render_belief" in names_
    @test "render_rollout" in names_
    @test "render_diagnostics" in names_
    @test "render_animation" in names_

    # the two future ones must be reserved, not implemented — FJ10 runs before
    # FJ9 precisely so the renderer is not built around the latent state alone
    for p in pts
        if p.name in ("render_observation", "render_belief")
            @test occursin("reserve", p.status)
            @test occursin("do not implement", p.status)
        else
            @test occursin("buildable now", p.status)
        end
    end
    # The RESERVED points must not exist. The buildable ones may — and since
    # FJ9.0 landed, four of them do.
    #
    # This assertion previously required all six to be undefined and fired the
    # moment FJ9.0 declared the contract. That is the audit working as
    # designed: it is written to fail when the situation changes rather than
    # to go quietly stale. What changed is the situation, so the assertion now
    # tracks the STATUS rather than a snapshot of one day's package contents.
    for p in pts
        if occursin("do not implement", p.status)
            @test !isdefined(DuckietownDecisionModels, Symbol(p.name))
        end
    end
    @test isdefined(DuckietownDecisionModels, :render_world)
    @test isdefined(DuckietownDecisionModels, :render_projection)
    @test isdefined(DuckietownDecisionModels, :render_policy)
    @test isdefined(DuckietownDecisionModels, :render_search)
end
