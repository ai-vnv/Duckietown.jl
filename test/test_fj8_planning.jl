# FJ8.1 — the generic online-planner contract, tested with NO solver present.
#
# The point of this gate is that the package can host a solver without being
# shaped by one. So every test here uses either a hand-written stand-in policy
# or the model itself; nothing imports a planning library. If these pass, a
# real solver's only remaining job is to satisfy the POMDPs.jl interface.

using DuckietownDecisionModels
using POMDPs
using Test
using Random

const FJ81_QCFG = joinpath(pkgdir(DuckietownDecisionModels), "..", "duckduck",
    "policies", "q_learning", "training_config.yaml")
const FJ81_CCFG = joinpath(pkgdir(DuckietownDecisionModels), "..", "duckduck",
    "policies", "sac", "training_config.yaml")

"""A stand-in for an external POMDPs.jl planner: subtypes `POMDPs.Policy` and
answers the two-argument `action(planner, s)`, exactly as `solve` products do.
It also consumes the generative model, so the call counter has something to
count."""
struct StandInPlanner{M} <: POMDPs.Policy
    mdp::M
    rollouts::Int
end

function POMDPs.action(p::StandInPlanner, s::DuckieWorldState)
    rng = MersenneTwister(4)
    best_a, best_v = nothing, -Inf
    for a in POMDPs.actions(p.mdp)
        v = 0.0
        for _ in 1:p.rollouts
            v += POMDPs.gen(p.mdp, s, a, rng).r
        end
        if v > best_v
            best_v, best_a = v, a
        end
    end
    return best_a
end

"""Same, but reports solver-specific numbers through the open `extra` slot."""
struct DiagnosticPlanner{M} <: POMDPs.Policy
    mdp::M
    rollouts::Int
end

POMDPs.action(p::DiagnosticPlanner, s::DuckieWorldState) =
    POMDPs.action(StandInPlanner(p.mdp, p.rollouts), s)

function DuckietownDecisionModels.plan_action(p::DiagnosticPlanner, m,
    s::DuckieWorldState)
    before = model_calls(m)
    t0 = time_ns()
    a = POMDPs.action(p, s)
    dt = (time_ns() - t0) / 1e9
    used = model_calls(m) < 0 ? -1 : model_calls(m) - before
    n = length(POMDPs.actions(p.mdp))
    return a, PlanningDiagnostics(dt, used,
        (tree_nodes=n * p.rollouts, max_depth=1, iterations=p.rollouts))
end

@testset "FJ8.1 InstrumentedMDP is transparent" begin
    mdp = DuckietownMDP(FJ81_QCFG; action_space=:discrete)
    imdp = InstrumentedMDP(mdp)

    # same type parameters, so a solver dispatching on the action type is
    # unaffected by the wrapper
    @test imdp isa MDP{DuckieWorldState,MacroAction}
    @test imdp isa AnyMDPLike
    @test imdp isa MDPLike{MacroAction}

    # every interface function agrees
    @test discount(imdp) === discount(mdp)
    @test actions(imdp) === actions(mdp)
    @test length(actions(imdp)) == 7
    s = rand(MersenneTwister(11), initialstate(imdp))
    s_ref = rand(MersenneTwister(11), initialstate(mdp))
    @test worlds_identical(s, s_ref)
    @test actions(imdp, s) === actions(mdp, s)
    @test isterminal(imdp, s) === isterminal(mdp, s)
    @test is_truncated(imdp, s) === is_truncated(mdp, s)
    for a in actions(mdp)
        @test actionindex(imdp, a) == actionindex(mdp, a)
    end

    # fields forward, so code written against DuckietownMDP works unchanged
    @test imdp.transition === mdp.transition
    @test imdp.config === mdp.config
    @test imdp.map === mdp.map
    @test imdp.discount === mdp.discount
    @test :transition in propertynames(imdp)

    # BITWISE transparency of the transition: this is the property that makes
    # the wrapper a measuring device and not a translation layer
    for a in actions(mdp), seed in (1, 2, 7)
        x = gen(imdp, s, a, MersenneTwister(seed))
        y = gen(mdp, s, a, MersenneTwister(seed))
        @test worlds_identical(x.sp, y.sp)
        @test x.r === y.r
    end
    # and it does not mutate the state it is given, exactly like the model
    snapshot = branch(s)
    gen(imdp, s, FAST_STRAIGHT, MersenneTwister(1))
    @test worlds_identical(s, snapshot)
end

@testset "FJ8.1 the call counter counts what a planner consumes" begin
    mdp = DuckietownMDP(FJ81_QCFG; action_space=:discrete)
    imdp = InstrumentedMDP(mdp)
    @test model_calls(imdp) == 0
    @test model_calls(mdp) == -1              # not measured, not zero
    @test reset_model_calls!(mdp) == -1

    s = rand(MersenneTwister(11), initialstate(imdp))
    for _ in 1:5
        gen(imdp, s, FAST_STRAIGHT, MersenneTwister(1))
    end
    @test model_calls(imdp) == 5
    @test reset_model_calls!(imdp) == 5
    @test model_calls(imdp) == 0

    # the environment's own step is NOT charged as planning cost
    simulate_decision(imdp.transition, s, FAST_STRAIGHT, MersenneTwister(1))
    @test model_calls(imdp) == 0

    # a planner's consumption is counted exactly
    planner = StandInPlanner(imdp, 3)
    a = action(planner, s)
    @test a in actions(mdp)
    @test model_calls(imdp) == 7 * 3
end

@testset "FJ8.1 any POMDPs.Policy plugs in without a wrapper" begin
    mdp = DuckietownMDP(FJ81_QCFG; action_space=:discrete)
    imdp = InstrumentedMDP(mdp)
    s = rand(MersenneTwister(11), initialstate(imdp))
    planner = StandInPlanner(imdp, 2)

    # the evaluator's three-argument convention resolves to the solver's own
    # two-argument method — this one bridge is why no adapter is needed
    @test policy_action(planner, imdp, s) == action(planner, s)
    @test policy_action(planner, mdp, s) == action(planner, s)

    # ... and it does NOT come from adding a three-argument method to
    # POMDPs.action. Loading this package must not change how a generic
    # POMDPs.jl policy behaves for anyone else, so the adaptation lives on a
    # function this package owns. This guard fails if that ever regresses.
    @test !applicable(POMDPs.action, planner, imdp, s)
    @test !applicable(POMDPs.action, planner, mdp, s)
    @test applicable(POMDPs.action, planner, s)
    @test applicable(policy_action, planner, mdp, s)

    # this package's own policies keep the three-argument form they define
    qpath = joinpath(pkgdir(DuckietownDecisionModels), "..", "duckduck",
        "policies", "q_learning", "policy.npy")
    if isfile(qpath)
        qpol = QTablePolicy(qpath; solver=:q_learning)
        @test qpol isa DuckietownDecisionModels.AbstractPolicy
        @test !(qpol isa POMDPs.Policy)
        @test policy_action(qpol, imdp, s) == action(qpol, imdp, s)
    end

    # and it goes straight through the shared FJ7.6 evaluator
    ms = evaluate_policy(imdp, planner; seeds=1:2, max_steps=6)
    @test length(ms) == 2
    @test all(m -> 1 <= m.decisions <= 6, ms)
    @test all(m -> isfinite(m.ret), ms)
end

@testset "FJ8.1 diagnostics are generic, and open for solver specifics" begin
    mdp = DuckietownMDP(FJ81_QCFG; action_space=:discrete)
    imdp = InstrumentedMDP(mdp)
    s = rand(MersenneTwister(11), initialstate(imdp))

    # default path: works for a solver that reports nothing at all
    reset_model_calls!(imdp)
    a, d = plan_action(StandInPlanner(imdp, 2), imdp, s)
    @test a in actions(mdp)
    @test d isa PlanningDiagnostics
    @test d.planning_time > 0
    @test d.model_calls == 14
    @test d.extra === NamedTuple()
    @test occursin("calls", sprint(show, d))

    # uninstrumented model: reported as unmeasured, never as free
    _, d2 = plan_action(StandInPlanner(InstrumentedMDP(mdp), 2), mdp, s)
    @test d2.model_calls == -1
    @test occursin("n/a", sprint(show, d2))

    # a solver may fill `extra` with anything; the core never inspects it
    reset_model_calls!(imdp)
    _, d3 = plan_action(DiagnosticPlanner(imdp, 4), imdp, s)
    @test d3.extra.tree_nodes == 28
    @test d3.extra.max_depth == 1
    @test d3.extra.iterations == 4
    @test d3.model_calls == 28

    # a learned policy reports no `extra` and that is not a defect
    qpath = joinpath(pkgdir(DuckietownDecisionModels), "..", "duckduck",
        "policies", "q_learning", "policy.npy")
    if isfile(qpath)
        qpol = QTablePolicy(qpath; solver=:q_learning)
        aq, dq = plan_action(qpol, imdp, s)
        @test aq in actions(mdp)
        @test dq.extra === NamedTuple()
        @test dq.model_calls == 0          # it never consults the model
    end
end

@testset "FJ8.1 planning cost aggregates without assuming a tree" begin
    mdp = DuckietownMDP(FJ81_QCFG; action_space=:discrete)
    imdp = InstrumentedMDP(mdp)

    res = evaluate_planner(imdp, DiagnosticPlanner(imdp, 2); seeds=1:2,
        max_steps=5)
    @test length(res.episodes) == 2
    c = res.cost
    @test c isa PlannerCost
    @test c.decisions == sum(m -> m.decisions, res.episodes)
    @test c.total_time > 0
    @test c.latency_mean ≈ c.total_time / c.decisions
    @test c.latency_p50 <= c.latency_p95 <= c.latency_max
    @test c.model_calls_total == 14 * c.decisions
    @test c.model_calls_per_action ≈ 14.0
    @test c.extra[:tree_nodes] ≈ 14.0
    @test c.extra[:iterations] ≈ 2.0

    # a planner reporting nothing extra still aggregates
    plain = evaluate_planner(imdp, StandInPlanner(imdp, 2); seeds=1:2,
        max_steps=5)
    @test isempty(plain.cost.extra)
    @test plain.cost.model_calls_per_action ≈ 14.0

    # an unmeasured model reports -1 rather than a misleading zero
    unmeasured = summarize_planning([PlanningDiagnostics(0.01, -1, NamedTuple())])
    @test unmeasured.model_calls_total == -1
    @test isnan(unmeasured.model_calls_per_action)
    @test summarize_planning(PlanningDiagnostics[]).decisions == 0

    # recording must not change the episode itself
    a1 = evaluate_policy(imdp, StandInPlanner(imdp, 2); seeds=1:2, max_steps=5)
    rec = PlanningDiagnostics[]
    a2 = evaluate_policy(imdp, StandInPlanner(imdp, 2); seeds=1:2, max_steps=5,
        record=rec)
    @test [m.ret for m in a1] == [m.ret for m in a2]
    @test [m.decisions for m in a1] == [m.decisions for m in a2]
    @test length(rec) == sum(m -> m.decisions, a1)
end

@testset "FJ8.1 model capabilities are measured, not declared" begin
    mdp = DuckietownMDP(FJ81_QCFG; action_space=:discrete)
    cmdp = DuckietownMDP(FJ81_CCFG; action_space=:continuous)

    d = model_capabilities(mdp)
    c = model_capabilities(cmdp)

    @test d.generative_transition && c.generative_transition
    @test d.discrete_actions && !d.continuous_actions
    @test c.continuous_actions && !c.discrete_actions
    @test d.enumerable_actions            # a Vector of 7 macro actions
    @test !c.enumerable_actions           # a box: a solver must widen, not enumerate
    @test d.continuous_state && c.continuous_state
    @test d.terminal_states && c.terminal_states
    @test d.truncation_separate_from_termination
    @test d.discount && c.discount
    @test d.initial_state_sampler && c.initial_state_sampler
    # honest negatives: this is an MDP, and no explicit T(.|s,a) is offered
    @test !d.explicit_transition_distribution
    @test !d.observation_model
    @test !d.belief_updater

    # the wrapper must not change any capability
    @test model_capabilities(InstrumentedMDP(mdp)) == d

    # The model is CONDITIONALLY stochastic: the pedestrian trigger only draws
    # when a duck is armed, ahead and inside the trigger window. Reported as
    # three separate facts so a state-widening planner is not misled into
    # widening states whose successor is a function of (s, a) alone.
    @test 0.0 <= d.stochastic_state_fraction <= 1.0
    @test 0.0 <= c.stochastic_state_fraction <= 1.0
    @test d.consumes_rng == (d.stochastic_state_fraction > 0)
    @test !d.stochastic_outcomes || d.consumes_rng   # outcomes imply draws

    # Under a constant action the vehicle leaves the road within a few
    # decisions and never reaches a trigger window, so the default probe sees
    # no randomness at all. That is a property of the PROBE, not of the model,
    # and is recorded as such.
    @test d.stochastic_state_fraction == 0.0
    @test !d.consumes_rng

    # Driving competently reaches the states that matter.
    qpath = joinpath(pkgdir(DuckietownDecisionModels), "..", "duckduck",
        "policies", "q_learning", "policy.npy")
    if isfile(qpath)
        qpol = QTablePolicy(qpath; solver=:q_learning)
        dq = model_capabilities(mdp; policy=qpol, probe_states=200,
            state_seed=1, seeds=1:12)
        @test dq.consumes_rng                    # the trigger IS reachable
        @test dq.stochastic_state_fraction > 0
        @test dq.stochastic_state_fraction < 0.05
        # ... but with p_cross = 1.0 in the shipped configs, `rand() < 1.0` is
        # always true, so the draw never changes the outcome. The model is
        # therefore deterministic in OUTCOME on these baselines.
        @test !dq.stochastic_outcomes
        @test mdp.config.duck_controller.p_cross == 1.0
        @test mdp.config.duck_controller.max_crossings_per_episode == 1
        @info "FJ8.1 stochasticity along a competent trajectory" consumes_rng =
            dq.consumes_rng  fraction_of_states =
            round(dq.stochastic_state_fraction; digits=4)  outcomes_differ =
            dq.stochastic_outcomes  p_cross = mdp.config.duck_controller.p_cross max_crossings =
            mdp.config.duck_controller.max_crossings_per_episode
    end

    @info "FJ8.1 capability report (discrete, constant-action probe)\n" *
        capability_report(mdp)
    @info "FJ8.1 capability report (continuous, constant-action probe)\n" *
        capability_report(cmdp)
end

@testset "FJ8.1 the core names no solver" begin
    # the formulation must not acquire solver-specific vocabulary. If a future
    # change adds MCTS/DPW concepts to src/, this fails and the code belongs in
    # an extension instead.
    srcdir = joinpath(pkgdir(DuckietownDecisionModels), "src")
    offenders = String[]
    for (root, _, files) in walkdir(srcdir), f in files
        endswith(f, ".jl") || continue
        text = read(joinpath(root, f), String)
        for token in ("MCTSSolver", "DPWSolver", "MCTSPlanner", "using MCTS",
            "import MCTS")
            occursin(token, text) &&
                push!(offenders, relpath(joinpath(root, f), srcdir) * ": " * token)
        end
    end
    @test isempty(offenders)
    isempty(offenders) || @info "solver vocabulary leaked into src/" offenders
end
