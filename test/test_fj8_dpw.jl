# FJ8.3 — continuous-action planning with double progressive widening.
#
# Four things, in this order, because each one makes the next meaningful:
#
#   8.3a  the action proposal really is continuous and reproducible
#   8.3b  action widening is really active, and sublinear in visits
#   8.3c  state widening really is unnecessary on this baseline — measured on
#         the transition, not inferred from a config value
#   8.3d  the planner integrates like any other POMDPs.jl policy
#
# Baseline configuration is deliberately conservative: action PW on, state PW
# off, uniform proposals in the action box, no learned guidance, no
# hyper-parameter sweep. FJ8.0 measured a continuous `gen` at ~109 us and
# 237 KiB, so the first question is correctness and scaling, not tuning.

using DuckietownDecisionModels
using POMDPs
using Test
using Random

const FJ83_OK = try
    @eval using MCTS
    true
catch err
    @info "FJ8.3: skipped (MCTS.jl not available)" err
    false
end

const FJ83_CCFG = joinpath(pkgdir(DuckietownDecisionModels), "..", "duckduck",
    "policies", "sac", "training_config.yaml")
const FJ83_QCFG = joinpath(pkgdir(DuckietownDecisionModels), "..", "duckduck",
    "policies", "q_learning", "training_config.yaml")

fj83_cmdp() = DuckietownMDP(FJ83_CCFG; action_space=:continuous)
fj83_state(m, seed=11) = rand(MersenneTwister(seed), initialstate(m))

"""The FJ8.3 baseline solver. Action widening ON, state widening OFF — the
latter justified by the measurement in the 8.3c test set, not by convention.
Uniform proposals (MCTS.jl's default `RandomActionGenerator`), no learned
guidance, no tree reuse."""
fj83_solver(; n=100, depth=10, seed=1, action_pw=true, state_pw=false) =
    DPWSolver(n_iterations=n, depth=depth, exploration_constant=5.0,
        enable_action_pw=action_pw, enable_state_pw=state_pw,
        k_action=4.0, alpha_action=0.5,
        rng=MersenneTwister(seed), keep_tree=false)

# ---------------------------------------------------------------------------
# 8.3a — the action proposal contract
# ---------------------------------------------------------------------------

@testset "FJ8.3a the action proposal is genuinely continuous" begin
    cmdp = fj83_cmdp()
    space = actions(cmdp)
    @test space isa DuckieActionSpace

    n = 10_000
    rng = MersenneTwister(2026)
    props = [rand(rng, space) for _ in 1:n]

    # every proposal inside the reference action box
    @test all(a -> a in space, props)
    @test all(a -> 0.0 <= a.v <= 0.41, props)
    @test all(a -> -1.5 <= a.omega <= 1.5, props)
    @test isapprox(space.v_max, 0.41; atol=1e-7)
    @test space.v_min == 0.0
    @test space.omega_min == -1.5 && space.omega_max == 1.5

    # genuinely continuous, not a hidden lattice
    uv = length(unique(a -> a.v, props))
    uw = length(unique(a -> a.omega, props))
    @test uv > n ÷ 2
    @test uw > n ÷ 2

    # and specifically NOT the seven macro actions in disguise
    table = build_action_table(cmdp.transition.action_cfg)
    lattice = Set((round(sp.v; digits=12), round(sp.omega; digits=12))
                  for sp in table)
    on_lattice = count(a -> (round(a.v; digits=12),
        round(a.omega; digits=12)) in lattice, props)
    @test on_lattice == 0
    @test length(lattice) <= 7

    # the box is actually covered, not clustered in one corner
    @test minimum(a -> a.v, props) < 0.01
    @test maximum(a -> a.v, props) > 0.40
    @test minimum(a -> a.omega, props) < -1.45
    @test maximum(a -> a.omega, props) > 1.45

    # reproducible per stream, different across streams
    seq(seed) = [rand(MersenneTwister(seed), space) for _ in 1:5]
    @test seq(4) == seq(4)
    @test seq(4) != seq(5)
    r1, r2 = MersenneTwister(9), MersenneTwister(9)
    @test [rand(r1, space) for _ in 1:50] == [rand(r2, space) for _ in 1:50]

    @info "FJ8.3a proposal distribution" proposals = n unique_v = uv unique_omega =
        uw on_macro_lattice = on_lattice
end

FJ83_OK && @testset "FJ8.3a the actions the planner really proposed" begin
    # Read the proposals out of a REAL search rather than calling the
    # generator artificially: `tree.a_labels` is exactly the set of actions
    # MCTS.jl's default `RandomActionGenerator` produced while widening, so
    # this is the contract the planner actually got.
    cmdp = fj83_cmdp()
    s = fj83_state(cmdp)
    planner = solve(fj83_solver(; n=300), cmdp)
    action(planner, s)
    labels = planner.tree.a_labels

    @test !isempty(labels)
    @test all(a -> a isa DuckieAction, labels)
    @test all(a -> a in actions(cmdp), labels)
    @test all(a -> 0.0 <= a.v <= 0.41 && -1.5 <= a.omega <= 1.5, labels)

    # continuous, not the macro-action lattice
    table = build_action_table(cmdp.transition.action_cfg)
    lattice = Set((round(sp.v; digits=12), round(sp.omega; digits=12))
                  for sp in table)
    @test count(a -> (round(a.v; digits=12),
        round(a.omega; digits=12)) in lattice, labels) == 0
    @test length(unique(a -> (a.v, a.omega), labels)) == length(labels)

    @info "FJ8.3a actions proposed during a real search" proposed = length(labels) distinct =
        length(unique(a -> (a.v, a.omega), labels))  v_range =
        (round(minimum(a -> a.v, labels); digits=3),
         round(maximum(a -> a.v, labels); digits=3))  omega_range =
        (round(minimum(a -> a.omega, labels); digits=3),
         round(maximum(a -> a.omega, labels); digits=3))
end

# ---------------------------------------------------------------------------
# 8.3b — progressive widening is really active
# ---------------------------------------------------------------------------

FJ83_OK && @testset "FJ8.3b action widening is active and sublinear" begin
    cmdp = fj83_cmdp()
    imdp = InstrumentedMDP(cmdp)
    s = fj83_state(imdp)

    budgets = (25, 50, 100, 250, 500)
    rows = NamedTuple[]
    for n in budgets
        reset_model_calls!(imdp)
        planner = solve(fj83_solver(; n=n), imdp)
        a, d = plan_action(planner, imdp, s)
        @test a isa DuckieAction
        push!(rows, (n=n, visits=d.extra.root_visits,
            root_children=d.extra.root_action_children,
            action_nodes=d.extra.action_nodes,
            state_nodes=d.extra.state_nodes,
            max_state_children=d.extra.max_state_children,
            gen_calls=d.model_calls, ms=1e3 * d.planning_time))
    end

    @info "FJ8.3b widening vs budget\n" * join([
        "  n=$(lpad(r.n,4))  visits=$(lpad(r.visits,4))" *
        "  root_action_children=$(lpad(r.root_children,3))" *
        "  action_nodes=$(lpad(r.action_nodes,5))" *
        "  state_nodes=$(lpad(r.state_nodes,5))" *
        "  max_state_children=$(r.max_state_children)" *
        "  gen=$(lpad(r.gen_calls,6))  $(round(r.ms; digits=1)) ms"
        for r in rows], "\n")

    # widening happens at all: more budget buys more root actions
    @test rows[end].root_children > rows[1].root_children
    # ... and it is monotone in budget
    @test issorted([r.root_children for r in rows])

    # SUBLINEAR in visits: the whole point of progressive widening is that the
    # action set grows slower than the visit count. Checked as a ratio that
    # must decrease, which needs no assumed constant.
    ratios = [r.root_children / r.visits for r in rows]
    @test ratios[end] < ratios[1]
    @test rows[end].root_children < rows[end].visits

    # consistent with |A(s)| ~ k * N^alpha for the configured k=4, alpha=0.5:
    # the realised count must not exceed the widening bound at any budget
    for r in rows
        @test r.root_children <= ceil(4.0 * r.visits^0.5) + 1
    end
    @info "FJ8.3b children/visits ratio" ratios = round.(ratios; digits=3)
end

FJ83_OK && @testset "FJ8.3b negative control: widening off changes the tree" begin
    # With a continuous action space and action PW disabled, DPW must fall back
    # to enumerating `actions(mdp, s)` — which a box cannot answer. Whatever
    # happens, it must NOT silently behave like the widening run.
    cmdp = fj83_cmdp()
    imdp = InstrumentedMDP(cmdp)
    s = fj83_state(imdp)

    on = plan_action(solve(fj83_solver(; n=100), imdp), imdp, s)[2]
    off_result = try
        (:ok, plan_action(solve(fj83_solver(; n=100, action_pw=false), imdp),
            imdp, s)[2])
    catch err
        (:error, err)
    end

    if off_result[1] === :error
        # the expected outcome: an unenumerable action space cannot be widened
        # off. Recorded as a capability fact, not swept under a try/catch.
        @test off_result[2] isa Exception
        @info "FJ8.3b action PW disabled on a continuous space" error =
            sprint(showerror, off_result[2])[1:min(160, end)]
    else
        d = off_result[2]
        @test d.extra.root_action_children != on.extra.root_action_children
        @info "FJ8.3b action PW disabled still ran" with_pw =
            on.extra.root_action_children  without_pw = d.extra.root_action_children
    end

    # The same control on the DISCRETE model, where disabling widening is at
    # least well-defined. Two regimes, and both are worth recording:
    dmdp = InstrumentedMDP(DuckietownMDP(FJ83_QCFG; action_space=:discrete))
    ds = fj83_state(dmdp)
    widen(sol) = plan_action(solve(sol, dmdp), dmdp, ds)[2].extra.root_action_children

    # (i) with the BASELINE coefficients the widening bound at N = 100 is
    #     k*N^alpha = 4*10 = 40, far above the seven available actions, so
    #     widening admits all of them and PW on/off coincide. Progressive
    #     widening is simply vacuous on a small discrete action set — recorded
    #     rather than tuned away.
    off_baseline = widen(fj83_solver(; n=100, action_pw=false))
    on_baseline = widen(fj83_solver(; n=100, action_pw=true))
    @test off_baseline == 7
    @test on_baseline == 7
    @test 4.0 * 100^0.5 > 7

    # (ii) with a bound deliberately BELOW the action count, widening bites and
    #      the tree really does differ. This is the control proper.
    tight = DPWSolver(n_iterations=100, depth=10, exploration_constant=5.0,
        enable_action_pw=true, enable_state_pw=false,
        k_action=1.0, alpha_action=0.3, rng=MersenneTwister(1),
        keep_tree=false)
    on_tight = widen(tight)
    @test 1.0 * 100^0.3 < 7
    @test on_tight < 7
    @test on_tight != off_baseline

    @info "FJ8.3b discrete control" widening_off = off_baseline baseline_on =
        on_baseline  baseline_bound = round(4.0 * 100^0.5; digits=1) tight_on =
        on_tight  tight_bound = round(1.0 * 100^0.3; digits=2)
end

# ---------------------------------------------------------------------------
# 8.3c — state widening is unnecessary, measured on the transition
# ---------------------------------------------------------------------------

"""States along a competently driven trajectory at which the transition
actually draws from the caller's RNG — i.e. the crossing-trigger region."""
function fj83_trigger_states(mdp, policy; max_steps=250, seed=1)
    s = rand(MersenneTwister(seed), initialstate(mdp))
    rng = MersenneTwister(99)
    hits = Tuple{DuckieWorldState,Any}[]
    for _ in 1:max_steps
        a = policy_action(policy, mdp, s)
        probe = MersenneTwister(1)
        POMDPs.gen(mdp, s, a, probe)
        probe == MersenneTwister(1) || push!(hits, (s, a))
        r = simulate_decision(mdp.transition, s, a, rng)
        s = r.sp
        (r.terminated || r.truncated) && break
    end
    return hits
end

@testset "FJ8.3c the baseline transition is a point mass where it matters" begin
    # This is the justification for `enable_state_pw = false`, and it is a
    # measurement of the TRANSITION, not a reading of a config value.
    dmdp = DuckietownMDP(FJ83_QCFG; action_space=:discrete)
    cmdp = fj83_cmdp()
    qpath = joinpath(pkgdir(DuckietownDecisionModels), "..", "duckduck",
        "policies", "q_learning", "policy.npy")

    if !isfile(qpath)
        @test_skip "tabular checkpoint unavailable"
    else
        qpol = QTablePolicy(qpath; solver=:q_learning)
        hits = Tuple{DuckieWorldState,Any}[]
        for seed in 1:4
            append!(hits, fj83_trigger_states(dmdp, qpol; seed=seed))
        end
        @test !isempty(hits)          # the trigger region is reachable at all

        seeds = 1:64
        worst_unique = 1
        for (s, a) in hits
            # discrete action at the state the policy actually took ...
            succ = [POMDPs.gen(dmdp, s, a, MersenneTwister(k)).sp for k in seeds]
            uniq = 1
            for x in succ[2:end]
                worlds_identical(succ[1], x) || (uniq += 1)
            end
            worst_unique = max(worst_unique, uniq)

            # ... and a continuous action at the same world state, since the
            # two models share the state space and the same duck trigger
            ca = DuckieAction(0.2, 0.3)
            csucc = [POMDPs.gen(cmdp, s, ca, MersenneTwister(k)).sp
                     for k in seeds]
            for x in csucc[2:end]
                @test worlds_identical(csucc[1], x)
            end
        end
        @test worst_unique == 1

        @info "FJ8.3c successor multiplicity in the trigger region" trigger_states =
            length(hits)  rng_seeds = length(seeds)  unique_successors =
            worst_unique  p_cross = dmdp.config.duck_controller.p_cross max_crossings =
            dmdp.config.duck_controller.max_crossings_per_episode
    end
end

FJ83_OK && @testset "FJ8.3c the tree shows no state-widening activity" begin
    # the structural consequence: with state PW off, every state-action node
    # has exactly one sampled successor
    cmdp = fj83_cmdp()
    imdp = InstrumentedMDP(cmdp)
    s = fj83_state(imdp)
    for n in (100, 250)
        _, d = plan_action(solve(fj83_solver(; n=n), imdp), imdp, s)
        @test d.extra.state_pw == false
        @test d.extra.max_state_children == 1
        @test d.extra.unique_transitions == d.extra.action_nodes
    end
end

# ---------------------------------------------------------------------------
# 8.3d — planner integration
# ---------------------------------------------------------------------------

FJ83_OK && @testset "FJ8.3d DPW integrates like any other POMDPs.jl policy" begin
    cmdp = fj83_cmdp()
    s = fj83_state(cmdp)

    planner = solve(fj83_solver(; n=120), cmdp)
    @test planner isa POMDPs.Policy
    a = action(planner, s)
    @test a isa DuckieAction
    @test a in actions(cmdp)
    @test 0.0 <= a.v <= 0.41
    @test -1.5 <= a.omega <= 1.5

    # the model is untouched by the search
    imdp = InstrumentedMDP(cmdp)
    s2 = fj83_state(imdp)
    snapshot = branch(s2)
    reset_model_calls!(imdp)
    a2 = action(solve(fj83_solver(; n=120), imdp), s2)
    @test a2 in actions(cmdp)
    @test isempty(world_differences(s2, snapshot))
    @test rng_frozen([s2], snapshot.controller_rng)
    @test model_calls(imdp) >= 120        # native gen only

    # reproducible under the same planner RNG
    b1 = action(solve(fj83_solver(; n=120, seed=17), cmdp), s)
    b2 = action(solve(fj83_solver(; n=120, seed=17), cmdp), s)
    @test b1 == b2
    @test b1 isa DuckieAction

    # and the shared evaluator takes it with no DPW-specific harness
    res = evaluate_planner(imdp, solve(fj83_solver(; n=40, depth=6), imdp);
        seeds=1:2, max_steps=6)
    @test length(res.episodes) == 2
    @test all(m -> 1 <= m.decisions <= 6, res.episodes)
    @test all(m -> isfinite(m.ret), res.episodes)
    c = res.cost
    @test c.model_calls_per_action >= 40
    @test haskey(c.extra, :root_action_children)
    @test c.extra[:max_state_children] == 1.0
    @info "FJ8.3d DPW through the shared evaluator" decisions = c.decisions latency_mean_ms =
        round(1e3 * c.latency_mean; digits=1)  latency_p95_ms =
        round(1e3 * c.latency_p95; digits=1)  gen_per_action =
        round(c.model_calls_per_action; digits=1)  mean_root_children =
        round(c.extra[:root_action_children]; digits=2)
end
