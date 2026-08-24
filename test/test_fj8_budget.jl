# FJ8.4a — the cost–search curve.
#
# Not "what is the best budget". The deliverable is the relationship between
# what a solver is *told* to do (iterations) and what it actually *costs*
# (generative calls, milliseconds, bytes) — because two solvers quoting the
# same iteration count are not on the same computational budget.
#
# Everything here runs on the DEVELOPMENT seed set. The evaluation seeds are
# frozen in configs/planning/fj8_seeds.yaml and are not read by this file, so
# no configuration choice can be made against the seeds FJ8.4b will report.

using DuckietownDecisionModels
using POMDPs
using Test
using Random

const FJ84_OK = try
    @eval using MCTS
    true
catch err
    @info "FJ8.4a: skipped (MCTS.jl not available)" err
    false
end

const FJ84_QCFG = joinpath(pkgdir(DuckietownDecisionModels), "..", "duckduck",
    "policies", "q_learning", "training_config.yaml")
const FJ84_CCFG = joinpath(pkgdir(DuckietownDecisionModels), "..", "duckduck",
    "policies", "sac", "training_config.yaml")

@testset "FJ8.4a the seed split is frozen and disjoint" begin
    cfg = planning_seed_config()
    @test isfile(cfg.path)
    @test length(cfg.development.states) == 5
    @test length(cfg.evaluation.episodes) == 20
    @test isempty(intersect(cfg.development.states, cfg.evaluation.episodes))
    @test cfg.evaluation.max_steps == 150
    @test cfg.evaluation.planner isa Int
    @info "FJ8.4a seed split" development_states = cfg.development.states evaluation_episodes =
        length(cfg.evaluation.episodes)  horizon = cfg.evaluation.max_steps
end

FJ84_OK && @testset "FJ8.4a cost-search curve for both planners" begin
    cfg = planning_seed_config()
    dev_seeds = cfg.development.states

    dmdp = InstrumentedMDP(DuckietownMDP(FJ84_QCFG; action_space=:discrete))
    cmdp = InstrumentedMDP(DuckietownMDP(FJ84_CCFG; action_space=:continuous))

    # the SAME states at every budget, drawn from the development seeds
    dstates = [rand(MersenneTwister(k), initialstate(dmdp)) for k in dev_seeds]
    cstates = [rand(MersenneTwister(k), initialstate(cmdp)) for k in dev_seeds]
    @test length(dstates) == length(cstates) == 5

    mcts_of(b, seed) = solve(MCTSSolver(n_iterations=b, depth=10,
        exploration_constant=5.0, rng=MersenneTwister(seed),
        reuse_tree=false), dmdp)
    dpw_of(b, seed) = solve(DPWSolver(n_iterations=b, depth=10,
        exploration_constant=5.0, enable_action_pw=true,
        enable_state_pw=false, k_action=4.0, alpha_action=0.5,
        rng=MersenneTwister(seed), keep_tree=false), cmdp)

    budgets = (5, 10, 25, 50, 100, 250, 500, 1000)
    mcts_curve = budget_study(dmdp, mcts_of, budgets; states=dstates,
        repeats=2, label="MCTS (discrete)")
    dpw_curve = budget_study(cmdp, dpw_of, budgets; states=cstates,
        repeats=2, label="DPW (continuous)")

    for curve in (mcts_curve, dpw_curve)
        @test length(curve) == length(budgets)
        @test [p.budget for p in curve] == collect(budgets)
        @test all(p -> p.decisions == 10, curve)      # 5 states x 2 repeats
        @test all(p -> p.latency_mean > 0, curve)
        @test all(p -> p.model_calls_mean > 0, curve)
        @test all(p -> p.bytes_per_decision > 0, curve)
        # both cost axes must grow with the budget
        @test issorted([p.model_calls_mean for p in curve])
        @test curve[end].latency_mean > curve[1].latency_mean
        # quantile ordering
        @test all(p -> p.latency_p50 <= p.latency_p95 <= p.latency_max, curve)
    end

    # Iterations are not a computational budget — but MEASURED, the reason is
    # not the one that was assumed. Both solvers consume nearly the same work
    # per iteration on this state set; what varies is the state (next test set).
    # The ratio is reported, not asserted to differ.
    m_per_iter = [p.model_calls_per_iteration for p in mcts_curve]
    d_per_iter = [p.model_calls_per_iteration for p in dpw_curve]
    @test all(isfinite, m_per_iter)
    @test all(isfinite, d_per_iter)
    @test all(>(0), m_per_iter)
    @test all(>(0), d_per_iter)
    ratio = sum(d_per_iter) / sum(m_per_iter)
    # the rate is stable across budgets for each solver, which is what makes
    # inverting it a usable estimator
    @test maximum(m_per_iter) / minimum(m_per_iter) < 1.5
    @test maximum(d_per_iter) / minimum(d_per_iter) < 1.5

    @info "FJ8.4a MCTS cost-search curve\n" * budget_table(mcts_curve;
        extra_fields=[:tree_nodes, :action_nodes, :root_children])
    @info "FJ8.4a DPW cost-search curve\n" * budget_table(dpw_curve;
        extra_fields=[:state_nodes, :action_nodes, :root_action_children,
            :max_state_children])
    @info "FJ8.4a generative calls per iteration" mcts =
        round.(m_per_iter; digits=2)  dpw = round.(d_per_iter; digits=2) dpw_over_mcts =
        round(ratio; digits=2)

    # compute-matched operating points: matched on MEASURED generative calls,
    # never on iteration count, with the realised value reported
    curves = Dict("MCTS (discrete)" => mcts_curve,
        "DPW (continuous)" => dpw_curve)
    targets = (500, 1_000, 2_000)
    rows = matched_operating_points(curves, targets)
    @test length(rows) == 2 * length(targets)
    @test all(r -> r.budget in budgets, rows)
    @test all(r -> r.actual_calls > 0, rows)
    @info "FJ8.4a nearest grid point\n" * operating_point_table(rows)

    # Nearest-on-a-coarse-grid can be tens of percent off, so the operating
    # points FJ8.4b will actually use are obtained by inverting the measured
    # calls-per-iteration rate and then MEASURING what that budget realises.
    tuned = NamedTuple[]
    for target in targets
        for (label, curve, mk, m, sts) in (
            ("MCTS (discrete)", mcts_curve, mcts_of, dmdp, dstates),
            ("DPW (continuous)", dpw_curve, dpw_of, cmdp, cstates))
            b = estimate_budget_for_calls(curve, target)
            @test b >= 1
            pt = only(budget_study(m, mk, (b,); states=sts, repeats=1,
                label=label))
            push!(tuned, (label=label, target=Float64(target), budget=b,
                actual_calls=pt.model_calls_mean,
                relative_error=(pt.model_calls_mean - target) / target,
                latency_mean=pt.latency_mean))
        end
    end
    # inverting a measured, stable rate must land far closer than the grid
    @test all(r -> abs(r.relative_error) < 0.15, tuned)
    @info "FJ8.4a compute-matched operating points (estimated, then measured)\n" *
        operating_point_table(tuned)

    # write the artefact
    dir = joinpath(pkgdir(DuckietownDecisionModels), "artifacts", "fj8")
    mkpath(dir)
    path = joinpath(dir, "budget_study.md")
    open(path, "w") do io
        println(io, "# FJ8.4a — planner cost–search curve")
        println(io)
        println(io, "Generated by `test/test_fj8_budget.jl`. Development seeds ",
            "only (`", basename(cfg.path), "`): states ",
            dev_seeds, ", ", 2, " repeats per (state, budget), warm-up run ",
            "discarded. Latency is wall clock and varies run to run; ",
            "generative calls and node counts are exactly reproducible.")
        println(io)
        println(io, "## MCTS, discrete macro actions\n\n```")
        print(io, budget_table(mcts_curve;
            extra_fields=[:tree_nodes, :action_nodes, :root_children]))
        println(io, "```\n")
        println(io, "## DPW, continuous action box\n\n```")
        print(io, budget_table(dpw_curve;
            extra_fields=[:state_nodes, :action_nodes,
                :root_action_children, :max_state_children]))
        println(io, "```\n")
        println(io, "## Compute-matched operating points\n")
        println(io, "Matched on measured generative calls per decision, never ",
            "on iteration count. Nearest point of the measured grid:\n\n```")
        print(io, operating_point_table(rows))
        println(io, "```\n")
        println(io, "Budget estimated by inverting the measured ",
            "calls-per-iteration rate, then run and measured. These are the ",
            "operating points FJ8.4b uses:\n\n```")
        print(io, operating_point_table(tuned))
        println(io, "```")
    end
    @test isfile(path)
end

FJ84_OK && @testset "FJ8.4a what actually sets the cost per iteration" begin
    # An earlier reading of FJ8.2/8.3 suggested MCTS and DPW differ in
    # generative calls per iteration (~15.6 vs ~23). Averaged over the same
    # development states they do not — both sit near 28. The variable is the
    # STATE, not the solver, and the mechanism is rollout survival: a rollout
    # ends when the episode terminates, which happens far sooner from some
    # spawns than others. Measured here so the earlier reading is corrected
    # rather than carried forward.
    imdp = InstrumentedMDP(DuckietownMDP(FJ84_QCFG; action_space=:discrete))
    rows = NamedTuple[]
    for sd in (11, 101, 102, 103, 104, 105)
        s = rand(MersenneTwister(sd), initialstate(imdp))
        # how long a random-action rollout survives from this state
        rng = MersenneTwister(7)
        t, len = s, 0
        for _ in 1:60
            r = simulate_decision(imdp.transition, t,
                ALL_MACRO_ACTIONS[rand(rng, 1:7)], rng)
            t = r.sp
            len += 1
            (r.terminated || r.truncated) && break
        end
        per_iter = Float64[]
        for depth in (10, 12)
            reset_model_calls!(imdp)
            p = solve(MCTSSolver(n_iterations=20, depth=depth,
                exploration_constant=5.0, rng=MersenneTwister(1),
                reuse_tree=false), imdp)
            action(p, s)
            push!(per_iter, model_calls(imdp) / 20)
        end
        push!(rows, (seed=sd, rollout_len=len, d10=per_iter[1],
            d12=per_iter[2]))
    end

    # depth is NOT binding: rollouts die of termination well before the cap,
    # so raising the depth limit buys nothing on this model
    @test all(r -> r.d10 == r.d12, rows)

    # the state is what varies, and it varies a lot
    rates = [r.d10 for r in rows]
    @test maximum(rates) / minimum(rates) > 2.0

    # and it correlates strongly with rollout survival. Correlation, not an
    # exact argmin/argmax correspondence: one random rollout is a crude proxy
    # for the many rollouts a search actually performs, and two of these seeds
    # share a rollout length while differing in cost.
    lens = Float64[r.rollout_len for r in rows]
    function pearson(x, y)
        mx, my = sum(x) / length(x), sum(y) / length(y)
        dx, dy = x .- mx, y .- my
        return sum(dx .* dy) / sqrt(sum(abs2, dx) * sum(abs2, dy))
    end
    r_corr = pearson(lens, rates)
    @test r_corr > 0.7
    @test rates[argmax(lens)] == maximum(rates)

    @info "FJ8.4a cost per iteration is a property of the state\n" *
        join(["  seed=$(lpad(r.seed,3))  random_rollout_len=$(lpad(r.rollout_len,3))" *
              "  gen/iter@depth10=$(lpad(r.d10,6))  @depth12=$(lpad(r.d12,6))"
              for r in rows], "\n") *
        "\n  pearson(rollout_len, gen/iter) = $(round(r_corr; digits=3))" *
        "\n  spread max/min = $(round(maximum(rates) / minimum(rates); digits=2))x"
end

FJ84_OK && @testset "FJ8.4a matching is on measured calls, not iterations" begin
    # a small synthetic curve, so the selection logic is tested independently
    # of how expensive any real planner happens to be
    pts = [BudgetPoint("x", b, 1, 1, 1, 1e-3 * b, 1e-3 * b, 1e-3 * b,
        1e-3 * b, 10.0 * b, 10.0, 0.0, Dict{Symbol,Float64}())
           for b in (25, 50, 100, 250)]
    m = compute_matched_budget(pts, 1_000)
    @test m.budget == 100                 # 100 iterations -> 1000 calls
    @test m.actual_calls == 1_000.0
    @test m.error == 0.0
    m2 = compute_matched_budget(pts, 900)
    @test m2.budget == 100                # nearest, and the error is reported
    @test m2.actual_calls == 1_000.0
    @test m2.relative_error ≈ 100 / 900

    # an unmeasured model is refused rather than silently matched on zeros
    unmeasured = [BudgetPoint("y", 10, 1, 1, 1, 1e-3, 1e-3, 1e-3, 1e-3,
        -1.0, NaN, 0.0, Dict{Symbol,Float64}())]
    @test_throws ArgumentError compute_matched_budget(unmeasured, 100)
end

@testset "FJ8.4a a learned policy has a flat curve" begin
    # the study is solver-agnostic: a policy that never consults the model
    # must report zero generative calls at every budget — 0, not -1, because
    # this is measured knowledge rather than a missing measurement
    qpath = joinpath(pkgdir(DuckietownDecisionModels), "..", "duckduck",
        "policies", "q_learning", "policy.npy")
    if !isfile(qpath)
        @test_skip "tabular checkpoint unavailable"
    else
        imdp = InstrumentedMDP(DuckietownMDP(FJ84_QCFG; action_space=:discrete))
        qpol = QTablePolicy(qpath; solver=:q_learning)
        states = [rand(MersenneTwister(k), initialstate(imdp))
                  for k in planning_seed_config().development.states]
        curve = budget_study(imdp, (b, seed) -> qpol, (1, 100, 1000);
            states=states, repeats=2, label="Q-learning")
        @test length(curve) == 3
        @test all(p -> p.model_calls_mean == 0.0, curve)
        @test all(p -> isempty(p.extra), curve)
        @test all(p -> p.latency_mean > 0, curve)
        @info "FJ8.4a learned-policy curve\n" * budget_table(curve)
    end
end
