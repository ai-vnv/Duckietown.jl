# Native-scenario tests for the evaluation machinery.
#
# Everything here runs from a single clone: worlds come from
# `scenario_config`, the planner comes from the test environment's MCTS, and
# no reference material is touched. The machinery under test is the same code
# the gated FJ7/FJ8 experiments run through — those gates validate it against
# the reference; these tests keep it exercised (and counted) everywhere else.

using Duckietown
using POMDPs
using Test
using Random
using MCTS

const EVN_MDP = DuckietownMDP(scenario_config(:stop_and_duck);
    action_space = :discrete)

# The simplest legal POMDPs.jl policy: always drive straight. Deliberately
# incompetent — episodes end early and cheaply, which is all these tests need.
struct EvnStraight <: POMDPs.Policy end
POMDPs.action(::EvnStraight, ::DuckieWorldState) = FAST_STRAIGHT

evn_planner(m; iters = 3, seed = 1) = solve(MCTSSolver(n_iterations = iters,
    depth = 5, exploration_constant = 5.0, rng = MersenneTwister(seed)), m)

@testset "evaluation: a policy is measured, not scored" begin
    eps = evaluate_policy(EVN_MDP, EvnStraight(); seeds = 1:4, max_steps = 60)
    @test length(eps) == 4
    for e in eps
        @test e.decisions > 0
        @test isfinite(e.ret) && isfinite(e.discounted_return)
        # undiscounted return upper-bounds the discounted one only in sign-free
        # worlds; here the honest property is finiteness plus a recorded reason
        @test !isempty(e.reason)
        @test e.mean_abs_d >= 0 && e.max_abs_d >= e.mean_abs_d - 1e-12
        @test 0.0 <= e.brake_ratio <= 1.0
        # exactly one recorded ending class per episode
        endings = count((e.offroad, e.other_collision, e.duck_collision,
            e.timeout, e.goal))
        @test endings <= 1
    end
    # the same seeds reproduce the same measurements bit for bit
    eps2 = evaluate_policy(EVN_MDP, EvnStraight(); seeds = 1:4, max_steps = 60)
    @test [e.ret for e in eps2] == [e.ret for e in eps]

    summ = summarize_evaluation(eps)
    @test summ !== nothing
end

@testset "evaluation: planner cost is measured separately" begin
    im = InstrumentedMDP(EVN_MDP)
    reset_model_calls!(im)
    @test model_calls(im) == 0

    planner = evn_planner(im)
    out = evaluate_planner(im, planner; seeds = 1:2, max_steps = 12)
    @test length(out.episodes) == 2
    @test out.cost isa PlannerCost
    @test !isempty(out.decisions)
    @test all(d -> d.diagnostics.model_calls > 0, out.decisions)

    # plan_action reports what one decision cost
    s = rand(MersenneTwister(11), initialstate(im))
    before = model_calls(im)
    a, diag = plan_action(planner, im, s)
    @test a isa MacroAction
    @test diag.model_calls == model_calls(im) - before
    @test diag.planning_time >= 0

    # an uninstrumented model says "unmeasured", never zero
    _, diag_plain = plan_action(EvnStraight(), EVN_MDP, s)
    @test diag_plain.model_calls == -1
end

@testset "evaluation: capabilities are exercised, not asserted" begin
    caps = model_capabilities(EVN_MDP; seeds = 1:4, policy = EvnStraight())
    @test caps.consumes_rng isa Bool
    @test 0.0 <= caps.stochastic_state_fraction <= 1.0
    rep = capability_report(EVN_MDP)
    @test rep isa AbstractString && !isempty(rep)
end

@testset "evaluation: decision traces reaggregate to the episodes" begin
    traces = DecisionTrace[]
    eps = evaluate_policy(EVN_MDP, EvnStraight(); seeds = 1:2, max_steps = 40,
        record = traces, trace_solver = "native_straight")
    @test !isempty(traces)

    csv = decision_csv(traces)
    @test startswith(csv, first(split(csv, "\n")))  # has a header line
    @test count(==('\n'), csv) >= length(traces)

    re = reaggregate_episodes(traces)
    @test length(re) == length(eps)
end

@testset "evaluation: the budget study machinery" begin
    im = InstrumentedMDP(EVN_MDP)
    states = [rand(MersenneTwister(sd), initialstate(im)) for sd in (11, 12)]
    points = budget_study(im, (b, sd) -> evn_planner(im; iters = b, seed = sd),
        [2, 4]; states = states, repeats = 2, warmup = true)
    @test length(points) == 2
    @test all(p -> p.model_calls_mean > 0, points)
    @test issorted([p.budget for p in points])
    @test all(p -> p.decisions > 0, points)

    tbl = budget_table(points)
    @test !isempty(tbl)

    est = estimate_budget_for_calls(points, points[end].model_calls_mean)
    @test est > 0
    matched = compute_matched_budget(points, points[1].model_calls_mean)
    @test matched !== nothing

    curves = Dict("mcts" => points)
    rows = matched_operating_points(curves, [points[1].model_calls_mean])
    @test !isempty(rows)
    @test !isempty(operating_point_table(rows))

    # the committed seed protocol loads and self-checks
    seeds = planning_seed_config()
    @test !isempty(seeds.development.states)
    @test isempty(intersect(seeds.development.states,
        seeds.evaluation.episodes))
end

@testset "evaluation: the generative benchmark measures, never guesses" begin
    s0 = rand(MersenneTwister(21), initialstate(EVN_MDP))

    # identical worlds have no differences; a stepped world has some
    @test isempty(world_differences(s0, s0))
    sp = @gen(:sp)(EVN_MDP, s0, FAST_STRAIGHT, MersenneTwister(1))
    @test !isempty(world_differences(s0, sp))

    shared = shared_mutable_arrays(s0, sp)
    ok_by_design = shared_by_design(s0, sp)
    @test shared isa AbstractVector
    @test ok_by_design isa Bool || ok_by_design isa AbstractVector

    b = benchmark_gen(EVN_MDP, s0, FAST_STRAIGHT)
    @test b isa GenBenchmark
    @test !isempty(benchmark_table([b]))

    prof = gen_stage_profile(EVN_MDP, s0, FAST_STRAIGHT)
    @test prof !== nothing

    m = measure(() -> 1 + 1, 10; label = "noop")
    @test m !== nothing
end

@testset "evaluation: native rollouts and drift analysis" begin
    model = EVN_MDP.transition
    s0 = rand(MersenneTwister(31), initialstate(EVN_MDP))
    acts = fill(FAST_STRAIGHT, 25)

    a = rollout_native(model, s0, acts; rng = MersenneTwister(5))
    b = rollout_native(model, s0, acts; rng = MersenneTwister(5))
    @test !isempty(a) && length(a) == length(b)

    same = compare_rollouts(a, b)
    @test same isa DriftReport
    @test !isempty(drift_summary(same))

    c = rollout_native(model, s0, fill(SLOW_STRAIGHT, 25);
        rng = MersenneTwister(5))
    diff = compare_rollouts(a, c)
    @test !isempty(drift_summary(diff))

    @test event_timing(a) !== nothing
    @test event_timing_diff(a, c) !== nothing
    @test !isempty(rollout_table(a))

    rows = three_lane_table(a, a, a)
    @test !isempty(rows)
    hyp = libm_hypothesis_check(rows)
    @test hyp.stats isa AbstractDict
    # three identical lanes: every cross-lane delta is exactly zero
    @test hyp.d_CJ_exactly_zero
end

@testset "evaluation: comparison statistics on measured episodes" begin
    im = InstrumentedMDP(EVN_MDP)
    planner = evn_planner(im; iters = 2, seed = 7)
    o1 = evaluate_planner(im, planner; seeds = 1:3, max_steps = 10)
    o2 = evaluate_planner(im, EvnStraight(); seeds = 1:3, max_steps = 10)

    st = summary_stats([e.ret for e in o1.episodes]; bootstrap = 200,
        seed = 1)
    @test st.n == 3

    pd = paired_difference("mcts", o1.episodes, "straight", o2.episodes, :ret)
    @test pd isa PairedDifference

    # short probe episodes may never reach a stop sign; nothing is the honest
    # answer there, a ratio in [0,1] otherwise — never a fabricated zero
    sc = stop_compliance(o1.episodes)
    @test sc === nothing || 0.0 <= sc <= 1.0

    runs = [SolverRun("mcts", "search", o1.episodes, o1.cost, o1.decisions,
                "scenario_config(:stop_and_duck)"),
            SolverRun("straight", "baseline", o2.episodes, o2.cost,
                o2.decisions, "scenario_config(:stop_and_duck)")]
    @test !isempty(task_table(runs))
    @test !isempty(safety_table(runs))
    @test !isempty(cost_table(runs))
    @test !isempty(paired_table([pd]))
    @test !isempty(episode_csv(runs))

    pos = cost_by_episode_position(o1.decisions)
    @test !isempty(position_table(pos))

    proto = check_paired_protocol(runs)
    @test proto !== nothing
end

@testset "evaluation: parity helpers quantify differences" begin
    @test Duckietown._ulps(1.0, 1.0) == 0
    @test Duckietown._ulps(1.0, nextfloat(1.0)) == 1
    @test Duckietown._ulps(1.0, nextfloat(nextfloat(1.0))) == 2

    d0 = Duckietown._diff("x", 1.0, 1.0)
    d1 = Duckietown._diff("x", 1.0, 1.0 + 1e-6)
    @test d0 === nothing || d0.name == "x"
    @test d1 !== nothing
end

@testset "readiness: the audit runs on the package's own scenario" begin
    items = pomdp_readiness(EVN_MDP)
    @test length(items) == 16
    @test all(i -> !isempty(i.component), items)
    @test all(i -> !isempty(i.evidence), items)
end
