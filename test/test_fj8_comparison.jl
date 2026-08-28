# FJ8.4b — the comparison machinery.
#
# The expensive experiment lives in `tools/solver_comparison.jl`: twenty seeds at
# a 150-decision horizon with two planners at ~1 000 generative calls per
# decision is roughly twenty minutes of wall time, which does not belong in a
# regression suite. What belongs here is proof that the machinery is right —
# the pairing, the denominators, the statistics, the two-block separation and
# the protocol guard — exercised end to end on a small configuration.

using DuckietownDecisionModels
using POMDPs
using Test
using Random
using YAML

const FJ84B_QCFG = joinpath(pkgdir(DuckietownDecisionModels), "..", "duckduck",
    "policies", "q_learning", "training_config.yaml")

struct FixedActionPolicy <: DuckietownDecisionModels.AbstractPolicy
    a::MacroAction
end
POMDPs.action(p::FixedActionPolicy, ::AnyMDPLike, ::DuckieWorldState) = p.a

"""Build a SolverRun from a real (small) evaluation, so the tests exercise the
same path the experiment uses."""
function small_run(name, family, mdp, policy; seeds=1:4, horizon=8)
    res = evaluate_planner(mdp, policy; seeds=seeds, max_steps=horizon)
    return SolverRun(name, family, res.episodes, res.cost, res.decisions, "")
end

@testset "FJ8.4b the evaluation protocol is frozen on disk" begin
    root = pkgdir(DuckietownDecisionModels)
    path = joinpath(root, "configs", "planning", "evaluation.yaml")
    @test isfile(path)
    proto = YAML.load_file(path)

    @test proto["protocol"]["horizon"] == 150
    @test proto["protocol"]["primary_operating_point"] == 1000
    @test length(proto["solvers"]) == 6

    # the two planners must be pinned to the FJ8.4a calibration, and the
    # widening decisions must match what was measured there
    @test proto["solvers"]["mcts"]["n_iterations"] == 36
    @test proto["solvers"]["dpw"]["n_iterations"] == 35
    @test proto["solvers"]["dpw"]["enable_action_pw"] == true
    @test proto["solvers"]["dpw"]["enable_state_pw"] == false
    @test proto["solvers"]["dpw"]["k_action"] == 4.0
    @test proto["solvers"]["dpw"]["alpha_action"] == 0.5
    @test proto["solvers"]["mcts"]["reuse_tree"] == false

    # sensitivity is planner-only and separate from the six-solver table
    @test proto["sensitivity"]["targets"] == [500, 1000, 2000]
    @test length(proto["sensitivity"]["mcts_iterations"]) == 3

    # and the evaluation seeds are still disjoint from the development ones
    cfg = planning_seed_config()
    @test isempty(intersect(cfg.development.states, cfg.evaluation.episodes))
    @test cfg.evaluation.max_steps == proto["protocol"]["horizon"]
end

@testset "FJ8.4b summary statistics" begin
    s = summary_stats([1.0, 2.0, 3.0, 4.0, 100.0])
    @test s.n == 5
    @test s.mean ≈ 22.0
    @test s.median == 3.0            # median is not dragged by the outlier
    @test s.min == 1.0 && s.max == 100.0
    @test s.q25 <= s.median <= s.q75
    @test s.ci_lo <= s.mean <= s.ci_hi
    @test s.sd > 0

    # the bootstrap is reproducible, which is why a seed is fixed
    @test summary_stats([1.0, 5.0, 9.0]).ci_lo ==
        summary_stats([1.0, 5.0, 9.0]).ci_lo

    # degenerate inputs do not throw
    @test summary_stats(Float64[]).n == 0
    one = summary_stats([7.0])
    @test one.n == 1 && one.mean == 7.0 && one.sd == 0.0
end

@testset "FJ8.4b episodes are paired by seed" begin
    mdp = InstrumentedMDP(DuckietownMDP(FJ84B_QCFG; action_space=:discrete))
    a = small_run("straight", "test", mdp, FixedActionPolicy(FAST_STRAIGHT))
    b = small_run("brake", "test", mdp, FixedActionPolicy(BRAKE))

    d = paired_difference("straight", a.episodes, "brake", b.episodes, :ret)
    @test d.stats.n == 4
    @test d.n_a_better + d.n_b_better + d.n_tied == 4
    # the difference really is per-seed, not a difference of means
    manual = [ea.ret - eb.ret for (ea, eb) in
              zip(sort(a.episodes; by=e -> e.seed),
                  sort(b.episodes; by=e -> e.seed))]
    @test d.stats.mean ≈ sum(manual) / 4
    @test d.stats.median ≈ summary_stats(manual).median

    # a mismatched seed set is a protocol error, not something to intersect
    short = small_run("short", "test", mdp, FixedActionPolicy(BRAKE); seeds=1:3)
    @test_throws ArgumentError paired_difference("a", a.episodes, "s",
        short.episodes, :ret)
    @test_throws ArgumentError check_paired_protocol([a, short])

    ok = check_paired_protocol([a, b])
    @test ok.seeds == [1, 2, 3, 4]
    @test ok.solvers == 2 && ok.episodes == 4
end

@testset "FJ8.4b denominators are explicit" begin
    # stop compliance divides by stop ENCOUNTERS, not by episodes
    mk(passed, viol) = EpisodeMetrics(1, 10, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, false, false, false, false, false, 0, viol, passed, 0, nothing,
        0, 0, 0, "in_progress")
    @test stop_compliance([mk(4, 1), mk(6, 0)]) ≈ 9 / 10
    @test stop_compliance([mk(2, 2)]) == 0.0
    # never reaching a stop sign is "n/a", never perfect compliance
    @test stop_compliance([mk(0, 0), mk(0, 0)]) === nothing
    @test stop_compliance(EpisodeMetrics[]) === nothing
end

@testset "FJ8.4b task and cost blocks stay separate" begin
    mdp = InstrumentedMDP(DuckietownMDP(FJ84B_QCFG; action_space=:discrete))
    runs = [small_run("fixed-straight", "test (learned)", mdp,
            FixedActionPolicy(FAST_STRAIGHT)),
        small_run("fixed-brake", "test (learned)", mdp,
            FixedActionPolicy(BRAKE))]

    task = task_table(runs)
    cost = cost_table(runs)
    safety = safety_table(runs)

    # the task block carries no timing and no model-call column
    @test !occursin("ms ", task)
    @test !occursin("gen/act", task)
    @test occursin("return", task)
    @test occursin("max|d|", task)
    # the cost block carries no return
    @test !occursin("return", cost)
    @test occursin("gen/act", cost)
    @test occursin("ms p95", cost)
    # both name every solver
    for r in runs
        @test occursin(r.name, task)
        @test occursin(r.name, cost)
        @test occursin(r.name, safety)
    end
    # a policy that never consults the model reports a measured zero
    @test all(r -> r.cost.model_calls_per_action == 0.0, runs)
    @test occursin("0.0", cost)
    # explicit denominator in the safety block
    @test occursin("stop enc", safety)
    @test occursin("compliance", safety)

    @info "FJ8.4b task block\n" * task
    @info "FJ8.4b cost block\n" * cost
end

@testset "FJ8.4b per-decision records keep episode structure" begin
    mdp = InstrumentedMDP(DuckietownMDP(FJ84B_QCFG; action_space=:discrete))
    res = evaluate_planner(mdp, FixedActionPolicy(SLOW_STRAIGHT); seeds=1:3,
        max_steps=10)
    recs = res.decisions
    @test recs isa Vector{DecisionRecord}
    @test length(recs) == sum(e -> e.decisions, res.episodes)
    @test Set(r.seed for r in recs) == Set(1:3)
    for s in 1:3
        steps = [r.step for r in recs if r.seed == s]
        @test steps == collect(1:length(steps))     # contiguous, in order
    end

    rows = cost_by_episode_position(recs; bins=4)
    @test !isempty(rows)
    @test sum(r -> r.n, rows) == length(recs)
    @test all(r -> 0.0 <= r.from < r.to <= 1.0, rows)
    @test issorted([r.bin for r in rows])
    @test occursin("episode fraction", position_table(rows))
    @test isempty(cost_by_episode_position(DecisionRecord[]))

    # recording per-decision changes nothing about the episodes themselves
    plain = evaluate_policy(mdp, FixedActionPolicy(SLOW_STRAIGHT); seeds=1:3,
        max_steps=10)
    @test [e.ret for e in plain] == [e.ret for e in res.episodes]
end

@testset "FJ8.4b the episode CSV preserves the paired structure" begin
    mdp = InstrumentedMDP(DuckietownMDP(FJ84B_QCFG; action_space=:discrete))
    runs = [small_run("a", "test", mdp, FixedActionPolicy(FAST_STRAIGHT)),
        small_run("b", "test", mdp, FixedActionPolicy(SLOW_STRAIGHT))]
    csv = episode_csv(runs)
    lines = split(strip(csv), "\n")
    @test length(lines) == 1 + 2 * 4          # header + one row per (solver, seed)
    @test startswith(lines[1], "solver,family,seed,")
    @test occursin("max_abs_d", lines[1])
    ncols = length(split(lines[1], ","))
    @test all(l -> length(split(l, ",")) == ncols, lines)
    # every solver/seed pair appears exactly once
    keys_ = [(split(l, ",")[1], split(l, ",")[3]) for l in lines[2:end]]
    @test length(unique(keys_)) == 8
end
