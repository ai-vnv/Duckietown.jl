# FJ8.4c — evaluation artefact enrichment.
#
#   FJ8.4c DOES NOT SUPERSEDE FJ8.4b. It reproduces the frozen evaluation
#   protocol while enriching the observational record.
#
# The expensive replication runs in tools/run_enrich.sh. These tests check the
# two things that make it trustworthy: the logger is observational (it changes
# nothing about the episode) and the enriched rows re-aggregate to exactly the
# original episode artefact.

using DuckietownDecisionModels
using POMDPs
using Test
using Random
using JSON3

const FJ84C_DIR = joinpath(pkgdir(DuckietownDecisionModels), "artifacts",
    "fj8", "enriched")
const FJ84C_QCFG = joinpath(pkgdir(DuckietownDecisionModels), "..", "duckduck",
    "policies", "q_learning", "training_config.yaml")

@testset "FJ8.4c the logger is observational" begin
    mdp = InstrumentedMDP(DuckietownMDP(FJ84C_QCFG; action_space=:discrete))
    qpath = joinpath(pkgdir(DuckietownDecisionModels), "..", "duckduck",
        "policies", "q_learning", "policy.npy")
    if !isfile(qpath)
        @test_skip "tabular checkpoint unavailable"
    else
        pol = QTablePolicy(qpath; solver=:q_learning)

        plain = evaluate_policy(mdp, pol; seeds=1001:1003, max_steps=25)
        traced = DecisionTrace[]
        with = evaluate_policy(mdp, pol; seeds=1001:1003, max_steps=25,
            record=traced, trace_solver="q_learning")

        # tracing must not change the episode in any respect
        @test length(plain) == length(with)
        for (a, b) in zip(plain, with)
            for f in fieldnames(EpisodeMetrics)
                @test getfield(a, f) === getfield(b, f)
            end
        end
        @test length(traced) == sum(e -> e.decisions, plain)

        # the model-call counter must see the same consumption either way: a
        # logger that ran an extra gen would show up here
        reset_model_calls!(mdp)
        evaluate_policy(mdp, pol; seeds=1001:1003, max_steps=25)
        without_calls = model_calls(mdp)
        reset_model_calls!(mdp)
        evaluate_policy(mdp, pol; seeds=1001:1003, max_steps=25,
            record=DecisionTrace[], trace_solver="q_learning")
        @test model_calls(mdp) == without_calls
    end
end

@testset "FJ8.4c the trace re-aggregates to the episode metrics" begin
    mdp = InstrumentedMDP(DuckietownMDP(FJ84C_QCFG; action_space=:discrete))
    qpath = joinpath(pkgdir(DuckietownDecisionModels), "..", "duckduck",
        "policies", "q_learning", "policy.npy")
    if !isfile(qpath)
        @test_skip "tabular checkpoint unavailable"
    else
        pol = QTablePolicy(qpath; solver=:q_learning)
        traced = DecisionTrace[]
        eps = evaluate_policy(mdp, pol; seeds=1001:1004, max_steps=40,
            record=traced, trace_solver="q_learning")
        re = reaggregate_episodes(traced)

        @test length(re) == length(eps)
        for (a, b) in zip(sort(eps; by=e -> e.seed), sort(re; by=e -> e.seed))
            for f in fieldnames(EpisodeMetrics)
                @test getfield(a, f) === getfield(b, f)
            end
        end

        # the CSV round-trips the schema
        csv = decision_csv(traced)
        lines = split(strip(csv), "\n")
        @test length(lines) == length(traced) + 1
        @test split(lines[1], ",") == collect(DECISION_TRACE_SCHEMA)
        @test all(l -> length(split(l, ",")) == length(DECISION_TRACE_SCHEMA),
            lines)
    end
end

@testset "FJ8.4c the frozen replication reproduced FJ8.4b exactly" begin
    report_path = joinpath(FJ84C_DIR, "reproduction_report.json")
    if !isfile(report_path)
        @test_skip "enrichment has not been run"
    else
        r = JSON3.read(read(report_path, String))
        @test r.gate == "FJ8.4c"
        @test occursin("does not supersede", r.statement)
        @test r.episodes_checked == 120
        @test r.fields_per_episode >= 20
        @test isempty(r.mismatches)
        @test r.exact == true
        @test r.horizon == 150
        @test length(r.seeds) == 20
        @test length(r.solvers) == 6
        @test r.decision_rows > 10_000

        fp = JSON3.read(read(joinpath(FJ84C_DIR, "fingerprints.json"), String))
        # the enriched run and the original artefact describe the same episodes
        @test fp.episode_fingerprint_enriched == fp.episode_fingerprint_original
        @test !isempty(fp.decision_log_fingerprint)
        @test !isempty(fp.experiment_fingerprint)

        # the original artefact was not overwritten
        original = joinpath(pkgdir(DuckietownDecisionModels), "artifacts",
            "fj8", "six_solver_episodes.csv")
        @test isfile(original)
        @test load_rollout_artifact(original).provenance.rows == 120

        @info "FJ8.4c reproduction" episodes = r.episodes_checked decision_rows =
            r.decision_rows mismatches = length(r.mismatches) exact = r.exact
    end
end

@testset "FJ8.4c the enriched log carries per-decision planning cost" begin
    path = joinpath(FJ84C_DIR, "decisions.csv")
    if !isfile(path)
        @test_skip "enrichment has not been run"
    else
        header = strip.(split(first(eachline(path)), ","))
        @test "model_calls" in header
        @test "planning_time" in header
        @test "ego_x" in header && "ego_z" in header && "ego_angle" in header
        @test "decision" in header && "seed" in header && "solver" in header
        @test "reward_total" in header && "reward_progress" in header
        @test "full_stop" in header && "passed_stop" in header
        @test "reason" in header && "terminated" in header
        @test length(header) == length(DECISION_TRACE_SCHEMA)

        # the per-decision record answers what FJ8.4b could only average:
        # a learned policy consumes zero generative calls, a planner does not,
        # and DPW's consumption collapses as its episodes deteriorate
        idx = Dict(h => k for (k, h) in enumerate(header))
        rows = [split(l, ",") for l in Iterators.drop(eachline(path), 1)]
        calls_of(s) = [parse(Int, r[idx["model_calls"]]) for r in rows
                       if r[idx["solver"]] == s]
        @test all(==(0), calls_of("q_learning"))
        @test all(>(0), calls_of("mcts@1k"))
        dpw = calls_of("dpw@1k")
        @test !isempty(dpw)
        @test maximum(dpw) > 900
        @test minimum(dpw) < 400        # the collapse is in the record
        @info "FJ8.4c per-decision model calls" q_learning =
            length(calls_of("q_learning")) mcts_range =
            extrema(calls_of("mcts@1k")) dpw_range = extrema(dpw)
    end
end
