# FJ9.6a/b — the decision-artifact contract and the DiagnosticSeries core.
#
# No environment, no policy, no planner, no solver runs anywhere in this file.
# The FJ8.4c enriched log is the sole evidence, and these tests check that the
# loader refuses anything it cannot interpret, that units and semantics are
# decided in the core, and that a missing value stays missing.

using DuckietownDecisionModels
using Test

const FJ96_LOG = joinpath(pkgdir(DuckietownDecisionModels), "artifacts", "fj8",
    "enriched", "decisions.csv")

# a small hand-built log, so the loader's validation can be tested without the
# artefact and without inventing data that claims to be an experiment
function fj96_synthetic(; rows = nothing)
    header = join(collect(DECISION_LOG_REQUIRED), ",")
    default = Dict("solver" => "toy", "seed" => "1", "decision" => "1",
        "ego_x" => "0.5", "ego_z" => "0.5", "ego_angle" => "0.0",
        "ego_speed" => "0.2", "action_kind" => "macro", "action_id" => "3",
        "v_cmd" => "0.2", "omega_cmd" => "0.0", "d" => "0.01", "phi" => "0.02",
        "v" => "0.2", "kappa" => "0.0", "d_stop" => "", "sigma_stop" => "false",
        "stop_hold_progress" => "0.0", "duck_present" => "false",
        "duck_longitudinal" => "0.0", "duck_lateral" => "0.0",
        "duck_active" => "false", "reward_total" => "1.0",
        "reward_progress" => "1.0", "reward_lateral" => "0.0",
        "reward_heading" => "0.0", "reward_events" => "0.0",
        "full_stop" => "false",
        "passed_stop" => "false", "stop_violation" => "false",
        "offroad" => "false", "other_collision" => "false",
        "duck_collision" => "false", "goal" => "false", "timeout" => "false",
        "terminated" => "false", "truncated" => "false",
        "reason" => "in_progress", "planning_time" => "0.001",
        "model_calls" => "10")
    specs = rows === nothing ?
        [Dict("decision" => "1"), Dict("decision" => "2"),
         Dict("decision" => "3", "terminated" => "true", "reason" => "offroad",
              "offroad" => "true")] : rows
    body = String[]
    for sp in specs
        r = merge(default, sp)
        push!(body, join([r[c] for c in DECISION_LOG_REQUIRED], ","))
    end
    path = tempname() * ".csv"
    write(path, header * "\n" * join(body, "\n") * "\n")
    return path
end

@testset "FJ9.6a the loader validates before it interprets" begin
    good = fj96_synthetic()
    log = load_decision_log(good)
    @test length(log) == 3
    @test log.episodes == [("toy", 1)]
    @test !isempty(log.fingerprint)

    # a missing required column is refused, named
    trimmed = tempname() * ".csv"
    lines = readlines(good)
    keep = [k for (k, h) in enumerate(split(lines[1], ",")) if h != "d_stop"]
    write(trimmed, join([join(split(l, ",")[keep], ",") for l in lines], "\n"))
    @test_throws ArgumentError load_decision_log(trimmed)
    try
        load_decision_log(trimmed)
    catch e
        @test occursin("d_stop", e.msg)
    end

    # a gap in the decision indices is a corrupt record, not a short episode
    gapped = fj96_synthetic(rows = [Dict("decision" => "1"),
        Dict("decision" => "3")])
    @test_throws ArgumentError load_decision_log(gapped)

    # terminating before the final row is refused
    early = fj96_synthetic(rows = [Dict("decision" => "1",
            "terminated" => "true"), Dict("decision" => "2")])
    @test_throws ArgumentError load_decision_log(early)

    # a ragged row is refused
    ragged = tempname() * ".csv"
    write(ragged, join(readlines(good)[1:3], "\n") * "\n1,2,3\n")
    @test_throws ArgumentError load_decision_log(ragged)

    @test_throws ArgumentError load_decision_log(tempname() * ".csv")
    foreach(rm, [good, trimmed, gapped, early, ragged])
end

@testset "FJ9.6b series carry unit, category and semantics from the core" begin
    path = fj96_synthetic()
    ep = episode_diagnostics(load_decision_log(path), "toy", 1)

    @test ep.solver == "toy" && ep.seed == 1
    @test ep.decisions == [1, 2, 3]
    @test ep.progress ≈ [0.0, 1 / 3, 2 / 3]

    # every declared series is present, and every one is annotated
    @test length(ep.series) == length(DIAGNOSTIC_SERIES_SPEC) + 1
    for s in ep.series
        @test length(s) == 3
        @test !isempty(s.semantics)
        @test s.category isa SeriesCategory
        @test s.kind isa SeriesKind
    end
    # a renderer must never have to guess a unit
    @test series_named(ep, "d").unit == "m"
    @test series_named(ep, "phi").unit == "rad"
    @test series_named(ep, "omega_cmd").unit == "rad/s"
    @test series_named(ep, "model_calls").unit == "calls"
    @test series_named(ep, "planning_time").unit == "s"

    # panels are separated by unit, not by convenience
    @test series_named(ep, "d").category === NAVIGATION
    @test series_named(ep, "v_cmd").category === MOTION_COMMAND
    @test series_named(ep, "d_stop").category === STOP_SUBSYS
    @test series_named(ep, "duck_lateral").category === DUCK_SUBSYS
    @test series_named(ep, "reward_total").category === REWARD
    @test series_named(ep, "model_calls").category === COMPUTE
    @test !isempty(series_in(ep, COMPUTE))

    # d and phi live in different coordinate spaces, and the core says so
    @test occursin("coordinate space", series_named(ep, "phi").semantics)

    # the derived series is labelled as derived, and is an exact identity
    cum = series_named(ep, "cumulative_return")
    @test cum.kind === CUMULATIVE
    @test occursin("identity", cum.semantics)
    @test cum.values == [1.0, 2.0, 3.0]

    # flags are flags, not lines
    @test series_named(ep, "sigma_stop").kind === FLAG
    @test series_named(ep, "duck_present").kind === FLAG
    rm(path)
end

@testset "FJ9.6b missing survives to the renderer" begin
    path = fj96_synthetic(rows = [Dict("decision" => "1", "d_stop" => ""),
        Dict("decision" => "2", "d_stop" => "0.0"),
        Dict("decision" => "3", "d_stop" => "0.4")])
    ep = episode_diagnostics(load_decision_log(path), "toy", 1)
    ds = series_named(ep, "d_stop")

    # the three cases stay three cases: absent, exactly zero, and 0.4
    @test ismissing(ds.values[1])
    @test ds.values[2] === 0.0
    @test ds.values[3] === 0.4
    @test n_missing(ds) == 1
    @test !isequal(ds.values[1], ds.values[2])   # missing is not zero
    @test !isequal(ds.values[1], 0.0)
    @test occursin("not the same as zero", ds.semantics)
    rm(path)
end

@testset "FJ9.6b events come from logged indices only" begin
    path = fj96_synthetic(rows = [Dict("decision" => "1"),
        Dict("decision" => "2", "full_stop" => "true"),
        Dict("decision" => "3", "terminated" => "true",
             "reason" => "other_collision", "other_collision" => "true")])
    ep = episode_diagnostics(load_decision_log(path), "toy", 1)

    @test Dict(ep.events)["full_stop"] == [2]
    @test Dict(ep.events)["other_collision"] == [3]
    # a flag that never fired produces no marker rather than an empty one
    @test !haskey(Dict(ep.events), "goal")
    @test !haskey(Dict(ep.events), "passed_stop")

    # the outcome is read from the record
    @test ep.outcome === ENV_TERMINATED
    @test ep.reason == "other_collision"
    rm(path)
end

@testset "FJ9.6b horizon and environment termination are distinguished" begin
    open_path = fj96_synthetic(rows = [Dict("decision" => "1"),
        Dict("decision" => "2")])
    ep = episode_diagnostics(load_decision_log(open_path), "toy", 1)
    @test ep.outcome === HORIZON_REACHED
    @test ep.reason == "in_progress"

    dead = fj96_synthetic()
    @test episode_diagnostics(load_decision_log(dead), "toy", 1).outcome ===
        ENV_TERMINATED
    foreach(rm, [open_path, dead])
end

@testset "FJ9.6b the fingerprint is figure identity" begin
    path = fj96_synthetic()
    log = load_decision_log(path)
    ep = episode_diagnostics(log, "toy", 1)

    base = diagnostics_fingerprint(ep; fields = ["d", "phi"])
    @test base == diagnostics_fingerprint(ep; fields = ["phi", "d"])  # order
    @test base != diagnostics_fingerprint(ep; fields = ["d"])
    @test base != diagnostics_fingerprint(ep; fields = ["d", "phi"],
        mode = NORMALIZED_PROGRESS)
    @test any(l -> occursin(ep.source_fingerprint, l),
        diagnostics_provenance(ep))
    rm(path)
end

@testset "FJ9.6a negative control: one edited cell moves what it should" begin
    # change a single model_calls value. The computational series and the
    # source fingerprint must move; nothing else may.
    original = fj96_synthetic()
    edited = tempname() * ".csv"
    lines = readlines(original)
    k = findfirst(==("model_calls"), split(lines[1], ","))
    f = split(lines[2], ",")
    f[k] = "999"
    write(edited, join([lines[1], join(f, ","), lines[3], lines[4]], "\n"))

    a = episode_diagnostics(load_decision_log(original), "toy", 1)
    b = episode_diagnostics(load_decision_log(edited), "toy", 1)

    @test a.source_fingerprint != b.source_fingerprint
    @test diagnostics_fingerprint(a) != diagnostics_fingerprint(b)
    @test series_named(a, "model_calls").values !=
          series_named(b, "model_calls").values
    @test series_named(b, "model_calls").values[1] == 999.0
    for s in ("d", "phi", "v_cmd", "reward_total", "cumulative_return",
              "d_stop", "planning_time")
        @test isequal(series_named(a, s).values, series_named(b, s).values)
    end
    @test a.events == b.events
    @test a.outcome === b.outcome
    foreach(rm, [original, edited])
end

@testset "FJ9.6a the enriched artefact supports the contract" begin
    if !isfile(FJ96_LOG)
        @test_skip "FJ8.4c enrichment has not been run"
    else
        log = load_decision_log(FJ96_LOG)
        @test length(log) == 16_522
        @test length(log.episodes) == 120
        @test length(unique(first.(log.episodes))) == 6
        @test length(unique(last.(log.episodes))) == 20

        items = decision_log_audit(log)
        wanted = [it for it in items if it.quantity != "not recorded"]
        # every quantity the contract asks for is in the record
        @test all(it -> it.status !== FIELD_ABSENT, wanted)
        @test count(it -> it.status === DERIVED_IDENTITY, items) == 2
        # and what the experiment never stored is reported ABSENT, not filled
        absent = [it for it in items if it.status === FIELD_ABSENT]
        @test !isempty(absent)
        @test any(it -> occursin("belief", it.column), absent)
        @test any(it -> occursin("observation", it.column), absent)
        @test occursin("ABSENT", decision_audit_table(items))

        # the missingness the audit reports is d_stop's, and only d_stop's
        blanks = [it for it in items if it.status === LOGGED && it.blanks > 0]
        @test length(blanks) == 1
        @test only(blanks).column == "d_stop"
        @test only(blanks).blanks == 12_814

        @info "FJ9.6a decision-artifact audit" rows = length(log) episodes =
            length(log.episodes) logged = count(it -> it.status === LOGGED,
            items) derived = count(it -> it.status === DERIVED_IDENTITY,
            items) absent = length(absent)
    end
end

@testset "FJ9.6b the three FJ9.6c episodes are available" begin
    if !isfile(FJ96_LOG)
        @test_skip "FJ8.4c enrichment has not been run"
    else
        log = load_decision_log(FJ96_LOG)
        for solver in ("mcts@1k", "dpw@1k", "td3")
            lens = episode_lengths(log, solver)
            @test length(lens) == 20
            @test all(>(0), lens)
            ep = episode_diagnostics(log, solver, 1001)
            @test ep.solver == solver
            @test length(ep.decisions) == length(ep.progress)
            @test ep.decisions == collect(1:length(ep.decisions))
            @test all(0.0 .<= ep.progress .< 1.0)
            @test ep.outcome in (ENV_TERMINATED, HORIZON_REACHED)
        end

        # the compute collapse FJ8.4c measured is in the series, per decision
        dpw = episode_diagnostics(log, "dpw@1k", 1001)
        mcts = episode_diagnostics(log, "mcts@1k", 1001)
        @test maximum(skipmissing(series_named(mcts, "model_calls").values)) >
            900
        @test all(==(0.0), skipmissing(
            series_named(episode_diagnostics(log, "q_learning", 1001),
                "model_calls").values))

        # every episode either terminates or reaches the evaluator's horizon,
        # and in this artefact the horizon is a single value
        horizon = maximum(length(episode_diagnostics(log, s, sd).decisions)
                          for (s, sd) in log.episodes)
        @test horizon == 150
        for (s, sd) in log.episodes
            e = episode_diagnostics(log, s, sd)
            if e.outcome === HORIZON_REACHED
                @test length(e.decisions) == horizon
                @test e.reason == "in_progress"
            else
                @test length(e.decisions) < horizon
                @test e.reason != "in_progress"
            end
        end
        @info "FJ9.6b dpw compute" first_call =
            series_named(dpw, "model_calls").values[1] last_call =
            series_named(dpw, "model_calls").values[end] decisions =
            length(dpw.decisions)
    end
end

@testset "FJ9.6c the series expose what the aggregate could not" begin
    if !isfile(FJ96_LOG)
        @test_skip "FJ8.4c enrichment has not been run"
    else
        log = load_decision_log(FJ96_LOG)
        # TD3 stops at the stop line and never proceeds. FJ8.4b reported its
        # compliance as n/a, correctly, but the accompanying prose said it
        # "never reaches a stop sign" — the opposite of what happened. The
        # per-decision record settles it, so the claim is pinned here.
        stalled = 0
        for seed in 1001:1020
            ep = episode_diagnostics(log, "td3", seed)
            ev = Dict(ep.events)
            haskey(ev, "full_stop") && !haskey(ev, "passed_stop") &&
                (stalled += 1)
        end
        @test stalled == 20

        ep = episode_diagnostics(log, "td3", 1001)
        @test ep.outcome === HORIZON_REACHED
        @test only(Dict(ep.events)["full_stop"]) == 28
        # it is inside the stop zone for most of the episode, not absent from it
        @test count(==(1.0), series_named(ep, "sigma_stop").values) == 123
        # and it is stationary there
        @test all(<(0.05), skipmissing(series_named(ep, "v").values[60:end]))
        # the return is the stagnation penalty, not a collision
        @test series_named(ep, "cumulative_return").values[end] < -200

        # by contrast the tabular policies complete every encounter they meet.
        # Not every seed puts them at a sign — 8 of 20 do — and these counts
        # are the FJ8.4b denominators, reproduced from the per-decision log
        for s in ("q_learning", "sarsa")
            passed = count(seed -> haskey(
                Dict(episode_diagnostics(log, s, seed).events), "passed_stop"),
                1001:1020)
            stopped = count(seed -> haskey(
                Dict(episode_diagnostics(log, s, seed).events), "full_stop"),
                1001:1020)
            @test passed == 8
            @test stopped == 8
        end
        @info "FJ9.6c td3 stop-and-stall" episodes_stalled = stalled
    end
end

@testset "FJ9.6d progress bins aggregate without interpolating" begin
    if !isfile(FJ96_LOG)
        @test_skip "FJ8.4c enrichment has not been run"
    else
        log = load_decision_log(FJ96_LOG)
        b = progress_bins(log, "dpw@1k", "model_calls"; bins = 5)
        @test length(b) == 5
        @test [x.bin for x in b] == 1:5
        # every bin reports its own n; none is padded to a common length
        @test all(x -> x.n > 0, b)
        @test sum(x.n for x in b) == sum(episode_lengths(log, "dpw@1k"))
        @test all(x -> x.min <= x.q25 <= x.median <= x.q75 <= x.max, b)
        # the collapse survives binning
        @test b[1].mean > b[end].mean
        @test b[1].n > b[end].n          # fewer episodes reach the last bin

        # a column with missing values reports how many it dropped rather
        # than substituting zeros
        ds = progress_bins(log, "dpw@1k", "d_stop"; bins = 5)
        @test all(x -> x.absent > 0, ds)
        @test sum(x.n for x in ds) + first(ds).absent ==
            sum(episode_lengths(log, "dpw@1k"))

        # this binning was written independently of tools/enrich_decision_log.jl and
        # lands on the same five numbers FJ8.4c reported. Two implementations
        # agreeing is a check on both; one of them agreeing with itself is not
        @test [round(Int, x.mean) for x in b] == [1019, 835, 940, 602, 195]
        @test all(==(0), (round(Int, x.mean) for x in
            progress_bins(log, "q_learning", "model_calls"; bins = 5)))

        @test_throws ArgumentError progress_bins(log, "dpw@1k", "no_such_col")
        @test_throws ArgumentError progress_bins(log, "no_such_solver", "d")
        @info "FJ9.6d dpw model_calls by progress" bins =
            [(x.bin, round(x.mean; digits = 1), x.n) for x in b]
    end
end
