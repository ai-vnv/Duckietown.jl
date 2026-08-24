# FJ9.9 — reproducibility closure.
#
# Every check here is executable and self-invalidating. A documentation rule
# that lives in a checklist goes stale the first time someone edits a file; one
# that runs here fails.

using DuckietownDecisionModels
using POMDPs
using Test
using JSON3

const FJ99_ROOT = pkgdir(DuckietownDecisionModels)
const FJ99_CFG = joinpath(FJ99_ROOT, "..", "duckduck", "policies",
    "q_learning", "training_config.yaml")

@testset "FJ9.9c the artifact ledger distinguishes kinds of evidence" begin
    led = artifact_ledger(FJ99_ROOT)
    @test length(led) >= 12
    @test all(a -> a.status isa ArtifactStatus, led)

    # the recorded experiment is PERSISTED_SOURCE and must never be "rebuilt":
    # re-running it would be a different experiment wearing its name
    src = [a for a in led if a.status === PERSISTED_SOURCE]
    @test any(a -> occursin("six_solver_episodes", a.path), src)
    @test any(a -> occursin("decisions.csv", a.path), src)
    @test any(a -> occursin("search_snapshot", a.path), src)

    # the actor weights are provisioned from a read-only checkpoint
    prov = [a for a in led if a.status === PROVISIONED_FROZEN_INPUT]
    @test !isempty(prov)
    @test all(a -> occursin("weights", a.path), prov)

    # figures are the only things expected to regenerate
    reb = [a for a in led if a.status === REBUILT]
    @test any(a -> occursin("figure", a.path), reb)

    if all(a -> a.present, led)
        @test all(a -> !isempty(a.fingerprint), led)
        @test all(a -> a.bytes > 0, led)
        @info "FJ9.9c ledger" total=length(led) persisted=length(src) provisioned=length(prov) rebuilt=length(reb)
    else
        @test_skip "some artifacts have not been built in this checkout"
    end
end

@testset "FJ9.9d the documentation audit finds no stale claim" begin
    issues = documentation_audit(FJ99_ROOT)
    stale = [i for i in issues if i.kind === :stale_claim]
    dead = [i for i in issues if i.kind === :dead_link]
    missing_ = [i for i in issues if i.kind === :missing_path]

    @test isempty(stale)
    @test isempty(dead)
    @test isempty(missing_)
    isempty(issues) || @info "FJ9.9d issues" issues

    # the guard itself must still be armed: the audit has to be capable of
    # catching the claim, not merely silent about it
    @test length(STALE_CLAIMS) >= 2
    @test any(sc -> occursin("stop sign", sc.claim), STALE_CLAIMS)
    for sc in STALE_CLAIMS
        @test !isempty(sc.correction)
        @test !isempty(sc.allow)
    end

    # a synthetic offending file is detected
    tmp = mktempdir()
    write(joinpath(tmp, "bad.md"), "TD3 never reaches a stop sign at all.\n")
    bad = documentation_audit(tmp)
    @test any(i -> i.kind === :stale_claim, bad)
    # and a dead link is detected
    write(joinpath(tmp, "link.md"), "see [x](does/not/exist.md)\n")
    @test any(i -> i.kind === :dead_link, documentation_audit(tmp))
    rm(tmp; recursive=true)
end

@testset "FJ9.9e optional packages do not alter the core formulation" begin
    if !isfile(FJ99_CFG)
        @test_skip "reference config unavailable"
    else
        mdp = DuckietownMDP(FJ99_CFG; action_space=:discrete)
        fp = core_fingerprint(mdp)
        @test !isempty(fp)
        @test length(fp) == 16
        # deterministic, and independent of whatever is loaded in this session
        @test core_fingerprint(DuckietownMDP(FJ99_CFG; action_space=:discrete)) == fp
        # an InstrumentedMDP is a counter, not a different model
        @test core_fingerprint(InstrumentedMDP(mdp)) == fp

        # a different formulation IS a different fingerprint
        cont = DuckietownMDP(FJ99_CFG; action_space=:continuous)
        @test core_fingerprint(cont) != fp

        # the full matrix is measured by tools/run_manifest.sh; if it has run,
        # every configuration must agree
        man = joinpath(FJ99_ROOT, "artifacts", "fj9",
            "reproducibility_manifest.json")
        if isfile(man)
            m = JSON3.read(read(man, String))
            @test m.isolation_all_match == true
            @test length(m.isolation_matrix) >= 3
            @test all(r -> r.core_fingerprint == m.core_fingerprint,
                m.isolation_matrix)
        end
    end
end

@testset "FJ9.9e src/ imports no optional backend" begin
    issues = source_import_audit(FJ99_ROOT)
    @test isempty(issues)
    isempty(issues) || @info "FJ9.9e banned imports" issues

    @test "PythonCall" in SOURCE_IMPORT_BAN
    @test "MCTS" in SOURCE_IMPORT_BAN
    @test "Makie" in SOURCE_IMPORT_BAN

    # the lint must still fire on real code while ignoring a docstring example
    tmp = mktempdir()
    mkpath(joinpath(tmp, "src"))
    write(joinpath(tmp, "src", "a.jl"), "using MCTS\n")
    @test length(source_import_audit(tmp)) == 1
    write(joinpath(tmp, "src", "a.jl"),
        "\"\"\"\nExample:\n\n```julia\nusing MCTS\n```\n\"\"\"\nf() = 1\n")
    @test isempty(source_import_audit(tmp))
    rm(tmp; recursive=true)
end

@testset "FJ9.9f known limitations are recorded, not hidden" begin
    @test length(KNOWN_LIMITATIONS) >= 5
    text = join((k * " " * v for (k, v) in KNOWN_LIMITATIONS), " ")
    # each deferred decision the project actually took must be named
    for topic in ("controller_rng", "Duckiematrix", "belief", "gen")
        @test occursin(topic, text)
    end
    @test all(kv -> !isempty(last(kv)), KNOWN_LIMITATIONS)
end

@testset "FJ9.9f the manifest is complete when it has been built" begin
    man = joinpath(FJ99_ROOT, "artifacts", "fj9",
        "reproducibility_manifest.json")
    if !isfile(man)
        @test_skip "manifest has not been generated"
    else
        m = JSON3.read(read(man, String))
        @test m.schema == "fj99.manifest.1"
        @test !isempty(m.julia_version)
        @test !isempty(m.core_fingerprint)
        @test isempty(m.documentation_issues)
        @test isempty(m.source_import_issues)
        @test length(m.known_limitations) >= 5
        @test !isempty(m.artifacts)
        @test all(a -> a.present, m.artifacts)

        # the captured searches are recorded with their validity
        if haskey(m, :search_snapshots) && !isempty(m.search_snapshots)
            for (_, v) in pairs(m.search_snapshots)
                @test v.valid == true
                @test !isempty(v.state)
            end
        end
        @info "FJ9.9f manifest" artifacts=length(m.artifacts) limitations=length(m.known_limitations)
    end
end

@testset "FJ9.9b the structured report agrees with the parsers" begin
    rp = joinpath(FJ99_ROOT, "artifacts", "fj9", "test_report.json")
    if !isfile(rp)
        @test_skip "no structured report from a previous full run"
    else
        r = JSON3.read(read(rp, String))
        @test r.schema == "fj99.reporter.1"
        @test r.testsets > 100
        @test r.assertions > 100_000

        # Deliberately NOT asserted here: failures == 0. This file is written
        # at the END of a run, so the copy on disk while this testset executes
        # belongs to the PREVIOUS run. Asserting a pass/fail verdict from a
        # different run is the same mistake as a renderer reporting data the
        # experiment never produced. The current run's verdict is PKGTEST_EXIT.
        #
        # What IS checkable here is the report's internal consistency, which
        # is what caught the reporter's own first bug: it read `n_passed` only
        # at the top level and reported 88 213 against the parsers' 148 758.
        @test sum(t -> t.passes, r.testsets_detail) == r.assertions
        @test sum(t -> t.fails, r.testsets_detail) == r.failures
        @test sum(t -> t.errors, r.testsets_detail) == r.errors
        @test length(r.testsets_detail) == r.testsets
        @test all(t -> !isempty(t.name), r.testsets_detail)
        @test all(t -> t.passes >= 0, r.testsets_detail)
    end
end
