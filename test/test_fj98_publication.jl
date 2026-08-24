# FJ9.8 — publication composites.
#
# The point of this gate is that a caption is data, not decoration. FJ9.6
# found a false sentence living beside correct numbers for weeks; these tests
# make that specific failure unrepresentable, and check the machinery that
# generalises it.

using DuckietownDecisionModels
using Test

const FJ98_ROOT = pkgdir(DuckietownDecisionModels)
const FJ98_LOG = joinpath(FJ98_ROOT, "artifacts", "fj8", "enriched",
    "decisions.csv")

@testset "FJ9.8a the inventory classifies before any layout is chosen" begin
    inv = publication_inventory(FJ98_ROOT)
    @test length(inv) >= 15
    @test all(a -> a.role isa FigureRole, inv)

    # every main figure is assigned to a numbered figure, not left loose
    for a in inv
        a.role === MAIN_FIGURE && @test startswith(a.figure, "Figure ")
    end
    # videos are supplementary and never main: a printed paper must stand alone
    vids = [a for a in inv if occursin("video", a.id)]
    @test length(vids) == 4
    @test all(a -> a.role === SUPPLEMENTARY, vids)
    @test all(a -> startswith(a.figure, "Supplementary"), vids)

    # availability is probed, not asserted
    @test all(a -> a.available isa Bool, inv)
    fake = [PublicationArtifact("x", "y", MAIN_FIGURE, "Figure 9",
        "artifacts/does_not_exist.png", false)]
    @test occursin("**no**", inventory_table(fake))
    @test occursin("| Artifact |", inventory_table(inv))
end

@testset "FJ9.8b layout and wrapping are core concerns" begin
    @test grid_layout(4) == [(1, 1, 1, 1), (1, 2, 1, 1), (2, 1, 1, 1),
        (2, 2, 1, 1)]
    @test grid_layout(3; columns=3) == [(1, 1, 1, 1), (1, 2, 1, 1),
        (1, 3, 1, 1)]

    # wrapping happens on word boundaries and loses nothing
    long = join(fill("word", 60), " ")
    w = wrap_text(long, 40)
    @test all(l -> length(l) <= 40, split(w, "\n"))
    @test replace(w, "\n" => " ") == long
    @test wrap_text("a\nb", 40) == "a\nb"
end

@testset "FJ9.8d a caption is checked, not trusted" begin
    # the rules themselves must forbid the sentence FJ9.6 had to correct
    r4 = caption_rule("Figure 4")
    @test "never reaches a stop sign" in r4.forbidden
    @test "reaches the stop sign" in r4.required
    @test "passed_stops = 0" in r4.required
    @test "combined score" in r4.forbidden      # FJ8.4b's no-ranking rule

    r2 = caption_rule("Figure 2")
    @test "not guaranteed reachable" in r2.required
    @test "feature space" in r2.required
    r1 = caption_rule("Figure 1")
    @test "not an observation" in r1.required

    # a hand-made composite carrying the OLD false claim must fail
    bad = PublicationComposite("Figure 4", "t", PanelSpec[], grid_layout(0),
        "TD3 never reaches a stop sign, so compliance is n/a.",
        Pair{String,String}[], "0")
    chk = check_caption(bad)
    @test !chk.ok
    @test "never reaches a stop sign" in chk.present
    @test !isempty(chk.absent)

    # a caption that omits a required qualifier also fails
    thin = PublicationComposite("Figure 2", "t", PanelSpec[], grid_layout(0),
        "Slices of the Q-table over d and phi.", Pair{String,String}[], "0")
    @test !check_caption(thin).ok
    @test "feature space" in check_caption(thin).absent
end

@testset "FJ9.8c Figure 3 refuses to compare two different states" begin
    if !isfile(joinpath(FJ98_ROOT, "artifacts", "fj9",
            "search_snapshot_mcts.json"))
        @test_skip "search snapshots have not been captured"
    else
        m = load_snapshot(joinpath(FJ98_ROOT, "artifacts", "fj9",
            "search_snapshot_mcts.json"))
        d = load_snapshot(joinpath(FJ98_ROOT, "artifacts", "fj9",
            "search_snapshot_dpw.json"))
        @test m.state_fingerprint == d.state_fingerprint

        f = figure_search(m, d)
        @test f.figure_id == "Figure 3"
        @test check_caption(f).ok
        @test panel_ids(f) == ["3A", "3B", "3C", "3D"]
        # the shared state is in the caption AND in the metadata
        @test occursin(m.state_fingerprint, f.caption)
        @test any(kv -> last(kv) == m.state_fingerprint, f.metadata)

        # the numbers in the caption are the measured ones
        sm = search_statistics(m)
        sd = search_statistics(d)
        @test occursin("$(sm.root_actions) distinct", f.caption)
        @test occursin(string(sd.root_actions), f.caption)
        @test occursin("max $(sm.max_visits)", f.caption)

        # comparing searches from different states is refused outright
        other = SearchSnapshot(d.id, d.solver, "deadbeefdeadbeef",
            d.config_fingerprint, d.planner_seed, d.root_visits,
            d.selected_action, d.nodes, d.extra)
        @test_throws ArgumentError figure_search(m, other)
    end
end

@testset "FJ9.8c Figure 4 carries the FJ9.6 correction" begin
    if !isfile(FJ98_LOG)
        @test_skip "FJ8.4c enrichment has not been run"
    else
        log = load_decision_log(FJ98_LOG)
        td3 = episode_diagnostics(log, "td3", 1001)
        dpw = episode_diagnostics(log, "dpw@1k", 1002)
        bins = progress_bins(log, "dpw@1k", "model_calls"; bins=5)
        sels = [select_episode(log, "td3"; rule=:first_stagnation),
            select_episode(log, "dpw@1k"; rule=:median_length_terminating)]

        f = figure_episode(["a", "b"], td3, dpw, bins, sels)
        @test f.figure_id == "Figure 4"
        chk = check_caption(f)
        @test chk.ok
        @test isempty(chk.present)

        # the corrected narrative, in the caption, from the data
        @test occursin("reaches the stop sign", f.caption)
        @test occursin("full stop", f.caption)
        @test occursin("never proceeds", f.caption)
        @test occursin("horizon", lowercase(f.caption))
        @test occursin("passed_stops = 0", f.caption)
        @test !occursin("never reaches", f.caption)

        # and the numbers are the measured ones, not literals in the prose
        @test occursin("decision 28", f.caption)
        @test occursin("123 of 150", f.caption)
        @test occursin("1019 → 835 → 940 → 602 → 195", f.caption)

        # the selection rule reaches the reader
        @test occursin("first_stagnation", f.caption)
        @test occursin("median_length_terminating", f.caption)
        @test any(kv -> first(kv) == "selection_rules", f.metadata)

        # provenance ties the figure to the artefact
        @test any(kv -> last(kv) == td3.source_fingerprint, f.metadata)
        @test occursin("4B.stop_zone_decisions = 123", provenance_block(f))
        @test occursin("layout_version", provenance_block(f))

        # the layout gives the wide table its own full-width row
        @test f.layout[1] == (1, 1, 1, 2)
        @test length(f.layout) == length(f.panels)
    end
end

@testset "FJ9.8 the figure fingerprint tracks its content" begin
    if !isfile(FJ98_LOG)
        @test_skip "FJ8.4c enrichment has not been run"
    else
        log = load_decision_log(FJ98_LOG)
        td3 = episode_diagnostics(log, "td3", 1001)
        dpw = episode_diagnostics(log, "dpw@1k", 1002)
        bins = progress_bins(log, "dpw@1k", "model_calls"; bins=5)
        sels = [select_episode(log, "td3"; rule=:first_stagnation)]

        a = figure_episode(["a"], td3, dpw, bins, sels)
        b = figure_episode(["a"], td3, dpw, bins, sels)
        @test a.fingerprint == b.fingerprint          # deterministic

        # a different episode is a different figure
        c = figure_episode(["a"], td3,
            episode_diagnostics(log, "dpw@1k", 1001), bins, sels)
        @test c.fingerprint != a.fingerprint
        # so is a different binning
        d = figure_episode(["a"], td3, dpw,
            progress_bins(log, "dpw@1k", "model_calls"; bins=4), sels)
        @test d.fingerprint != a.fingerprint
    end
end

@testset "FJ9.8e the exported artefacts exist in vector and raster form" begin
    out = joinpath(FJ98_ROOT, "artifacts", "fj9", "publication")
    if !isdir(out)
        @test_skip "publication build has not been run"
    else
        for stem in ("figure1", "figure2", "figure3", "figure4")
            for ext in ("pdf", "svg", "png")
                p = joinpath(out, "$stem.$ext")
                @test isfile(p)
                @test filesize(p) > 5_000
            end
            cap = joinpath(out, "$stem.caption.txt")
            @test isfile(cap)
            txt = read(cap, String)
            @test occursin("figure_id", txt)
            @test occursin("layout_version", txt)
        end
        # the corrected claim survives into the exported caption file
        f4 = read(joinpath(out, "figure4.caption.txt"), String)
        @test occursin("reaches the stop sign", f4)
        @test !occursin("never reaches a stop sign", f4)
        @test isfile(joinpath(out, "inventory.md"))

        @info "FJ9.8e exports" pdf=filesize(joinpath(out, "figure4.pdf")) svg=filesize(joinpath(out, "figure4.svg"))
    end
end
