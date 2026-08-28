# FJ9.7 — artifact-driven animation.
#
# Animation is playback of recorded evidence. These tests check the two things
# that make that claim true rather than decorative: the frame at t exposes
# rows 1..t and nothing after, and one edited row changes exactly what it
# should. No environment, policy or planner runs here.

using DuckietownDecisionModels
using Test

const FJ97_LOG = joinpath(pkgdir(DuckietownDecisionModels), "artifacts", "fj8",
    "enriched", "decisions.csv")

# an edited copy of the real log, so the negative controls compare like with
# like instead of against a synthetic stand-in
function fj97_edit(col::AbstractString, solver::AbstractString, seed::Integer,
        decision::Integer, value::AbstractString)
    lines = readlines(FJ97_LOG)
    header = split(lines[1], ",")
    k = findfirst(==(col), header)
    si = findfirst(==("solver"), header)
    di = findfirst(==("seed"), header)
    ci = findfirst(==("decision"), header)
    out = [lines[1]]
    hits = 0
    for l in Iterators.drop(lines, 1)
        f = split(l, ",")
        if f[si] == solver && parse(Int, f[di]) == seed &&
           parse(Int, f[ci]) == decision
            f[k] = value
            hits += 1
        end
        push!(out, join(f, ","))
    end
    hits == 1 || error("edit matched $hits rows, expected 1")
    path = tempname() * ".csv"
    write(path, join(out, "\n") * "\n")
    return path
end

@testset "FJ9.7a the sequence is built from logged rows only" begin
    if !isfile(FJ97_LOG)
        @test_skip "FJ8.4c enrichment has not been run"
    else
        log = load_decision_log(FJ97_LOG)
        seq = animation_sequence(log, "td3", 1001)

        @test length(seq) == 150
        @test seq.horizon == 150
        @test seq.outcome === HORIZON_REACHED
        @test seq.timeline == ANIMATION_TIMELINE_LABEL
        @test seq.source_fingerprint == log.fingerprint
        @test [f.decision for f in seq.frames] == collect(1:150)
        @test all(!isempty(f.fingerprint) for f in seq.frames)
        @test length(unique(f.fingerprint for f in seq.frames)) == 150

        # a terminating episode has exactly as many frames as it has decisions
        dpw = animation_sequence(log, "dpw@1k", 1001)
        @test length(dpw) == 116
        @test dpw.horizon == 116
        @test dpw.outcome === ENV_TERMINATED
        @test dpw.reason == "offroad"

        # the cumulative return is the running sum, and its last value is the
        # episode return
        @test seq.frames[end].cumulative_return ≈
            sum(f.reward_total for f in seq.frames)

        @test_throws ArgumentError animation_sequence(log, "td3", 99)
    end
end

@testset "FJ9.7a the timeline never claims wall-clock time" begin
    if !isfile(FJ97_LOG)
        @test_skip "FJ8.4c enrichment has not been run"
    else
        seq = animation_sequence(load_decision_log(FJ97_LOG), "td3", 1001)
        @test occursin("Decision-index", ANIMATION_TIMELINE_LABEL)
        @test !occursin("real-time", lowercase(ANIMATION_TIMELINE_LABEL))
        @test any(l -> occursin("wall-clock", l), animation_absent_lines())

        # model time exists only as an exact identity, and only when the
        # caller supplies the timestep the protocol used
        @test model_time(seq, 1; frame_skip=6, dt=1 / 30) == 0.0
        @test model_time(seq, 11; frame_skip=6, dt=1 / 30) ≈ 2.0
        @test occursin("NOT recorded wall-clock", MODEL_TIME_LABEL)
        # planning_time must not be reachable as a pacing source
        @test !occursin("planning_time", MODEL_TIME_LABEL)
    end
end

@testset "FJ9.7b the frame never shows the future" begin
    if !isfile(FJ97_LOG)
        @test_skip "FJ8.4c enrichment has not been run"
    else
        log = load_decision_log(FJ97_LOG)
        seq = animation_sequence(log, "td3", 1001)

        # the trajectory grows one point per frame and stops at the frame
        for t in (1, 2, 20, 75, 150)
            @test length(trajectory_through(seq, t)) == t
            @test trajectory_through(seq, t)[end] == seq.frames[t].ego_position
        end
        @test trajectory_through(seq, 20) ==
            trajectory_through(seq, 150)[1:20]
        @test_throws BoundsError trajectory_through(seq, 151)
        @test_throws BoundsError trajectory_through(seq, 0)

        # every series is a prefix, never the whole episode
        for nm in ("d", "phi", "model_calls", "cumulative_return")
            @test length(series_through(seq, 33, nm)) == 33
            @test isequal(series_through(seq, 33, nm),
                series_through(seq, 150, nm)[1:33])
        end
        @test_throws ArgumentError series_through(seq, 5, "no_such_series")
    end
end

@testset "FJ9.7b an event appears at its frame and not before" begin
    if !isfile(FJ97_LOG)
        @test_skip "FJ8.4c enrichment has not been run"
    else
        seq = animation_sequence(load_decision_log(FJ97_LOG), "td3", 1001)
        # td3 seed 1001 logs exactly one event: full_stop at decision 28
        @test isempty(events_through(seq, 27))
        @test events_through(seq, 28) == ["full_stop" => 28]
        @test events_through(seq, 150) == ["full_stop" => 28]
        for t in 1:27
            @test isempty(events_through(seq, t))
        end
        # the caption follows the same rule
        @test !any(l -> occursin("full_stop", l), frame_caption(seq, 27))
        @test any(l -> occursin("full_stop", l), frame_caption(seq, 28))
    end
end

@testset "FJ9.7b missing d_stop stays missing in playback" begin
    if !isfile(FJ97_LOG)
        @test_skip "FJ8.4c enrichment has not been run"
    else
        log = load_decision_log(FJ97_LOG)
        seq = animation_sequence(log, "td3", 1001)
        early = series_through(seq, 5, "d_stop")
        @test all(ismissing, early)
        @test !any(x -> x === 0.0, early)
        @test ismissing(seq.frames[1].d_stop)
        # and it becomes a real value once the sign is a candidate
        late = series_through(seq, 150, "d_stop")
        @test any(!ismissing, late)
        @test count(ismissing, late) > 0
    end
end

@testset "FJ9.7b horizon and termination are visually distinct" begin
    if !isfile(FJ97_LOG)
        @test_skip "FJ8.4c enrichment has not been run"
    else
        log = load_decision_log(FJ97_LOG)
        hz = animation_sequence(log, "td3", 1001)
        tm = animation_sequence(log, "dpw@1k", 1001)
        @test any(l -> occursin("HORIZON REACHED", l),
            frame_caption(hz, length(hz)))
        @test any(l -> occursin("TERMINATED", l),
            frame_caption(tm, length(tm)))
        # and the banner appears only on the final frame
        @test !any(l -> occursin("TERMINATED", l), frame_caption(tm, 50))
    end
end

@testset "FJ9.7c selection rules are stated, not chosen by eye" begin
    if !isfile(FJ97_LOG)
        @test_skip "FJ8.4c enrichment has not been run"
    else
        log = load_decision_log(FJ97_LOG)

        st = select_episode(log, "td3"; rule=:first_stagnation)
        @test st.seed == 1001
        @test occursin("never passes", st.justification)

        dp = select_episode(log, "dpw@1k"; rule=:median_length_terminating)
        @test dp.rule === :median_length_terminating
        @test occursin("terminating", dp.justification)
        @test animation_sequence(log, "dpw@1k", dp.seed).outcome ===
            ENV_TERMINATED

        # the median-return rule reproduces the FJ8.4b median column exactly,
        # from the per-decision log — two independent paths to one number
        mc = select_episode(log, "mcts@1k"; rule=:median_return)
        @test round(animation_sequence(log, "mcts@1k",
            mc.seed).frames[end].cumulative_return; digits=2) == 7.75
        dpm = select_episode(log, "dpw@1k"; rule=:median_return)
        @test round(animation_sequence(log, "dpw@1k",
            dpm.seed).frames[end].cumulative_return; digits=2) == -211.59

        # solvers that never terminate have no terminating median, and say so
        @test_throws ArgumentError select_episode(log, "mcts@1k";
            rule=:median_length_terminating)
        @test_throws ArgumentError select_episode(log, "sac";
            rule=:first_stagnation)
        @test_throws ArgumentError select_episode(log, "td3"; rule=:nonsense)

        @info "FJ9.7c canonical selections" stagnation=st.seed dpw=dp.seed mcts=mc.seed
    end
end

@testset "FJ9.7d pairing is by absolute decision, with a freeze" begin
    if !isfile(FJ97_LOG)
        @test_skip "FJ8.4c enrichment has not been run"
    else
        log = load_decision_log(FJ97_LOG)
        a = animation_sequence(log, "td3", 1001)        # 150, horizon
        b = animation_sequence(log, "dpw@1k", 1001)     # 116, terminated

        @test paired_frames(a, b) == 150
        # while both run, absolute decision t is frame t in both panels
        for t in (1, 50, 116)
            @test frame_index(a, t) == t
            @test frame_index(b, t) == t
            @test !is_frozen(b, t)
        end
        # after the shorter one ends its panel holds the terminal frame
        for t in (117, 130, 150)
            @test frame_index(a, t) == t
            @test frame_index(b, t) == 116
            @test is_frozen(b, t)
            @test !is_frozen(a, t)
        end
        # no looping, no restart: the held frame is the last one, every time
        @test frame_index(b, 150) == length(b)
        @test trajectory_through(b, frame_index(b, 150)) ==
            trajectory_through(b, 116)
    end
end

@testset "FJ9.7 negative control 1: edit an ego position" begin
    if !isfile(FJ97_LOG)
        @test_skip "FJ8.4c enrichment has not been run"
    else
        base = animation_sequence(load_decision_log(FJ97_LOG), "td3", 1001)
        p = fj97_edit("ego_x", "td3", 1001, 20, "0.4242")
        ed = animation_sequence(load_decision_log(p), "td3", 1001)

        @test ed.frames[20].ego_position[1] == 0.4242
        @test ed.fingerprint != base.fingerprint
        @test ed.source_fingerprint != base.source_fingerprint
        # the trajectory changes from frame 20 on, and not before
        @test trajectory_through(ed, 19) == trajectory_through(base, 19)
        @test trajectory_through(ed, 20) != trajectory_through(base, 20)
        # nothing else moves
        for nm in ("d", "phi", "reward_total", "model_calls", "planning_time")
            @test isequal(series_through(ed, 150, nm),
                series_through(base, 150, nm))
        end
        @test events_through(ed, 150) == events_through(base, 150)
        rm(p)
    end
end

@testset "FJ9.7 negative control 2: shift an event by one decision" begin
    if !isfile(FJ97_LOG)
        @test_skip "FJ8.4c enrichment has not been run"
    else
        base = animation_sequence(load_decision_log(FJ97_LOG), "td3", 1001)
        @test events_through(base, 150) == ["full_stop" => 28]

        off = fj97_edit("full_stop", "td3", 1001, 28, "false")
        on = fj97_edit("full_stop", "td3", 1001, 29, "true")
        # apply both edits to one file
        lines = readlines(on)
        header = split(lines[1], ",")
        k = findfirst(==("full_stop"), header)
        ci = findfirst(==("decision"), header)
        si = findfirst(==("solver"), header)
        di = findfirst(==("seed"), header)
        out = [lines[1]]
        for l in Iterators.drop(lines, 1)
            f = split(l, ",")
            if f[si] == "td3" && parse(Int, f[di]) == 1001 &&
               parse(Int, f[ci]) == 28
                f[k] = "false"
            end
            push!(out, join(f, ","))
        end
        both = tempname() * ".csv"
        write(both, join(out, "\n") * "\n")

        ed = animation_sequence(load_decision_log(both), "td3", 1001)
        @test events_through(ed, 150) == ["full_stop" => 29]
        # the marker moves by exactly one frame
        @test isempty(events_through(ed, 28))
        @test events_through(ed, 29) == ["full_stop" => 29]
        @test ed.fingerprint != base.fingerprint
        # and nothing else about the episode changed
        for nm in ("d", "phi", "model_calls", "cumulative_return")
            @test isequal(series_through(ed, 150, nm),
                series_through(base, 150, nm))
        end
        @test [f.ego_position for f in ed.frames] ==
            [f.ego_position for f in base.frames]
        foreach(rm, [off, on, both])
    end
end

@testset "FJ9.7 negative control 3: edit model_calls" begin
    if !isfile(FJ97_LOG)
        @test_skip "FJ8.4c enrichment has not been run"
    else
        base = animation_sequence(load_decision_log(FJ97_LOG), "dpw@1k", 1001)
        p = fj97_edit("model_calls", "dpw@1k", 1001, 40, "7777")
        ed = animation_sequence(load_decision_log(p), "dpw@1k", 1001)

        @test series_through(ed, 150 |> t -> min(t, length(ed)),
            "model_calls")[40] == 7777.0
        @test ed.fingerprint != base.fingerprint
        # the computational panel is the only one that moves
        for nm in ("d", "phi", "v_cmd", "omega_cmd", "reward_total",
                   "cumulative_return", "d_stop", "planning_time")
            @test isequal(series_through(ed, length(ed), nm),
                series_through(base, length(base), nm))
        end
        @test [f.ego_position for f in ed.frames] ==
            [f.ego_position for f in base.frames]
        @test events_through(ed, length(ed)) ==
            events_through(base, length(base))
        rm(p)
    end
end

@testset "FJ9.7 negative control 4: turn a missing d_stop into zero" begin
    if !isfile(FJ97_LOG)
        @test_skip "FJ8.4c enrichment has not been run"
    else
        base = animation_sequence(load_decision_log(FJ97_LOG), "td3", 1001)
        @test ismissing(base.frames[3].d_stop)

        p = fj97_edit("d_stop", "td3", 1001, 3, "0.0")
        ed = animation_sequence(load_decision_log(p), "td3", 1001)

        # the semantics change: what was "no candidate" is now "distance zero"
        @test ed.frames[3].d_stop === 0.0
        @test !ismissing(ed.frames[3].d_stop)
        @test count(ismissing, series_through(ed, length(ed), "d_stop")) ==
            count(ismissing, series_through(base, length(base), "d_stop")) - 1
        @test ed.fingerprint != base.fingerprint
        # and only that column moves
        for nm in ("d", "phi", "v_cmd", "model_calls", "cumulative_return")
            @test isequal(series_through(ed, length(ed), nm),
                series_through(base, length(base), nm))
        end
        rm(p)
    end
end

@testset "FJ9.7 the absent list is reported, not reconstructed" begin
    lines = animation_absent_lines()
    @test length(lines) == length(ANIMATION_ABSENT)
    @test all(l -> startswith(l, "ABSENT — "), lines)
    for q in ("duck world position", "wall-clock timestamp", "belief state",
              "per-decision search tree")
        @test any(l -> occursin(q, l), lines)
    end
    # the reason duck world position is absent must name the lane frame, so a
    # future reader does not "fix" it by drawing the reset pose
    @test any(l -> occursin("duck world position", l) &&
                   occursin("lane frame", l), lines)
end
