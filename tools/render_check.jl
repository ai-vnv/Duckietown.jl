# FJ9.9 (started early, because it is the cheapest way to keep FJ9 honest) —
# render in a FRESH process with nothing but CairoMakie: no Python, no Gym,
# no planning library. If this exits 0 the figures are reproducible on a
# headless machine, which is the only kind that matters for CI and papers.
#
#     tools/run_render_check.sh

using DuckietownDecisionModels
using POMDPs
using Random
using CairoMakie
CairoMakie.activate!(type="png")

const ROOT = pkgdir(DuckietownDecisionModels)
const OUT = joinpath(ROOT, "artifacts", "fj9")
mkpath(OUT)

cfg = joinpath(ROOT, "..", "duckduck", "policies", "q_learning",
    "training_config.yaml")
mdp = DuckietownMDP(cfg; action_space=:discrete)
s = rand(MersenneTwister(11), initialstate(mdp))

# A short ground track so the trajectory overlay is exercised. Wrapped in a
# function: a bare top-level loop puts `cur` in soft scope and Julia treats it
# as a fresh local, which is the same trap tools/native_gen_check.jl hit.
function ground_track(mdp, s0, policy, n)
    states = DuckieWorldState[s0]
    rng = MersenneTwister(4)
    cur = s0
    for _ in 1:n
        a = policy === nothing ? FAST_STRAIGHT : policy_action(policy, mdp, cur)
        r = simulate_decision(mdp.transition, cur, a, rng)
        cur = r.sp
        push!(states, cur)
        (r.terminated || r.truncated) && break
    end
    return states
end

qpath = joinpath(ROOT, "..", "duckduck", "policies", "q_learning", "policy.npy")
pol = isfile(qpath) ? QTablePolicy(qpath; solver=:q_learning) : nothing
states = ground_track(mdp, s, pol, 80)

ext = Base.get_extension(DuckietownDecisionModels, :DuckietownMakieExt)
println("EXT_LOADED=", ext !== nothing)
println("RENDER_WORLD_METHODS=", length(methods(render_world)))

fig = render_world(mdp, states[end]; trajectory=trajectory_points(states),
    title="small_loop — Q-learning ground track")
p1 = joinpath(OUT, "world.png")
save(p1, fig)
println("WORLD_PNG=", isfile(p1) && filesize(p1) > 5_000)

# vector output too: paper figures need it, and it exercises a different
# CairoMakie path
p2 = joinpath(OUT, "world.svg")
CairoMakie.activate!(type="svg")
save(p2, render_world(mdp, states[end]; trajectory=trajectory_points(states)))
CairoMakie.activate!(type="png")
println("WORLD_SVG=", isfile(p2) && filesize(p2) > 5_000)

raw, _ = get_raw_state(states[end], mdp.transition.state_cfg)
cont = get_continuous_state(states[end], raw, mdp.transition.state_cfg,
    mdp.transition.continuous_cfg; controller_cfg=mdp.transition.duck_cfg)
p3 = joinpath(OUT, "projection.png")
save(p3, render_projection(raw, cont))
println("PROJECTION_PNG=", isfile(p3) && filesize(p3) > 5_000)

art = joinpath(ROOT, "artifacts", "fj8", "six_solver_episodes.csv")
if isfile(art)
    agg = load_rollout_artifact(art)
    p4 = joinpath(OUT, "rollout_comparison.png")
    save(p4, render_rollout(agg))
    println("ROLLOUT_PNG=", isfile(p4) && filesize(p4) > 5_000)
else
    println("ROLLOUT_PNG=skipped")
end

snaps = [joinpath(OUT, f) for f in
    ("search_snapshot_mcts.json", "search_snapshot_dpw.json")]
if all(isfile, snaps)
    ok = all(s -> check_snapshot(load_snapshot(s)).ok, snaps)
    println("SNAPSHOTS_LOADED=", ok)
else
    println("SNAPSHOTS_LOADED=skipped")
end

if all(isfile, snaps)
    m = load_snapshot(snaps[1])
    d = load_snapshot(snaps[2])
    save(joinpath(OUT, "search_tree_mcts.png"), render_search(m; max_depth=2))
    save(joinpath(OUT, "search_tree_dpw.png"), render_search(d; max_depth=2))
    save(joinpath(OUT, "search_action_plane_dpw.png"),
        render_search_action_plane(d))
    println("SEARCH_TREE_PNG=",
        isfile(joinpath(OUT, "search_tree_mcts.png")) &&
        filesize(joinpath(OUT, "search_tree_mcts.png")) > 5_000)
    println("ACTION_PLANE_PNG=",
        isfile(joinpath(OUT, "search_action_plane_dpw.png")) &&
        filesize(joinpath(OUT, "search_action_plane_dpw.png")) > 5_000)
else
    println("SEARCH_TREE_PNG=skipped")
    println("ACTION_PLANE_PNG=skipped")
end

# FJ9.6c/d — diagnostic time series. Read from the frozen decision log; no
# environment, policy or planner is touched anywhere below.
dlog = joinpath(ROOT, "artifacts", "fj8", "enriched", "decisions.csv")

# accumulate inside a function — a bare top-level loop puts the accumulator in
# soft scope and Julia makes it a fresh local (tools/enrich_decision_log.jl, FJ8.4c)
function render_episode_diagnostics(log, picks, out)
    lines = String[]
    for (solver, seed) in picks
        ep = episode_diagnostics(log, solver, seed)
        tag = replace(solver, "@" => "", "_" => "")
        p = joinpath(out, "diagnostics_$(tag)_$(seed).png")
        save(p, render_diagnostics(ep))
        push!(lines, "DIAG_$(uppercase(tag))_$(seed)=" *
            string(isfile(p) && filesize(p) > 5_000))
    end
    return lines
end

if isfile(dlog)
    dl = load_decision_log(dlog)
    println("DECISION_LOG_ROWS=", length(dl))
    println("DECISION_LOG_FP=", dl.fingerprint)

    audit = decision_log_audit(dl)
    write(joinpath(OUT, "decision_contract.md"),
        "# FJ9.6a — decision-artifact contract\n\n" *
        "Source: `artifacts/fj8/enriched/decisions.csv`, fingerprint " *
        "`$(dl.fingerprint)`, $(length(dl)) decisions over " *
        "$(length(dl.episodes)) episodes.\n\n" * decision_audit_table(audit))
    println("CONTRACT_MD=", isfile(joinpath(OUT, "decision_contract.md")))
    println("CONTRACT_LOGGED=", count(it -> it.status === LOGGED, audit))
    println("CONTRACT_ABSENT=", count(it -> it.status === FIELD_ABSENT, audit))

    # a planner that survives, a planner whose compute collapses, a learned
    # continuous policy, and a tabular one — the last both exercises the flag
    # shading (dpw@1k seed 1001 never enters the stop zone, so its flags are
    # legitimately blank) and shows model_calls = 0 as a measured zero
    for line in render_episode_diagnostics(dl,
            [("mcts@1k", 1001), ("dpw@1k", 1001), ("td3", 1001),
             ("q_learning", 1001)], OUT)
        println(line)
    end

    ep = episode_diagnostics(dl, "dpw@1k", 1001)
    psvg = joinpath(OUT, "diagnostics_dpw1k_1001.svg")
    CairoMakie.activate!(type="svg")
    save(psvg, render_diagnostics(ep))
    CairoMakie.activate!(type="png")
    println("DIAG_SVG=", isfile(psvg) && filesize(psvg) > 5_000)

    pnorm = joinpath(OUT, "diagnostics_dpw1k_1001_progress.png")
    save(pnorm, render_diagnostics(ep; mode=NORMALIZED_PROGRESS))
    println("DIAG_PROGRESS_PNG=", isfile(pnorm) && filesize(pnorm) > 5_000)
    println("DIAG_MODE_FP_DIFFERS=",
        diagnostics_fingerprint(ep; mode=ABSOLUTE_DECISION) !=
        diagnostics_fingerprint(ep; mode=NORMALIZED_PROGRESS))

    pagg = joinpath(OUT, "diagnostics_compute_aggregate.png")
    save(pagg, render_diagnostics_aggregate(dl,
        ["mcts@1k", "dpw@1k", "q_learning"]; column="model_calls", bins=5))
    println("DIAG_AGGREGATE_PNG=", isfile(pagg) && filesize(pagg) > 5_000)

    pd = joinpath(OUT, "diagnostics_lateral_aggregate.png")
    save(pd, render_diagnostics_aggregate(dl, ["mcts@1k", "dpw@1k", "td3"];
        column="d", bins=5, ylabel="lateral offset d (m)"))
    println("DIAG_LATERAL_PNG=", isfile(pd) && filesize(pd) > 5_000)

    bins = progress_bins(dl, "dpw@1k", "model_calls"; bins=5)
    println("DPW_COLLAPSE_BINS=", join([round(Int, x.mean) for x in bins], ","))
    println("DPW_COLLAPSE_N=", join([x.n for x in bins], ","))
else
    println("DECISION_LOG_ROWS=skipped")
end

# FJ9.7 — animation. Playback of the same frozen log. The mdp is built only
# for the static track description; `imdp` counts generative calls so
# ENV_STEPPED is measured rather than asserted by intent.
function export_animation(sw, seq, out, stem)
    for (ext, fr) in (("mp4", 10), ("gif", 8))
        p = joinpath(out, "$stem.$ext")
        try
            render_animation(sw, seq, p; framerate=fr)
            isfile(p) && filesize(p) > 5_000 && return (ext, p)
        catch e
            println("ANIM_$(uppercase(ext))_FAILED=", sprint(showerror, e)[1:min(120, end)])
        end
    end
    return ("none", "")
end

if isfile(dlog)
    imdp = InstrumentedMDP(DuckietownMDP(cfg; action_space=:discrete))
    reset_model_calls!(imdp)
    ref = rand(MersenneTwister(7), initialstate(imdp))
    sw = static_world(imdp, ref)
    # the track description must not depend on which reset produced it
    sw2 = static_world(imdp, rand(MersenneTwister(4242), initialstate(imdp)))
    println("STATIC_WORLD_SEED_INVARIANT=",
        sw.sign_positions == sw2.sign_positions &&
        sw.view_extent == sw2.view_extent &&
        length(sw.tiles) == length(sw2.tiles))

    dl2 = load_decision_log(dlog)
    picks = [("td3", :first_stagnation, "anim_td3_stagnation"),
             ("dpw@1k", :median_length_terminating, "anim_dpw_terminal"),
             ("mcts@1k", :median_return, "anim_mcts_reference")]
    function run_animations(dl2, sw, picks, out)
        lines = String[]
        for (solver, rule, stem) in picks
            sel = select_episode(dl2, solver; rule=rule)
            seq = animation_sequence(dl2, solver, sel.seed)
            push!(lines, "ANIM_SELECT_$(uppercase(stem))=seed $(sel.seed) " *
                "[$(rule)] $(sel.justification)")
            # one still, from the same layout the video uses
            png = joinpath(out, "$(stem)_frame.png")
            save(png, render_frame(sw, seq, cld(length(seq), 2)))
            push!(lines, "ANIM_FRAME_$(uppercase(stem))=" *
                string(isfile(png) && filesize(png) > 5_000))
            ext, p = export_animation(sw, seq, out, stem)
            push!(lines, "ANIM_VIDEO_$(uppercase(stem))=$ext " *
                (isempty(p) ? "" : string(filesize(p))))
        end
        return lines
    end
    for l in run_animations(dl2, sw, picks, OUT)
        println(l)
    end

    # FJ9.7d — paired, absolute decision index, shorter panel freezes
    pa = animation_sequence(dl2, "td3", 1001)
    pb = animation_sequence(dl2, "dpw@1k", 1001)
    println("PAIRED_FRAMES=", paired_frames(pa, pb),
        " a=", length(pa), " b=", length(pb))
    # a still from the frozen region, so the freeze banner is inspectable
    pfr = joinpath(OUT, "anim_paired_frame130.png")
    save(pfr, render_frame(sw, pa, pb, 130))
    println("PAIRED_FRAME_PNG=", isfile(pfr) && filesize(pfr) > 5_000)
    println("PAIRED_FROZEN_AT_130=", is_frozen(pb, 130), " ",
        frame_index(pb, 130), "/", length(pb))

    ppath = joinpath(OUT, "anim_paired_td3_dpw_1001.mp4")
    ok = try
        render_paired_animation(sw, pa, pb, ppath; framerate=10)
        isfile(ppath) && filesize(ppath) > 5_000
    catch e
        println("PAIRED_MP4_FAILED=", sprint(showerror, e)[1:min(120, end)])
        false
    end
    println("PAIRED_VIDEO=", ok)

    # nothing above stepped the environment
    println("ENV_STEPPED=", model_calls(imdp) != 0)
    println("ENV_MODEL_CALLS=", model_calls(imdp))
else
    println("STATIC_WORLD_SEED_INVARIANT=skipped")
end

loaded = [string(m.name) for m in keys(Base.loaded_modules)]
python_like = filter(m -> m in ("PythonCall", "CondaPkg", "PyCall"), loaded)
println("PYTHON_MODULES=", isempty(python_like) ? "none" : join(python_like, ","))
println("MCTS_LOADED=", any(==("MCTS"), loaded))
println("RESERVED_STILL_RESERVED=",
    !isdefined(DuckietownDecisionModels, :render_observation) &&
    !isdefined(DuckietownDecisionModels, :render_belief))
println("RENDER_CHECK_OK=true")
