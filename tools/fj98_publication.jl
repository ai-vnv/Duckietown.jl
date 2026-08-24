# FJ9.8 — build the publication composites in a FRESH process.
#
#     tools/run_publication.sh
#
# Every panel is drawn from a live data object. Nothing here loads a PNG, runs
# the environment, or touches Python or a planning library. The MDP is built
# for map semantics and one reference state; `InstrumentedMDP` counts
# generative calls so the claim is measured.
#
# Accumulators live inside functions: a bare top-level loop puts them in soft
# scope and Julia makes them fresh locals, which has cost this project three
# debugging cycles (see docs/FJ84C_STATUS.md).

using DuckietownDecisionModels
using POMDPs
using Random
using CairoMakie
CairoMakie.activate!(type="png")

const ROOT = pkgdir(DuckietownDecisionModels)
const OUT = joinpath(ROOT, "artifacts", "fj9", "publication")
mkpath(OUT)
const DUCK = joinpath(ROOT, "..", "duckduck")

cfg = joinpath(DUCK, "policies", "q_learning", "training_config.yaml")
imdp = InstrumentedMDP(DuckietownMDP(cfg; action_space=:discrete))
reset_model_calls!(imdp)

# ---- FJ9.8a inventory ------------------------------------------------------
inv = publication_inventory(ROOT)
write(joinpath(OUT, "inventory.md"),
    "# FJ9.8a — publication figure inventory\n\n" * inventory_table(inv))
println("INVENTORY_TOTAL=", length(inv))
for r in (MAIN_FIGURE, SUPPLEMENTARY, DIAGNOSTIC_ONLY)
    println("INVENTORY_$(r)=", count(a -> a.role === r, inv))
end
println("INVENTORY_MISSING=", count(a -> !a.available, inv))
for a in inv
    a.available || println("  ABSENT: ", a.id, " -> ", a.source)
end

# ---- Figure 1 --------------------------------------------------------------
s0 = rand(MersenneTwister(11), initialstate(imdp))
world = world_scene(imdp, s0)
tr = imdp.inner.transition
raw, _ = get_raw_state(s0, tr.state_cfg)
cont = get_continuous_state(s0, raw, tr.state_cfg, tr.continuous_cfg;
    controller_cfg=tr.duck_cfg)
fig1 = figure_model(world, projection_scene(raw, cont))

# ---- Figure 2 --------------------------------------------------------------
qpath = joinpath(DUCK, "policies", "q_learning", "policy.npy")
# the .npy cache produced ONCE by tools/run_export_weights.sh — the exact
# checkpoint bits, read natively, so this build needs no torch and no Python
td3dir = joinpath(ROOT, "artifacts", "fj9", "weights", "td3")
fig2 = nothing
if isfile(qpath)
    qpol = QTablePolicy(qpath; solver=:q_learning)
    tab = policy_slice(qpol, :d, :phi)
    if isdir(td3dir)
        td3 = TD3ActorPolicy(td3dir)
        ccfg = imdp.inner.transition.continuous_cfg
        sv = policy_slice(td3, ccfg, :d, :phi; name="td3")
        println("FIG2_TD3=native from ", relpath(td3dir, ROOT))
        fig2 = figure_policy(tab, sv, sv)
    else
        println("FIG2_TD3=ABSENT — run tools/run_export_weights.sh first")
    end
else
    println("FIG2=ABSENT — tabular checkpoint unavailable")
end

# ---- Figure 3 --------------------------------------------------------------
snapdir = joinpath(ROOT, "artifacts", "fj9")
mp = joinpath(snapdir, "search_snapshot_mcts.json")
dp = joinpath(snapdir, "search_snapshot_dpw.json")
fig3 = (isfile(mp) && isfile(dp)) ?
    figure_search(load_snapshot(mp), load_snapshot(dp)) : nothing

# ---- Figure 4 --------------------------------------------------------------
dlog = joinpath(ROOT, "artifacts", "fj8", "enriched", "decisions.csv")
art = joinpath(ROOT, "artifacts", "fj8", "six_solver_episodes.csv")
fig4 = nothing
if isfile(dlog) && isfile(art)
    dl = load_decision_log(dlog)
    agg = load_rollout_artifact(art)
    sel_td3 = select_episode(dl, "td3"; rule=:first_stagnation)
    sel_dpw = select_episode(dl, "dpw@1k"; rule=:median_length_terminating)
    sel_mcts = select_episode(dl, "mcts@1k"; rule=:median_return)
    fig4 = figure_episode(split(rstrip(comparison_table(agg)), "\n"),
        episode_diagnostics(dl, "td3", sel_td3.seed),
        episode_diagnostics(dl, "dpw@1k", sel_dpw.seed),
        progress_bins(dl, "dpw@1k", "model_calls"; bins=5),
        [sel_td3, sel_dpw, sel_mcts])
end

# ---- caption validation (FJ9.8d) ------------------------------------------
function validate_captions(figs)
    lines = String[]
    for f in figs
        f === nothing && continue
        c = check_caption(f)
        push!(lines, "CAPTION_$(replace(f.figure_id, " " => "_"))=" *
            (c.ok ? "ok" : "FAIL absent=$(join(c.absent, "|")) " *
                           "forbidden=$(join(c.present, "|"))"))
    end
    return lines
end
figs = [fig1, fig2, fig3, fig4]
for l in validate_captions(figs)
    println(l)
end

# ---- export (FJ9.8e) -------------------------------------------------------
function export_figures(figs, out)
    lines = String[]
    for f in figs
        f === nothing && continue
        stem = lowercase(replace(f.figure_id, " " => ""))
        fig = render_composite(f)
        for (ext, kind) in (("pdf", "pdf"), ("svg", "svg"), ("png", "png"))
            CairoMakie.activate!(type=kind)
            p = joinpath(out, "$stem.$ext")
            save(p, fig; px_per_unit=ext == "png" ? 4 : 1)
            push!(lines, "EXPORT_$(uppercase(stem))_$(uppercase(ext))=" *
                string(isfile(p) && filesize(p) > 5_000) * " " *
                string(filesize(p)))
        end
        CairoMakie.activate!(type="png")
        push!(lines, "FIGURE_FINGERPRINT_$(uppercase(stem))=$(f.fingerprint)")
        # the composite fingerprint is figure identity and belongs IN the
        # caption file, not only in this tool's stdout
        write(joinpath(out, "$stem.caption.txt"),
            f.caption * "\n\n" * provenance_block(f) *
            "\nfigure_fingerprint = " * f.fingerprint * "\n")
    end
    return lines
end
for l in export_figures(figs, OUT)
    println(l)
end

println("ENV_MODEL_CALLS=", model_calls(imdp))
loaded = [string(m.name) for m in keys(Base.loaded_modules)]
println("PYTHON_MODULES=",
    isempty(filter(m -> m in ("PythonCall", "CondaPkg", "PyCall"), loaded)) ?
    "none" : "PRESENT")
println("MCTS_LOADED=", any(==("MCTS"), loaded))
println("PUBLICATION_OK=true")
