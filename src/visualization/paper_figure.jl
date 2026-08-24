# FJ9.8 — publication composites.
#
#     validated scene / data objects -> PublicationComposite -> backend
#
# A composite is built from the SAME objects the individual renderers consume,
# never by pasting PNGs together. Assembling rasters would give inconsistent
# fonts and axes, scaling artefacts, no vector output, and — worst — a caption
# written by hand beside numbers nobody recomputed.
#
# FJ9.6 is why the caption matters as much as the panel. Every aggregate
# number about TD3 was correct while the sentence attached to them said the
# opposite of what happened. So captions here are generated from the data and
# then CHECKED against required and forbidden claims, the same way a value is
# checked against a tolerance.

"""
    FigureRole

`MAIN_FIGURE` — carries an argument the paper cannot make without it.
`SUPPLEMENTARY` — supports a main figure; a video belongs here and never in
the main body, because a printed paper has to stand on its own.
`DIAGNOSTIC_ONLY` — an internal check that was never meant for a reader.
"""
@enum FigureRole MAIN_FIGURE SUPPLEMENTARY DIAGNOSTIC_ONLY

"""
    PublicationArtifact

One line of the FJ9.8a inventory: what exists, what it is for, and whether it
is actually on disk.
"""
struct PublicationArtifact
    id::String
    description::String
    role::FigureRole
    figure::String
    source::String
    available::Bool
end

const _INVENTORY = (
    ("world_scene", "top-down latent world state", MAIN_FIGURE, "Figure 1",
     "artifacts/fj9/world.png"),
    ("projection_panel", "privileged ContinuousState projection",
     MAIN_FIGURE, "Figure 1", "artifacts/fj9/projection.png"),
    ("tabular_slice", "Q-learning action / value / tie / margin slice",
     MAIN_FIGURE, "Figure 2", "computed"),
    ("continuous_slice", "TD3 v_cmd and omega_cmd surfaces", MAIN_FIGURE,
     "Figure 2", "computed"),
    ("search_tree_mcts", "MCTS tree from the frozen state", MAIN_FIGURE,
     "Figure 3", "artifacts/fj9/search_snapshot_mcts.json"),
    ("search_tree_dpw", "DPW tree from the same frozen state", MAIN_FIGURE,
     "Figure 3", "artifacts/fj9/search_snapshot_dpw.json"),
    ("action_plane_dpw", "DPW sampled continuous root actions", MAIN_FIGURE,
     "Figure 3", "artifacts/fj9/search_snapshot_dpw.json"),
    ("six_solver_table", "frozen six-solver outcome summary", MAIN_FIGURE,
     "Figure 4", "artifacts/fj8/six_solver_episodes.csv"),
    ("diagnostics_td3", "TD3 stagnation time series", MAIN_FIGURE,
     "Figure 4", "artifacts/fj8/enriched/decisions.csv"),
    ("diagnostics_dpw", "DPW deterioration time series", MAIN_FIGURE,
     "Figure 4", "artifacts/fj8/enriched/decisions.csv"),
    ("compute_by_progress", "realised model calls by episode progress",
     MAIN_FIGURE, "Figure 4", "artifacts/fj8/enriched/decisions.csv"),
    ("rollout_comparison", "aggregate return / length comparison",
     SUPPLEMENTARY, "Supplementary", "artifacts/fj9/rollout_comparison.png"),
    ("video_td3", "TD3 stagnation playback", SUPPLEMENTARY,
     "Supplementary Video 1", "artifacts/fj9/anim_td3_stagnation.mp4"),
    ("video_dpw", "DPW termination playback", SUPPLEMENTARY,
     "Supplementary Video 2", "artifacts/fj9/anim_dpw_terminal.mp4"),
    ("video_mcts", "MCTS reference playback", SUPPLEMENTARY,
     "Supplementary Video 3", "artifacts/fj9/anim_mcts_reference.mp4"),
    ("video_paired", "paired TD3 / DPW playback", SUPPLEMENTARY,
     "Supplementary Video 4", "artifacts/fj9/anim_paired_td3_dpw_1001.mp4"),
    ("decision_contract", "FJ9.6a decision-artifact contract",
     DIAGNOSTIC_ONLY, "—", "artifacts/fj9/decision_contract.md"),
    ("budget_study", "FJ8.4a planner cost curve", DIAGNOSTIC_ONLY, "—",
     "artifacts/fj8/budget_study.md"),
    ("lap_completion", "FJ8 lap analysis", DIAGNOSTIC_ONLY, "—",
     "artifacts/fj8/lap_completion.txt"),
)

"""
    PUBLICATION_LAYOUT_VERSION
"""
const PUBLICATION_LAYOUT_VERSION = "fj98.1"

"""
    publication_inventory(root) -> Vector{PublicationArtifact}

FJ9.8a. Every artefact the project has produced, classified before any layout
is chosen, and probed for existence rather than assumed.

Not everything belongs in the main body. A composite assembled from
"everything we have" is a contact sheet, not an argument.
"""
function publication_inventory(root::AbstractString)
    out = PublicationArtifact[]
    for (id, desc, role, fig, src) in _INVENTORY
        avail = src == "computed" || isfile(joinpath(root, src))
        push!(out, PublicationArtifact(id, desc, role, fig, src, avail))
    end
    return out
end

"""
    inventory_table(items) -> String
"""
function inventory_table(items::AbstractVector{PublicationArtifact})
    io = IOBuffer()
    println(io, "| Artifact | Role | Figure | Source | Present |")
    println(io, "|---|---|---|---|---|")
    for a in items
        println(io, "| $(a.id) — $(a.description) | $(a.role) | $(a.figure) ",
            "| `$(a.source)` | ", a.available ? "yes" : "**no**", " |")
    end
    return String(take!(io))
end

"""
    PanelSpec

One panel of a composite: its data, and the metadata its caption is generated
from. `payload` is a live scene/data object — a `WorldScene`, `PolicySlice`,
`SearchSnapshot`, `EpisodeDiagnostics` — never an image.
"""
struct PanelSpec
    id::String
    title::String
    kind::Symbol
    payload::Any
    provenance::Vector{Pair{String,String}}
    semantics::String
end

"""
    PublicationComposite

A figure: its panels, the caption generated from them, and machine-readable
metadata carrying every fingerprint, selection rule and semantic qualifier a
reader needs to reproduce or contest it.
"""
struct PublicationComposite
    figure_id::String
    title::String
    panels::Vector{PanelSpec}
    layout::Vector{NTuple{4,Int}}
    caption::String
    metadata::Vector{Pair{String,String}}
    fingerprint::String
end

panel_ids(c::PublicationComposite) = [p.id for p in c.panels]

"""
    grid_layout(n; columns) -> Vector{NTuple{4,Int}}

The default placement: `n` panels down a fixed number of columns, each one
cell. Figures whose panels differ in natural shape override it — a text table
in a square cell is mostly whitespace.
"""
grid_layout(n::Integer; columns::Integer = 2) =
    [(cld(k, columns), mod1(k, columns), 1, 1) for k in 1:n]

"""
    wrap_text(text, width) -> String

Hard-wrap on word boundaries. Makie does not reflow, so an unwrapped caption
runs off the canvas and the end of the sentence is simply lost — which for a
caption means losing the part that qualifies the claim.
"""
function wrap_text(text::AbstractString, width::Integer)
    out = String[]
    for para in split(text, "
")
        line = ""
        for w in split(para, " ")
            if isempty(line)
                line = String(w)
            elseif length(line) + 1 + length(w) <= width
                line *= " " * w
            else
                push!(out, line)
                line = String(w)
            end
        end
        push!(out, line)
    end
    return join(out, "
")
end

function _composite(figure_id, title, panels, caption, extra;
        layout = grid_layout(length(panels)))
    meta = Pair{String,String}["figure_id" => figure_id,
        "panel_ids" => join((p.id for p in panels), ","),
        "layout_version" => PUBLICATION_LAYOUT_VERSION]
    for p in panels, kv in p.provenance
        push!(meta, "$(p.id).$(first(kv))" => last(kv))
    end
    append!(meta, extra)
    fp = string(hash((figure_id, Tuple(string(k, "=", v) for (k, v) in meta),
        caption)); base = 16, pad = 16)
    return PublicationComposite(figure_id, title, panels, collect(layout),
        caption, meta, fp)
end

# ---------------------------------------------------------------------------
# Caption requirements — checked, not trusted
# ---------------------------------------------------------------------------

"""
    CaptionRule

What a caption must and must not say. `forbidden` exists because FJ9.6 found a
false claim living happily beside correct numbers; a rule that only requires
content cannot catch that.
"""
struct CaptionRule
    figure_id::String
    required::Vector{String}
    forbidden::Vector{String}
end

const CAPTION_RULES = (
    CaptionRule("Figure 1",
        ["privileged", "not an observation", "latent"],
        ["belief state estimate", "sensor measurement"]),
    CaptionRule("Figure 2",
        ["feature space", "not guaranteed reachable", "fixed context",
         "distinct tabular states", "tied"],
        ["reachable state space", "over all states"]),
    CaptionRule("Figure 3",
        ["same", "visits", "root actions", "budget"],
        ["is random", "dpw is worse", "optimal action"]),
    CaptionRule("Figure 4",
        ["reaches the stop sign", "full stop", "never proceeds", "horizon",
         "passed_stops = 0", "generative calls"],
        ["never reaches a stop sign", "never reaches the stop sign",
         "combined score", "overall ranking"]),
)

"""
    caption_rule(figure_id) -> CaptionRule
"""
caption_rule(id::AbstractString) =
    only(filter(r -> r.figure_id == id, collect(CAPTION_RULES)))

"""
    check_caption(composite) -> NamedTuple

Validate a caption against its rule. Returns `(ok, absent, present)`, where
`present` lists forbidden phrases that appeared.
"""
function check_caption(c::PublicationComposite)
    rule = caption_rule(c.figure_id)
    low = lowercase(c.caption)
    absent = [s for s in rule.required if !occursin(lowercase(s), low)]
    present = [s for s in rule.forbidden if occursin(lowercase(s), low)]
    return (ok = isempty(absent) && isempty(present), absent = absent,
        present = present)
end

"""
    provenance_block(composite) -> String
"""
provenance_block(c::PublicationComposite) =
    join(("$k = $v" for (k, v) in c.metadata), "\n")

# ---------------------------------------------------------------------------
# The four main figures
# ---------------------------------------------------------------------------

"""
    figure_model(world, projection) -> PublicationComposite

**Figure 1 — model and representation.** What is modelled, and the separation
FJ10 established between the latent world, the privileged projection the
policies consume, and the observation/belief layer that does not exist yet.
"""
function figure_model(world::WorldScene, proj::ProjectionScene)
    obs = continuous_state_observability()
    est = count(o -> o.class === SENSOR_ESTIMABLE, obs)
    # AGENT_MEMORY is FJ10's "no physical counterpart" class: sigma_stop and
    # stop_hold_progress are the agent's own memory, not world quantities a
    # sensor could ever estimate
    mem = count(o -> o.class === AGENT_MEMORY, obs)
    panels = [
        PanelSpec("1A", "Latent world state", :world, world,
            ["source" => "world_scene",
             "ego_speed" => string(round(world.ego_speed; digits = 4)),
             "stop_lines" => string(length(world.stop_lines))],
            "the latent truth the transition operates on"),
        PanelSpec("1B", PROJECTION_PANEL_TITLE, :projection, proj,
            ["source" => string(proj.source),
             "entries" => string(length(proj.entries))],
            "the privileged vector the policies consume"),
        PanelSpec("1C", "Observability classification", :observability, obs,
            ["components" => string(length(obs)),
             "sensor_estimable" => string(est),
             "agent_memory" => string(mem)],
            "FJ10's per-component classification")]
    cap = "**Figure 1. Model and representation.** (1A) The latent world " *
        "state: tile geometry, lane centrelines, the ego's true collision " *
        "footprint, and the stop line the model measures `d_stop` against. " *
        "(1B) The $(length(proj.entries))-entry privileged projection the " *
        "policies consume. This is a privileged model state and **not an " *
        "observation**: it is read from the latent world, not estimated from " *
        "sensors. (1C) Of the $(length(obs)) components, $(est) are " *
        "sensor-estimable, while $(mem) are the agent's own memory with no " *
        "physical counterpart a sensor could estimate, which is why an " *
        "observation model cannot be obtained by relabelling this vector. No belief representation exists in this formulation."
    return _composite("Figure 1", "Model and representation", panels, cap,
        ["observability.sensor_estimable" => string(est)];
        layout = [(1, 1, 2, 1), (1, 2, 1, 1), (2, 2, 1, 1)])
end

"""
    figure_policy(tabular, td3_v, td3_omega) -> PublicationComposite

**Figure 2 — policy structure and ambiguity.** A selected-action map alone
makes a table look more decisive than it is; the tie and margin surfaces are
what show the ambiguity.
"""
function figure_policy(tab::PolicySlice, td3v::PolicySlice,
        td3w::PolicySlice)
    ties = tie_surface(tab)
    cells = length(ties)
    tied = count(>(1), ties)
    fixed = join(fixed_context_lines(tab), "; ")
    panels = [
        PanelSpec("2A", "Q-learning selected action", :slice_action, tab,
            ["policy" => tab.policy_name, "mode" => string(tab.mode),
             "fingerprint" => slice_fingerprint(tab),
             "cells" => string(cells),
             "distinct_tabular_states" => string(tab.distinct_states)],
            "argmax over the table, with the validated near-tie rule"),
        PanelSpec("2B", "Value and ambiguity", :slice_value, tab,
            ["tied_cells" => string(tied),
             "decisive_cells" => string(cells - tied)],
            "V(s), tie count and Q-margin on the same grid"),
        PanelSpec("2C", "TD3 v_cmd", :slice_continuous, td3v,
            ["policy" => td3v.policy_name,
             "fingerprint" => slice_fingerprint(td3v)],
            "commanded forward speed"),
        PanelSpec("2D", "TD3 omega_cmd", :slice_continuous, td3w,
            ["policy" => td3w.policy_name,
             "fingerprint" => slice_fingerprint(td3w)],
            "commanded angular rate — a different unit, a separate panel")]
    cap = "**Figure 2. Policy structure and ambiguity.** Slices over " *
        "`$(tab.x.field)` × `$(tab.y.field)` in **feature space**: every " *
        "cell is a coordinate at which the policy can be evaluated, and is " *
        "**not guaranteed reachable** by the dynamics. Fixed context: " *
        "$(fixed). (2A) The selected action. (2B) Value with the tie and " *
        "margin surfaces: $(tied) of $(cells) cells are **tied**, against " *
        "$(cells - tied) decisive ones, so a one-action-per-cell map " *
        "overstates how decisive the table is. The discretizer indexes " *
        "`bin(d)` and `bin(phi + d)`, so these $(cells) cells collapse to " *
        "$(tab.distinct_states) **distinct tabular states**. (2C, 2D) TD3's " *
        "continuous outputs, on separate panels because m/s and rad/s do not " *
        "share an axis."
    return _composite("Figure 2", "Policy structure and ambiguity", panels,
        cap, ["slice.tied_cells" => string(tied),
              "slice.cells" => string(cells),
              "slice.distinct_states" => string(tab.distinct_states)])
end

"""
    figure_search(mcts, dpw) -> PublicationComposite

**Figure 3 — search behaviour.** Both snapshots come from the same frozen
state, which is the only reason the panels are comparable at all.
"""
function figure_search(mcts::SearchSnapshot, dpw::SearchSnapshot)
    mcts.state_fingerprint == dpw.state_fingerprint || throw(ArgumentError(
        "Figure 3 compares two searches from the SAME state; got " *
        "$(mcts.state_fingerprint) and $(dpw.state_fingerprint)"))
    sm = search_statistics(mcts)
    sd = search_statistics(dpw)
    panels = [
        PanelSpec("3A", "MCTS tree", :search_tree, mcts,
            ["solver" => mcts.solver, "state" => mcts.state_fingerprint,
             "root_actions" => string(sm.root_actions),
             "mean_visits" => string(round(sm.mean_visits; digits = 2)),
             "median_visits" => string(sm.median_visits),
             "max_visits" => string(sm.max_visits)],
            "discrete macro actions at the root"),
        PanelSpec("3B", "MCTS root-action visits", :search_summary, mcts,
            ["total_root_visits" => string(sm.total_root_visits)],
            "visits grouped by ACTION, not by child node"),
        PanelSpec("3C", "DPW tree", :search_tree, dpw,
            ["solver" => dpw.solver, "state" => dpw.state_fingerprint,
             "root_actions" => string(sd.root_actions),
             "mean_visits" => string(round(sd.mean_visits; digits = 2)),
             "median_visits" => string(sd.median_visits),
             "max_visits" => string(sd.max_visits)],
            "progressively widened continuous actions"),
        PanelSpec("3D", "DPW sampled action plane", :action_plane, dpw,
            ["sampled_actions" => string(sd.distinct_actions)],
            "only sampled points; no interpolated surface")]
    cap = "**Figure 3. Search behaviour at a single frozen state.** Both " *
        "planners searched from the **same** state (fingerprint " *
        "`$(mcts.state_fingerprint)`), at a matched **budget**. (3A, 3B) " *
        "MCTS expands $(sm.root_actions) distinct **root actions** with " *
        "$(round(sm.mean_visits; digits = 2)) **visits** per action on " *
        "average (median $(sm.median_visits), max $(sm.max_visits)). " *
        "(3C, 3D) DPW expands $(sd.root_actions) distinct continuous root " *
        "actions with mean $(round(sd.mean_visits; digits = 2)) (median " *
        "$(sd.median_visits), max $(sd.max_visits)); only sampled actions " *
        "are drawn, since a smoothed surface would imply combinations the " *
        "planner never tried. Both planners leave many candidates lightly " *
        "explored, but MCTS concentrates substantially more evidence on its " *
        "most-preferred discrete action, whereas DPW spreads the same budget " *
        "across many continuous actions."
    return _composite("Figure 3", "Search behaviour", panels, cap,
        ["state_fingerprint" => mcts.state_fingerprint,
         "mcts.root_actions" => string(sm.root_actions),
         "dpw.root_actions" => string(sd.root_actions)])
end

"""
    figure_episode(summary, td3, dpw, bins, selections) -> PublicationComposite

**Figure 4 — episode behaviour and failure mechanisms.** This is where the
FJ9.6 correction has to survive into print.
"""
function figure_episode(summary::AbstractVector, td3::EpisodeDiagnostics,
        dpw::EpisodeDiagnostics, bins::AbstractVector,
        selections::AbstractVector{EpisodeSelection})
    ev = Dict(td3.events)
    stop_at = haskey(ev, "full_stop") ? first(ev["full_stop"]) : -1
    zone = count(==(1.0), series_named(td3, "sigma_stop").values)
    vs = collect(skipmissing(series_named(td3, "v").values))
    tail = vs[max(1, length(vs) ÷ 2):end]
    meanv = isempty(tail) ? 0.0 : sum(tail) / length(tail)
    calls = join((string(round(Int, b.mean)) for b in bins), " → ")
    selrule = join(("$(s.solver) $(s.rule) seed $(s.seed)"
                    for s in selections), "; ")

    panels = [
        PanelSpec("4A", "Six-solver outcome summary", :summary_table, summary,
            ["episodes" => "120", "seeds" => "20", "solvers" => "6"],
            "task performance and cost reported separately, never combined"),
        PanelSpec("4B", "TD3 — stagnation at the stop line", :diagnostics,
            td3, ["solver" => td3.solver, "seed" => string(td3.seed),
             "outcome" => string(td3.outcome), "full_stop" => string(stop_at),
             "stop_zone_decisions" => string(zone),
             "source" => td3.source_fingerprint],
            "the mechanism behind a return no aggregate explained"),
        PanelSpec("4C", "DPW — deterioration to termination", :diagnostics,
            dpw, ["solver" => dpw.solver, "seed" => string(dpw.seed),
             "outcome" => string(dpw.outcome), "reason" => dpw.reason,
             "decisions" => string(length(dpw.decisions))],
            "navigation degrading into an environment termination"),
        PanelSpec("4D", "Generative calls by episode progress", :bins, bins,
            ["bins" => string(length(bins)), "means" => calls,
             "n" => join((string(b.n) for b in bins), ",")],
            "realised compute, binned without interpolation")]

    cap = "**Figure 4. Episode behaviour and failure mechanisms.** (4A) The " *
        "frozen six-solver comparison over 20 paired seeds; task performance " *
        "and computational cost are reported on separate axes and never " *
        "combined into one score. (4B) TD3, seed $(td3.seed): it **reaches " *
        "the stop sign** and performs a **full stop** at decision " *
        "$(stop_at), then remains in the stop zone for $(zone) of " *
        "$(length(td3.decisions)) decisions at a mean speed of " *
        "$(round(meanv; digits = 3)) m/s and **never proceeds** past it — " *
        "`passed_stops = 0` across all 20 seeds — so the episode ends only " *
        "when the evaluation **horizon** is reached, with the return " *
        "dominated by the stagnation penalty. (4C) DPW, seed $(dpw.seed): " *
        "navigation degrades until the environment terminates the episode at " *
        "decision $(length(dpw.decisions)) ($(dpw.reason)). (4D) Realised " *
        "**generative calls** per decision by normalised episode progress " *
        "for DPW: $(calls). Compute is endogenous — it falls as the " *
        "trajectory deteriorates and the planner has less tree to expand. " *
        "Episodes were selected by stated rule ($(selrule)), not by " *
        "inspection."
    return _composite("Figure 4", "Episode behaviour and failure mechanisms",
        panels, cap,
        ["td3.full_stop_decision" => string(stop_at),
         "td3.stop_zone_decisions" => string(zone),
         "dpw.terminated_at" => string(length(dpw.decisions)),
         "selection_rules" => selrule];
        layout = [(1, 1, 1, 2), (2, 1, 1, 1), (2, 2, 1, 1), (3, 1, 1, 2)])
end
