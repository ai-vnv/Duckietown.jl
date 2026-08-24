# FJ9.0 / FJ9.5b — the solver-neutral search representation.
#
# `render_search` must never import a planning library, for the same reason
# the core names no solver: the next solver should plug in without the
# renderer changing. A solver extension converts its own tree into this
# snapshot, the snapshot is serialised, and the renderer only ever sees the
# snapshot — so a search figure can be reproduced with MCTS.jl absent.
#
#     solver tree -> (extension) -> SearchSnapshot -> artifact -> renderer
#
# Quantities a solver does not define are stored as `missing`. They are never
# back-computed from something else: FJ9.5a showed that aggregates do not
# determine a tree, and the same discipline applies to individual fields.

"""
    capture_search(planner, state; id, planner_seed) -> SearchSnapshot

Convert a solver-specific search tree into a [`SearchSnapshot`](@ref). Given
methods by a solver extension; the core declares it so the conversion has a
home without the core knowing any solver type.
"""
function capture_search end

"""
    SearchNode

One node. `parent == 0` marks the root. `value` is `missing` when the solver
does not define one for that node. `action` is the action that led here from
the parent (`nothing` at the root) and is stored as `Any` so a snapshot can
hold `MacroAction`s or continuous `DuckieAction`s without the renderer caring.
"""
struct SearchNode
    id::Int
    parent::Int
    depth::Int
    visits::Int
    value::Union{Float64,Missing}
    action::Any
end

"""
    SearchSnapshot

A planner's search, in a form no solver owns, with the provenance needed to
tie it to one identified decision.

`state_fingerprint`, `config_fingerprint` and `planner_seed` exist so a figure
cannot be silently re-attributed: two snapshots of the same shape from
different states or configurations are different evidence.

`extra` carries solver-specific scalars, the same open-slot convention as
`PlanningDiagnostics`, so this type never needs a field for the next planner.
"""
struct SearchSnapshot
    id::String
    solver::String
    state_fingerprint::String
    config_fingerprint::String
    planner_seed::Int
    root_visits::Int
    selected_action::Any
    nodes::Vector{SearchNode}
    extra::NamedTuple
end

SearchSnapshot(solver, root_visits, nodes; id="", state_fingerprint="",
    config_fingerprint="", planner_seed=-1, selected_action=nothing,
    extra=NamedTuple()) =
    SearchSnapshot(String(id), String(solver), String(state_fingerprint),
        String(config_fingerprint), Int(planner_seed), Int(root_visits),
        selected_action, nodes, extra)

"""
    state_fingerprint(state) -> String

Deterministic fingerprint of a latent world state, so a snapshot can name the
state it searched from without embedding it.
"""
function state_fingerprint(s::DuckieWorldState)
    h = hash((s.ego.pos, s.ego.angle, s.ego.v_long, s.ego.omega, s.ego.speed,
        s.ego.step_count, s.ego.timestamp,
        Tuple(s.ego.command_history), Tuple(s.ego.q0), Tuple(s.ego.v0),
        Tuple((d.pos, d.angle, d.vel, d.visible, d.pedestrian_active,
               d.walk_distance, d.time) for d in s.ducks),
        Tuple((g.pos, g.angle) for g in s.stop_signs),
        (s.stop_memory.sigma_stop, s.stop_memory.hold_steps,
         s.stop_memory.last_stop_id, s.stop_memory.last_d_stop),
        s.lane_fallback, Tuple(s.crossings_started), Tuple(s.crossing_armed)))
    return string(h; base=16, pad=16)
end

"""
    snapshot_fingerprint(snapshot) -> String

Content fingerprint over the tree AND its provenance. Changing any parent,
action, visit count or value changes it.
"""
function snapshot_fingerprint(s::SearchSnapshot)
    h = hash((s.solver, s.state_fingerprint, s.config_fingerprint,
        s.planner_seed, s.root_visits, string(s.selected_action),
        [(n.id, n.parent, n.depth, n.visits,
          n.value === missing ? "missing" : string(n.value),
          string(n.action)) for n in s.nodes]))
    return string(h; base=16, pad=16)
end

"""
    root_children(snapshot) -> Vector{SearchNode}

The actions considered at the root, which is what "what did the planner think?"
usually means.
"""
root_children(s::SearchSnapshot) = filter(n -> n.parent == 1, s.nodes)

"""
    search_max_depth(snapshot) -> Int
"""
search_max_depth(s::SearchSnapshot) =
    isempty(s.nodes) ? 0 : maximum(n -> n.depth, s.nodes)

"""
    check_snapshot(snapshot; action_space) -> NamedTuple

Structural validation, so a malformed snapshot fails where it is built rather
than inside a renderer.

Checks: ids are `1:n`; exactly one root and it is node 1; every non-root
parent exists and precedes its child (no dangling edges); depths follow the
parent chain; visits are non-negative; the root carries no action and every
other node carries one; and, when a `selected_action` is recorded, it appears
among the root's children.

Pass `action_space` to also check that continuous actions lie inside the
reference box.
"""
function check_snapshot(s::SearchSnapshot; action_space=nothing)
    n = length(s.nodes)
    n == 0 && return (ok=true, nodes=0, roots=0, issues=String[])
    issues = String[]
    [x.id for x in s.nodes] == collect(1:n) || push!(issues, "ids are not 1:n")
    roots = count(x -> x.parent == 0, s.nodes)
    roots == 1 || push!(issues, "expected exactly one root, found $roots")
    s.nodes[1].parent == 0 || push!(issues, "node 1 is not the root")
    s.nodes[1].action === nothing || push!(issues, "the root carries an action")
    for x in s.nodes
        if x.parent == 0
            x.id == 1 || push!(issues, "node $(x.id) is a second root")
            continue
        end
        x.action === nothing &&
            push!(issues, "non-root node $(x.id) carries no action")
        if !(1 <= x.parent <= n)
            push!(issues, "node $(x.id) has a dangling parent $(x.parent)")
            continue
        end
        if x.parent >= x.id
            push!(issues, "node $(x.id) has an out-of-order parent $(x.parent)")
        elseif x.depth != s.nodes[x.parent].depth + 1
            push!(issues,
                "node $(x.id) depth $(x.depth) disagrees with its parent")
        end
        x.visits >= 0 || push!(issues, "node $(x.id) has negative visits")
    end
    if s.selected_action !== nothing
        any(c -> c.action == s.selected_action, root_children(s)) ||
            push!(issues, "the selected action is not among the root children")
    end
    if action_space !== nothing
        for x in s.nodes
            x.action isa DuckieAction || continue
            x.action in action_space || push!(issues,
                "node $(x.id) action $(x.action) is outside the action space")
        end
    end
    return (ok=isempty(issues), nodes=n, roots=roots, issues=issues)
end

# ---------------------------------------------------------------------------
# Serialisation — so a figure can be redrawn with no solver installed
# ---------------------------------------------------------------------------

_action_to_json(::Nothing) = nothing
_action_to_json(a::MacroAction) = Dict("kind" => "macro", "id" => Int(a))
_action_to_json(a::DuckieAction) =
    Dict("kind" => "continuous", "v" => a.v, "omega" => a.omega)
_action_to_json(a) = Dict("kind" => "other", "repr" => string(a))

function _action_from_json(d)
    d === nothing && return nothing
    kind = String(d["kind"])
    kind == "macro" && return ALL_MACRO_ACTIONS[Int(d["id"]) + 1]
    kind == "continuous" &&
        return DuckieAction(Float64(d["v"]), Float64(d["omega"]))
    return String(d["repr"])
end

"""
    save_snapshot(path, snapshot)

Write a snapshot as JSON. `missing` values are preserved as `null` and read
back as `missing`, never as `0.0`.
"""
function save_snapshot(path::AbstractString, s::SearchSnapshot)
    doc = Dict(
        "schema" => "fj9.5b-search-snapshot-1",
        "id" => s.id, "solver" => s.solver,
        "state_fingerprint" => s.state_fingerprint,
        "config_fingerprint" => s.config_fingerprint,
        "planner_seed" => s.planner_seed,
        "root_visits" => s.root_visits,
        "selected_action" => _action_to_json(s.selected_action),
        "fingerprint" => snapshot_fingerprint(s),
        "extra" => Dict(String(k) => s.extra[k] for k in keys(s.extra)),
        "nodes" => [Dict("id" => n.id, "parent" => n.parent,
            "depth" => n.depth, "visits" => n.visits,
            "value" => n.value === missing ? nothing : n.value,
            "action" => _action_to_json(n.action)) for n in s.nodes])
    mkpath(dirname(path))
    open(path, "w") do io
        JSON3.pretty(io, doc)
    end
    return path
end

"""
    load_snapshot(path) -> SearchSnapshot

Read a snapshot back. The stored fingerprint is verified against the
recomputed one, so a tampered or truncated file is an error rather than a
plausible figure.
"""
function load_snapshot(path::AbstractString)
    isfile(path) || throw(ArgumentError("snapshot not found: $path"))
    doc = JSON3.read(read(path, String))
    String(doc.schema) == "fj9.5b-search-snapshot-1" ||
        throw(ArgumentError("unknown snapshot schema: $(doc.schema)"))
    nodes = [SearchNode(Int(n.id), Int(n.parent), Int(n.depth), Int(n.visits),
        n.value === nothing ? missing : Float64(n.value),
        _action_from_json(n.action)) for n in doc.nodes]
    extra = NamedTuple(Symbol(k) => v for (k, v) in pairs(doc.extra))
    s = SearchSnapshot(String(doc.id), String(doc.solver),
        String(doc.state_fingerprint), String(doc.config_fingerprint),
        Int(doc.planner_seed), Int(doc.root_visits),
        _action_from_json(doc.selected_action), nodes, extra)
    snapshot_fingerprint(s) == String(doc.fingerprint) || throw(ArgumentError(
        "snapshot fingerprint mismatch in $path: the file has been modified"))
    return s
end

"""
    search_statistics(snapshot) -> NamedTuple

How the search spent its budget at the root, computed from the snapshot alone.

`single_visit_fraction` is the quantity that distinguishes a search which
*evaluated* its actions from one which merely *sampled* them: the share of root
actions visited exactly once.
"""
function search_statistics(s::SearchSnapshot)
    rc = root_children(s)
    isempty(rc) && return (root_actions=0, root_children=0, mean_visits=NaN,
        median_visits=NaN, max_visits=0, min_visits=0,
        single_visit_fraction=NaN, total_root_visits=s.root_visits,
        distinct_actions=0)
    # Grouped by ACTION, not by child node. A vanilla MCTS tree has one root
    # child per simulation (states never merge, FJ8.2), so several children
    # repeat the same action and the same action-node visit count; counting
    # children would report that count many times over and distort every
    # statistic below.
    byaction = Dict{Any,Int}()
    for c in rc
        byaction[c.action] = max(get(byaction, c.action, 0), c.visits)
    end
    v = sort(collect(values(byaction)))
    n = length(v)
    return (root_actions=n,
        root_children=length(rc),
        mean_visits=sum(v) / n,
        median_visits=v[cld(n, 2)],
        max_visits=v[end],
        min_visits=v[1],
        single_visit_fraction=count(==(1), v) / n,
        total_root_visits=s.root_visits,
        distinct_actions=n)
end

"""
    visible_nodes(snapshot; max_depth, min_visits, top_k) -> Vector{Int}

Node ids a renderer should draw. **Display filtering only** — the snapshot is
never modified, and every filter is a property of the figure rather than of the
evidence. A node is kept only if its parent is kept, so the drawn subtree stays
connected.
"""
function visible_nodes(s::SearchSnapshot; max_depth::Integer=typemax(Int),
    min_visits::Integer=0, top_k::Integer=typemax(Int))
    isempty(s.nodes) && return Int[]
    keep = falses(length(s.nodes))
    keep[1] = true
    rc = root_children(s)
    allowed_root = Set(c.id for c in
        (top_k < length(rc) ?
         sort(rc; by=c -> -c.visits)[1:top_k] : rc))
    for n in s.nodes
        n.parent == 0 && continue
        keep[n.parent] || continue
        n.depth <= max_depth || continue
        n.visits >= min_visits || continue
        n.parent == 1 && !(n.id in allowed_root) && continue
        keep[n.id] = true
    end
    return findall(keep)
end

"""
    search_summary(snapshot) -> String
"""
function search_summary(s::SearchSnapshot)
    st = search_statistics(s)
    io = IOBuffer()
    println(io, s.solver, "  (", s.id, ")")
    println(io, "state      ", s.state_fingerprint)
    println(io, "config     ", s.config_fingerprint, "   seed ", s.planner_seed)
    println(io, "nodes      ", length(s.nodes), "   depth ",
        search_max_depth(s))
    println(io, "root       ", st.root_children, " child nodes, ",
        st.distinct_actions, " distinct actions, ", st.total_root_visits,
        " visits")
    println(io, "per action mean ", round(st.mean_visits; digits=2),
        "  median ", st.median_visits, "  max ", st.max_visits,
        "  once-only ", round(100 * st.single_visit_fraction; digits=1), "%")
    println(io, "selected   ", s.selected_action)
    if haskey(s.extra, :value_semantics)
        # wrapped: an unwrapped sentence runs off the edge of a figure, and a
        # provenance note that is clipped mid-word is not provenance
        words = split(String(s.extra.value_semantics))
        line = "value      "
        for w in words
            if length(line) + length(w) + 1 > 88
                println(io, line)
                line = "           " * w
            else
                line *= (endswith(line, " ") ? "" : " ") * w
            end
        end
        println(io, line)
    end
    return String(take!(io))
end

# ---------------------------------------------------------------------------
# FJ9.5a — what search data does the project actually have?
# ---------------------------------------------------------------------------
#
# FJ9.4's lesson, applied before any search figure is designed: audit the
# artefacts first. A renderer must never reconstruct a tree from aggregate
# counters — `tree_nodes = 259` and `root_children = 7` do not determine a
# tree, and a picture built from them would be an invention.

"""
    SearchDataStatus

`PERSISTED` — recoverable from a stored artefact.
`AGGREGATE_ONLY` — only a summary statistic exists; the underlying structure
does not, and cannot be recovered from the summary.
`ABSENT` — not recorded anywhere.
"""
@enum SearchDataStatus PERSISTED AGGREGATE_ONLY ABSENT

"""
    SearchDataItem

One quantity a search figure might need, with what the project actually holds.
"""
struct SearchDataItem
    quantity::String
    status::SearchDataStatus
    evidence::String
end

_has_column(path, names) = isfile(path) &&
    any(n -> n in strip.(split(first(eachline(path)), ",")), names)

_mentions(path, needle) = isfile(path) && occursin(needle, read(path, String))

"""
    search_artifact_audit(dir) -> Vector{SearchDataItem}

Probe `artifacts/fj8` for everything FJ9.5 would need to draw a search.

Executable rather than prose, like the FJ10 audit: if FJ9.5b later captures
snapshots, this reports `PERSISTED` and the test pinning the current answer
fails until the finding is updated.
"""
function search_artifact_audit(dir::AbstractString)
    budget = joinpath(dir, "budget_study.md")
    # Scan the whole artifacts tree, not just this directory: an audit that
    # only looks where the answer used to be "absent" would stay stale the
    # moment a capture lands somewhere else, which is exactly the failure an
    # executable audit exists to prevent.
    root = dirname(rstrip(dir, [Char(47)]))
    snapshots = String[]
    if isdir(root)
        for (base, _, files) in walkdir(root), f in files
            occursin("search", lowercase(f)) && endswith(f, ".json") &&
                push!(snapshots, relpath(joinpath(base, f), root))
        end
    end

    persisted(x) = !isempty(snapshots) ? PERSISTED : x
    items = SearchDataItem[]
    src = isempty(snapshots) ?
        "no search snapshot artefact exists; the episode CSVs carry no such columns" :
        "captured in " * join(snapshots, ", ")

    for q in ("node ids and parent/child edges", "per-node visit counts",
        "per-node Q estimates", "node depths",
        "root action labels (continuous v, omega)")
        push!(items, SearchDataItem(q, isempty(snapshots) ? ABSENT : PERSISTED,
            src))
    end

    push!(items, SearchDataItem("aggregate tree counters", AGGREGATE_ONLY,
        "tree_nodes / action_nodes appear in " * basename(budget) *
        " as prose table columns: " * string(_mentions(budget, "tree_nodes")) *
        "; they are means over decisions and do not determine a tree"))

    push!(items, SearchDataItem("per-decision diagnostics (PlanningDiagnostics.extra)",
        AGGREGATE_ONLY,
        "aggregated into PlannerCost.extra as means; episode_csv writes no " *
        "extra columns, so per-decision values are not persisted"))

    push!(items, SearchDataItem("planner configuration",
        isfile(joinpath(dir, "..", "..", "configs", "planning",
            "fj8_evaluation.yaml")) ? PERSISTED : ABSENT,
        "frozen in configs/planning/fj8_evaluation.yaml"))

    return items
end

"""
    search_audit_table(items) -> String
"""
function search_audit_table(items::AbstractVector{SearchDataItem})
    w = maximum(length(i.quantity) for i in items)
    io = IOBuffer()
    println(io, rpad("quantity", w), "  ", rpad("status", 16), "  evidence")
    println(io, "-"^(w + 60))
    for i in items
        println(io, rpad(i.quantity, w), "  ", rpad(string(i.status), 16),
            "  ", i.evidence)
    end
    return String(take!(io))
end

"""
    search_visualisation_supported(items) -> Bool

Whether a faithful search figure can be drawn from what exists today.
"""
search_visualisation_supported(items::AbstractVector{SearchDataItem}) =
    any(i -> i.status == PERSISTED &&
        occursin("node", i.quantity), items)
