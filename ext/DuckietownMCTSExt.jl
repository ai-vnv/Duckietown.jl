"""
    DuckietownMCTSExt

FJ8.2 — the first *external* solver integration, and deliberately the smallest
one that can exist.

MCTS.jl already works with `DuckietownMDP` without this extension:

```julia
using DuckietownDecisionModels, MCTS
mdp     = DuckietownMDP("…/training_config.yaml")
planner = solve(MCTSSolver(n_iterations = 200, depth = 20), mdp)
a       = action(planner, s)
```

because the model implements the standard POMDPs.jl generative contract and
nothing else is required. This file therefore adds **no conversion, no wrapper
model and no convenience layer** — doing so would make the model's usability
depend on this package having anticipated the solver, which is exactly what
FJ8 is meant to disprove.

What it adds is one thing: a `plan_action` method that reports MCTS-specific
numbers through the open `extra` slot of `PlanningDiagnostics`, so tree
statistics reach the evaluator without the core ever learning what a tree is.

It must not, and does not, touch state, transition, reward, action bounds,
termination or discount.
"""
module DuckietownMCTSExt

using DuckietownDecisionModels
using POMDPs
using MCTS

const DDM = DuckietownDecisionModels

"""
    tree_statistics(planner) -> NamedTuple

Solver-specific numbers, read from whatever the planner actually exposes.

Only quantities the tree really records are reported. Vanilla `MCTSTree` keeps
no transition structure unless tree visualisation is enabled, so its realised
depth is not recoverable; `depth_limit` (the configured cap) is reported
instead of a fabricated `max_depth`. `DPWTree` does record transitions, so its
depth is computed.
"""
function tree_statistics(p::MCTS.MCTSPlanner)
    t = p.tree
    t === nothing && return (iterations=p.solver.n_iterations,
        depth_limit=p.solver.depth, tree_nodes=0, action_nodes=0,
        root_visits=0, root_children=0)
    return (
        iterations=p.solver.n_iterations,
        depth_limit=p.solver.depth,
        tree_nodes=length(t.s_labels),
        action_nodes=length(t.a_labels),
        root_visits=isempty(t.total_n) ? 0 : t.total_n[1],
        root_children=isempty(t.child_ids) ? 0 : length(t.child_ids[1]),
    )
end

function tree_statistics(p::MCTS.DPWPlanner)
    t = p.tree
    sol = p.solver
    t === nothing && return (iterations=sol.n_iterations,
        depth_limit=sol.depth, tree_nodes=0, state_nodes=0, action_nodes=0,
        root_visits=0, root_children=0, root_action_children=0,
        max_action_children=0, max_state_children=0, max_depth=0,
        unique_transitions=0, action_pw=sol.enable_action_pw,
        state_pw=sol.enable_state_pw, k_action=sol.k_action,
        alpha_action=sol.alpha_action, k_state=sol.k_state,
        alpha_state=sol.alpha_state)
    root_children = isempty(t.children) ? 0 : length(t.children[1])
    return (
        iterations=sol.n_iterations,
        depth_limit=sol.depth,
        # `tree_nodes` keeps the name vanilla MCTS uses; `state_nodes` is the
        # same quantity under the name the DPW literature uses.
        tree_nodes=length(t.s_labels),
        state_nodes=length(t.s_labels),
        action_nodes=length(t.a_labels),
        root_visits=isempty(t.total_n) ? 0 : t.total_n[1],
        root_children=root_children,
        root_action_children=root_children,
        # how wide action widening ever got at a single state ...
        max_action_children=isempty(t.children) ? 0 :
            maximum(length, t.children),
        # ... and how wide STATE widening ever got at a single state-action.
        # With `enable_state_pw = false` this must stay 1: one sampled
        # successor per state-action, which is the whole point of turning it
        # off on a transition kernel measured to be a point mass.
        max_state_children=isempty(t.n_a_children) ? 0 :
            maximum(t.n_a_children),
        max_depth=_dpw_depth(t),
        unique_transitions=length(t.unique_transitions),
        # the configuration echoed back, so a report can never disagree with
        # the solver that produced it
        action_pw=sol.enable_action_pw,
        state_pw=sol.enable_state_pw,
        k_action=sol.k_action,
        alpha_action=sol.alpha_action,
        k_state=sol.k_state,
        alpha_state=sol.alpha_state,
    )
end

"""Realised depth of a `DPWTree`, in state nodes, by breadth-first walk from
the root. Guarded against revisits so a merged state cannot loop."""
function _dpw_depth(t::MCTS.DPWTree)
    isempty(t.s_labels) && return 0
    seen = falses(length(t.s_labels))
    frontier = [1]
    seen[1] = true
    depth = 0
    while !isempty(frontier)
        next = Int[]
        for si in frontier
            si <= length(t.children) || continue
            for sanode in t.children[si]
                sanode <= length(t.transitions) || continue
                for (sp, _) in t.transitions[sanode]
                    (1 <= sp <= length(seen) && !seen[sp]) || continue
                    seen[sp] = true
                    push!(next, sp)
                end
            end
        end
        isempty(next) && break
        depth += 1
        frontier = next
    end
    return depth
end

"""
    plan_action(planner, mdp, s) -> (action, PlanningDiagnostics)

The generic contract from FJ8.1, with tree statistics filled in. Timing and
`model_calls` are read exactly as the default method does — this only adds
`extra`.
"""
function DDM.plan_action(p::MCTS.AbstractMCTSPlanner, m::DDM.AnyMDPLike,
    s::DuckieWorldState)
    before = model_calls(m)
    t0 = time_ns()
    a = POMDPs.action(p, s)
    dt = (time_ns() - t0) / 1e9
    after = model_calls(m)
    used = (before < 0 || after < 0) ? -1 : after - before
    return a, PlanningDiagnostics(dt, used, tree_statistics(p))
end


# ---------------------------------------------------------------------------
# FJ9.5b — capture a solver's own tree as a solver-neutral SearchSnapshot
# ---------------------------------------------------------------------------
#
# This is the ONLY place that knows what `MCTS.MCTSTree` and `MCTS.DPWTree`
# are. The core sees a `SearchSnapshot` and nothing else, which is what lets a
# search figure be redrawn with MCTS.jl absent.
#
# Nothing is inferred. A quantity the solver does not define is stored as
# `missing`; FJ9.5a established that reconstructing structure from summaries
# is invention, and the same rule applies field by field.

# Only fields with a stable textual form contribute: rollout estimators,
# callbacks and RNG objects print addresses or closures that would make the
# fingerprint differ between runs of the same configuration.
function _config_fingerprint(sol)
    parts = String[string(typeof(sol))]
    for f in fieldnames(typeof(sol))
        v = getfield(sol, f)
        v isa Union{Real,Bool,Symbol,AbstractString} || continue
        push!(parts, string(f, "=", v))
    end
    return string(hash(join(parts, ";")); base=16, pad=16)
end

"""
    capture_search(planner, state; id, planner_seed) -> SearchSnapshot

Convert a planner's own tree into the solver-neutral snapshot.

**Vanilla MCTS**: `MCTSTree` records each state's action children with their
visit counts and Q values, but records the action -> next-state edges only
when the solver was built with `enable_tree_vis = true`. Capture therefore
requires it; without it only the root's action children are recoverable and
the deeper structure is genuinely absent, not merely inconvenient.
"""
function DDM.capture_search(p::MCTS.MCTSPlanner, s::DuckieWorldState;
    id::AbstractString="", planner_seed::Integer=-1,
    selected_action=nothing)
    t = p.tree
    t === nothing && throw(ArgumentError(
        "the planner has no tree; call action(planner, state) first"))
    vis = t._vis_stats
    isempty(vis) && throw(ArgumentError(
        "MCTSTree records no action -> next-state edges unless the solver was " *
        "built with enable_tree_vis = true; rebuild the solver for capture " *
        "rather than inferring the structure"))

    root_si = get(t.state_map, s, 0)
    root_si == 0 && throw(ArgumentError(
        "the given state is not this tree's root"))

    # state node id -> snapshot node id, assigned breadth-first from the root
    nodes = SearchNode[]
    ids = Dict{Int,Int}()
    push!(nodes, SearchNode(1, 0, 0, t.total_n[root_si], missing, nothing))
    ids[root_si] = 1
    frontier = [root_si]
    depth = 0
    while !isempty(frontier)
        depth += 1
        next = Int[]
        for si in frontier
            si <= length(t.child_ids) || continue
            for sanode in t.child_ids[si]
                a = t.a_labels[sanode]
                q = t.q[sanode]
                n = t.n[sanode]
                # every next state this action reached, from the vis stats
                sps = [pr.second for (pr, _) in vis if pr.first == sanode]
                if isempty(sps)
                    push!(nodes, SearchNode(length(nodes) + 1, ids[si], depth,
                        n, q, a))
                else
                    for sp in sps
                        haskey(ids, sp) && continue
                        push!(nodes, SearchNode(length(nodes) + 1, ids[si],
                            depth, n, q, a))
                        ids[sp] = length(nodes)
                        push!(next, sp)
                    end
                end
            end
        end
        frontier = next
    end

    # The caller passes the action THIS search returned. Calling
    # `action(planner, state)` again would rebuild the tree with the RNG
    # already advanced and report a different search decision, which the
    # snapshot validator caught the first time this was written that way.
    selected = selected_action
    return SearchSnapshot(String(id), "MCTS.MCTSSolver",
        state_fingerprint(s), _config_fingerprint(p.solver),
        Int(planner_seed), t.total_n[root_si], selected, nodes,
        (value_semantics="mean of backed-up returns (MCTS.jl updates q += (return - q)/n, i.e. an incremental sample mean; the first value is seeded by init_Q, not by a rollout)",
            iterations=p.solver.n_iterations, depth_limit=p.solver.depth,
            exploration_constant=p.solver.exploration_constant,
            state_nodes=length(t.s_labels), action_nodes=length(t.a_labels),
            distinct_root_actions=length(unique(n.action for n in nodes if n.parent == 1))))
end

"""
    capture_search(planner::DPWPlanner, state; id, planner_seed)

`DPWTree` records `transitions` per state-action node, so the edges are
directly available and no capture-time configuration change is needed.
"""
function DDM.capture_search(p::MCTS.DPWPlanner, s::DuckieWorldState;
    id::AbstractString="", planner_seed::Integer=-1,
    selected_action=nothing)
    t = p.tree
    t === nothing && throw(ArgumentError(
        "the planner has no tree; call action(planner, state) first"))
    root_si = get(t.s_lookup, s, 0)
    root_si == 0 && throw(ArgumentError(
        "the given state is not this tree's root"))

    nodes = SearchNode[]
    ids = Dict{Int,Int}()
    push!(nodes, SearchNode(1, 0, 0, t.total_n[root_si], missing, nothing))
    ids[root_si] = 1
    frontier = [root_si]
    depth = 0
    while !isempty(frontier)
        depth += 1
        next = Int[]
        for si in frontier
            si <= length(t.children) || continue
            for sanode in t.children[si]
                a = t.a_labels[sanode]
                q = t.q[sanode]
                n = t.n[sanode]
                sps = sanode <= length(t.transitions) ?
                    unique(first.(t.transitions[sanode])) : Int[]
                if isempty(sps)
                    push!(nodes, SearchNode(length(nodes) + 1, ids[si], depth,
                        n, q, a))
                else
                    for sp in sps
                        haskey(ids, sp) && continue
                        push!(nodes, SearchNode(length(nodes) + 1, ids[si],
                            depth, n, q, a))
                        ids[sp] = length(nodes)
                        push!(next, sp)
                    end
                end
            end
        end
        frontier = next
    end

    # The caller passes the action THIS search returned. Calling
    # `action(planner, state)` again would rebuild the tree with the RNG
    # already advanced and report a different search decision, which the
    # snapshot validator caught the first time this was written that way.
    selected = selected_action
    sol = p.solver
    return SearchSnapshot(String(id), "MCTS.DPWSolver",
        state_fingerprint(s), _config_fingerprint(sol),
        Int(planner_seed), t.total_n[root_si], selected, nodes,
        (value_semantics="mean of backed-up returns (MCTS.jl updates q += (return - q)/n, i.e. an incremental sample mean; the first value is seeded by init_Q, not by a rollout)",
            iterations=sol.n_iterations, depth_limit=sol.depth,
            exploration_constant=sol.exploration_constant,
            k_action=sol.k_action, alpha_action=sol.alpha_action,
            enable_action_pw=sol.enable_action_pw,
            enable_state_pw=sol.enable_state_pw,
            state_nodes=length(t.s_labels), action_nodes=length(t.a_labels),
            distinct_root_actions=length(unique(n.action for n in nodes if n.parent == 1))))
end

end # module
