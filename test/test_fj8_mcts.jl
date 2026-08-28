# FJ8.2 — an external, standard POMDPs.jl solver drives the model unchanged.
#
# The milestone is NOT that MCTS scores well. It is:
#
#   DuckietownDecisionModels.jl is usable by an external, standard POMDPs.jl
#   online-planning solver without changing the model.
#
# So the tests check the integration, not the search quality: the standard
# call sequence works with no conversion, the model comes out bit-identical,
# only native `gen` is used, and the planner drops into the shared FJ7.6
# evaluator. A negative control makes sure the integration could actually fail
# — that the planner is really reading this model and not being rubber-stamped.

using DuckietownDecisionModels
using POMDPs
using Test
using Random

const FJ82_OK = try
    @eval using MCTS
    true
catch err
    @info "FJ8.2: skipped (MCTS.jl not available)" err
    false
end

const FJ82_CFG = joinpath(pkgdir(DuckietownDecisionModels), "..", "duckduck",
    "policies", "q_learning", "training_config.yaml")

fj82_mdp() = DuckietownMDP(FJ82_CFG; action_space=:discrete)
fj82_state(m, seed=11) = rand(MersenneTwister(seed), initialstate(m))

"""The FJ8.2 baseline solver: vanilla MCTS, no tree reuse, default random
rollout, no learned guidance. Deliberately the plainest configuration that
exists, so a pass means the MODEL is usable — not that the planner was tuned."""
fj82_solver(; n=60, depth=12, seed=1) = MCTSSolver(n_iterations=n, depth=depth,
    exploration_constant=5.0, rng=MersenneTwister(seed), reuse_tree=false)

FJ82_OK && @testset "FJ8.2 the standard POMDPs.jl call sequence works" begin
    mdp = fj82_mdp()
    s = fj82_state(mdp)

    # exactly the sequence the ecosystem documents — no model conversion
    solver = fj82_solver()
    planner = solve(solver, mdp)
    @test planner isa POMDPs.Policy
    a = action(planner, s)
    @test a isa MacroAction
    @test a in actions(mdp)

    # the extension is loaded and is the only thing MCTS added
    @test Base.get_extension(DuckietownDecisionModels, :DuckietownMCTSExt) !== nothing

    # the planner works on the bare model AND on the instrumented wrapper,
    # because the wrapper is transparent
    imdp = InstrumentedMDP(mdp)
    ip = solve(fj82_solver(), imdp)
    @test action(ip, s) isa MacroAction
    @test model_calls(imdp) > 0
end

FJ82_OK && @testset "FJ8.2 planning does not disturb the model" begin
    mdp = fj82_mdp()
    imdp = InstrumentedMDP(mdp)
    s = fj82_state(imdp)
    snapshot = branch(s)

    planner = solve(fj82_solver(; n=80), imdp)
    a = action(planner, s)
    @test a in actions(mdp)

    # the root state the planner searched from is bit-identical afterwards
    diffs = world_differences(s, snapshot)
    @test isempty(diffs)
    isempty(diffs) || @info "planning mutated the root" diffs[1:min(5, end)]

    # the shared legacy stream was never advanced by the search
    @test rng_frozen([s], snapshot.controller_rng)

    # and the model still answers exactly as before the search
    for act in actions(mdp)
        x = gen(mdp, s, act, MersenneTwister(3))
        y = gen(mdp, snapshot, act, MersenneTwister(3))
        @test worlds_identical(x.sp, y.sp)
        @test x.r === y.r
    end
end

FJ82_OK && @testset "FJ8.2 the search runs on native Julia gen" begin
    mdp = fj82_mdp()
    imdp = InstrumentedMDP(mdp)
    s = fj82_state(imdp)

    for n in (20, 40, 80)
        reset_model_calls!(imdp)
        planner = solve(fj82_solver(; n=n), imdp)
        action(planner, s)
        used = model_calls(imdp)
        # every simulation consumes the generative model, and nothing else can
        # produce those calls — the reference backends are never constructed
        @test used >= n
        @info "FJ8.2 model consumption" iterations = n gen_calls = used calls_per_iteration =
            round(used / n; digits=2)
    end
end

FJ82_OK && @testset "FJ8.2 the planner is reproducible" begin
    mdp = fj82_mdp()
    s = fj82_state(mdp)

    # same solver seed, same state -> same decision
    a1 = action(solve(fj82_solver(; seed=7), mdp), s)
    a2 = action(solve(fj82_solver(; seed=7), mdp), s)
    @test a1 == a2

    # a fresh planner from the same solver spec behaves the same as a reused
    # one on a fresh tree (reuse_tree = false)
    p = solve(fj82_solver(; seed=7), mdp)
    @test action(p, s) == a1

    # the search really depends on its own rng: at least one other seed is
    # allowed to differ, and each seed is itself reproducible
    decisions = Dict(k => action(solve(fj82_solver(; seed=k), mdp), s)
                     for k in 1:6)
    for (k, a) in decisions
        @test action(solve(fj82_solver(; seed=k), mdp), s) == a
        @test a in actions(mdp)
    end
    @info "FJ8.2 decisions by solver seed" decisions
end

FJ82_OK && @testset "FJ8.2 diagnostics arrive through the generic slot" begin
    mdp = fj82_mdp()
    imdp = InstrumentedMDP(mdp)
    s = fj82_state(imdp)
    planner = solve(fj82_solver(; n=50), imdp)

    reset_model_calls!(imdp)
    a, d = plan_action(planner, imdp, s)
    @test a in actions(mdp)
    @test d isa PlanningDiagnostics
    @test d.planning_time > 0
    @test d.model_calls >= 50
    # MCTS-specific numbers live in `extra`; the core struct gained no fields
    @test d.extra.iterations == 50
    @test d.extra.depth_limit == 12
    @test d.extra.tree_nodes > 0
    @test d.extra.action_nodes > 0
    @test d.extra.root_visits > 0
    @test d.extra.root_children == 7
    @test fieldnames(PlanningDiagnostics) ==
        (:planning_time, :model_calls, :extra)
    @info "FJ8.2 tree statistics" d.extra
end

FJ82_OK && @testset "FJ8.2 the planner enters the shared evaluator unchanged" begin
    mdp = fj82_mdp()
    imdp = InstrumentedMDP(mdp)
    planner = solve(fj82_solver(; n=30, depth=8), imdp)

    # no MCTS-specific harness: the same function the four baselines use
    res = evaluate_planner(imdp, planner; seeds=1:2, max_steps=8)
    @test length(res.episodes) == 2
    @test all(m -> 1 <= m.decisions <= 8, res.episodes)
    @test all(m -> isfinite(m.ret), res.episodes)

    c = res.cost
    @test c.decisions == sum(m -> m.decisions, res.episodes)
    @test c.model_calls_per_action >= 30
    @test c.latency_p50 <= c.latency_p95 <= c.latency_max
    @test haskey(c.extra, :tree_nodes)
    @info "FJ8.2 planner cost through the shared evaluator" decisions = c.decisions latency_mean_ms =
        round(1e3 * c.latency_mean; digits=1)  latency_p95_ms =
        round(1e3 * c.latency_p95; digits=1)  gen_per_action =
        round(c.model_calls_per_action; digits=1)  mean_tree_nodes =
        round(c.extra[:tree_nodes]; digits=1)
end

FJ82_OK && @testset "FJ8.2 the ecosystem's own requirements check" begin
    # The compatibility claim should not rest on our word. POMDPLinter is the
    # POMDPs.jl ecosystem's requirements checker; its verdict on this model is
    # captured as an FJ8.2 artefact.
    ok = try
        @eval using POMDPLinter
        true
    catch err
        @info "FJ8.2 requirements check skipped (POMDPLinter unavailable)" err
        false
    end
    if !ok
        @test_skip "POMDPLinter unavailable"
    else
        mdp = fj82_mdp()
        s = fj82_state(mdp)
        planner = solve(fj82_solver(), mdp)

        # The requirements MCTS declares are attached to `action`, not to
        # `solve` (`get_requirements(solve, ...)` returns `Unspecified`).
        reqs = POMDPLinter.get_requirements(POMDPs.action, (planner, s))
        @test reqs isa POMDPLinter.RequirementSet

        lines = String[]
        unspecified = Ref(0)
        satisfied = Ref(0)
        missing_reqs = String[]
        function walk(r, indent="")
            if !(r isa POMDPLinter.RequirementSet)
                unspecified[] += 1
                push!(lines, indent * "(sub-requirements not declared by the solver)")
                return
            end
            push!(lines, indent * "requirer: " * string(r.requirer))
            for (f, types) in r.reqs
                impl = POMDPLinter.implemented(f, types)
                sig = string(f) * "(" * join(collect(types.parameters), ", ") * ")"
                push!(lines, indent * "   " * (impl ? "[OK] " : "[NO] ") * sig)
                impl ? (satisfied[] += 1) : push!(missing_reqs, sig)
            end
            for d in r.deps
                walk(d, indent * "      ")
            end
        end
        walk(reqs)

        # The verdict that matters: nothing the solver declares is missing.
        @test isempty(missing_reqs)
        isempty(missing_reqs) || @info "unmet solver requirements" missing_reqs
        @test satisfied[] >= 5

        # `check_requirements` is FALSE here, and that is not a failure: the
        # tree contains `Unspecified` nodes because the rollout estimator does
        # not declare its own requirements. Recorded so the false is not read
        # as "the model is missing something".
        @test unspecified[] > 0

        dir = joinpath(pkgdir(DuckietownDecisionModels), "artifacts", "fj8")
        mkpath(dir)
        path = joinpath(dir, "mcts_requirements.txt")
        open(path, "w") do io
            println(io, "POMDPLinter requirements check")
            println(io, "solver : MCTSSolver (vanilla, reuse_tree=false)")
            println(io, "model  : DuckietownMDP{MacroAction}")
            println(io, "entry  : POMDPs.action(planner, state)")
            println(io, "generated by test/test_fj8_mcts.jl")
            println(io)
            for l in lines
                println(io, l)
            end
            println(io)
            println(io, "satisfied              : ", satisfied[])
            println(io, "missing                : ", length(missing_reqs))
            println(io, "undeclared sub-requirement sets : ", unspecified[])
            println(io)
            println(io, "Notes")
            println(io, "  * check_requirements() returns false only because of the")
            println(io, "    undeclared sub-requirement sets above, not because any")
            println(io, "    declared requirement is unmet.")
            println(io, "  * show_requirements() cannot render this tree: MCTS.jl's own")
            println(io, "    @POMDP_require block raises UndefVarError(:s) when printed.")
            println(io, "    That is an upstream rendering defect, not a model defect;")
            println(io, "    the tree above was walked directly instead.")
            println(io, "  * isequal/hash on DuckieWorldState resolve to the identity")
            println(io, "    defaults, so tree states never merge. Measured consequence")
            println(io, "    recorded in the state-identity test set.")
        end
        @test isfile(path)
        @info "FJ8.2 requirements check\n" * join(lines, "\n")

        # the upstream rendering defect, pinned so a future MCTS release that
        # fixes it is noticed rather than silently assumed
        rendered = try
            tmp = tempname()
            open(tmp, "w") do io
                redirect_stdout(() -> POMDPLinter.show_requirements(reqs), io)
            end
            read(tmp, String)
        catch err
            "UNRENDERABLE: " * string(typeof(err))
        end
        @info "FJ8.2 show_requirements rendering" status =
            startswith(rendered, "UNRENDERABLE") ? rendered : "rendered"
    end
end

FJ82_OK && @testset "FJ8.2 state identity: a measured consequence, not a blocker" begin
    # `DuckieWorldState` is a mutable struct, so `hash`/`==` are identity-based
    # and every successor `gen` produces is a distinct key. MCTS.jl's tree
    # therefore never MERGES states: each simulation contributes exactly one
    # new state node. Recorded here as a measurement because it explains the
    # tree shape, and deliberately NOT "fixed" — defining structural equality
    # on the canonical state is a formulation change, and the integration does
    # not need it.
    mdp = fj82_mdp()
    imdp = InstrumentedMDP(mdp)
    s = fj82_state(imdp)
    for n in (25, 50, 100)
        planner = solve(fj82_solver(; n=n), imdp)
        _, d = plan_action(planner, imdp, s)
        @test d.extra.tree_nodes == n + 1        # root + one per simulation
        @test d.extra.root_visits == n
    end

    # the mechanism, stated directly
    x = gen(mdp, s, FAST_STRAIGHT, MersenneTwister(1)).sp
    y = gen(mdp, s, FAST_STRAIGHT, MersenneTwister(1)).sp
    @test worlds_identical(x, y)      # same content ...
    @test x !== y                     # ... different objects ...
    @test hash(x) != hash(y)          # ... so different tree keys
end

FJ82_OK && @testset "FJ8.2 negative control: the planner really reads THIS model" begin
    # If the integration were a rubber stamp — a wrapper that ignores the model
    # and returns something plausible — perturbing the reward would change
    # nothing. It must change the search.
    base = fj82_mdp()
    s = fj82_state(base)

    # a NEW config object with one reward coefficient inverted and amplified.
    # The shipped baseline file is untouched — this perturbation exists only in
    # memory, for the control, exactly as a corrected scenario would have to be
    # a separate named configuration rather than an edit.
    replace_field(x::T, name::Symbol, value) where {T} =
        T((f === name ? value : getfield(x, f) for f in fieldnames(T))...)

    cfg = base.config
    rc = cfg.reward
    flipped = replace_field(rc, :alpha_lateral, -50.0 * rc.alpha_lateral)
    perturbed = DuckietownMDP(replace_field(cfg, :reward, flipped);
        action_space=:discrete)

    @test perturbed.transition.reward_cfg.alpha_lateral !=
        base.transition.reward_cfg.alpha_lateral
    @test base.transition.reward_cfg.alpha_lateral == rc.alpha_lateral

    # the perturbation must be visible in the model itself ...
    xb = gen(base, s, FAST_STRAIGHT, MersenneTwister(1))
    xp = gen(perturbed, s, FAST_STRAIGHT, MersenneTwister(1))
    @test worlds_identical(xb.sp, xp.sp)      # dynamics unchanged
    @test xb.r != xp.r                        # reward changed

    # ... and it must reach the planner's decisions
    changed = 0
    for seed in 1:8
        st = fj82_state(base, seed)
        ab = action(solve(fj82_solver(; n=60, seed=3), base), st)
        ap = action(solve(fj82_solver(; n=60, seed=3), perturbed), st)
        ab == ap || (changed += 1)
    end
    @test changed > 0
    @info "FJ8.2 negative control" states_whose_decision_changed = changed  of = 8
end
