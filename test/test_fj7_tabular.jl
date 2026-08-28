# FJ7.1 / FJ7.2 — tabular reference policies (Q-learning, SARSA).
#
# Question this gate answers: given the SAME discrete state, does the Julia
# adapter select the SAME action as the reference implementation, for every
# one of the 9 000 states?
#
# The greedy rule is not a plain `argmax`. The reference's reproducible
# evaluator (`q_policy_adapter.QPolicyAdapter`) reads the float32 row as
# float64, treats every allowed action within `atol = 1e-12` of the maximum as
# TIED, and selects the lowest such action id. (Its training-time evaluator
# resolves ties randomly, which is why the deterministic adapter exists at
# all.) The Julia port implements that rule verbatim, and this test compares
# against the real Python object — action ids, tie sets and Q-margins.
#
# The native half of the test needs no Python: the `.npy` reader is Julia.

using DuckietownDecisionModels
using POMDPs
using Test
using Random

const POLICIES = joinpath(pkgdir(DuckietownDecisionModels), "..", "duckduck",
    "policies")
qpath(name) = joinpath(POLICIES, name, "policy.npy")

const FJ7_PYCALL = try
    @eval using PythonCall
    occursin("ddm-ref", pyconvert(String, pyimport("sys").executable))
catch
    false
end

@testset "FJ7.1 native .npy loading (no Python involved)" begin
    for name in ("q_learning", "sarsa")
        p = QTablePolicy(qpath(name); solver=Symbol(name))
        @test size(p.table) == Q_SHAPE == (5, 5, 3, 3, 4, 2, 5, 7)
        @test eltype(p.table) == Float32
        @test all(isfinite, p.table)
        @test p.allowed_actions == collect(0:6)
        @test length(p.action_table) == 7
        @test p.action_table[1].name == "fast_left"
        @test p.action_table[7].name == "brake"
    end
    # guards mirror the reference adapter's own validation
    @test_throws ArgumentError QTablePolicy(qpath("q_learning");
        allowed_actions=Int[])
    @test_throws ArgumentError QTablePolicy(qpath("q_learning");
        allowed_actions=[0, 0, 1])
    @test_throws ArgumentError QTablePolicy(qpath("q_learning");
        allowed_actions=[0, 7])
end

@testset "FJ7.1 greedy rule semantics" begin
    p = QTablePolicy(qpath("q_learning"))
    idx = all_state_indices()
    @test length(idx) == 9000
    @test length(unique(idx)) == 9000
    @test first(idx) == (0, 0, 0, 0, 0, 0, 0)
    @test last(idx) == STATE_SHAPE .- 1

    # every state decides, and the decision is self-consistent
    for k in (1, 137, 4500, 9000)
        d = decide(p, idx[k])
        @test d.action_id in 0:6
        @test d.action == ALL_MACRO_ACTIONS[d.action_id + 1]
        @test d.action_id == minimum(d.ties)          # lowest_action_id
        @test !isempty(d.ties)
        best = maximum(d.q_values)
        for a in d.ties                                # every tie is within atol
            @test abs(d.q_values[a + 1] - best) <= TIE_ATOL
        end
        @test d.q_margin >= 0.0
    end

    # restricting the action set must be honoured
    restricted = QTablePolicy(qpath("q_learning"); allowed_actions=[5, 6])
    for k in (1, 900, 9000)
        @test decide(restricted, idx[k]).action_id in (5, 6)
    end

    # invalid indices are rejected
    @test_throws ArgumentError decide(p, (5, 0, 0, 0, 0, 0, 0))
    @test_throws ArgumentError decide(p, (0, 0, 0, 0, 0, 0, -1))
end

@testset "FJ7.1 policy drives the validated MDP" begin
    mdp = DuckietownMDP(joinpath(POLICIES, "q_learning", "training_config.yaml"))
    p = QTablePolicy(qpath("q_learning");
        allowed_actions=mdp.config.solver.allowed_actions,
        action_cfg=mdp.config.actions)
    s = rand(MersenneTwister(53), initialstate(mdp))
    a = action(p, mdp, s)
    @test a isa MacroAction
    @test a in actions(mdp)
    # deterministic: same state -> same action, and no RNG is consumed
    @test action(p, mdp, s) == a
    raw, _ = get_raw_state(s, mdp.transition.state_cfg)
    @test act(p, raw) == a
    @test act(p, discretize(raw)) == a
    # a short greedy rollout runs end to end on the native model
    rng = MersenneTwister(7)
    w = s
    for _ in 1:20
        w = gen(mdp, w, action(p, mdp, w), rng).sp
        isterminal(mdp, w) && break
    end
    @test w.ego.step_count > 0
end

if !FJ7_PYCALL
    @info "FJ7: 9000-state reference parity skipped (needs PythonCall on ddm-ref)"
end

FJ7_PYCALL && @testset "FJ7.1/7.2 9000-state greedy parity vs the reference adapter" begin
    sys = pyimport("sys")
    dd = abspath(joinpath(pkgdir(DuckietownDecisionModels), "..", "duckduck"))
    pyconvert(Bool, sys.path.__contains__(dd)) || sys.path.insert(0, dd)
    adapter_mod = pyimport("src.explainability.q_policy_adapter")
    kinds = pyimport("src.explainability.schema").SolverKind

    idx = all_state_indices()
    for (name, kind) in (("q_learning", kinds.Q_LEARNING),
        ("sarsa", kinds.SARSA))
        jl = QTablePolicy(qpath(name); solver=Symbol(name))
        py = adapter_mod.QPolicyAdapter.from_checkpoint(qpath(name);
            solver_kind=kind)

        # the two sides really loaded the same numbers
        pytab = pyconvert(Array{Float32}, py.q_table)
        @test size(pytab) == size(jl.table)
        @test pytab == jl.table

        mismatches = NTuple{7,Int}[]
        tie_mismatches = 0
        margin_mismatches = 0
        for state in idx
            d = decide(jl, state)
            r = py.decide_index(pytuple(state))
            got = pyconvert(Int, r.action.action_id)
            got == d.action_id || push!(mismatches, state)
            diag = r.diagnostics
            pyties = pyconvert(Vector{Int}, diag[pystr("greedy_ties")])
            pyties == d.ties || (tie_mismatches += 1)
            pym = diag[pystr("q_margin")]
            if !pyis(pym, pybuiltins.None)
                pyconvert(Float64, pym) == d.q_margin || (margin_mismatches += 1)
            end
        end
        @test isempty(mismatches)
        @test tie_mismatches == 0
        @test margin_mismatches == 0

        # the action NAME/commands must agree too, not just the id
        for state in (idx[1], idx[2500], idx[9000])
            d = decide(jl, state)
            r = py.decide_index(pytuple(state))
            @test pyconvert(String, r.action.action_name) == d.spec.name
            @test pyconvert(Float64, r.action.v_cmd) == d.spec.v
            @test pyconvert(Float64, r.action.omega_cmd) == d.spec.omega
        end
    end
end

FJ7_PYCALL && @testset "FJ7.1 restricted action set parity" begin
    sys = pyimport("sys")
    dd = abspath(joinpath(pkgdir(DuckietownDecisionModels), "..", "duckduck"))
    pyconvert(Bool, sys.path.__contains__(dd)) || sys.path.insert(0, dd)
    adapter_mod = pyimport("src.explainability.q_policy_adapter")
    allowed = [1, 4, 6]
    jl = QTablePolicy(qpath("q_learning"); allowed_actions=allowed)
    py = adapter_mod.QPolicyAdapter.from_checkpoint(qpath("q_learning");
        allowed_actions=pytuple(allowed))
    bad = 0
    for state in all_state_indices()
        pyconvert(Int, py.decide_index(pytuple(state)).action.action_id) ==
            decide(jl, state).action_id || (bad += 1)
    end
    @test bad == 0
end
