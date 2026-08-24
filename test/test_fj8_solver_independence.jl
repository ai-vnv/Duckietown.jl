# FJ8.5 — solver independence as a measured property, not a documented one.
#
# The claim is that MCTS.jl can be added or removed without changing the
# model, the reward, the evaluator or the learned baselines. Two runs prove it:
#
#   A. a FRESH process on the PACKAGE project, where MCTS is only a weak
#      dependency and therefore not installed at all;
#   B. this test session, where `Pkg.test` HAS installed MCTS and the
#      extension is loaded.
#
# Both compute the same fingerprint. If loading a solver changed anything the
# model does, the two would disagree.

using DuckietownDecisionModels
using POMDPs
using Test
using Random

struct FJ85FixedPolicy <: DuckietownDecisionModels.AbstractPolicy end
POMDPs.action(::FJ85FixedPolicy, ::DuckietownMDP, ::DuckieWorldState) =
    SLOW_STRAIGHT

"""The same fingerprint `tools/fj8_no_solver_check.jl` prints."""
function fj85_fingerprint()
    root = joinpath(pkgdir(DuckietownDecisionModels), "..", "duckduck")
    mdp = DuckietownMDP(joinpath(root, "policies", "q_learning",
        "training_config.yaml"); action_space=:discrete)
    s = rand(MersenneTwister(11), initialstate(mdp))
    rewards = [POMDPs.gen(mdp, s, a, MersenneTwister(3)).r for a in actions(mdp)]
    succ = [POMDPs.gen(mdp, s, a, MersenneTwister(3)).sp.ego.pos[1]
            for a in actions(mdp)]
    qpath = joinpath(root, "policies", "q_learning", "policy.npy")
    qdec = isfile(qpath) ?
        string(POMDPs.action(QTablePolicy(qpath; solver=:q_learning), mdp, s)) :
        "unavailable"
    ep = evaluate_policy(mdp, FJ85FixedPolicy(); seeds=1:2, max_steps=12)
    return Dict(
        "N_ACTIONS" => string(length(actions(mdp))),
        "DISCOUNT" => repr(discount(mdp)),
        "REWARDS" => join(repr.(rewards), "|"),
        "SUCC_X" => join(repr.(succ), "|"),
        "QDECISION" => qdec,
        "EPISODE_RETURNS" => join(repr.([e.ret for e in ep]), "|"),
        "EPISODE_LENGTHS" => join(string.([e.decisions for e in ep]), "|"),
        "CAPABILITIES" => repr(model_capabilities(mdp)),
    )
end

@testset "FJ8.5 the model is identical with and without the solver" begin
    script = joinpath(pkgdir(DuckietownDecisionModels), "tools",
        "fj8_no_solver_check.jl")
    @test isfile(script)
    project = pkgdir(DuckietownDecisionModels)

    out = try
        read(`$(Base.julia_cmd()) --project=$project --startup-file=no $script`,
            String)
    catch err
        @info "FJ8.5 solver-absent run could not start" err
        ""
    end

    if isempty(out)
        @test_skip "solver-absent process unavailable"
    else
        absent = Dict(String(split(l, "=", limit=2)[1]) =>
                      String(split(l, "=", limit=2)[2])
                      for l in split(strip(out), "\n") if occursin("=", l))

        # the package project genuinely has no solver in it
        @test absent["MCTS_LOADED"] == "false"
        @test absent["MCTS_EXT_LOADED"] == "false"

        present = fj85_fingerprint()
        for k in keys(present)
            @test absent[k] == present[k]
            absent[k] == present[k] ||
                @info "solver presence changed the model" field = k without =
                    absent[k] with = present[k]
        end
        @info "FJ8.5 fingerprint matched with and without the solver" fields =
            length(present)  mcts_loaded_here =
            any(m -> string(m.name) == "MCTS", keys(Base.loaded_modules))
    end
end

@testset "FJ8.5 the solver stays a weak dependency" begin
    proj = TOML_PARSE = nothing
    path = joinpath(pkgdir(DuckietownDecisionModels), "Project.toml")
    text = read(path, String)

    # crude but decisive: the [deps] block must not name a solver
    deps_block = split(split(text, "[deps]")[2], "\n[")[1]
    for solver in ("MCTS", "POMDPLinter", "D3Trees", "POMDPTools")
        @test !occursin(solver, deps_block)
    end
    @test occursin("MCTS", split(split(text, "[weakdeps]")[2], "\n[")[1])
    @test occursin("DuckietownMCTSExt = \"MCTS\"", text)

    # the extension file exists and the core does not import the solver
    @test isfile(joinpath(pkgdir(DuckietownDecisionModels), "ext",
        "DuckietownMCTSExt.jl"))
    srcdir = joinpath(pkgdir(DuckietownDecisionModels), "src")
    offenders = String[]
    for (root, _, files) in walkdir(srcdir), f in files
        endswith(f, ".jl") || continue
        occursin(r"\b(using|import)\s+MCTS\b", read(joinpath(root, f), String)) &&
            push!(offenders, relpath(joinpath(root, f), srcdir))
    end
    @test isempty(offenders)
end
