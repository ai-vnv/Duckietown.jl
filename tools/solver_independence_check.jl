# FJ8.5 evidence that removing the solver removes nothing else.
#
# Run in a FRESH process whose environment is the PACKAGE project, where MCTS
# is only a weak dependency and therefore not installed. Checking this inside
# the test session would prove nothing: `Pkg.test` installs MCTS there on
# purpose so FJ8.2 can exercise the extension.
#
# It reports a fingerprint of the model, the reward, the evaluator and the
# tabular baseline. `tools/fj8_solver_present_check.jl` prints the same
# fingerprint with MCTS loaded; the two must agree exactly.

using DuckietownDecisionModels
using POMDPs
using Random

root = joinpath(pkgdir(DuckietownDecisionModels), "..", "duckduck")
cfg = joinpath(root, "policies", "q_learning", "training_config.yaml")
mdp = DuckietownMDP(cfg; action_space=:discrete)
s = rand(MersenneTwister(11), initialstate(mdp))

# model + reward fingerprint over every action at a fixed seed
rewards = [POMDPs.gen(mdp, s, a, MersenneTwister(3)).r for a in actions(mdp)]
succ = [POMDPs.gen(mdp, s, a, MersenneTwister(3)).sp.ego.pos[1]
        for a in actions(mdp)]

# the tabular baseline must decide identically
qpath = joinpath(root, "policies", "q_learning", "policy.npy")
qdec = "unavailable"
if isfile(qpath)
    qpol = QTablePolicy(qpath; solver=:q_learning)
    qdec = string(POMDPs.action(qpol, mdp, s))
end

# the shared evaluator must produce the same episode
struct FixedPolicy <: DuckietownDecisionModels.AbstractPolicy end
POMDPs.action(::FixedPolicy, ::DuckietownMDP, ::DuckieWorldState) = SLOW_STRAIGHT
ep = evaluate_policy(mdp, FixedPolicy(); seeds=1:2, max_steps=12)

loaded = [string(m.name) for m in keys(Base.loaded_modules)]

println("MCTS_LOADED=", any(==("MCTS"), loaded))
println("MCTS_EXT_LOADED=",
    Base.get_extension(DuckietownDecisionModels, :DuckietownMCTSExt) !== nothing)
println("N_ACTIONS=", length(actions(mdp)))
println("DISCOUNT=", repr(discount(mdp)))
println("REWARDS=", join(repr.(rewards), "|"))
println("SUCC_X=", join(repr.(succ), "|"))
println("QDECISION=", qdec)
println("EPISODE_RETURNS=", join(repr.([e.ret for e in ep]), "|"))
println("EPISODE_LENGTHS=", join(string.([e.decisions for e in ep]), "|"))
println("CAPABILITIES=", repr(model_capabilities(mdp)))
