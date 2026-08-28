# FJ8.0 evidence that the generative model a planner will drive is pure Julia.
#
# Run in a FRESH process (see test/test_fj8_gen_native.jl): it loads the
# package, runs a few thousand `gen` calls, and then asserts that no Python
# module was ever loaded. Checking this inside the main test session would be
# meaningless — FJ5-R deliberately loads PythonCall there.

using DuckietownDecisionModels
using POMDPs
using Random

cfg = joinpath(pkgdir(DuckietownDecisionModels), "..", "duckduck", "policies",
    "q_learning", "training_config.yaml")
mdp = DuckietownMDP(cfg; action_space=:discrete)
s = rand(MersenneTwister(11), initialstate(mdp))

const N_CALLS = 5_000

# wrapped in a function: a bare top-level loop puts `acc` in soft scope
function run_calls(mdp, s, n)
    rng = MersenneTwister(3)
    acc = 0.0
    for i in 1:n
        a = ALL_MACRO_ACTIONS[mod1(i, 7)]
        acc += POMDPs.gen(mdp, s, a, rng).r
    end
    return acc
end

n = N_CALLS
acc = run_calls(mdp, s, n)

loaded = [string(m.name) for m in keys(Base.loaded_modules)]
python_like = filter(m -> occursin(r"^(PythonCall|CondaPkg|PyCall|Conda)$", m),
    loaded)

println("GEN_CALLS=", n)
println("REWARD_SUM_FINITE=", isfinite(acc))
println("PYTHON_MODULES=", isempty(python_like) ? "none" : join(python_like, ","))
println("EXT_LOADED=", any(m -> occursin("DuckietownPythonCallExt", m), loaded))
println("NATIVE_OK=", isempty(python_like) && isfinite(acc))
