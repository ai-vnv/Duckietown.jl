"""
    AbstractPolicy

Interface boundary for reference-policy adapters (`solvers/adapters.jl`) and
future solver policies. An adapter wraps one shipped artefact
(`policies/*/policy.npy` or `policy.pt`) and maps the appropriate projection
(discrete or encoded continuous observation) to an action.
"""
abstract type AbstractPolicy end

"""
    act(policy, observation, rng) -> action

Return the policy's action for the given observation (post-encoding
continuous vector for SAC/TD3 adapters, raw state for tabular adapters).
`rng` is required for stochastic exploration behaviour; deterministic
evaluation must not consume it.
"""
function act end