# Is the Duckietown transition actually deterministic given (s, a)?
#
# model_capabilities reports consumes_rng=false, stochastic_outcomes=false,
# stochastic_state_fraction=0.0. If that holds, then
#
#     transition(mdp, s, a) = Deterministic(simulate_decision(...).sp)
#
# is EXACT — not an estimate — and DORA can consume the model through the
# documented `key` mechanism with no sampled kernel anywhere.
#
# That claim is load-bearing, so it gets checked rather than quoted.

using DuckietownDecisionModels
using POMDPs, Random, Printf

mdp = DuckietownMDP(scenario_config(:stop_and_duck); action_space = :discrete)
ACTS = collect(POMDPs.actions(mdp))
tr = mdp.transition

"""Same (s,a), many different rng streams. If the successor is identical every
time, the transition does not consume randomness."""
function determinism_check(; states = 60, seeds = 8)
    rng0 = MersenneTwister(11)
    identical = 0
    total = 0
    worst = 0.0
    for k in 1:states
        s = rand(rng0, initialstate(mdp))
        # walk a few steps so we test more than just spawn states
        for _ in 1:(k % 7)
            s = simulate_decision(tr, s, rand(rng0, ACTS), MersenneTwister(1)).sp
        end
        for a in ACTS
            ref = simulate_decision(tr, s, a, MersenneTwister(1))
            for sd in 2:seeds
                r = simulate_decision(tr, s, a, MersenneTwister(sd * 7919))
                total += 1
                same = r.sp.ego.pos == ref.sp.ego.pos &&
                       r.sp.ego.angle == ref.sp.ego.angle &&
                       r.reward.total == ref.reward.total &&
                       r.terminated == ref.terminated
                same && (identical += 1)
                d = maximum(abs.(collect(r.sp.ego.pos) .- collect(ref.sp.ego.pos)))
                worst = max(worst, d)
            end
        end
    end
    return identical, total, worst
end

id, tot, worst = determinism_check()
@printf("identical successors : %d / %d\n", id, tot)
@printf("largest position difference across rng streams : %.3e m\n", worst)
println(id == tot ? "=> the transition is DETERMINISTIC given (s, a)" :
                    "=> the transition DOES depend on the rng")

# Does the duck controller introduce randomness over a longer horizon?
"""p_cross drives a coin flip somewhere; check whether an EPISODE diverges."""
function episode_divergence(; episodes = 20, horizon = 40)
    div = 0
    for k in 1:episodes
        s0 = rand(MersenneTwister(1000 + k), initialstate(mdp))
        traj = map((1, 2)) do sd
            s = s0
            out = NTuple{2,Float64}[]
            r = MersenneTwister(sd * 104729)
            for t in 1:horizon
                res = simulate_decision(tr, s, ACTS[1 + t % length(ACTS)], r)
                push!(out, (res.sp.ego.pos[1], res.sp.ego.pos[3]))
                (res.terminated || res.truncated) && break
                s = res.sp
            end
            out
        end
        traj[1] == traj[2] || (div += 1)
    end
    return div
end

@printf("\nepisodes diverging between two rng streams : %d / 20\n",
        episode_divergence())

# And the capability report's own answer, for the record
println("\nmodel_capabilities says:")
for (k, v) in pairs(model_capabilities(mdp))
    k in (:consumes_rng, :stochastic_outcomes, :stochastic_state_fraction,
          :explicit_transition_distribution, :continuous_state) &&
        @printf("  %-34s %s\n", string(k), string(v))
end
