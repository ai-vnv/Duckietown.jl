# %% [markdown]
# # DuckietownDecisionModels.jl — quickstart
#
# Everything below runs with **no Python, no solver library, and no external
# data file**. The map is embedded in the package.
#
# What this notebook cannot do is reproduce the reported experiments: those
# need the frozen `training_config.yaml` and policy checkpoints, which live in
# the separate `duckduck` supplementary package. See the last section.

# %%
using DuckietownDecisionModels
using POMDPs
using Random

# %% [markdown]
# ## 1. Build a world
#
# `default_config` gives the Python source defaults, which contain **no stop
# sign** — the stop subsystem is inert there. Use `scenario_config` for a world
# that actually exercises the task.

# %%
cfg = scenario_config(:stop_and_duck)
mdp = DuckietownMDP(cfg; action_space=:discrete)

rng = MersenneTwister(1001)
s0 = rand(rng, initialstate(mdp))

println("actions      : ", length(POMDPs.actions(mdp)))
println("discount     : ", POMDPs.discount(mdp))
println("stop signs   : ", length(s0.stop_signs))
println("ducks        : ", length(s0.ducks))
println("ego (x, z)   : ", round.((s0.ego.pos[1], s0.ego.pos[3]); digits = 4))

# %% [markdown]
# ## 2. Step the generative model
#
# This is the standard POMDPs.jl interface — any solver that consumes `gen`
# can drive this model with no adapter.

# %%
# `simulate_decision` gives the whole `TransitionResult` — reward, event flags
# and the termination reason — where `gen` returns only `(sp, r)`.
function rollout(mdp, s, policy, n; rng = MersenneTwister(7))
    total, states, k, reason = 0.0, [s], 0, "horizon"
    for i in 1:n
        a = policy(mdp, s, rng)
        res = simulate_decision(mdp.transition, s, a, rng)
        total += res.reward.total
        s = res.sp
        push!(states, s)
        k = i
        if res.terminated || res.truncated
            reason = string(res.reason)
            break
        end
    end
    return (states = states, ret = total, steps = k, reason = reason)
end

# A hand-written lane follower: read the projection, pick a macro action.
# Not a trained policy — just enough to show the perception/action loop.
function lane_follower(mdp, s, rng)
    raw, _ = get_raw_state(s, mdp.transition.state_cfg)
    err = raw.d + 0.5 * raw.phi          # offset, nudged by heading
    err < -0.04 && return SLOW_LEFT
    err > 0.04 && return SLOW_RIGHT
    return abs(raw.phi) < 0.15 ? FAST_STRAIGHT : SLOW_STRAIGHT
end

res = rollout(mdp, s0, lane_follower, 150)

println("lane follower  : ", res.steps, " steps, return ",
    round(res.ret; digits = 3), ", ended by ", res.reason)

# For contrast: drive straight regardless of where the lane goes.
straight = rollout(mdp, s0, (m, s, r) -> FAST_STRAIGHT, 150)
println("always straight: ", straight.steps, " steps, return ",
    round(straight.ret; digits = 3), ", ended by ", straight.reason)

# Both leave the road, and the naive follower survives roughly twice as long
# while scoring notably worse. That is a real question, not a rhetorical one:
# a longer episode accumulating a worse return means the extra decisions are
# costing more than they earn. Section 4's projections and the diagnostic
# tools in the package are how you would find out which reward term is
# responsible — that workflow is what FJ9.6 was built for.
#
# Neither controller reacts to the stop sign or the duck at all. Handling
# those is what the trained policies are for, and what FJ8/FJ9 measure.

# %% [markdown]
# ## 3. Look at the state the policies actually consume
#
# The 7-component `RawState` is what the tabular policies see after
# discretization; the 15-component `ContinuousState` is what SAC/TD3 consume.
# Both are **privileged projections of the latent world**, not observations —
# FJ10 settled that distinction quantitatively.

# %%
raw, _ = get_raw_state(s0, mdp.transition.state_cfg)
println("d      = ", round(raw.d; digits = 4), "  (m, lateral offset)")
println("phi    = ", round(raw.phi; digits = 4), "  (rad, heading error)")
println("index  = ", discretize(raw))

for c in continuous_state_observability()[1:5]
    println(rpad(string(c.name), 20), c.class)
end

# %% [markdown]
# ## 4. Geometry without a plotting backend
#
# All scene geometry is computed in the core and is inspectable with no
# plotting package installed at all. A backend only draws it.

# %%
scene = world_scene(mdp, s0; trajectory = trajectory_points(res.states))
println("tiles          : ", length(scene.tiles))
println("lane segments  : ", length(scene.lane_centrelines))
println("stop lines     : ", length(scene.stop_lines))
println("footprint pts  : ", length(scene.ego_footprint))
println("trajectory pts : ", length(scene.trajectory))

# %% [markdown]
# ## 5. Draw it (optional)
#
# Add `CairoMakie` to the environment and the renderer becomes available. It
# renders on the CPU with no display, so it works inline in a notebook and in
# CI. Without it, everything above still works.

# %%
try
    @eval using CairoMakie
    CairoMakie.activate!(type = "png")
    fig = render_world(mdp, s0; trajectory = trajectory_points(res.states))
    save("quickstart_world.png", fig)
    println("wrote quickstart_world.png")
catch err
    println("CairoMakie not installed — skipping the figure.")
    println("  add it with: import Pkg; Pkg.add(\"CairoMakie\")")
end

# %% [markdown]
# ## 6. What needs the supplementary package
#
# The four trained policies and the exact evaluated environment are **not**
# shipped here. Without them you can explore the model, but you cannot
# reproduce a reported number.

# %%
for (what, path) in (("q-learning / SARSA table", "policies/q_learning/policy.npy"),
                     ("SAC / TD3 actor", "policies/td3/policy.pt"),
                     ("frozen evaluation config", "policies/q_learning/training_config.yaml"))
    println(rpad(what, 26), isfile(joinpath(pkgdir(DuckietownDecisionModels),
        path)) ? "available" : "NOT in this package")
end

println()
println("scenario_config(:stop_and_duck) is the right shape for the task, but")
println("it is NOT byte-identical to the frozen evaluation config: the reward")
println("weights and spawn settings there differ per algorithm.")
