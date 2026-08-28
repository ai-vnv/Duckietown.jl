# FJ7.4 / FJ7.5 — SAC and TD3 actor inference parity.
#
#   Julia native actor   vs   PyTorch reference actor (ddm-torch oracle)
#
# The oracle runs the REAL reference classes (`SquashedGaussianActor`,
# `DeterministicActor` from `duckduck/src/agents/`) with the frozen
# checkpoints, in eval mode, under `no_grad`. It also exports every actor
# parameter as `.npy`, which the Julia side reads with its own native reader —
# so the weights cross bit-exactly and Julia needs no PyTorch.
#
# The comparison is LAYER BY LAYER, not just on the final action, so any
# divergence can be localised to the first layer where it appears:
#
#   SAC : layer0 -> relu -> layer2 -> relu -> {mean, log_std}
#         -> tanh(mean) -> scale/bias -> agent action        (no clip)
#   TD3 : layer0 -> relu -> layer2 -> relu -> layer4
#         -> tanh -> scale/bias -> agent action -> clip      (always clipped)
#
# Acceptance follows the FJ5/FJ6 philosophy: weights and inputs must be
# bit-exact; activations may differ at BLAS/vectorisation scale, and what
# matters is that the difference stays negligible and never changes the
# action materially. No blanket `isapprox` over everything.

using DuckietownDecisionModels
using POMDPs
using Test
using Random

const FJ7A_OK = torch_policy_available()

if !FJ7A_OK
    @info "FJ7.4/7.5: skipped (needs the ddm-torch env)"
end

_f32(v) = Float32[Float64(x) for x in v]

"""Worst absolute difference between a Julia vector and the oracle's list."""
function maxabs(jl::AbstractVector, py)
    m = 0.0
    for (i, x) in enumerate(jl)
        m = max(m, abs(Float64(x) - Float64(py[i])))
    end
    return m
end

"""Observation set: reference-shaped, boundary and random cases."""
function observation_suite(n_random::Int)
    lo = Float32[-1, -1, 0, -1, 0, 0, 0, 0, -1, -1, -1, -1, 0, 0, 0]
    hi = ones(Float32, 15)
    obs = Vector{Vector{Float32}}()
    push!(obs, zeros(Float32, 15))                       # zero vector
    push!(obs, copy(lo))                                 # lower bound
    push!(obs, copy(hi))                                 # upper bound
    push!(obs, (lo .+ hi) ./ 2)                          # midpoint
    # near-bound perturbations
    for eps in (1.0f-7, 1.0f-3)
        push!(obs, clamp.(lo .+ eps, lo, hi))
        push!(obs, clamp.(hi .- eps, lo, hi))
    end
    # one-hot style: each coordinate at its extreme, the rest at zero
    for k in 1:15
        v = zeros(Float32, 15); v[k] = hi[k]; push!(obs, copy(v))
        v = zeros(Float32, 15); v[k] = lo[k]; push!(obs, copy(v))
    end
    # random valid observations inside the declared box
    rng = MersenneTwister(20260819)
    for _ in 1:n_random
        push!(obs, lo .+ (hi .- lo) .* rand(rng, Float32, 15))
    end
    return obs
end

FJ7A_OK && @testset "FJ7.4a/7.5a torch oracle + weight export" begin
    b = TorchPolicyReferenceBackend()
    try
        @test torch_call(b, Dict("cmd" => "ping")).pong == true
        for (name, params) in ("sac" => ["backbone.0.weight", "backbone.0.bias",
                "backbone.2.weight", "backbone.2.bias", "mean.weight",
                "mean.bias", "log_std.weight", "log_std.bias",
                "action_scale", "action_bias"],
            "td3" => ["net.0.weight", "net.0.bias", "net.2.weight",
                "net.2.bias", "net.4.weight", "net.4.bias",
                "action_scale", "action_bias"])
            meta = torch_policy_init!(b, name)
            @test meta["obs_dim"] == 15
            @test meta["hidden"] == 256
            @test meta["eval_mode"] == true
            @test Float64.(meta["action_low"]) == [0.0, -1.5]
            @test isapprox(Float64(meta["action_high"][1]), 0.41; atol=1e-7)
            @test Float64(meta["action_high"][2]) == 1.5
            shapes = meta["param_shapes"]
            for p in params
                @test haskey(shapes, Symbol(p))
            end
            @test occursin("2.", String(meta["torch"]))   # torch 2.x
        end
    finally
        close(b)
    end
end

FJ7A_OK && @testset "FJ7.4b SAC inference parity (layer by layer)" begin
    b = TorchPolicyReferenceBackend()
    try
        meta = torch_policy_init!(b, "sac")
        pol = SACActorPolicy(String(meta["weights_dir"]))

        # weights crossed bit-exactly
        @test size(pol.l0.W) == (256, 15)
        @test size(pol.mean.W) == (2, 256)
        @test pol.action_scale ≈ Float32[0.205, 1.5]
        @test pol.action_bias ≈ Float32[0.205, 0.0]

        # Weight provenance, measured rather than assumed. On the zero
        # observation the first layer reduces to its bias; on a one-hot
        # observation it reduces to one weight column plus the bias, and the
        # 14 remaining products are exactly zero — so in both cases the sum is
        # exact in IEEE and any weight discrepancy would show up bitwise.
        zero_py = torch_policy_infer(b, "sac", zeros(Float32, 15))
        @test _f32(zero_py["layer0_linear"]) == pol.l0.b
        @test _f32(zero_py["layer0_linear"]) == forward(pol, zeros(Float32, 15)).layer0_linear
        for k in 1:15
            e = zeros(Float32, 15); e[k] = 1.0f0
            py_k = torch_policy_infer(b, "sac", e)
            @test _f32(py_k["layer0_linear"]) == pol.l0.W[:, k] .+ pol.l0.b
        end

        obs = observation_suite(962)
        @test length(obs) == 1000

        worst = Dict(k => 0.0 for k in ("layer0_linear", "layer1_relu",
            "layer2_linear", "layer3_relu", "mean", "log_std",
            "log_std_clamped", "tanh_mean", "scaled_action", "agent_action"))
        n_exact_action = 0
        for o in obs
            py = torch_policy_infer(b, "sac", o)
            jl = forward(pol, o)
            # the oracle received exactly the bits we sent
            @test _f32(py["obs"]) == o
            for k in keys(worst)
                worst[k] = max(worst[k], maxabs(getfield(jl, Symbol(k)), py[k]))
            end
            _f32(py["agent_action"]) == jl.agent_action && (n_exact_action += 1)
        end

        # provenance: differences must grow only through the network, and stay
        # at float32 round-off scale (measured, then bounded with headroom)
        @test worst["layer0_linear"] <= 1e-5
        @test worst["layer1_relu"] <= 1e-5
        @test worst["layer2_linear"] <= 1e-4
        @test worst["layer3_relu"] <= 1e-4
        @test worst["mean"] <= 1e-4
        @test worst["log_std"] <= 1e-4
        @test worst["tanh_mean"] <= 1e-4
        @test worst["scaled_action"] <= 1e-4
        @test worst["agent_action"] <= 1e-4
        @info "FJ7.4b SAC worst per-layer |Δ|" * "
" * join(["  $k = $(worst[k])" for k in sort(collect(keys(worst)))], "
") exact_actions = n_exact_action total = length(obs)

        # SAC evaluation does NOT clip: the reference's own clipped value must
        # equal its unclipped one for every observation (the tanh squash keeps
        # it inside the box by construction)
        for o in obs[1:50]
            py = torch_policy_infer(b, "sac", o)
            @test _f32(py["agent_action"]) == _f32(py["clipped_action"])
        end

        # the policy is usable as a POMDPs policy on the validated MDP
        mdp = DuckietownMDP(joinpath(pkgdir(DuckietownDecisionModels), "..",
            "duckduck", "policies", "sac", "training_config.yaml");
            action_space=:continuous)
        s = rand(MersenneTwister(73), initialstate(mdp))
        a = action(pol, mdp, s)
        @test a isa DuckieAction
        @test a in actions(mdp)
        @test action(pol, mdp, s) == a          # deterministic
    finally
        close(b)
    end
end

FJ7A_OK && @testset "FJ7.5b TD3 inference parity (layer by layer)" begin
    b = TorchPolicyReferenceBackend()
    try
        meta = torch_policy_init!(b, "td3")
        pol = TD3ActorPolicy(String(meta["weights_dir"]);
            action_low=Float32.(Float64.(meta["action_low"])),
            action_high=Float32.(Float64.(meta["action_high"])))

        @test size(pol.l0.W) == (256, 15)
        @test size(pol.l4.W) == (2, 256)

        # same weight-provenance measurement as SAC
        zero_py = torch_policy_infer(b, "td3", zeros(Float32, 15))
        @test _f32(zero_py["layer0_linear"]) == pol.l0.b
        for k in 1:15
            e = zeros(Float32, 15); e[k] = 1.0f0
            py_k = torch_policy_infer(b, "td3", e)
            @test _f32(py_k["layer0_linear"]) == pol.l0.W[:, k] .+ pol.l0.b
        end

        obs = observation_suite(962)
        @test length(obs) == 1000
        worst = Dict(k => 0.0 for k in ("layer0_linear", "layer1_relu",
            "layer2_linear", "layer3_relu", "layer4_linear", "pre_tanh",
            "tanh", "scaled_action", "agent_action", "clipped_action"))
        n_exact_action = 0
        n_clipped = 0
        for o in obs
            py = torch_policy_infer(b, "td3", o)
            jl = forward(pol, o)
            @test _f32(py["obs"]) == o
            for k in keys(worst)
                worst[k] = max(worst[k], maxabs(getfield(jl, Symbol(k)), py[k]))
            end
            _f32(py["clipped_action"]) == jl.clipped_action && (n_exact_action += 1)
            _f32(py["agent_action"]) != _f32(py["clipped_action"]) && (n_clipped += 1)
        end

        @test worst["layer0_linear"] <= 1e-5
        @test worst["layer2_linear"] <= 1e-4
        @test worst["layer4_linear"] <= 1e-4
        @test worst["pre_tanh"] <= 1e-4
        @test worst["tanh"] <= 1e-4
        @test worst["scaled_action"] <= 1e-4
        @test worst["clipped_action"] <= 1e-4
        @info "FJ7.5b TD3 worst per-layer |Δ|" * "
" * join(["  $k = $(worst[k])" for k in sort(collect(keys(worst)))], "
") exact_actions = n_exact_action total = length(obs) clipped = n_clipped

        mdp = DuckietownMDP(joinpath(pkgdir(DuckietownDecisionModels), "..",
            "duckduck", "policies", "td3", "training_config.yaml");
            action_space=:continuous)
        s = rand(MersenneTwister(73), initialstate(mdp))
        a = action(pol, mdp, s)
        @test a isa DuckieAction
        @test a in actions(mdp)
        @test action(pol, mdp, s) == a
    finally
        close(b)
    end
end

FJ7A_OK && @testset "FJ7.4/7.5 observations from real rollouts" begin
    # the observations a policy will actually see, not only synthetic ones
    b = TorchPolicyReferenceBackend()
    try
        mdp = DuckietownMDP(joinpath(pkgdir(DuckietownDecisionModels), "..",
            "duckduck", "policies", "sac", "training_config.yaml");
            action_space=:continuous)
        model = mdp.transition
        s = rand(MersenneTwister(73), initialstate(mdp))
        obs = Vector{Vector{Float32}}()
        rng = MersenneTwister(3)
        for _ in 1:60
            raw, _ = get_raw_state(s, model.state_cfg)
            cont = get_continuous_state(s, raw, model.state_cfg,
                model.continuous_cfg; controller_cfg=model.duck_cfg)
            push!(obs, encode_continuous_state(cont, model.continuous_cfg))
            r = simulate_decision(model, s, DuckieAction(0.2, 0.3 * (rand(rng) - 0.5)), rng)
            s = r.sp
            (r.terminated || r.truncated) && break
        end
        @test length(obs) >= 20

        for (name, ctor) in ("sac" => SACActorPolicy, "td3" => TD3ActorPolicy)
            meta = torch_policy_init!(b, name)
            pol = ctor(String(meta["weights_dir"]))
            worst_action = 0.0
            for o in obs
                py = torch_policy_infer(b, name, o)
                jl = forward(pol, o)
                key = name == "sac" ? "agent_action" : "clipped_action"
                got = name == "sac" ? jl.agent_action : jl.clipped_action
                worst_action = max(worst_action, maxabs(got, py[key]))
            end
            @test worst_action <= 1e-4
            @info "FJ7 rollout-observation action |Δ|" policy = name worst_action n = length(obs)
        end
    finally
        close(b)
    end
end
