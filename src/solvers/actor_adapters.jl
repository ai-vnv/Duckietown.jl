# FJ7.4b / FJ7.5b: native Julia inference for the SAC and TD3 reference actors.
#
# The weights come from the checkpoint via the oracle's `.npy` export and are
# read with the package's own native reader, so running a learned policy in
# Julia needs no Python and no PyTorch. The forward passes reproduce the
# reference definitions in `duckduck/src/agents/`:
#
#   SAC  SquashedGaussianActor
#        backbone: Linear(15,h) -> ReLU -> Linear(h,h) -> ReLU
#        heads:    mean = Linear(h,2), log_std = Linear(h,2) [clamp -5, 2]
#        deterministic evaluation (SACAgent.select_action(deterministic=True)):
#            tanh(mean) * action_scale + action_bias        -- NO clipping
#
#   TD3  DeterministicActor
#        net: Linear(15,h) -> ReLU -> Linear(h,h) -> ReLU -> Linear(h,2)
#        forward:  tanh(net(obs)) * action_scale + action_bias
#        evaluation (TD3Agent.select_action): the agent ALWAYS applies
#            np.clip(action, action_low, action_high)
#
# The clipping asymmetry is a real semantic difference between the two agents
# and is preserved, not smoothed over.
#
# Arithmetic is Float32 throughout, matching PyTorch. Layer outputs are kept
# so a divergence can be localised to the first layer where it appears rather
# than only observed at the action.

"""
    LinearLayer

One `torch.nn.Linear`: `y = W * x + b` with `W` of size `(out, in)`.
"""
struct LinearLayer
    W::Matrix{Float32}
    b::Vector{Float32}
end

(l::LinearLayer)(x::AbstractVector{Float32}) = l.W * x .+ l.b

relu(x::AbstractVector{Float32}) = max.(x, 0.0f0)

"""
    SACActorPolicy <: AbstractPolicy

Native SAC actor (deterministic evaluation).
"""
struct SACActorPolicy <: AbstractPolicy
    l0::LinearLayer
    l2::LinearLayer
    mean::LinearLayer
    log_std::LinearLayer
    action_scale::Vector{Float32}
    action_bias::Vector{Float32}
    dir::String
end

"""
    TD3ActorPolicy <: AbstractPolicy

Native TD3 actor (deterministic by construction).
"""
struct TD3ActorPolicy <: AbstractPolicy
    l0::LinearLayer
    l2::LinearLayer
    l4::LinearLayer
    action_scale::Vector{Float32}
    action_bias::Vector{Float32}
    action_low::Vector{Float32}
    action_high::Vector{Float32}
    dir::String
end

const SAC_LOG_STD_MIN = -5.0f0
const SAC_LOG_STD_MAX = 2.0f0

_npy32(dir, name) = convert(Array{Float32}, read_npy(joinpath(dir, name)))
_layer(dir, stem) = LinearLayer(_npy32(dir, stem * "_weight.npy"),
    vec(_npy32(dir, stem * "_bias.npy")))

"""
    SACActorPolicy(dir)

Load from a directory of exported parameter files (`backbone_0_weight.npy`,
`backbone_0_bias.npy`, `backbone_2_*`, `mean_*`, `log_std_*`,
`action_scale.npy`, `action_bias.npy`).
"""
function SACActorPolicy(dir::AbstractString)
    p = SACActorPolicy(_layer(dir, "backbone_0"), _layer(dir, "backbone_2"),
        _layer(dir, "mean"), _layer(dir, "log_std"),
        vec(_npy32(dir, "action_scale.npy")),
        vec(_npy32(dir, "action_bias.npy")), String(dir))
    size(p.l0.W, 2) == 15 ||
        throw(ArgumentError("SAC actor expects a 15-D observation, got " *
            "$(size(p.l0.W, 2))"))
    return p
end

"""
    TD3ActorPolicy(dir; action_low, action_high)
"""
function TD3ActorPolicy(dir::AbstractString;
    action_low=Float32[0.0, -1.5], action_high=Float32[0.41, 1.5])
    p = TD3ActorPolicy(_layer(dir, "net_0"), _layer(dir, "net_2"),
        _layer(dir, "net_4"), vec(_npy32(dir, "action_scale.npy")),
        vec(_npy32(dir, "action_bias.npy")),
        Float32.(action_low), Float32.(action_high), String(dir))
    size(p.l0.W, 2) == 15 ||
        throw(ArgumentError("TD3 actor expects a 15-D observation, got " *
            "$(size(p.l0.W, 2))"))
    return p
end

"""
    forward(policy, obs) -> NamedTuple

Full activation trace. Field names match the oracle's response keys so the
two can be compared layer by layer.
"""
function forward(p::SACActorPolicy, obs::AbstractVector{Float32})
    length(obs) == 15 || throw(ArgumentError("observation must be 15-D"))
    h0 = p.l0(obs)
    h1 = relu(h0)
    h2 = p.l2(h1)
    h3 = relu(h2)
    mean = p.mean(h3)
    log_std = p.log_std(h3)
    log_std_clamped = clamp.(log_std, SAC_LOG_STD_MIN, SAC_LOG_STD_MAX)
    tanh_mean = tanh.(mean)
    scaled = tanh_mean .* p.action_scale .+ p.action_bias
    return (layer0_linear=h0, layer1_relu=h1, layer2_linear=h2,
        layer3_relu=h3, mean=mean, log_std=log_std,
        log_std_clamped=log_std_clamped, tanh_mean=tanh_mean,
        scaled_action=scaled, agent_action=scaled)   # SAC does not clip
end

function forward(p::TD3ActorPolicy, obs::AbstractVector{Float32})
    length(obs) == 15 || throw(ArgumentError("observation must be 15-D"))
    h0 = p.l0(obs)
    h1 = relu(h0)
    h2 = p.l2(h1)
    h3 = relu(h2)
    pre_tanh = p.l4(h3)
    squashed = tanh.(pre_tanh)
    scaled = squashed .* p.action_scale .+ p.action_bias
    clipped = clamp.(scaled, p.action_low, p.action_high)
    return (layer0_linear=h0, layer1_relu=h1, layer2_linear=h2,
        layer3_relu=h3, layer4_linear=pre_tanh, pre_tanh=pre_tanh,
        tanh=squashed, scaled_action=scaled, agent_action=scaled,
        clipped_action=clipped)
end

"""
    act(policy, obs[, rng]) -> DuckieAction

The evaluation action: for SAC `tanh(mean)*scale+bias`; for TD3 the same
followed by the agent's clip. Deterministic — `rng` is never consumed.
"""
function act(p::SACActorPolicy, obs::AbstractVector{Float32},
    rng::AbstractRNG=Random.default_rng())
    a = forward(p, obs).agent_action
    return DuckieAction(Float64(a[1]), Float64(a[2]))
end

function act(p::TD3ActorPolicy, obs::AbstractVector{Float32},
    rng::AbstractRNG=Random.default_rng())
    a = forward(p, obs).clipped_action
    return DuckieAction(Float64(a[1]), Float64(a[2]))
end

"""
    POMDPs.action(policy, mdp, s) -> DuckieAction

Drive a learned continuous policy from a world state: project to the
15-component continuous state and encode it exactly as the reference does,
then run the actor.
"""
function POMDPs.action(p::Union{SACActorPolicy,TD3ActorPolicy},
    mdp::AnyMDPLike, s::DuckieWorldState;
    stop_hold_progress::Float64=0.0)
    raw, _ = get_raw_state(s, mdp.transition.state_cfg)
    cont = get_continuous_state(s, raw, mdp.transition.state_cfg,
        mdp.transition.continuous_cfg;
        controller_cfg=mdp.transition.duck_cfg,
        stop_hold_progress=stop_hold_progress)
    return act(p, encode_continuous_state(cont, mdp.transition.continuous_cfg))
end
