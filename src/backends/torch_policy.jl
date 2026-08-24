# FJ7.4a / FJ7.5a: client for the PyTorch policy oracle.
#
# Deliberately separate from `ProcessReferenceBackend`: that one drives the
# Duckietown SIMULATOR in `ddm-ref` (python 3.9), this one only runs the
# reference ACTOR networks in the isolated `ddm-torch` env (python 3.11 +
# CPU torch). Keeping them apart means FJ7 never forces the validated
# simulator reference — or the PythonCall interpreter bound to it — onto a
# different Python.
#
#     DuckietownDecisionModels.jl
#              │ 15-D Float32 observation
#              ▼
#     TorchPolicyReferenceBackend  (out-of-process, ddm-torch)
#              ├── SAC actor
#              └── TD3 actor
#              ▼
#     per-layer activations + reference action

"""
    TorchPolicyReferenceBackend

Handle on the running `torch_policy_server.py` process. It answers `init`
(metadata + exports the actor weights as `.npy`) and `infer` (per-layer
activations for one 15-D observation).
"""
mutable struct TorchPolicyReferenceBackend
    proc::Base.Process
    open::Bool
    meta::Dict{String,Any}
end

const TORCH_CONDA_ENV = Ref("ddm-torch")

"""
    torch_policy_available() -> Bool

Whether the `ddm-torch` oracle can be started (WSL reachable and the env
present). Parity tests skip cleanly when it cannot.
"""
function torch_policy_available()
    try
        inner = "test -x ~/miniconda3/envs/$(TORCH_CONDA_ENV[])/bin/python && echo yes"
        cmd = `wsl.exe -d $(REFERENCE_WSL_DISTRO[]) -- bash -lc $inner`
        return strip(read(cmd, String)) == "yes"
    catch
        return false
    end
end

"""
    TorchPolicyReferenceBackend()

Launch the oracle. Uses the same clean-shutdown discipline as the simulator
backend (`close(proc.in)` then reap — `close(::Process)` alone leaves the
child running).
"""
function TorchPolicyReferenceBackend()
    script = "$(REFERENCE_REPO[])/tools/parity/torch_policy_server.py"
    inner = "source ~/miniconda3/etc/profile.d/conda.sh && " *
        "conda activate $(TORCH_CONDA_ENV[]) && " *
        "cd $(REFERENCE_REPO[]) && exec python $script"
    cmd = `wsl.exe -d $(REFERENCE_WSL_DISTRO[]) -- bash -lc $inner`
    proc = open(pipeline(cmd; stderr=devnull), "r+")
    return TorchPolicyReferenceBackend(proc, true, Dict{String,Any}())
end

function torch_call(b::TorchPolicyReferenceBackend, message::AbstractDict)
    b.open || error("torch policy backend is closed")
    println(b.proc, JSON3.write(message))
    flush(b.proc)
    line = readline(b.proc)
    isempty(line) && error("torch policy backend closed the connection")
    reply = JSON3.read(line)
    reply.ok || error("torch policy backend error:\n" * String(reply.error))
    return reply.result
end

function Base.close(b::TorchPolicyReferenceBackend)
    b.open || return nothing
    try
        torch_call(b, Dict("cmd" => "quit"))
    catch
    end
    b.open = false
    try
        close(b.proc.in)
    catch
    end
    try
        wait(b.proc)
    catch
    end
    return nothing
end

"""
    torch_policy_init!(backend, policy) -> Dict

Load `"sac"` or `"td3"` in the oracle and return its metadata: `obs_dim`,
`hidden`, action bounds, parameter shapes, the exported weight files, the
torch/numpy/python versions and the evaluation rule read from the reference
agent source.
"""
function torch_policy_init!(b::TorchPolicyReferenceBackend,
    policy::AbstractString)
    res = torch_call(b, Dict("cmd" => "init", "policy" => String(policy)))
    meta = Dict{String,Any}(String(k) => v for (k, v) in pairs(res))
    b.meta[String(policy)] = meta
    return meta
end

"""
    torch_policy_infer(backend, policy, obs) -> Dict

Reference activations for one 15-D observation: every linear/ReLU output, the
head outputs, the squashed action, the scaled action, the action the agent's
own `select_action(deterministic=true)` returns, and the clipped action.
"""
function torch_policy_infer(b::TorchPolicyReferenceBackend,
    policy::AbstractString, obs::AbstractVector)
    res = torch_call(b, Dict("cmd" => "infer", "policy" => String(policy),
        "obs" => Float64.(obs)))
    return Dict{String,Any}(String(k) => v for (k, v) in pairs(res))
end

"""
    torch_policy_infer_batch(backend, policy, observations) -> Vector{Dict}
"""
function torch_policy_infer_batch(b::TorchPolicyReferenceBackend,
    policy::AbstractString, observations::AbstractVector)
    res = torch_call(b, Dict("cmd" => "infer_batch",
        "policy" => String(policy),
        "obs" => [Float64.(o) for o in observations]))
    return [Dict{String,Any}(String(k) => v for (k, v) in pairs(row))
        for row in res.rows]
end
