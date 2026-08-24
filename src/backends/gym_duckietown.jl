# FJ5.1 / FJ5-R: the Python reference backend clients.
#
# The reference stack is `ddm-ref` (Python 3.9 / numpy 1.20.0 / gym 0.23.1 /
# duckietown-gym-daffy 6.1.34) and is NEVER modified: both transports drive
# the same `Session` class in `tools/parity/reference_server.py`, which only
# reads the real wrapper's attributes and (for matched-state parity) writes
# back the documented mutable latent attributes.
#
# Two transports, same semantics:
#
# - FJ5 `ProcessReferenceBackend` — out-of-process JSON-lines server. This was
#   REQUIRED originally: the project ran a native Windows Julia against the
#   ELF/Linux conda env inside WSL, and PythonCall loads libpython
#   in-process, which cannot cross that boundary. It remains fully valid and
#   is the regression path (and the only option from Windows Julia).
#
# - FJ5-R `PythonCallReferenceBackend` — in-process, now possible because a
#   Linux Julia exists inside WSL alongside `ddm-ref`. Lives in the
#   `DuckietownPythonCallExt` package extension so PythonCall stays optional:
#   `using DuckietownDecisionModels` never touches Python.

"""
    AbstractReferenceBackend <: AbstractBackend

Common supertype of the reference (Python) backends. Two transports exist for
the SAME reference runtime and the same `Session` semantics:

- [`ProcessReferenceBackend`](@ref) — out-of-process JSON-lines server
  (`tools/parity/reference_server.py`). Works from a Windows Julia against a
  WSL Python; this is what FJ5 validated.
- `PythonCallReferenceBackend` — in-process via PythonCall, available when
  Julia and the reference Python live on the same platform (FJ5-R). Provided
  by the `DuckietownPythonCallExt` package extension, so PythonCall stays an
  optional dependency: `using DuckietownDecisionModels` never touches Python.

Both expose the same verbs — `ref_reset!`, `ref_get_state`, `ref_set_state!`,
`ref_step!`, `ref_probe_stop`, `close` — and share one state mapping
([`world_to_ref`](@ref) / [`ref_to_world`](@ref)), so callers only choose at
construction.
"""
abstract type AbstractReferenceBackend <: AbstractBackend end

"""
    PythonCallReferenceBackend(config; seed, action_space, overrides, map)

In-process reference backend (FJ5-R). Requires the PythonCall extension:

```julia
using PythonCall                      # loads DuckietownPythonCallExt
ref = PythonCallReferenceBackend("q_learning"; seed = 53)
```

Without PythonCall loaded this throws a message saying exactly that.
"""
function PythonCallReferenceBackend(args...; kwargs...)
    error("PythonCallReferenceBackend requires PythonCall: run `using " *
          "PythonCall` first (it loads the DuckietownPythonCallExt " *
          "extension). The reference interpreter must be the validated " *
          "`ddm-ref` env — set JULIA_CONDAPKG_BACKEND=Null and " *
          "JULIA_PYTHONCALL_EXE=\$(which python) with that env active.")
end

"""
    ProcessReferenceBackend <: AbstractReferenceBackend

Handle on a running reference-server process.

```julia
ref = ReferenceBackend("q_learning"; seed = 53)
s, _ = ref_reset!(ref, 53)                # DuckieWorldState from the reference
ref_set_state!(ref, world)                # inject a Julia state
sp, dump = ref_step!(ref, FAST_STRAIGHT)  # reference transition from it
close(ref)
```

[`reference_backend_available`](@ref) reports whether the backend can start at
all (WSL + the `ddm-ref` env present), so parity test sets skip cleanly on
machines without the reference environment.
"""
mutable struct ProcessReferenceBackend <: AbstractReferenceBackend
    proc::Base.Process
    info::Dict{Symbol,Any}
    map::RoadMap
    open::Bool
end

"""
    ReferenceBackend

Compatibility alias kept so every FJ5 caller and test keeps working:
`ReferenceBackend(...)` still constructs a [`ProcessReferenceBackend`](@ref).
"""
const ReferenceBackend = ProcessReferenceBackend

const REFERENCE_WSL_DISTRO = Ref("Ubuntu-Baru")
const REFERENCE_CONDA_ENV = Ref("ddm-ref")
const REFERENCE_REPO = Ref("/home/pannntastic/aivnv/DuckietownDecisionModels.jl")

"""
    reference_backend_available() -> Bool

Whether a reference server can be launched here (WSL distro reachable and the
`ddm-ref` python present).
"""
function reference_backend_available()
    try
        inner = "test -x ~/miniconda3/envs/$(REFERENCE_CONDA_ENV[])/bin/python && echo yes"
        cmd = `wsl.exe -d $(REFERENCE_WSL_DISTRO[]) -- bash -lc $inner`
        return strip(read(cmd, String)) == "yes"
    catch
        return false
    end
end

"""
    ReferenceBackend(config; seed, action_space=:discrete, overrides=Dict())

Launch the reference server and build the reference env from the experiment
YAML `config` (`"q_learning"`, `"sarsa"`, `"sac"`, `"td3"`). `overrides` is a
nested `section => Dict(key => value)` mapping applied on top of the YAML —
used only for explicitly-labelled variant scenarios; the baseline configs on
disk are never modified.
"""
function ProcessReferenceBackend(config::AbstractString; seed::Integer=53,
    action_space::Symbol=:discrete, overrides=Dict{String,Any}(),
    map::RoadMap=small_loop_map())
    script = "$(REFERENCE_REPO[])/tools/parity/reference_server.py"
    inner = "source ~/miniconda3/etc/profile.d/conda.sh && " *
        "conda activate $(REFERENCE_CONDA_ENV[]) && " *
        "cd $(REFERENCE_REPO[]) && exec python $script"
    cmd = `wsl.exe -d $(REFERENCE_WSL_DISTRO[]) -- bash -lc $inner`
    # `open(cmd, "r+")` gives a Julia-managed bidirectional pipe; hand-rolled
    # `Pipe()` plumbing crashed the GC on this Windows build.
    proc = open(pipeline(cmd; stderr=devnull), "r+")
    backend = ProcessReferenceBackend(proc, Dict{Symbol,Any}(), map, true)
    res = ref_call(backend, Dict("cmd" => "init", "config" => String(config),
        "seed" => Int(seed), "action_space" => String(action_space),
        "overrides" => overrides))
    backend.info = Dict(Symbol(k) => v for (k, v) in pairs(res))
    return backend
end

"""
    ref_call(backend, message) -> result

Send one protocol message and return its `result`, raising the reference
traceback as an `ErrorException` on failure.
"""
function ref_call(b::ProcessReferenceBackend, message::AbstractDict)
    b.open || error("reference backend is closed")
    println(b.proc, JSON3.write(message))
    flush(b.proc)
    line = readline(b.proc)
    isempty(line) && error("reference backend closed the connection")
    reply = JSON3.read(line)
    reply.ok || error("reference backend error:\n" * String(reply.error))
    return reply.result
end

"""
    close(backend)

Shut the reference process down cleanly: ask it to quit, close its **stdin**
so the server's `for line in sys.stdin` loop ends, then reap it.

`close(::Base.Process)` is NOT sufficient here — it returns successfully while
`process_running` stays `true`, leaving an orphaned child holding a live pipe.
On this Windows build that half-closed state later crashed the GC
(`EXCEPTION_ACCESS_VIOLATION` in `gc_mark_stack`) once a second backend was
created in the same session. Closing `proc.in` explicitly and waiting is what
actually terminates it.
"""
function Base.close(b::ProcessReferenceBackend)
    b.open || return nothing
    try
        ref_call(b, Dict("cmd" => "quit"))
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
    try
        close(b.proc.out)
    catch
    end
    return nothing
end

# ---------------------------------------------------------------------------
# FJ5.2: the state bridge (reference dump <-> DuckieWorldState)
# ---------------------------------------------------------------------------

_ref_num(x) = x === nothing ? nothing :
    x isa AbstractDict ? (x["nonfinite"] == "nan" ? NaN :
        x["nonfinite"] == "inf" ? Inf :
        x["nonfinite"] == "-inf" ? -Inf : -0.0) : Float64(x)

_ref_tuple3(v) = (_ref_num(v[1]), _ref_num(v[2]), _ref_num(v[3]))

_ref_matrix(v) = Float64[_ref_num(v[i][j]) for i in 1:length(v),
    j in 1:length(v[1])]

"""
    ref_to_world(dump, map) -> DuckieWorldState

Import a reference state dump into the canonical Julia world state. Every
dynamics-relevant field crosses: the DB18 pose/velocity matrices `q0`/`v0`,
the wheel-axis angles, the trimmed delayed-command window, each duckie's full
object state, the controller counters, and the stop/lane memory.
"""
function ref_to_world(s, map::RoadMap)
    e = s.ego
    history = Tuple{Float64,Float64,Float64}[
        (_ref_num(c[1]), _ref_num(c[2]), _ref_num(c[3])) for c in e.commands]
    q0 = _ref_matrix(e.q0)
    v0 = _ref_matrix(e.v0)
    # v_long / omega live in v0; the reference keeps no separate copy
    (v_long, _), omega = linear_angular_from_se2(v0)
    ego = DuckieEgoState(_ref_tuple3(e.pos), _ref_num(e.angle), v_long, omega,
        _ref_num(e.speed), Int(e.step_count), _ref_num(e.timestamp),
        history, q0, v0, _ref_num(e.axis_left_rad), _ref_num(e.axis_right_rad))

    ducks = [DuckieState(_ref_tuple3(d.pos), _ref_tuple3(d.center),
        _ref_tuple3(d.start), _ref_num(d.angle), _ref_tuple3(d.heading),
        _ref_num(d.vel), Bool(d.visible), Bool(d.active), _ref_num(d.wait),
        _ref_num(d.time), _ref_num(d.walk_distance), _ref_num(d.scale),
        _ref_num(d.safety_radius), _ref_tuple3(d.min_coords),
        _ref_tuple3(d.max_coords),
        [(_ref_num(c[1]), _ref_num(c[2])) for c in d.corners],
        _ref_matrix(d.norm)) for d in s.ducks]

    signs = [StopSignState(_ref_tuple3(sg.pos), _ref_num(sg.angle))
        for sg in s.signs]

    m = s.memory
    lid = m.last_stop_id === nothing ? nothing : Int(m.last_stop_id)
    memory = StopMemory(Bool(m.sigma_stop), Int(m.hold_steps), lid,
        _ref_num(m.last_d_stop))

    return DuckieWorldState(ego, ducks, signs, map, memory,
        (_ref_num(m.lane_fallback[1]), _ref_num(m.lane_fallback[2])),
        Int[Int(v) for v in s.controller.crossings_started],
        Bool[Bool(v) for v in s.controller.crossing_armed],
        MersenneTwister(0))
end

_ref_enc(x::Float64) = isnan(x) ? Dict("nonfinite" => "nan") :
    isinf(x) ? Dict("nonfinite" => x > 0 ? "inf" : "-inf") :
    (x == 0.0 && signbit(x)) ? Dict("nonfinite" => "-0.0") : x

_ref_enc_vec(v) = [_ref_enc(Float64(x)) for x in v]
_ref_enc_mat(m::AbstractMatrix) =
    [[_ref_enc(m[i, j]) for j in 1:size(m, 2)] for i in 1:size(m, 1)]

"""
    world_to_ref(world) -> Dict

Export a canonical world state in the reference server's schema, so the same
latent state `x_t` can be injected into the real Python simulator
(`set_state`) and stepped side by side with the native model.
"""
function world_to_ref(w::DuckieWorldState)
    e = w.ego
    return Dict(
        "ego" => Dict(
            "pos" => _ref_enc_vec(collect(e.pos)),
            "angle" => _ref_enc(e.angle),
            "speed" => _ref_enc(e.speed),
            "step_count" => e.step_count,
            "timestamp" => _ref_enc(e.timestamp),
            "state_t" => _ref_enc(e.timestamp),
            "inner_t0" => _ref_enc(e.timestamp),
            "q0" => _ref_enc_mat(e.q0),
            "v0" => _ref_enc_mat(e.v0),
            "axis_left_rad" => _ref_enc(e.axis_left_rad),
            "axis_right_rad" => _ref_enc(e.axis_right_rad),
            "commands" => [[_ref_enc(c[1]), _ref_enc(c[2]), _ref_enc(c[3])]
                for c in e.command_history],
        ),
        "ducks" => [Dict(
            "pos" => _ref_enc_vec(collect(d.pos)),
            "center" => _ref_enc_vec(collect(d.center)),
            "start" => _ref_enc_vec(collect(d.start)),
            "angle" => _ref_enc(d.angle),
            "heading" => _ref_enc_vec(collect(d.heading)),
            "vel" => _ref_enc(d.vel),
            "visible" => d.visible,
            "active" => d.pedestrian_active,
            "wait" => _ref_enc(d.pedestrian_wait_time),
            "time" => _ref_enc(d.time),
            "walk_distance" => _ref_enc(d.walk_distance),
            "scale" => _ref_enc(d.scale),
            "safety_radius" => _ref_enc(d.safety_radius),
            "min_coords" => _ref_enc_vec(collect(d.min_coords)),
            "max_coords" => _ref_enc_vec(collect(d.max_coords)),
            "corners" => [[_ref_enc(c[1]), _ref_enc(c[2])] for c in d.obj_corners],
            "norm" => _ref_enc_mat(d.obj_norm),
        ) for d in w.ducks],
        "signs" => [Dict("pos" => _ref_enc_vec(collect(sg.pos)),
            "angle" => _ref_enc(sg.angle)) for sg in w.stop_signs],
        "controller" => Dict(
            "crossings_started" => w.crossings_started,
            "crossing_armed" => w.crossing_armed,
        ),
        "memory" => Dict(
            "sigma_stop" => w.stop_memory.sigma_stop,
            "env_sigma_stop" => w.stop_memory.sigma_stop,
            "hold_steps" => w.stop_memory.hold_steps,
            "last_stop_id" => w.stop_memory.last_stop_id,
            "last_d_stop" => w.stop_memory.last_d_stop === nothing ? nothing :
                _ref_enc(w.stop_memory.last_d_stop),
            "lane_fallback" => _ref_enc_vec(collect(w.lane_fallback)),
        ),
    )
end

# ---------------------------------------------------------------------------
# Backend operations
# ---------------------------------------------------------------------------

"""
    ref_reset!(backend, seed=nothing) -> (world, dump)

Reference `reset()`; returns the sampled state as a [`DuckieWorldState`](@ref).
"""
function ref_reset!(b::ProcessReferenceBackend, seed=nothing)
    msg = Dict{String,Any}("cmd" => "reset")
    seed === nothing || (msg["seed"] = Int(seed))
    res = ref_call(b, msg)
    return ref_to_world(res, b.map), res
end

"""
    ref_get_state(backend) -> (world, dump)
"""
function ref_get_state(b::ProcessReferenceBackend)
    res = ref_call(b, Dict("cmd" => "get_state"))
    return ref_to_world(res, b.map), res
end

"""
    ref_set_state!(backend, world)

Inject a canonical world state into the live reference simulator (FJ5.2).
"""
ref_set_state!(b::ProcessReferenceBackend, w::DuckieWorldState) =
    ref_call(b, Dict("cmd" => "set_state", "state" => world_to_ref(w)))

"""
    ref_step!(backend, action) -> (world, dump)

One reference decision (`DuckieMDPEnv.step` / `ContinuousDuckieMDPEnv.step`).
`action` is a `MacroAction`/`Int` for the discrete env or a
[`DuckieAction`](@ref)/vector for the continuous one. `dump.result` carries
the reward breakdown, events, termination reason and wheel commands.
"""
function ref_step!(b::ProcessReferenceBackend, action)
    payload = if action isa MacroAction
        Int(action)
    elseif action isa DuckieAction
        [action.v, action.omega]
    else
        action
    end
    res = ref_call(b, Dict("cmd" => "step", "action" => payload))
    return ref_to_world(res, b.map), res
end

"""
    ref_probe_stop(backend; decisions=400, policy="lane_follow") -> rows

FJ5.4: record every stop-candidate filter quantity per decision from the LIVE
reference runtime (observation only; the baseline config is untouched).
"""
ref_probe_stop(b::ProcessReferenceBackend; decisions::Integer=400,
    policy::AbstractString="lane_follow") =
    ref_call(b, Dict("cmd" => "probe_stop", "decisions" => Int(decisions),
        "policy" => policy)).rows
