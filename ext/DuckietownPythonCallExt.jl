# FJ5-R: the in-process reference backend.
#
# This extension loads only when the user does `using PythonCall`, so the
# native package stays Python-free.
#
# Design note — ONE source of semantics. The backend does NOT re-implement the
# reference interaction; it imports the very same `Session` class that the
# FJ5 JSON-lines server uses (`tools/parity/reference_server.py`) and calls
# its methods directly. `Session` speaks plain JSON-able dicts, so the only
# difference between the two transports is how those dicts travel:
#
#     ProcessReferenceBackend :  Julia --JSON over pipe--> Session
#     PythonCallReferenceBackend : Julia --pydict--> Session   (same process)
#
# Consequently the two backends cannot drift apart semantically, and the
# Julia-side state mapping (`world_to_ref` / `ref_to_world`) is shared
# verbatim.
module DuckietownPythonCallExt

using DuckietownDecisionModels
using PythonCall
using Random

const DDM = DuckietownDecisionModels

"""
    PythonCallRefBackend <: DDM.AbstractReferenceBackend

In-process handle: a live Python `Session` object plus the same metadata the
process backend carries.
"""
mutable struct PythonCallRefBackend <: DDM.AbstractReferenceBackend
    session::Py
    info::Dict{Symbol,Any}
    map::DDM.RoadMap
    open::Bool
end

"""
    _reference_module() -> Py

Import `tools/parity/reference_server.py` (the module, never its `main()`),
after putting the duckduck repository on `sys.path`. The module guards its
stdout hijack behind `claim_protocol_channel()`, which only server mode
calls — importing it here leaves Julia's stdout alone.
"""
function _reference_module()
    sys = pyimport("sys")
    os = pyimport("os")
    os.environ["PYGLET_HEADLESS"] = "1"
    tools = joinpath(pkgdir(DDM), "tools", "parity")
    # `pkgdir` may be a UNC path when the package is read from Windows; the
    # in-process backend only runs where Julia and Python share a filesystem,
    # so a plain absolute path is what we need here.
    for p in (tools, joinpath(pkgdir(DDM), "..", "duckduck"))
        pp = abspath(p)
        pycontains = pyconvert(Bool, sys.path.__contains__(pp))
        pycontains || sys.path.insert(0, pp)
    end
    return pyimport("reference_server")
end

# --- construction ---------------------------------------------------------

function DDM.PythonCallReferenceBackend(config::AbstractString;
    seed::Integer=53, action_space::Symbol=:discrete,
    overrides=Dict{String,Any}(), map::DDM.RoadMap=DDM.small_loop_map())
    mod = _reference_module()
    session = mod.Session()
    res = _from_py(session.init(String(config), Int(seed),
        String(action_space), _to_py(overrides)))::Dict{String,Any}
    info = Dict{Symbol,Any}(Symbol(k) => v for (k, v) in res)
    return PythonCallRefBackend(session, info, map, true)
end

# --- conversion helpers ---------------------------------------------------
#
# The Session schema is JSON-able: nested Dict/Vector of Float64/Int/Bool/
# String/nothing, plus the `{"nonfinite": tag}` markers the FJ5 encoder uses
# for NaN/Inf/-0.0. These helpers move that shape across without ever going
# through a JSON string.

_to_py(x::AbstractDict) = pydict(Dict(String(k) => _to_py(v) for (k, v) in x))
_to_py(x::AbstractVector) = pylist([_to_py(v) for v in x])
_to_py(x::Nothing) = pybuiltins.None
_to_py(x::Bool) = Py(x)
_to_py(x::Integer) = Py(Int(x))
_to_py(x::AbstractFloat) = Py(Float64(x))
_to_py(x::AbstractString) = Py(String(x))
_to_py(x) = Py(x)

function _from_py(x::Py)
    pyisinstance(x, pybuiltins.bool) && return pyconvert(Bool, x)
    pyisinstance(x, pybuiltins.int) && return pyconvert(Int, x)
    pyisinstance(x, pybuiltins.float) && return pyconvert(Float64, x)
    pyisinstance(x, pybuiltins.str) && return pyconvert(String, x)
    pyis(x, pybuiltins.None) && return nothing
    if pyisinstance(x, pybuiltins.dict)
        d = Dict{String,Any}()
        for k in pylist(x.keys())
            d[pyconvert(String, k)] = _from_py(x[k])
        end
        return d
    end
    if pyisinstance(x, pybuiltins.list) || pyisinstance(x, pybuiltins.tuple)
        return Any[_from_py(v) for v in pylist(x)]
    end
    return pyconvert(Any, x)
end
_from_py(x) = x

"""
    _wrap(dump) -> named access

`ref_to_world` and the FJ5 comparators address the reference dump with
property syntax (`s.ego.pos`), which JSON3 provides. The in-process path
returns plain `Dict`s, so they are wrapped in a tiny property-access view —
no schema duplication, just the same field names.
"""
struct RefView
    data::Dict{String,Any}
end

Base.getproperty(v::RefView, name::Symbol) =
    name === :data ? getfield(v, :data) : _view(getfield(v, :data)[String(name)])
Base.getindex(v::RefView, k) = _view(getfield(v, :data)[String(k)])
Base.haskey(v::RefView, k::Symbol) = haskey(getfield(v, :data), String(k))
Base.keys(v::RefView) = Symbol.(keys(getfield(v, :data)))
Base.propertynames(v::RefView) = keys(v)

# `{"nonfinite": tag}` is a LEAF value (NaN/Inf/-0.0), not a nested object:
# it must stay a plain Dict so the existing `x isa AbstractDict` decoders in
# `ref_to_world` and `compare_step` recognise it.
_view(x::Dict{String,Any}) =
    haskey(x, "nonfinite") ? x : RefView(x)
_view(x::AbstractVector) = Any[_view(v) for v in x]
_view(x) = x

# `ref_to_world` calls `_ref_num` on values that may be the nonfinite marker
# dict; RefView exposes those as RefView too, so unwrap them back to a Dict.
DDM._ref_num(x::RefView) = DDM._ref_num(getfield(x, :data))

# --- interface ------------------------------------------------------------

_call(b::PythonCallRefBackend, f, args...) = begin
    b.open || error("reference backend is closed")
    RefView(_from_py(f(args...))::Dict{String,Any})
end

function DDM.ref_reset!(b::PythonCallRefBackend, seed=nothing)
    res = _call(b, b.session.reset, seed === nothing ? pybuiltins.None : Int(seed))
    return DDM.ref_to_world(res, b.map), res
end

function DDM.ref_get_state(b::PythonCallRefBackend)
    res = _call(b, b.session.export_state)
    return DDM.ref_to_world(res, b.map), res
end

DDM.ref_set_state!(b::PythonCallRefBackend, w::DDM.DuckieWorldState) =
    _call(b, b.session.import_state, _to_py(DDM.world_to_ref(w)))

function DDM.ref_step!(b::PythonCallRefBackend, action)
    payload = if action isa DDM.MacroAction
        Py(Int(action))
    elseif action isa DDM.DuckieAction
        pylist([action.v, action.omega])
    else
        _to_py(action)
    end
    res = _call(b, b.session.step, payload)
    return DDM.ref_to_world(res, b.map), res
end

DDM.ref_probe_stop(b::PythonCallRefBackend; decisions::Integer=400,
    policy::AbstractString="lane_follow") =
    _call(b, b.session.probe_stop, Int(decisions), String(policy)).rows

function Base.close(b::PythonCallRefBackend)
    b.open || return nothing
    b.open = false
    # the Session holds a gym env; drop the reference and let Python collect it
    b.session = pybuiltins.None
    return nothing
end

end # module
