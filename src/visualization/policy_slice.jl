# FJ9.3 — policy, value and ambiguity slices, computed in the core.
#
# Same rule as FJ9.1 and FJ9.2: the core produces finished semantics, the
# backend only draws them. The backend never runs a policy — every decision in
# a slice comes from the validated adapters, so a figure cannot disagree with
# the evaluator.
#
# Two honesty requirements are structural here rather than editorial:
#
#   1. A slice over two feature dimensions is a POLICY-INPUT slice, not a
#      reachable-state manifold. Varying `d`, `phi`, `d_stop` and the duck
#      fields independently produces combinations no `DuckieWorldState` can
#      generate. `mode` records this as data, not as a caption someone might
#      drop.
#
#   2. `d` and `phi` are NOT two independent tabular axes. The discretizer
#      indexes on `bin(d)` and `bin(phi + d)` — the tracking error — so a
#      `d x phi` grid maps many cells onto the same tabular state. Every slice
#      records how many distinct states it actually touched, and each tabular
#      cell carries the index it collapsed to.

"""
    SliceMode

`FEATURE_SPACE` — axes are policy *input* features, swept independently.
Combinations are not guaranteed to correspond to any reachable latent world
state. A future `REACHABLE_STATES` mode would sample from the world instead;
it does not exist yet and must not be implied.
"""
@enum SliceMode FEATURE_SPACE

"""
    SLICE_FEATURE_SPACE_CAVEAT

The sentence a feature-space figure must carry. Held in the core so a renderer
cannot forget it or reword it into something weaker.
"""
const SLICE_FEATURE_SPACE_CAVEAT =
    "Policy-input feature-space slice; combinations are not guaranteed to " *
    "correspond to reachable latent world states."

"""
    SliceAxis
"""
struct SliceAxis
    field::Symbol
    values::Vector{Float64}
    unit::String
end

"""
    TabularSliceCell

One cell of a tabular slice.

`selected_action` comes from [`decide`](@ref) — the FJ7-validated near-tie rule
(`|Q - max| <= TIE_ATOL`, lowest action id) — and never from a plain `argmax`.
`qmax` is deliberately the **raw** maximum over the allowed actions, with no
tie-breaking mixed in, so the value surface and the policy overlay stay
separable. `tie_count` and `q_margin` are the ambiguity diagnostics: on the
shipped checkpoints 8 689 of 9 000 states are near-tied, so a
one-action-per-cell map looks far more decisive than the table actually is.
"""
struct TabularSliceCell
    selected_action::MacroAction
    action_id::Int
    qmax::Float64
    tie_count::Int
    q_margin::Union{Nothing,Float64}
    index::NTuple{7,Int}
end

"""
    ContinuousSliceCell

One cell of a continuous slice: the two commanded outputs, kept as separate
fields rather than folded into one ambiguous symbol.
"""
struct ContinuousSliceCell
    v_cmd::Float64
    omega_cmd::Float64
end

"""
    PolicySlice

A slice as data. `fixed` is part of the slice's identity, not an annotation:
two slices over the same grid with different fixed context are different
objects and get different fingerprints, so a stale artefact cannot be mistaken
for a current one.
"""
struct PolicySlice{C}
    policy_name::String
    x::SliceAxis
    y::SliceAxis
    fixed::Vector{Pair{Symbol,Any}}
    mode::SliceMode
    coordinate_note::String
    cells::Matrix{C}
    distinct_states::Int
end

const TabularPolicySlice = PolicySlice{TabularSliceCell}
const ContinuousPolicySlice = PolicySlice{ContinuousSliceCell}

"""
    slice_fingerprint(slice) -> String

Stable identity of a slice, including the fixed context. Changing one fixed
value changes the fingerprint, which is what stops an old figure from being
reused under new assumptions.
"""
function slice_fingerprint(s::PolicySlice)
    h = hash((s.policy_name, s.x.field, s.y.field, s.x.values, s.y.values,
        string(s.mode), [(k, string(v)) for (k, v) in s.fixed]))
    return string(h; base=16, pad=16)
end

"""
    fixed_context_lines(slice) -> Vector{String}

The fixed dimensions, rendered for display. Every dimension not on an axis
appears here — a two-dimensional figure that hides the other dimensions is not
an honest depiction of a seven- or fifteen-dimensional policy.
"""
fixed_context_lines(s::PolicySlice) =
    [string(k, " = ", _fixed_value(v)) for (k, v) in s.fixed]

# `Bool <: Real` in Julia, so booleans must be handled first or `false`
# renders as `0.0` and the context reads like a measurement.
_fixed_value(v::Bool) = v ? "true" : "false"
_fixed_value(v::Real) = string(round(Float64(v); digits=4))
_fixed_value(::Nothing) = "none"
_fixed_value(v) = string(v)

_axis(field, values, unit) = SliceAxis(field, collect(Float64.(values)), unit)

const _SLICE_UNITS = Dict(:d => "m", :phi => "rad", :v => "m/s",
    :d_stop => "m", :kappa => "1/m", :duck_longitudinal => "m",
    :duck_lateral => "m", :duck_v_longitudinal_relative => "m/s",
    :duck_v_lateral_relative => "m/s")

_unit_of(f) = get(_SLICE_UNITS, f, "")

# ---------------------------------------------------------------------------
# Tabular slices
# ---------------------------------------------------------------------------

const _TABULAR_BASE = Dict{Symbol,Any}(:d => 0.0, :phi => 0.0, :v => 0.10,
    :tile => STRAIGHT, :d_stop => nothing, :sigma_stop => false, :duck => NONE)

const TABULAR_SLICE_FIELDS = (:d, :phi, :v, :tile, :d_stop, :sigma_stop, :duck)

"""
    raw_state_grid(x, y, xs, ys, fixed) -> Matrix{RawState}

Build the `RawState` of every cell by varying two fields and holding the rest
at `fixed`, defaulting to a straight-lane, no-stop, no-duck situation. The
defaults are reported as fixed context, never left implicit.
"""
function raw_state_grid(x::Symbol, y::Symbol, xs, ys, fixed::AbstractDict)
    get_(f, xv, yv) = f === x ? xv : f === y ? yv :
        haskey(fixed, f) ? fixed[f] : _TABULAR_BASE[f]
    return [RawState(get_(:d, xv, yv), get_(:phi, xv, yv), get_(:v, xv, yv),
        get_(:tile, xv, yv), get_(:d_stop, xv, yv), get_(:sigma_stop, xv, yv),
        get_(:duck, xv, yv)) for yv in ys, xv in xs]
end

function _fixed_pairs(x, y, fixed::AbstractDict, all_keys, base)
    out = Pair{Symbol,Any}[]
    for f in all_keys
        (f === x || f === y) && continue
        push!(out, f => (haskey(fixed, f) ? fixed[f] : base[f]))
    end
    return out
end

"""
    policy_slice(policy::QTablePolicy, x, y; xs, ys, fixed, name) -> PolicySlice

Tabular policy / value / ambiguity slice.

Each cell's `RawState` goes through the **real discretizer** — including
`tracking_error = phi + d` — and then through the validated [`decide`](@ref).
The visualisation therefore inherits FJ7's near-tie semantics instead of
re-deriving them, and `distinct_states` records how many tabular states the
grid actually reached, which on a `d x phi` grid is far fewer than the number
of cells.
"""
function policy_slice(policy::QTablePolicy, x::Symbol, y::Symbol;
    xs=range(-0.25, 0.25; length=61), ys=range(-0.6, 0.6; length=61),
    fixed::AbstractDict=Dict{Symbol,Any}(), name::AbstractString="tabular")
    grid = raw_state_grid(x, y, xs, ys, fixed)
    cells = Matrix{TabularSliceCell}(undef, size(grid)...)
    seen = Set{NTuple{7,Int}}()
    for k in eachindex(grid)
        idx = discretize(grid[k])
        push!(seen, idx)
        dec = decide(policy, idx)
        qmax = maximum(dec.q_values[a + 1] for a in policy.allowed_actions)
        cells[k] = TabularSliceCell(dec.action, dec.action_id, qmax,
            length(dec.ties), dec.q_margin, idx)
    end
    note = "tabular axes are bin(d) and bin(phi + d), not $(x) and $(y) " *
           "directly; this grid collapses onto $(length(seen)) distinct " *
           "tabular states"
    return PolicySlice(String(name), _axis(x, xs, _unit_of(x)),
        _axis(y, ys, _unit_of(y)),
        _fixed_pairs(x, y, fixed, TABULAR_SLICE_FIELDS, _TABULAR_BASE),
        FEATURE_SPACE, note, cells, length(seen))
end

"""
    value_surface(slice) -> Matrix{Float64}

`V(s) = max_a Q(s, a)` over the allowed actions — the raw maximum, with no
tie-breaking applied.
"""
value_surface(s::PolicySlice{TabularSliceCell}) = [c.qmax for c in s.cells]

"""
    action_surface(slice) -> Matrix{Int}
"""
action_surface(s::PolicySlice{TabularSliceCell}) = [c.action_id for c in s.cells]

"""
    tie_surface(slice) -> Matrix{Int}

Number of actions within `TIE_ATOL` of the best. A one-action-per-cell policy
map cannot show this, and without it the map looks more decisive than the
table is.
"""
tie_surface(s::PolicySlice{TabularSliceCell}) = [c.tie_count for c in s.cells]

"""
    margin_surface(slice) -> Matrix{Float64}

`Q_(1) - Q_(2)` over the allowed actions; `0.0` where the top two are tied.
"""
margin_surface(s::PolicySlice{TabularSliceCell}) =
    [c.q_margin === nothing ? 0.0 : c.q_margin for c in s.cells]

# ---------------------------------------------------------------------------
# Continuous slices
# ---------------------------------------------------------------------------

const _CONTINUOUS_BASE = (d=0.0, phi=0.0, v=0.10, kappa=0.0,
    stop_present=false, d_stop=nothing, sigma_stop=false, duck_present=false,
    duck_longitudinal=0.0, duck_lateral=0.0,
    duck_v_longitudinal_relative=0.0, duck_v_lateral_relative=0.0,
    duck_active=false, duck_crossing_available=true, stop_hold_progress=0.0)

"""
    continuous_state_grid(x, y, xs, ys, fixed) -> Matrix{ContinuousState}
"""
function continuous_state_grid(x::Symbol, y::Symbol, xs, ys,
    fixed::AbstractDict)
    get_(f, xv, yv) = f === x ? xv : f === y ? yv :
        haskey(fixed, f) ? fixed[f] : getfield(_CONTINUOUS_BASE, f)
    fields = fieldnames(ContinuousState)
    return [ContinuousState((get_(f, xv, yv) for f in fields)...)
            for yv in ys, xv in xs]
end

"""
    policy_slice(policy, cfg, x, y; xs, ys, fixed, name) -> PolicySlice

Continuous policy slice for a SAC or TD3 actor.

Each cell is encoded with the same `encode_continuous_state` the policy sees at
run time and evaluated through the FJ7-validated actor, so the surface is the
policy's own mapping rather than a re-implementation. The two commanded
outputs stay in separate fields — `v_cmd` and `omega_cmd` — because folding
them into one symbol makes the figure ambiguous.
"""
function policy_slice(policy::Union{SACActorPolicy,TD3ActorPolicy},
    cfg::ContinuousStateConfig, x::Symbol, y::Symbol;
    xs=range(-0.25, 0.25; length=41), ys=range(-0.6, 0.6; length=41),
    fixed::AbstractDict=Dict{Symbol,Any}(),
    name::AbstractString=policy isa SACActorPolicy ? "sac" : "td3")
    grid = continuous_state_grid(x, y, xs, ys, fixed)
    cells = Matrix{ContinuousSliceCell}(undef, size(grid)...)
    for k in eachindex(grid)
        a = act(policy, encode_continuous_state(grid[k], cfg))
        cells[k] = ContinuousSliceCell(a.v, a.omega)
    end
    fixedpairs = Pair{Symbol,Any}[]
    for f in fieldnames(ContinuousState)
        (f === x || f === y) && continue
        push!(fixedpairs, f => (haskey(fixed, f) ? fixed[f] :
            getfield(_CONTINUOUS_BASE, f)))
    end
    note = "continuous policy input; the axes are two of the 15 encoded " *
           "features and the other 13 are held fixed"
    return PolicySlice(String(name), _axis(x, xs, _unit_of(x)),
        _axis(y, ys, _unit_of(y)), fixedpairs, FEATURE_SPACE, note, cells,
        length(cells))
end

"""
    v_surface(slice) -> Matrix{Float64}
"""
v_surface(s::PolicySlice{ContinuousSliceCell}) = [c.v_cmd for c in s.cells]

"""
    omega_surface(slice) -> Matrix{Float64}
"""
omega_surface(s::PolicySlice{ContinuousSliceCell}) =
    [c.omega_cmd for c in s.cells]

"""
    slice_summary(slice) -> String
"""
function slice_summary(s::PolicySlice)
    io = IOBuffer()
    println(io, s.policy_name, ": ", s.x.field, " x ", s.y.field,
        "  (", size(s.cells, 2), " x ", size(s.cells, 1), " cells)")
    println(io, "mode: ", s.mode, " — ", SLICE_FEATURE_SPACE_CAVEAT)
    println(io, s.coordinate_note)
    println(io, "fingerprint: ", slice_fingerprint(s))
    println(io, "fixed context:")
    for l in fixed_context_lines(s)
        println(io, "  ", l)
    end
    return String(take!(io))
end
