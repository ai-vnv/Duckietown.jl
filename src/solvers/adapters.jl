# FJ7.1/FJ7.2: inference adapters for the shipped tabular policies
# (`policies/q_learning/policy.npy`, `policies/sarsa/policy.npy`).
#
# The checkpoints are read-only inputs and are never rewritten. The `.npy`
# reader below is native Julia, so loading and running a reference policy
# needs no Python at all.
#
# The greedy rule is NOT a plain `argmax`. It is ported verbatim from the
# reference's own deterministic evaluator
# (`duckduck/src/explainability/q_policy_adapter.py::_decide_index`), because
# that is the only reproducible convention the reference defines — its
# training-time evaluator resolves exact ties RANDOMLY, which cannot be
# compared across implementations:
#
#     values         = q_table[index]            (float32 widened to float64)
#     allowed_values = values[allowed_actions]
#     best           = maximum(allowed_values)
#     ties           = allowed actions with |value - best| <= 1e-12   (atol,
#                      rtol = 0 — a NEAR-tie window, not exact equality)
#     selected       = minimum(ties)             ("lowest_action_id")
#     q_margin       = top1 - top2 over allowed_values
#
# The 1e-12 window matters: an action can win the raw maximum yet lose the
# decision to a lower-numbered action that is within 1e-12 of it.

"""
    read_npy(path) -> Array

Minimal reader for the NumPy `.npy` v1/v2 format restricted to what the
shipped checkpoints use: plain (non-pickled) little-endian numeric arrays.
Returns a Julia array with the file's own shape, honouring `fortran_order`.
"""
function read_npy(path::AbstractString)
    open(path, "r") do io
        magic = read(io, 6)
        magic == UInt8[0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59] ||
            throw(ArgumentError("not a .npy file: $path"))
        major = read(io, UInt8)
        read(io, UInt8)                                  # minor
        header_len = major == 0x01 ? Int(ltoh(read(io, UInt16))) :
            Int(ltoh(read(io, UInt32)))
        header = String(read(io, header_len))

        descr = match(r"'descr':\s*'([^']+)'", header)
        fortran = match(r"'fortran_order':\s*(True|False)", header)
        shape_m = match(r"'shape':\s*\(([^)]*)\)", header)
        (descr === nothing || fortran === nothing || shape_m === nothing) &&
            throw(ArgumentError("unsupported .npy header: $header"))

        dt = descr.captures[1]
        T = dt in ("<f4", "|f4", "=f4") ? Float32 :
            dt in ("<f8", "|f8", "=f8") ? Float64 :
            dt in ("<i8", "|i8") ? Int64 :
            dt in ("<i4", "|i4") ? Int32 :
            throw(ArgumentError("unsupported .npy dtype: $dt"))

        dims = [parse(Int, strip(s)) for s in
            split(strip(shape_m.captures[1]), ',') if !isempty(strip(s))]
        n = isempty(dims) ? 1 : prod(dims)
        raw = Vector{T}(undef, n)
        read!(io, raw)
        raw .= ltoh.(raw)

        isempty(dims) && return raw[1]
        if fortran.captures[1] == "True"
            return reshape(raw, dims...)
        end
        # C order: the LAST index varies fastest, so read into the reversed
        # shape and permute back
        return permutedims(reshape(raw, reverse(dims)...),
            reverse(1:length(dims)))
    end
end

"""
    QTablePolicy <: AbstractPolicy

A shipped tabular reference policy (Q-learning or SARSA) exposed through the
validated MDP. Holds the Q-table exactly as stored (`Float32`, shape
[`Q_SHAPE`](@ref)), the allowed action ids, and the action table the ids map
to.

```julia
pol = QTablePolicy("../duckduck/policies/q_learning/policy.npy")
a   = act(pol, raw_state)          # MacroAction
d   = decide(pol, discretize(raw)) # full decision record incl. ties/margin
```
"""
struct QTablePolicy <: AbstractPolicy
    table::Array{Float32,8}
    allowed_actions::Vector{Int}
    action_table::NTuple{7,ActionSpec}
    solver::Symbol
    path::String
end

"""
    QTablePolicy(path; allowed_actions=0:6, action_cfg=ActionConfig(), solver=:q_learning)

Load a `policy.npy` checkpoint. Validates the shape against `Q_SHAPE`, that
every entry is finite, and that the allowed action ids are unique and inside
`0:6` — the same guards the reference adapter applies.
"""
function QTablePolicy(path::AbstractString;
    allowed_actions=collect(0:6), action_cfg::ActionConfig=ActionConfig(),
    solver::Symbol=:q_learning)
    table = read_npy(path)
    size(table) == Q_SHAPE ||
        throw(ArgumentError("Wrong Q-table shape: $(size(table))"))
    all(isfinite, table) ||
        throw(ArgumentError("Q-table contains non-finite values"))
    allowed = collect(Int, allowed_actions)
    (!isempty(allowed) && length(unique(allowed)) == length(allowed)) ||
        throw(ArgumentError("allowed_actions must be non-empty and unique"))
    all(a -> 0 <= a < Q_SHAPE[end], allowed) ||
        throw(ArgumentError("allowed_actions must contain ids in [0, 6]"))
    return QTablePolicy(convert(Array{Float32,8}, table), allowed,
        build_action_table(action_cfg), solver, String(path))
end

"""
    QDecision

One greedy decision plus the evidence the reference records with it: the full
Q-value row, the near-tied action set, and the top-1/top-2 margin.
"""
struct QDecision
    action_id::Int
    action::MacroAction
    spec::ActionSpec
    q_values::NTuple{7,Float64}
    ties::Vector{Int}
    q_margin::Union{Nothing,Float64}
end

"""
    TIE_ATOL

Absolute tolerance of the reference's near-tie window
(`np.isclose(..., rtol=0.0, atol=1e-12)`).
"""
const TIE_ATOL = 1e-12

"""
    decide(policy, index) -> QDecision

Greedy decision for a **0-based** discrete state index (as produced by
[`discretize`](@ref)), following the reference rule verbatim.
"""
function decide(p::QTablePolicy, index::NTuple{7,Int})
    all(0 <= index[k] < STATE_SHAPE[k] for k in 1:7) ||
        throw(ArgumentError("invalid Q-table state index: $index"))
    row = ntuple(a -> Float64(p.table[(index .+ 1)..., a]), 7)
    best = -Inf
    for a in p.allowed_actions
        v = row[a + 1]
        v > best && (best = v)
    end
    ties = Int[]
    for a in p.allowed_actions
        abs(row[a + 1] - best) <= TIE_ATOL && push!(ties, a)
    end
    selected = minimum(ties)
    allowed_values = sort([row[a + 1] for a in p.allowed_actions]; rev=true)
    margin = length(allowed_values) < 2 ? nothing :
        allowed_values[1] - allowed_values[2]
    return QDecision(selected, ALL_MACRO_ACTIONS[selected + 1],
        p.action_table[selected + 1], row, ties, margin)
end

decide(p::QTablePolicy, raw::RawState) = decide(p, discretize(raw))

"""
    act(policy, raw_state[, rng]) -> MacroAction

Deterministic greedy action. The `rng` is accepted for interface uniformity
and deliberately never consumed.
"""
act(p::QTablePolicy, raw::RawState, rng::AbstractRNG=Random.default_rng()) =
    decide(p, raw).action

act(p::QTablePolicy, index::NTuple{7,Int},
    rng::AbstractRNG=Random.default_rng()) = decide(p, index).action

"""
    POMDPs.action(policy, mdp, s) -> MacroAction

Drive the policy directly from a world state: project with the validated
observer, discretize, then decide.
"""
function POMDPs.action(p::QTablePolicy, mdp::AnyMDPLike,
    s::DuckieWorldState)
    raw, _ = get_raw_state(s, mdp.transition.state_cfg)
    return decide(p, discretize(raw)).action
end

"""
    all_state_indices() -> Vector{NTuple{7,Int}}

Every valid 0-based discrete state index, in the reference's C-order
enumeration (last dimension varies fastest) — 9 000 of them.
"""
function all_state_indices()
    out = Vector{NTuple{7,Int}}(undef, prod(STATE_SHAPE))
    k = 0
    for i1 in 0:STATE_SHAPE[1]-1, i2 in 0:STATE_SHAPE[2]-1,
        i3 in 0:STATE_SHAPE[3]-1, i4 in 0:STATE_SHAPE[4]-1,
        i5 in 0:STATE_SHAPE[5]-1, i6 in 0:STATE_SHAPE[6]-1,
        i7 in 0:STATE_SHAPE[7]-1
        k += 1
        out[k] = (i1, i2, i3, i4, i5, i6, i7)
    end
    return out
end

"""
    greedy_action_table(policy) -> Vector{Int}

The selected action id for every discrete state, in [`all_state_indices`](@ref)
order. This is the object FJ7 compares against the reference.
"""
greedy_action_table(p::QTablePolicy) =
    [decide(p, idx).action_id for idx in all_state_indices()]

"""
    tie_statistics(policy) -> NamedTuple

How often the near-tie window actually decides something: the number of
states with more than one tied action, how often the tie changes the outcome
relative to a plain `argmax`, and the margin distribution.
"""
function tie_statistics(p::QTablePolicy)
    n_tied = 0
    n_changed = 0
    n_zero_margin = 0
    min_margin = Inf
    for idx in all_state_indices()
        d = decide(p, idx)
        length(d.ties) > 1 && (n_tied += 1)
        # plain argmax (first maximal allowed action) for comparison
        best = -Inf
        plain = p.allowed_actions[1]
        for a in p.allowed_actions
            v = d.q_values[a + 1]
            if v > best
                best = v
                plain = a
            end
        end
        plain != d.action_id && (n_changed += 1)
        if d.q_margin !== nothing
            min_margin = min(min_margin, d.q_margin)
            d.q_margin <= TIE_ATOL && (n_zero_margin += 1)
        end
    end
    return (states=prod(STATE_SHAPE), tied=n_tied,
        argmax_differs=n_changed, zero_margin=n_zero_margin,
        min_margin=min_margin)
end
