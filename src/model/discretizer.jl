"""
    D_BINS, TRACKING_ERROR_BINS, V_BINS

Bin edges of the tabular discretizer (`src/discretizer.py`), in Python order.
`np.digitize(x, bins)` (right-open bins, `bins[i-1] ≤ x < bins[i]`) equals
`searchsortedlast(bins, x)`, i.e. the count of edges `≤ x`.
"""
const D_BINS = [-0.15, -0.05, 0.05, 0.15]
const TRACKING_ERROR_BINS = [-0.50, -0.10, 0.10, 0.50]
const V_BINS = [0.04, 0.16]

"""
    STATE_SHAPE, Q_SHAPE

Number of bins per dimension of the discretized state (7 dims), and the
Q-table shape `STATE_SHAPE + (7,)` for the macro-action dimension.
"""
const STATE_SHAPE = (5, 5, 3, 3, 4, 2, 5)
const Q_SHAPE = (STATE_SHAPE..., 7)

"""
    digitize(x, bins) -> Int

NumPy `np.digitize` semantics (default `right=false`): the number of bin edges
`≤ x`, i.e. `searchsortedlast(bins, x)`. Valid on monotone `bins`.
"""
digitize(x::Real, bins::AbstractVector{<:Real}) = searchsortedlast(bins, x)

"""
    discretize(state::RawState) -> NTuple{7,Int}

Exact 7-D index of the Q-table for a [`RawState`](@ref)
(`src/discretizer.py::discretize`):

    s_bar = (bin(d), bin(phi + d), bin(v), tile, bin(d_stop), sigma_stop, duck)

`e = phi + d` is the tracking error; binning it (not `phi` alone) removes the
state aliasing between poses with equal heading but different offsets.
`d_stop` classes: `none → 0`, `> 1.0 → 1`, `≥ 0.3 → 2`, else `3`.

The range guard mirrors Python (`IndexError(index)`); with the four bin edges
of `D_BINS`/`TRACKING_ERROR_BINS`/`V_BINS` and valid enums it is unreachable
(max `digitize` = `length(bins)` < shape), exactly as in the reference.
"""
function discretize(state::RawState)
    stop = state.d_stop === nothing ? 0 :
        state.d_stop > 1.0 ? 1 :
        state.d_stop >= 0.3 ? 2 : 3
    tracking_error = state.phi + state.d
    index = (
        digitize(state.d, D_BINS),
        digitize(tracking_error, TRACKING_ERROR_BINS),
        digitize(state.v, V_BINS),
        Int(state.tile),
        stop,
        Int(state.sigma_stop),
        Int(state.duck),
    )
    if any(i < 0 || i >= n for (i, n) in zip(index, STATE_SHAPE))
        throw(IndexError(index))
    end
    return index
end