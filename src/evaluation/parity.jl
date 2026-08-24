# FJ5.3: live matched-state one-step parity between the native Julia model and
# the real Python runtime.
#
#                       same latent state x_t
#                       same action a_t
#                   +-----------+-----------+
#                   |                       |
#          reference runtime           native Julia
#      (ref_set_state! + ref_step!)  (simulate_decision)
#                   |                       |
#                   +-------- compare ------+
#
# The comparison is field-level and ULP-quantified, over the full latent
# state, the solver-facing projections, the reward breakdown, the events and
# the termination classification. Unlike the FJ3 fixtures this runs the two
# implementations LIVE, and unlike a seeded-reset comparison it removes every
# initial-condition and accumulation confound: the reference simulator is
# loaded with exactly the state the Julia model starts from.
#
# Accumulated drift over a free-running episode is deliberately NOT measured
# here — that is FJ6.

"""
    FieldDiff

One compared quantity: its `name`, the two values, the ULP distance (for
finite Float64 pairs) and the absolute difference. `ulps == 0` means
bit-identical.
"""
struct FieldDiff
    name::String
    julia::Float64
    reference::Float64
    ulps::Int
    absdiff::Float64
    bitdiff::Bool     # differing bit patterns even when numerically equal
end

"""
    _ulps(a, b) -> Int

ULP distance under the IEEE-754 total order, saturating instead of
overflowing. Numerically equal values are 0 ULP apart — including
`0.0` vs `-0.0`, whose raw `reinterpret` difference is `typemin(Int64)` and
overflowed an earlier `abs(...)`-based implementation into a negative
"distance" (a measurement-tool bug, caught by FJ5.3 refusing to accept
otherwise clean steps).
"""
function _ulps(a::Float64, b::Float64)
    (isnan(a) && isnan(b)) && return 0
    a == b && return 0                       # covers +0.0 vs -0.0
    (isnan(a) || isnan(b)) && return typemax(Int)
    (!isfinite(a) || !isfinite(b)) && return typemax(Int)
    # map to a monotonic unsigned ordering, then take the unsigned distance
    ord(x) = (k = reinterpret(UInt64, x);
        (k & 0x8000000000000000) != 0 ? 0xffffffffffffffff - k + 1 :
        k + 0x8000000000000000)
    ua, ub = ord(a), ord(b)
    d = ua >= ub ? ua - ub : ub - ua
    return d > UInt64(typemax(Int)) ? typemax(Int) : Int(d)
end

function _diff(name::AbstractString, a, b)
    x = Float64(a)
    y = Float64(b)
    # equal non-finite values (Inf == Inf, NaN vs NaN) are a zero difference,
    # not the NaN that `Inf - Inf` would produce
    absdiff = (x == y || (isnan(x) && isnan(y))) ? 0.0 : abs(x - y)
    bitdiff = reinterpret(UInt64, x) != reinterpret(UInt64, y) &&
        !(isnan(x) && isnan(y))
    return FieldDiff(String(name), x, y, _ulps(x, y), absdiff, bitdiff)
end

"""
    SIGNED_ZERO_FIELDS

Entries that can carry a different ZERO SIGN between the runtimes with no
numerical difference whatsoever (`0.0 == -0.0`).

Measured: the se(2) body-velocity diagonal `ego.v0[1,1]` / `ego.v0[2,2]` on
turning actions (the reference builds the matrix through NumPy products that
can yield `-0.0`). These entries are **never read**: the only consumers of
`v0` are `linear_angular_from_se2` (which reads `[1,3]`, `[2,3]`, `[2,1]`)
and `SE2_from_se2` (which reads `[2,1]` and `[1:2,3]`). The diagonal is inert,
so this cannot affect any transition.
"""
const SIGNED_ZERO_FIELDS = Set(["ego.v0[1,1]", "ego.v0[2,2]"])

_push_diff!(v, name, a, b) = push!(v, _diff(name, a, b))

"""
    StepParityReport

Result of one matched-state comparison: the field-level `diffs`, the worst
ULP/absolute distances, and the discrete agreements (`reason`, `terminated`,
`truncated`, `events`, `tile`, `duck` class, counters). `exact` means every
compared quantity was bit-identical and every discrete field agreed.
"""
struct StepParityReport
    action::Any
    diffs::Vector{FieldDiff}
    max_ulps::Int
    max_absdiff::Float64
    discrete_mismatches::Vector{String}
    reason_julia::TerminationReason
    reason_reference::String
end

exact(r::StepParityReport) =
    r.max_ulps == 0 && isempty(r.discrete_mismatches)

"""
    LIBM_DERIVED_FIELDS

The only quantities allowed to differ at all, and the reason each may.

**Root cause (measured, not assumed).** The dynamical state itself is
bit-identical: after a matched-state step the DB18 pose/velocity matrices
`q0`/`v0` agree bit-for-bit. The two runtimes diverge only where they read
that state back through their own libm. Recomputing `atan2(q0[2,1],
q0[1,1])` in Julia **on the reference's own q0** reproduces the Julia angle
exactly while the reference's stored angle differs by 1 ULP, so the source is
`atan2` (OpenLibm vs glibc) — the deviation class already recorded in
FJ2/FJ3, now confirmed live.

**Propagation chain (measured worst case, Julia 1.11.3 vs `ddm-ref`):**

```
ego.angle          1 ULP   atan2 pose readback              <- root
  lane_fallback[2] 2 ULP   acos(dot(dir(angle), tangent)), ill-conditioned
    raw.phi        2 ULP   the clamped lane angle
      reward.heading 4 ULP  -alpha*phi^2 doubles the relative error
      reward.total   4 ULP  inherits the heading term
  reward.progress  1 ULP   alpha*v*cos(phi)
```

Worst absolute difference over a 40-decision matched-state sweep:
**2.22e-16** — twelve orders of magnitude below the quantities involved
(rewards ~1e-1 rad/units). `raw.d`, the ego pose, speed, the duckie state,
the delay window, the events and the termination classification are all
bit-identical.

**Second, independent libm source (found by FJ6).** The quadratic reward
terms are written `state.phi ** 2` / `state.d ** 2` in the reference, and
CPython's `float.__pow__` is a libm `pow()` call — not a multiplication.
Measured at `phi = 0.16103364894924665`:

```
Python  phi ** 2   = 0.02593183609390921    (glibc pow)
Python  phi * phi  = 0.025931836093909207
Julia   phi^2      = 0.025931836093909207   (x*x)
Julia   phi^2.0    = 0.025931836093909207   (OpenLibm pow agrees with x*x)
```

so glibc's `pow` is 1 ULP off the correctly-rounded square here. Julia cannot
reproduce that without emulating glibc's `pow`, which would make the model
platform-dependent — a worse outcome than a 1-ULP reward difference that
never touches the state. `reward.lateral` is included for the same reason
(`d ** 2`). Unlike the `atan2` source, this one is NOT affected by in-process
interposition: embedded Python still returned glibc's value.

The exact set is toolchain-dependent: on Julia 1.10.11 only `ego.angle`
deviated (1 ULP); 1.11.3 propagates it a little further. The invariant that
matters — and that the tests assert — is STRUCTURAL: nothing outside this
libm-derived chain may deviate at all, and no discrete field may disagree.
"""
const LIBM_DERIVED_FIELDS = Set(["ego.angle", "lane_fallback[2]", "raw.phi",
    "cont.phi", "cont.kappa", "reward.progress", "reward.heading",
    "reward.lateral", "reward.total"])

# kept as the historical name used by the FJ5.3 attribution note
const LIBM_1ULP_FIELDS = LIBM_DERIVED_FIELDS

"""
    LIBM_MAX_ULPS / LIBM_MAX_ABSDIFF

Numeric bounds for [`LIBM_DERIVED_FIELDS`], derived from measurement rather
than chosen (master prompt §33): the observed worst case is 4 ULP and
2.22e-16 absolute; the bounds keep 2x ULP headroom for other toolchains while
staying far below any physically meaningful scale.
"""
const LIBM_MAX_ULPS = 8
const LIBM_MAX_ABSDIFF = 1e-14

"""
    parity_accepted(report; max_libm_ulps=LIBM_MAX_ULPS,
                    max_libm_absdiff=LIBM_MAX_ABSDIFF) -> Bool

The FJ5 acceptance criterion: every discrete field agrees and every compared
quantity is bit-identical, EXCEPT the attributed [`LIBM_DERIVED_FIELDS`],
which may differ within the measured bounds.
"""
function parity_accepted(r::StepParityReport;
    max_libm_ulps::Integer=LIBM_MAX_ULPS,
    max_libm_absdiff::Real=LIBM_MAX_ABSDIFF)
    # written as a plain loop on purpose: the `all(...) do d ... end` form,
    # whose closure captured the keyword argument, reproducibly crashed the
    # Julia 1.10.11 Windows compiler (EXCEPTION_ACCESS_VIOLATION in
    # gc_mark_stack during the inlining pass)
    isempty(r.discrete_mismatches) || return false
    for d in r.diffs
        d.ulps == 0 && continue
        (d.name in LIBM_1ULP_FIELDS && d.ulps <= max_libm_ulps) || return false
    end
    return true
end

"""
    nonzero_fields(reports) -> Vector{String}

Every field name that was ever non-bit-identical across a sweep. The FJ5
evidence is that this set equals [`LIBM_1ULP_FIELDS`] or is empty.
"""
function nonzero_fields(reports::AbstractVector{StepParityReport})
    names = String[]
    for r in reports, d in r.diffs
        d.ulps > 0 && push!(names, d.name)
    end
    return unique(names)
end

"""
    bitwise_only_fields(reports) -> Vector{String}

Fields that are numerically equal but differ in bit pattern — in practice the
signed-zero entries described in [`SIGNED_ZERO_FIELDS`].
"""
function bitwise_only_fields(reports::AbstractVector{StepParityReport})
    names = String[]
    for r in reports, d in r.diffs
        (d.bitdiff && d.ulps == 0) && push!(names, d.name)
    end
    return unique(names)
end

worst(r::StepParityReport; n::Integer=5) =
    sort(r.diffs; by=d -> -d.ulps)[1:min(n, length(r.diffs))]

function Base.show(io::IO, r::StepParityReport)
    print(io, "StepParityReport(action=", r.action, ", fields=",
        length(r.diffs), ", max_ulps=", r.max_ulps, ", max_absdiff=",
        r.max_absdiff, ", discrete_mismatches=",
        length(r.discrete_mismatches), ", exact=", exact(r), ")")
end

"""
    compare_worlds(julia_world, reference_world) -> Vector{FieldDiff}

Field-level comparison of the full latent state: ego pose/speed/DB18
matrices/wheel axes/delay window, every duckie's object state, and the
memories. Discrete fields are checked by [`compare_step`](@ref).
"""
function compare_worlds(a::DuckieWorldState, b::DuckieWorldState)
    d = FieldDiff[]
    for k in 1:3
        _push_diff!(d, "ego.pos[$k]", a.ego.pos[k], b.ego.pos[k])
    end
    _push_diff!(d, "ego.angle", a.ego.angle, b.ego.angle)
    _push_diff!(d, "ego.speed", a.ego.speed, b.ego.speed)
    _push_diff!(d, "ego.timestamp", a.ego.timestamp, b.ego.timestamp)
    _push_diff!(d, "ego.axis_left_rad", a.ego.axis_left_rad, b.ego.axis_left_rad)
    _push_diff!(d, "ego.axis_right_rad", a.ego.axis_right_rad, b.ego.axis_right_rad)
    for i in 1:3, j in 1:3
        _push_diff!(d, "ego.q0[$i,$j]", a.ego.q0[i, j], b.ego.q0[i, j])
        _push_diff!(d, "ego.v0[$i,$j]", a.ego.v0[i, j], b.ego.v0[i, j])
    end
    n = min(length(a.ego.command_history), length(b.ego.command_history))
    for k in 1:n, c in 1:3
        _push_diff!(d, "ego.cmd[$k][$c]", a.ego.command_history[k][c],
            b.ego.command_history[k][c])
    end
    for (i, (da, db)) in enumerate(zip(a.ducks, b.ducks))
        for k in 1:3
            _push_diff!(d, "duck$i.pos[$k]", da.pos[k], db.pos[k])
            _push_diff!(d, "duck$i.center[$k]", da.center[k], db.center[k])
            _push_diff!(d, "duck$i.start[$k]", da.start[k], db.start[k])
            _push_diff!(d, "duck$i.heading[$k]", da.heading[k], db.heading[k])
        end
        _push_diff!(d, "duck$i.angle", da.angle, db.angle)
        _push_diff!(d, "duck$i.vel", da.vel, db.vel)
        _push_diff!(d, "duck$i.time", da.time, db.time)
        _push_diff!(d, "duck$i.wait", da.pedestrian_wait_time,
            db.pedestrian_wait_time)
        for k in 1:4, c in 1:2
            _push_diff!(d, "duck$i.corner[$k][$c]", da.obj_corners[k][c],
                db.obj_corners[k][c])
        end
        for r in 1:2, c in 1:2
            _push_diff!(d, "duck$i.norm[$r,$c]", da.obj_norm[r, c],
                db.obj_norm[r, c])
        end
    end
    if a.stop_memory.last_d_stop !== nothing &&
        b.stop_memory.last_d_stop !== nothing
        _push_diff!(d, "memory.last_d_stop", a.stop_memory.last_d_stop,
            b.stop_memory.last_d_stop)
    end
    for k in 1:2
        _push_diff!(d, "lane_fallback[$k]", a.lane_fallback[k],
            b.lane_fallback[k])
    end
    return d
end

const _REF_REASONS = Dict(
    "duck_collision" => DUCK_COLLISION, "other_collision" => OTHER_COLLISION,
    "timeout" => TIMEOUT, "offroad" => OFFROAD, "goal" => GOAL,
    "in_progress" => IN_PROGRESS)

_pnum(x) = x === nothing ? nothing :
    x isa AbstractDict ? (x["nonfinite"] == "nan" ? NaN :
        x["nonfinite"] == "inf" ? Inf :
        x["nonfinite"] == "-inf" ? -Inf : -0.0) : Float64(x)

"""
    compare_step(backend, model, world, action; rng) -> StepParityReport

Inject `world` into the live reference simulator, step BOTH implementations
with `action` from that same state, and compare everything.

Note on stochasticity: the reference pedestrian trigger draws from its own
`RandomState`. Under the shipped configs `p_cross = 1.0`, so the trigger
outcome is draw-independent (`u < 1.0` for every `u`) and the comparison is
unambiguous. FJ3.8 separately pins the draw values, positions and call
semantics, so a `p_cross < 1` scenario can be compared by seeding
[`NumpyMT19937`](@ref) identically.
"""
function compare_step(backend::AbstractReferenceBackend,
    model::DuckieTransitionModel,
    world::DuckieWorldState, action; rng::AbstractRNG=MersenneTwister(0))
    ref_set_state!(backend, world)
    ref_world, dump = ref_step!(backend, action)
    res = dump.result

    jl = simulate_decision(model, world, action, rng)

    diffs = compare_worlds(jl.sp, ref_world)

    # wheel commands, raw-state projection, reward breakdown
    for k in 1:2
        _push_diff!(diffs, "wheels[$k]", Float64(jl.wheel_commands[k]),
            _pnum(res.wheels[k]))
    end
    rw = res.raw_state
    _push_diff!(diffs, "raw.d", jl.raw_state.d, _pnum(rw.d))
    _push_diff!(diffs, "raw.phi", jl.raw_state.phi, _pnum(rw.phi))
    _push_diff!(diffs, "raw.v", jl.raw_state.v, _pnum(rw.v))
    if jl.raw_state.d_stop !== nothing && rw.d_stop !== nothing
        _push_diff!(diffs, "raw.d_stop", jl.raw_state.d_stop, _pnum(rw.d_stop))
    end
    rt = res.reward_terms
    for f in (:progress, :lateral, :heading, :time, :pedestrian, :stagnation,
        :stop_approach, :steering, :events, :total)
        _push_diff!(diffs, "reward.$f", getfield(jl.reward, f),
            _pnum(rt[f]))
    end
    if haskey(res, :continuous_state)
        cs = res.continuous_state
        for f in (:d, :phi, :v, :kappa, :d_stop, :duck_longitudinal,
            :duck_lateral, :duck_v_longitudinal_relative,
            :duck_v_lateral_relative, :stop_hold_progress)
            jv = getfield(jl.continuous_state, f)
            rv = cs[f]
            (jv === nothing || rv === nothing) && continue
            _push_diff!(diffs, "cont.$f", jv, _pnum(rv))
        end
    end

    # discrete agreements
    mism = String[]
    ref_reason = String(res.reason)
    jl.reason == _REF_REASONS[ref_reason] || push!(mism, "reason")
    jl.terminated == Bool(res.terminated) || push!(mism, "terminated")
    jl.truncated == Bool(res.truncated) || push!(mism, "truncated")
    ev = res.events
    for (name, got) in (("collision_duck", jl.events.collision_duck),
        ("other_collision", jl.events.other_collision),
        ("offroad", jl.events.offroad), ("timeout", jl.events.timeout),
        ("stop_violation", jl.events.stop_violation),
        ("full_stop", jl.events.full_stop),
        ("passed_stop", jl.events.passed_stop), ("goal", jl.events.goal))
        got == Bool(ev[Symbol(name)]) || push!(mism, "events.$name")
    end
    Int(jl.raw_state.tile) == Int(rw.tile) || push!(mism, "raw.tile")
    Int(jl.raw_state.duck) == Int(rw.duck) || push!(mism, "raw.duck")
    jl.raw_state.sigma_stop == Bool(rw.sigma_stop) || push!(mism, "raw.sigma")
    (jl.raw_state.d_stop === nothing) == (rw.d_stop === nothing) ||
        push!(mism, "raw.d_stop_present")
    jl.sp.ego.step_count == ref_world.ego.step_count ||
        push!(mism, "ego.step_count")
    jl.sp.crossings_started == ref_world.crossings_started ||
        push!(mism, "crossings_started")
    jl.sp.crossing_armed == ref_world.crossing_armed ||
        push!(mism, "crossing_armed")
    jl.sp.stop_memory.sigma_stop == ref_world.stop_memory.sigma_stop ||
        push!(mism, "memory.sigma_stop")
    jl.sp.stop_memory.hold_steps == ref_world.stop_memory.hold_steps ||
        push!(mism, "memory.hold_steps")
    jl.sp.stop_memory.last_stop_id == ref_world.stop_memory.last_stop_id ||
        push!(mism, "memory.last_stop_id")
    length(jl.sp.ego.command_history) ==
        length(ref_world.ego.command_history) ||
        push!(mism, "ego.command_window_length")
    for (i, (da, db)) in enumerate(zip(jl.sp.ducks, ref_world.ducks))
        da.pedestrian_active == db.pedestrian_active ||
            push!(mism, "duck$i.active")
        da.visible == db.visible || push!(mism, "duck$i.visible")
    end

    return StepParityReport(action, diffs,
        isempty(diffs) ? 0 : maximum(d -> d.ulps, diffs),
        isempty(diffs) ? 0.0 : maximum(d -> d.absdiff, diffs),
        mism, jl.reason, ref_reason)
end

"""
    matched_state_sweep(backend, model, world, actions; rng, advance)
        -> Vector{StepParityReport}

Run [`compare_step`](@ref) for each action from the SAME `world` (branching
comparison). With `advance = true` the world is instead advanced by each
action in turn, so the sweep walks a trajectory while every individual step
stays matched-state.
"""
function matched_state_sweep(backend::AbstractReferenceBackend,
    model::DuckieTransitionModel, world::DuckieWorldState, actions;
    rng::AbstractRNG=MersenneTwister(0), advance::Bool=false)
    reports = StepParityReport[]
    w = world
    for a in actions
        r = compare_step(backend, model, w, a; rng=rng)
        push!(reports, r)
        advance && (w = simulate_decision(model, w, a, rng).sp)
    end
    return reports
end

"""
    parity_summary(reports) -> NamedTuple

Aggregate a sweep: number of steps, how many were bit-exact, the worst ULP and
absolute distances with the field that produced them, and every discrete
mismatch seen.
"""
function parity_summary(reports::AbstractVector{StepParityReport})
    isempty(reports) && return (steps=0, exact=0, max_ulps=0,
        max_absdiff=0.0, worst_field="", discrete_mismatches=String[])
    worst_d = reduce((x, y) -> x.ulps >= y.ulps ? x : y,
        reduce(vcat, [r.diffs for r in reports]))
    return (steps=length(reports),
        exact=count(exact, reports),
        accepted=count(parity_accepted, reports),
        bitwise_only_fields=bitwise_only_fields(reports),
        max_ulps=maximum(r -> r.max_ulps, reports),
        max_absdiff=maximum(r -> r.max_absdiff, reports),
        worst_field=worst_d.name,
        nonzero_fields=nonzero_fields(reports),
        discrete_mismatches=unique(reduce(vcat,
            [r.discrete_mismatches for r in reports])))
end
