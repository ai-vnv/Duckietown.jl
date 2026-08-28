# FJ6: free-running rollout harness and drift analysis.
#
# FJ5 compared ONE decision at a time with the reference simulator reloaded
# from the Julia state. FJ6 removes that correction entirely:
#
#     x0 (matched)
#       ├─ Python reference runs freely:  x1_py -> x2_py -> ... -> xT_py
#       └─ native Julia runs freely:      x1_jl -> x2_jl -> ... -> xT_jl
#
# so small readback differences are allowed to feed back into the dynamics and
# accumulate. The point is to classify what accumulates and when.
#
# Two kinds of divergence are tracked SEPARATELY, because they mean different
# things (FJ5-R established why):
#
#   Type 1 — readback drift. `q0`/`v0` (the actual dynamical state) are still
#            bit-identical, but the pose read back through `atan2` differs,
#            and with it phi and the reward terms derived from phi. The
#            physical state has NOT diverged.
#   Type 2 — dynamical drift. `q0`/`v0` themselves differ. Measured cause in
#            the one case where it happened (see docs/validation/FJ6_STATUS.md): NOT
#            feedback from the readback drift, but a fresh 1-ULP disagreement
#            in `cos` INSIDE the SE(2) exponential — glibc returns
#            0.9999531746252679 for w = 0.009677369495122897 where OpenLibm
#            returns 0.999953174625268. Every other operation of that step
#            (sin, the exponential, the matrix product, the wheel commands,
#            v0) was bit-identical.
#
# Milestone decisions (first index where each first happens):
#   D_readback  pose/observer values differ while q0/v0 still agree
#   D_dynamic   q0/v0 differ                      (Type 2 begins)
#   D_discrete  any categorical/event field differs
#   D_terminal  termination timing or reason differs
#
# Backend roles (FJ5-R finding — do not swap them):
#   ProcessReferenceBackend  isolated Python + glibc  -> NUMERICAL reference
#   PythonCallRefBackend     in-process, Julia's libm -> SEMANTIC oracle

"""
    RolloutRecord

One decision of a free-running rollout. Everything needed to compare two runs
without re-deriving anything: the dynamical state (`q0`, `v0`, delay window
length), the pose readback, the tabular projection, the stop and duck state,
the action actually applied, the full reward breakdown and the episode flags.
"""
struct RolloutRecord
    decision::Int
    # dynamical state (Type 2 evidence)
    q0::Matrix{Float64}
    v0::Matrix{Float64}
    n_commands::Int
    step_count::Int
    # pose readback (Type 1 evidence)
    pos::NTuple{3,Float64}
    angle::Float64
    speed::Float64
    # observers
    d::Float64
    phi::Float64
    v::Float64
    tile::Int
    d_stop::Union{Nothing,Float64}
    sigma_stop::Bool
    duck_threat::Int
    # stop / duck state
    hold_steps::Int
    stop_id::Union{Nothing,Int}
    duck_center::NTuple{3,Float64}
    duck_vel::Float64
    duck_active::Bool
    crossings_started::Int
    # action and outcome
    action::String
    wheels::NTuple{2,Float64}
    reward::RewardBreakdown
    cumulative_return::Float64
    events::EventFlags
    terminated::Bool
    truncated::Bool
    reason::String
end

"""
    rollout_native(model, x0, actions; rng, discount=1.0) -> Vector{RolloutRecord}

Free-running native rollout: `simulate_decision` applied repeatedly, feeding
its own successor state back in. Stops early on a genuine terminal.
"""
function rollout_native(model::DuckieTransitionModel, x0::DuckieWorldState,
    actions; rng::AbstractRNG=MersenneTwister(0), discount::Real=1.0)
    records = RolloutRecord[]
    w = x0
    total = 0.0
    disc = 1.0
    for (k, a) in enumerate(actions)
        r = simulate_decision(model, w, a, rng)
        total += disc * r.reward.total
        disc *= discount
        push!(records, _native_record(k, a, r, total))
        w = r.sp
        # episode boundary: the reference wrapper reports done = terminated ||
        # truncated, so both runs must stop at the same place
        (r.terminated || r.truncated) && break
    end
    return records
end

function _native_record(k::Int, action, r::TransitionResult, total::Float64)
    sp = r.sp
    duck = isempty(sp.ducks) ? nothing : sp.ducks[1]
    return RolloutRecord(k, copy(sp.ego.q0), copy(sp.ego.v0),
        length(sp.ego.command_history), sp.ego.step_count,
        sp.ego.pos, sp.ego.angle, sp.ego.speed,
        r.raw_state.d, r.raw_state.phi, r.raw_state.v, Int(r.raw_state.tile),
        r.raw_state.d_stop, r.raw_state.sigma_stop, Int(r.raw_state.duck),
        sp.stop_memory.hold_steps, sp.stop_memory.last_stop_id,
        duck === nothing ? (0.0, 0.0, 0.0) : duck.center,
        duck === nothing ? 0.0 : duck.vel,
        duck === nothing ? false : duck.pedestrian_active,
        isempty(sp.crossings_started) ? 0 : sp.crossings_started[1],
        _action_label(action),
        (Float64(r.wheel_commands[1]), Float64(r.wheel_commands[2])),
        r.reward, total, r.events, r.terminated, r.truncated,
        lowercase(string(r.reason)))
end

_action_label(a::MacroAction) = string(a)
_action_label(a::DuckieAction) = "v=$(a.v),w=$(a.omega)"
_action_label(a) = string(a)

"""
    rollout_reference(backend, x0, actions; discount=1.0) -> Vector{RolloutRecord}

Free-running REFERENCE rollout. `x0` is injected once, at the start; after
that the reference simulator advances on its own — no per-step re-injection,
which is exactly what distinguishes FJ6 from FJ5.
"""
function rollout_reference(backend::AbstractReferenceBackend,
    x0::DuckieWorldState, actions; discount::Real=1.0)
    ref_set_state!(backend, x0)
    records = RolloutRecord[]
    total = 0.0
    disc = 1.0
    for (k, a) in enumerate(actions)
        sp, dump = ref_step!(backend, a)
        res = dump.result
        rw = res.reward_terms
        reward = RewardBreakdown(_pnum(rw.progress), _pnum(rw.lateral),
            _pnum(rw.heading), _pnum(rw.time), _pnum(rw.pedestrian),
            _pnum(rw.stagnation), _pnum(rw.stop_approach), _pnum(rw.steering),
            _pnum(rw.events), _pnum(rw.total))
        total += disc * reward.total
        disc *= discount
        ev = res.events
        events = EventFlags(collision_duck=Bool(ev.collision_duck),
            other_collision=Bool(ev.other_collision), offroad=Bool(ev.offroad),
            timeout=Bool(ev.timeout), stop_violation=Bool(ev.stop_violation),
            full_stop=Bool(ev.full_stop), passed_stop=Bool(ev.passed_stop),
            goal=Bool(ev.goal))
        raw = res.raw_state
        duck = isempty(sp.ducks) ? nothing : sp.ducks[1]
        push!(records, RolloutRecord(k, copy(sp.ego.q0), copy(sp.ego.v0),
            length(sp.ego.command_history), sp.ego.step_count,
            sp.ego.pos, sp.ego.angle, sp.ego.speed,
            _pnum(raw.d), _pnum(raw.phi), _pnum(raw.v), Int(raw.tile),
            raw.d_stop === nothing ? nothing : _pnum(raw.d_stop),
            Bool(raw.sigma_stop), Int(raw.duck),
            sp.stop_memory.hold_steps, sp.stop_memory.last_stop_id,
            duck === nothing ? (0.0, 0.0, 0.0) : duck.center,
            duck === nothing ? 0.0 : duck.vel,
            duck === nothing ? false : duck.pedestrian_active,
            isempty(sp.crossings_started) ? 0 : sp.crossings_started[1],
            _action_label(a),
            (_pnum(res.wheels[1]), _pnum(res.wheels[2])),
            reward, total, events, Bool(res.terminated), Bool(res.truncated),
            String(res.reason)))
        (Bool(res.terminated) || Bool(res.truncated)) && break
    end
    return records
end

# ---------------------------------------------------------------------------
# Drift analysis
# ---------------------------------------------------------------------------

"""
    DriftReport

Per-decision drift between two free-running rollouts, plus the milestone
decisions that matter more than any aggregate error.
"""
struct DriftReport
    n::Int                              # decisions compared
    dx::Vector{Float64}                 # |Δ ego x|
    dz::Vector{Float64}                 # |Δ ego z|
    dangle::Vector{Float64}
    dspeed::Vector{Float64}
    dd::Vector{Float64}                 # |Δ lane d|
    dphi::Vector{Float64}
    dreward::Vector{Float64}
    dreturn::Vector{Float64}
    dq0::Vector{Float64}                # max |Δ q0| entrywise (Type 2)
    dv0::Vector{Float64}
    d_readback::Union{Nothing,Int}
    d_dynamic::Union{Nothing,Int}
    d_discrete::Union{Nothing,Int}
    d_terminal::Union{Nothing,Int}
    discrete_first_field::String
    len_a::Int
    len_b::Int
    reason_a::String
    reason_b::String
end

_maxabsdiff(A::AbstractMatrix, B::AbstractMatrix) =
    maximum(abs.(A .- B))

"""
    compare_rollouts(a, b) -> DriftReport

Compare two free-running rollouts decision by decision. `a` is the reference
run, `b` the run under test (native Julia, or the other transport).
"""
function compare_rollouts(a::Vector{RolloutRecord}, b::Vector{RolloutRecord})
    n = min(length(a), length(b))
    dx = zeros(n); dz = zeros(n); dang = zeros(n); dsp = zeros(n)
    dd = zeros(n); dphi = zeros(n); dr = zeros(n); dg = zeros(n)
    dq = zeros(n); dv = zeros(n)
    d_readback = nothing
    d_dynamic = nothing
    d_discrete = nothing
    d_terminal = nothing
    discrete_field = ""
    for k in 1:n
        ra, rb = a[k], b[k]
        dx[k] = abs(ra.pos[1] - rb.pos[1])
        dz[k] = abs(ra.pos[3] - rb.pos[3])
        dang[k] = abs(ra.angle - rb.angle)
        dsp[k] = abs(ra.speed - rb.speed)
        dd[k] = abs(ra.d - rb.d)
        dphi[k] = abs(ra.phi - rb.phi)
        dr[k] = abs(ra.reward.total - rb.reward.total)
        dg[k] = abs(ra.cumulative_return - rb.cumulative_return)
        dq[k] = _maxabsdiff(ra.q0, rb.q0)
        dv[k] = _maxabsdiff(ra.v0, rb.v0)

        if d_dynamic === nothing && (dq[k] > 0.0 || dv[k] > 0.0)
            d_dynamic = k
        end
        if d_readback === nothing && (dang[k] > 0.0 || dphi[k] > 0.0 ||
            dd[k] > 0.0 || dx[k] > 0.0 || dz[k] > 0.0)
            d_readback = k
        end
        if d_discrete === nothing
            f = _first_discrete_difference(ra, rb)
            if f !== nothing
                d_discrete = k
                discrete_field = f
            end
        end
        if d_terminal === nothing &&
            (ra.terminated != rb.terminated || ra.truncated != rb.truncated ||
             ra.reason != rb.reason)
            d_terminal = k
        end
    end
    if d_terminal === nothing && length(a) != length(b)
        d_terminal = n + 1     # one run ended and the other did not
    end
    return DriftReport(n, dx, dz, dang, dsp, dd, dphi, dr, dg, dq, dv,
        d_readback, d_dynamic, d_discrete, d_terminal, discrete_field,
        length(a), length(b),
        isempty(a) ? "" : a[end].reason, isempty(b) ? "" : b[end].reason)
end

function _first_discrete_difference(ra::RolloutRecord, rb::RolloutRecord)
    ra.tile == rb.tile || return "tile"
    ra.duck_threat == rb.duck_threat || return "duck_threat"
    ra.sigma_stop == rb.sigma_stop || return "sigma_stop"
    ra.hold_steps == rb.hold_steps || return "hold_steps"
    ra.stop_id == rb.stop_id || return "stop_id"
    (ra.d_stop === nothing) == (rb.d_stop === nothing) || return "d_stop_present"
    ra.duck_active == rb.duck_active || return "duck_active"
    ra.crossings_started == rb.crossings_started || return "crossings_started"
    ra.step_count == rb.step_count || return "step_count"
    ra.n_commands == rb.n_commands || return "n_commands"
    for f in (:collision_duck, :other_collision, :offroad, :timeout,
        :stop_violation, :full_stop, :passed_stop, :goal)
        getfield(ra.events, f) == getfield(rb.events, f) || return "events.$f"
    end
    ra.wheels == rb.wheels || return "wheels"
    return nothing
end

"""
    drift_summary(report) -> NamedTuple

Compact, JSON-ready summary: milestone decisions, final and maximum drifts,
and the Type-1/Type-2 classification.
"""
function drift_summary(r::DriftReport)
    last_or(v) = isempty(v) ? 0.0 : v[end]
    max_or(v) = isempty(v) ? 0.0 : maximum(v)
    kind = r.d_dynamic === nothing ?
        (r.d_readback === nothing ? "IDENTICAL" : "TYPE1_READBACK_ONLY") :
        "TYPE2_DYNAMICAL"
    return (decisions=r.n, len_reference=r.len_a, len_test=r.len_b,
        reason_reference=r.reason_a, reason_test=r.reason_b,
        divergence_kind=kind,
        D_readback=r.d_readback, D_dynamic=r.d_dynamic,
        D_discrete=r.d_discrete, D_terminal=r.d_terminal,
        discrete_first_field=r.discrete_first_field,
        max_dq0=max_or(r.dq0), max_dv0=max_or(r.dv0),
        max_dx=max_or(r.dx), max_dz=max_or(r.dz),
        max_dangle=max_or(r.dangle), max_dspeed=max_or(r.dspeed),
        max_dd=max_or(r.dd), max_dphi=max_or(r.dphi),
        max_dreward=max_or(r.dreward),
        final_dx=last_or(r.dx), final_dz=last_or(r.dz),
        final_dangle=last_or(r.dangle),
        final_dreturn=last_or(r.dreturn), max_dreturn=max_or(r.dreturn))
end

"""
    event_timing(records) -> Dict{String,Union{Nothing,Int}}

Decision index at which each tracked event/condition first occurs in a
rollout (`nothing` if it never does).
"""
function event_timing(records::Vector{RolloutRecord})
    out = Dict{String,Union{Nothing,Int}}()
    firsts = ("duck_activation" => r -> r.duck_active,
        "duck_crossing_started" => r -> r.crossings_started > 0,
        "d_stop_visible" => r -> r.d_stop !== nothing,
        "stop_zone_entry" => r -> r.d_stop !== nothing && r.d_stop <= 0.45,
        "hold_steps_started" => r -> r.hold_steps > 0,
        "sigma_stop" => r -> r.sigma_stop,
        "full_stop" => r -> r.events.full_stop,
        "passed_stop" => r -> r.events.passed_stop,
        "stop_violation" => r -> r.events.stop_violation,
        "collision_duck" => r -> r.events.collision_duck,
        "other_collision" => r -> r.events.other_collision,
        "offroad" => r -> r.events.offroad,
        "timeout" => r -> r.events.timeout,
        "goal" => r -> r.events.goal)
    for (name, pred) in firsts
        idx = findfirst(pred, records)
        out[name] = idx
    end
    return out
end

"""
    event_timing_diff(a, b) -> Dict{String,Any}

Per-event first-occurrence indices in both runs and their difference.
"""
function event_timing_diff(a::Vector{RolloutRecord}, b::Vector{RolloutRecord})
    ta, tb = event_timing(a), event_timing(b)
    out = Dict{String,Any}()
    for k in sort(collect(keys(ta)))
        ia, ib = ta[k], tb[k]
        out[k] = Dict("reference" => ia, "test" => ib,
            "difference" => (ia === nothing || ib === nothing) ? nothing : ib - ia)
    end
    return out
end

"""
    rollout_table(records) -> String

CSV text of a rollout log (one row per decision), for the FJ6 artifacts.
"""
function rollout_table(records::Vector{RolloutRecord})
    io = IOBuffer()
    println(io, "decision,step_count,x,z,angle,speed,d,phi,v,tile,d_stop," *
        "sigma_stop,duck_threat,hold_steps,duck_x,duck_z,duck_vel,duck_active," *
        "crossings,action,wheel_l,wheel_r,r_progress,r_lateral,r_heading," *
        "r_time,r_pedestrian,r_stagnation,r_stop_approach,r_steering," *
        "r_events,r_total,cumulative_return,terminated,truncated,reason")
    for r in records
        print(io, r.decision, ",", r.step_count, ",", r.pos[1], ",", r.pos[3],
            ",", r.angle, ",", r.speed, ",", r.d, ",", r.phi, ",", r.v, ",",
            r.tile, ",", r.d_stop === nothing ? "" : r.d_stop, ",",
            r.sigma_stop, ",", r.duck_threat, ",", r.hold_steps, ",",
            r.duck_center[1], ",", r.duck_center[3], ",", r.duck_vel, ",",
            r.duck_active, ",", r.crossings_started, ",", r.action, ",",
            r.wheels[1], ",", r.wheels[2], ",", r.reward.progress, ",",
            r.reward.lateral, ",", r.reward.heading, ",", r.reward.time, ",",
            r.reward.pedestrian, ",", r.reward.stagnation, ",",
            r.reward.stop_approach, ",", r.reward.steering, ",",
            r.reward.events, ",", r.reward.total, ",", r.cumulative_return,
            ",", r.terminated, ",", r.truncated, ",", r.reason)
        println(io)
    end
    return String(take!(io))
end

"""
    three_lane_table(process, pycall, julia, fields) -> Vector{Dict}

The FJ5-R-motivated three-value log: for each decision and each
libm-sensitive field, the value from the isolated Python reference, the
in-process Python oracle and native Julia, plus

    Δ_PJ = process - julia      (true cross-runtime difference)
    Δ_PP = process - pycall     (libm isolation effect)
    Δ_CJ = pycall  - julia      (expected ~0 if the libm hypothesis holds)
"""
function three_lane_table(process::Vector{RolloutRecord},
    pycall::Vector{RolloutRecord}, julia::Vector{RolloutRecord};
    fields=(:angle, :phi, :d))
    n = min(length(process), length(pycall), length(julia))
    rows = Dict{String,Any}[]
    for k in 1:n
        p, c, j = process[k], pycall[k], julia[k]
        row = Dict{String,Any}("decision" => k)
        for f in fields
            pv, cv, jv = getfield(p, f), getfield(c, f), getfield(j, f)
            row[String(f)] = Dict("process" => pv, "pycall" => cv,
                "julia" => jv, "d_PJ" => pv - jv, "d_PP" => pv - cv,
                "d_CJ" => cv - jv)
        end
        for (name, sel) in ("reward.heading" => r -> r.reward.heading,
            "reward.progress" => r -> r.reward.progress,
            "reward.total" => r -> r.reward.total)
            pv, cv, jv = sel(p), sel(c), sel(j)
            row[name] = Dict("process" => pv, "pycall" => cv, "julia" => jv,
                "d_PJ" => pv - jv, "d_PP" => pv - cv, "d_CJ" => cv - jv)
        end
        push!(rows, row)
    end
    return rows
end

"""
    libm_hypothesis_check(rows) -> NamedTuple

Aggregate the three-lane table into the FJ5-R prediction test:
`Δ_PJ ≈ Δ_PP` and `Δ_CJ ≈ 0` for every libm-sensitive field.

`d_CJ_exactly_zero` is the strict form. It does NOT always hold, and that is
itself informative: the in-process interposition makes CPython's `math.atan2`
resolve to Julia's libm, but NumPy's ufunc path has its own inner loop, so for
occasional inputs the embedded reference still lands 1 ULP away from Julia.
`max_abs_d_CJ` therefore reports the measured magnitude instead of hiding it
behind a boolean.
"""
function libm_hypothesis_check(rows::Vector{Dict{String,Any}})
    names = String[]
    for r in rows, (k, v) in r
        k == "decision" && continue
        k in names || push!(names, k)
    end
    out = Dict{String,Any}()
    for name in names
        max_cj = 0.0
        max_gap = 0.0     # |Δ_PJ - Δ_PP|
        for r in rows
            v = r[name]
            max_cj = max(max_cj, abs(v["d_CJ"]))
            max_gap = max(max_gap, abs(v["d_PJ"] - v["d_PP"]))
        end
        out[name] = Dict("max_abs_d_CJ" => max_cj,
            "max_abs_dPJ_minus_dPP" => max_gap)
    end
    max_cj_overall = isempty(out) ? 0.0 :
        maximum(v -> v["max_abs_d_CJ"], values(out))
    max_gap_overall = isempty(out) ? 0.0 :
        maximum(v -> v["max_abs_dPJ_minus_dPP"], values(out))
    return (fields=names, stats=out,
        max_abs_d_CJ=max_cj_overall,
        max_abs_dPJ_minus_dPP=max_gap_overall,
        d_CJ_exactly_zero=(max_cj_overall == 0.0))
end
