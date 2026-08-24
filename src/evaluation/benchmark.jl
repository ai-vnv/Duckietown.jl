# FJ8.0 — the generative-model contract and its cost, measured before any
# planner exists.
#
# MCTS and DPW call `gen` thousands of times per decision, and every call must
# be safe to branch from the same node. Two things therefore have to be
# established first, in this order:
#
#   1. CONTRACT   determinism, root immutability, no aliasing between branches
#   2. COST       time, allocations and bytes per `gen`, and where they go
#
# A planner built on a `gen` that quietly mutates its input produces results
# that look plausible and are wrong, so the contract is checked structurally
# (bitwise field-by-field), not by spot-checking a few numbers.
#
# Nothing here asserts a performance threshold. The cost is *measured and
# reported*; what counts as acceptable is a decision for the planner gates,
# not a constant invented here.

# ---------------------------------------------------------------------------
# Structural state comparison
# ---------------------------------------------------------------------------

"""
    WorldMismatch

One field where two world states differ, with both values rendered for the
report. Floats are compared **bitwise** (`===`), so `0.0` and `-0.0` count as
different and `NaN` equals `NaN` — a planner-purity check must not be fooled
by numeric coincidence.
"""
struct WorldMismatch
    field::String
    a::String
    b::String
end

Base.show(io::IO, m::WorldMismatch) =
    print(io, m.field, ": ", m.a, " vs ", m.b)

_same(x::Float64, y::Float64) = x === y
_same(x, y) = isequal(x, y)

function _cmp!(d::Vector{WorldMismatch}, name::AbstractString, x, y)
    _same(x, y) || push!(d, WorldMismatch(String(name), repr(x), repr(y)))
    return d
end

function _cmp_len!(d::Vector{WorldMismatch}, name::AbstractString, x, y)
    if length(x) != length(y)
        push!(d, WorldMismatch(String(name) * ".length",
            repr(length(x)), repr(length(y))))
        return false
    end
    return true
end

function _cmp_ego!(d, pre, a::DuckieEgoState, b::DuckieEgoState)
    for k in 1:3
        _cmp!(d, "$pre.pos[$k]", a.pos[k], b.pos[k])
    end
    _cmp!(d, "$pre.angle", a.angle, b.angle)
    _cmp!(d, "$pre.v_long", a.v_long, b.v_long)
    _cmp!(d, "$pre.omega", a.omega, b.omega)
    _cmp!(d, "$pre.speed", a.speed, b.speed)
    _cmp!(d, "$pre.step_count", a.step_count, b.step_count)
    _cmp!(d, "$pre.timestamp", a.timestamp, b.timestamp)
    _cmp!(d, "$pre.axis_left_rad", a.axis_left_rad, b.axis_left_rad)
    _cmp!(d, "$pre.axis_right_rad", a.axis_right_rad, b.axis_right_rad)
    if _cmp_len!(d, "$pre.command_history", a.command_history, b.command_history)
        for (k, (ca, cb)) in enumerate(zip(a.command_history, b.command_history))
            for c in 1:3
                _cmp!(d, "$pre.command_history[$k][$c]", ca[c], cb[c])
            end
        end
    end
    for (nm, x, y) in (("q0", a.q0, b.q0), ("v0", a.v0, b.v0))
        if size(x) == size(y)
            for i in eachindex(x)
                _cmp!(d, "$pre.$nm[$i]", x[i], y[i])
            end
        else
            push!(d, WorldMismatch("$pre.$nm.size", repr(size(x)), repr(size(y))))
        end
    end
    return d
end

function _cmp_duck!(d, pre, a::DuckieState, b::DuckieState)
    for (nm, x, y) in (("pos", a.pos, b.pos), ("center", a.center, b.center),
        ("start", a.start, b.start), ("heading", a.heading, b.heading),
        ("min_coords", a.min_coords, b.min_coords),
        ("max_coords", a.max_coords, b.max_coords))
        for k in 1:3
            _cmp!(d, "$pre.$nm[$k]", x[k], y[k])
        end
    end
    _cmp!(d, "$pre.angle", a.angle, b.angle)
    _cmp!(d, "$pre.vel", a.vel, b.vel)
    _cmp!(d, "$pre.visible", a.visible, b.visible)
    _cmp!(d, "$pre.pedestrian_active", a.pedestrian_active, b.pedestrian_active)
    _cmp!(d, "$pre.pedestrian_wait_time", a.pedestrian_wait_time,
        b.pedestrian_wait_time)
    _cmp!(d, "$pre.time", a.time, b.time)
    _cmp!(d, "$pre.walk_distance", a.walk_distance, b.walk_distance)
    _cmp!(d, "$pre.scale", a.scale, b.scale)
    _cmp!(d, "$pre.safety_radius", a.safety_radius, b.safety_radius)
    if _cmp_len!(d, "$pre.obj_corners", a.obj_corners, b.obj_corners)
        for (k, (ca, cb)) in enumerate(zip(a.obj_corners, b.obj_corners))
            _cmp!(d, "$pre.obj_corners[$k][1]", ca[1], cb[1])
            _cmp!(d, "$pre.obj_corners[$k][2]", ca[2], cb[2])
        end
    end
    if size(a.obj_norm) == size(b.obj_norm)
        for i in eachindex(a.obj_norm)
            _cmp!(d, "$pre.obj_norm[$i]", a.obj_norm[i], b.obj_norm[i])
        end
    else
        push!(d, WorldMismatch("$pre.obj_norm.size",
            repr(size(a.obj_norm)), repr(size(b.obj_norm))))
    end
    return d
end

"""
    world_differences(a, b) -> Vector{WorldMismatch}

Every field in which two world states differ, compared bitwise for floats and
exactly for everything else, including container lengths. `map` is compared by
identity: it is a shared read-only description of the track, not per-branch
state.
"""
function world_differences(a::DuckieWorldState, b::DuckieWorldState)
    d = WorldMismatch[]
    _cmp_ego!(d, "ego", a.ego, b.ego)
    if _cmp_len!(d, "ducks", a.ducks, b.ducks)
        for (i, (da, db)) in enumerate(zip(a.ducks, b.ducks))
            _cmp_duck!(d, "ducks[$i]", da, db)
        end
    end
    if _cmp_len!(d, "stop_signs", a.stop_signs, b.stop_signs)
        for (i, (sa, sb)) in enumerate(zip(a.stop_signs, b.stop_signs))
            for k in 1:3
                _cmp!(d, "stop_signs[$i].pos[$k]", sa.pos[k], sb.pos[k])
            end
            _cmp!(d, "stop_signs[$i].angle", sa.angle, sb.angle)
        end
    end
    a.map === b.map || push!(d, WorldMismatch("map", "identity", "differs"))
    _cmp!(d, "stop_memory.sigma_stop", a.stop_memory.sigma_stop,
        b.stop_memory.sigma_stop)
    _cmp!(d, "stop_memory.hold_steps", a.stop_memory.hold_steps,
        b.stop_memory.hold_steps)
    _cmp!(d, "stop_memory.last_stop_id", a.stop_memory.last_stop_id,
        b.stop_memory.last_stop_id)
    _cmp!(d, "stop_memory.last_d_stop", a.stop_memory.last_d_stop,
        b.stop_memory.last_d_stop)
    _cmp!(d, "lane_fallback[1]", a.lane_fallback[1], b.lane_fallback[1])
    _cmp!(d, "lane_fallback[2]", a.lane_fallback[2], b.lane_fallback[2])
    if _cmp_len!(d, "crossings_started", a.crossings_started, b.crossings_started)
        for i in eachindex(a.crossings_started)
            _cmp!(d, "crossings_started[$i]", a.crossings_started[i],
                b.crossings_started[i])
        end
    end
    if _cmp_len!(d, "crossing_armed", a.crossing_armed, b.crossing_armed)
        for i in eachindex(a.crossing_armed)
            _cmp!(d, "crossing_armed[$i]", a.crossing_armed[i],
                b.crossing_armed[i])
        end
    end
    _cmp!(d, "controller_rng", a.controller_rng, b.controller_rng)
    return d
end

"""
    worlds_identical(a, b) -> Bool

`true` when [`world_differences`](@ref) is empty.
"""
worlds_identical(a::DuckieWorldState, b::DuckieWorldState) =
    isempty(world_differences(a, b))

"""
    shared_mutable_arrays(a, b) -> Vector{String}

Names of the *per-branch* mutable containers that the two states hold as the
**same object** (`===`). Two branches sharing one of these would alias:
writing through one would silently change the other. An empty result is the
branch-purity property a planner needs.

Fields shared **by design** are excluded and reported separately by
[`shared_by_design`](@ref); see that docstring for why each one is safe.
"""
function shared_mutable_arrays(a::DuckieWorldState, b::DuckieWorldState)
    shared = String[]
    a.ego.command_history === b.ego.command_history &&
        push!(shared, "ego.command_history")
    a.ego.q0 === b.ego.q0 && push!(shared, "ego.q0")
    a.ego.v0 === b.ego.v0 && push!(shared, "ego.v0")
    a.ducks === b.ducks && push!(shared, "ducks")
    a.crossings_started === b.crossings_started &&
        push!(shared, "crossings_started")
    a.crossing_armed === b.crossing_armed && push!(shared, "crossing_armed")
    for i in 1:min(length(a.ducks), length(b.ducks))
        a.ducks[i].obj_corners === b.ducks[i].obj_corners &&
            push!(shared, "ducks[$i].obj_corners")
        a.ducks[i].obj_norm === b.ducks[i].obj_norm &&
            push!(shared, "ducks[$i].obj_norm")
    end
    return shared
end

"""
    shared_by_design(a, b) -> Vector{String}

The objects two states are *expected* to share, listed explicitly so the
sharing stays visible rather than accidental:

- `map` and `stop_signs` — the read-only track description. Copying these per
  node would dominate a planner's memory and nothing ever writes to them.
- `controller_rng` — a legacy field. Since FJ3.8 the transition's stochasticity
  comes from the caller's `rng` (`x' ~ T(·|x,a)` with externally supplied
  noise); the decision chain never draws from the state's stream. It is
  carried by reference because copying a `MersenneTwister` costs ~9 % of one
  `gen` call's total allocation for a field with no semantic role.

The sharing is only safe while the object is never written to, which is a
measurable property, not an assumption — [`rng_frozen`](@ref) checks it, and
`world_differences` compares the stream position so any advance shows up as a
`controller_rng` mismatch.
"""
function shared_by_design(a::DuckieWorldState, b::DuckieWorldState)
    shared = String[]
    a.map === b.map && push!(shared, "map")
    a.stop_signs === b.stop_signs && push!(shared, "stop_signs")
    a.controller_rng === b.controller_rng && push!(shared, "controller_rng")
    return shared
end

"""
    rng_frozen(states, reference) -> Bool

`true` when every state's `controller_rng` still sits at exactly the stream
position of `reference` — i.e. the shared legacy stream was never drawn from
while the branches were built. This is what makes sharing it equivalent to
sharing the map.
"""
rng_frozen(states, reference::MersenneTwister) =
    all(s -> s.controller_rng == reference, states)

# ---------------------------------------------------------------------------
# Cost measurement
# ---------------------------------------------------------------------------

"""
    GenBenchmark

Measured cost of `calls` repetitions of one operation. Per-call figures come
from [`per_call_us`](@ref), [`bytes_per_call`](@ref) and
[`allocs_per_call`](@ref).
"""
struct GenBenchmark
    label::String
    calls::Int
    seconds::Float64
    bytes::Int
    allocs::Int
end

per_call_us(b::GenBenchmark) = b.calls == 0 ? NaN : 1e6 * b.seconds / b.calls
bytes_per_call(b::GenBenchmark) = b.calls == 0 ? NaN : b.bytes / b.calls
allocs_per_call(b::GenBenchmark) = b.calls == 0 ? NaN : b.allocs / b.calls
calls_per_second(b::GenBenchmark) = b.seconds == 0 ? Inf : b.calls / b.seconds

_alloc_count(d::Base.GC_Diff) = d.malloc + d.realloc + d.poolalloc + d.bigalloc

"""
    measure(f, calls; label) -> GenBenchmark

Run `f()` `calls` times and record wall time, bytes allocated and allocation
count. `f` is called once first to force compilation, and the result of every
call is retained so the work cannot be optimised away. A `GC.gc()` runs before
the measurement so the reported bytes belong to this loop.
"""
function measure(f, calls::Integer; label::AbstractString="")
    sink = Ref{Any}(f())                     # warm up + keep the result alive
    GC.gc()
    before = Base.gc_num()
    t0 = time_ns()
    for _ in 1:calls
        sink[] = f()
    end
    elapsed = (time_ns() - t0) / 1e9
    diff = Base.GC_Diff(Base.gc_num(), before)
    sink[] === nothing && error("unreachable")   # keep `sink` observable
    return GenBenchmark(String(label), Int(calls), elapsed, diff.allocd,
        _alloc_count(diff))
end

"""
    benchmark_gen(mdp, s0, a; calls, mode, seed) -> GenBenchmark

Cost of `POMDPs.gen` on the validated MDP.

- `mode = :branch` — every call starts from the same `s0`. This is what a
  planner does when it expands one node, and it is the figure that multiplies
  by the per-decision node budget.
- `mode = :chain` — each call continues from the previous successor, resetting
  to `s0` on a terminal or truncated state. This is the rollout cost, and it
  differs from `:branch` because the command history and duck state evolve.
"""
function benchmark_gen(mdp::DuckietownMDP, s0::DuckieWorldState, a;
    calls::Integer=1000, mode::Symbol=:branch, seed::Integer=1,
    label::AbstractString="gen($mode)")
    rng = MersenneTwister(seed)
    if mode === :branch
        return measure(() -> POMDPs.gen(mdp, s0, a, rng), calls; label=label)
    elseif mode === :chain
        s = Ref(s0)
        f = function ()
            r = simulate_decision(mdp.transition, s[], a, rng)
            s[] = (r.terminated || r.truncated) ? s0 : r.sp
            return (sp=r.sp, r=r.reward.total)
        end
        return measure(f, calls; label=label)
    end
    throw(ArgumentError("mode must be :branch or :chain"))
end

"""
    gen_scaling(mdp, s0, a; calls, mode, seed) -> Vector{GenBenchmark}

The same measurement repeated at several call counts, so a per-call figure
distorted by one-off costs is visible rather than hidden.
"""
gen_scaling(mdp::DuckietownMDP, s0::DuckieWorldState, a;
    calls=(1, 100, 1_000, 10_000), mode::Symbol=:branch, seed::Integer=1) =
    [benchmark_gen(mdp, s0, a; calls=n, mode=mode, seed=seed,
        label="gen($mode) n=$n") for n in calls]

"""
    gen_stage_profile(mdp, s0, a; calls, seed) -> Vector{GenBenchmark}

Where one `gen` call's time and memory actually go, measured stage by stage
along the locked transition order. `branch(world)` is included as a reference
point: it is the pure cost of deep-copying a world state, so any stage costing
less than a branch is not worth optimising first.
"""
function gen_stage_profile(mdp::DuckietownMDP, s0::DuckieWorldState, a;
    calls::Integer=1000, seed::Integer=1)
    m = mdp.transition
    rng = MersenneTwister(seed)
    w0 = before_step(s0, m.duck_cfg, MersenneTwister(seed))
    wheels64 = (0.5, 0.5)
    w1 = ego_tick(w0, wheels64)
    raw, _ = get_raw_state(w1, m.state_cfg; sigma_stop=false)
    events = EventFlags()
    out = GenBenchmark[]
    push!(out, measure(() -> branch(s0), calls; label="branch(world)"))
    push!(out, measure(() -> before_step(s0, m.duck_cfg, rng), calls;
        label="before_step"))
    push!(out, measure(() -> ego_tick(w0, wheels64), calls; label="ego_tick"))
    if !isempty(w0.ducks)
        push!(out, measure(() -> duck_step(w0, 1), calls; label="duck_step"))
    end
    push!(out, measure(() -> get_raw_state(w1, m.state_cfg; sigma_stop=false),
        calls; label="get_raw_state"))
    push!(out, measure(() -> next_stop_candidate(w1, m.state_cfg), calls;
        label="next_stop_candidate"))
    push!(out, measure(() -> termination_reason(m, w1), calls;
        label="termination_reason"))
    push!(out, measure(() -> compute_reward(raw, events, m.reward_cfg), calls;
        label="compute_reward"))
    push!(out, measure(() -> get_continuous_state(w1, raw, m.state_cfg,
            m.continuous_cfg; controller_cfg=m.duck_cfg), calls;
        label="get_continuous_state"))
    push!(out, benchmark_gen(mdp, s0, a; calls=calls, mode=:branch,
        label="FULL gen"))
    return out
end

"""
    benchmark_table(benchmarks) -> String

Fixed-width report: per-call microseconds, allocations and kilobytes, plus the
achievable call rate. Printed by the FJ8 tests and pasted into the gate
document.
"""
function benchmark_table(bs::AbstractVector{GenBenchmark})
    width = maximum(length(b.label) for b in bs; init=10)
    io = IOBuffer()
    println(io, rpad("operation", width), "  ", lpad("calls", 7), "  ",
        lpad("us/call", 10), "  ", lpad("allocs", 9), "  ",
        lpad("KiB/call", 10), "  ", lpad("calls/s", 10))
    println(io, "-"^(width + 56))
    for b in bs
        println(io, rpad(b.label, width), "  ", lpad(b.calls, 7), "  ",
            lpad(round(per_call_us(b); digits=2), 10), "  ",
            lpad(round(allocs_per_call(b); digits=1), 9), "  ",
            lpad(round(bytes_per_call(b) / 1024; digits=2), 10), "  ",
            lpad(round(Int, calls_per_second(b)), 10))
    end
    return String(take!(io))
end

"""
    planning_budget_estimate(gen_bench, nodes) -> NamedTuple

What a per-decision node budget costs at the measured `gen` rate: seconds per
decision and megabytes allocated per decision. This is the number that decides
whether a planner budget is affordable, and it is derived from measurement
rather than assumed.
"""
planning_budget_estimate(b::GenBenchmark, nodes::Integer) = (
    nodes=Int(nodes),
    seconds_per_decision=nodes * b.seconds / b.calls,
    mib_per_decision=nodes * (b.bytes / b.calls) / 1024^2,
)
