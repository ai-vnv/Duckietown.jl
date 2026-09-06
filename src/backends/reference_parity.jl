# The evaluation functions that REQUIRE a live reference backend: matched
# one-step comparison (FJ5) and the reference side of episode rollouts (FJ6).
# They live in backends/ because not one of their lines can execute without
# the reference installation — the same reason the transport bridges do.
# Moved verbatim from src/evaluation/{parity,rollout}.jl; behaviour, names
# and exports are unchanged, and the gated FJ5/FJ6 suites still run them.

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

