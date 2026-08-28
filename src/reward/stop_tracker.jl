"""
    StopTracker

Dwell-counter state of the stop-compliance mechanism
(`src/reward.py::StopTracker`).

- `zone`: distance (m) within which a slow ego may accumulate dwell.
- `speed`: velocity (m/s) below which the ego counts as slow.
- `pass_distance`: distance (m) below which the stop is considered passed;
  clamped to `≥ zone` (Python: `pass_distance = max(zone, pass_distance)`).
- `hold_steps_required`: dwell steps needed to set `sigma_stop`; clamped to
  `≥ 1` (experiments: 1 tabular, 3 SAC/TD3).

The dwell counter is a mutable memory cell, updated in place by
[`update!`](@ref) exactly like the Python attribute mutations; rollouts must
call it on a branched copy (see [`branch`](@ref)).
"""
mutable struct StopTracker
    zone::Float64
    speed::Float64
    pass_distance::Float64
    hold_steps_required::Int
    hold_steps::Int
    sigma_stop::Bool
end

function StopTracker(zone=0.45, speed=0.02, pass_distance=0.55, hold_steps=1)
    StopTracker(
        Float64(zone),
        Float64(speed),
        max(Float64(zone), Float64(pass_distance)),
        max(1, Int(hold_steps)),
        0,
        false,
    )
end

"""
    hold_progress(tracker) -> Float64

Normalized dwell progress: `1.0` once stopped, else `hold_steps /
hold_steps_required` clipped to `[0, 1]`. This feature keeps the process
Markov (append-only 15th continuous feature).
"""
hold_progress(tracker::StopTracker) =
    tracker.sigma_stop ? 1.0 : min(1.0, tracker.hold_steps / tracker.hold_steps_required)

"""
    reset_tracker(tracker) -> StopTracker

Fresh tracker with the same configuration and cleared dwell memory.
"""
reset_tracker(tracker::StopTracker) =
    StopTracker(tracker.zone, tracker.speed, tracker.pass_distance,
        tracker.hold_steps_required)

"""
    update!(tracker, previous, current, previous_stop_id=nothing, current_stop_id=nothing) -> (sigma_stop, events)

Advance the stop memory over one macro-decision and return `(sigma_stop,
events)` (`src/reward.py::StopTracker.update`, exact semantics).

Passed detection (awarded once; resets dwell):
- with stop ids available (`previous_stop_id !== nothing`): passed iff the
  sign changed AND `previous.d_stop ≤ pass_distance`;
- without ids: passed iff `previous.d_stop ≤ pass_distance` AND (`current` has
  no stop OR its distance jumped `> 0.5` beyond the previous).

A passed stop sets `passed_stop`, and `stop_violation` iff `sigma_stop` was
not yet set. A sign change without a pass resets the memory. Otherwise the
dwell increments only while `near (d_stop ≤ zone) AND slow (v < speed)`, must
be consecutive (`hold_steps` resets on any non-qualifying step), and latches
`sigma_stop` at `hold_steps_required` steps with `full_stop` raised exactly
once.

Mirrors Python exactly, including mutating the tracker in place (the
canonical dwell memory lives in the world state; call this on the branched
state in rollouts).
"""
function update!(tracker::StopTracker, previous::RawState, current::RawState,
    previous_stop_id::Union{Nothing,Integer}=nothing,
    current_stop_id::Union{Nothing,Integer}=nothing)
    events = EventFlags()
    ids_available = previous_stop_id !== nothing || current_stop_id !== nothing
    if ids_available
        stop_changed = previous_stop_id !== nothing &&
            previous_stop_id != current_stop_id
        passed = stop_changed && previous.d_stop !== nothing &&
            previous.d_stop <= tracker.pass_distance
    else
        passed = previous.d_stop !== nothing &&
            previous.d_stop <= tracker.pass_distance &&
            (current.d_stop === nothing || current.d_stop > previous.d_stop + 0.5)
    end

    if passed
        events = EventFlags(passed_stop=true,
            stop_violation=!tracker.sigma_stop)
        tracker.sigma_stop = false
        tracker.hold_steps = 0
        return tracker.sigma_stop, events
    end

    if ids_available && stop_changed
        tracker.sigma_stop = false
        tracker.hold_steps = 0
    end

    near = current.d_stop !== nothing && current.d_stop <= tracker.zone
    slow = current.v < tracker.speed
    if !tracker.sigma_stop
        if near && slow
            tracker.hold_steps += 1
            if tracker.hold_steps >= tracker.hold_steps_required
                tracker.sigma_stop = true
                tracker.hold_steps = tracker.hold_steps_required
                events = EventFlags(full_stop=true)
            end
        else
            tracker.hold_steps = 0
        end
    end
    return tracker.sigma_stop, events
end