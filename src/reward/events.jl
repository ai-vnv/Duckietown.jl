"""
    EventFlags

Discrete events that add a bonus or penalty on top of the dense reward
(`src/reward.py::EventFlags`). `passed_stop` is informational only; it never
enters the reward. `timeout` marks truncation (non-absorbing).
"""
struct EventFlags
    collision_duck::Bool
    other_collision::Bool
    offroad::Bool
    timeout::Bool
    stop_violation::Bool
    full_stop::Bool
    passed_stop::Bool
    goal::Bool
end

EventFlags(; collision_duck=false, other_collision=false, offroad=false,
    timeout=false, stop_violation=false, full_stop=false, passed_stop=false,
    goal=false) =
    EventFlags(collision_duck, other_collision, offroad, timeout,
        stop_violation, full_stop, passed_stop, goal)