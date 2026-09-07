# The suite tests internals as well as the public API. These names were
# removed from the export list at registry review (generic words and
# underscore-prefixed internals collide with other packages in user code),
# but the tests that pin their behaviour keep first-name access through this
# one explicit import. Users write `Duckietown.decide`; the suite earns the
# shorthand by being the thing that validates them.
using Duckietown: _collision, _drivable_pos, _get_tile, _inconvenient_spawn,
    _valid_pose,
    act, decide, forward, measure, branch, update!,
    intersects, intersects_single_obj, random_sample, exact, worst, outcome,
    digitize, wrap_text, n_missing, is_frozen, model_time, grid_layout,
    select_episode, root_children, panel_ids, series_in, hold_progress,
    NONE, STRAIGHT, READY, NOT_READY, NEEDS_REFACTOR, GOAL, REWARD, COMPUTE,
    FLAG, ABSENT, PERSISTED, LOGGED, CUMULATIVE, INSTANTANEOUS, TIMEOUT,
    OFFROAD, IN_PROGRESS, MAIN_FIGURE, SUPPLEMENTARY, AGGREGATE_ONLY,
    NAVIGATION
