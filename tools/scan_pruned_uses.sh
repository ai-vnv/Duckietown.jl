#!/usr/bin/env bash
# For each pruned export, list the test files that use it as a bare token.
set -u
cd "$(dirname "$0")/.."
NAMES="act decide forward measure branch update! intersects intersects_single_obj random_sample exact worst outcome digitize wrap_text n_missing is_frozen model_time grid_layout select_episode root_children panel_ids series_in hold_progress _collision _drivable_pos _get_tile _inconvenient_spawn _valid_pose NONE STRAIGHT READY NOT_READY NEEDS_REFACTOR GOAL REWARD COMPUTE FLAG ABSENT PERSISTED LOGGED CUMULATIVE INSTANTANEOUS TIMEOUT OFFROAD IN_PROGRESS MAIN_FIGURE SUPPLEMENTARY AGGREGATE_ONLY NAVIGATION"
for n in $NAMES; do
  hits=$(grep -rlE "(^|[^.\w!])${n}(\\(|[^\w!(]|$)" test/*.jl 2>/dev/null | tr '\n' ' ')
  [ -n "$hits" ] && echo "$n :: $hits"
done
exit 0
