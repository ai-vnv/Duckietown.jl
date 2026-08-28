# FJ9.6a — decision-artifact contract

Source: `artifacts/fj8/enriched/decisions.csv`, fingerprint `952e2a164018018a`, 16522 decisions over 120 episodes.

| Quantity | Column | Status | Empty rows | Evidence |
|---|---|---|---|---|
| episode identity | `solver` | LOGGED | 0 | 16522 rows, 6 distinct, none empty |
| episode identity | `seed` | LOGGED | 0 | 16522 rows, 20 distinct, none empty |
| episode identity | `decision` | LOGGED | 0 | 16522 rows, 150 distinct, none empty |
| ego world pose | `ego_x` | LOGGED | 0 | 16522 rows, 13705 distinct, none empty |
| ego world pose | `ego_z` | LOGGED | 0 | 16522 rows, 13705 distinct, none empty |
| ego world pose | `ego_angle` | LOGGED | 0 | 16522 rows, 13669 distinct, none empty |
| ego world pose | `ego_speed` | LOGGED | 0 | 16522 rows, 13666 distinct, none empty |
| lane projection | `d` | LOGGED | 0 | 16522 rows, 13620 distinct, none empty |
| lane projection | `phi` | LOGGED | 0 | 16522 rows, 13742 distinct, none empty |
| lane projection | `kappa` | LOGGED | 0 | 16522 rows, 6 distinct, none empty |
| action identity and command | `action_kind` | LOGGED | 0 | 16522 rows, 2 distinct, none empty |
| action identity and command | `action_id` | LOGGED | 0 | 16522 rows, 8 distinct, none empty |
| action identity and command | `v_cmd` | LOGGED | 0 | 16522 rows, 7628 distinct, none empty |
| action identity and command | `omega_cmd` | LOGGED | 0 | 16522 rows, 7713 distinct, none empty |
| reward total and components | `reward_total` | LOGGED | 0 | 16522 rows, 13771 distinct, none empty |
| reward total and components | `reward_progress` | LOGGED | 0 | 16522 rows, 13768 distinct, none empty |
| reward total and components | `reward_lateral` | LOGGED | 0 | 16522 rows, 13620 distinct, none empty |
| reward total and components | `reward_heading` | LOGGED | 0 | 16522 rows, 13742 distinct, none empty |
| reward total and components | `reward_time` | LOGGED | 0 | 16522 rows, 1 distinct, none empty |
| reward total and components | `reward_pedestrian` | LOGGED | 0 | 16522 rows, 2 distinct, none empty |
| reward total and components | `reward_stagnation` | LOGGED | 0 | 16522 rows, 2 distinct, none empty |
| reward total and components | `reward_stop_approach` | LOGGED | 0 | 16522 rows, 3 distinct, none empty |
| reward total and components | `reward_steering` | LOGGED | 0 | 16522 rows, 4253 distinct, none empty |
| reward total and components | `reward_events` | LOGGED | 0 | 16522 rows, 6 distinct, none empty |
| stop subsystem | `d_stop` | LOGGED | 12814 | 16522 rows, 12814 empty (77.6%) — MISSING, never zero |
| stop subsystem | `sigma_stop` | LOGGED | 0 | 16522 rows, 2 distinct, none empty |
| stop subsystem | `stop_hold_progress` | LOGGED | 0 | 16522 rows, 4 distinct, none empty |
| stop subsystem | `full_stop` | LOGGED | 0 | 16522 rows, 2 distinct, none empty |
| stop subsystem | `passed_stop` | LOGGED | 0 | 16522 rows, 2 distinct, none empty |
| stop subsystem | `stop_violation` | LOGGED | 0 | 16522 rows, 2 distinct, none empty |
| duck subsystem | `duck_present` | LOGGED | 0 | 16522 rows, 2 distinct, none empty |
| duck subsystem | `duck_longitudinal` | LOGGED | 0 | 16522 rows, 6701 distinct, none empty |
| duck subsystem | `duck_lateral` | LOGGED | 0 | 16522 rows, 6701 distinct, none empty |
| duck subsystem | `duck_active` | LOGGED | 0 | 16522 rows, 2 distinct, none empty |
| duck subsystem | `duck_class` | LOGGED | 0 | 16522 rows, 5 distinct, none empty |
| duck subsystem | `duck_active_state` | LOGGED | 0 | 16522 rows, 2 distinct, none empty |
| duck subsystem | `crossings_started` | LOGGED | 0 | 16522 rows, 2 distinct, none empty |
| episode outcome | `terminated` | LOGGED | 0 | 16522 rows, 2 distinct, none empty |
| episode outcome | `truncated` | LOGGED | 0 | 16522 rows, 1 distinct, none empty |
| episode outcome | `reason` | LOGGED | 0 | 16522 rows, 3 distinct, none empty |
| planning cost | `planning_time` | LOGGED | 0 | 16522 rows, 13663 distinct, none empty |
| planning cost | `model_calls` | LOGGED | 0 | 16522 rows, 1287 distinct, none empty |
| reward | `cumulative_return` | DERIVED (exact identity) | — | running sum of reward_total; FJ8.4c showed the final value equals the FJ8.4b episode return exactly |
| episode outcome | `horizon_reached` | DERIVED (exact identity) | — | terminated == false on the final row; 'truncated' is false on all 16522 rows and carries no signal on its own |
| not recorded | `observation as the policy saw it` | ABSENT | — | the evaluation never stored it; FJ9.5 persists search trees only for the two explicitly captured snapshots |
| not recorded | `belief state` | ABSENT | — | the evaluation never stored it; FJ9.5 persists search trees only for the two explicitly captured snapshots |
| not recorded | `per-decision search tree` | ABSENT | — | the evaluation never stored it; FJ9.5 persists search trees only for the two explicitly captured snapshots |
| not recorded | `wall-clock timestamp` | ABSENT | — | the evaluation never stored it; FJ9.5 persists search trees only for the two explicitly captured snapshots |
