# FJ1 STATUS — Package skeleton, typed data model, config hierarchy, interfaces

**Status: READY FOR REVIEW** · 2026-08-18 · package `DuckietownDecisionModels.jl` v0.1.0
(UUID `b9206cf4-d873-49ff-bdb0-b4cd6bc56877`)

Reference contract: `duckduck/docs/validation/FJ0_repository_audit.md`
(accepted audit). FJ1 scope is strictly: skeleton + typed data model + config
hierarchy + interface boundaries. No dynamics, MCTS, or visualization
implementation is included.

## Deliverables

1. Package skeleton: `Project.toml` (Julia 1.10 compat; POMDPs.jl v1,
   YAML.jl), main module with dependency-ordered includes, full `src/` tree
   with gate-marked stubs (FJ2–FJ4, FJ7–FJ8).
2. Typed data model mirroring the Python reference:
   - `RawState` (7-D tabular) + `StateConfig` + `TileType`
   - `ContinuousState` (15-D, `stop_hold_progress` last) +
     `ContinuousStateConfig` + `DuckRelativeState` + `OBSERVATION_NAMES`
   - `MacroAction` (7 macro actions) + `DuckieAction` + `ActionConfig`
   - `EventFlags`, `StopTracker` (+ `hold_progress`, `reset_tracker`)
   - `RewardConfig` (23 fields) + `RewardBreakdown`
   - `DuckieWorldState` (canonical branchable dynamics state, mutable),
     `DuckieEgoState` (with `command_history` delay window),
     `DuckieState`, `StopSignState`, `StopMemory`, `TileSpec`, `RoadMap`,
     `branch()` deep-copy
3. Config hierarchy with authoritative precedence *experiment YAML > Python
   source defaults per missing key*, exactly the Python
   `Config(**yaml_dict)` semantics:
   - `load_config(path)` — typed parse of one `training_config.yaml`
   - `default_config(algorithm)` — Python source defaults
   - validation mirrors Python constructors (`spawn_route_direction`,
     `spawn_min_route_alignment ∈ [0,1]`, spawn bounds min ≤ max)
   - unknown keys ignored (Python loader behaviour)
4. Interface boundaries:
   - `AbstractBackend` — `reset!`, `step!`, `get_raw_state`,
     `get_continuous_state`; docstring locks the reference transition order
   - `AbstractPolicy` — `act`

## Locked constraints (user-approved, binding for all later gates)

1. **Structural equivalence.** The native Julia backend aims for structural
   equivalence with the audited simulator implementation; numerical
   equivalence must be established by runtime parity testing (FJ6). Docs must
   not claim "float rounding level" parity.
2. **Canonical state.** `DuckieWorldState` is the canonical branchable
   dynamics state; `RawState` and `ContinuousState` are projections. Never
   define the POMDP canonically as `MDP{ContinuousState, ...}`.
3. **Reward semantics.** SAC/TD3 steering penalty uses pre-action `kappa`
   with clipped `omega_cmd`; `reward(m, s, a, sp)` never reads `sp.kappa`.
4. **Transition order** (locked): previous state → `before_step` → action →
   wheels → `frame_skip` delayed-DB18 ticks → raw extraction → StopTracker →
   collision/termination → reward.

## Verification

- `Pkg.instantiate()` clean on the UNC path (Windows Julia 1.10.11 driving
  the WSL checkout); POMDPs.jl v1.0.0.
- `Pkg.test()`: **259 assertions pass, 0 fail**:
  - `load_config parses all four reference configs` — 186 assertions pinning
    every field of `q_learning` / `sarsa` / `sac` / `td3` against the real
    YAMLs (seeds, map, spawn, state/continuous-state blocks, duck controller,
    rewards, solvers, lane teacher, transition model, training, evaluation)
  - `default_config matches Python source defaults` — 26 assertions (e.g.
    `stop_orientation_cos 0.70710678`, `duck_corridor_width 0.35`,
    `duck_max_distance 2.0`, `goal 50.0`, `max_steer_command 1.5`)
  - `validation mirrors Python constructors` — 4
  - `enums match Python integer values` — 8
  - `OBSERVATION_NAMES is 15 features, hold progress last` — 2
  - `RawState / ContinuousState construction` — 6
  - `StopTracker state semantics` — 14
  - `DuckieWorldState is branchable` — 9
  - `EventFlags defaults` — 4

## Notable port decisions (documented in source docstrings)

- `DuckieWorldState` is mutable so `stop_memory` / `ego` / `ducks` can be
  swapped on reset and updated in place, mirroring the Python environment;
  `branch()` still deep-copies all mutable fields for rollouts.
- `initial_q_table` lives under the `training:` block in the YAMLs, not the
  solver block; it is therefore a `TrainingConfig` field (tabular adapters
  read it from there in FJ7).
- `DuckieWorldState.controller_rng` is a placeholder native
  `MersenneTwister`; the Python stream is `np.random.RandomState` (MT19937
  legacy seeding) — exact stream compatibility is an FJ3 concern.
- Enums carry the Python integer values (`TileType` 0–2, `DuckThreat` 0–4,
  `MacroAction` 0–6) for cross-language encoding parity.

## What FJ1 does NOT cover

Native DB18 dynamics, duckie motion, RNG stream parity, POMDPs.jl `gen`,
PythonCall reference backend, solver adapters, visualization, experiments.
Those are FJ2–FJ10.

## Gate exit criteria

- [x] Skeleton compiles and precompiles cleanly
- [x] All four reference configs parse with full field pinning
- [x] Python source defaults replicated and tested
- [x] Data model constructs, branches, and matches Python integer semantics
- [x] Interface boundaries + transition order documented
- [ ] User acceptance of FJ1 (this document)