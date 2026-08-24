# DuckietownDecisionModels.jl

Native Julia reimplementation of the decision-making stack of
[`duckduck`](https://github.com/duckietown/duckduck) — the Duckietown
deadline-driven SLAM/planning reward-machinery experiment suite — as a
POMDPs.jl ecosystem package.

The reference semantics are pinned by the repository audit
[`../duckduck/docs/FJ0_repository_audit.md`](../duckduck/docs/FJ0_repository_audit.md), which is the contract every port gate (`FJ2`–`FJ10` below)
is verified against.

## Quickstart

No Python, no solver library, no external data file — the map is embedded.

```julia
using Pkg
Pkg.activate("duckie-nb")          # a project of its own; the compat bounds
Pkg.add(url="https://github.com/PannnTastic/DuckietownDecisionModels.jl")
Pkg.add(["POMDPs", "CairoMakie"])  # CairoMakie is optional, for figures
```

```julia
using DuckietownDecisionModels, POMDPs, Random

mdp = DuckietownMDP(scenario_config(:stop_and_duck); action_space=:discrete)
s   = rand(MersenneTwister(1001), initialstate(mdp))
sp, r = @gen(:sp, :r)(mdp, s, FAST_STRAIGHT, MersenneTwister(7))
```

Runnable version, with a lane follower, the privileged-state projections and
an optional figure: [`examples/quickstart.jl`](examples/quickstart.jl), or the
same thing as a notebook, [`examples/quickstart.ipynb`](examples/quickstart.ipynb).
The notebook is generated from the script by `tools/make_notebook.jl` and CI
fails if the two drift apart.

> **Which config?** `scenario_config(:stop_and_duck)` builds a world that
> exercises the task. `default_config` returns the Python source defaults,
> which contain **no stop sign at all** — useful for provenance, useless for
> driving. Neither reproduces a reported experiment: those need the frozen
> `training_config.yaml` and policy checkpoints from the separate `duckduck`
> supplementary package, which are not distributed here.

## Status

| Gate | Scope | Status |
|------|-------|--------|
| FJ1 | Package skeleton, typed data model, config hierarchy, interface boundaries | Done (accepted) |
| FJ2 | Semantic port of deterministic components: actions, discretizer, encoding, StopTracker, reward, state projections | Done — full PYTHON ↔ JULIA parity (see [docs/FJ2_STATUS.md](docs/FJ2_STATUS.md)) |
| FJ3 | Native dynamics, observers, full transition chain, RNG | **Complete** (FJ3.1–FJ3.8). Map/collision/spawn, DB18 delayed dynamics + delay buffer, duckie walk, `DuckController.before_step`, state observers, `simulate_decision`/`TransitionResult` (discrete + continuous, pinned against the real `DuckieMDPEnv.step`/`ContinuousDuckieMDPEnv.step`), and exact NumPy RNG streams (`NumpyMT19937`, `NumpySeedSequence`/`NumpyPCG64`) — see [docs/FJ3_STATUS.md](docs/FJ3_STATUS.md) |
| FJ4 | POMDPs.jl interface: `DuckietownMDP{MacroAction}` / `DuckietownMDP{DuckieAction}`, `gen` as a thin adapter over `simulate_decision`, `initialstate`, `isterminal` (terminal vs truncation), `discount`, `actions` | **Done** (2 028 assertions — see [docs/FJ4_STATUS.md](docs/FJ4_STATUS.md)) |
| FJ5 | Live reference backend (out-of-process JSON-lines) + matched-state one-step parity + stop-sign reachability probe | **Done** (94 assertions — see [docs/FJ5_STATUS.md](docs/FJ5_STATUS.md)) |
| FJ5-R | In-process PythonCall reference backend (WSL Julia ↔ `ddm-ref`), transport equivalence, libm-interposition finding | **Done** (66 assertions — see [docs/FJ5R_STATUS.md](docs/FJ5R_STATUS.md)) |
| FJ6 | Free-running full-episode rollout parity: three lanes, Type-1/Type-2 drift split, event timing, stop/duck exercise | **Done** (5 trajectories, 539 decisions; 38-assertion regression guard — see [docs/FJ6_STATUS.md](docs/FJ6_STATUS.md)) |
| FJ7 | Solver baselines: all four shipped policies | **Done** — Q-learning/SARSA exact on all 9 000 states (action, tie set, margin); SAC/TD3 weights and inputs bit-exact with activations matched layer by layer over 1 000 observations (worst action Δ 4.5e-6 / 5.1e-6); one shared evaluator for all four. See [docs/FJ7_STATUS.md](docs/FJ7_STATUS.md) |
| FJ8 | Solver compatibility: online planning | **Done** (FJ8.0–8.5) — generative contract (~100 µs / 213 KiB / 4 929 allocations per `gen`), solver-agnostic planner contract, and MCTS.jl driving both models through the standard `solve`/`action` sequence with no conversion: vanilla MCTS on the 7 macro actions, DPW on the continuous box with action widening measured to follow `⌈4√N⌉` and state widening disabled on measured grounds. FJ8.4a measured the planner cost curve and compute-matched operating points (±2 % on generative calls); FJ8.4b compared all six solvers on 20 frozen seeds with task performance and computational cost as separate blocks and no combined ranking — see [docs/FJ8_STATUS.md](docs/FJ8_STATUS.md) |
| FJ10 | POMDP readiness audit (run **before** FJ9, so the renderer is built around the right extension points) | **Done** — 8 READY / 1 NEEDS_REFACTOR / 7 NOT_READY, each decided by probing the package. `Observation ≠ ContinuousState` settled quantitatively: only 6 of the 15 privileged components are sensor-estimable and 2 have no physical counterpart. See [docs/FJ10_STATUS.md](docs/FJ10_STATUS.md) |
| FJ8.4c | Evaluation artefact enrichment (frozen protocol, richer log) | **Done** — 120/120 episodes reproduced **exactly** (0 mismatches across 22 fields) with 16 522 per-decision rows added; the FJ8.4b artefact is untouched and not superseded. See [docs/FJ84C_STATUS.md](docs/FJ84C_STATUS.md) |
| FJ9 | Visualization suite | **FJ9.0–9.5 done** — all geometry lives in the core and is tested with no backend installed (193 assertions); Makie is a third weak dependency and only draws. The renderer caught a real defect: the stop line had been drawn along the sign's facing rather than the ego's lane frame, and is now pinned by an identity test against `d_stop`. The projection panel's semantics (labels, units, subsystem, FJ10 privilege class) are core data; the backend only lays them out. See [docs/FJ9_STATUS.md](docs/FJ9_STATUS.md) |
| FJ9.6 | Diagnostic time series | **Done** — read from the frozen FJ8.4c log with no environment, policy or planner run; 42 columns LOGGED / 2 derived by exact identity / 4 reported ABSENT; `missing` survives to the axis; the aggregate reproduces FJ8.4c's compute-collapse table from an independent binning. Surfaced a real defect in FJ8's prose: TD3 does reach the stop sign, stops, and never proceeds in **20 of 20** seeds — a stall that every aggregate metric described correctly and no aggregate metric revealed. See [docs/FJ96_STATUS.md](docs/FJ96_STATUS.md) |
| FJ9.7 | Artifact-driven animation | **Done** — playback of the same frozen log, with `ENV_MODEL_CALLS=0` measured through an `InstrumentedMDP` on every render. The timeline is the decision index, because the artefact records no wall-clock time. The trajectory at frame *t* is rows 1..*t* and an event marker appears at the frame it was logged, both enforced structurally and checked by four negative controls on edited copies of the real log. Paired playback keeps the absolute decision index and freezes the shorter episode rather than stretching it. See [docs/FJ97_STATUS.md](docs/FJ97_STATUS.md) |
| FJ9.8 | Publication composites | **Done** — four main figures built from live data objects (never assembled from PNGs), exported as PDF/SVG/PNG from a fresh process with `ENV_MODEL_CALLS=0`, no Python and no MCTS. Captions are generated from the payloads and **validated against required and forbidden phrases before the figure is drawn**, so the FJ9.6 correction cannot regress: Figure 4 forbids the superseded TD3 wording and requires "reaches the stop sign", "never proceeds" and `passed_stops = 0`. See [docs/FJ98_STATUS.md](docs/FJ98_STATUS.md) |
| FJ9.9 | Reproducibility closure | **Done** — structured JSON reporter is now authoritative and agrees exactly with both legacy parsers (200 / 148 823 / 0). Executable audits: an artefact ledger separating `PERSISTED_SOURCE` from `REBUILT`, a documentation audit that bans superseded claims and resolves every link, a `src/` import lint, and a core-formulation fingerprint proven identical with Makie and MCTS loaded. Known limitations are recorded in `artifacts/fj9/reproducibility_manifest.json` rather than omitted. See [docs/FJ99_STATUS.md](docs/FJ99_STATUS.md) |

Full suite (`Pkg.test`, exit 0 everywhere):

| Runtime | Testsets | Assertions | Scope |
|---|---|---|---|
| WSL Julia 1.11.3 + `ddm-ref` + `ddm-torch` + MCTS.jl | 200 | **148 823** | FJ1–FJ8, FJ8.4c, FJ10, FJ9.0–9.9 (live reference transports, rollout parity, four learned baselines, two external planners, cost curve, six-solver comparison, POMDP readiness audit, world renderer, projection panel, policy slices, diagnostic time series, artifact-driven animation, publication composites, reproducibility closure) |

Reproduce with `tools/run_full_suite.sh`. Since FJ9.9b the authoritative
count is the structured report the suite writes to
`artifacts/fj9/test_report.json`, built from the `Test` result tree rather
than by parsing terminal output. `tools/suite_summary.sh` and the independent
`tools/suite_summary.py` are kept as cross-checks and must agree with it —
they disagreed on the reporter's very first run, which is how a bug in the
reporter was caught.

**Correction.** Suite totals reported in the earlier gate documents (80 583,
82 328, 82 360, 82 442) were undercounts produced by a faulty counting command
— they are smaller than the sum of their own per-gate figures (FJ2 alone
records 49 012 assertions and FJ3.7 alone 59 870). The number above is the sum
of the Pass column of every top-level testset row, measured twice by
unrelated parsers. The historical documents are left as written; only this
index is corrected.

**Julia version note.** The package targets Julia 1.10, but the *live parity*
test sets require **Julia ≥ 1.11**: Julia 1.10.11 on Windows crashes its own
runtime (`EXCEPTION_ACCESS_VIOLATION` in `gc_mark_stack`, raised from the
compiler's inlining pass) while compiling them. They skip themselves with an
explicit message on 1.10; everything else is unaffected. Details in
[docs/FJ5_STATUS.md](docs/FJ5_STATUS.md).

The validation environment is pinned per directory with juliaup
(`juliaup override set 1.11.3` on this repo), so the rest of the machine keeps
its own default. Manifests are split per Julia version (`Manifest.toml` for
1.10, `Manifest-v1.11.toml` for 1.11).

**Python is never required to use this package.** `using
DuckietownDecisionModels` loads no Python at all; PythonCall is a *weak*
dependency powering only the FJ5-R in-process reference backend
(`ext/DuckietownPythonCallExt.jl`), pinned to the exact validated version
`=0.9.25` because newer releases refuse the reference interpreter's Python 3.9.
Both parity test sets skip cleanly without the reference environment.

**No solver is required either.** MCTS.jl is a weak dependency powering only
`ext/DuckietownMCTSExt.jl`, which adds tree statistics to the generic planning
diagnostics and nothing else. FJ8.5 verifies in a separate process that the
model, reward, evaluator and all four learned baselines produce an identical
fingerprint with the solver absent and with it loaded.

## Quick start

```julia
using DuckietownDecisionModels, POMDPs, Random

mdp = DuckietownMDP("../duckduck/policies/q_learning/training_config.yaml")
rng = MersenneTwister(53)
s0  = rand(rng, initialstate(mdp))
x   = gen(mdp, s0, FAST_STRAIGHT, rng)      # x.sp, x.r

# continuous variant — same problem, different action representation
mdpc = DuckietownMDP("../duckduck/policies/sac/training_config.yaml";
                     action_space = :continuous)
a = rand(rng, actions(mdpc))                # DuckieAction(v, omega)

# the full audit trail (reward breakdown, events, reason, projections)
r = simulate_decision(mdp.transition, s0, FAST_STRAIGHT, rng)
r.reward.progress, r.events.full_stop, r.reason, r.continuous_state
```

## Design constraints (locked with the user)

1. **Structural, not numerical, equivalence.** The native Julia backend aims
   for structural equivalence with the audited simulator implementation.
   Numerical equivalence is *not* established by this audit and must be
   demonstrated by runtime parity testing (`FJ6`).
2. **Canonical state.** `DuckieWorldState` is the canonical branchable
   dynamics state. `RawState` (7-D) is the tabular projection `f_tab`, and
   `ContinuousState` (15-D) is the continuous projection `f_cont`. The POMDP
   is never canonically `MDP{ContinuousState, ...}`.
3. **Reward uses pre-action curvature.** For SAC/TD3, the steering penalty
   uses the *pre-action* curvature `kappa` with the clipped `omega_cmd`.
   `reward(m, s, a, sp)` must never read `sp.kappa`.
4. **Locked transition order** (per physics substep, then per macro-action):

   ```
   previous state
     → DuckController.before_step
     → action → wheel commands
     → frame_skip × delayed DB18 physics ticks
     → raw-state extraction
     → StopTracker
     → collision/termination
     → reward
   ```

   This order is the reference semantics; do not reorder it in the port.

## Repository layout

```
src/
  DuckietownDecisionModels.jl   # module, include order, exports
  config/    config.jl          # typed config structs
             yaml_loader.jl     # load_config / default_config (YAML > Python defaults)
  model/     tabular_state.jl   # RawState, StateConfig, TileType
             continuous_state.jl# ContinuousState, ContinuousStateConfig, DuckRelativeState
             actions.jl         # ActionConfig, MacroAction, DuckieAction, ActionSpec,
                                # build_action_table, vw_to_wheels, action_to_wheels
             discretizer.jl     # STATE_SHAPE/Q_SHAPE bins, digitize, discretize
             encoding.jl        # gate_duck_visibility, build/encode_continuous_state
             state_projection.jl# classify_tile, ego_relative_curve, terminal_lane_fallback
  reward/    events.jl          # EventFlags
             stop_tracker.jl    # StopTracker (sigma, hold, pass-zone), update!
             reward.jl          # RewardConfig, RewardBreakdown, compute_reward
  dynamics/  world_state.jl     # DuckieWorldState, DuckieEgoState, DuckieState,
                                # StopSignState, StopMemory, TileSpec, RoadMap, branch()
             lane_geometry.jl   # bezier_point/tangent, curve_matrix,
                                # curve_signed_curvature (FJ2 parity, FJ3 geometry)
  backends/  abstract_backend.jl# AbstractBackend + locked transition order
  interfaces/policies.jl        # AbstractPolicy
  solvers/   adapters.jl        # QTablePolicy (Q-learning/SARSA), native .npy reader
             actor_adapters.jl  # SACActorPolicy, TD3ActorPolicy — no Python needed
  evaluation/parity.jl          # one-step matched-state comparison (FJ5)
             rollout.jl         # free-running rollout parity, drift split (FJ6)
             metrics.jl         # evaluate_policy / compare_policies (FJ7.6)
  visualization/stubs (FJ9)
  stubs      FJ3–FJ4 module boundaries
test/        runtests.jl, test_configs.jl, test_data_model.jl, test_fj2_parity.jl
tools/parity/ gen_fj2_fixtures.py  # parity fixture generator (runs the real duckduck/src functions)
test/fixtures/ fj2_parity.json     # 6 372 recorded Python outputs for FJ2
```

## Configuration

```julia
using DuckietownDecisionModels

cfg = load_config("../duckduck/policies/q_learning/training_config.yaml")
cfg.reward.full_stop      # 15.0
cfg.solver::QLearningConfig
```

Config hierarchy mirrors Python `Config(**yaml_dict)` semantics: keys absent
from the YAML fall back to the Python source defaults encoded in the config
structs; unknown keys are ignored, exactly like the Python loader. The test
suite pins every field against the four reference `training_config.yaml`
files (q_learning, sarsa, sac, td3) and against the Python defaults.

## Testing

```julia
julia --project="." -e 'using Pkg; Pkg.test()'
```

The config tests read the real YAMLs from the sibling `duckduck` repository
(`../duckduck/policies/*/training_config.yaml` relative to the package root).
The FJ2 parity tests read `test/fixtures/fj2_parity.json`; regenerate it after
touching the ported components with
`python3 tools/parity/gen_fj2_fixtures.py` (WSL Python 3 + numpy 1.26.x).

## Roadmap gates

- `FJ2` — semantic port of deterministically comparable components
  (`discretizer`, `actions.vw_to_wheels`/`action_to_wheels`, `StopTracker`,
  reward, `encoding`, state structures) with cross-language unit parity.
- `FJ3` — native DB18 dynamics: map/geometry, kinematics, duckie crossing
  motion, `before_step`, RNG-stream semantics (`np.random.RandomState` MT19937
  vs Julia RNG).
- `FJ4` — POMDPs.jl `MDP`/`gen` with `DuckieWorldState` as the state type,
  tabular/continuous observation spaces.
- `FJ5` — PythonCall-based reference backend loading the pinned
  `duckietown-gym-daffy==6.1.34` + `duckietown-world-daffy==6.4.3` sources.
- `FJ6` — runtime parity: identical seeds, identical action traces,
  state/reward divergence thresholds.
- `FJ7` — solver adapters reproducing the four reference policies
  (tabulated Q tables, SAC/TD3 checkpoints) under reference configs.
- `FJ8` — online planning: MCTS on the discrete action set, DPW on the
  continuous one, evaluated through the same `evaluate_policy` harness.
- `FJ9` — visualization of rollouts and learned behaviour.
- `FJ10` — POMDP readiness note (what a partially observed variant would need).

Reference artefacts under `duckduck/policies/*/policy.npy` / `policy.pt` are
read-only inputs; they are never overwritten.