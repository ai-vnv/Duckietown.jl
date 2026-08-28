# How Duckietown.jl was built — what, why, and how

This document explains how this package was constructed: what each part is,
why it exists, and how it was validated.

**The provenance rule.** Every factual claim in this document traces to one
of five sources, and says which:

1. a **gate document** in `docs/src/validation/FJ*_STATUS.md` (the validation record),
2. a **test** in `test/` (executable, re-runnable),
3. an **artifact** in `artifacts/` (recorded evidence),
4. a **script + recording** in `notebooks/` (the DORA case study), or
5. the **file's own header comment** (each source file states its scope and
   the reference code it ports — the per-file tables below quote those
   headers verbatim).

Anything not traceable to one of these is labelled a *design decision*, with
its reason stated.

---

## 1 · What this is, and why a port instead of a wrapper

The package is a native Julia reimplementation of a Python Duckietown MDP —
a small autonomous-driving decision problem (lane following, a crossing
duck, a stop sign) — exposed through the
[POMDPs.jl](https://github.com/JuliaPOMDP/POMDPs.jl) interface so that the
Julia planning ecosystem can be run against it.

Why a port rather than calling the Python simulator from Julia:

- **Planners need branchable states.** Tree search calls the transition many
  times from the same state with different actions. The Python simulator
  mutates a single global environment; the Julia transition is *branch-pure*
  — `simulate_decision` never mutates its input, so a planner may branch
  freely (stated and enforced in `src/generative/transition.jl`, validated
  in FJ8.0's generative-model contract, `src/evaluation/benchmark.jl`).
- **Speed.** The native backend exists to be "branchable, fast"
  (`src/backends/native_julia.jl` header); planner budget studies
  (`artifacts/fj8/budget_study.md`) are measured in generative calls, which
  a cross-language bridge would dominate with overhead.
- **The Python code stays the ground truth.** The port never replaces the
  reference — it is validated *against* it, live, at three levels
  (section 4).

The reference implementation is the sibling `duckduck` repository
(supplementary material of the study this benchmark reproduces). Its trained
checkpoints (`policy.npy`, `policy.pt`) are **read-only frozen inputs**:
they are loaded, never rewritten (rule stated in `src/solvers/adapters.jl`
header; enforcement: the FJ9.9c artifact ledger classifies them as
provisioned frozen inputs, `src/interfaces/reproducibility.jl`).

---

## 2 · The method: validation gates

The port was built as a sequence of **FJ gates**. A gate is a bounded claim
("the ego dynamics match", "a full episode matches", "the trained policies
match") with:

- a **status document** (`docs/src/validation/FJ<N>_STATUS.md`) recording what was
  measured, what deviated, and what was left undone;
- **tests** that re-run the measurement (`test/test_fj<N>_*.jl`);
- where applicable, **frozen artifacts** of the evidence (`artifacts/`).

A gate that fails is fixed or its failure is documented — it is never
skipped silently. The gate sequence (from the README's build table, each
entry linked to its document):

| Gate | Claim | Record |
|---|---|---|
| FJ1 | package skeleton, typed data model, config hierarchy | [FJ1](validation/FJ1_STATUS.md) |
| FJ2 | semantic parity of the pure functions (actions, discretizer, reward) against fixtures generated from the Python source | [FJ2](validation/FJ2_STATUS.md), `test/test_fj2_parity.jl`, `test/fixtures/fj2_parity.json` |
| FJ3 | native dynamics, observers, transition, exact NumPy RNG | [FJ3](validation/FJ3_STATUS.md), `test/test_fj3_*.jl` |
| FJ4 | the POMDPs.jl model interface | [FJ4](validation/FJ4_STATUS.md), `test/test_fj4_pomdps.jl` |
| FJ5 / FJ5-R | live reference backends (out-of-process and in-process) + matched-state one-step parity | [FJ5](validation/FJ5_STATUS.md), [FJ5-R](validation/FJ5R_STATUS.md), `test/test_fj5_reference.jl` |
| FJ6 | free-running full-episode rollout parity | [FJ6](validation/FJ6_STATUS.md), `artifacts/fj6/` |
| FJ7 | the four trained baselines (Q-learning, SARSA, SAC, TD3), matched layer by layer | [FJ7](validation/FJ7_STATUS.md), `test/test_fj7_*.jl` |
| FJ8 | online planning: solver contract, budget curves, cross-family comparison | [FJ8](validation/FJ8_STATUS.md), [FJ8.4c](validation/FJ84C_STATUS.md), `artifacts/fj8/` |
| FJ9 | scientific visualization through publication figures and reproducibility closure | [FJ9](validation/FJ9_STATUS.md), [9.6](validation/FJ96_STATUS.md), [9.7](validation/FJ97_STATUS.md), [9.8](validation/FJ98_STATUS.md), [9.9](validation/FJ99_STATUS.md) |
| FJ10 | POMDP readiness audit ("an audit, not an implementation" — its own header) | [FJ10](validation/FJ10_STATUS.md), `src/interfaces/pomdp_readiness.jl` |

### Standing rules the gates produced

Each of these exists because breaking it once cost real debugging time; each
is enforced in code, not by convention:

- **A changed world gets a new name, never an edit.** `default_config` means
  "the Python source defaults", permanently. Corrected or extended worlds
  are new named scenarios. Enforced by the testset
  *"named scenarios are new configs, not edits to the defaults"*
  (`test/test_configs.jl`), which pins each scenario to exactly its
  documented deltas.
- **The core must load with nothing but Julia.** `using
  DuckietownDecisionModels` may not require Python, a solver, or a plotting
  library; solver and plotting code lives in package extensions (`ext/`,
  see section 6). Enforced by `test/test_fj8_solver_independence.jl` and the
  source-import audit in `src/interfaces/reproducibility.jl`, which bans
  solver tokens from `src/`.
- **Artifacts are classed, and the class decides what a rebuild may do**
  (`ArtifactStatus` in `src/interfaces/reproducibility.jl`): *rebuilt*
  artifacts are regenerated and compared; *persisted-source* artifacts are
  recorded evidence that can only be checked, never regenerated;
  *provisioned frozen inputs* (the trained checkpoints) are never written.
- **Stale claims are banned by string.** The documentation audit
  (`src/interfaces/reproducibility.jl`, run by
  `test/test_fj99_closure.jl`) scans the repository for claims that were
  once true and later falsified — for example, the claim that the car can
  never encounter the stop sign, which stopped being true when the scenario
  layer landed — so a fixed bug cannot survive as prose.
- **Never judge a suite from truncated output.** The suite writes a
  structured report (`artifacts/fj9/test_report.json`) built from the
  `Test` result tree; two independent summarizers
  (`tools/suite_summary.sh`, `tools/suite_summary.py`) must agree with it.
  This rule exists because the reporter's very first run *disagreed* with
  the summarizers — the reporter had a counting bug (finished testsets
  discard `Pass` objects and keep only `n_passed`; `test/reporter.jl` now
  reads counts at every level). The README records this incident.
- **`missing` stays `missing` all the way to the renderer** — never coerced
  to zero (FJ9 visualization contract, `src/visualization/scene.jl`).

---

## 3 · The source tree, file by file

Descriptions below are the files' **own header lines** (quoted or lightly
compressed), so this table cannot drift from the code without the code's
headers drifting too.

### `src/dynamics/` — the world and its physics (FJ3)

| File | Header says |
|---|---|
| `world_state.jl` | the typed state: `DuckieEgoState`, duckies, stop signs, map, stop memory, controller RNG |
| `ego.jl` | "FJ3.2: DB18 nominal motor model + 0.15 s delayed dynamics" |
| `map_loading.jl` | "FJ3.1: map reconstruction — tiles, lane curves, world↔tile coordinates" — verbatim port of the simulator's `_get_curve` / `_interpret_map` |
| `lane_geometry.jl` | lane frames and closest-point queries (FJ3, used by the observers) |
| `pedestrian.jl` | "FJ3.4: DuckieObj walk/finish\_walk/step parity (gym\_duckietown 6.1.34)" |
| `collision.jl` | "FJ3.1: collision geometry and validity checks" |

### `src/rng/` — exact randomness (FJ3.8)

| File | Header says |
|---|---|
| `numpy_rng.jl` | "exact NumPy RNG stream reproduction (RNG-C compatibility layer)" — so a Julia reset draws the *same* spawn as the Python reset |
| `ziggurat_constants.jl` | "Ziggurat tables for random\_standard\_normal, generated verbatim" from the reference |

*Why this exists at all:* episode-level parity (FJ6) is impossible unless
both sides consume identical random streams; approximating the RNG would
turn every parity mismatch into an unanswerable question.

### `src/model/` — what the agent sees (FJ2, FJ3.6)

| File | Header says |
|---|---|
| `observers.jl` | "FJ3.6: env-dependent state extraction" — `get_raw_state`, duck classification, stop-sign candidate geometry |
| `discretizer.jl` | the tabular bins (`D_BINS`, …): the 7-tuple discretisation the Q-learning baseline uses |
| `continuous_state.jl`, `encoding.jl` | the 15-dimensional continuous observation and its encoder |
| `tabular_state.jl`, `state_projection.jl`, `actions.jl` | tile classification, state projection, the action table |

### `src/reward/` — how behaviour is scored (FJ2)

| File | Header says |
|---|---|
| `reward.jl` | `RewardConfig` + the shaped reward breakdown (progress, lateral, heading, pedestrian, stop-approach, events) |
| `stop_tracker.jl` | `StopTracker` — the stop-sign memory: dwell, `sigma_stop`, pass/violation events |
| `events.jl` | `EventFlags`, ported from the reference `reward.py` |

### `src/generative/` — the transition (FJ3.7)

| File | Header says |
|---|---|
| `transition.jl` | "the full one-decision generative transition, verbatim" — frame-skip physics, duck steps, stop tracking, reward; branch-pure |
| `initial_state.jl` | "spawn-pose sampling (raw simulator reset)" + the task-map injection semantics of the reference duck controller |

### `src/backends/` — the reference, live (FJ5)

| File | Header says |
|---|---|
| `abstract_backend.jl` | the backend contract |
| `native_julia.jl` | "full generative backend over DuckieWorldState (branchable, fast)" |
| `gym_duckietown.jl` | "FJ5.1 / FJ5-R: the Python reference backend clients" — including `world_to_ref`, the state encoder used for parity and for the case-study renderers |
| `torch_policy.jl` | "FJ7.4a / FJ7.5a: client for the PyTorch policy oracle" |

### `src/solvers/` — the frozen baselines (FJ7)

| File | Header says |
|---|---|
| `adapters.jl` | "inference adapters for the shipped tabular policies" — a native `.npy` reader + `QTablePolicy`; checkpoints read-only |
| `actor_adapters.jl` | "native Julia inference for the SAC and TD3 reference actors" — the forward passes reproduced layer by layer, Float32 like PyTorch, with the SAC/TD3 clipping asymmetry preserved, not smoothed over (its header states this) |

*Why native inference:* running a learned policy must not require Python
(the dependency rule of section 2). The weight exports live at
`artifacts/fj9/weights/` (produced by `tools/export_actor_weights.jl`).

### `src/evaluation/` — measurement without solver bias (FJ5–FJ8)

| File | Header says |
|---|---|
| `parity.jl` | "FJ5.3: live matched-state one-step parity" |
| `rollout.jl` | "FJ6: free-running rollout harness and drift analysis" |
| `metrics.jl` | "FJ7.6: one solver-independent evaluation harness" |
| `benchmark.jl` | "FJ8.0 — the generative-model contract and its cost, measured before any" solver was connected |
| `budget.jl` | "FJ8.4a — the cost–search curve, measured solver-agnostically" |
| `comparison.jl` | "FJ8.4b — cross-family solver comparison" |

### `src/interfaces/` — the contracts

| File | Header says |
|---|---|
| `pomdps.jl` | "FJ4: the POMDPs.jl model interface" |
| `rl_environment.jl` | "FJ4: step/reset info-style interface mirroring DuckieMDPEnv semantics" |
| `planning.jl` | "FJ8.1 — the solver-facing contract, written without reference to any solver" |
| `policies.jl` | the `AbstractPolicy` / `act` contract |
| `reproducibility.jl` | "FJ9.9 — reproducibility closure": artifact ledger, stale-claim audit, source-import ban, core fingerprint |
| `pomdp_readiness.jl` | "FJ10 — POMDP readiness audit. AN AUDIT, NOT AN IMPLEMENTATION." |

### `src/visualization/` — figures that cannot lie quietly (FJ9)

| File | Header says |
|---|---|
| `scene.jl` | "FJ9.0 — the visualisation contract. Backend-free, by construction." |
| `policy_slice.jl` | "policy, value and ambiguity slices, computed in the core" |
| `rollout.jl` | "rollout comparison driven by the FROZEN FJ8.4b artefacts" |
| `search_tree.jl` | "the solver-neutral search representation" |
| `diagnostics.jl` | "FJ9.6 — diagnostic time series from the frozen FJ8.4c decision log" |
| `animation.jl` | "FJ9.7 — artifact-driven animation" — playback of recorded evidence, never a simulator re-run |
| `paper_figure.jl` | "FJ9.8 — publication composites" — figures whose captions are checked against rules, because a figure must not merely reproduce correct numbers; its caption must reproduce the correct *interpretation* |
| `world_view.jl` | "FJ8: Makie top-down world render + state panel" |

### `src/config/`

| File | Header says |
|---|---|
| `config.jl` | the typed config hierarchy (environment, state, actions, duck controller, reward, …) mirroring the reference constructors |
| `yaml_loader.jl` | `load_config(path)` for the frozen training configs, plus the named scenarios (`SCENARIOS`, `scenario_config`) |

### `ext/` — optional capability, isolated

`DuckietownMCTSExt.jl`, `DuckietownMakieExt.jl`,
`DuckietownPythonCallExt.jl` — package extensions
(`Project.toml [extensions]`) that activate only when the user loads MCTS,
Makie, or PythonCall. *Why:* the dependency rule of section 2; the core
package must never pull a solver, a plotting stack, or Python into a plain
`using`.

---

## 4 · How "valid" is established: the three-layer parity architecture

1. **Fixture parity (FJ2/FJ3).** Pure functions are compared against JSON
   fixtures generated from the Python source
   (`test/fixtures/fj2_parity.json`, `fj3_*.json`, `fj37_transition.json`,
   `fj38_rng.json`).
2. **Live one-step parity (FJ5).** A running Python reference is driven to a
   state, the state is imported/exported through `world_to_ref`, both sides
   take the same action, successors are compared
   (`src/evaluation/parity.jl`, `tools/parity/reference_server.py`).
3. **Free-running episode parity (FJ6).** Both sides run whole episodes from
   the same seed and the trajectories are compared step by step
   (`artifacts/fj6/rollout_*.csv`, drift summarised in
   `artifacts/fj6/drift_summary.json`).

The suite runs all of it when the reference environment is present, and
**degrades honestly** when it is not: `test/reference_guard.jl` skips the 26
reference-dependent test files *by name and says so* (README, section
"Testing"). Measured counts (README): 201 test sets / 148 855 assertions
with the reference, 17 / 167 without. The authoritative count is always
`artifacts/fj9/test_report.json`.

---

## 5 · The scenario layer: making the task reachable without touching the defaults

`default_config` reproduces the Python defaults — including their inert
switches: the duck crosses with probability 0.02 and no stop sign is
injected, so the stop-and-duck machinery is effectively unreachable in
casual use. Rather than edit the defaults (banned, section 2), named
scenarios were added (`scenario_config` in `src/config/yaml_loader.jl`):

- `:lane_following` — the defaults, unchanged (test-pinned field by field).
- `:stop_and_duck` — the switches on (`p_cross = 1.0`, stop sign injected),
  plus spawn-sanity bounds; scoring untouched (test-pinned:
  `same(sc.reward, d.reward)`).
- `:stop_and_duck_safe` — the same world with the safety shaping switched on
  and the stop sign made *reachable*. Both changes are measured, not
  aesthetic:
  - With the source-default weights, the pedestrian term is zero whether
    the car brakes for a crossing duck or drives through it — and braking
    scores strictly worse (measured in `notebooks/audit_reward.jl` before
    the scenario existed; the comment block in `yaml_loader.jl` records the
    numbers). The weights used are the project's own SAC/TD3 values.
  - The source places the sign at rotate 180° "facing vehicles travelling
    east" — but the route travels **west** there, so traffic sees the back
    of the board and the detector rejects it on every frame; even turned
    around, the corner placement leaves the stop line geometrically
    unreachable (detection dies at d\_stop = 0.395 m), and a mid-straight
    placement gives less than one decision of detection runway. All three
    facts were measured (`notebooks/audit_sign_orientation.jl`,
    `notebooks/audit_stop_geometry.jl`) before the final placement was
    chosen. `:stop_and_duck` keeps the source's own placement, warts
    included.
  - The testset in `test/test_configs.jl` pins the safe scenario to
    **exactly** these deltas — four reward weights and the sign pose — and
    nothing else.

---

## 6 · The DORA case study: what the finished model is for

With the model validated, an external solver
([DORASolvers.jl](https://github.com/ai-vnv/DORASolvers.jl), an online
tabular-SSP solver) was brought to it. Every design step is a measured fact
with its script in `notebooks/`:

| Fact | Where measured |
|---|---|
| the decision step is deterministic (identical successors under different RNGs) — so the *known transition kernel* DORA requires exists exactly, no estimation | re-measured live in `notebooks/DORA_on_Duckietown.jl` (step 2); used by `dora_lap_safe.jl` |
| 0.2 s actions collapse the tabularisation to a single orbit; holding a command for 8 decisions makes the graph branch (K = 8/6/4 solve, K = 2 collapses to 22 states) | `notebooks/probe_zigzag_k.jl` |
| coarse keys alias: states 0.19 m / 15° apart share a key and diverge under the same action; a single open-loop plan crashes | `notebooks/audit_divergence.jl` |
| the fix is receding horizon: re-plan from the true state before every macro action, making the chosen action's prediction exact by construction | `dora_lap_safe.jl`, `dora_zigzag.jl` |
| results: `small_loop` lap in 102 decisions / 13 plans (cost 131.39 vs plan 133.76) with a duck yield to v = 0.000 and a full stop at the sign (FULL\_STOP, then PASSED\_STOP, no violation); `zigzag_dists` 26/26 tiles in 341 decisions / 43 plans | recordings `notebooks/lap_states_safe.json`, `notebooks/zigzag_lap.json` |
| the zigzag map loaded by the package's own parser is bit-identical to the reference simulator's on all 26 tiles | `notebooks/check_zigzag_curves.py` |
| every rendered frame is a physics substep replayed with the package's own primitives and asserted bit-identical to `simulate_decision` | assertions in `dora_lap_safe.jl` / `dora_zigzag.jl`; rendering via the *real* gym-duckietown renderer (`render_lap_safe.py`, `render_zigzag.py`) |

Two Pluto notebooks make this runnable by anyone:
`notebooks/DORA_on_Duckietown.jl` (the formulation, executed cell by cell,
plus tile-by-tile replays) and `notebooks/Playground.jl` (pick a scenario ×
a driver: random, MCTS, DORA, or the four frozen baselines).

---

## 7 · How to verify any claim in this document

```bash
# the whole validation suite, complete log, real exit code
bash tools/run_full_suite.sh

# the authoritative result
cat artifacts/fj9/test_report.json

# any single gate
julia --project=. -e 'using Pkg; Pkg.test()'   # honest subset without the reference

# the case study, live
julia -e 'using Pluto; Pluto.run()'            # open notebooks/DORA_on_Duckietown.jl
```

Known limitations are not in this document by design — they are recorded
where the audit checks them: `artifacts/fj9/reproducibility_manifest.json`
and the README's *Known limitations* section.
