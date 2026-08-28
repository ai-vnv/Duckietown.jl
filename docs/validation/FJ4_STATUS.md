# FJ4 — POMDPs.jl Model Interface

Date: 2026-08-19. Status: **PASSED**.

FJ4 turns the FJ3 generative core into a `POMDPs.jl` problem. Scope was kept
deliberately narrow: the model interface only — no solvers, no MCTS.

API pinned against the **installed** POMDPs.jl (v1.0.0), not from memory:
`gen(m, s, a, rng)` returns a `NamedTuple`, `initialstate(m)` returns a
sampleable, plus `isterminal(m, s)`, `discount(m)`, `actions(m)`,
`actionindex(m, a)`.

## Public surface

```julia
using DuckietownDecisionModels, POMDPs, Random

mdp  = DuckietownMDP("../duckduck/policies/q_learning/training_config.yaml")
mdpc = DuckietownMDP("../duckduck/policies/sac/training_config.yaml";
                     action_space = :continuous)

rng = MersenneTwister(53)
s0  = rand(rng, initialstate(mdp))
x   = gen(mdp, s0, FAST_STRAIGHT, rng)     # x.sp, x.r
isterminal(mdp, x.sp)
discount(mdp)                              # solver gamma from the YAML
actions(mdp)                               # 7 MacroActions
actions(mdpc)                              # DuckieActionSpace box
```

Both variants are explicit and share ONE problem definition — same world
state, dynamics, reward, initial-state distribution; only the action
representation differs:

| Variant | Type | For |
|---|---|---|
| `DuckietownMDP{MacroAction}` | `MDP{DuckieWorldState, MacroAction}` | Q-learning, SARSA, MCTS |
| `DuckietownMDP{DuckieAction}` | `MDP{DuckieWorldState, DuckieAction}` | SAC, TD3, DPW |

## Design decisions

1. **`gen` is a thin adapter.** `gen(m, s, a, rng)` is
   `simulate_decision(m.transition, s, a, rng)` reduced to `(sp, r)`. Every
   consumer that needs more (parity, diagnostics, visualization, search-tree
   logging) calls `simulate_decision` directly for the full
   `TransitionResult`. No dynamics/reward/termination logic is duplicated.
2. **`isterminal` is the genuine-terminal half only.** `termination_reason`
   was factored OUT of `_decision_chain` so the chain and `isterminal` share
   one implementation; it is a pure function of the post-transition world
   state (invalid pose / horizon / duckie SAT / full collision / goal tile).
   `isterminal` is `true` for `duck_collision`, `other_collision`, `offroad`,
   `goal` — never for `timeout`, which is horizon truncation, not an
   absorbing physical state (master prompt §17: TD bootstrapping must stay
   intact). `is_truncated(mdp, s)` exposes the horizon separately.
3. **`initialstate` is an implicit distribution.** `rand(rng, d)` runs the
   reference reset structure: pick the start tile (`user_tile_start`, else a
   uniform drivable tile), sample a spawn pose (`Simulator.reset` loop,
   FJ3.1), rebuild the world with the injected duckie and stop sign, extract
   the raw state, and test the wrapper's acceptance predicate
   (`spawn_max_abs_d`, `spawn_max_abs_phi`, `spawn_position_bounds_xz`,
   `spawn_route_direction`/`spawn_min_route_alignment`), up to
   `spawn_attempts` times. Reset also clears the stop/lane memory and records
   the first stop candidate as the previous-decision memory.
4. **Any `AbstractRNG` works, including the FJ3.8 reference streams.**
   `NumpyPCG64` is now `<: Random.AbstractRNG` with `rand(g)` ≡
   `Generator.random()`, so `rand(NumpyPCG64(seed), initialstate(mdp))`
   drives the same sampler through the reference draw stream, while a native
   Julia RNG gives an equally valid `rho_0`. No formulation changes.
5. **Object placement is derived, not hardcoded.** `object_world_pose`
   implements the verbatim tile→world rule
   (`Simulator.interpret_object` → `duckietown_world.get_transform` →
   `weird_from_cartesian`): `x = pos₁·ts`, `y_cart = (W − 1 − pos₂)·ts`,
   `z = grid_height·ts − y_cart`, `angle = −deg2rad(rotate)`. For the square
   `small_loop` this is `(pos₁·ts, 0, (pos₂+1)·ts)`, which reproduces the
   FJ3.1-verified anchors: sign `[1.20, 2.10]` → `(0.702, 1.8135)`, duckie
   `[1.62, 0.50]` → `(0.9477, 0.8775)`.

## Verification (2 028 assertions, no new fixture needed)

Reference evidence is reused from the FJ3 fixtures, so FJ4 is checked against
real reference data rather than against itself:

| Test set | Assertions | What it pins |
|---|---|---|
| model construction + interface | 225 | types, `discount` from YAML, action sets, `actionindex`, continuous box sampling/membership, YAML-over-defaults |
| injected objects vs reference reset | 41 | the derived stop sign equals the FJ3.1 anchor; `initial_duckie` matches `fj3_duck.json::duck_init` field by field (pos/center/start/angle/heading/vel/wait/time/corners/norm) |
| initialstate distribution | 71 | seed reproducibility, fresh-episode invariants, curriculum predicate actually satisfied, reset stop memory, `NumpyPCG64` path |
| `spawn_accepted` vs reference resets | 14 | our acceptance predicate accepts every recorded reference reset pose in `fj37_transition.json` |
| `gen` thin-adapter + branch purity | 64 | `gen` ≡ `simulate_decision` reduced; parent state unmutated; no aliasing; continuous variant |
| isterminal / truncation semantics | 1 610 | state-only reason ≡ chain reason on every replayed decision; `isterminal` ≡ `terminated`; `is_truncated` ≡ `truncated`; explicit timeout-is-not-absorbing invariant |
| end-user rollout | 3 | the package drives like a benchmark MDP (`initialstate` → `actions` → `gen` → `discount`) |

`walk_distance` note: `duck_init` (0.585 = the simulator default) predates
`DuckController.__init__`, which stamps `cfg.walk_distance` (0.90). Both
values are asserted explicitly so the FJ3.5 finding stays pinned.

## Carried forward (unchanged, not "fixed")

FJ4 preserved the stop-sign semantics without touching the baseline placement,
and deferred the reachability question to FJ5. **Resolved there:** the FJ5.4
probe of the live reference runtime returns `A_REACHABLE` — the baseline sign
does become a candidate (19 of 400 decisions). The earlier
"possibly unreachable" note was an artifact of a single trajectory and is
superseded; see `docs/validation/FJ5_STATUS.md` §FJ5.4.

## Next gate

**FJ5** — reference/runtime parity layer: a PythonCall-based reference
backend and matched-initial-state comparison harness, then FJ6 full rollout
parity. `NumpyMT19937`/`NumpyPCG64` are available there for
reference-identical replay.
