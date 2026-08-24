# FJ2 — Semantic Parity Port (PYTHON ↔ JULIA)

Date: 2026-08-18. Status: **ready for acceptance** (all tests green).

FJ2 ports the deterministic, dynamics-independent slice of the reference
`duckduck/src` semantics into Julia and pins it with a cross-language parity
suite. The dynamics themselves (physics, map layout, duckie motion, RNG
streams) are FJ3; here we lock the pure functions that sit on top of them.

## Method

`tools/parity/gen_fj2_fixtures.py` imports the **real `duckduck/src` modules**
(`actions.py`, `discretizer.py`, `reward.py`, `state.py`,
`continuous_state.py`) with `gym`/`gym_duckietown` stubbed by `sys.modules`
injection. The stub carries the verbatim `bezier_point`/`bezier_tangent` from
the pinned `duckietown-gym-daffy-6.1.34/src/gym_duckietown/graphics.py`. The
Python functions are executed over generated sweeps and every output is
recorded to `test/fixtures/fj2_parity.json` (6 372 cases).

Julia re-implements each function and `test/test_fj2_parity.jl` compares
outputs **bit-for-bit**:

- Float64 via `reinterpret(UInt64, ...)`;
- Float32 outputs stored as their exact float64 image, re-rounded to Float32
  and compared via `reinterpret(UInt32, ...)`;
- non-finite floats and `-0.0` are stored as `{"nonfinite": "nan"|"inf"|"-inf"|"-0.0"}`
  markers because JSON cannot round-trip them (JSON3 parses `-0.0` as `+0.0`);
- NaN outputs (only `bezier_tangent` of a zero-length segment) are compared
  as NaN — Julia's `NaN` literal has the sign bit set while NumPy produces
  positive NaN; the values are IEEE-identical otherwise.

## Parity table

| Component | Python source | Julia port | Cases | Assertions | Result |
|---|---|---|---|---|---|
| `vw_to_wheels`, `action_to_wheels`, action table | `actions.py` | `src/model/actions.jl` | 162 | 361 | bit-exact |
| `discretize` (7-D tabular index) | `discretizer.py` | `src/model/discretizer.jl` | 2 700 | 2 703 | bit-exact |
| `encode_continuous_state` (15-D observation) | `continuous_state.py` | `src/model/encoding.jl` | 1 810 | 28 963 | bit-exact (Float32) |
| `gate_duck_visibility` | `continuous_state.py` | `src/model/encoding.jl` | 70 | 70 | bit-exact |
| `StopTracker.update` (event semantics, dwell, pass zone) | `reward.py` | `src/reward/stop_tracker.jl` | 16 sequences | 160 | exact |
| `compute_reward` (10-component breakdown) | `reward.py` | `src/reward/reward.jl` | 1 617 | 16 170 | bit-exact |
| `classify_tile` | `state.py` | `src/model/state_projection.jl` | 12 | 12 | exact |
| `bezier_point` / `bezier_tangent` | `gym_duckietown/graphics.py` | `src/dynamics/lane_geometry.jl` | 70 | 210 | bit-exact |
| `curve_signed_curvature` | `continuous_state.py` | `src/dynamics/lane_geometry.jl` | 13 | 13 | bit-exact except `atan2` (1 ULP, below) |
| `terminal_lane_fallback` | `state.py` | `src/model/state_projection.jl` | 20 | 20 | bit-exact |

Suite total: **49 012 assertions, 0 failures** (incl. the 259 FJ1 assertions).

## Semantics locked by this gate

- `digitize(x, bins)` = number of bins `≤ x` (`searchsortedlast`), matching
  `np.digitize`. The `IndexError` guard in the Python `discretize` is
  **unreachable for valid inputs** (max digitize value < each `STATE_SHAPE`
  dimension); both sides keep it defensively, and the fixture asserts the
  Python error list is empty.
- Stop classes: `none → 0`, `> 1.0 → 1`, `≥ 0.3 → 2`, else `3`; tracking
  error is `phi + d`.
- `vw_to_wheels`: float64 arithmetic, **converted to float32, then clipped to
  ±1** — order matters and is preserved.
- `StopTracker`: `passed_stop` fires when the previous stop sign was within
  `pass_distance` and the candidate changed (ids available) or the distance
  jumped `> 0.5` (no ids); a passed or switched sign **resets** the memory;
  dwell requires `near && slow` on **consecutive** steps; `sigma_stop` latches
  at `hold_steps_required` and awards `full_stop` once.
- Reward: exact term order (`progress, lateral, heading, time, pedestrian,
  stagnation, stop_approach, steering, events`) and exact event sum order
  (`collision_duck, other_collision, offroad, stop_violation, full_stop,
  goal`) so float sums are reproducible.
- `classify_tile` lowercases `kind`; `straight`/`3way*`/`4way` → `STRAIGHT`,
  `curve_left`/`curve_right` direct; anything else or non-drivable →
  `ValueError` (Julia `ArgumentError`).
- `curve_signed_curvature`: tangents at `t = 0.05` and `0.95`, cross-y sign,
  `atan2` heading change clamped via `dot`, `samples < 3` raises, `|Δ| ≤
  threshold` → `0.0`, arc length by `range(0, 1; length=samples)` (bit-identical
  to `np.linspace`), `> 1e-9` else `0.0`.
- `-0.0` is a real output (e.g. `-10.0 * 0.0^2`) and round-trips correctly
  through the fixture markers.

## Known deviations (documented, not defects)

1. **`atan2` 1-ULP boundary.** The Windows Julia build links OpenLibm, the
   WSL fixture generator uses glibc; their `atan2` differ in the last bit for
   some arguments (observed: 2/13 curvature cases, ±1 ULP ≈ 6e-17 at
   |κ| ≈ 0.27). Every other intermediate — tangents, cross, dot, arc length,
   division — is bit-identical. The test allows exactly ≤ 1 ULP on the
   curvature value only; runtime parity (FJ6) is unaffected at any practical
   threshold and this is recorded in the FJ0 audit wording (structural
   equivalence).
2. **NaN sign bit.** Julia's `NaN` is negative-NaN by convention; NumPy's is
   positive. Only reachable via `bezier_tangent` on a zero-length segment.

## Exception mapping

| Python | Julia |
|---|---|
| `ValueError` (classify_tile, curvature_samples, non-finite encoding inputs) | `ArgumentError` |
| `IndexError` (discretize guard, unreachable) | `IndexError` |
| `ValueError` (encodings on NaN/Inf inputs) | `ArgumentError` |

## Deliverables

- Ported modules: `src/model/{actions,discretizer,encoding,state_projection}.jl`,
  `src/reward/{stop_tracker,reward}.jl`, `src/dynamics/lane_geometry.jl`.
- Parity harness: `tools/parity/gen_fj2_fixtures.py`; fixture
  `test/fixtures/fj2_parity.json`.
- Tests: `test/test_fj2_parity.jl`; `test/runtests.jl` includes them.
- Dependencies added: `JSON3` (fixture reading), `LinearAlgebra` (norm/dot/
  cross).

## Next gate (FJ3)

Latent world state construction, map/lane geometry end-to-end, DB18 kinematics
+ delayed action application (`frame_skip`), duckie crossing mechanics,
`before_step`, and RNG-stream semantics (`np.random.RandomState` MT19937 vs
Julia `MersenneTwister`; structural parity vs exact-stream parity). Includes
the user-required branch-purity test:
`s_original = deepcopy(s); sp1 = branch(s); step!(sp1, a1); sp2 = branch(s);
step!(sp2, a2)` with `@test s == s_original` and `@test sp1 != sp2`.