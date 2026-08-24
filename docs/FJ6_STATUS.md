# FJ6 — Free-Running Full-Episode Rollout Parity

Date: 2026-08-19. Status: **PASSED**.

> **Objective.** Validate free-running rollout parity between the isolated
> Python/glibc reference runtime and the native Julia model, using the
> in-process PythonCall backend as a secondary semantic oracle to distinguish
> true model divergence from process-level math-library effects.

FJ5 reloaded the reference from the Julia state before every comparison. FJ6
removes that correction: one matched `x0`, then both runs advance on their
own, so any difference is free to feed back into the dynamics.

## ENVIRONMENT

WSL Linux · Julia **1.11.3** (juliaup directory override) · Python **3.9.25**
(`ddm-ref`) · PythonCall **0.9.25** · numpy 1.20.0 · gym 0.23.1 ·
gym_duckietown 6.1.34.

## THREE LANES (roles set by the FJ5-R finding)

```
                    Python source semantics
                             │
             ┌───────────────┴───────────────┐
             ▼                               ▼
   ProcessReferenceBackend           PythonCallRefBackend
   isolated process, glibc           in-process, Julia's libm
             │                               │
     NUMERICAL reference               SEMANTIC oracle
             └───────────────┬───────────────┘
                             ▼
                       Native Julia
```

| Lane | Comparison | Purpose |
|---|---|---|
| A | process vs Julia | true cross-runtime accumulated drift |
| B | pycall vs Julia | semantic/event parity under a near-identical libm |
| C | process vs pycall | libm isolation diagnostic |

## RESULTS — 5 trajectories, 539 free-running decisions

| Case | Decisions | Terminal | Lane A | D_readback | D_dynamic | D_discrete | D_terminal | max Δq0 | max ΔG |
|---|---|---|---|---|---|---|---|---|---|
| discrete_lanefollow | 120 | in_progress | TYPE1 | 4 | — | — | — | 0 | 8.9e-16 |
| continuous_lanefollow | 41 | offroad | TYPE1 | 40 | — | — | — | 0 | 0 |
| discrete_long | 250 | timeout | TYPE1 | 4 | — | — | — | 0 | 1.8e-15 |
| continuous_stop_compliance | 65 | offroad | **TYPE2** | 10 | **65** | — | — | 1.11e-16 | 2.2e-16 |
| continuous_stop_pass | 63 | offroad | TYPE1 | 24 | — | — | — | 0 | 2.2e-16 |

Lane B was `IDENTICAL` (literally zero differences in every compared field) in
4 of 5 cases; in the fifth `max|Δ_CJ| = 5.55e-17` on `angle` alone. Lane C
mirrored Lane A in every case.

**The prediction from the FJ5-R analysis held:**

```
Δ_PJ ≈ Δ_PP        (Lane A ≡ Lane C in all five cases)
Δ_CJ ≈ 0           (Lane B identical in 4/5; ≤ 1 ULP in the 5th)
```

### What did NOT diverge, anywhere

- **No discrete or semantic divergence at any decision** (`D_discrete` is
  `nothing` in all five cases): tile class, duck-threat class, `sigma_stop`,
  `hold_steps`, stop-candidate identity, duck activity, crossing counters,
  step counts, delay-window length, wheel commands and every event flag agreed
  at every decision.
- **No termination divergence** (`D_terminal` `nothing` everywhere): identical
  episode length and identical reason.
- **Event timing difference was 0 for every event that fired**, in every case:
  duck activation and crossing start (decision 41), timeout (250), offroad
  (41/63/65), `d_stop` first visible (28/46), stop-zone entry (28/46),
  `hold_steps` start (49), `sigma_stop` and `full_stop` (51).
- **Cumulative return** agreed to ≤ 3.55e-15 over whole episodes.

## THE ONE TYPE-2 EVENT, FULLY ATTRIBUTED

`continuous_stop_compliance` is the only case where the *dynamical* state
diverged, at the final decision (65) with `max|Δq0| = 1.11e-16`. It was
tracked to its cause rather than assumed:

| Checked | Result |
|---|---|
| wheel commands at that decision | identical |
| `v0` (body velocity) | identical |
| `sin`/`cos` vs **in-process** numpy | identical |
| SE(2) exponential vs `geometry.SE2.group_from_algebra` | identical |
| matrix product vs `np.dot` | identical |
| `cos` vs **standalone/glibc** numpy | **differs, 1 ULP** |

```
w = 0.009677369495122897        (5th physics tick of decision 65)
glibc   cos(w) = 0.9999531746252679
OpenLibm cos(w) = 0.999953174625268
```

So the Type-2 event is **not** accumulated feedback from the readback drift:
it is a fresh, independent 1-ULP libm disagreement occurring *inside* the
dynamics path (the SE(2) exponential), which is why it appears abruptly and
only once. Classification: `LIBM_ACCUMULATION` — no `UNKNOWN` divergence
anywhere in FJ6.

This also explains the lane pattern: in-process numpy resolves `cos` to
Julia's libm, so Lane B stays identical while Lanes A and C show the same
difference.

## TYPE-1 VS TYPE-2 — why the distinction matters

In four of five trajectories `q0`/`v0` — the actual dynamical state — stayed
**bit-identical for the entire episode** (250 free-running decisions in the
longest), while `angle`, `phi` and the phi-derived reward terms differed at
~1e-16 from the first decisions onward. Reporting only "pose angle drift"
would have made the port look less faithful than it is: the physical state had
not diverged at all. Even `ego.pos` was bit-identical throughout, because the
position is read from `q0`'s translation entries directly while the angle goes
through `atan2`.

## STOP AND DUCK BEHAVIOUR EXERCISED (baseline config, unmodified)

- **Stop**: `d_stop` observed on 14–19 decisions, minimum 0.0934 m; stop-zone
  entry, dwell start, `sigma_stop` and `full_stop` all fired — with **zero**
  timing difference between reference and Julia. `passed_stop` and
  `stop_violation` were *not* reached: in both stop trajectories the ego went
  off-road a dozen decisions after completing the stop, so it never drove past
  the sign. Recorded as a coverage gap, not a parity failure.
- **Duck**: activation and crossing start at decision 41 in both discrete
  cases, 7 decisions with the duckie walking, 1 crossing completed — identical
  in both runs.

## NEW DEFECT / DEVIATION FOUND

A second, independent libm source was discovered by the long rollouts and
added to `LIBM_DERIVED_FIELDS`: the quadratic reward terms are written
`state.phi ** 2` / `state.d ** 2`, and CPython's `float.__pow__` is a libm
`pow()` call, not a multiplication:

```
Python  phi ** 2 = 0.02593183609390921    (glibc pow)
Python  phi * phi = 0.025931836093909207
Julia   phi^2     = 0.025931836093909207  (x*x; OpenLibm pow agrees)
```

glibc's `pow` is 1 ULP off the correctly-rounded square here. Reproducing it
would require emulating glibc's `pow`, making the model platform-dependent —
a worse outcome than a 1-ULP reward difference that never touches the state.
`reward.lateral` was added to the derived set for the same reason. Unlike the
`atan2` source, this one is *not* affected by in-process interposition.

## FILES

- `src/evaluation/rollout.jl` — `RolloutRecord`, `rollout_native`,
  `rollout_reference`, `compare_rollouts`, `drift_summary`, `event_timing`,
  `event_timing_diff`, `rollout_table`, `three_lane_table`,
  `libm_hypothesis_check`.
- `tools/parity/run_fj6_rollout.jl` — the five-case experiment.
- `test/test_fj6_rollout.jl` — regression guard (**38 assertions**), including
  a negative control: a deliberately perturbed action sequence must be
  detected as `TYPE2_DYNAMICAL`, so the comparator cannot pass everything.
- `artifacts/fj6/` — `rollout_<case>_{process,pycall,julia}.csv`,
  `drift_summary.json`, `event_timing.json`, `stop_parity.json`,
  `duck_parity.json`.

## EXIT CRITERIA

```
[x] same matched x0 (exported from the reference reset, injected once)
[x] no per-step reinjection
[x] discrete rollout completed          (120 and 250 decisions)
[x] continuous rollout completed        (41, 63, 65 decisions)
[x] drift quantified                    (per-decision, per-field, milestones)
[x] event timing compared               (difference 0 for every fired event)
[x] stop behavior exercised             (through full_stop; passed_stop not reached)
[x] duck behavior exercised             (activation + completed crossing)
[x] termination compared                (identical length and reason, all cases)
[x] cumulative reward compared          (<= 3.55e-15)
[x] every meaningful divergence classified  (all LIBM_*, none UNKNOWN)
[x] full test suite green
```

Full suite in the canonical environment (WSL Julia 1.11.3, `ddm-ref` active):
**82 360 assertions, 0 failures, 0 errors**, `Pkg.test` exit 0 — verified from
the complete log including the final summary line.

## SCIENTIFIC INTERPRETATION

Over 539 free-running decisions the native Julia model and the isolated Python
reference produced the same episodes: same length, same termination reason,
same events at the same decisions, same discrete state at every step, and
returns equal to within 4e-15. The dynamical state stayed bit-identical in
four of five trajectories; the single exception was traced to a 1-ULP `cos`
disagreement between glibc and OpenLibm, not to the port.

This is a stronger statement than one-step parity: it says the two
implementations are the same dynamical system, and that the residual
differences are properties of the platforms' math libraries rather than of the
model. The Type-1/Type-2 split, plus the three-lane design, is what makes that
claim falsifiable — and the negative control in the test set shows the
comparator would have caught a real divergence.

## NEXT GATE

**FJ7** — existing solver baselines (Q-learning, SARSA, SAC, TD3) on the
validated problem. Policy inference must adapt to the model, never the other
way round.
