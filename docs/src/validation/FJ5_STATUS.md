# FJ5 — Live Reference Integration + Matched-State One-Step Parity

Date: 2026-08-19. Status: **PASSED** (FJ5.1–FJ5.4 complete).

FJ5 stops using recorded fixtures and runs the REAL Python reference stack
next to the native Julia model, from the same latent state.

## FJ5.0 — the PythonCall blocker (measured, not assumed)

The gate was specified as a PythonCall backend. That is not possible on this
workstation, and the evidence is concrete:

| Check | Result |
|---|---|
| Julia build | native **Windows** (`julia 1.10.11`, `x64.w64.mingw32`) |
| Reference env `ddm-ref` python | `ELF 64-bit LSB executable, x86-64` (WSL/Linux) |
| Julia inside WSL | none installed (`which julia` → empty) |
| Windows python + reference stack | `import gym_duckietown` fails (Python 3.13; the stack is pinned to 3.9 and Linux) |

PythonCall loads `libpython` **in-process**; a Windows process cannot load a
Linux ELF shared library. So FJ5.1 is implemented as an **out-of-process
JSON-lines bridge** to a server running inside `ddm-ref`. This delivers the
same capability the gate actually asks for — live side-by-side execution with
matched states — and has two side benefits:

1. `using DuckietownDecisionModels` needs **no Python dependency at all**
   (the client is `open(cmd, "r+")` + JSON3; nothing else), which satisfies
   the "native package must stay pure Julia" constraint even more cleanly
   than PythonCall behind weakdeps would.
2. `ReferenceBackend` is an ordinary `AbstractBackend`, so a PythonCall
   implementation of the same interface can replace it later (e.g. if a Linux
   Julia is installed) without touching a single caller.

## FJ5.1 — reference backend

`tools/parity/reference_server.py` runs inside `ddm-ref` and constructs the
REAL `DuckieMDPEnv` / `ContinuousDuckieMDPEnv` through the REAL factories
(`build_env` / `build_continuous_env`). The reference implementation is never
modified. Protocol: `init`, `reset`, `get_state`, `set_state`, `step`,
`probe_stop`, `ping`, `quit`.

One implementation detail worth recording: the reference stack prints to
stdout on import (pyglet dumps its options dict), which would corrupt a
line protocol. The server therefore dups the real stdout to a private fd used
only for responses and points fd 1 at stderr, so library chatter can never
desynchronise the channel.

Julia side (`src/backends/gym_duckietown.jl`):

```julia
ref = ReferenceBackend("q_learning"; seed = 53)
w, dump = ref_reset!(ref, 53)          # -> DuckieWorldState
ref_set_state!(ref, world)             # inject a Julia state
sp, out = ref_step!(ref, FAST_STRAIGHT)
close(ref)
```

`reference_backend_available()` gates the whole test set, so the suite stays
green on machines without WSL + `ddm-ref`.

## FJ5.2 — the state bridge

`world_to_ref` / `ref_to_world` move the FULL latent state across, not just a
pose: the DB18 `q0`/`v0` matrices, the wheel-axis angles, the trimmed delayed
command window, every duckie's complete object state (pose, walk origin,
heading, velocity, activity, wait/time, corners, SAT normals, mesh extents),
the controller counters, and the stop/lane memory.

Injection maps onto documented reference structures:
`DynamicModel(parameters, (q0, v0), t0, axis_left_rad, axis_right_rad)` and
`DelayedDynamics(state, delay, t0, u0, commands, timestamps)` — whose
constructor re-applies exactly the window trim the simulator applies after
each tick, so injecting an already-trimmed window is a no-op.

Verified: export → inject → export is **bit-identical** on every field
(28 assertions), including after 30 physics ticks of driving, and a
Julia-sampled state the reference has never seen injects cleanly.

## FJ5.3 — matched-state one-step parity

```
                      same latent state x_t, same action a_t
                   +--------------------+--------------------+
          reference runtime                        native Julia
     (ref_set_state! + ref_step!)               (simulate_decision)
                   +--------- compare (field-level, ULP) -----+
```

The comparison covers the full latent successor state, the wheel commands,
the raw-state projection, the 10-component reward breakdown, the continuous
projection (continuous variant), the event flags, and the termination
classification — 80+ quantities per step.

### Result: the dynamical state is bit-identical; deviations are confined to
### one attributed libm-derived chain

**Root cause proven, not guessed.** After a matched-state step the DB18
`q0`/`v0` matrices — the actual dynamical state — are **bit-identical**, as
are `ego.pos`, `ego.speed`, the delay window, the whole duckie state,
`raw.d`, the events and the termination classification. Recomputing
`atan2(q0[2,1], q0[1,1])` in Julia **on the reference's own bit-identical
q0** reproduces the Julia angle exactly, while the reference's stored angle
differs by 1 ULP. The pose readback (`translation_angle_from_SE2`) is the one
place each runtime calls its own libm, so the source is `atan2`
(OpenLibm vs glibc) — the deviation class recorded since FJ2, now confirmed
live and end to end.

**Propagation chain** (worst case over a 40-decision matched-state sweep,
Julia 1.11.3 vs `ddm-ref`):

| Field | ULP | Why |
|---|---|---|
| `ego.angle` | 1 | `atan2` pose readback — the root |
| `lane_fallback[2]` | 2 | `acos(dot(dir(angle), tangent))`, ill-conditioned near alignment |
| `raw.phi` | 2 | the clamped lane angle |
| `reward.heading` | 4 | `-alpha*phi^2` doubles the relative error |
| `reward.total` | 4 | inherits the heading term |
| `reward.progress` | 1 | `alpha*v*cos(phi)` |

Worst absolute difference anywhere: **2.22e-16** — about twelve orders of
magnitude below the quantities involved. Zero discrete mismatches: every
event flag, termination reason, `terminated`/`truncated` split, tile class,
duck threat class, controller counter and stop-memory field agreed on every
compared step.

**Signed zeros (numerically identical, bitwise different).** On turning
actions the se(2) body-velocity diagonal `ego.v0[1,1]` / `ego.v0[2,2]` can be
`0.0` on one side and `-0.0` on the other (the reference builds the matrix
through NumPy products). `0.0 == -0.0`, so this is a zero numerical
difference, and the entries are provably **inert**: the only consumers of
`v0` are `linear_angular_from_se2` (reads `[1,3]`, `[2,3]`, `[2,1]`) and
`SE2_from_se2` (reads `[2,1]`, `[1:2,3]`) — the diagonal is never read. It is
tracked separately as `SIGNED_ZERO_FIELDS` / `bitwise_only_fields` so it is
reported rather than hidden.

Finding this exposed a **bug in the measurement tool itself**: the original
ULP helper computed `abs(reinterpret(Int64,a) - reinterpret(Int64,b))`, which
overflows to `typemin(Int64)` for `0.0` vs `-0.0` and produced a *negative*
"distance" that slipped past a `ulps > 0` filter while still failing
acceptance. `_ulps` now uses the IEEE-754 total order with saturation, and
treats numerically equal values (including signed zeros) as 0 ULP.

**The deviation set is toolchain-dependent.** On Julia 1.10.11 only
`ego.angle` deviated (1 ULP); 1.11.3 propagates the same root a little
further down the chain. The invariant the tests assert is therefore
STRUCTURAL rather than a blanket tolerance:

1. no discrete field may disagree, ever;
2. nothing outside `LIBM_DERIVED_FIELDS` (the named chain above) may deviate
   at all — in particular `q0`, `v0`, `ego.pos`, and the duckie state are
   asserted bit-identical on every action;
3. within that chain, deviations stay under the measured bounds
   `LIBM_MAX_ULPS = 8` and `LIBM_MAX_ABSDIFF = 1e-14` (measured worst: 4 ULP,
   2.22e-16 — 2x ULP headroom for other toolchains, absolute cap far below
   any physically meaningful scale);
4. bitwise-only differences stay inside `SIGNED_ZERO_FIELDS`.

### Test result

`test/test_fj5_reference.jl` on Julia 1.11.3 against the live `ddm-ref`
runtime: **94 assertions, 0 failures** (15 lifecycle + 28 state bridge +
40 discrete matched-state + 11 continuous matched-state), covering all 7
macro actions branched from one state, a 40-decision matched-state
trajectory, a Julia-sampled initial state the reference never produced, the
8 continuous action-space key points (including an out-of-range clip), a
30-decision continuous trajectory, and the explicit atan2 attribution check.

### Julia 1.10.11 Windows GC crash (toolchain defect, recorded)

Running the FJ5.3 comparison under the project's Julia 1.10.11 reproducibly
crashed the runtime with `EXCEPTION_ACCESS_VIOLATION` in `gc_mark_stack`,
raised from the **compiler's inlining pass** — i.e. inside Julia's own
optimizer, not in package code. It reproduced in a standalone script as well
as under `@testset`, and disappeared entirely on Julia 1.11.3.

Two things came out of investigating it:

- `close(::Base.Process)` returns successfully while `process_running` stays
  `true`, leaving an orphaned child holding a live pipe. `close(backend)` now
  closes `proc.in` explicitly (EOF → the server's stdin loop ends) and then
  reaps the process; four create/step/close cycles plus forced GC are clean.
- `parity_accepted` was rewritten from `all(r.diffs) do d ... end` (a closure
  capturing a keyword argument) into a plain loop, since that construct was
  present in every crashing variant.

Both fixes were kept (they are correct regardless), but **the crash
persisted on 1.10.11** — so it is a toolchain defect, not something in the
package. `test_fj5_reference.jl` therefore gates itself on
`VERSION >= v"1.11"` with an explicit `@info`, rather than being silently
weakened: on Julia 1.10 the suite runs FJ1–FJ4 and reports FJ5 as skipped;
on Julia 1.11.3 the whole suite including FJ5 runs.

**Recommendation recorded for the project:** run the live FJ5 parity tests on
Julia ≥ 1.11 on Windows.

## FJ5.4 — stop-sign reachability probe

### Verdict: **A_REACHABLE** — and the earlier FJ3.6 note was WRONG

`tools/parity/run_stop_probe.jl` drives the LIVE reference runtime for
400 decisions on the **unmodified baseline config**, recording every
stop-candidate filter quantity per decision (raw data:
`docs/src/validation/fj54_stop_probe.json`).

| Quantity (baseline, 400 decisions, 1 reset) | Value |
|---|---|
| orientation condition passes | 63 |
| lateral condition passes | 148 |
| ahead condition passes | 221 |
| range condition passes | 221 |
| **all conditions pass simultaneously** | **19** |
| **decisions with `d_stop != None`** | **19** |
| min lateral offset reached | 0.0023 m |
| min geometric distance reached | 0.227 m |

The candidate appears at decisions **381–399**, with `d_stop` decreasing
monotonically as the ego approaches:

```
dec 381  d_stop 0.336  ahead 0.536  lateral 0.073  orient -0.716
dec 383  d_stop 0.301  ahead 0.501  lateral 0.002  orient -0.810
dec 385  d_stop 0.257  ahead 0.457  lateral 0.057  orient -0.883
dec 388  d_stop 0.181  ahead 0.381  lateral 0.125  orient -0.950
```

**Correction to the record.** `docs/src/validation/FJ3_STATUS.md` recorded a "scenario
observation" that the baseline sign may never pass the candidate filter
(`d_stop = None` across a 300-decision rollout). That conclusion was
over-generalized from ONE trajectory: the hits above need a particular
approach geometry that this run first reached after roughly two and a half
laps and one reset. The baseline configuration is **not** broken, and nothing
about it should be "fixed". The FJ3 note is superseded by this measurement.

A control condition with an on-route placement (a separate, explicitly
labelled scenario — the baseline YAML was never edited) also returns
`A_REACHABLE`, with 69 hits instead of 19: the baseline placement is simply
*harder* to encounter, not unreachable.

Consequence for later gates: stop-compliance behaviour (`full_stop`,
`passed_stop`, `stop_violation`) IS exercisable under the baseline config, so
FJ6/FJ7 can evaluate it without inventing a new scenario — though a
longer-horizon or targeted-approach policy is needed to reach it reliably.

## Next gate

**FJ6** — full-episode rollout parity: free-running trajectories (no
per-step re-injection) to measure accumulated drift, event timing, stop
compliance, duck crossing, termination time and reason.
