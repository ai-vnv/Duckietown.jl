# FJ10 — POMDP Readiness Audit

Date: 2026-08-20.

## STATUS

**PASSED.** This gate is an **audit, not an implementation**. No observation
model, belief representation, belief updater, detector or filter was built,
and none should have been. Executed before FJ9 on purpose: the audit fixes the
extension points that the visualisation layer must be built around.

The one question:

> Can `DuckietownDecisionModels.jl` support a future partially observable
> formulation **without modifying or contaminating the validated MDP core**?

**Answer: yes.** Everything the latent side needs is already there and needs no
change; everything the observation/belief side needs is absent and can be added
as an extension that wraps the validated model rather than altering it. One
existing field is flagged as technical debt.

## ENVIRONMENT

WSL Julia 1.11.3. The audit needs no Python and no solver — it probes the
package's own type and method tables.

## WHAT WAS IMPLEMENTED

`src/interfaces/pomdp_readiness.jl`. The audit is **executable rather than
prose**: every status is decided by probing the live package, so the day
someone adds an observation model the audit changes with it and the tests that
pin it fail until this document is updated. An audit that can go stale silently
is worth very little.

- `ReadinessStatus` — `READY` / `NEEDS_REFACTOR` / `NOT_READY`
- `ReadinessItem` — component, status, **evidence** (the probe), needed change
- `pomdp_readiness(mdp)`, `readiness_table`, `readiness_counts`
- `ObservabilityClass`, `ComponentObservability`,
  `continuous_state_observability()`, `observability_table`
- `VISUALIZATION_EXTENSION_POINTS` — the renderer signatures FJ9 must respect

## THE READINESS MATRIX

```
component                           status            needed
latent world state                  READY             reuse unchanged — already the branchable x_t
transition model                    READY             reuse unchanged
reward                              READY             reuse unchanged — R(x,a) stays latent
terminal semantics                  READY             reuse unchanged
discount                            READY             reuse unchanged
initial state distribution          READY             reuse as the LATENT prior
discrete action space               READY             reuse unchanged
continuous action space             READY             reuse unchanged
observation type                    NOT_READY         define one, explicitly NOT ContinuousState
observation model                   NOT_READY         add O(o | x', a) as an extension
observation randomness              NOT_READY         adopt the transition's caller-supplied-rng contract
belief representation               NOT_READY         define a belief type
belief initialisation               NOT_READY         derive b_0 from the latent prior explicitly
belief updater                      NOT_READY         extension point, explicit updater rng
POMDPs.jl POMDP interface           NOT_READY         a DuckietownPOMDP should WRAP, not replace
legacy controller_rng in the state  NEEDS_REFACTOR    technical debt; do not extend into POMDP layer
```

**8 READY · 1 NEEDS_REFACTOR · 7 NOT_READY.**

Every `READY` row is verified against the live package rather than asserted —
`statetype`, the `gen` signature and its `(sp, r)` keys, that `compute_reward`
has a `RawState` method and **no** `ContinuousState` method, that `isterminal`
and `is_truncated` are distinct, that the continuous action space refuses
`length`. Every `NOT_READY` row is verified absent, which is what makes the
audit self-invalidating when the situation changes.

## THE STATE HIERARCHY THIS GATE EXISTS TO PROTECT

```
DuckieWorldState   latent, privileged world state                    x_t
RawState           tabular projection of the latent state
ContinuousState    15-D PRIVILEGED policy feature vector
Observation        does not exist yet, and is NOT ContinuousState    o_t
Belief             does not exist yet, and is NOT ContinuousState    b_t
```

A partially observable formulation is `x_t → sensor → o_t → update → b_t`.
Calling the 15-D privileged feature vector an "observation" because it happens
to be fifteen numbers a network consumes would silently collapse that chain,
and is the single most likely way this port could go wrong later. So the claim
`Observation ≠ ContinuousState` is not left as a statement of intent — every
component is classified:

```
component                     class                   why
d                             SENSOR_ESTIMABLE        lane detection, with error
phi                           SENSOR_ESTIMABLE        lane detection, with error
v                             SENSOR_ESTIMABLE        wheel encoders
kappa                         MAP_PRIVILEGED          curvature AHEAD, from the map's Bezier curves
stop_present                  MAP_PRIVILEGED          candidate chosen from the map's object list
d_stop                        MAP_PRIVILEGED          metric distance from the sign's true world pose
sigma_stop                    AGENT_MEMORY            the StopTracker's latched flag
duck_present                  SENSOR_ESTIMABLE        a detector could report it
duck_longitudinal             SENSOR_ESTIMABLE        relative position from a detection
duck_lateral                  SENSOR_ESTIMABLE        relative position from a detection
duck_v_longitudinal_relative  TEMPORALLY_DERIVED      needs tracking across frames
duck_v_lateral_relative       TEMPORALLY_DERIVED      needs tracking across frames
duck_active                   SIMULATOR_PRIVILEGED    the duck's internal pedestrian_active flag
duck_crossing_available       SIMULATOR_PRIVILEGED    crossings_started / crossing_armed bookkeeping
stop_hold_progress            AGENT_MEMORY            the tracker's hold counter
```

**6 sensor-estimable · 2 temporally derived · 3 map-privileged · 2
simulator-privileged · 2 agent memory.**

Only 6 of 15 components could come from a sensor at all. Two —
`duck_crossing_available` and `duck_active` — are simulator bookkeeping with
**no physical counterpart**: no sensor, however good, can measure them. Two
more are the agent's own memory and belong in a belief or an agent-state
object, not in an observation. That settles the question quantitatively:
the 15-D vector is a privileged feature projection, and a future observation
type must be built from the six estimable components plus explicit noise, not
by renaming this one.

## THE RNG CONTRACT, FIXED NOW RATHER THAN LATER

FJ3–FJ8 established how much damage a hidden RNG can do, so the contract for
the partially observable layer is set before anything is built:

```
transition randomness     caller-supplied rng           (already enforced)
observation randomness    caller-supplied rng           (to be enforced)
belief-update randomness  explicit updater rng if any   (to be enforced)
```

Tests confirm the transition half today: `gen` has no method that invents an
rng for the caller, and the only `AbstractRNG` field anywhere in
`DuckieWorldState` is `controller_rng`.

**`controller_rng` is recorded as technical debt.** FJ8.0 measured it shared by
every node of a search tree, proved it frozen (never drawn from), and rejected
a defensive copy that would have cost 8.96 % of one `gen` call's allocation for
a field with no semantic role. It is a candidate for removal once parity work
no longer needs it. It must **not** be extended into an observation or belief
object.

## SOLVER READINESS — CAPABILITY MATCHING, NOT MODEL SURGERY

Following FJ8.1: the question is whether a solver's requirements are met by the
model, never how to bend the model until the solver runs.

| Requirement | Met today | Note |
|---|---|---|
| generative transition `(sp, r)` | **yes** | FJ8.0/8.2/8.3 |
| discrete and continuous actions | **yes** | both variants shipped |
| non-enumerable action space handled by widening | **yes** | FJ8.3 |
| terminal / truncation split | **yes** | FJ4 |
| generative `(sp, o, r)` | **no** | needs the observation model |
| `obstype` / observation space | **no** | needs the observation type |
| belief prior and updater | **no** | needs both |
| continuous-observation widening | **no** | follows once observations exist |

POMCP needs the `(sp, o, r)` generative form plus a belief prior; POMCPOW adds
observation widening for a continuous observation space. Both are therefore
blocked on exactly the four `NOT_READY` observation/belief rows and on nothing
else — no change to the state, transition, reward, action bounds or termination
semantics is implied. **These two rows are audited against the documented
POMDPs.jl interface, not executed**: neither package is installed, and
installing a solver to audit it would be the scope creep this gate is meant to
avoid.

## EXTENSION POINTS FJ9 MUST RESPECT

The reason this gate ran first. A renderer whose only entry point takes a
`DuckieWorldState` hardens the assumption that the thing being drawn is the
latent truth — and belief-space visualisation is precisely the case where that
is false.

| Signature | Input | Status |
|---|---|---|
| `render_world` | `DuckieWorldState` | buildable now — the latent truth |
| `render_projection` | `RawState` / `ContinuousState` | buildable now — **label it privileged** |
| `render_policy` | policy + model | buildable now |
| `render_search` | `PlanningDiagnostics.extra` | buildable now — already solver-agnostic |
| `render_observation` | a future observation type | **reserve the signature; do not implement** |
| `render_belief` | a future belief type | **reserve the signature; do not implement** |

A test asserts the two future entry points do not exist yet, so FJ9 cannot
quietly implement them under a different shape.

## TESTS AND RESULTS

`test/test_fj10_readiness.jl` — **156 assertions, 0 failures**.

| Test set | Assertions | What it pins |
|---|---|---|
| the audit is well formed | 14 | 16 items, unique, all with evidence; works on both variants and the wrapper |
| every READY claim is true of the live package | 17 | signatures, `(sp, r)` keys, reward's argument type, terminal split |
| every NOT_READY claim is genuinely absent | 88 | no POMDP subtype, no observation/belief method or type anywhere |
| `ContinuousState` is a privileged projection | 11 | 15/15 fields classified; the counts above |
| the RNG contract holds and is not extended | 8 | no rng-inventing `gen`; `controller_rng` is the only one, and frozen |
| the visualization extension points are recorded | 18 | six points; the two future ones reserved and non-existent |

### Full suite

```
top-level testsets : 128
assertions passed  : 144 546   (144 390 before FJ10, so FJ10 contributes 156)
failed / errored   : 0
Pkg verdict        : tests passed
PKGTEST_EXIT       : 0
```

Both parsers agree exactly. Every earlier gate still passes: FJ10 adds an
introspection module and its tests and changes no package behaviour.

## KNOWN DEVIATIONS

1. **POMCP / POMCPOW rows are documentation-based, not executed.** Neither
   package is installed. Labelled as such above rather than presented as a
   measurement.
2. **The observability classification is a modelling judgement**, not a
   measurement — it says what a sensor *could* produce, which no experiment in
   this repository can settle. The parts that are not judgement (which fields
   exist, that `duck_crossing_available` is computed from `crossings_started`
   and `crossing_armed`) are verified in code.
3. **`controller_rng` remains in the canonical state.** Removing it changes the
   state struct that FJ2–FJ8 validated, so it stays recorded as debt rather
   than being changed inside an audit gate.

## NEW DEFECTS

None. No package behaviour was changed by this gate: it only adds an
introspection module and its tests.

## SCIENTIFIC INTERPRETATION

The MDP core is already the latent half of a POMDP. `DuckieWorldState` is a
privileged world state, the transition consumes caller-supplied noise, and the
reward is a function of the latent transition — which is exactly the structure
`x_t → o_t → b_t` requires underneath it. Nothing in the validated core has to
move.

What is missing is the sensing half, and the audit's most useful output is that
it cannot be improvised from what exists: only 6 of the 15 components of the
privileged feature vector are estimable by any sensor, and 2 have no physical
counterpart at all. A partially observable formulation here is a modelling
project — choosing a sensor, its noise, and a belief representation — not a
type alias over the vector SAC already consumes. Recording that now is what
stops FJ9, or a future gate, from making the cheap substitution.

## NEXT

FJ9 — scientific visualisation, built on the four entry points marked
buildable, with `render_observation` and `render_belief` reserved but
unimplemented.
