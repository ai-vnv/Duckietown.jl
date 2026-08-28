# FJ9.7 — Artifact-Driven Animation

Date: 2026-08-23.

> Animation is playback of recorded evidence, not a simulator run again.

## STATUS

**PASSED.** FJ9.7a–9.7d closed. Three canonical single-solver animations and
one paired animation export as MP4 from a fresh process, with
`ENV_MODEL_CALLS=0`, `PYTHON_MODULES=none` and `MCTS_LOADED=false` measured
rather than asserted by intent.

## ENVIRONMENT

```
WSL Ubuntu-Baru, juliaup override 1.11.3
Pkg.test with ddm-ref + ddm-torch + MCTS.jl : 186 testsets, 148 645 assertions
render check (fresh process, CairoMakie only): RENDER_EXIT=0
                                               ENV_STEPPED=false
                                               ENV_MODEL_CALLS=0
                                               PYTHON_MODULES=none
                                               MCTS_LOADED=false
```

`ENV_MODEL_CALLS` is the strongest of these. The check builds an
`InstrumentedMDP`, renders every frame and every video through it, and prints
the counter. A single `gen` anywhere on the rendering path would make it
non-zero.

## THE RENDERING PATH

```
decisions.csv -> load_decision_log -> animation_sequence -> AnimationSequence
                                                          -> frame_scene
                                                          -> Makie -> MP4
```

The MDP is constructed for one purpose: the **static track description**.
FJ8.0 certified `map` and `stop_signs` as the shared-by-design, never-written
part of a world state, and the render check confirms empirically that two
different reset seeds produce the same `StaticWorld`
(`STATIC_WORLD_SEED_INVARIANT=true`). Nothing episode-specific can enter
through it, and the reference state's ego pose, ducks, stop memory and RNG are
all discarded.

Everything that moves comes from the log. Per-frame geometry is computed from
the logged pose by the same functions the physics uses — `get_agent_corners`
for the footprint, `closest_curve_point` for the lane frame the stop line is
drawn in — applied to recorded values rather than to a re-simulated state.

## FJ9.7a — THE DATA CONTRACT

`AnimationFrame` carries one logged decision: pose, lane projection, stop
subsystem, duck subsystem (lane-relative), action, reward and running return,
the events logged **at that decision**, planning cost, and episode status.
`AnimationSequence` carries the frames, the outcome, the source fingerprint,
and its own fingerprint over the ordered frame identities.

`horizon` is the episode's own length. An episode that terminated at 86
decisions has 86 frames; padding it to 150 would invent stationary frames that
never happened.

### The timeline is the decision index

FJ9.6a found no wall-clock timestamp in the artefact, so this is not
real-time playback and the figure never says it is. The caption carries
`Decision-index playback`, and `framerate` is documented as a display rate
with no claim about elapsed world time.

`model_time` exists, as an exact identity only:

```
model_time(t) = (t - 1) * frame_skip * dt
```

It is labelled **model time, NOT recorded wall-clock time**, and neither
`frame_skip` nor `dt` is in the log, so the caller must supply them from the
configuration the protocol used — there is no default that could silently be
wrong. `planning_time` is never used for pacing; it is computational latency,
and a test asserts it is not reachable from the model-time label.

### What is ABSENT

| Quantity | Why it is not drawn |
|---|---|
| duck world position | the log records `duck_longitudinal` / `duck_lateral` in the **lane frame** only |
| wall-clock timestamp | never recorded |
| observation as the policy saw it | never recorded (FJ9.6a) |
| belief state | no partially observable formulation exists (FJ10) |
| per-decision search tree | FJ9.5 persists trees only for the two captured snapshots |

The duck is the one that would have been easy to fake. Its **reset** pose is
seed-invariant and sitting right there in the reference state, so drawing it
would have cost one line and produced a stationary duck for 150 frames that
looks exactly like data. The world panel omits it; the lane-relative panel
shows what was actually recorded. A test asserts the stated reason names the
lane frame, so a later reader does not "fix" the omission.

## FJ9.7b — SINGLE-EPISODE PLAYBACK

Layout: world view, a text panel of the current decision, and eight history
axes — one per unit, following the FJ9.6 rule (lateral offset m, heading rad,
speed m/s, angular rate rad/s, distance to stop line m, cumulative return,
generative calls, planning time s).

Two invariants carry the gate, because both failure modes look perfectly
convincing on screen:

```
trajectory at frame t = rows 1..t          never the whole episode
event marker at frame t                    only if logged at decision <= t
```

Both are enforced structurally: the panels can only reach the data through
`trajectory_through`, `series_through` and `events_through`, all of which
slice `1:t`. Tests check the prefix property directly
(`series_through(seq, 33, nm) == series_through(seq, 150, nm)[1:33]`) and walk
every frame from 1 to 27 asserting the `full_stop` at decision 28 is invisible.

The camera is fixed to the map and the signs, never to the trajectory. A view
fitted to the episode's full extent would hold the vehicle's future in the
empty space around it, and a view refitted per frame would jitter.

Missing `d_stop` renders as a gap plus a grey tick, never as zero, in both the
history axis and the numeric readout (`MISSING (no candidate)`).

## FJ9.7c — THREE CANONICAL ANIMATIONS

Selected by stated rule, never by inspection. `select_episode` implements the
rules and returns the justification, which is printed by the render check and
written into the figure.

| Animation | Rule | Selected | Justification |
|---|---|---|---|
| `anim_td3_stagnation` | `:first_stagnation` | seed 1001 | lowest seed that performs a full stop and never passes the sign (123 of 150 decisions in the stop zone) |
| `anim_dpw_terminal` | `:median_length_terminating` | seed 1002 | lower-median length among the 19 terminating episodes (79 decisions, other_collision) |
| `anim_mcts_reference` | `:median_return` | seed 1018 | lower-median return over 20 seeds (7.75) |

The median-return rule reproduces the FJ8.4b median column exactly from the
per-decision log — 7.75 for `mcts@1k` and −211.59 for `dpw@1k` — which is a
third independent path to those numbers and is pinned by test.

The TD3 animation is the one this gate exists for. It shows the FJ9.6
correction as behaviour rather than as a table: approach, `full_stop` at
decision 28, `sigma_stop` from there to the end, `d_stop → 0`, speed collapsing
to ~0.005 m/s, and the cumulative return falling linearly for 120 decisions
under the stagnation penalty while the vehicle does not move.

## FJ9.7d — PAIRED PLAYBACK

Same seed, side by side, on the **absolute decision index**. Decision 40 on
the left is decision 40 on the right, and both panels share one x-span so it
is also decision 40 at the same place on the page — the first version gave
each panel its own span, which was numerically correct and visually
misaligned.

When the shorter episode ends, its panel **freezes** on the terminal frame
with a banner (`FROZEN — TERMINATED at decision 116 (offroad)`) and the header
reports both states. It is never looped, restarted, or stretched.

Normalised progress is deliberately not offered for animation. It would align
DPW's decision 40 of 80 with TD3's decision 75 of 150, which are not the same
decision and were not taken under the same conditions. Aggregate plots
(FJ9.6d) may use it; playback may not.

## NEGATIVE CONTROLS

All four run against edited copies of the real log, not synthetic stand-ins.

| # | Edit | Required | Result |
|---|---|---|---|
| 1 | `ego_x` at decision 20 | trajectory changes from frame 20 on, and only there | trajectory identical through 19, differs at 20; `d`, `phi`, reward, cost, events unchanged |
| 2 | move `full_stop` from decision 28 to 29 | marker moves exactly one frame | invisible at 28, visible at 29; every series and every pose unchanged |
| 3 | `model_calls` at decision 40 | computational panel only | all eight other series byte-identical; poses and events unchanged |
| 4 | `d_stop` MISSING → `0.0` at decision 3 | gap becomes a zero | missing count drops by exactly one; all other series unchanged |

Every one also changes the source fingerprint and the animation fingerprint.

## FILES

| File | Purpose |
|---|---|
| `src/visualization/animation.jl` | the whole core: sequence, frames, prefixes, selection rules, static/frame geometry |
| `src/visualization/scene.jl` | `render_frame`, `render_animation`, `render_paired_animation` declarations |
| `ext/DuckietownMakieExt.jl` | one observable layout, driven by `setframe!` |
| `src/interfaces/pomdp_readiness.jl` | `render_animation` added to the extension points (now 9) |
| `test/test_fj97_animation.jl` | 167 assertions across 13 testsets |
| `tools/render_check.jl` | fresh-process export, MP4 with GIF fallback |
| `artifacts/fj9/anim_*.mp4` | three single + one paired |
| `artifacts/fj9/anim_*_frame.png`, `anim_paired_frame130.png` | stills from the same layout |

MP4 is primary; the exporter falls back to GIF and reports which it produced,
so a codec problem degrades the artefact format without touching the
scientific content.

## KNOWN DEVIATIONS

1. **No duck in the world view.** The log has no world-frame duck track.
   Reported ABSENT; the lane-relative pair is shown instead.
2. **The paired layout shows four history axes per side, not eight.** Sixteen
   axes plus two world views is unreadable at video resolution. The single
   animation carries all eight.
3. **The MDP is constructed** to obtain the static track description. It is
   never stepped, and `ENV_MODEL_CALLS=0` is measured every run.

## NEW DEFECTS

Three of my own, all in the renderer, all found by looking at output rather
than at exit codes.

`vlines!` with an empty vector makes Makie reduce over an empty collection
while computing data limits, and a panel legitimately has no events for most
of an episode. Event markers are now line segments spanning the current data
range. The same crash was reachable a second way: a panel whose values are
all `missing` so far never set explicit y-limits and fell through to
autolimits. It now gets a stated range and renders as an empty axis, which is
also the honest picture — "no data yet", not "zero".

The paired figure was never initialised before `record` first laid it out, so
an empty `poly!` observable raised a `BoundsError`; the single-episode path
had been initialising via `setframe!(1)` all along. And the caption rounded
`planning_time` to four decimal places, which printed a tabular policy's
4×10⁻⁵ s as `0.0` — a measured value displayed as if it were absent, which is
the exact confusion FJ9.6 spent a gate preventing. It now rounds to
significant digits.

## ACCEPTANCE

```
[x] decisions.csv sole dynamic evidence source
[x] no solver/environment execution      ENV_MODEL_CALLS=0, measured
[x] one logged decision = one canonical frame
[x] trajectory grows only through current frame
[x] future events never shown early
[x] missing d_stop stays missing
[x] horizon vs environment termination visually distinct
[x] absolute decision indexing is canonical timeline
[x] unequal episode lengths handled without temporal stretching
[x] per-decision planning cost visible   (calls and seconds, separate axes)
[x] TD3 stagnation behaviour faithfully reproducible
[x] DPW compute collapse visible from logged values
[x] PNG/frame rendering works fresh-process
[x] GIF/MP4 headless export works        (MP4 produced; GIF fallback in place)
[x] no Python / no MCTS required
```

## SCIENTIFIC INTERPRETATION

FJ9.6 established that TD3's failure was invisible in the aggregate. FJ9.7
makes it watchable, which is a different and useful thing: the stall is not a
number that needs explaining but a vehicle that stops and stays stopped while
the return counts down behind it.

The paired animation adds something neither gate had. On one seed, from one
initial condition, two planners diverge into two distinct failure modes —
TD3 halts permanently at the sign while DPW drifts off the road at decision
116 — and the pairing that FJ8.4b was built on is what makes that comparison
mean anything. Freezing rather than stretching the shorter panel is what
preserves it.

## NEXT GATE

FJ9.8 — publication composite. The TD3 correction should reach the final
figures.
