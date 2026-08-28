# FJ9 — Scientific Visualization

Date: 2026-08-21.

## STATUS

| Sub-gate | Scope | Status |
|---|---|---|
| FJ9.0 | Visualization contract + backend isolation | **PASSED** |
| FJ9.1 | World renderer | **PASSED** |
| FJ9.2 | Privileged projection panel | **PASSED** |
| FJ9.3 | Policy / value / ambiguity slices | **PASSED** |
| FJ9.4 | Rollout comparison from frozen artefacts | **PASSED (aggregate views); trajectory views BLOCKED — see below)** |
| FJ9.5a | Search-artifact availability audit | **PASSED** — verdict: nothing that determines a tree is stored |
| FJ9.5b | Search capture (MCTS + DPW) | **PASSED** |
| FJ9.5c | Generic search rendering | **PASSED** |
| FJ9.5d | DPW continuous action plane | **PASSED** |
| FJ9.6 | Diagnostic time series | Not started |
| FJ9.7 | Animation | Not started |
| FJ9.8 | Publication figure | Not started |
| FJ9.9 | Reproducibility / headless export | Partly done — the headless check runs and passes |

`render_observation` and `render_belief` remain **unimplemented**, as FJ10
reserved them. A test asserts it.

## ENVIRONMENT

WSL Julia 1.11.3. The core needs no plotting library; the extension was
validated against **Makie 0.24.13** via **CairoMakie 0.15.13**, installed only
in `experiments/`.

## FJ9.0 — THE CONTRACT

The organising decision: **all geometry is computed in the core, and the
backend only draws it.**

```
src/visualization/scene.jl      world_scene(mdp, state) -> WorldScene
src/visualization/search_tree.jl SearchSnapshot (solver-neutral)
        │  finished coordinates
        ▼
ext/DuckietownMakieExt.jl       puts ink on them
```

Three consequences, each of them the point rather than a side effect:

1. **The geometry is testable with no backend installed.** FJ9's 3 246 assertions
   run in the ordinary suite, where Makie is absent.
2. **No plotting package is a dependency.** Makie joins PythonCall and MCTS as
   a weak dependency — the third time the same rule has applied: a library
   needed to *look at* the model must not be required to *use* it.
3. **The renderer never reads the Python simulator.** Every coordinate comes
   from `DuckieWorldState`, the `RoadMap` and the model's own projections. The
   headless check confirms no Python module is loaded while rendering.

`SearchSnapshot` is declared now, in FJ9.0, so FJ9.5 cannot be tempted to make
`render_search` import a planning library. A solver extension will convert its
own tree into the snapshot; the renderer only ever sees the snapshot — the same
`solver-specific extraction → generic representation → generic renderer`
pattern FJ8 established.

## FJ9.1 — THE WORLD RENDERER

`render_world(mdp, state; trajectory, ...)` draws tiles, lane centrelines, the
ego footprint with heading and velocity arrows, stop signs and their stop
lines, duckies with their own corner polygons, and an optional ground track.

The tests are **geometric, not pixel-based**. "Is this pixel #4C72B0" is
brittle and proves nothing; what is asserted is that the drawn objects *are*
the model's objects:

- `ego_position === (s.ego.pos[1], s.ego.pos[3])`, `ego_angle === s.ego.angle`,
  heading `=== get_dir_vec(angle)` projected and unit-norm;
- the ego polygon is `get_agent_corners` row for row — the **true collision
  polygon**, not a decorative box;
- lane centrelines start and end exactly on `bezier_point(curve, 0)` and
  `bezier_point(curve, 1)` of the tile's own curves — the same curves the
  observer measures `d` and `phi` against;
- tile corners land exactly on `(i·ts, j·ts)`;
- duck markers and polygons are the latent duck positions and `obj_corners`;
- extracting a scene leaves the state bitwise unchanged.

A deliberate non-equality is asserted too: the footprint centroid is **not**
the ego pose, because `get_agent_corners` builds the box around
`_actual_center`. Testing for equality there would have forced a wrong
renderer.

### The defect this gate was designed to catch, and did

The first implementation drew the stop line as `sign_to_line_offset` ahead of
the sign **along the sign's own facing**. It produced a plausible red line in a
plausible place. It corresponded to nothing the model computes.

The model has **no stop-line object at all**. `next_stop_candidate` computes

```
rel      = sign.pos - ego.pos
ahead    = dot(rel, forward)              # forward = the EGO's lane frame
d_stop   = max(0, ahead - sign_to_line_offset)
```

so the stop line is the locus of points at along-track offset
`sign_to_line_offset` before the sign, measured along the **ego's** direction
of travel, and its width is the model's own acceptance gate
`stop_lateral_limit` (a sign only counts while
`|dot(rel, right)| ≤ stop_lateral_limit`).

The corrected function is now pinned by an identity test rather than by
inspection: for every state where the observer reports a stop candidate, the
along-track distance from the ego to the **drawn** line reproduces `d_stop`
exactly (`atol = 1e-9`). Those states cannot be spawn poses — a sign is only
ever an accepted candidate mid-episode — so the test drives with the shipped
Q-learning policy to find them, and asserts it found some. Verified on **12
(state, sign) pairs**.

### small_loop's stop sign is outside the tile grid

Rendering surfaced this immediately: the injected stop sign sits at
z = 1.8135 while the 3×3 grid ends at 1.755. That is the reference's own
placement — `object_world_pose` maps tile position `[1.20, 2.10]` to
`(0.702, 1.8135)`, validated in FJ3 map loading and exercised live in FJ5/FJ6 —
so the fix was to widen the view, not to move anything:

```
map extent   (0.0,    1.755,  0.0,    1.755)
view extent  (-0.029, 1.784, -0.029,  2.279)
```

A view clipped to the map would have hidden a real feature of the scenario.

## FJ9.2 — THE PROJECTION PANEL

Built on the same rule as FJ9.1: **the core decides the semantics, the backend
only lays them out.** `projection_scene(raw, cont)` returns a
`ProjectionScene` of `ProjectionEntry` values, each carrying display order,
source field, raw value, rendered string, unit, subsystem category and FJ10
privilege class. The Makie method receives that and does nothing but place
text.

The division matters because the FJ10 classification —

```
sensor-estimable · temporally-derived · map-privileged
simulator-privileged · agent-memory
```

— is **package semantics, not visual style**. A backend that decided labels,
units or privilege classes could quietly disagree with the audit.

Category and privilege are deliberately orthogonal: `sigma_stop` is part of the
`STOP_SUBSYSTEM` *and* is `AGENT_MEMORY`.

### Row k is component k of the policy input

Entries follow `fieldnames(ContinuousState)`, which is the order
`encode_continuous_state` produces, so the panel's row *k* is component *k* of
the vector SAC and TD3 actually consume. Values are read straight off the
`ContinuousState` object — asserted with `===`, not recomputed — so the panel
cannot drift from what the policy saw.

The panel draws a rule at each category transition. Because the row order is
the *encoded vector's* order rather than a tidy grouping, `stop_hold_progress`
appears after the duck block and gets a rule of its own. Preserving
"row k = component k" is worth more than a neater layout, and the tests pin the
ordering rather than the appearance.

Tabular-projection facts the 15-D vector does not carry (`tile`, the tabular
duck class) are shown as **context**, visually separated, so they are never
mistaken for policy inputs.

### Acceptance

All eight requirements are asserted in
`FJ9.2 the panel's semantics come from the core, not the backend` (85
assertions): every component exactly once in deterministic order; values
identical (`===`) to the `ContinuousState` the encoder consumes; labels, units
and classes from core metadata; the title contains "Privileged" and not
"observation"; `render_observation`/`render_belief` still absent; a different
state produces a different panel while leaving the world model bitwise
unchanged; and the headless PNG/SVG export still runs with no Python and no
solver.

## FJ9.9 — HEADLESS EXPORT (partial)

`tools/run_render_check.sh` renders in a fresh process and reports:

```
EXT_LOADED=true
RENDER_WORLD_METHODS=1
WORLD_PNG=true            artifacts/fj9/world.png
WORLD_SVG=true            artifacts/fj9/world.svg
PROJECTION_PNG=true       artifacts/fj9/projection.png
PYTHON_MODULES=none
MCTS_LOADED=false
RESERVED_STILL_RESERVED=true
RENDER_EXIT=0
```

Both raster and vector output, on the CPU, with no display, no Python and no
planning library.

## FILES

| File | Purpose |
|---|---|
| `src/visualization/scene.jl` | `WorldScene`, `world_scene`, tile/lane/stop-line geometry, `ProjectionScene` |
| `src/visualization/search_tree.jl` | `SearchSnapshot` and its validator |
| `ext/DuckietownMakieExt.jl` | `render_world`, `render_projection` |
| `test/test_fj9_visualization.jl` | 193 assertions, no backend required |
| `tools/fj9_render_check.jl`, `tools/run_render_check.sh` | headless export |

## TESTS AND RESULTS

`test/test_fj9_visualization.jl` — **3 246 assertions, 0 failures**, with no
plotting backend installed.

| Test set | Assertions | What it pins |
|---|---|---|
| the contract exists and needs no backend | 16 | four entry points declared, zero methods, `MethodError` on call; FJ10 reservations intact; no plotting package in `[deps]` |
| the drawn world IS the model's geometry | 65 | ego, footprint, tiles, lane curves, ducks, purity |
| the stop line is the model's measurement | 49 | centre, perpendicularity, width, and the `d_stop` identity on 12 pairs |
| the view shows what lies outside the tile grid | 12 | the out-of-grid stop sign is inside the view |
| trajectory overlay | 5 | ground track matches the state sequence |
| the panel's semantics come from the core | 85 | the eight FJ9.2 acceptance requirements |
| the projection panel is labelled privileged | 34 | title wording; every component carries its FJ10 class |
| the search snapshot is solver-neutral and validated | 12 | discrete and continuous actions in one type; malformed snapshots rejected |
| FJ9.3a the slice contract is core data | 14 | plain arrays, no backend, typed slices |
| FJ9.3b the action comes from `decide` | 1 125 | every cell equals the validated decision |
| FJ9.3b near-tie is not `argmax` | 7 | synthetic tied table |
| FJ9.3c value and ambiguity are separate | 1 178 | tied cells have zero margin; both layers present |
| FJ9.3e coordinates, mode, fixed context | 16 | collapse recorded, caveat present, fingerprint identity |
| FJ9.3d continuous slices | 19 | two surfaces, bounds, actor pipeline, TD3 reuse |
| FJ9.3e the fixed context is not neutral | 8 | the TD3 measurement above |

### Full suite

```
top-level testsets : 158
assertions passed  : 147 805
failed / errored   : 0
Pkg verdict        : tests passed
PKGTEST_EXIT       : 0
```

Both parsers agree exactly. Two tooling problems surfaced on the way there and
are recorded under NEW DEFECTS.

## KNOWN DEVIATIONS

1. **Only Makie 0.24 is claimed.** The compat bound was narrowed from a
   speculative range to `"0.24"` after FJ9.1 moved to `arrows2d!`, which
   replaced the deprecated `arrows` recipe. A wider range would be a claim
   about versions this repository has never rendered with.
2. **The extension is not exercised by `Pkg.test`.** Installing Makie into the
   test target would add a large dependency and a long precompile to every
   suite run. The geometry — the substantive part — is tested there; the
   drawing is verified by the headless check in `experiments/`.
3. **The velocity arrow is scaled** (×1.5 by default). At 0.14 m/s a
   true-to-scale arrow is invisible on a 1.755 m map. The scale is printed in
   the axis subtitle so the figure never implies a length it does not have.

## NEW DEFECTS

1. **The stop line was drawn along the wrong axis** (this gate's own new code),
   described above. Found by asking whether the drawn geometry was the model's
   geometry, fixed, and now pinned by an identity test against `d_stop` rather
   than by inspection.

2. **The FJ10 boundary test fired — as designed.** FJ10 asserted that none of
   the six visualisation entry points existed. FJ9.0 declared the four that
   FJ10 itself marked "buildable now", so the assertion failed on the first
   full run after this gate. That is the executable audit doing its job: it
   was written to fail when the situation changes rather than to go quietly
   stale. The assertion now tracks each point's *status* — reserved points must
   not exist, buildable ones may — instead of a snapshot of one day's contents.

3. **Both suite parsers dropped rows again, twice over.** A test set whose
   elapsed time was **negative** (`-0.5s`) was silently excluded, and so was
   every *failing* test set, whose row carries three counts (`Pass Fail
   Total`) instead of two. Together they hid 43 assertions and made the two
   parsers disagree — which is how they were caught. Both now accept a leading
   minus and 2–5 count columns. This is the third variation of the same defect
   class recorded in the README correction; the lesson is holding that a
   summary parser must fail loudly rather than skip a line it does not
   recognise.

   The negative durations come from the WSL clock stepping backwards mid-run.
   They are cosmetic: Julia's `@testset` timer reads the wall clock, while
   every measurement in this repository (`measure`, `plan_action`,
   `budget_study`) uses monotonic `time_ns()`, so the FJ8.0/8.4a latency
   figures are unaffected.

## SCIENTIFIC INTERPRETATION

The stop-line defect is the argument for this gate's structure. A renderer that
invents geometry produces figures that look right and are wrong, and a
screenshot test would have accepted it forever — the line was in a plausible
place, the right colour, the right length. What caught it was the requirement
that every drawn coordinate be traceable to something the model computes, and
what fixed it is now an identity test rather than a comment.

That is also why the geometry lives in the core rather than in the backend.
A figure in a paper is a claim about the model; keeping the claim in tested
code and the ink in an extension is what makes the claim checkable.

## FJ9.3 — POLICY, VALUE AND AMBIGUITY SLICES

Same rule again: `policy_slice(...)` returns a `PolicySlice` of finished
semantics and the backend never runs a policy. Every decision in a slice comes
from the validated adapters, so a figure cannot disagree with the evaluator.

### The tabular action is the validated one

Each cell's `RawState` goes through the **real discretizer** and then through
`decide` — FJ7's near-tie rule (`|Q − max| ≤ 1e-12`, lowest action id) — never
through a re-implemented `argmax`. Every cell is asserted equal to `decide`'s
own answer, field by field.

That test alone would pass for the wrong reason, because FJ7 measured
`argmax_differs = 0` on the shipped checkpoints: the two rules happen to agree
everywhere. So a **synthetic** table is also used, with action 3 as the raw
first maximum and action 1 tied with it. The reference rule must return 1, and
the slice must return what the reference rule returns.

### Value and ambiguity are separate layers

```
value_surface   V(s) = max_a Q(s,a)   RAW maximum, no tie-breaking
action_surface  the deterministic selection
tie_surface     how many actions are within TIE_ATOL of the best
margin_surface  Q_(1) − Q_(2)
```

On a 61×61 `d × phi` slice of the Q-learning table: tie counts span **1 to 7**,
margins **0 to 8.34**, and **1 174 of 3 721 cells are tied** against 2 547
decisive ones. A one-action-per-cell map cannot show that, and without it the
policy looks far more decisive than the table is.

### `d` and `phi` are not two tabular axes — measured

The discretizer indexes on `bin(d)` and `bin(phi + d)`. A `d × phi` grid
therefore collapses:

```
cells                    3 721
distinct tabular states     23
```

The slice records both, and its `coordinate_note` says so in the figure rather
than in a comment. Drawing that grid as if the two were independent tabular
axes would be a straightforward misrepresentation.

### Fixed context is identity, and it is not neutral

Every dimension not on an axis is recorded in `fixed` and included in
`slice_fingerprint`, so two slices over the same grid with different context
are different objects. Booleans render as `false`, not `0.0`; `nothing`
renders as `none`.

The reason this matters turned out to be sharper than expected. For TD3:

| context | commanded `v` |
|---|---|
| states actually reached while driving | 0.034 – 0.391, varied |
| this module's **default** slice context | 0.403 – 0.410, near-saturated |
| context with `kappa = 2.2`, `duck_crossing_available = false` | 0.021 – 0.409 |

The two pictures **do not overlap**, and the encoded inputs differ in exactly
two of fifteen components — `kappa` and `duck_crossing_available`. A default
context that looks innocuous put the policy in a regime it does not occupy.

That is what `mode = FEATURE_SPACE` and `SLICE_FEATURE_SPACE_CAVEAT` are for,
and why both are core data rather than a caption someone might drop:

> Policy-input feature-space slice; combinations are not guaranteed to
> correspond to reachable latent world states.

A future `REACHABLE_STATES` mode would sample from the world instead. It does
not exist, and nothing here implies it does.

### Continuous slices keep the two commands apart

`v_cmd` and `omega_cmd` are separate surfaces, never one arrow. Cells are
encoded with the same `encode_continuous_state` the policy sees at run time and
evaluated through the FJ7-validated actor; asserted `===` against `act` on
sampled cells. The same `policy_slice` serves TD3 and any future continuous
policy — there is no `plot_sac_d_phi`.

Measured on SAC over `d × phi`: `v ∈ [0.0003, 0.329]`, `ω ∈ [−1.497, 1.500]`.

## FJ9.4 — ROLLOUT COMPARISON FROM FROZEN ARTEFACTS

```
frozen artifact -> validated loader -> RolloutAggregate -> backend
```

Nothing in `src/visualization/rollout.jl` constructs an MDP, loads a checkpoint
or steps a transition. The loader reads a CSV, validates it, and every number
in a figure traces to a row of it. No solver was re-run.

### What the frozen artefact actually contains — and what it does not

`artifacts/fj8/six_solver_episodes.csv` is **episode-level**: 6 solvers × 20
seeds × 25 columns. It carries outcomes, event **counts** and per-episode
summary statistics.

It does **not** carry ego positions, per-decision series, or event timestamps.

So two of the views specified for this gate — the world-trajectory overlay and
the speed/heading-versus-decision panels — **cannot be built from it**, and are
not faked. `render_rollout(::RolloutComparison)` throws an error saying so
rather than inferring a trajectory from summary statistics. Producing them
requires extending what the experiment *records*, which is a decision about
the experiment and not about the renderer; the safe form would be to add
trajectory logging to `tools/fj8_comparison.jl`, re-run the same frozen
protocol, and assert every episode-level number reproduces the existing CSV
bitwise — the same experiment with more recorded, rather than a new one.

### Validation is strict, because a renderer must not forgive evidence

The loader refuses a renamed column, an episode longer than the declared
horizon, and any solver whose seed set differs from the others (which would
silently break pairing). A missing seed raises rather than producing an empty
figure.

### Negative control

Copying the artefact unchanged reproduces the fingerprint and the table
exactly. Changing **one** return by 1.0 changes the fingerprint, moves that
solver's summary, and leaves the other five untouched. Changing one
termination reason changes the fingerprint and that solver's
`env_terminated` count. The renderer cannot normalise perturbed evidence away.

### Horizon expiry is not environment termination

`in_progress` in FJ8.4b means the **evaluation horizon** expired while the
environment was still running. `EpisodeOutcome` separates `HORIZON_REACHED`
from `ENV_TERMINATED`, and the figure gives them different markers. Both occur
in the data, so the distinction is exercised rather than hypothetical.

### The two failure modes are finally legible

```
solver       eps   mean ret    med ret     len   env end   horizon   offroad   collide   stop enc   compliance
dpw@1k        20    -214.46    -211.59    85.6        19         1        13         6          7         0.0%
mcts@1k       20       6.78       7.75   150.0         0        20         0         0          4       100.0%
q_learning    20      11.63      12.86   145.6         1        19         1         0          8       100.0%
sac           20      -1.32       4.75   150.0         0        20         0         0         20        90.0%
sarsa         20      11.92      12.86   144.8         1        19         1         0          8       100.0%
td3           20    -213.78    -216.08   150.0         0        20         0         0          0          N/A
```

TD3 and DPW score within 0.7 of each other and fail in completely different
ways: TD3 never terminates, never leaves the road and **never reaches a stop
sign at all**; DPW terminates in 19 of 20 episodes, goes off-road 13 times and
collides 6 times, on episodes averaging 85.6 decisions. A return column alone
would have made them look like the same result.

`N/A` survives to the rendered table: TD3 has no compliance rate because it
faced no test. It is never printed as 0 % or 100 %.

### Seeds are explicit, never chosen for effect

`comparison_at_seed(aggregate, seed)` requires a named seed. For a paper's
"representative" seed, `median_return_seed` offers a **stated rule** — the seed
closest to that solver's median return — so the choice is recorded rather than
made by which picture looks most dramatic.

### One honest note on the figure

In the paired per-seed panel the `q_learning` line is hidden underneath
`sarsa`. That is not a plotting bug: the two agree to 0.3 in mean return and
share a median of 12.86. Jittering them apart would misrepresent how close
they are.

## FJ9.5a — SEARCH-ARTIFACT AVAILABILITY AUDIT

FJ9.4's lesson applied before designing anything: audit the artefacts first.

```
quantity                                              status            
node ids and parent/child edges                       ABSENT
per-node visit counts                                 ABSENT
per-node Q estimates                                  ABSENT
node depths                                           ABSENT
root action labels (continuous v, omega)              ABSENT
aggregate tree counters                               AGGREGATE_ONLY
per-decision diagnostics (PlanningDiagnostics.extra)  AGGREGATE_ONLY
planner configuration                                 PERSISTED
```

**Verdict: `search_visualisation_supported == false`.** Nothing that
determines a tree was ever written. `PlanningDiagnostics.extra` is aggregated
into `PlannerCost.extra` as *means*, and `episode_csv` writes no `extra`
columns at all, so even the per-decision counters are gone. What survives —
`tree_nodes = 259`, `root_children = 7` — are means over decisions in a prose
table.

The tests check this directly rather than trusting the audit: both episode
CSVs are inspected for `visits`, `q`, `value`, `depth`, `parent`, `node`,
`tree_nodes`, `action_nodes`, `v_cmd`, `omega_cmd` columns, and none is
present. No snapshot artefact exists.

### Why aggregates cannot be promoted to a tree

Demonstrated rather than argued. Two snapshots are constructed that agree on
**every** aggregate a figure could be built from — node count, root visits,
number of root children, maximum depth, and the multiset of visit counts — yet
are different trees: the depth-2 node hangs off a different parent and the root
actions differ. Reconstructing a tree from summary counters is therefore
invention, not depiction, and FJ9.5c stays blocked until a capture exists.

### What FJ9.5b would need

Unlike FJ9.4's trajectory gap, capturing this is **not** a benchmark re-run: it
answers a new question — *what did the planner simulate at one decision?* — so
it needs one frozen state, one frozen planner config and one frozen planner
RNG, captured once:

```
solver tree  ->  (solver extension)  ->  SearchSnapshot  ->  artifact  ->  renderer
```

`SearchSnapshot` and its validator already exist from FJ9.0, and the core must
continue to know nothing of `MCTS.Tree` or `DPWTree`.

## FJ9.5b — SEARCH CAPTURE

A diagnostic capture, not a benchmark run: it does not touch FJ8.4b and makes
no performance claim. It answers one question — *what did this planner
simulate from this one identified state?*

### The state was fixed before any tree was seen

```
seed 1001 (the first frozen evaluation seed)
driven by the shipped Q-learning policy
decision 30
```

Deterministic and reconstructible, and the **same latent state** for both
planners — only the action representation differs, which is the distinction
FJ4 kept explicit. Both snapshots record `state_fingerprint =
65ecc28dcdc8e0c5`, and a test asserts they match.

### Capture lives in the extension; the core sees only a snapshot

`ext/DuckietownMCTSExt.jl` is the only place that knows `MCTS.MCTSTree` or
`MCTS.DPWTree` exist. Snapshots are serialised to JSON with a fingerprint over
the tree *and* its provenance, and `tools/fj9_render_check.jl` confirms in a
fresh process that they load and validate with **`MCTS_LOADED=false`** — the
property that makes FJ9.5c possible.

`missing` is preserved end to end: the root's value is `missing`, stored as
`null`, and read back as `missing`, never `0.0`.

### What the two searches actually did

```
solver             nodes   root children   distinct root actions   visits/action   depth
MCTS.MCTSSolver      289              36                       7            1–24       2
MCTS.DPWSolver        36              24                      24             1–3       2
```

This is the quantitative form of the FJ8.3 explanation for DPW's behaviour.
MCTS spreads 36 simulations over **7** actions and concentrates up to 24 on
one; DPW spreads 35 over **24 distinct continuous actions**, so no action
receives more than **3** visits. At this budget the continuous search has
almost no evidence per action, which is what near-random action selection
looks like from the inside.

MCTS's 36 root children against 7 distinct actions is the FJ8.2 identity-hash
finding again: states never merge, so each simulation contributes its own
state node. `distinct_root_actions` is recorded so the count cannot be
misread.

### Two defects, both caught by the validator and both mine

1. **The selected action came from a different search.** `capture_search`
   originally called `action(planner, state)` again to learn what was chosen —
   which, with `keep_tree = false`, rebuilds the tree with the RNG already
   advanced. `check_snapshot` reported *"the selected action is not among the
   root children"*, which is exactly what a validator is for. The caller now
   passes the action its own search returned; nothing is re-run.

2. **The FJ9.5a audit went stale.** It scanned only `artifacts/fj8`, so
   capturing into `artifacts/fj9` left it still reporting `ABSENT` — an
   executable audit that fails to notice the very change it was written to
   notice. It now scans the whole artefact tree and reports `PERSISTED`, and
   its test tracks the *status* (all five tree fields move together, and only
   by an actual capture) rather than one day's answer. The aggregate-only
   verdicts are unchanged: means over decisions still do not determine a tree.

## FJ9.5c / FJ9.5d — DRAWING THE CAPTURED SEARCH

```
search_snapshot_*.json -> load + validate -> SearchSnapshot -> render
```

No planner, no `solve`, no `action` on the canonical path.
`tools/fj9_render_check.jl` proves it in a fresh process:
`SEARCH_TREE_PNG=true`, `ACTION_PLANE_PNG=true`, **`MCTS_LOADED=false`**.
`render_search` refuses an invalid snapshot before drawing anything.

### The value label was verified before it was written

The user's warning — do not colour by "Q" until you know what the field is —
was checked against MCTS.jl's source rather than assumed:

```julia
vanilla.jl:274   tree.q[said]   += (q - tree.q[said])   / tree.n[said]
dpw.jl:151       tree.q[sanode] += (q - tree.q[sanode]) / tree.n[sanode]
```

Both are the incremental sample-mean update, so the field really is a mean of
backed-up returns and DPW itself exports it as `best_Q`. The label is
therefore justified — with the qualification that the first value comes from
`init_Q` rather than a rollout, which the snapshot records verbatim in
`extra.value_semantics` and every figure prints.

### Statistics are per action, and that correction mattered

The first version aggregated over root **child nodes**. A vanilla MCTS tree has
one root child per simulation (states never merge, FJ8.2), so several children
repeat the same action and the same action-node visit count — reporting
"median 24 visits" for MCTS, which overstated its concentration. Grouped by
action:

```
solver             actions   visits/action mean   median   max   visited once
MCTS.MCTSSolver          7                 5.14        2    24         42.9 %
MCTS.DPWSolver          24                 1.46        1     3         58.3 %
```

The honest contrast is **`max_visits`: 24 against 3**. Both searches try some
actions only once; only MCTS ever accumulates enough evidence on one candidate
to distinguish it. That is evidence dilution in the continuous action space,
measured from the inside rather than inferred from return.

### Display filtering never touches the evidence

`visible_nodes(snapshot; max_depth, min_visits, top_k)` returns *ids*. The
snapshot is immutable and its fingerprint is asserted unchanged after
filtering, and a filtered subtree stays connected — a node is kept only if its
parent is.

### The action plane draws only what was sampled

`x = v_cmd`, `y = ω_cmd`, marker area ∝ visits, colour = the search's value
estimate, selected action marked, and the reference box
(`v ∈ [0, 0.41]`, `ω ∈ [−1.5, 1.5]`) drawn as a dashed rectangle. **No
interpolation and no heatmap**: with 24 samples a smoothed surface would imply
the planner evaluated the whole plane, which it did not. The figure states
`24 sampled actions, 58.3 % visited once` in its own subtitle.

### One paired result, no returns involved

Both snapshots come from `state_fingerprint = 65ecc28dcdc8e0c5` — the same
latent state, the same provenance rule, different action representations. Side
by side they explain *how* the action space changes the search: MCTS
concentrates 24 of 36 simulations on `SLOW_STRAIGHT`, DPW spreads 35 across 24
continuous actions and never exceeds 3 on any of them.

## TECHNICAL DEBT — the suite parsers should stop parsing

The same defect class has now appeared three times: minute-format durations,
negative durations, and the extra column a *failing* test set prints. The
problem is no longer any particular regex — it is that
**`Pkg.test`'s human-readable output is being used as a machine-readable
source of truth**, and its format is a presentation detail free to change.

The fix, recorded here rather than improvised at the end of a visualisation
gate: emit a structured report from the test run itself —

```json
{"testsets": 158, "passed": 147805, "failed": 0,
 "errored": 0, "broken": 0, "exit_code": 0}
```

— by wrapping each `include` in `test/runtests.jl` in a named `@testset`,
collecting the returned `DefaultTestSet` objects and writing JSON. After that:

```
structured reporter   authoritative
terminal parsers A/B  independent sanity check only
```

Two independent parsers remain worthwhile as defence in depth — they are how
all three defects were caught — but they should be checking a number, not
producing it. This is a test-harness change and deliberately not made inside
this gate.

## NEXT

Two capture decisions, both deliberately left to the user because both change
what an *experiment* records rather than how a figure is drawn:

1. **Trajectory logging** (FJ9.4) — needed for world-track overlays and
   speed/heading time series. Safe form: extend `tools/fj8_comparison.jl`,
   re-run the same frozen protocol, and assert every episode-level number
   reproduces the existing CSV. Should be named as its own step
   (e.g. `FJ8.4c — evaluation artefact enrichment`) so FJ8.4b stays the
   original experiment in the record.
2. ~~**Search capture** (FJ9.5b)~~ — done.

Superseded plan, kept for the record — FJ9.5 (search visualisation, converting a solver's
tree into the `SearchSnapshot` already defined in FJ9.0).

Superseded plan, kept for the record — 9.3a tabular policy slice, 9.3b value/Q slice, 9.3c
continuous policy slice (two panels, `v_cmd` and `omega_cmd`, never one
ambiguous symbol), 9.3d fixed-context honesty. Two things carry over from
earlier gates and must be respected there:

- the policy overlay must use FJ7's **deterministic near-tie rule**
  (`|Q - max| <= 1e-12`, lowest action id), not a plain `argmax` — 8 689 of
  9 000 states are near-tied, so the distinction is the norm rather than an
  edge case, and the tie count and `q_margin` deserve their own overlay;
- a slice over two feature dimensions is a **policy-input slice**, not a
  reachable-state manifold. Some combinations cannot arise from any
  `DuckieWorldState`, and the figure must say which mode it is showing.
