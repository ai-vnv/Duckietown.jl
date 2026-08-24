# FJ8 — Online Planning

Date: 2026-08-19.

## STATUS

FJ8 is a **solver-compatibility gate**, not an MCTS gate. The deliverable is
that an external solver can drive this model unchanged; a particular solver is
only the proof.

| Sub-gate | Scope | Status |
|---|---|---|
| FJ8.0 | Generative solver contract + cost profile | **PASSED** |
| FJ8.1 | Generic online-planner adapter contract (no solver present) | **PASSED** |
| FJ8.2 | MCTS.jl `MCTSSolver`, discrete `MacroAction` | **PASSED** |
| FJ8.3 | MCTS.jl `DPWSolver`, continuous `DuckieAction` | **PASSED** |
| FJ8.4a | Planner budget / latency study | **PASSED** |
| FJ8.4b | Six-way task comparison | **PASSED** |
| FJ8.5 | Solver-compatibility regression tests | **PASSED** |
| FJ8.X | Additional solver integrations | Open |

The ordering is deliberate. A planner built on a `gen` that quietly mutates its
input or aliases its branches produces results that look plausible and are
wrong (FJ8.0), and an integration designed around one solver's API quietly
becomes that solver's model (FJ8.1). Both are cheaper to prevent than to
detect later.

The architectural rule, enforced by a test rather than by convention:

```
DuckietownDecisionModels.jl  --implements-->  POMDPs.jl contracts
                                                     ^
                                   any solver --------+
```

`src/` may not name a solver. `test/test_fj8_planning.jl` greps the whole
source tree for solver vocabulary (`MCTSSolver`, `DPWSolver`, `MCTSPlanner`,
`using MCTS`, `import MCTS`) and fails if any of it appears. Solver
integrations belong in `ext/`.

## ENVIRONMENT

WSL Julia 1.11.3. FJ8.0 requires **no Python at all** — one of its tests
proves exactly that (see below). Timings were taken on the WSL host with
`ddm-ref` active but unused; they are single-run wall-clock figures, not
statistically repeated benchmarks, and are reported as an order of magnitude
for budget selection rather than as precise constants.

## WHAT WAS IMPLEMENTED

`src/evaluation/benchmark.jl`:

- `world_differences(a, b)` / `worlds_identical(a, b)` — full structural
  comparison of two world states, field by field including container lengths.
  Floats are compared **bitwise** (`===`), so `0.0` and `-0.0` count as
  different and `NaN` equals `NaN`; a purity check must not be satisfied by
  numeric coincidence.
- `shared_mutable_arrays(a, b)` — the per-branch mutable containers two states
  hold as the *same object*. Empty is the branch-purity property.
- `shared_by_design(a, b)` — the objects that are *expected* to be shared, so
  the sharing is visible rather than accidental.
- `rng_frozen(states, reference)` — the legacy state RNG is still at the
  reference stream position.
- `measure`, `GenBenchmark`, `benchmark_gen`, `gen_scaling`,
  `gen_stage_profile`, `benchmark_table`, `planning_budget_estimate`.

No `gen`, dynamics, reward or state code was changed. FJ8.0 only observes.

## FILES

| File | Purpose |
|---|---|
| `src/evaluation/benchmark.jl` | contract predicates + cost measurement |
| `test/test_fj8_gen.jl` | the contract and the profile |
| `test/test_fj8_gen_native.jl` | fresh-process proof that `gen` needs no Python |
| `tools/fj8_native_check.jl` | the script that fresh process runs |

## TESTS AND RESULTS

`test/test_fj8_gen.jl` + `test/test_fj8_gen_native.jl` — **181 assertions, 0
failures**.

| Test set | Assertions | What it pins |
|---|---|---|
| `gen` is deterministic | 63 | same `(s, a, rng)` → bitwise identical successor and reward, discrete and continuous |
| `gen` never mutates the state it is called on | 10 | root unchanged after 21 expansions per state, via `gen` and via `simulate_decision` |
| sibling branches do not alias | 33 | no shared per-branch container root↔child or child↔child, plus a write-through proof |
| the shared legacy RNG is frozen | 8 | see the finding below |
| branch order does not change the result | 9 | expansion schedule cannot affect the children |
| terminal semantics survive branching | 39 | `isterminal`/`is_truncated` agree with the chain's own reason at every step |
| the action sets a planner will search | 9 | 7 macro actions; the continuous space is **not** enumerable |
| `gen` cost profile | 4 | the measurements below |
| `gen` runs without any Python | 6 | fresh process, 5 000 calls |

### The aliasing finding

The sibling-aliasing test failed on first run, on exactly one field:

```
FJ8.0 sibling branches do not alias: Test Failed
  Evaluated: isempty(["controller_rng"])
```

Every node in the tree — root, children, siblings — held **the same
`MersenneTwister` object**. Tracing it: the transition path never calls
`branch`. It builds successors functionally, and `before_step` copies `ducks`,
`crossings_started` and `crossing_armed` once at the start of a decision, after
which `ego_tick`/`duck_step` share those already-private copies. Every field is
therefore private per branch **except** `controller_rng`, which is passed by
reference the whole way through because nothing on this path ever copies it.

Two ways to resolve it, and the choice was made by measurement rather than
taste:

| Option | Cost |
|---|---|
| copy the RNG once per decision in `before_step` | +3.5 µs and +19.1 KiB per `gen` — **3.65 % of time, 8.96 % of allocation** |
| share it, and prove it is never advanced | free |

At a 10 000-node budget the defensive copy would add ~190 MiB of allocation per
decision to protect a field with no semantic role: since FJ3.8 the transition's
stochasticity comes from the caller's `rng` (`x' ~ T(·|x,a)` with externally
supplied noise) and the decision chain never draws from the state's stream.
The copy was therefore **rejected**, and the sharing was converted from an
accident into an enforced invariant:

- `controller_rng` moved from `shared_mutable_arrays` into `shared_by_design`,
  alongside `map` and `stop_signs`, so it is listed explicitly;
- a dedicated test builds a wide, deep tree and asserts every node still sits
  at the root's exact stream position (`rng_frozen`), that the root's own
  stream is unmoved, and — behaviourally — that the next draw from the shared
  stream is the value it would have produced before the tree existed;
- `world_differences` compares the stream position, so any future advance
  fails the root-immutability test too.

This is safe **while** nothing writes to it, which is now measured on every
test run rather than assumed. The structural alternative — removing the live
RNG from the state entirely, which FJ3.8's design already implies — changes
the canonical state struct and is left as a decision for the user, not taken
unilaterally.

### Cost of one `gen`

Per macro-decision (`frame_skip = 6` physics ticks), discrete action set:

```
operation                 calls     us/call     allocs    KiB/call     calls/s
gen(branch) n=1               1      231.25     4929.0      213.48        4324
gen(branch) n=100           100       69.08     4929.0      213.48       14477
gen(branch) n=1000         1000      111.89     4929.0      213.48        8937
gen(branch) n=10000       10000       94.46     4929.0      213.48       10587
```

The allocation figure is exactly reproducible across scales (4 929 allocations,
213.48 KiB every time); the timing varies by ±20 % between runs, so **~100 µs
per `gen`, ~10 000 `gen`/s** is the honest summary and the `n = 1` row is
dominated by one-off effects.

Variants:

```
operation                  calls     us/call     allocs    KiB/call     calls/s
gen(branch) n=10000        10000       94.46     4929.0      213.48       10587
gen(chain)                 10000       95.62     4993.3      217.69       10458
gen(:branch) continuous    10000      109.24     5535.0       237.7         9154
```

Branching from one root and chaining a rollout cost essentially the same, so a
planner's expansion and rollout phases can be budgeted with one number. The
continuous variant costs ~16 % more, from the extra pre-action curvature
lookup its reference semantics require.

### Where the cost goes

Per-tick stages weighted by `frame_skip = 6`:

```
operation               calls     us/call     allocs    KiB/call     calls/s
branch(world)            2000        5.92       28.0       20.33      168835
before_step              2000        4.61      482.0        19.7       216800
ego_tick                 2000        3.71      105.0        5.62       269802
duck_step                2000        0.09        7.0        0.77     10990460
get_raw_state            2000       33.10     2007.0       80.41        30209
next_stop_candidate      2000        4.12      477.0       19.05       242935
termination_reason       2000        7.04      214.0       11.89       141965
compute_reward           2000        0.01        0.0         0.0    126863305
get_continuous_state     2000       10.16     1070.0       43.00        98388
FULL gen                 2000      111.62     4929.0      213.48         8959
```

Accounting: the weighted stages sum to 93.3 µs of 111.8 µs (**83.4 %** of the
time) and 212.4 of 213.5 KiB (**99.5 %** of the allocation), so the profile
explains essentially all of the memory and most of the time — the remainder is
the chain's own glue.

**The dominant stage is `get_raw_state`: 34.5 % of the time and 37.7 % of the
allocation of every decision**, at 2 007 allocations for one lane-frame
projection. `get_continuous_state` is second. `compute_reward` is free, and
`duck_step` is negligible.

This is recorded as the optimisation target and **deliberately not optimised
here**: `get_raw_state` is pinned bit-exact against the Python reference by
FJ2, FJ3.6, FJ5 and FJ6, so touching it risks the parity the whole project
rests on. Any rewrite must be its own gate with the parity suite as the
acceptance criterion.

### What a planner budget will cost

Derived from the measured rate, not assumed:

| Nodes per decision | Seconds per decision | Allocated per decision |
|---|---|---|
| 100 | 0.009 s | 21 MiB |
| 1 000 | 0.094 s | 208 MiB |
| 10 000 | 0.94 s | 2.0 GiB |

For the FJ7.6 evaluation shape (5 seeds × 150 decisions = 750 decisions) that
is roughly **70 s and 150 GiB of churn at 1 000 nodes**, or **12 minutes and
1.5 TiB of churn at 10 000**. Allocation rate, not raw speed, is the binding
constraint — which is what the stage profile above is for.

### `gen` needs no Python

Checked in a **fresh process**, because FJ5-R deliberately loads PythonCall
into the main test session and asserting its absence there would prove nothing:

```
GEN_CALLS=5000
REWARD_SUM_FINITE=true
PYTHON_MODULES=none
EXT_LOADED=false
NATIVE_OK=true
```

## FJ8.1 AMENDMENT — the `action` bridge was narrowed

The first FJ8.1 implementation added a three-argument method to
`POMDPs.action`:

```julia
POMDPs.action(p::POMDPs.Policy, ::AnyMDPLike, s::DuckieWorldState) =
    POMDPs.action(p, s)
```

Both argument types were ours, so it was not type piracy — but it still
extended a generic function the ecosystem owns, which means loading this
package changed how *every* POMDPs.jl policy responds to a three-argument
call. That is backwards for a package whose whole claim is that solvers plug
in without the model reshaping anything around them.

The adaptation now lives on a function this package owns:

```julia
policy_action(p::POMDPs.Policy,   m, s) = POMDPs.action(p, s)      # solvers
policy_action(p::AbstractPolicy,  m, s) = POMDPs.action(p, m, s)   # ours
policy_action(p,                  m, s) = POMDPs.action(p, m, s)   # fallback
```

`evaluate_policy`, `plan_action` and the capability probe all call it, and a
regression guard asserts the old method has not come back:

```julia
@test !applicable(POMDPs.action, planner, mdp, s)
@test  applicable(POMDPs.action, planner, s)
@test  applicable(policy_action, planner, mdp, s)
```

## FJ8.2 — WHAT WAS IMPLEMENTED

`ext/DuckietownMCTSExt.jl`, and it is deliberately the smallest thing that can
be called an integration. MCTS.jl already drives `DuckietownMDP` **without**
the extension, because the model implements the standard generative contract
and nothing else is required:

```julia
using DuckietownDecisionModels, MCTS
mdp     = DuckietownMDP("…/q_learning/training_config.yaml")
planner = solve(MCTSSolver(n_iterations = 60, depth = 12), mdp)
a       = action(planner, s)     # a valid MacroAction
```

The extension adds exactly one thing: a `plan_action` method that reports tree
statistics through the open `extra` slot, so the evaluator receives them
without the core ever learning what a tree is. It defines no model, no
conversion and no convenience constructors — a convenience layer would make
usability depend on this package having anticipated the solver, which is what
FJ8 exists to disprove.

`Project.toml`: MCTS is a **weak dependency** with an extension entry, next to
PythonCall. POMDPLinter is a test-only extra.

### Baseline solver configuration

Vanilla `MCTSSolver`, `reuse_tree = false`, default random rollout, no learned
guidance — the plainest configuration that exists, so a pass means the *model*
is usable rather than that the planner was tuned. Tree reuse is left as a
separate experiment: `DuckieWorldState` hash/equality semantics must not be a
blocker for the first integration.

## FJ8.2 / FJ8.5 — RESULTS

`test/test_fj8_mcts.jl` — **77 assertions**; `test/test_fj8_solver_independence.jl`
— **19 assertions**; 0 failures.

| Test set | Assertions | What it pins |
|---|---|---|
| the standard POMDPs.jl call sequence works | 6 | `solve` → `action`, no conversion; extension loaded; works on the bare model and the wrapper |
| planning does not disturb the model | 17 | root **bit-identical** after the search, RNG still frozen, model answers unchanged |
| the search runs on native Julia gen | 3 | counted generative consumption at three budgets |
| the planner is reproducible | 14 | same solver seed → same decision, for six seeds |
| diagnostics arrive through the generic slot | 11 | tree statistics in `extra`; `PlanningDiagnostics` gained no fields |
| the planner enters the shared evaluator unchanged | 7 | `evaluate_planner`, no MCTS-specific harness |
| the ecosystem's own requirements check | 5 | see below |
| state identity: a measured consequence | 9 | see below |
| negative control | 5 | see below |
| FJ8.5 model identical with and without the solver | 11 | 8-field fingerprint, two processes |
| FJ8.5 the solver stays a weak dependency | 8 | `[deps]` names no solver; `src/` imports none |

### Measured behaviour

```
iterations   gen calls   calls/iteration
        20         311             15.6
        40         573             14.3
        80        1217             15.2
```

Tree statistics at `n_iterations = 50, depth = 12`:

```
iterations = 50, depth_limit = 12, tree_nodes = 51,
action_nodes = 357, root_visits = 50, root_children = 7
```

Through the shared evaluator (`n = 30, depth = 8`, 16 decisions): mean latency
86 ms, p95 136 ms, **930 generative calls per action**, mean 31 tree nodes.
Reported separately from the episode metrics, never folded into one score.

### The ecosystem's verdict

POMDPLinter, run on `POMDPs.action(planner, s)` — the entry point where MCTS
actually declares its requirements (`solve` declares none):

```
[OK] discount(DuckietownMDP{MacroAction})
[OK] isterminal(DuckietownMDP{MacroAction}, DuckieWorldState)
[OK] gen(DuckietownMDP{MacroAction}, DuckieWorldState, MacroAction, MersenneTwister)
[OK] isequal(DuckieWorldState, DuckieWorldState)
[OK] hash(DuckieWorldState)
```

**All five declared requirements are satisfied; none are missing.** Saved as
`artifacts/fj8/mcts_requirements.txt`.

Two honest caveats, both recorded rather than smoothed over:

1. `POMDPLinter.check_requirements` returns **false**, and that is *not* a
   model defect. The tree contains `Unspecified` nodes because the rollout
   estimator (`MCTS.estimate_value`) declares no requirements of its own. The
   test asserts `isempty(missing_requirements)` rather than trusting that
   boolean, which would otherwise read as "the model is missing something".
2. `POMDPLinter.show_requirements` **cannot render this tree** — MCTS.jl's own
   `@POMDP_require` block raises `UndefVarError(:s)` while printing. That is an
   upstream rendering defect, not ours; the artefact was produced by walking
   the requirement tree directly, and the test pins the current status so a
   future MCTS release that fixes it is noticed.

### State identity — measured, not fixed

MCTS declares `isequal` and `hash` on the state, and `DuckieWorldState` is a
mutable struct, so both resolve to the **identity** defaults. Every successor
`gen` produces is a distinct key, so the tree never merges states:

```
n_iterations   tree_nodes
          25           26      = n + 1, exactly
          50           51
         100          101
```

The mechanism, asserted directly: two `gen` calls with the same seed give
`worlds_identical(x, y) == true`, `x !== y`, and `hash(x) != hash(y)`.

This is recorded as a property of the integration and deliberately **not
"fixed"**. Defining structural equality on the canonical state would be a
formulation change, the integration does not need it, and making hash/equality
semantics a prerequisite would have blocked the first solver for no benefit.
It does mean the search is a trajectory tree rather than a merged graph — a
real efficiency note for FJ8.4, not a correctness problem.

### Negative control

If the integration were a rubber stamp — a wrapper that ignores the model and
returns something plausible — perturbing the reward would change nothing. A
**new in-memory config** with `alpha_lateral` inverted and amplified (the
shipped baseline file untouched) leaves the dynamics bit-identical
(`worlds_identical(xb.sp, xp.sp)`) while changing the reward (`xb.r != xp.r`),
and changes the planner's decision at **6 of 8 probed states**. The planner is
genuinely reading this model.

## FJ8.3 — CONTINUOUS DPW

Baseline configuration, chosen conservatively and justified by measurement
rather than convention:

```julia
DPWSolver(n_iterations = …, depth = 10, exploration_constant = 5.0,
          enable_action_pw = true,      # the action space is a box
          enable_state_pw  = false,     # the kernel is a measured point mass
          k_action = 4.0, alpha_action = 0.5,
          rng = MersenneTwister(seed), keep_tree = false)
```

Uniform proposals (MCTS.jl's default `RandomActionGenerator`), no learned
guidance, no tree reuse, and **no hyper-parameter sweep** — FJ8.0 showed a
continuous `gen` costs ~109 µs and 237 KiB, so the first question is
correctness and scaling, not tuning.

`test/test_fj8_dpw.jl` — **322 assertions, 0 failures**.

### 8.3a — the proposal is genuinely continuous

| Source | Proposals | Distinct `v` | Distinct `ω` | On the macro-action lattice |
|---|---|---|---|---|
| `rand(rng, DuckieActionSpace)` | 10 000 | 10 000 | 10 000 | **0** |
| labels from a real 300-iteration search | 300 | 300 | 300 | **0** |

Every proposal satisfies `0 ≤ v ≤ 0.41` and `−1.5 ≤ ω ≤ 1.5`; the box is
covered rather than clustered (`v ∈ [0.001, 0.409]`, `ω ∈ [−1.482, 1.493]` in
the real search). Same stream → identical sequence; different stream →
different sequence.

The second row matters more than the first: it reads `tree.a_labels` out of an
actual search, so it is the proposal path the planner really used, not a
parallel implementation of it.

### 8.3b — action widening is active and sublinear

```
   n   visits   root_action_children   action_nodes   state_nodes   max_state_children      gen      ms
  25       25                     20             25            26                    1      604    87.6
  50       50                     29             50            51                    1     1193   143.5
 100      100                     40            100           101                    1     2341   268.7
 250      250                     64            250           251                    1     5793   678.4
 500      500                     90            500           501                    1    11522  1374.6
```

Children per visit falls monotonically — **0.80 → 0.58 → 0.40 → 0.256 → 0.18** —
which is the sublinearity progressive widening exists to produce. The realised
counts track `⌈k·N^α⌉ = ⌈4√N⌉` almost exactly (20, 29, 40, 64, 90 against
20.0, 28.3, 40.0, 63.2, 89.4), so the configured coefficients are the ones
actually in force.

**Negative controls.**

- *Continuous, widening off*: the search fails with
  `MethodError: no method matching iterate(::DuckieActionSpace)`. An
  unenumerable box cannot be enumerated — recorded as a capability fact rather
  than caught and ignored. This is also why `enable_action_pw = true` is not a
  tuning choice on this model but a requirement.
- *Discrete, baseline coefficients*: PW on and off both give **7** root
  children, because the widening bound at `N = 100` is `4·10 = 40`, far above
  the seven available actions. Progressive widening is simply **vacuous on a
  small discrete action set**. Reported rather than tuned away.
- *Discrete, bound below the action count* (`k = 1.0, α = 0.3` → bound 3.98):
  widening bites, giving **4** root children against 7 with it disabled. This
  is the control proper.

### 8.3c — state widening is unnecessary, measured on the transition

The justification for `enable_state_pw = false` is a measurement of the
transition, not a reading of a config value.

Trigger-region states were located by driving the model with the shipped
Q-learning policy across four spawn seeds and keeping every state at which
`gen` actually draws from the caller's stream. At each such state, the
successor was sampled with **64 different RNG seeds**:

```
trigger states found : 4
rng seeds per state  : 64
unique successors    : 1        (bitwise, via world_differences)
p_cross              : 1.0
max_crossings        : 1
```

Both action types were checked at the same world states — the discrete action
the policy actually chose, and a continuous `DuckieAction(0.2, 0.3)` — since
the two models share the state space and the same duck trigger.

> State PW is disabled because repeated generative samples from the relevant
> baseline transition produce one unique successor outcome.

The structural consequence is confirmed in the tree itself: with state PW off,
`max_state_children == 1` and `unique_transitions == action_nodes` at every
budget, i.e. exactly one sampled successor per state-action node.

### 8.3d — integration

`solve(DPWSolver(...), mdpc)` → `action(planner, s)` returns a `DuckieAction`
inside the box, with the root state **bit-identical** afterwards, the legacy
RNG still frozen, only native `gen` consumed, and the same planner seed giving
the same decision. Through the shared evaluator (`n = 40, depth = 6`, 12
decisions): mean latency 168 ms, p95 185 ms, **1 590 generative calls per
action**, mean 25 root action children — reported separately from the episode
metrics.

## FJ8.4a — THE COST–SEARCH CURVE

Not "what is the best budget". The deliverable is the relation between what a
solver is *told* to do and what it actually costs.

### The seed split is frozen first

`configs/planning/fj8_seeds.yaml` fixes two disjoint sets, and a test asserts
they do not overlap:

| Set | Contents | Used by |
|---|---|---|
| development | states `[101…105]`, planner `[201, 202]` | FJ8.4a, and **any** configuration choice |
| evaluation | 20 episode seeds `[1001…1020]`, planner `2026`, horizon 150 | FJ8.4b only |

FJ8.4a never reads the evaluation list. A configuration therefore cannot be
chosen against the seeds it will later be reported on.

### Method

`src/evaluation/budget.jl` — `budget_study(mdp, make_planner, budgets; states,
repeats)`. The planner arrives as a closure `(budget, seed) -> planner`, which
is what keeps the study solver-agnostic; it never names a solver type. The same
five states are used at every budget, one warm-up decision per budget is
discarded, and each (state, budget) is repeated with an identical planner seed
so the search is identical and the repeats measure **timing noise only**.

### MCTS, discrete macro actions

```
solver            iters    gen/act   gen/iter    ms mean    ms p50    ms p95   MiB/act    tree_nodes  action_nodes
MCTS (discrete)      25      697.4       27.9      69.48     67.07    129.03    170.04          26.0         182.0
MCTS (discrete)      50     1402.2      28.04     125.74    133.43     165.0    340.27          51.0         357.0
MCTS (discrete)     100     2814.0      28.14     250.56    288.58    317.11    682.18         101.0         707.0
MCTS (discrete)     250     7049.4       28.2      631.0    743.75    805.76   1709.06         251.0        1757.0
MCTS (discrete)     500    14252.8      28.51    1322.71   1550.01   1665.98   3455.21         501.0        3507.0
MCTS (discrete)    1000    28619.8      28.62    2685.75    3178.1   3394.58   6935.54        1001.0        7007.0
```

### DPW, continuous action box

```
solver             iters    gen/act   gen/iter    ms mean    ms p50    ms p95   MiB/act   state_nodes  root_action_children  max_state_children
DPW (continuous)      25      710.6      28.42      85.56     73.38     173.9    212.06          26.0                  20.0                 1.0
DPW (continuous)      50     1444.8       28.9      163.5    157.54    215.07    430.61          51.0                  29.0                 1.0
DPW (continuous)     100     2877.2      28.77     331.32    325.91    401.66    858.34         101.0                  40.0                 1.0
DPW (continuous)     250     7151.8      28.61     826.59    738.89   1108.22    2135.8         251.0                  64.0                 1.0
DPW (continuous)     500    14304.2      28.61    1578.77   1468.05   2029.13   4280.74         501.0                  90.0                 1.0
DPW (continuous)    1000    28533.8      28.53    3208.25   2994.68   3880.41   8539.72        1001.0                 127.0                 1.0
```

Allocation is the binding constraint, exactly as FJ8.0 predicted: **6.9 GiB per
decision** for MCTS at 1 000 iterations, 8.5 GiB for DPW.

### A correction to the premise we were both using

Earlier readings of FJ8.2/8.3 suggested the two solvers differ in generative
calls per iteration — roughly 15.6 for MCTS against 23.0 for DPW. **Averaged
over the same development states they do not**: MCTS 27.9–28.6, DPW 28.4–28.9,
ratio **1.01**. Those earlier figures came from different single states, not
from a solver difference.

What the cost per iteration actually depends on is the **state**, and the
mechanism is rollout survival — a rollout ends when the episode terminates,
which happens far sooner from some spawns than others:

```
  seed= 11  random_rollout_len=  9  gen/iter@depth10= 15.55  @depth12= 15.55
  seed=101  random_rollout_len=  9  gen/iter@depth10= 10.45  @depth12= 10.45
  seed=102  random_rollout_len= 45  gen/iter@depth10= 39.85  @depth12= 39.85
  seed=103  random_rollout_len= 26  gen/iter@depth10=  33.2  @depth12=  33.2
  seed=104  random_rollout_len= 21  gen/iter@depth10= 29.75  @depth12= 29.75
  seed=105  random_rollout_len= 26  gen/iter@depth10= 25.75  @depth12= 25.75

  pearson(rollout_len, gen/iter) = 0.922      spread max/min = 3.81x
```

Two consequences, both measured rather than assumed:

1. **`depth` is not binding on this model.** Depth 10 and depth 12 give
   *identical* call counts at every seed: rollouts die of termination long
   before the cap. Raising the depth limit buys nothing here, which removes it
   from the FJ8.4b tuning surface.
2. **Cost per iteration varies 3.8× across spawn states.** Compute-matching
   must therefore be done on calls measured over the state set being used, not
   on a per-solver constant.

The conclusion the user drew still holds, and now holds for a better reason:
iteration counts are not a computational budget. They are just not unequal
*between these two solvers* — they are unequal *between states*.

### Compute-matched operating points

Nearest point of the measured grid is too coarse to be fair:

```
solver             target gen   iters   actual gen    rel err    ms mean
DPW (continuous)          500      25        710.6      42.1%      83.23
MCTS (discrete)           500      25        697.4      39.5%      67.21
DPW (continuous)         1000      25        710.6     -28.9%      83.23
MCTS (discrete)          1000      25        697.4     -30.3%      67.21
DPW (continuous)         2000      50       1444.8     -27.8%     169.78
MCTS (discrete)          2000      50       1402.2     -29.9%     129.06
```

`estimate_budget_for_calls` instead inverts the measured calls-per-iteration
rate; the estimate is then **run and measured**, never trusted:

```
solver             target gen   iters   actual gen    rel err    ms mean
MCTS (discrete)           500      18        494.2      -1.2%      59.46
DPW (continuous)          500      18        510.0       2.0%      59.93
MCTS (discrete)          1000      36       1006.8       0.7%     101.60
DPW (continuous)         1000      35       1002.8       0.3%     120.99
MCTS (discrete)          2000      71       2009.4       0.5%     195.52
DPW (continuous)         2000      70       2012.8       0.6%     237.20
```

Every point lands within **±2 %** of its target. These are the operating points
FJ8.4b will use. Saved as `artifacts/fj8/budget_study.md`.

At equal generative work the two planners cost similar wall time, with DPW
19–21 % slower at the higher budgets — consistent with FJ8.0's measurement that
a continuous `gen` costs ~16 % more than a discrete one.

### A learned policy has a flat curve

The study is solver-agnostic, so the same function runs on a Q-table:

```
solver         iters    gen/act   gen/iter    ms mean    ms p50    ms p95   MiB/act
Q-learning         1        0.0        0.0       0.06      0.04      0.29      0.08
Q-learning       100        0.0        0.0       0.03      0.03      0.03      0.08
Q-learning      1000        0.0        0.0       0.03      0.03      0.03      0.08
```

`model_calls` is **0**, not `-1`: the policy demonstrably performs no
generative planning, which is measured knowledge. `-1` remains reserved for
*unmeasured* — an uninstrumented model — so "free" and "unknown" never collapse
into the same number.

### Full suite

`tools/run_full_suite.sh`, WSL Julia 1.11.3 with `ddm-ref` active:

```
top-level testsets : 122
assertions passed  : 144 390
failed / errored   : 0
broken             : 0
Pkg verdict        : tests passed
PKGTEST_EXIT       : 0
```

Both parsers agree exactly, with zero unparsed headers — which they did not
before the elapsed-time bug above was fixed.

| Stage | Testsets | Assertions | Added |
|---|---|---|---|
| after FJ7 | 76 | 143 516 | — |
| after FJ8.0 | 85 | 143 697 | 181 |
| after FJ8.1 | 92 | 143 836 | 139 |
| after the FJ8.1 amendment | 92 | 143 843 | 7 |
| after FJ8.2 + FJ8.5 | 103 | 143 939 | 96 |
| after FJ8.3 | 110 | 144 261 | 322 |
| after FJ8.4a | 115 | 144 318 | 57 |
| after FJ8.4b | 122 | 144 390 | 72 |

Every earlier gate still passes unchanged. FJ8.0 added observation code only;
FJ8.1 widened three existing signatures from `DuckietownMDP` to `MDPLike` and
added an opt-in `record` keyword whose default leaves the FJ7.6 evaluator
byte-identical; FJ8.2 added a weak dependency and an extension. FJ7.6's own
determinism tests still pass, which is what confirms none of it leaked into
the evaluator.

## FJ8.1 — WHAT WAS IMPLEMENTED

`src/interfaces/planning.jl`. It mentions no solver, and the guard test
enforces that.

- **`PlanningDiagnostics(planning_time, model_calls, extra)`** — only two
  universal fields; everything solver-specific goes in the open `extra`
  `NamedTuple`. A tree search fills `(tree_nodes, max_depth, iterations)`, a
  particle method `(particles, belief_nodes)`, a learned policy nothing at all.
  The evaluator never looks inside, so it cannot assume a planner has a tree.
  `model_calls = -1` means "not measured" and is deliberately distinct from
  `0`, so an uninstrumented run never reads as free.
- **`InstrumentedMDP(mdp)`** — the same model, counting generative calls. It
  forwards every POMDPs.jl function and, through `getproperty`, every field, so
  code written against `DuckietownMDP` works on it unchanged. It is a
  measuring device, not a translation layer, and the tests assert that
  **bitwise**: for every action and several seeds, `gen` through the wrapper
  returns a successor identical field-by-field to `gen` on the bare model, with
  `r === r`.
- **`MDPLike{A}` / `AnyMDPLike`** — the model or a transparent wrapper. The
  existing tabular, actor and evaluator signatures were widened to this, so
  no code needed a second implementation.
- **`POMDPs.action(policy, mdp, s) = POMDPs.action(policy, s)`** for
  `POMDPs.Policy` — the single bridge between the two-argument form solvers
  produce and the three-argument form this evaluator uses. **This is why an
  external solver needs no adapter at all**:

  ```julia
  planner = solve(SomeSolver(...), mdp)
  evaluate_policy(mdp, planner)
  ```
- **`plan_action(policy, mdp, s) -> (action, PlanningDiagnostics)`** — the
  default times `action` and reads the call counter, which already works for
  any solver. An extension may add a method to fill `extra`; that is the only
  thing an extension is ever expected to add, and it is optional.
- **`model_capabilities(mdp; policy, ...)` / `capability_report`** — what the
  model offers, determined by exercising the interface.

`src/evaluation/metrics.jl` gained planning cost, kept strictly apart from task
performance:

- `evaluate_policy(...; record=Vector{PlanningDiagnostics})` — same episode,
  same actions, optionally observed. Default `nothing` leaves FJ7.6 behaviour
  byte-identical.
- `PlannerCost` — decisions, total time, latency mean/p50/p95/max, model calls
  total and per action, plus the mean of every numeric `extra` field the
  planner reported.
- `evaluate_planner(mdp, planner)` → `(episodes, cost)`, two values, never one
  score. A slow planner that drives well and a fast one that drives badly stay
  distinguishable.

## FJ8.1 — RESULTS

`test/test_fj8_planning.jl` — **139 assertions, 0 failures**, with **no
planning library installed**. Every planner in these tests is a hand-written
stand-in subtyping `POMDPs.Policy`, which is the point: if these pass, a real
solver's only remaining job is to satisfy the POMDPs.jl interface.

| Test set | Assertions | What it pins |
|---|---|---|
| `InstrumentedMDP` is transparent | 65 | same type parameters, same interface answers, bitwise-identical transitions, forwards fields, does not mutate |
| the call counter counts what a planner consumes | 9 | planner calls counted exactly; the evaluator's own environment step is *not* charged |
| any `POMDPs.Policy` plugs in without a wrapper | 5 | `action(planner, mdp, s) == action(planner, s)`, and it runs through the FJ7.6 evaluator |
| diagnostics are generic, open for solver specifics | 15 | default path, unmeasured path, custom `extra`, learned policy reporting nothing |
| planning cost aggregates without assuming a tree | 18 | latency quantiles, per-action call rate, `extra` means, and recording changes no episode |
| capabilities are measured, not declared | 26 | the table below |
| the core names no solver | 1 | `src/` contains no solver vocabulary |

### Capability report

Measured by exercising the interface, not asserted:

| Capability | Discrete model | Continuous model |
|---|---|---|
| generative transition | YES | YES |
| discrete actions | YES | NO |
| continuous actions | NO | YES |
| enumerable actions | YES (7 macro actions) | **NO** (a box — a solver must widen) |
| continuous state | YES | YES |
| terminal states | YES | YES |
| truncation separate from termination | YES | YES |
| discount | YES | YES |
| initial-state sampler | YES | YES |
| explicit transition distribution | NO | NO |
| observation model | NO (this is an MDP) | NO |
| belief updater | NO | NO |

### The stochasticity finding — this changes how FJ8.3 should be configured

Probing revealed something a planner must know, and it is reported as three
separate facts rather than one flag:

| Probe | states | rng-consuming states | outcomes differ |
|---|---|---|---|
| constant action, from spawn | 7–32 (all end `OFFROAD`) | 0 | no |
| Q-learning policy, 200 decisions | 200 | **1 (0.5 %)** | **no** |

Two distinct causes, both traced rather than guessed:

1. **A constant action never reaches the interesting states.** Driving one
   macro action leaves the road within 7–32 decisions, well before the ego
   comes within the trigger window, so a naive probe reports "no randomness"
   as a property of the probe, not of the model. `model_capabilities` therefore
   takes a `policy` argument, and the report says which probe produced it.
2. **On the shipped configurations the model is deterministic in *outcome*.**
   Both `q_learning/training_config.yaml` and `sac/training_config.yaml` set

   ```yaml
   p_cross: 1.00
   max_crossings_per_episode: 1
   ```

   so the transition *does* draw from the caller's stream — once per episode,
   at the single eligible trigger window — but `rand() < 1.0` is always true.
   The draw is consumed and the successor is the same for every seed. Twelve
   seeds at that state produced bitwise-identical successors.

Consequence for FJ8.3, derived from measurement rather than convention:
`T(·|s,a)` is a point mass on these baselines, so **DPW's state widening has
nothing to widen over**. `k_state = 1, alpha_state = 0` is the configuration
the model justifies; any larger setting spends budget generating identical
children. Action widening is where a continuous planner's budget belongs.

If genuinely stochastic transitions are wanted for planning experiments, that
requires a **new, explicitly named configuration** with `p_cross < 1` — never a
silent edit of a reference baseline.

## KNOWN DEVIATIONS

1. **`controller_rng` is shared across every branch.** Documented, measured and
   enforced as a frozen invariant rather than copied; rationale and cost above.
2. **Timings are single-run wall clock.** Allocation counts are exactly
   reproducible; times vary ~±20 % run to run. No timing threshold is asserted
   anywhere, and none should be read as a guarantee.
3. **`get_raw_state` is the bottleneck and stays unoptimised** for now, to
   protect the bit-exact parity established in FJ2/FJ3.6/FJ5/FJ6.

## FJ8.4b — SIX-SOLVER COMPARISON

Four solver families on one formulation and one evaluator: tabular learned,
deep learned, discrete online planning, continuous online planning.

The protocol was frozen in `configs/planning/fj8_evaluation.yaml` **before** the
evaluation seeds were touched, and a test asserts its contents (planner
budgets, widening flags, horizon, seed disjointness). Twenty evaluation seeds,
horizon 150, planner RNG 2026, one shared `evaluate_policy`. Nothing was tuned
after reading a result.

The experiment is `tools/run_comparison.sh` → `tools/fj8_comparison.jl`, run in
its own `experiments/` environment because MCTS is a weak dependency. It is
**not** part of the regression suite: twenty minutes of wall time does not
belong there. `test/test_fj8_comparison.jl` (72 assertions) exercises the
machinery — pairing, denominators, statistics, two-block separation, protocol
guard — on a small configuration.

Artefacts: `artifacts/fj8/six_solver_comparison.md`,
`six_solver_episodes.csv` (episode-level paired records),
`planner_sensitivity_episodes.csv`.

### Task performance

```
solver         return     median                95% CI   progress   mean|d|    max|d|  mean|phi|    speed     len
q_learning      11.63      12.86         [-13.9, 27.5]      20.08    0.0342      0.25     0.1494   0.1391   145.6
sarsa           11.92      12.86         [-13.0, 27.5]      20.07    0.0323      0.25     0.1436   0.1408   144.8
sac             -1.32       4.75           [-9.2, 5.0]      14.52    0.0596    0.1394     0.1171   0.0979   150.0
td3           -213.78    -216.08      [-222.4, -205.0]       4.48     0.033    0.0723     0.1241   0.0303   150.0
mcts@1k          6.78       7.75            [4.6, 8.8]      15.96    0.0859      0.25     0.2522   0.1125   150.0
dpw@1k        -214.46    -211.59      [-228.2, -197.7]       7.13    0.0744      0.25     0.3558   0.0889    85.6
```

### Safety, stops and ducks

```
solver        eps   offroad   collide   duck hit   stop enc   full stop   violation   compliance   crossings
q_learning     20         1         0          0          8           8           0       100.0%          20
sarsa          20         1         0          0          8           8           0       100.0%          20
sac            20         0         0          0         20          18           2        90.0%          20
td3            20         0         0          0          0          20           0          n/a           0
mcts@1k        20         0         0          0          4           5           0       100.0%          20
dpw@1k         20        13         6          0          7           4           7         0.0%           1
```

TD3's `n/a` is the denominator rule doing its job: it completes no stop
encounter — `passed_stops = 0` over all 20 seeds — so it has no compliance
rate at all, and reporting 100 % there would have credited a policy that never
finished the test.

> **Corrected 2026-08-23 (FJ9.6).** This paragraph previously read that TD3
> "never reaches a stop sign". That is not what happened, and this artefact
> said so at the time: `stop_zone_decisions = 2289` for TD3, about 114 of each
> episode's 150 decisions. TD3 reaches the sign in every episode, performs a
> full stop, and then never proceeds — its mean speed of 0.030 m/s is the
> stall, not an approach. The per-decision series in FJ9.6 made this visible;
> the `n/a` verdict and every number in the tables above are unchanged.

### Computational cost — a separate axis, not comparable to the above

```
solver                      family    ms mean    ms p50    ms p95    gen/act    iters   act nodes
q_learning       tabular (learned)      0.036     0.021     0.061        0.0        -           -
sarsa            tabular (learned)      0.035     0.020     0.033        0.0        -           -
sac                 deep (learned)      0.059     0.040     0.054        0.0        -           -
td3                 deep (learned)      0.052     0.036     0.051        0.0        -           -
mcts@1k        discrete online plan    125.464   128.458   169.266      999.4     36.0       259.0
dpw@1k       continuous online plan    106.847   104.390   213.620      721.7     35.0        34.9
```

A learned policy decides in ~0.04 ms by tabular lookup or one network forward
pass; a planner spends ~100 ms running roughly a thousand model simulations.
These are not two points on one efficiency axis and no combined score is
formed anywhere in this gate.

### Paired per-seed differences in return

All six ran the same twenty seeds, so these are differences on identical
initial conditions rather than a comparison of independent group means.

```
pair                      metric   mean diff     median                  95% CI    a>b    b>a
mcts@1k - q_learning         ret       -4.85     -9.504         [-21.08, 20.86]      1     19
mcts@1k - sarsa              ret      -5.142     -9.504         [-21.09, 19.98]      1     19
mcts@1k - sac                ret       8.094      2.854            [1.26, 16.9]     14      6
mcts@1k - td3                ret     220.559    221.419        [211.03, 229.97]     20      0
dpw@1k - q_learning          ret     -226.09   -241.939      [-249.65, -196.16]      0     20
dpw@1k - sarsa               ret    -226.382   -241.939      [-249.66, -197.03]      0     20
dpw@1k - sac                ret     -213.146   -212.588      [-227.83, -195.99]      0     20
dpw@1k - td3                 ret      -0.681      -8.39         [-15.95, 17.53]      9     11
mcts@1k - dpw@1k             ret     221.239    218.459        [204.38, 234.94]     20      0
```

Note how the pairing changes the reading of MCTS vs the tabular policies: the
mean difference is −4.85 with a CI spanning zero, yet the tabular policy wins
on **19 of 20 individual seeds**. Group means alone would have hidden that.
With n = 20 all of this is descriptive; **no superiority claim is made**.

### The compute match did not transfer — and the data says why

This is the most important methodological result of FJ8.4b.

The operating points were calibrated in FJ8.4a on development states, where
both planners realised ~1 000 generative calls per decision to within 1 %. On
the evaluation seeds:

| Solver | target | realised | miss |
|---|---|---|---|
| mcts@1k | 1 000 | **999.4** | −0.1 % |
| dpw@1k | 1 000 | **721.7** | **−28 %** |

So the primary DPW row is **not** compute-matched to MCTS, and must not be read
as if it were. The per-position table shows the mechanism exactly:

```
episode fraction     mcts gen/act     dpw gen/act
       0.0-0.2             1019.2          1019.1
       0.2-0.4             1020.9           835.3
       0.4-0.6             1024.8           940.3
       0.6-0.8              913.9           602.0
       0.8-1.0             1018.3           195.1
```

DPW starts at the calibrated cost and collapses to 195 calls per decision by
the end of its episodes. The cause is the FJ8.4a finding applied to itself:
cost per iteration is set by rollout survival, DPW drives off the road, its
states become near-terminal, and rollouts then die immediately. **A planner's
realised compute budget is endogenous to how well it drives.** Compute matching
calibrated on one state distribution transfers only to solvers that stay in
that distribution.

### Why DPW does badly here, and why that is not a defect

At 35 iterations the widening rule admits `⌈4√35⌉ ≈ 24` root actions, so each
sampled action receives about 1.5 visits — close to picking uniformly among 24
random actions drawn from the box. `mean|phi| = 0.3558` is the highest of all
six, i.e. the steering swerves. The baseline was deliberately specified with
uniform proposals and no learned guidance, so this is the honest behaviour of
an under-budgeted unguided continuous planner, not a broken integration —
FJ8.3 already showed the same solver returns valid in-box actions, leaves the
model bit-identical and consumes only native `gen`.

The budget sensitivity confirms it is a budget effect:

```
solver        return     median                95% CI      len     gen/act
mcts@500         7.00       4.70           [3.6, 12.9]     60.0       489.9
mcts@1000        5.87       3.24           [2.9, 11.0]     60.0      1047.2
mcts@2000        6.73       4.57           [3.6, 12.0]     60.0      2030.0
dpw@500       -165.27    -199.22       [-210.1, -86.1]     48.0       292.8
dpw@1000       -30.15     -41.39        [-43.9, -16.4]     60.0       886.2
dpw@2000        -26.90     -39.69        [-43.1, -10.7]     60.0      1834.8
```

MCTS is flat across a 4× budget range (7.0 / 5.9 / 6.7 — within noise), while
DPW improves by an order of magnitude (−165 → −30 → −27) and stops terminating
early. Continuous action search is budget-hungry in a way discrete search over
seven macro actions is not. This block is planner-only, on five seeds at
horizon 60, and is deliberately **not** four extra rows of the six-solver
table.

## WHICH SOLVER COMPLETES `small_loop`?

The FJ8.4b table cannot answer this, for two reasons that both had to be found
before the question could be posed properly.

**1. There is no goal.** All four shipped configs set `goal_tile: null`, so the
`GOAL` terminal can never fire. "Completing the loop" is not a terminal
condition of the baseline task at all — the shipped objective is to drive well
and survive, not to arrive somewhere. Completion therefore has to be *defined*:
here it is the winding number of the ego about the loop centre reaching 2π.

**2. The comparison horizon was too short for a lap.** `small_loop` is 3×3
tiles of 0.585 m; the lane centreline is **4.258 m** per lap. At the fastest
speed any solver achieved (0.14 m/s) one lap needs ~30.4 s = **152 decisions**,
and FJ8.4b ran 150. The six-solver table was, by construction, measuring
sub-lap behaviour.

Re-run to each configuration's **own** step limit (`tools/run_lap_analysis.sh`,
artefact `artifacts/fj8/lap_completion.txt`):

```
solver       seeds   laps mean    max   >=1 lap    path      time   tiles  crashed  ends by
q_learning      20        1.59   1.72     19/20   6.87 m    48.1 s     7.8        1  timeout/offroad
sarsa           20        1.58   1.72     19/20   6.83 m    47.5 s     7.8        2  timeout/offroad
sac             20        2.63   2.67     20/20  12.34 m   120.0 s     8.0        0  horizon
td3             20        0.91   1.15      8/20   4.60 m   120.0 s     7.4        0  horizon
mcts@1k          5        1.49   1.64       5/5   5.41 m    50.0 s     8.0        0  timeout
dpw@1k           5        0.34   0.40       0/5   1.70 m    19.3 s     3.6        5  offroad/collision
```

The winding measure is corroborated by path length: `laps × 4.258 m` matches
the measured path to within the lane deviation in every row, so the number is
distance actually covered around the ring and not a vehicle spinning in place.
`tiles` is how many of the 8 drivable tiles were visited.

**Answer.** Four of the six complete the loop:

- **SAC — the only one that completes it on every seed** (20/20) with no
  crashes, and keeps going: 2.63 laps in 120 s.
- **Q-learning and SARSA — 19/20**, then stopped by their own config's
  `max_steps = 1500` physics ticks (250 decisions), not by failing. One seed
  each goes off-road.
- **MCTS — 5/5**, no crashes, at a similar pace to the tabular policies.
- **TD3 — mostly cannot**: 8/20, and only because it is slow (0.030 m/s), not
  because it crashes. It never leaves the road in 120 s.
- **DPW — cannot at this budget**: 0/5, all five end off-road or in a
  collision after 1.7 m, having reached only 3.6 of the 8 tiles.

One caveat on comparability, stated because the numbers invite the wrong
reading: the horizons are **not** equal here. The tabular pair is capped at 250
decisions by its own configuration while SAC and TD3 were given 600 (their
configs allow 1500), so SAC's 2.63 laps against Q-learning's 1.59 is partly
more time, not only more speed. What *is* horizon-independent is the ≥1-lap
column and the crash count.

Note also that this ordering is not the FJ8.4b return ordering. Q-learning has
the best mean return over 150 decisions while SAC completes more laps over its
own horizon — the reward function and lap completion are different objectives,
and the shipped task optimises the former.

## NEW DEFECTS FOUND IN FJ8.4a — IN THE MEASURING TOOLS

Both were found by the two-parser cross-check disagreeing, which is the reason
that check exists.

1. **The suite parsers dropped any test set that ran longer than a minute.**
   Julia prints the elapsed column as `3.5s`, but as `3m17.5s` once a test set
   passes sixty seconds. Both `tools/suite_summary.sh` and
   `tools/suite_summary.py` accepted only the seconds form, so FJ8.4a's
   3-minute cost-search test set — 36 assertions — was silently excluded from
   both totals. Reported total would have been 144 282 instead of 144 318.

   This is exactly the defect class recorded in the README correction about
   historical undercounts, reproduced by my own tooling. Both parsers now
   accept `1h2m3.4s`, and `tools/recount_logs.sh` re-summarises every captured
   log whenever the parsers change.

   **Every previously reported total was re-checked** with the fixed parser:
   no earlier log contains a minute-format row, so 143 516 / 143 697 /
   143 836 / 143 939 / 144 261 all stand. Only this gate's figure moved.

2. **Windows-side text-mode writes introduced CRLF into 18 files.** Patching
   files with a Python heredoc from the Windows host translated `\n` to
   `\r\n`, which Julia tolerates but `bash` does not: `tools/suite_summary.sh`
   became unexecutable through a nested call, producing empty output rather
   than a wrong number. All 18 files were normalised, and the authoritative
   suite was re-run afterwards so the reported log comes from the final bytes.

## NEW DEFECTS

One, in the FJ8.0 scaffolding itself: `tools/fj8_native_check.jl` put its
accumulator in a bare top-level loop, so Julia's soft-scope rule made it a new
local and the script raised `UndefVarError`. The test reported this as an
unavailable check rather than passing — it did not silently succeed — and the
script now wraps the loop in a function. No package code was involved.

## SCIENTIFIC INTERPRETATION

The generative model is safe to search. Determinism, root immutability,
branch independence and schedule independence are established structurally and
bitwise, not by sampling a few numbers, and the one place where two branches
did share state was found by the test rather than by inspection — which is the
point of writing the contract before the planner.

The cost profile changes what FJ8.1/8.2 should aim for. At ~100 µs and 213 KiB
per `gen`, the planner's node budget is bounded by allocation pressure well
before it is bounded by time, and roughly three quarters of both goes into two
observation projections rather than into the physics. A planner that reuses a
node's already-computed projection instead of recomputing it will therefore
buy more than any tree-policy tuning — which is worth knowing *before* choosing
an exploration constant, not after.

## NEXT

FJ8.2 — MCTS.jl `MCTSSolver` on the discrete model, as an **extension**
(`ext/DuckietownMCTSExt.jl`) behind a weak dependency, so `using
DuckietownDecisionModels` continues to work with no planning library installed.
The extension may only bridge interfaces; it may not touch state, reward,
transition, action bounds or termination semantics, and the `src/` guard test
keeps solver vocabulary out of the core.

The milestone FJ8.2 establishes is not the planner's score. It is:

> `DuckietownDecisionModels.jl` is usable by an external, standard POMDPs.jl
> online-planning solver without changing the model.

FJ8.5 then makes solver independence a measured property rather than a
documented principle: the package must behave identically with the planning
library absent, present-but-unloaded, and loaded — with `gen`, `initialstate`,
`actions`, reward, evaluation and the four learned baselines unaffected in
every case.
