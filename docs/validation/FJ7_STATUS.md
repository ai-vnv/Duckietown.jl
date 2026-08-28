# FJ7 — Existing Solver Baselines

Date: 2026-08-19.

## STATUS

| Sub-gate | Scope | Status |
|---|---|---|
| FJ7.1 | Q-learning: Q-table import, 9 000-state greedy parity, tie-breaking | **PASSED** |
| FJ7.2 | SARSA: same | **PASSED** |
| FJ7.3 | Continuous observation adapter (exact 15-D policy input) | **PASSED** |
| FJ7.4 | SAC action-inference parity (oracle + native actor, layer by layer) | **PASSED** |
| FJ7.5 | TD3 action-inference parity (oracle + native actor, layer by layer) | **PASSED** |
| FJ7.6 | Common evaluation harness across all four policies | **PASSED** |

FJ7.4/7.5 were previously blocked for lack of any PyTorch on this machine.
The block was cleared by creating a NEW, isolated inference environment
(option A), not by touching the validated simulator reference.

## ENVIRONMENT

Two Python environments, deliberately kept apart, plus a Python-free Julia
path:

| Env | Python | Role | Used by |
|---|---|---|---|
| `ddm-ref` | 3.9.25 | validated **simulator** reference (gym-duckietown) | FJ5, FJ5-R, FJ6, FJ7.1/7.2 parity |
| `ddm-torch` | 3.11.15, torch **2.13.0+cpu**, numpy 2.4.6, CUDA `False` | **actor inference oracle only** | FJ7.4a/7.5a |
| — | none | native Julia (`.npy` reader + own forward pass) | FJ7.1–7.6 runtime |

Julia 1.11.3 (WSL), PythonCall 0.9.25. `ddm-ref` was **not** upgraded,
recreated, or rebound; the PythonCall interpreter is still the `ddm-ref` one.
The torch oracle is reached **out of process** over the JSON-lines protocol, so
it can never displace the validated interpreter. The checkpoints
(`policy.npy`, `policy.pt`) were only ever read.

## WHAT WAS IMPLEMENTED

### FJ7.1 / FJ7.2 — tabular policies (`src/solvers/adapters.jl`)

- `read_npy` — minimal native NumPy v1/v2 reader (little-endian numeric
  arrays, C or Fortran order).
- `QTablePolicy` — one shipped tabular policy (`Float32`, shape
  `(5,5,3,3,4,2,5,7)`), allowed action ids, action table, with the reference
  adapter's validation (shape, finiteness, unique ids in `0:6`).
- `decide` / `QDecision` — the greedy decision plus the evidence the reference
  records with it: full Q-value row, tied action set, top-1/top-2 margin.
- `act`, `POMDPs.action(policy, mdp, s)`, `all_state_indices`,
  `greedy_action_table`, `tie_statistics`.

#### The greedy rule is not `argmax`

Ported verbatim from `duckduck/src/explainability/q_policy_adapter.py`:

```
values         = q_table[index]                  # float32 widened to float64
allowed_values = values[allowed_actions]
best           = max(allowed_values)
ties           = allowed actions with |value - best| <= 1e-12   (atol, rtol=0)
selected       = min(ties)                       # "lowest_action_id"
q_margin       = top1 - top2 over allowed_values
```

The tie test is a **near-tie window**, not equality, so an action can hold the
raw maximum yet lose to a lower-numbered action within 1e-12 of it. This
deterministic rule exists because the reference's *training-time* evaluator
resolves exact ties **randomly** — a path that is not reproducible across
implementations, so the reference itself defines this adapter as the
reproducible evaluation convention. It is preserved as written.

### FJ7.4a / FJ7.5a — the torch oracle

`tools/parity/torch_policy_server.py` (runs in `ddm-torch`) and
`src/backends/torch_policy.jl` (the Julia client,
`TorchPolicyReferenceBackend`).

The oracle imports the **real reference classes** from `duckduck/src/agents/`
(`SquashedGaussianActor`, `DeterministicActor`) and loads the frozen
checkpoints into them, in `eval()` mode under `no_grad`. For SAC it calls the
class's own `actor.sample(x)` and takes the deterministic branch — it does not
re-implement the evaluation rule. It returns every intermediate activation and
exports every actor parameter as `.npy`.

`torch.load(..., map_location="cpu", weights_only=False)` is required and
documented in the server: these checkpoints store numpy arrays alongside the
tensors, so the safe-loader path rejects them. They are the project's own
frozen artefacts, read-only.

Deliberately separate from `ProcessReferenceBackend`, which drives the
simulator in `ddm-ref`:

```
DuckietownDecisionModels.jl
         │ 15-D Float32 observation
         ▼
TorchPolicyReferenceBackend  (out-of-process, ddm-torch)
         ├── SAC actor  ──┐
         └── TD3 actor  ──┴─►  per-layer activations + reference action
```

### FJ7.4b / FJ7.5b — native Julia actors (`src/solvers/actor_adapters.jl`)

`SACActorPolicy` and `TD3ActorPolicy` load the exported `.npy` parameters with
the package's own reader, so **running a learned policy in Julia needs neither
Python nor PyTorch**. `forward` returns the full activation trace (field names
match the oracle's response keys) so a divergence can be localised to the
first layer where it appears.

The clipping asymmetry between the two agents is a real semantic difference
and is preserved, not smoothed over:

```
SAC  SquashedGaussianActor
     backbone: Linear(15,256) → ReLU → Linear(256,256) → ReLU
     heads:    mean = Linear(256,2),  log_std = Linear(256,2)  [clamp -5, 2]
     eval:     tanh(mean) * action_scale + action_bias        — NO clip

TD3  DeterministicActor
     net:      Linear(15,256) → ReLU → Linear(256,256) → ReLU → Linear(256,2)
     eval:     tanh(net(obs)) * action_scale + action_bias
               then the agent ALWAYS applies np.clip(a, low, high)
```

`POMDPs.action(policy, mdp, s)` projects the world state through the validated
observer, encodes it exactly as the reference does, and runs the actor.

### FJ7.6 — the common evaluator (`src/evaluation/metrics.jl`)

`evaluate_policy(mdp, policy; seeds, max_steps)` → `Vector{EpisodeMetrics}`,
`summarize_evaluation`, `compare_policies`. `policy` is anything answering
`POMDPs.action(policy, mdp, state)`, so **no solver family gets its own
environment semantics** — the only thing that differs between them is how an
action is chosen. Every metric is derived from the rollout log, never from a
parallel bookkeeping path.

## FILES

| File | Gate |
|---|---|
| `src/solvers/adapters.jl` | 7.1, 7.2 |
| `src/backends/torch_policy.jl` | 7.4a, 7.5a |
| `tools/parity/torch_policy_server.py` | 7.4a, 7.5a |
| `src/solvers/actor_adapters.jl` | 7.4b, 7.5b |
| `src/evaluation/metrics.jl` | 7.6 |
| `test/test_fj7_tabular.jl`, `test/test_fj7_actors.jl`, `test/test_fj7_evaluation.jl` | tests |

## REFERENCE SOURCE

- `duckduck/src/explainability/q_policy_adapter.py` — `QPolicyAdapter`
- `duckduck/src/agents/` — `SquashedGaussianActor`, `DeterministicActor`,
  `SACAgent.select_action`, `TD3Agent.select_action`
- `duckduck/policies/{q_learning,sarsa}/policy.npy`,
  `duckduck/policies/{sac,td3}/policy.pt` — frozen, read-only

## TESTS AND RESULTS

### FJ7.1 / FJ7.2 — 106 assertions, 0 failures

| Test set | Assertions | Result |
|---|---|---|
| native `.npy` loading (no Python) + validation guards | 17 | PASS |
| greedy rule semantics (tie set, margin, restricted action sets, index guards) | 54 | PASS |
| policy drives the validated MDP | 6 | PASS |
| **9 000-state greedy parity vs the reference adapter**, both solvers | 28 | PASS |
| restricted action set `[1,4,6]` parity, all 9 000 states | 1 | PASS |

Compared against the **real Python object** (`QPolicyAdapter.from_checkpoint`),
and on more than the action id: the natively loaded table is **bitwise equal**
to what NumPy loads; the **action id**, the **tie set** and the **q_margin**
all match on all 9 000 states for both solvers (0 mismatches); action name,
`v_cmd` and `omega_cmd` match on sampled states.

Measured property of the shipped Q-learning table (`tie_statistics`):

```
states          9000
tied            8689     (more than one action within 1e-12 of the best)
zero_margin     8689
argmax_differs     0
min_margin       0.0
```

Most states are tied because large parts of the table were never visited and
remain exactly `0.0`. On these checkpoints the near-tie rule and a plain
first-maximum `argmax` happen to agree everywhere (`argmax_differs = 0`) — the
rule is still implemented as written, because that agreement is a property of
these tables, not of the rule.

### FJ7.4 / FJ7.5 — actor parity

**1 000 observations per actor**: the zero vector, both declared bounds, the
midpoint, four near-bound perturbations, 30 one-hot extremes and 962
pseudo-random points inside the declared observation box; plus 35 observations
taken from a real rollout of the validated MDP.

Provenance is established stage by stage rather than by a blanket tolerance.

**1. Weights cross bit-exactly — measured, not assumed.** On the zero
observation the first layer reduces to its bias; on a one-hot observation it
reduces to one weight column plus the bias, and the other 14 products are
exactly zero, so the sum is exact in IEEE and any weight discrepancy would
appear bitwise. All 16 such checks per actor are **bitwise equal** (`==`, not
`≈`).

**2. Inputs cross bit-exactly.** Every observation the oracle echoes back is
compared with `==` against the Float32 vector Julia sent: 1 000/1 000 exact,
for both actors.

**3. Activations — worst |Δ| over 1 000 observations.**

| Stage | SAC | TD3 |
|---|---|---|
| `layer0_linear` (15→256) | 7.15e-7 | 7.15e-7 |
| `layer1_relu` | 7.15e-7 | 7.15e-7 |
| `layer2_linear` (256→256) | 1.14e-5 | 9.54e-6 |
| `layer3_relu` | 2.86e-6 | 3.81e-6 |
| `mean` / `layer4_linear` | 4.41e-6 | 1.34e-5 |
| `log_std` (and clamped) | 3.93e-6 | — |
| `tanh_mean` / `tanh` | 4.24e-6 | 6.94e-6 |
| `scaled_action` | 4.53e-6 | 5.13e-6 |
| **agent action** | **4.53e-6** | **5.13e-6** (clipped: 5.13e-6) |
| bit-identical actions | 437 / 1 000 | 562 / 1 000 |

Rollout-derived observations (35 each): worst action |Δ| **1.79e-7** (SAC),
**1.67e-6** (TD3).

**Attribution.** The pattern is float32 accumulation order, not a semantic
difference. `layer0` sums 15 products and lands at 7.15e-7 = 1 ulp at
magnitude ~6; `layer2` sums 256 products and grows by roughly an order of
magnitude; the heads inherit that. A transposed weight, a wrong activation or
a misread bias would produce an O(1) discrepancy at the first affected layer,
not a monotone round-off growth. Nothing anomalous appears at any stage.

**4. The decision is materially identical.** `scaled_action` differs by at most
5.13e-6 against action ranges of `v ∈ [0, 0.41]` and `ω ∈ [-1.5, 1.5]` — a
relative error of ~1.3e-5 on `v` and ~1.7e-6 on `ω`, well below the
Float32 resolution the reference itself carries into the simulator.

**5. The clipping asymmetry is confirmed empirically.** For SAC, the
reference's own unclipped and clipped values are equal on every checked
observation — the tanh squash keeps the action inside the box by construction,
which is why SAC omits the clip. For TD3, `clipped = 0` over 1 000
observations: the agent's `np.clip` is a genuine part of its evaluation rule
but never actually binds on these weights. Both facts are reported as
measurements; neither was used to justify dropping the other agent's rule.

### FJ7.3 — 15-D observation adapter

The policy input is validated end to end rather than assumed: `build_continuous_state`
and `encode_continuous_state` were pinned bit-exact in FJ2 (28 963 assertions
over 1 810 cases), the env-dependent observers in FJ3.6, and the whole
continuous projection again live against the running reference in FJ5/FJ5-R
and free-running in FJ6. FJ7.4/7.5 close the loop by feeding that same encoder
into both actors and confirming the oracle receives exactly the bits Julia
produced.

### FJ7.6 — 132 assertions, 0 failures

Harness properties, not policy performance:

| Test set | Checks |
|---|---|
| determinism and seed dependence | same seeds → identical returns, lengths, reasons; different seeds → different episodes |
| bookkeeping consistency | counters bounded by the decision count; exactly one termination reason and it agrees with the event flags |
| braking is recorded as braking | `BRAKE` → `brake_ratio == 1.0`, `FAST_STRAIGHT` → `0.0`, and braking covers less ground |
| aggregation matches the episode list | every summary field recomputed from the episodes |
| the same harness accepts continuous actions | a constant `DuckieAction` policy evaluates identically |
| all four reference policies | run through `compare_policies`, and re-running is bit-identical |

All four shipped policies, same 5 seeds, same 150-decision horizon, same MDP:

| Policy | mean return | mean length | mean \|d\| | mean speed | brake ratio | offroad | full stops / violations | reasons |
|---|---|---|---|---|---|---|---|---|
| q_learning | −33.397 | 129.4 | 0.0442 | 0.1375 | 0.095 | 1 | 1 / 1 | 4 in_progress, 1 offroad |
| sarsa | −32.062 | 127.0 | 0.0371 | 0.1456 | 0.044 | 1 | 1 / 1 | 4 in_progress, 1 offroad |
| sac | +6.245 | 150.0 | 0.0599 | 0.1011 | 0.0 | 0 | 5 / 0 | 5 in_progress |
| td3 | −237.369 | 150.0 | 0.0369 | 0.0298 | 0.031 | 0 | 5 / 0 | 5 in_progress |

`in_progress` here means the harness horizon was reached, not that the
environment terminated. No episode reached the goal tile or the environment's
own step limit within 150 decisions. These numbers are a **comparability
demonstration for the harness**, not a claim about which policy is better —
five seeds is far too few for that, and no such claim is made.

### Full suite

Reproduced with `tools/run_full_suite.sh`, which exists because an inline
heredoc crossing the Windows/WSL shell boundary was expanded at *write* time in
an earlier attempt: `$?` froze into a stale literal `127` and
`JULIA_PYTHONCALL_EXE` was emptied, so the run printed a failure-looking exit
code while the log said the tests had passed. The script is a real file, sets
the environment at run time, and reports Julia's actual status.

WSL Julia 1.11.3, `ddm-ref` active, `ddm-torch` reachable:

```
top-level testsets : 76
assertions passed  : 143 516
failed / errored   : 0
broken             : 0
Pkg verdict        : tests passed
PKGTEST_EXIT       : 0
```

FJ7's own rows in that run:

| Test set | Assertions |
|---|---|
| FJ7.1 native `.npy` loading (no Python involved) | 17 |
| FJ7.1 greedy rule semantics | 54 |
| FJ7.1 policy drives the validated MDP | 6 |
| FJ7.1/7.2 9 000-state greedy parity vs the reference adapter | 28 |
| FJ7.1 restricted action set parity | 1 |
| FJ7.4a/7.5a torch oracle + weight export | 33 |
| FJ7.4b SAC inference parity (layer by layer) | 1 084 |
| FJ7.5b TD3 inference parity (layer by layer) | 1 029 |
| FJ7.4/7.5 observations from real rollouts | 3 |
| FJ7.6 evaluation harness (solver independent) | 97 |
| FJ7.6 reference policies through the shared harness | 35 |

The total is the sum of the Pass column over all 76 top-level testset rows,
measured twice by unrelated parsers (`tools/suite_summary.sh` and
`tools/suite_summary.py`) which agree exactly, with zero unparsed headers and
zero rows where Pass ≠ Total.

## KNOWN DEVIATIONS

1. **Actor activations are not bit-exact** (worst 1.34e-5 at a pre-tanh layer,
   4.53e-6 / 5.13e-6 at the action). Attributed above to float32 accumulation
   order between PyTorch's GEMM and Julia's BLAS. Weights and inputs *are*
   bit-exact, so the deviation is confined to arithmetic ordering. Bit-exact
   neural forward is not claimed and was not demanded.
2. **`weights_only=False`** is required to load these checkpoints. Justified in
   the server comment: the artefacts store numpy arrays alongside tensors and
   are the project's own frozen files.
3. **Weight export path.** Julia reads `.npy` files the oracle writes from the
   checkpoint tensors, so there is exactly one transcription (torch → numpy →
   `.npy`, all lossless for float32) and the bitwise checks above confirm it.
   A native `.pt` reader (option B) is deliberately **not** implemented yet.
4. **`argmax_differs = 0`** on the shipped tables — a property of these
   checkpoints, not a licence to simplify the near-tie rule.

## NEW DEFECTS

None in the port itself. Four defects in the surrounding scaffolding were
found and fixed:

1. **Test authoring.** `observation_suite(950)` yielded 988 observations
   against an assertion of `>= 1000`. The suite now requests 962 random points
   and asserts `== 1000` exactly. No parity assertion was involved.
2. **False exit code in the runner.** An inline heredoc crossing the
   Windows/WSL shell boundary was expanded at write time, so the generated
   script contained a frozen `PKGTEST_EXIT=127` and an empty
   `JULIA_PYTHONCALL_EXE` — a run that had actually passed reported a
   failure-looking status, and the environment variable the run was supposed
   to exercise was unset. Replaced by the checked-in `tools/run_full_suite.sh`.
   This is exactly the failure mode the "never judge success from a truncated
   or unreliable signal" rule exists to catch, and it was caught by that rule.
3. **Undercounted suite totals in earlier documents.** The totals recorded in
   FJ3/FJ5-R/FJ6/FJ7 (80 583 / 82 328 / 82 360 / 82 442) are smaller than the
   sum of their own per-gate figures — FJ2 alone records 49 012 assertions and
   FJ3.7 alone 59 870. The counting command was wrong, not the tests. The
   correct total is now measured by two independent parsers and recorded in
   the README; the historical documents are left as written rather than
   retroactively edited.
4. **Silent single-duck assumption in the new evaluator.** `evaluate_policy`
   initially read `ducks[1]` and `crossings_started[1]`. All shipped configs
   have one duck, so no number changed, but the metric would have silently
   under-reported any multi-duck scenario. It now counts over every duck.

## SCIENTIFIC INTERPRETATION

All four shipped policies now run inside the Julia MDP with their reference
decision rules reproduced, and each rule is pinned against the real Python
object rather than a re-implementation. The tabular half is exact: identical
action, identical tie set, identical margin on the complete 9 000-state space.
The learned half is exact in everything that is representable — weights and
inputs cross bitwise — and differs only by float32 accumulation order, at a
magnitude five orders below the action resolution and localised layer by
layer, so no divergence is left unexplained.

The practical consequence is that the Julia package is now a *complete*
substitute for the reference at evaluation time: it can load and run every
shipped policy with no Python and no PyTorch in the loop, while the Python
environments remain available purely as oracles for verification. That is the
precondition for FJ8: an online planner needs a trustworthy generative model
*and* trustworthy baselines to be compared against, and both now exist under
one evaluator.

## NEXT GATE

FJ8 — MCTS / DPW on `DuckieWorldState`, evaluated through the same
`evaluate_policy` harness so the planner is directly comparable with the four
baselines recorded above.
