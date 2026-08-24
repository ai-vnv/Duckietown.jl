# FJ8.4c — Evaluation Artefact Enrichment

Date: 2026-08-23.

> **FJ8.4c does not supersede FJ8.4b. It reproduces the frozen evaluation
> protocol while enriching the observational record.**

## STATUS

**PASSED.** 120 of 120 episodes reproduced **exactly** — zero mismatches
across 22 metric fields per episode — with 16 522 per-decision rows added.
The original artefact was not overwritten.

## WHAT WAS FROZEN, AND WHAT CHANGED

Unchanged: the 20 evaluation seeds and their solver×seed pairing, horizon 150,
the tabular checkpoints and their deterministic near-tie rule, the SAC/TD3
checkpoints, the MCTS and DPW configurations, the planner RNG protocol, the
environment configuration, reset and termination semantics, the reward, and
the evaluator's metric definitions.

Changed: **more of what already happened is written down.**

## THE LOGGER IS OBSERVATIONAL

`DecisionTrace` is assembled from values the evaluator already held at the
moment the decision completed — the `TransitionResult`, the state the action
was taken from, the action, and the `PlanningDiagnostics` the decision already
carried. It calls no observer, no policy and no transition of its own, and
takes no draw from the evaluator's rng.

That restriction is enforced by test rather than by intent:

- every `EpisodeMetrics` field is identical with and without tracing;
- the model-call counter reports the **same** consumption either way, so a
  logger that slipped in an extra `gen` would be caught.

FJ9.5b is why this matters: a second `action` call there produced a second
search, and the snapshot validator caught it. The same mistake in a logger
would have produced a *different experiment* wearing the old one's name.

Row *k* means: from this pose, the policy took this action, and this is what
the decision produced. The pose is pre-transition (what the policy saw); the
projections, reward and events are post-transition (what the chain computed).
Both are stated in the schema rather than implied.

## THE AGGREGATE-BACK CHECK

```
enriched decision rows -> reaggregate_episodes -> compare to six_solver_episodes.csv
```

```
episodes checked        120 / 120
fields per episode      22
mismatches              0
exact                   true
episode fingerprint     enriched == original
original artefact       untouched
decision rows           16 522
```

No tolerance was used anywhere. The protocol is deterministic and frozen, so
exact reproduction was the expectation; a float that failed to match would
have been treated as evidence that the run differed, not as a reason to reach
for `isapprox`.

Two metrics needed care because they come from the successor **state** rather
than from the projections — `crossings_started` and the duck-active flag. They
are recorded explicitly; had they been left out, the aggregate-back check
would have failed for a reason that has nothing to do with reproduction.

## THREE-LEVEL FINGERPRINTS

`artifacts/fj8/enriched/fingerprints.json`:

| Level | Covers |
|---|---|
| `experiment_fingerprint` | horizon, seeds, planner RNG, frozen solver configs |
| `episode_fingerprint` | solver, seed, return, length, reason — computed for **both** runs and asserted equal |
| `decision_log_fingerprint` | the complete enriched rows |
| `original_artifact_fingerprint` | the FJ8.4b file's own content hash |

## PER-DECISION PLANNING COST

The thing FJ8.4b could only average is now in the record. Mean generative
calls per decision, by position within the episode:

```
solver        0-20%   20-40%   40-60%   60-80%   80-100%
dpw@1k         1019      835      940      602       195
mcts@1k        1019     1021     1025      914      1018
q_learning        0        0        0        0         0
```

These reproduce the FJ8.4b position table exactly, which is a further check
that the enriched run is the same run. DPW's compute collapses as its
trajectories deteriorate — the endogenous-budget finding, now visible
decision by decision rather than inferred from an aggregate.

`q_learning = 0` is measured, not missing: a tabular policy consumes no
generative calls.

## FILES

| File | Purpose |
|---|---|
| `src/evaluation/metrics.jl` | `DecisionTrace`, `decision_csv`, `reaggregate_episodes` |
| `tools/fj84c_enrich.jl`, `tools/run_enrich.sh` | the replication |
| `test/test_fj84c_enrichment.jl` | 203 assertions |
| `artifacts/fj8/enriched/decisions.csv` | 16 522 rows × 51 columns |
| `artifacts/fj8/enriched/episodes_reaggregated.csv` | |
| `artifacts/fj8/enriched/reproduction_report.json` | the verdict |
| `artifacts/fj8/enriched/fingerprints.json` | |

## NEW DEFECTS

One, in this gate's own tooling, and it was expensive: the comparison loop
accumulated `checked`, `mismatches` and `reagg` at **top level**, where Julia's
soft-scope rule makes each a fresh local. The script died with
`UndefVarError` *after* completing the twenty-minute six-solver replication,
so the entire run was wasted and had to be repeated.

This is the third tool script in the project to hit the same trap, after
`tools/fj8_native_check.jl` and `tools/fj9_render_check.jl`. The rule is now
recorded: in `tools/*.jl`, accumulate inside a function, never at top level.

## FULL SUITE

```
top-level testsets : 162
assertions passed  : 148 009
failed / errored   : 0
Pkg verdict        : tests passed
PKGTEST_EXIT       : 0
```

## NEXT

FJ9.6 — diagnostic time series, drawn from `decisions.csv`. There is now no
reason for a renderer to run the environment.
