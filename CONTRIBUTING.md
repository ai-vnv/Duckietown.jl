# Contributing

Thanks for looking. This document is mostly about the conventions that are not
obvious from the code, and about the ones that exist because breaking them
already caused a problem here at least once.

## Getting set up

```bash
git clone https://github.com/PannnTastic/Duckietown.jl
cd Duckietown.jl
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. -e 'using Pkg; Pkg.test()'
```

That works with nothing else installed. It runs 17 test sets and skips 26
files by name, because those compare against the Python reference.

To run everything, put the reference material from
[PannnTastic/DuckieMDP](https://github.com/PannnTastic/DuckieMDP) beside the
package:

```
parent/
├── Duckietown.jl/
└── duckduck/policies/{q_learning,sarsa,sac,td3}/
```

Then the suite runs 201 test sets and 148 855 assertions. Both numbers should
hold; if one moves, say so in the pull request and explain why.

## Rules that matter

**A changed scenario gets a new name.** Never edit a shipped config to fix or
adjust an experiment. `default_config` means "the Python source defaults" and
nothing else; if you need a different world, add it to `SCENARIOS` and
document what it changes. Silently changing a baseline makes every result that
cited it unreproducible, and nobody finds out.

**Recorded evidence is not regenerated.** `artifacts/fj9/reproducibility_manifest.json`
classifies every artefact. Anything marked `PERSISTED_SOURCE` — the FJ8.4b
evaluation, the FJ8.4c decision log, the two search snapshots — is the record
of an experiment that ran. Re-running that experiment produces a *different*
experiment wearing the old name. Check those files, do not rebuild them. Only
`REBUILT` entries (figures, reports, the contract table) are meant to
regenerate.

**The core names no solver, and imports no backend.** `src/` must not mention
`MCTSSolver`, `DPWSolver` or `using MCTS`, and must not import PythonCall or
Makie. Solver and plotting code lives in `ext/`. `using DuckietownDecisionModels`
has to work with none of them installed. Three separate lints enforce this and
they have no allowlist — a doc comment that merely *quotes* a banned token will
fail them, which is deliberate.

**Tests skip, they do not error.** Anything that needs the reference material
must go through `test/reference_guard.jl` and skip cleanly when it is absent.
Before that guard existed the suite errored 69 times on a clean checkout,
meaning it had only ever been runnable by its author.

**Missing is not zero.** `d_stop` is absent on 77 % of logged decisions
because no stop sign was a candidate — a different statement from `d_stop = 0`.
`missing` must survive from the loader to the axis. Likewise `model_calls = 0`
for a tabular policy is a measurement, not an absence.

**One axis per unit.** Metres, radians, m/s and counts do not share a y-axis.
This is not style: a shared axis flattened the lateral offset against the
curvature in one figure and hid a real effect, and it happened again after the
rule was written down.

**Captions are checked.** Publication figures validate their captions against
required and forbidden phrases before rendering, and a figure whose caption
fails is a build failure. This exists because a false sentence sat beside
correct numbers for two gates without anyone noticing. If you add a figure,
add its `CaptionRule`.

## Things that are generated

Do not hand-edit these; edit the source and regenerate.

| File | Generated from | Command |
|---|---|---|
| `examples/quickstart.ipynb` | `examples/quickstart.jl` | `julia --project=. tools/make_notebook.jl` |
| `artifacts/fj9/publication/*` | live data objects | `bash tools/run_publication.sh` |
| `artifacts/fj9/reproducibility_manifest.json` | the repository state | `bash tools/run_manifest.sh` |
| `artifacts/fj9/test_report.json` | the test run | written by `Pkg.test()` |

CI fails if the notebook and its script have drifted apart.

## Tests

New behaviour needs a test that would fail without it. Prefer tests that check
a property against measured data over tests that restate a constant.

The authoritative test count is `artifacts/fj9/test_report.json`, built from
the `Test` result tree. `tools/suite_summary.sh` and `tools/suite_summary.py`
parse terminal output as independent cross-checks and must agree with it. They
are kept precisely because they disagreed once — the reporter undercounted by
40 % on its first run and the disagreement is what caught it.

## Commits and pull requests

- Explain *why*, not just what. The change is visible in the diff; the reason
  is not.
- If you found a defect, say what it was and how it was caught. The status
  documents in `docs/` are written that way and they are the most useful thing
  in the repository.
- Do not add AI assistants as authors, co-authors or contributors. If a tool
  helped you write something, that is between you and the tool.
- CI runs the core on Ubuntu, macOS and Windows against Julia 1.10 and 1.11,
  plus the full suite on Ubuntu. All seven jobs must pass.

## Where things are

```
src/model/          actions, discretizer, state projections, reward
src/dynamics/       DB18 motor model, lane geometry, collision, pedestrians
src/generative/     the transition chain and initial-state distribution
src/interfaces/     POMDPs.jl interface, planner contract, audits
src/evaluation/     metrics, benchmarks, budgets, comparison statistics
src/visualization/  scene geometry, diagnostics, animation, figures (core only)
ext/                Makie, MCTS and PythonCall integrations
tools/              reproducible scripts for every artefact
docs/               one status document per validation gate
```

## Known weak points

Listed in the README and in the manifest. The two most likely to trip you up:

- The six-solver comparison spans two initial-condition sets and three reward
  functions, so cross-family comparisons in those tables are not paired.
  Within-family comparisons are.
- Parity is verified against the Python reference, not against reality. There
  is no validation against a real Duckiebot or a high-fidelity simulator.
