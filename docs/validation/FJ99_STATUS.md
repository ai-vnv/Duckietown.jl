# FJ9.9 — Final Reproducibility Closure

Date: 2026-08-24.

> Can someone else take this repository from a clean state and reproduce the
> FJ9 evidence, without a dependency leak, a stale artefact, or a claim the
> data no longer supports?

## STATUS

**PASSED.** FJ9.9a–9.9f all closed and verified.

```
WSL Ubuntu-Baru, juliaup override 1.11.3
Pkg.test              : 200 testsets, 148 823 assertions, 0 failures, exit 0
structured report     : 200 / 148 823 / 0   (authoritative)
tools/suite_summary.sh: 200 / 148 823 / 0   (cross-check)
tools/suite_summary.py: 200 / 148 823 / 0   (cross-check)
publication build     : PUBLICATION_EXIT=0, ENV_MODEL_CALLS=0, no Python, no MCTS
clean depot run       : 192 / 148 728 / 0, exit 0, no optional module loaded
isolation matrix      : ISOLATION_ALL_MATCH=true
documentation audit   : 0 issues
src import lint       : 0 issues
artifact ledger       : 15 entries, 0 missing
```

## FJ9.9b — THE STRUCTURED REPORTER (DEBT PAID)

Standing technical debt since FJ9.0. The terminal parsers dropped rows three
separate times — minute-format durations (`3m17.5s`), negative durations when
the WSL clock stepped backwards, and failing rows, which carry three count
columns instead of two. Each was caught only because two independently written
parsers disagreed.

The suite now runs under one parent testset. `test/reporter.jl` walks the
`Test` result tree and writes `artifacts/fj9/test_report.json`:

```json
{"schema": "fj99.reporter.1", "testsets": 200, "assertions": 148823,
 "failures": 0, "errors": 0, "broken": 0}
```

plus a per-testset detail array. The JSON is **authoritative**; both parsers
are retained as cross-checks and were updated for the new single-header
output, where an unindented root row precedes the per-testset rows and would
double every total if counted.

**The three-way check earned its keep immediately.** The reporter's first run
reported **88 213** assertions against the parsers' **148 758**. A finished
`DefaultTestSet` discards its individual `Pass` objects and keeps only the
`n_passed` counter, so recursing while looking for `Pass` results found
nothing below the top level. Had the reporter simply replaced the parsers, the
suite would have quietly lost 40 % of its assertions and still reported
success.

A synthetic fixture (nested passes, one fail, one error, one broken) verifies
the counts: `testsets=2 assertions=5 fails=1 errors=1 broken=1`.

## FJ9.9c — ARTIFACT LEDGER

Fifteen artefacts, each with a status that determines what "reproduce" means
for it:

| Status | Meaning | Examples |
|---|---|---|
| `PERSISTED_SOURCE` | the recorded evidence itself; **checked, never rebuilt** | FJ8.4b CSV, FJ8.4c `decisions.csv`, the two search snapshots |
| `PROVISIONED_FROZEN_INPUT` | extracted once from a read-only upstream checkpoint | `artifacts/fj9/weights/{td3,sac}` |
| `REBUILT` | regenerated from source data on every run | the four figures, the inventory, the decision contract, the test report |

The distinction is load-bearing. A `PERSISTED_SOURCE` that a rebuild
overwrites is not reproducibility, it is data loss: re-running the FJ8.4b
protocol produces a *different experiment* wearing the old one's name.

**Figure identity is semantic, not byte-level.** The four composite
fingerprints reproduced exactly across independent rebuilds
(`f4745f6fc31caf3c`, `7021ef589040d223`, `3947ab2b41697680`,
`8643d1703fa92018`) and are now written into each `figureN.caption.txt`. PDF
files are not byte-identical between runs because the vector backend embeds a
producer timestamp; that is recorded as a known limitation rather than
papered over by normalising the file.

## FJ9.9d — DOCUMENTATION AUDIT

`documentation_audit` walks every `.md` and `.jl` in the repository and checks
three things: no superseded claim outside its allowlist, every markdown link
resolves, and every backticked repository path exists.

It found **five issues on its first run**, which is the point of writing it as
code rather than as a checklist:

1. `README.md` quoted the banned TD3 phrase verbatim while describing the
   guard against it. Rephrased, so the guard stays strict on the most-read
   document rather than being allowlisted there.
2. and 3. `README.md` linked the FJ0 repository audit as a path inside this
   repository, when it lives in the `duckduck` repository — a dead link and a
   missing path, flagged twice by two different checks. Corrected to point
   across the repository boundary.
4. **`src/evaluation/comparison.jl` still carried the original wrong
   framing** in the comment explaining the compliance denominator: "when a
   solver never reaches a stop sign the rate is `nothing`". FJ9.6 corrected
   the FJ8 prose but not the source comment that generated it. It now reads
   "when a solver completes no stop encounter", with an explicit note that
   completing no encounter is not the same as never arriving.
5. `test/test_fj96_diagnostics.jl` quotes the phrase in the test that guards
   it — allowlisted, since that is the guard.

Two synthetic fixtures verify the audit still fires: a file containing the
claim, and a file with a dead link.

## FJ9.9e — DEPENDENCY ISOLATION

`core_fingerprint` hashes the formulation — model type, discount, action
semantics, the three state types' field structure, the reward config and the
state config. It must not move when an optional package loads:

```
core only              8ccb3d672ae7d281
core + Makie           8ccb3d672ae7d281
core + MCTS            8ccb3d672ae7d281
core + Makie + MCTS    8ccb3d672ae7d281
ISOLATION_ALL_MATCH = true
```

The continuous formulation is a *different* fingerprint, which is the control:
a hash that never changes proves nothing.

`source_import_audit` lints `src/` for optional-backend imports, skipping
docstring bodies so that a documented `using …` example is not mistaken for a
real import.

**This gate's own file broke two older guards.** FJ8.1 and FJ8.5 lint `src/`
for solver vocabulary, and the docstring I wrote for `source_import_audit`
spelled out the exact token they ban. The guards were right and the prose was
rewritten — they are stricter than the new lint and have no allowlist, which
is the correct trade.

## FJ9.9f — MANIFEST

`artifacts/fj9/reproducibility_manifest.json` (50 kB) records the git commit,
Julia version, core fingerprint, isolation matrix, the full test report, the
artefact ledger with fingerprints, both search snapshots with their validity,
the four figure fingerprints, both audits' results, and the known limitations.

Six limitations are recorded rather than omitted:

| Item | Why it is deferred |
|---|---|
| `controller_rng` shared by design | measured at 8.96 % of gen allocation to copy, never written to, enforced by `rng_frozen` |
| native `gen` cost | ~100 µs / 213 KiB / 4 929 allocations; planner budgets are quoted in generative calls because of it |
| no structural state merging in MCTS | transpositions are not merged; tree statistics count nodes, not distinct states |
| Duckiematrix not integrated | no live high-fidelity backend |
| observation and belief not implemented | the formulation is an MDP over a privileged state |
| PDF byte-reproducibility | vector exports embed a producer timestamp |

Omitting a deferred decision from a reproducibility statement is the same
class of error as a stale claim, so the manifest names them.

## FJ9.9a — CLEAN-ENVIRONMENT BOOTSTRAP

`tools/fj99_clean_bootstrap.sh` runs with a throwaway `JULIA_DEPOT_PATH` and
`--startup-file=no`, so nothing the user's global depot happens to provide can
satisfy an undeclared dependency.

```
INSTANTIATE_EXIT=0        registry added, project instantiated from scratch
CLEAN_TEST_EXIT=0         tests passed
                          192 testsets, 148 728 assertions, 0 failures
depot size                991 MB (throwaway, removed afterwards)

LOADED_PythonCall=false   LOADED_CondaPkg=false   LOADED_PyCall=false
LOADED_MCTS=false         LOADED_Makie=false      LOADED_CairoMakie=false
CLEAN_CORE_FINGERPRINT=8ccb3d672ae7d281
```

The core fingerprint from the clean depot is **identical** to the one from the
development environment, and `using DuckietownDecisionModels` loads no
optional module in a depot where several are installed.

### The 192 / 200 delta, explained rather than waved through

Eight testsets did not run, accounting for all 95 missing assertions:

```
FJ5-R environment identification                                 -7
FJ5-R backend construction + interface                          -19
FJ5-R libm interposition (why the transports differ)             -2
FJ5-R transport equivalence (process vs pythoncall)             -13
FJ5-R matched-state parity vs native Julia (discrete)           -16
FJ5-R matched-state parity vs native Julia (continuous)          -9
FJ7.1/7.2 9000-state greedy parity vs the reference adapter     -28
FJ7.1 restricted action set parity                               -1
```

Every one is a comparison against the **live Python reference**, which needs
the validated `ddm-ref` interpreter. A clean depot installs PythonCall and lets
CondaPkg provision a *fresh* pixi environment, which is not `ddm-ref`, so the
parity tests skip themselves as designed.

The important half of the result: **no testset that ran produced a different
count.** The 192 shared testsets are identical assertion for assertion. The
clean environment reproduces the entire native suite exactly; what it cannot
do is re-check parity against a reference stack it does not have.

**Finding worth stating:** a clean `Pkg.test()` pulls a full Python toolchain
and a 991 MB depot, because `PythonCall` is a test dependency. `using
DuckietownDecisionModels` remains pure Julia — first-time *testing* is simply
not a lightweight operation.

**Defect found and fixed:** the bootstrap wrote its structured report to the
canonical path, overwriting the development run's numbers with the clean
run's. A clean run is a different run; it now writes to its own path via
`DDM_TEST_REPORT`.

## FILES

| File | Purpose |
|---|---|
| `test/reporter.jl` | the structured reporter |
| `test/runtests.jl` | runs under one parent testset, writes the report |
| `src/interfaces/reproducibility.jl` | ledger, doc audit, core fingerprint, import lint, limitations |
| `test/test_fj99_closure.jl` | 60 assertions across 7 testsets |
| `tools/fj99_manifest.jl`, `run_manifest.sh` | isolation matrix + manifest |
| `tools/fj99_clean_bootstrap.sh` | clean-depot bootstrap |
| `tools/suite_summary.{sh,py}` | cross-checks, updated for the new output |

## NEW DEFECTS

Three, all in this gate's own work, all caught by the checks it added.

The reporter undercounted by 40 % on its first run, caught by parser
disagreement. A `MethodError` in `core_fingerprint`: it called `length` on the
action space, which the continuous formulation's box does not support — the
fingerprint claimed to cover both formulations and only worked for one. And my
FJ9.9b test initially asserted `failures == 0` against a report file written
by the *previous* run; asserting a verdict from a different run is the same
mistake as a renderer reporting data the experiment never produced. That test
now checks internal consistency only, and the current run's verdict is
`PKGTEST_EXIT`.

## ACCEPTANCE

```
[x] clean depot instantiate + test      INSTANTIATE_EXIT=0, CLEAN_TEST_EXIT=0
[x] structured reporter authoritative and verified
[x] both legacy parsers agree with reporter   200 / 148 823 / 0
[x] all publication artifacts rebuild         fingerprints reproduce exactly
[x] source/figure fingerprints validate
[x] no stale TD3 claim remains                audit: 0 issues
[x] docs/artifact links resolve
[x] optional deps do not alter core fingerprint
[x] publication build uses no Python/MCTS/environment execution
[x] manifest generated
[x] known limitations explicitly recorded     6 items
```

## SCIENTIFIC INTERPRETATION

Every check added in this gate found something on its first execution: the
reporter found its own undercount, the documentation audit found the original
wrong framing still living in `src/evaluation/comparison.jl` two gates after
the prose was corrected, and the older FJ8 guards found this gate's own
docstring. That is the argument for executable audits over checklists — not
that they are more rigorous in principle, but that a checklist cannot fail.

The `comparison.jl` finding is the one worth keeping. FJ9.6 corrected the
sentence in `docs/validation/FJ8_STATUS.md` and stopped there, because that was where the
wrong claim had been *read*. It was still sitting in the code comment that had
generated it, ready to be re-read by anyone maintaining the compliance metric.
A correction applied where a claim was found, rather than everywhere it lives,
is only half a correction.

## NEXT

FJ9.0–FJ9.9 closed. The visualization and reproducibility layer is complete,
and the Julia baseline is stable enough to serve as the comparison target for
a live high-fidelity backend — the DM0–DM4 Duckiematrix track.
