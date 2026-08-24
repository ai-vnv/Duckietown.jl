# FJ5-R — WSL PythonCall Reference Backend

Date: 2026-08-19. Status: **PASSED**.

FJ5-R adds a second, in-process transport to the reference runtime, now that a
Linux Julia exists inside WSL next to the validated `ddm-ref` Python. It does
**not** supersede FJ5: the original out-of-process JSON-lines backend was the
only option from a Windows Julia, it remains fully valid, and — as the
measurements below show — it is now the *numerical* reference of record.

## ENVIRONMENT

| Component | Value |
|---|---|
| Julia (validation) | **1.11.3** `x86_64-linux-gnu`, LLVM 18.1.7, official release, via juliaup **directory override** on the repo |
| Julia (global WSL default) | 1.12.7 — kept, reachable as `julia +release`; used only as a forward-compatibility check |
| Julia (Windows) | 1.10.11 — still works for FJ1–FJ4; FJ5/FJ5-R skip there (documented 1.10 GC crash) |
| Python | **3.9.25** — `/home/pannntastic/miniconda3/envs/ddm-ref/bin/python` |
| PythonCall | **0.9.25** (pinned, see below) |
| numpy / gym / gym_duckietown | 1.20.0 / 0.23.1 / 6.1.34 |
| Binding | `JULIA_CONDAPKG_BACKEND=Null`, `JULIA_PYTHONCALL_EXE=$(which python)` — no CondaPkg env is ever created |

Julia version pinning uses `juliaup override set 1.11.3` **on the repository
directory only**, so the rest of WSL keeps 1.12.7 as default. Manifests are
split per Julia version: `Manifest.toml` (1.10.11, Windows) and
`Manifest-v1.11.toml` (1.11.3, WSL); neither disturbs the other.

### PythonCall is pinned to the validated version 0.9.25

Current PythonCall refuses the reference interpreter outright:

```
ERROR: InitError: Only Python 3.10+ is supported, this is Python 3.9.25
```

Upgrading `ddm-ref` is forbidden (it is the validated reference). 0.9.15,
0.9.20, 0.9.23, 0.9.24 and 0.9.25 were each verified to load Python 3.9.25
successfully, and historically the Python-3.9 era extends past 0.9.25 (the
3.10 minimum lands later in the 0.9.x line). Reproducibility is better served
by standardising on the version actually validated here, so the compat entry
is the exact pin **`PythonCall = "=0.9.25"`**; a later release such as 0.9.28
can be revisited as a compatibility check, never as a blocker.

## WHAT WAS IMPLEMENTED

Backend hierarchy (no breaking API change):

```
AbstractBackend
└── AbstractReferenceBackend
    ├── ProcessReferenceBackend      out-of-process JSON-lines  (FJ5)
    │     const ReferenceBackend = ProcessReferenceBackend   ← compat alias
    └── PythonCallRefBackend         in-process                 (FJ5-R)
          constructed via PythonCallReferenceBackend(...)
```

`PythonCallReferenceBackend` exists in the main package as a stub that raises
an instructive error; the real method arrives with the
`DuckietownPythonCallExt` **package extension**, so PythonCall stays a
**weak dependency**. Verified: after `using DuckietownDecisionModels` the
PythonCall module is *not* loaded, and the full native path
(`DuckietownMDP`, `initialstate`, `gen`) works with no Python installed.

**One source of semantics.** The extension does not re-implement the
reference interaction: it imports the very same `Session` class the
JSON-lines server uses and calls its methods directly. Only the transport
differs.

```
                    tools/parity/reference_server.py :: Session
                          ▲                                  ▲
        JSON over a pipe  │                                  │  pydict, same process
              ProcessReferenceBackend            PythonCallRefBackend
                          ▲                                  ▲
                          └────── world_to_ref / ref_to_world ┘   (shared)
```

`reference_server.py` was refactored so importing it is side-effect free: the
stdout hijack it needs in server mode now lives in `claim_protocol_channel()`,
called only from `main()`. Without that, importing the module in-process
would have redirected Julia's own stdout.

## KEY FINDING — in-process Python does NOT use the reference libm

The transport-equivalence test initially failed on exactly the FJ5
libm-derived chain, which was suspicious: both transports run the same
Python. Measured cause, on the exact `q0` entries the pose readback uses:

| Where `math.atan2(0.45086989117635035, 0.8925896824580856)` runs | Result |
|---|---|
| standalone Python (`ddm-ref`, glibc) | `0.4677396694940821` |
| the same Python **inside** the Julia process (PythonCall) | `0.46773966949408213` |
| Julia's own `atan` | `0.46773966949408213` |

`np.arctan2` behaves identically. Julia loads its libm (OpenLibm) with global
scope, so CPython's and NumPy's `atan2` bind to **Julia's** implementation
instead of glibc's.

Two consequences, both important:

1. **The FJ5 attribution is now proven by construction.** FJ5 inferred that
   the 1-ULP `ego.angle` difference came from `atan2` (OpenLibm vs glibc).
   Running the *same Python code* with Julia's libm loaded reproduces Julia's
   value exactly — causal evidence, not correlation.
2. **`PythonCallReferenceBackend` is not bit-faithful for libm-sensitive
   quantities.** It agrees with native Julia *more* than the true reference
   does. Therefore:

   ```
   ProcessReferenceBackend   →  numerical reference of record (isolated glibc)
   PythonCallRefBackend      →  fast in-process path: semantics, events,
                                termination, state bridge, high-volume runs
   ```

   This **reverses the primary/secondary roles proposed for FJ6**: numerical
   drift must be measured against the process backend, or the port would look
   more faithful than it is. The in-process backend remains ideal for event
   timing, stop/duck semantics and long rollouts where per-step IPC dominates.

## TESTS

`test/test_fj5r_pythoncall.jl` — **66 assertions, 0 failures** (Julia 1.11.3,
WSL, `ddm-ref`):

| Test set | Assertions | Result |
|---|---|---|
| environment identification (interpreter, versions, OS, Julia ≥ 1.11) | 7 | PASS |
| backend construction + interface + state-bridge round trip | 19 | PASS |
| libm interposition (the measurement above, as an assertion) | 2 | PASS |
| transport equivalence, process vs pythoncall | 13 | PASS |
| matched-state parity vs native Julia, discrete (7 actions + 30-step sequence + branch purity) | 16 | PASS |
| matched-state parity vs native Julia, continuous (9 action points incl. clipping + 30-step sequence) | 9 | PASS |

Transport-equivalence result, stated precisely:

- reset from the same seed: **exact**;
- inject a state and read it back: **exact on both transports** (zero diffs,
  no bitwise-only differences either);
- stepping: exact on every discrete field, every event, the termination
  classification, the duckie state, the delay window and `q0`/`v0`; differs
  **only** within `LIBM_DERIVED_FIELDS`, for the interposition reason proven
  above.

Against native Julia the FJ5 structural criterion is applied unchanged
(Rules A–E) and passes with the same profile as FJ5.

Full suite in the canonical environment (WSL Julia 1.11.3, `ddm-ref` active):
**82 328 assertions, 0 failures, 0 errors**, `Pkg.test` exit 0 — verified
from the complete log, including the final summary line. This is the first
run in which FJ1–FJ5 **and** FJ5-R all execute together.

## KNOWN DEVIATIONS

1. The libm interposition described above (in-process only, ≤ 1 ULP at the
   root, confined to `LIBM_DERIVED_FIELDS`).
2. Everything inherited from FJ5: the same chain and the same
   `SIGNED_ZERO_FIELDS` note.
3. `rand(rng, 0:n-1)` differs between Julia 1.10 and 1.12, so
   `initialstate(mdp)` sampled with a plain `MersenneTwister` is
   Julia-version-dependent. The `NumpyPCG64` path is not. Relevant to FJ6
   reproducibility: pin the Julia version or use the NumPy-compatible RNG when
   an exact initial state matters.

## NEW DEFECTS FOUND

- `reference_server.py` hijacked stdout at import time — would have broken any
  in-process use. Now scoped to server mode.
- The comparator entry points were typed to the concrete process backend;
  widened to `AbstractReferenceBackend` so one comparator serves both.
- Non-finite marker dicts were being wrapped by the in-process property view,
  hiding them from the `isa AbstractDict` decoders. Markers are leaves now.

## SCIENTIFIC INTERPRETATION

The port now has two independent live references that agree exactly on all
semantics, and the single numerical difference between them is explained by a
mechanism that was *predicted* by the FJ5 analysis and then demonstrated
directly. That closes the loop on the one open numerical question from FJ5:
the deviation is a property of the mathematical library, not of the ported
model.

## NEXT GATE

**FJ6** — free-running full-episode rollout parity (no per-step
re-injection): accumulated drift, event timing, stop compliance, duck
crossing, termination. Per the finding above, the **process backend** carries
the numerical comparison; the PythonCall backend is used for the
high-volume/semantic passes and as a cross-check.
