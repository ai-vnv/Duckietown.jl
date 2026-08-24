# FJ9.8 — Publication Composites

Date: 2026-08-23.

> A figure must not only reproduce the right numbers. Its caption must
> reproduce the right interpretation.

## STATUS

**PASSED.** FJ9.8a–9.8e closed. Four main figures build from live data
objects, export as PDF/SVG/PNG from a fresh process, and every caption is
validated against required and forbidden claims before the figure is drawn.

## ENVIRONMENT

```
WSL Ubuntu-Baru, juliaup override 1.11.3
Pkg.test with ddm-ref + ddm-torch + MCTS.jl : 193 testsets, 148 758 assertions
publication build (fresh process)           : PUBLICATION_EXIT=0
                                              ENV_MODEL_CALLS=0
                                              PYTHON_MODULES=none
                                              MCTS_LOADED=false
```

## FJ9.8a — INVENTORY BEFORE LAYOUT

19 artefacts, classified before any figure was designed, and probed for
existence rather than assumed:

| Role | Count |
|---|---|
| `MAIN_FIGURE` | 11 |
| `SUPPLEMENTARY` | 5 |
| `DIAGNOSTIC_ONLY` | 3 |
| missing on disk | **0** |

The full table is `artifacts/fj9/publication/inventory.md`.

All four FJ9.7 videos are `SUPPLEMENTARY`. A printed paper has to stand on its
own, so nothing in the main body depends on being able to play a file. A test
asserts every video is supplementary and every main-figure artefact is
assigned to a numbered figure.

## FJ9.8b — COMPOSITE, NOT SCREENSHOT ASSEMBLY

```
WorldScene / ProjectionScene / PolicySlice / SearchSnapshot /
EpisodeDiagnostics / progress bins
        ↓
PanelSpec (payload + provenance + semantics)
        ↓
PublicationComposite (panels + layout + caption + metadata + fingerprint)
        ↓
Makie -> PDF / SVG / PNG
```

`PanelSpec.payload` holds a live data object. Nothing loads a PNG. That is
what makes the PDF and SVG genuinely vector, keeps one font stack and one axis
style across all panels, and — the reason that matters most here — lets the
caption be generated from the same objects the panels were drawn from.

Panels are placed by an explicit `layout` of `(row, col, rowspan, colspan)`.
The default is a two-column grid; figures whose panels differ in natural shape
override it. Figure 4 gives its summary table a full-width short row, because
a wide text table in a square cell is mostly whitespace.

## FJ9.8c — THE FOUR MAIN FIGURES

### Figure 1 — Model and representation

Latent world state; the 15-entry privileged projection; FJ10's per-component
observability classification. The caption states that the projection is a
privileged model state and **not an observation**, and reports that 6 of 15
components are sensor-estimable while 2 (`sigma_stop`, `stop_hold_progress`)
are the agent's own memory with no physical counterpart.

### Figure 2 — Policy structure and ambiguity

Recomputed live from the slice objects, not copied from FJ9.3's prose:

```
cells                     3 721
tied cells                1 174
decisive cells            2 547
distinct tabular states      23
```

The caption carries the three qualifiers that make the panel honest: **feature
space**, **not guaranteed reachable**, and the **fixed context**
(`v = 0.1; tile = STRAIGHT; d_stop = none; sigma_stop = false; duck = NONE`).
Tied cells are marked with a hatch marker rather than a second colour scale,
so the ambiguity survives greyscale.

### Figure 3 — Search behaviour

Both snapshots come from the same frozen state, and `figure_search` **refuses
to build** if the two state fingerprints differ — a test constructs a
mismatched snapshot and asserts the `ArgumentError`.

```
state fingerprint       65ecc28dcdc8e0c5   (identical for both)

MCTS     7 root actions   mean 5.14   median 2   max 24
DPW     24 root actions   mean 1.46   median 1   max  3
```

The caption says both planners leave many candidates lightly explored, and
that MCTS concentrates more evidence on its preferred discrete action while
DPW spreads the same budget across many continuous actions. It does not say
DPW is random, and `"is random"`, `"dpw is worse"` and `"optimal action"` are
all in that figure's forbidden list. The figure demonstrates evidence
dilution; it does not license a claim about search quality.

### Figure 4 — Episode behaviour and failure mechanisms

The frozen six-solver table; TD3's stagnation; DPW's deterioration; realised
generative calls by episode progress.

Every number in the caption is computed from the payload at build time:
`full stop at decision 28`, `123 of 150 decisions in the stop zone`,
`mean speed 0.005 m/s`, `1019 → 835 → 940 → 602 → 195`. Selection rules reach
the reader, not just the seeds:

```
td3     :first_stagnation            seed 1001
dpw@1k  :median_length_terminating   seed 1002
mcts@1k :median_return               seed 1018
```

The TD3 and DPW panels each use **two stacked axes**, m/s over m, with the
stop-zone flag shaded across both. The first version put `v`, `d_stop` and
`sigma_stop` on one axis — a regression of the FJ9.6 unit rule that flattened
both real quantities against a flag pinned at 1.0.

## FJ9.8d — CAPTIONS ARE CHECKED LIKE DATA

Each figure has a `CaptionRule` of required and forbidden phrases.
`render_composite` validates before drawing and **throws** on failure: a
figure whose caption fails its rule is a build failure, not a copy-editing
note.

The `forbidden` list is the half that matters. FJ9.6 found a false sentence
living beside correct numbers, and a rule that only required content could
never have caught it. Figure 4 forbids `"never reaches a stop sign"` and
requires `"reaches the stop sign"`, `"full stop"`, `"never proceeds"`,
`"horizon"` and `"passed_stops = 0"`. It also forbids `"combined score"` and
`"overall ranking"`, carrying FJ8.4b's no-composite-ranking rule into the
figure layer.

A test constructs a composite carrying the old false claim and asserts it is
rejected, and asserts the exported `figure4.caption.txt` contains the
corrected claim and not the old one.

## FJ9.8e — EXPORT

| Figure | PDF | SVG | PNG (4×) |
|---|---|---|---|
| 1 | 15.9 kB | 395 kB | 646 kB |
| 2 | 149 kB | 3.1 MB | 767 kB |
| 3 | 19.7 kB | 459 kB | 686 kB |
| 4 | 24.8 kB | 601 kB | 1.06 MB |

Each also writes `figureN.caption.txt` with the caption and the full
machine-readable provenance block.

```
[x] same composite renders fresh-process
[x] no Python                      PYTHON_MODULES=none
[x] no MCTS required               MCTS_LOADED=false
[x] no environment step            ENV_MODEL_CALLS=0, measured
[x] source fingerprints embedded   in metadata AND in the caption file
[x] text legible at column size    captions wrapped; panel fonts 7-10 pt
[x] no clipped titles/footers
[x] no overlapping legends
[x] all units visible              one axis per unit throughout
[x] raster panels high-resolution  px_per_unit = 4
[x] greyscale/print sanity
```

**Greyscale.** Colour is never the only channel. Action regions carry contour
boundaries; tied cells carry a hatch marker; time series use black with
distinct linestyles; the stop zone is a grey band; events are dashed rules
with text labels; bin counts are printed as `n=`. Sequential surfaces use
viridis, which is monotonic in luminance and therefore survives a greyscale
print as a legible gradient.

## THE TD3 WEIGHTS CACHE

Figure 2 needs the TD3 actor. Extracting it from the frozen `.pt` requires
torch, which would put Python on the publication path. So the extraction is a
separate provisioning step — `tools/run_export_weights.sh`, run once — which
writes the checkpoint's exact arrays to `artifacts/fj9/weights/{td3,sac}/`.
The publication build reads them with the native reader FJ7 validated layer by
layer, and then asserts `PYTHON_MODULES=none`. The `.pt` checkpoints are
read-only inputs and were not written to.

## FILES

| File | Purpose |
|---|---|
| `src/visualization/paper_figure.jl` | inventory, `PanelSpec`, `PublicationComposite`, caption rules, the four figure builders |
| `ext/DuckietownMakieExt.jl` | twelve panel renderers and `render_composite` |
| `test/test_fj98_publication.jl` | 108 assertions across 7 testsets |
| `tools/fj98_publication.jl`, `run_publication.sh` | the fresh-process build |
| `tools/fj98_export_weights.jl`, `run_export_weights.sh` | one-time actor export |
| `artifacts/fj9/publication/` | 12 exports, 4 caption files, the inventory |

## NEW DEFECTS

Four of my own, all in the composite renderer, all found by looking at the
rendered output.

The worst was a regression of my own rule: Figure 4's episode panels put m/s,
m and a 0/1 flag on one axis, exactly the mistake FJ9.6 spent a gate
establishing must not happen. The flag sat at 1.0 and compressed both physical
quantities into the bottom of the panel. Split into stacked per-unit axes with
the flag shaded.

The caption ran off the right edge unwrapped, losing the end of the sentence —
which for a caption means losing the clause that qualifies the claim. Makie
does not reflow, so `wrap_text` now hard-wraps on word boundaries in the core.

The outer grid used a doubled column index left over from when colorbars lived
in the outer layout; combined with content-driven column sizing, every figure
collapsed into a narrow band with wide empty margins. Colorbars now live in
each panel's own nested grid, and columns are sized explicitly.

Two smaller ones: `SearchNode.action` is deliberately `Any` (the snapshot is
solver-neutral), so grouping root visits by action needed `string(...)` rather
than a `String` key; and the action-plane panel read a `action_value` field
that does not exist, instead of filtering on `action isa DuckieAction`.

## SCIENTIFIC INTERPRETATION

The gate's own principle turned out to be testable, which was not obvious at
the start. "The caption must reproduce the right interpretation" sounds like a
style rule; implemented as required/forbidden phrase lists checked at build
time, it is a constraint with the same standing as a numerical tolerance, and
it fails the build the same way.

That matters because of what FJ9.6 found. The defect there was not a wrong
number — every number was right — but a sentence that misdescribed them, and
it survived because nothing in the pipeline treated prose as checkable. Figure
4 now cannot be built with that sentence in it, and cannot be built without
the corrected one.

## NEXT GATE

FJ9.9 — final reproducibility closure: fresh environment, full test run, every
publication artefact re-rendered, fingerprints verified, no optional-backend
leaks, documentation links checked, and the structured test reporter that has
been carried as tooling debt since FJ9.0.
