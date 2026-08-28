### A Pluto.jl notebook ###
# v1.0.3

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ a1b20002-0002-4002-8002-000000000002
begin
    import Pkg
    Pkg.activate(@__DIR__)
    using PlutoUI
    using JSON3
    using FFMPEG
    using Base64
    using DuckietownDecisionModels
    using POMDPs, POMDPTools, DORASolvers
    using Random, Printf
end

# ╔═╡ a1b20001-0001-4001-8001-000000000001
md"""
# DORA on Duckietown.jl

[DORASolvers.jl](https://github.com/ai-vnv/DORASolvers.jl) is an **online
stochastic-shortest-path (SSP) solver**: it operates on a finite tabular SSP
whose transition structure is *known* and whose traversal costs are *learned
online*. [DuckietownDecisionModels.jl](https://github.com/PannnTastic/DuckieMDP)
is a **continuous** driving MDP.

This notebook builds the bridge between them **step by step — every cell
below executes live**: import, model setup, the determinism measurement, the
SSP formulation, solving, and receding-horizon driving. The final sections
replay two full recorded laps through the real gym-duckietown renderer, tile
by tile.
"""

# ╔═╡ a1b20020-0020-4020-8020-000000000020
md"""
## Step 0 · Packages

The `notebooks/` project carries the model (`DuckietownDecisionModels`,
dev-ed at `".."`), the solver (`DORASolvers`, pinned to a public commit), and
the playback utilities.
"""

# ╔═╡ a1b20021-0021-4021-8021-000000000021
md"""
## Step 1 · The model

`scenario_config(:stop_and_duck_safe)` is the world with a crossing duck, a
stop sign facing the traffic, and the safety reward weights switched on
(with the source-default weights of `:stop_and_duck`, passing the duck at
speed costs *nothing* — measured — so nothing would ever yield).
"""

# ╔═╡ a1b20023-0023-4023-8023-000000000023
md"""
## Step 2 · The premise, measured: the transition is deterministic

DORA needs a **known** transition kernel and never estimates one. The bridge
is a fact, not an assumption: given `(s, a)`, the Duckietown decision step
always produces the same successor, whatever RNG you hand it. The original
experiment measured 2940/2940; the cell below re-measures a sample right now
— two *different* RNGs per pair, successors compared exactly:
"""

# ╔═╡ a1b20024-0024-4024-8024-000000000024
determinism = let
    s = rand(MersenneTwister(1002), initialstate(BASE))
    same = 0
    total = 0
    for rep in 1:5, a in ACTS
        total += 1
        r1 = simulate_decision(TR, s, a, MersenneTwister(3rep))
        r2 = simulate_decision(TR, s, a, MersenneTwister(7rep + 1000))
        same += (r1.sp.ego.pos == r2.sp.ego.pos &&
                 r1.sp.ego.angle == r2.sp.ego.angle)
        s = r1.sp
    end
    (identical = same, of = total)
end

# ╔═╡ a1b20025-0025-4025-8025-000000000025
md"""
**$(determinism.identical) / $(determinism.of)** successor pairs identical.
A deterministic kernel is a known kernel, so

```julia
POMDPs.transition(::LapMDP, ls, a) = Deterministic(first(macro_step(ls, a)))
```

is *exact* — no sampled model anywhere.
"""

# ╔═╡ a1b20026-0026-4026-8026-000000000026
md"""
## Step 3 · Lap progress: the goal DORA will chase

The model's 7-tuple observation is purely local lane geometry — it cannot
count laps. So we order the map's 8 drivable tiles into a ring and keep a
**monotone progress counter**: only a forward step to the next ring tile
advances it. (On non-convex maps the ring must be walked along the tiles'
lane-curve connectivity — see the `zigzag_dists` scripts; on this convex
ring an angular sort suffices.)
"""

# ╔═╡ a1b20027-0027-4027-8027-000000000027
begin
    const RING = let m = initial_map(CFG)
        t = collect(drivable_tiles(m))
        cx = sum(first.(t)) / length(t)
        cy = sum(last.(t)) / length(t)
        sort(t; by = p -> atan(p[2] - cy, p[1] - cx))
    end
    const RIX = Dict(t => i for (i, t) in enumerate(RING))
    const NRING = length(RING)

    tile_of(s) = get_grid_coords(s.map, collect(s.ego.pos))
    raw_of(s) = first(get_raw_state(s, SCFG))

    function advance(prog, prev, new)
        new == prev && return prog
        a = get(RIX, prev, 0)
        b = get(RIX, new, 0)
        (a == 0 || b == 0) && return prog
        return mod(b - a, NRING) == 1 ? prog + 1 : prog
    end

    md"""Ring of **$(NRING) tiles**: $(join(string.(RING), " → ")) — one lap = $(NRING) forward steps."""
end

# ╔═╡ a1b20028-0028-4028-8028-000000000028
md"""
## Step 4 · Cost and timescale

**Cost** is DORA's documented default form, `max(c_min, step_cost − reward)`
— the model's own shaped reward supplies the safety semantics, so the
cheapest route *is* the one that yields and stops. Nothing safety-related is
hand-coded into the cost.

**Timescale**: determinism means a breadth-first tabularisation with 0.2 s
actions explores a single forward orbit (measured: 4 states). One SSP action
therefore holds a command for **K = 8 decisions (1.6 s)** — long enough that
different commands reach different discretisation bins and the graph
branches. (Measured on zigzag: K = 8/6/4 all solve; K = 2 collapses to 22
states.)
"""

# ╔═╡ a1b20029-0029-4029-8029-000000000029
begin
    const K = 8
    const C_MIN = 0.05
    step_cost(r) = max(C_MIN, 1.0 - r.reward.total)
end

# ╔═╡ a1b2002a-002a-402a-802a-00000000002a
md"""
## Step 5 · The SSP itself

One cell holds the state type, the macro step, the tiny `MDP` wrapper with
its POMDPs methods, the `key`, and `classify` — deliberately: Pluto
re-defines structs on cell re-run, and methods dispatching on a struct must
be re-defined together with it. (Splitting these across cells is exactly
what broke an earlier version of this notebook.)

- `LapState` = world + ring progress; `key` = (progress, the package's own
  tabular discretisation) — `key` is DORA's documented mechanism for state
  types that do not hash by value.
- `classify` marks `prog ≥ goal` as `:goal` and terminal states as `:crash`.
- `plan_from` is one documented `solve` call (the `examples/04_custom_mdp.jl`
  pattern: explicit `start` / `classify` / `cost` / `key`), with
  `known_costs` left at its default.
"""

# ╔═╡ a1b2002b-002b-402b-802b-00000000002b
md"""
## Step 6 · Solve

Spawn the car (seed 1002 lands it outside the sign's detection corridor, so
the stop interaction happens on camera rather than at birth), set a small
goal, and solve. This cell runs live — a couple of seconds:
"""

# ╔═╡ a1b2002c-002c-402c-802c-00000000002c
begin
    GOAL[] = 2
    s0 = LapState(rand(MersenneTwister(1002), initialstate(BASE)), 0)
    t_solve = @elapsed planner0 = plan_from(s0)
    tab0 = planner0.tab
    V0, pistar0 = optimal_value(tab0)
    sr0, cr0, to0 = outcome_rates(tab0, pistar0)
    md"""
    Tabularised **$(tab0.S) states** in $(round(t_solve; digits = 1)) s.
    In-model, from this spawn to goal $(GOAL[]) tiles:
    optimal cost **$(round(V0[tab0.start]; digits = 2))**,
    success $(sr0), crash $(cr0), timeout $(to0).
    First action chosen: **$(action(planner0, s0))**.
    """
end

# ╔═╡ a1b2002d-002d-402d-802d-00000000002d
md"""
## Step 7 · Drive: receding horizon

The tabularisation is the one **lossy** step: two concrete states 0.19 m /
15° apart can share a key yet behave differently under the same action
(measured), and executing a single plan open-loop crashed — twice. So before
*every* macro action we re-plan from the true current state; the first BFS
expansion is from that state itself, which makes the chosen action's
predicted outcome exact by construction. This cell drives the goal live:
"""

# ╔═╡ a1b2002e-002e-402e-802e-00000000002e
"""Drive `goal` ring tiles under receding horizon; returns trajectory
points, the per-plan log, and the outcome."""
function live_run(goal; seed = 1002, horizon = 40)
    GOAL[] = goal
    rng = MersenneTwister(1)
    ls = LapState(rand(MersenneTwister(seed), initialstate(BASE)), 0)
    pts = [(ls.s.ego.pos[1], ls.s.ego.pos[3], 0.0)]
    log = String[]
    total = 0.0; dec = 0; plans = 0
    for _ in 1:horizon
        POMDPs.isterminal(BASE, ls.s) &&
            return (; outcome = :crash, total, dec, plans, pts, log)
        ls.prog >= goal &&
            return (; outcome = :goal, total, dec, plans, pts, log)
        t0 = time()
        pl = plan_from(ls)
        plans += 1
        a = action(pl, ls)
        push!(log, @sprintf("plan %d: %d states (%.1f s) -> %s",
                            plans, pl.tab.S, time() - t0, string(a)))
        s = ls.s; prog = ls.prog; tile = tile_of(s)
        for _ in 1:K
            dec += 1
            r = simulate_decision(TR, s, a, rng)
            total += step_cost(r)
            prog = advance(prog, tile, tile_of(r.sp))
            tile = tile_of(r.sp); s = r.sp
            push!(pts, (s.ego.pos[1], s.ego.pos[3], r.raw_state.v))
            (r.terminated || r.truncated || prog >= goal) && break
        end
        ls = LapState(s, prog)
    end
    return (; outcome = :timeout, total, dec, plans, pts, log)
end

# ╔═╡ a1b2002f-002f-402f-802f-00000000002f
"""Schematic top-down SVG: grey ring, trajectory coloured by speed
(blue = stopped, yellow = fast), green = start, gold = end."""
function demo_svg(pts; size_px = 420)
    world = 3 * 0.585
    sc = size_px / world
    px(x) = round(x * sc; digits = 1)
    cells = join("""<rect x="$(px(t[1] * 0.585))" y="$(px(t[2] * 0.585))"
        width="$(px(0.585))" height="$(px(0.585))"
        fill="#3a3a3a" stroke="#555"/>""" for t in RING)
    dots = join(begin
        f = clamp(p[3] / 0.24, 0.0, 1.0)
        r = round(Int, 80 + 150f); g = round(Int, 140 + 90f)
        b = round(Int, 255 - 200f)
        """<circle cx="$(px(p[1]))" cy="$(px(p[2]))" r="3"
            fill="rgb($r,$g,$b)"/>"""
    end for p in pts)
    s0_, s1_ = pts[1], pts[end]
    return """<svg width="$size_px" height="$size_px"
        style="background:#222;border-radius:6px">$cells$dots
        <circle cx="$(px(s0_[1]))" cy="$(px(s0_[2]))" r="7"
            fill="none" stroke="#7f7" stroke-width="2"/>
        <circle cx="$(px(s1_[1]))" cy="$(px(s1_[2]))" r="7"
            fill="none" stroke="#fd5" stroke-width="2"/></svg>"""
end

# ╔═╡ a1b20030-0030-4030-8030-000000000030
first_drive = live_run(2)

# ╔═╡ a1b20031-0031-4031-8031-000000000031
HTML("""
<div style="max-width:640px">
  <h4 style="margin:0 0 4px 0">LIVE: $(first_drive.outcome) —
    $(first_drive.dec) decisions, cost $(round(first_drive.total; digits = 2)),
    $(first_drive.plans) receding-horizon plans</h4>
  <div style="display:flex;gap:12px;align-items:flex-start">
    $(demo_svg(first_drive.pts))
    <pre style="font-size:0.75em;color:#aaa;margin:0">$(join(first_drive.log, "\n"))</pre>
  </div>
  <p style="margin:6px 0 0 0;color:#888;font-size:0.8em">
    schematic view, computed by the cells above just now (dot colour =
    speed) — the photorealistic scenes are in the replay section below</p>
</div>
""")

# ╔═╡ a1b20016-0016-4016-8016-000000000016
md"""
### Push it further

Pick a larger goal (8 = a full lap, a few minutes of solving) and tick run:
"""

# ╔═╡ a1b20018-0018-4018-8018-000000000018
@bind goal_tiles Slider(1:8; default = 4, show_value = true)

# ╔═╡ a1b20019-0019-4019-8019-000000000019
@bind run_live CheckBox(default = false)

# ╔═╡ a1b2001a-001a-401a-801a-00000000001a
demores = run_live ? live_run(goal_tiles) : nothing

# ╔═╡ a1b2001b-001b-401b-801b-00000000001b
if demores === nothing
    md"*(tick the box to run — nothing is precomputed)*"
else
    HTML("""
    <div style="max-width:640px">
      <h4 style="margin:0 0 4px 0">LIVE: $(demores.outcome) —
        $(demores.dec) decisions, cost $(round(demores.total; digits = 2)),
        $(demores.plans) receding-horizon plans</h4>
      <div style="display:flex;gap:12px;align-items:flex-start">
        $(demo_svg(demores.pts))
        <pre style="font-size:0.75em;color:#aaa;margin:0">$(join(demores.log, "\n"))</pre>
      </div>
    </div>
    """)
end

# ╔═╡ a1b20014-0014-4014-8014-000000000014
md"""
## The recorded laps, through the real renderer

The two full laps below were produced by exactly the pipeline above (scripts
`dora_lap_safe.jl` and `dora_zigzag.jl`), with every physics substep
recorded and **asserted bit-identical** to `simulate_decision`'s outcome,
exported through `world_to_ref` (the FJ5-validated parity encoder), and
rendered by the *real* gym-duckietown renderer — ego front camera beside the
top-down view, 30 recorded substeps per second = real time. Nothing in them
is a redrawing, and this notebook never regenerates them.

| lap | result |
|---|---|
| `small_loop`, `:stop_and_duck_safe` | lap in 102 decisions / 13 plans (cost 131.39 vs first plan 133.76) — **yields to the crossing duck** (v → 0.000) and **full-stops at the sign** (FULL\\_STOP dec 75, PASSED\\_STOP dec 93, no violation) |
| `zigzag_dists`, 26 tiles | lap 26/26 in 341 decisions / 43 plans (334.28 vs 340.38), lane offset mean 0.051 m |
"""

# ╔═╡ a1b20004-0004-4004-8004-000000000004
@bind lapchoice Select([
    "safe" => "small_loop — :stop_and_duck_safe (duck yield + full stop at the sign)",
    "zigzag" => "zigzag_dists — 26-tile lap, alternating left/right turns",
])

# ╔═╡ a1b20005-0005-4005-8005-000000000005
LAPS = Dict(
    "safe" => (json = "lap_states_safe.json", frames = "safe_frames",
               nring = 8, title = "small_loop :stop_and_duck_safe"),
    "zigzag" => (json = "zigzag_lap.json", frames = "zigzag_frames",
                 nring = 26, title = "zigzag_dists lane following"),
)

# ╔═╡ a1b20006-0006-4006-8006-000000000006
begin
    LAP = LAPS[lapchoice]
    jsonpath = joinpath(@__DIR__, LAP.json)
    framesdir = joinpath(@__DIR__, LAP.frames)
    isfile(jsonpath) || error("""recording not found: $(jsonpath)
        Run the experiment scripts first — see the reproduction section below.""")
    payload = JSON3.read(read(jsonpath))
    states = payload.states
    md"""
    **$(LAP.title)** — outcome `$(payload.outcome)`,
    executed cost $(round(payload.cost; digits = 2))
    vs first plan $(round(payload.cost_model; digits = 2)),
    $(length(states)) physics substeps
    = $(round((length(states) - 1) / 30; digits = 1)) s of model time.
    """
end

# ╔═╡ a1b20007-0007-4007-8007-000000000007
md"""
GIFs are cut per ring tile (segment ``t`` = substeps recorded while
`progress == t − 1`) and cached in `pluto_gifs/` — a committed cache means
the frame PNGs are not needed at all. Default is **one GIF frame per solver
decision**; tick for the smooth 30 Hz version (requires the rendered
frames):
"""

# ╔═╡ a1b20010-0010-4010-8010-000000000010
@bind smooth CheckBox(default = false)

# ╔═╡ a1b20008-0008-4008-8008-000000000008
"""Frame indices per tile segment: segment t = substeps with progress t-1;
the lap-closing frame joins the last segment. With `per_decision`, only the
LAST substep of each decision is kept."""
function tile_segments(states, nring; per_decision = true)
    keep = if per_decision
        last_of_dec = Dict{Int,Int}()
        for s in states
            last_of_dec[s.dec] = max(get(last_of_dec, s.dec, 0), s.i)
        end
        Set(values(last_of_dec))
    else
        Set(s.i for s in states)
    end
    segs = [Int[] for _ in 1:nring]
    for s in states
        s.i in keep || continue
        p = clamp(s.progress, 0, nring - 1)
        s.progress >= nring && (p = nring - 1)
        push!(segs[p + 1], s.i)
    end
    return segs
end

# ╔═╡ a1b20009-0009-4009-8009-000000000009
"""GIF from a list of PNG frames (concat demuxer + palettegen). Cached: an
existing file is returned untouched, so a committed `pluto_gifs/` works
without the frame PNGs."""
function build_gif(fdir, idxs, out; width = 560, dt = 0.0667)
    isfile(out) && return out
    isdir(fdir) || error("""GIF cache miss and no rendered frames at
        $(fdir) — run the render script for this lap first (see the
        reproduction section below).""")
    mkpath(dirname(out))
    list = tempname() * ".txt"
    open(list, "w") do io
        for i in idxs
            println(io, "file '", joinpath(fdir, "f" * lpad(i, 4, '0') * ".png"), "'")
            println(io, "duration ", dt)
        end
    end
    FFMPEG.exe(`-y -loglevel error -f concat -safe 0 -i $list
        -vf "scale=$(width):-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse"
        $out`)
    rm(list; force = true)
    return out
end

# ╔═╡ a1b2000a-000a-400a-800a-00000000000a
begin
    segs = tile_segments(states, LAP.nring; per_decision = !smooth)
    gifdir = joinpath(@__DIR__, "pluto_gifs")
    gmode = smooth ? "s" : "d"
    # 2x real speed in both modes: a real substep is 1/30 s -> shown 1/15 s;
    # a real decision is 0.2 s -> shown 0.1 s
    gdt = smooth ? 0.0667 : 0.1
    gifs = [build_gif(framesdir, seg,
                joinpath(gifdir, "$(lapchoice)_$(gmode)_tile$(lpad(t, 2, '0')).gif");
                dt = gdt)
            for (t, seg) in enumerate(segs)]
    modelabel = smooth ? "per substep, smooth" : "per decision, fast"
    md"**$(length(gifs)) GIFs** ($(modelabel)) ready in `pluto_gifs/`."
end

# ╔═╡ a1b2000c-000c-400c-800c-00000000000c
@bind tile Slider(1:26; default = 1, show_value = true)

# ╔═╡ a1b2000d-000d-400d-800d-00000000000d
"""Measured summary of one segment: decisions, speed range, and (where
recorded) the stop/duck events — numbers, not impressions."""
function segment_caption(states, idxs)
    ss = [s for s in states if s.i in Set(idxs)]
    vs = [s.v for s in ss]
    parts = ["decisions $(minimum(s.dec for s in ss))–$(maximum(s.dec for s in ss))",
             "v $(round(minimum(vs); digits = 3))–$(round(maximum(vs); digits = 3)) m/s"]
    if haskey(first(ss), :event)
        evs = unique(s.event for s in ss if s.event != "")
        isempty(evs) || push!(parts, "events: " * join(evs, ", "))
        ducks = unique(s.duck for s in ss if s.duck != "NONE")
        isempty(ducks) || push!(parts, "duck: " * join(ducks, ", "))
        any(s.sigma for s in ss) && push!(parts, "sigma_stop set")
    end
    return join(parts, " · ")
end

# ╔═╡ a1b2000e-000e-400e-800e-00000000000e
begin
    t_show = clamp(tile, 1, length(gifs))
    cap = segment_caption(states, segs[t_show])
    b64 = base64encode(read(gifs[t_show]))
    framemode = smooth ? "per substep (30 Hz)" : "1 frame per decision (0.2 s)"
    HTML("""
    <div style="max-width:600px">
      <h4 style="margin:0 0 4px 0">Tile $(t_show) / $(LAP.nring) — $(LAP.title)</h4>
      <p style="margin:0 0 8px 0;color:#888;font-size:0.9em">$(cap)</p>
      <img src="data:image/gif;base64,$(b64)" width="560"
           style="border:1px solid #444;border-radius:6px"/>
      <p style="margin:6px 0 0 0;color:#888;font-size:0.8em">
        2x playback · $(framemode) ·
        left: ego front camera · right: top-down view ·
        both from the real gym-duckietown renderer</p>
    </div>
    """)
end

# ╔═╡ a1b20013-0013-4013-8013-000000000013
md"""
## Where this follows the DORA docs, and where it extends them

The solver construction is exactly the documented custom-MDP pattern
(`examples/04_custom_mdp.jl`: explicit `start` / `classify` / `cost` /
`key`), the cost is the documented default form, and `known_costs` stays at
its default (decision-time planning). The **receding-horizon re-solve is an
extension**, not the documented deployment loop: the docs'
`observe!(planner, s, a, cost)` + lazy replan updates *cost statistics*
inside one planner, but our model-reality gap is in the *kernel* — the
determinized successor of an aliased key — which `observe!` cannot repair.
Re-tabularising from the true state can, and the guide itself names plain
"determinize-and-replan" (`correct = false`) as a known operating mode, so
the spirit is documented even though the per-step re-tabularisation is ours.
"""

# ╔═╡ a1b20015-0015-4015-8015-000000000015
md"""
## Reproducing the artefacts

From the repository root, with the sibling reference environment set up
(see `docs/` for the FJ5 bridge requirements):

```
# small_loop with duck + stop sign (safety-shaped scenario)
julia --project=notebooks notebooks/dora_lap_safe.jl      # solve + execute + record
julia --project=notebooks notebooks/export_lap_safe.jl    # states -> reference encoding
python notebooks/render_lap_safe.py                       # real renderer (conda: ddm-ref)
julia --project=notebooks notebooks/make_mp4_safe.jl      # 30 fps real-time video

# zigzag_dists lane-following lap
julia --project=notebooks notebooks/dora_zigzag.jl
python notebooks/render_zigzag.py
julia --project=notebooks notebooks/make_mp4_zigzag.jl

# GIF cache for this notebook (so clones need no frame PNGs)
julia --project=notebooks notebooks/build_gif_cache.jl
```

Diagnostics that shaped the design (all runnable): `probe_zigzag_k.jl`
(macro-length sweep), `probe_zigzag_greedy.jl` (lane followability),
`check_zigzag_curves.py` (map parity vs the reference — bit-identical on all
26 tiles), `audit_stop_geometry.jl` and `audit_sign_orientation.jl` (why the
source's stop-sign placement never fires), `audit_divergence.jl` (where a
single open-loop plan forks from reality).

For a pick-a-solver starter (random / MCTS / DORA on any scenario), see
[`Playground.jl`](./Playground.jl) next to this file.

---
### Artefact honesty

- Steps 0–7 and the *push it further* section compute everything live;
  nothing there is precomputed.
- The replay section never regenerates frames — it only cuts and packages
  recordings whose bit-identity assertions passed during the experiments,
  and `progress` is the experiment's own counter, not recomputed here.
- If an artefact is missing, the notebook stops with instructions — it will
  not silently substitute anything.
"""

# ╔═╡ a1b20022-0022-4022-8022-000000000022
begin
    const CFG = scenario_config(:stop_and_duck_safe)
    const BASE = DuckietownMDP(CFG; action_space = :discrete)
    const TR = BASE.transition
    const SCFG = TR.state_cfg
    const ACTS = collect(POMDPs.actions(BASE))
    md"""
    MDP ready: **$(length(ACTS)) macro actions** ($(join(string.(ACTS), ", "))),
    duck crossing probability $(CFG.duck_controller.p_cross),
    `duck_unsafe` = $(CFG.reward.duck_unsafe),
    `stop_approach_distance` = $(CFG.reward.stop_approach_distance) m.
    """
end

# ╔═╡ a1b20017-0017-4017-8017-000000000017
begin
    const GOAL = Ref(2)     # ring tiles to complete; threaded via a Ref

    struct LapState
        s::DuckieWorldState
        prog::Int
    end

    function macro_step(ls::LapState, a)
        s = ls.s; prog = ls.prog; tile = tile_of(s); c = 0.0
        for _ in 1:K
            r = simulate_decision(TR, s, a, MersenneTwister(1))
            c += step_cost(r)
            prog = advance(prog, tile, tile_of(r.sp))
            tile = tile_of(r.sp); s = r.sp
            (r.terminated || r.truncated || prog >= GOAL[]) &&
                return (LapState(s, prog), c, true)
        end
        return (LapState(s, prog), c, false)
    end

    struct LapMDP <: MDP{LapState,MacroAction} end
    POMDPs.actions(::LapMDP) = ACTS
    POMDPs.discount(::LapMDP) = 1.0
    POMDPs.isterminal(::LapMDP, ls) = POMDPs.isterminal(BASE, ls.s)
    POMDPs.transition(::LapMDP, ls, a) =
        Deterministic(first(macro_step(ls, a)))

    lapkey(ls) = (min(ls.prog, GOAL[]), discretize(raw_of(ls.s)))
    classify(ls) = POMDPs.isterminal(BASE, ls.s) ? :crash :
                   ls.prog >= GOAL[] ? :goal : :normal

    plan_from(from) = solve(DORASolver(
        start = from, classify = classify,
        cost = (ls, a, lsp) -> macro_step(ls, a)[2],
        key = lapkey,
        c_min = K * C_MIN, c_to = 2000.0, c_crash = 1000.0, horizon = 60,
    ), LapMDP())
end

# ╔═╡ Cell order:
# ╟─a1b20001-0001-4001-8001-000000000001
# ╟─a1b20020-0020-4020-8020-000000000020
# ╠═a1b20002-0002-4002-8002-000000000002
# ╟─a1b20021-0021-4021-8021-000000000021
# ╠═a1b20022-0022-4022-8022-000000000022
# ╟─a1b20023-0023-4023-8023-000000000023
# ╠═a1b20024-0024-4024-8024-000000000024
# ╟─a1b20025-0025-4025-8025-000000000025
# ╟─a1b20026-0026-4026-8026-000000000026
# ╠═a1b20027-0027-4027-8027-000000000027
# ╟─a1b20028-0028-4028-8028-000000000028
# ╠═a1b20029-0029-4029-8029-000000000029
# ╟─a1b2002a-002a-402a-802a-00000000002a
# ╠═a1b20017-0017-4017-8017-000000000017
# ╟─a1b2002b-002b-402b-802b-00000000002b
# ╠═a1b2002c-002c-402c-802c-00000000002c
# ╟─a1b2002d-002d-402d-802d-00000000002d
# ╠═a1b2002e-002e-402e-802e-00000000002e
# ╠═a1b2002f-002f-402f-802f-00000000002f
# ╠═a1b20030-0030-4030-8030-000000000030
# ╟─a1b20031-0031-4031-8031-000000000031
# ╟─a1b20016-0016-4016-8016-000000000016
# ╟─a1b20018-0018-4018-8018-000000000018
# ╟─a1b20019-0019-4019-8019-000000000019
# ╠═a1b2001a-001a-401a-801a-00000000001a
# ╟─a1b2001b-001b-401b-801b-00000000001b
# ╟─a1b20014-0014-4014-8014-000000000014
# ╟─a1b20004-0004-4004-8004-000000000004
# ╠═a1b20005-0005-4005-8005-000000000005
# ╟─a1b20006-0006-4006-8006-000000000006
# ╟─a1b20007-0007-4007-8007-000000000007
# ╟─a1b20010-0010-4010-8010-000000000010
# ╠═a1b20008-0008-4008-8008-000000000008
# ╠═a1b20009-0009-4009-8009-000000000009
# ╟─a1b2000a-000a-400a-800a-00000000000a
# ╟─a1b2000c-000c-400c-800c-00000000000c
# ╠═a1b2000d-000d-400d-800d-00000000000d
# ╟─a1b2000e-000e-400e-800e-00000000000e
# ╟─a1b20013-0013-4013-8013-000000000013
# ╟─a1b20015-0015-4015-8015-000000000015
