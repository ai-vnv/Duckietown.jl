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

# ╔═╡ c2d30002-0002-4002-8002-000000000002
begin
    import Pkg
    Pkg.activate(@__DIR__)
    using PlutoUI
    using DuckietownDecisionModels
    using POMDPs, POMDPTools, MCTS, DORASolvers
    using Random, Printf
end

# ╔═╡ c2d30001-0001-4001-8001-000000000001
md"""
# Duckietown.jl Playground

Pick a **scenario** and a **solver**, tick *run*, and watch what happens —
everything here executes live in this notebook. No Python, no pre-rendered
artefacts; the trajectory view is a deliberately schematic top-down plot.
For the full DORA case study with the photorealistic renderer, see
[`DORA_on_Duckietown.jl`](./DORA_on_Duckietown.jl) next to this file.

| solver | what it is | how it drives here |
|---|---|---|
| `random` | uniform over the 7 macro actions | baseline — expect it to leave the road |
| `MCTS` | Monte-Carlo tree search ([MCTS.jl](https://github.com/JuliaPOMDP/MCTS.jl)), plans per decision | the README's `solve` / `action` pattern, verbatim |
| `DORA` | online SSP solver ([DORASolvers.jl](https://github.com/ai-vnv/DORASolvers.jl)) on the tabularised model, receding horizon | the case-study formulation: ring-progress goal, reward-derived costs |

Scenarios (`scenario_config`): `:lane_following` (empty ring),
`:stop_and_duck` (duck + stop sign, but the safety reward weights are the
Python source defaults — all zero, so nothing asks the car to yield), and
`:stop_and_duck_safe` (same world with the yield / stop-approach shaping
switched on — the interesting one).
"""

# ╔═╡ c2d30003-0003-4003-8003-000000000003
# All model/solver definitions in ONE cell — Pluto redefines structs on
# re-run, and POMDPs method extensions must be redefined with the struct
# they dispatch on.
begin
    const K = 8                       # decisions one DORA macro action holds
    const C_MIN = 0.05
    const SEED_WORLD = 7              # controller rng inside the world state

    pg_cost(r) = max(C_MIN, 1.0 - r.reward.total)

    """Everything the runners need for one scenario, built once per choice."""
    function pg_setup(scenario::Symbol)
        cfg = scenario_config(scenario)
        mdp = DuckietownMDP(cfg; action_space = :discrete)
        tr = mdp.transition
        acts = collect(POMDPs.actions(mdp))
        ring = let m = initial_map(cfg)
            t = collect(drivable_tiles(m))
            cx = sum(first.(t)) / length(t)
            cy = sum(last.(t)) / length(t)
            sort(t; by = p -> atan(p[2] - cy, p[1] - cx))
        end
        return (; cfg, mdp, tr, acts, ring,
                rix = Dict(t => i for (i, t) in enumerate(ring)),
                nring = length(ring))
    end

    pg_tile(S, s) = get_grid_coords(s.map, collect(s.ego.pos))
    pg_raw(S, s) = first(get_raw_state(s, S.tr.state_cfg))

    function pg_advance(S, prog, prev, new)
        new == prev && return prog
        a = get(S.rix, prev, 0)
        b = get(S.rix, new, 0)
        (a == 0 || b == 0) && return prog
        return mod(b - a, S.nring) == 1 ? prog + 1 : prog
    end

    """One recorded decision step, shared by every solver."""
    function pg_record!(rec, S, r)
        raw = r.raw_state
        ev = r.events.full_stop ? "FULL_STOP" :
             r.events.stop_violation ? "STOP_VIOLATION" :
             r.events.passed_stop ? "PASSED_STOP" :
             r.events.collision_duck ? "DUCK_COLLISION" : ""
        push!(rec, (x = r.sp.ego.pos[1], z = r.sp.ego.pos[3], v = raw.v,
                    reward = r.reward.total, event = ev,
                    duck = string(raw.duck)))
    end

    # ── per-decision solvers (random / MCTS): step the real MDP directly ──
    function pg_run_decision(S, pick; seed, horizon)
        rng = MersenneTwister(1)
        s = rand(MersenneTwister(seed), initialstate(S.mdp))
        rec = NamedTuple[]
        prog = 0
        tile = pg_tile(S, s)
        for t in 1:horizon
            POMDPs.isterminal(S.mdp, s) && return (:terminal, prog, rec)
            a = pick(s, t)
            r = simulate_decision(S.tr, s, a, rng)
            pg_record!(rec, S, r)
            prog = pg_advance(S, prog, tile, pg_tile(S, r.sp))
            tile = pg_tile(S, r.sp)
            s = r.sp
            prog >= S.nring && return (:lap, prog, rec)
            (r.terminated || r.truncated) &&
                return (r.terminated ? :crash : :timeout, prog, rec)
        end
        return (:horizon, prog, rec)
    end

    # ── DORA: the case-study SSP adaptor, compact ─────────────────────────
    struct PGState
        s::DuckieWorldState
        prog::Int
    end
    const PGS = Ref{Any}(nothing)      # active setup, for the MDP methods
    const PGGOAL = Ref(2)

    function pg_macro(ps::PGState, a)
        S = PGS[]
        s = ps.s; prog = ps.prog; tile = pg_tile(S, s); c = 0.0
        for _ in 1:K
            r = simulate_decision(S.tr, s, a, MersenneTwister(1))
            c += pg_cost(r)
            prog = pg_advance(S, prog, tile, pg_tile(S, r.sp))
            tile = pg_tile(S, r.sp); s = r.sp
            (r.terminated || r.truncated || prog >= PGGOAL[]) &&
                return (PGState(s, prog), c, true)
        end
        return (PGState(s, prog), c, false)
    end

    struct PGMDP <: MDP{PGState,MacroAction} end
    POMDPs.actions(::PGMDP) = PGS[].acts
    POMDPs.discount(::PGMDP) = 1.0
    POMDPs.isterminal(::PGMDP, ps) = POMDPs.isterminal(PGS[].mdp, ps.s)
    POMDPs.transition(::PGMDP, ps, a) = Deterministic(first(pg_macro(ps, a)))

    pg_key(ps) = (min(ps.prog, PGGOAL[]), discretize(pg_raw(PGS[], ps.s)))
    pg_classify(ps) = POMDPs.isterminal(PGS[].mdp, ps.s) ? :crash :
                      ps.prog >= PGGOAL[] ? :goal : :normal

    function pg_run_dora(S; seed, goal, horizon = 40)
        PGS[] = S
        PGGOAL[] = goal
        rng = MersenneTwister(1)
        ps = PGState(rand(MersenneTwister(seed), initialstate(S.mdp)), 0)
        rec = NamedTuple[]
        plans = 0
        for _ in 1:horizon
            POMDPs.isterminal(S.mdp, ps.s) && return (:crash, ps.prog, rec, plans)
            ps.prog >= goal && return (:goal, ps.prog, rec, plans)
            planner = solve(DORASolver(
                start = ps, classify = pg_classify,
                cost = (a1, a2, a3) -> pg_macro(a1, a2)[2],
                key = pg_key,
                c_min = K * C_MIN, c_to = 2000.0, c_crash = 1000.0,
                horizon = 60,
            ), PGMDP())
            plans += 1
            a = action(planner, ps)
            s = ps.s; prog = ps.prog; tile = pg_tile(S, s)
            for _ in 1:K
                r = simulate_decision(S.tr, s, a, rng)
                pg_record!(rec, S, r)
                prog = pg_advance(S, prog, tile, pg_tile(S, r.sp))
                tile = pg_tile(S, r.sp); s = r.sp
                (r.terminated || r.truncated || prog >= goal) && break
            end
            ps = PGState(s, prog)
        end
        return (:timeout, ps.prog, rec, plans)
    end

    # ── schematic top-down SVG, coloured by speed ─────────────────────────
    function pg_svg(S, rec; size_px = 400)
        world = 3 * 0.585
        sc = size_px / world
        px(x) = round(x * sc; digits = 1)
        cells = join("""<rect x="$(px(t[1] * 0.585))" y="$(px(t[2] * 0.585))"
            width="$(px(0.585))" height="$(px(0.585))"
            fill="#3a3a3a" stroke="#555"/>""" for t in S.ring)
        dots = join(begin
            f = clamp(p.v / 0.24, 0.0, 1.0)
            r = round(Int, 80 + 150f); g = round(Int, 140 + 90f)
            b = round(Int, 255 - 200f)
            """<circle cx="$(px(p.x))" cy="$(px(p.z))" r="3"
                fill="rgb($r,$g,$b)"/>"""
        end for p in rec)
        return """<svg width="$size_px" height="$size_px"
            style="background:#222;border-radius:6px">$cells$dots</svg>"""
    end

    """Numbers, not impressions: outcome, progress, return, events, speeds."""
    function pg_summary(outcome, prog, rec, S)
        isempty(rec) && return "no decisions taken"
        vs = [p.v for p in rec]
        evs = unique(p.event for p in rec if p.event != "")
        ducks = unique(p.duck for p in rec if p.duck != "NONE")
        join(["outcome **$(outcome)** after $(length(rec)) decisions",
              "ring progress $(prog)/$(S.nring)",
              @sprintf("sum of rewards %.2f", sum(p.reward for p in rec)),
              @sprintf("speed %.3f–%.3f m/s", minimum(vs), maximum(vs)),
              "events: " * (isempty(evs) ? "none" : join(evs, ", ")),
              "duck classes seen: " * (isempty(ducks) ? "none" : join(ducks, ", ")),
             ], " · ")
    end
end

# ╔═╡ c2d30004-0004-4004-8004-000000000004
md"""
## Choose
"""

# ╔═╡ c2d30005-0005-4005-8005-000000000005
@bind scenario Select([
    :stop_and_duck_safe => "stop_and_duck_safe — duck + sign, safety shaping ON",
    :stop_and_duck => "stop_and_duck — duck + sign, source-default (zero) safety weights",
    :lane_following => "lane_following — empty ring",
])

# ╔═╡ c2d30006-0006-4006-8006-000000000006
@bind solver Select([
    "dora" => "DORA (receding horizon, macro actions)",
    "mcts" => "MCTS (plans every decision)",
    "random" => "random actions (baseline)",
])

# ╔═╡ c2d30007-0007-4007-8007-000000000007
md"""
episode seed — different seeds spawn the car in different lane poses:
"""

# ╔═╡ c2d30008-0008-4008-8008-000000000008
@bind seed Slider(1001:1020; default = 1002, show_value = true)

# ╔═╡ c2d30009-0009-4009-8009-000000000009
md"""
budget — decisions for random/MCTS, **ring tiles to complete** for DORA
(2 ≈ 10 s of solving; 8 = a full lap, minutes):
"""

# ╔═╡ c2d3000a-000a-400a-800a-00000000000a
@bind budget Slider(1:120; default = 40, show_value = true)

# ╔═╡ c2d3000b-000b-400b-800b-00000000000b
md"""
MCTS iterations per decision (only used by MCTS):
"""

# ╔═╡ c2d3000c-000c-400c-800c-00000000000c
@bind mcts_iters Slider(10:10:200; default = 50, show_value = true)

# ╔═╡ c2d3000d-000d-400d-800d-00000000000d
@bind go CheckBox(default = false)

# ╔═╡ c2d3000e-000e-400e-800e-00000000000e
result = if !go
    nothing
else
    S = pg_setup(scenario)
    if solver == "random"
        rrng = MersenneTwister(42)
        out, prog, rec = pg_run_decision(S,
            (s, t) -> rand(rrng, S.acts); seed, horizon = budget)
        (; S, out, prog, rec, note = "uniform random over $(length(S.acts)) actions")
    elseif solver == "mcts"
        planner = solve(MCTSSolver(n_iterations = mcts_iters, depth = 20,
                                   rng = MersenneTwister(9)), S.mdp)
        out, prog, rec = pg_run_decision(S,
            (s, t) -> action(planner, s); seed, horizon = budget)
        (; S, out, prog, rec,
           note = "MCTS, $(mcts_iters) iterations per decision, depth 20")
    else
        goal = clamp(budget, 1, 8)
        out, prog, rec, plans = pg_run_dora(S; seed, goal)
        (; S, out, prog, rec,
           note = "DORA receding horizon, goal $(goal) ring tiles, $(plans) plans")
    end
end

# ╔═╡ c2d3000f-000f-400f-800f-00000000000f
if result === nothing
    md"*(tick the box to run — nothing is precomputed)*"
else
    HTML("""
    <div style="max-width:640px">
      <h4 style="margin:0 0 4px 0">$(uppercase(String(solver))) on
        :$(scenario)</h4>
      <p style="margin:0 0 8px 0;color:#888;font-size:0.9em">$(result.note)</p>
      <div style="display:flex;gap:12px;align-items:flex-start">
        $(pg_svg(result.S, result.rec))
        <p style="font-size:0.85em;color:#ccc;max-width:200px">
          $(pg_summary(result.out, result.prog, result.rec, result.S))</p>
      </div>
      <p style="margin:6px 0 0 0;color:#888;font-size:0.8em">
        schematic top-down (dot colour = speed: blue stopped → yellow fast);
        just computed in this notebook</p>
    </div>
    """)
end

# ╔═╡ c2d30010-0010-4010-8010-000000000010
md"""
## Things worth trying

- **The reward is the safety spec:** run DORA on `:stop_and_duck` and then
  on `:stop_and_duck_safe` with the same seed and budget 8. Same solver,
  same world — only the reward weights differ, and only the second one
  yields and stops. (The measured demonstration of this is the recorded
  small\\_loop lap in the case-study notebook.)
- **Budget matters for MCTS:** at 10 iterations it drifts; raise the slider
  and watch the trajectory tighten.
- **Random is the honest floor:** it usually leaves the road within a few
  decisions — that is what "the task is nontrivial" looks like.
- To instrument planner cost in generative calls (the unit the README
  quotes), wrap the model in `InstrumentedMDP` — see the README's
  *Running a planner* section.
"""

# ╔═╡ Cell order:
# ╟─c2d30001-0001-4001-8001-000000000001
# ╠═c2d30002-0002-4002-8002-000000000002
# ╠═c2d30003-0003-4003-8003-000000000003
# ╟─c2d30004-0004-4004-8004-000000000004
# ╟─c2d30005-0005-4005-8005-000000000005
# ╟─c2d30006-0006-4006-8006-000000000006
# ╟─c2d30007-0007-4007-8007-000000000007
# ╟─c2d30008-0008-4008-8008-000000000008
# ╟─c2d30009-0009-4009-8009-000000000009
# ╟─c2d3000a-000a-400a-800a-00000000000a
# ╟─c2d3000b-000b-400b-800b-00000000000b
# ╟─c2d3000c-000c-400c-800c-00000000000c
# ╟─c2d3000d-000d-400d-800d-00000000000d
# ╠═c2d3000e-000e-400e-800e-00000000000e
# ╟─c2d3000f-000f-400f-800f-00000000000f
# ╟─c2d30010-0010-4010-8010-000000000010
