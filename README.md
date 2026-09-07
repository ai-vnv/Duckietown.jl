# Duckietown.jl

[![CI](https://github.com/ai-vnv/Duckietown.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/ai-vnv/Duckietown.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/ai-vnv/Duckietown.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/ai-vnv/Duckietown.jl)
[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://ai-vnv.github.io/Duckietown.jl/dev/)
[![V&V](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fai-vnv%2FDuckietown.jl%2Fgh-pages%2Fvnv-badge.json)](https://github.com/ai-vnv/Duckietown.jl/blob/main/.vnvspec/spec.yaml)

A Duckietown lane-following-with-obstacles MDP, written in Julia as a
[POMDPs.jl](https://github.com/JuliaPOMDP/POMDPs.jl) problem.

It is a native reimplementation of the Python environment in
[DuckieMDP](https://github.com/PannnTastic/DuckieMDP), validated
against it decision by decision — including exact NumPy RNG streams, so a
seeded episode reproduces bit for bit on the platform the evidence was
produced on: x86-64 under Julia 1.10/1.11, where CI keeps it pinned. On
platforms that fuse multiply-add (Apple Silicon on any Julia; every
architecture from Julia 1.12) a few derived read-back chains drift at the
last bits (measured in CI: at most 80 ULP, 1.8e-14 relative; tolerated and
documented in the suite) — dynamics, RNG streams and every
discrete decision remain identical.

![DORA completing a lap, drawn by the package's native renderer](https://raw.githubusercontent.com/PannnTastic/Duckietown-artifacts/main/docs/assets/native_dora_lap.gif)

*[DORASolvers.jl](https://github.com/ai-vnv/DORASolvers.jl) under receding
horizon completing a `:stop_and_duck_safe` lap — yielding to the crossing
duck on the way — drawn by
[`render_native`](#native-lookalike-renderer): solver, physics and renderer
all in Julia. The full formulation is the
[`notebooks/DORA_on_Duckietown.jl`](notebooks/DORA_on_Duckietown.jl) case
study. 2× speed; lookalike render, not parity evidence. Duckietown
environment by the [Duckietown Project](https://www.duckietown.org).*

```julia
using Duckietown, POMDPs, Random

mdp = DuckietownMDP(scenario_config(:stop_and_duck); action_space = :discrete)
s   = rand(MersenneTwister(1001), initialstate(mdp))
sp, r = @gen(:sp, :r)(mdp, s, FAST_STRAIGHT, MersenneTwister(7))
```

**No Python is involved.** `using Duckietown` loads no Python, no
plotting library and no solver; the map is embedded in the package.

---

## Install

Registration in the General registry is pending. Until it merges, install from
the repository:

```julia
using Pkg
Pkg.activate("duckie")            # a project of its own — see the note below
Pkg.add(url = "https://github.com/ai-vnv/Duckietown.jl")
Pkg.add("POMDPs")
```

Once the registration merges, the same two lines become:

```julia
Pkg.add("Duckietown")
Pkg.add("POMDPs")
```

> **Use a separate project.** The `[compat]` bounds here are deliberately
> narrow (`PythonCall = "=0.9.25"`, `Makie = "0.24"`) because those are the
> versions the parity results were produced with. In a shared environment the
> resolver will try to move your other packages to satisfy them.

Julia 1.10 or newer. Three optional extras, each enabling one extension:

| Add | Gives you |
|---|---|
| `CairoMakie` | figures and animations — renders headless, so notebooks and CI work |
| `MCTS` | run MCTS / DPW planners against the model |
| `PythonCall` | the in-process bridge to the Python reference, for parity work |

---

## Usage — from `using Duckietown` to an animation

The whole loop in one place: build the MDP, solve it, drive an episode, draw
it, animate it. This exact snippet is executed by
[`tools/readme_usage_example.jl`](tools/readme_usage_example.jl) in a fresh
project (so it also verifies the install instructions above); the numbers
quoted below are that run's measured output.

```julia
using Duckietown, POMDPs, Random

# 1. Build the MDP: a stop sign, and a duck that crosses the road
mdp = DuckietownMDP(scenario_config(:stop_and_duck); action_space = :discrete)
s   = rand(MersenneTwister(1001), initialstate(mdp))

# 2. Solve — any POMDPs.jl solver works; MCTS.jl shown here
using MCTS
planner = solve(MCTSSolver(n_iterations = 100, depth = 20,
                           exploration_constant = 5.0,
                           rng = MersenneTwister(2026)), mdp)

# 3. Drive one episode
rng, traj, total = MersenneTwister(7), NTuple{2,Float64}[], 0.0
while !isterminal(mdp, s) && length(traj) < 150
    a = action(planner, s)                      # plan from the current state
    global s, r = @gen(:sp, :r)(mdp, s, a, rng) # step the world
    global total += r
    push!(traj, (s.ego.pos[1], s.ego.pos[3]))
end

# 4. Draw the episode: the world at the final state, trajectory overlaid
using CairoMakie
save("episode.png", render_world(mdp, s; trajectory = traj,
    title = "MCTS episode, return $(round(total, digits = 1))"))

# 5. Animate: play back a recorded episode from the committed decision log
log = load_decision_log(joinpath(pkgdir(Duckietown),
    "artifacts", "fj8", "enriched", "decisions.csv"))
seq = animation_sequence(log, "td3", 1001)      # one episode, by solver + seed
sw  = static_world(mdp, s)
render_animation(sw, seq, "episode.gif")
```

Measured outcome of step 3, so expectations are set honestly: at a
100-iteration budget this episode ends off-road after 34 decisions with
return −67.6. That is the point of the benchmark — planning budget is the
variable, and the [budget study](artifacts/fj8/budget_study.md) measures the
whole curve. The animation in step 5 plays back a recorded TD3 episode from
the committed FJ8.4c decision log (playback of evidence, validated in FJ9.7 —
it never re-runs the experiment).

---

## Documentation

Everything below this point used to live here and now has a page of its own
on the [documentation site](https://ai-vnv.github.io/Duckietown.jl/dev/):

- [Guide](https://ai-vnv.github.io/Duckietown.jl/dev/guide/) — the model and
  its two privileged projections, scenarios, drawing (including the native
  lookalike renderer), running planners, the DORA case study, reproducing
  the recorded experiments, and the known limitations.
- [How it was built](https://ai-vnv.github.io/Duckietown.jl/dev/building/) —
  what every file is, why it exists, and where each claim's evidence lives.
- [Validation records](https://ai-vnv.github.io/Duckietown.jl/dev/validation/) —
  one status document per gate (FJ2–FJ10): what was measured, what
  deviated, what was left undone.
- [API](https://ai-vnv.github.io/Duckietown.jl/dev/api/model/) — per-layer
  reference.

The recorded media (rendered laps, animations, diagnostics, publication
figures) live in the
[companion repository](https://github.com/PannnTastic/Duckietown-artifacts), so
`Pkg.add` downloads source, tests and fixtures only.

---

## Acknowledgments

The environment semantics reimplemented here originate from the
[Duckietown Project](https://www.duckietown.org). In the wording their
software terms ask for: *the hardware/software used for the experiments was
developed by the Duckietown Project — www.duckietown.org.* The animated
media in this repository depict the Duckietown simulation environment, with
attribution to the Duckietown Project; the reference textures and meshes
themselves are **not** redistributed here (the optional native renderer
loads them from your own gym-duckietown installation).

---

## License & intellectual property

- **Source code:** [MIT](LICENSE). Parts of `src/dynamics/` and
  `src/visualization/` are a documented line-by-line port of
  [gym-duckietown](https://github.com/duckietown/gym-duckietown) (pinned
  6.1.34); those portions are released under MIT with the Duckietown
  Project's written permission.
- **Assets, maps, meshes, textures:** never bundled in this package. They are
  loaded at runtime from the user's own gym-duckietown installation (see
  `DUCKIETOWN_ASSETS`), remain the intellectual property of the
  [Duckietown Project](https://duckietown.com), and stay subject to the
  [Duckietown software terms](https://duckietown.com/sw-license/) — including
  their reservation of commercial use.
- The animated media (in the
  [companion repository](https://github.com/PannnTastic/Duckietown-artifacts))
  are renders of Duckietown environments and carry attribution in their
  captions.
