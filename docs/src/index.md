# Duckietown.jl

A Duckietown lane-following-with-obstacles MDP, written in Julia as a
[POMDPs.jl](https://github.com/JuliaPOMDP/POMDPs.jl) problem — a native
reimplementation of the decision layer of a
[gym-duckietown](https://github.com/duckietown/gym-duckietown)-based
environment, validated against it decision by decision, including exact
NumPy RNG streams, so a seeded episode reproduces bit for bit.

```@raw html
<img src="https://raw.githubusercontent.com/PannnTastic/Duckietown.jl/main/docs/assets/native_dora_lap.gif"
     alt="DORA completing a lap, drawn by the package's native renderer" style="max-width:100%"/>
```

*DORASolvers.jl under receding horizon completing a `:stop_and_duck_safe`
lap, drawn by the package's own native renderer — solver, physics and
renderer all in Julia. 2× speed; lookalike render, not parity evidence.
Duckietown environment by the [Duckietown Project](https://www.duckietown.org).*

## Install

```julia
using Pkg
Pkg.add(url = "https://github.com/ai-vnv/Duckietown.jl")
```

`using DuckietownDecisionModels` loads no Python, no plotting library and no
solver; the map is embedded in the package.

## Quickstart

```julia
using DuckietownDecisionModels, POMDPs, Random

mdp = DuckietownMDP(scenario_config(:stop_and_duck_safe); action_space = :discrete)
s   = rand(MersenneTwister(1001), initialstate(mdp))
sp, r = @gen(:sp, :r)(mdp, s, FAST_STRAIGHT, MersenneTwister(7))
```

Scenarios (`scenario_config`): `:lane_following` (empty ring),
`:stop_and_duck` (the source-default switches turned on, warts included),
and `:stop_and_duck_safe` (the same world with the yield / stop-approach
reward shaping switched on and the stop sign facing the traffic).

## Running a planner

Any POMDPs.jl solver drives the model through the standard
`solve` / `action` sequence:

```julia
using MCTS
planner = solve(MCTSSolver(n_iterations = 100, depth = 20), mdp)
a = action(planner, s)
```

For a full worked case study with an online SSP solver
([DORASolvers.jl](https://github.com/ai-vnv/DORASolvers.jl)) — formulation,
receding-horizon execution, and tile-by-tile replays of recorded laps — open
the Pluto notebook
[`notebooks/DORA_on_Duckietown.jl`](https://github.com/ai-vnv/Duckietown.jl/blob/main/notebooks/DORA_on_Duckietown.jl);
[`notebooks/Playground.jl`](https://github.com/ai-vnv/Duckietown.jl/blob/main/notebooks/Playground.jl)
is the pick-a-solver starter.

## Where to go next

- **[How it was built](building.md)** — what every file is, why it exists,
  and where each claim's evidence lives, under an explicit provenance rule.
- **[Validation record](validation/README.md)** — the gate-by-gate evidence
  behind every "validated" statement.
- **API** (sidebar) — the exported surface, one page per layer, from the docstrings.

## Acknowledgments

The environment semantics reimplemented here originate from the
[Duckietown Project](https://www.duckietown.org): *the hardware/software
used for the experiments was developed by the Duckietown Project —
www.duckietown.org.* The reference textures and meshes are not
redistributed by this package.
