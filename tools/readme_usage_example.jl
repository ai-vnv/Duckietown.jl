# Verifies, by running it, the end-to-end usage example shown in the README
# ("Usage: from `using Duckietown` to an animation"). The snippet between the
# BEGIN/END markers is kept byte-identical to the README code blocks.
#
# Run: julia --project=. tools/readme_usage_example.jl
# Outputs land in a temp dir, not the repository.

pkgpath = normpath(joinpath(@__DIR__, ".."))
outdir = mktempdir()
cd(outdir)
@info "README usage example" outdir

# Mirror the README install: a fresh project with the package plus the two
# optional extras the example uses (MCTS for planning, CairoMakie to draw).
using Pkg
Pkg.activate(joinpath(outdir, "duckie"))
Pkg.develop(path = pkgpath)
Pkg.add(["POMDPs", "MCTS", "CairoMakie"])

# ── BEGIN README SNIPPET ────────────────────────────────────────────────────
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
# ── END README SNIPPET ──────────────────────────────────────────────────────

@assert isfile("episode.png") && filesize("episode.png") > 10_000
@assert isfile("episode.gif") && filesize("episode.gif") > 50_000
@info "OK" decisions = length(traj) episode_return = round(total, digits = 1) png = filesize("episode.png") gif = filesize("episode.gif")
println("README_EXAMPLE_OK decisions=$(length(traj)) return=$(round(total, digits=1))")
