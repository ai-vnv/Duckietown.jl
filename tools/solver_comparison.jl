# FJ8.4b — run the frozen six-solver comparison and write the artefacts.
#
# Run as a script, not from the test suite: at ~1 000 generative calls per
# decision and a 150-decision horizon, the two planners alone need roughly
# twenty minutes of wall time over twenty seeds. The test suite exercises the
# comparison CODE on a small configuration; this produces the numbers.
#
#     tools/jl.sh tools/solver_comparison.jl
#
# Everything it uses is frozen in configs/planning/evaluation.yaml and
# configs/planning/seeds.yaml. Nothing here reads a result and changes a
# parameter.

using DuckietownDecisionModels
using POMDPs
using Random
using YAML
using MCTS

const ROOT = pkgdir(DuckietownDecisionModels)
const DUCK = joinpath(ROOT, "..", "duckduck")

seeds_cfg = planning_seed_config()
proto = YAML.load_file(joinpath(ROOT, "configs", "planning",
    "evaluation.yaml"))

EPISODES = seeds_cfg.evaluation.episodes
HORIZON = Int(proto["protocol"]["horizon"])
PLANNER_RNG = seeds_cfg.evaluation.planner

@info "FJ8.4b frozen protocol" episodes = length(EPISODES) horizon = HORIZON planner_rng =
    PLANNER_RNG  primary_gen = proto["protocol"]["primary_operating_point"]

qcfg = joinpath(DUCK, "policies", "q_learning", "training_config.yaml")
ccfg = joinpath(DUCK, "policies", "sac", "training_config.yaml")
tcfg = joinpath(DUCK, "policies", "td3", "training_config.yaml")

"""Run one solver under the frozen protocol and package the result."""
function run_solver(name, family, mdp, policy; horizon=HORIZON,
    seeds=EPISODES, config="")
    @info "running $name" family
    t0 = time_ns()
    res = evaluate_planner(mdp, policy; seeds=seeds, max_steps=horizon)
    @info "  done" seconds = round((time_ns() - t0) / 1e9; digits=1) mean_return =
        round(sum(e -> e.ret, res.episodes) / length(res.episodes); digits=2)
    return SolverRun(name, family, res.episodes, res.cost, res.decisions,
        config)
end

runs = SolverRun[]

# --- learned baselines -----------------------------------------------------
dmdp = InstrumentedMDP(DuckietownMDP(qcfg; action_space=:discrete))
for (name, dir) in ("q_learning" => "q_learning", "sarsa" => "sarsa")
    f = joinpath(DUCK, "policies", dir, "policy.npy")
    isfile(f) || continue
    push!(runs, run_solver(name, "tabular (learned)", dmdp,
        QTablePolicy(f; solver=Symbol(dir)); config="$dir/policy.npy"))
end

torch_ok = torch_policy_available()
if torch_ok
    b = TorchPolicyReferenceBackend()
    try
        for (name, cfgpath, ctor) in (("sac", ccfg, SACActorPolicy),
            ("td3", tcfg, TD3ActorPolicy))
            meta = torch_policy_init!(b, name)
            pol = ctor(String(meta["weights_dir"]))
            m = InstrumentedMDP(DuckietownMDP(cfgpath; action_space=:continuous))
            push!(runs, run_solver(name, "deep (learned)", m, pol;
                config="$name/policy.pt"))
        end
    finally
        close(b)
    end
else
    @warn "ddm-torch unavailable: SAC and TD3 omitted from this run"
end

# --- online planners at the primary operating point ------------------------
mcts_spec = proto["solvers"]["mcts"]
dpw_spec = proto["solvers"]["dpw"]

mcts_solver(n) = MCTSSolver(n_iterations=n, depth=Int(mcts_spec["depth"]),
    exploration_constant=Float64(mcts_spec["exploration_constant"]),
    rng=MersenneTwister(PLANNER_RNG), reuse_tree=false)
dpw_solver(n) = DPWSolver(n_iterations=n, depth=Int(dpw_spec["depth"]),
    exploration_constant=Float64(dpw_spec["exploration_constant"]),
    enable_action_pw=true, enable_state_pw=false,
    k_action=Float64(dpw_spec["k_action"]),
    alpha_action=Float64(dpw_spec["alpha_action"]),
    rng=MersenneTwister(PLANNER_RNG), keep_tree=false)

mcts_mdp = InstrumentedMDP(DuckietownMDP(qcfg; action_space=:discrete))
dpw_mdp = InstrumentedMDP(DuckietownMDP(ccfg; action_space=:continuous))

push!(runs, run_solver("mcts@1k", "discrete online planning", mcts_mdp,
    solve(mcts_solver(Int(mcts_spec["n_iterations"])), mcts_mdp);
    config="MCTSSolver n=$(mcts_spec["n_iterations"])"))
push!(runs, run_solver("dpw@1k", "continuous online planning", dpw_mdp,
    solve(dpw_solver(Int(dpw_spec["n_iterations"])), dpw_mdp);
    config="DPWSolver n=$(dpw_spec["n_iterations"])"))

# --- protocol check BEFORE any result is read ------------------------------
check = check_paired_protocol(runs)
@info "FJ8.4b paired protocol verified" check

# --- planner budget sensitivity (planner-only, separate) -------------------
sens = proto["sensitivity"]
sens_seeds = EPISODES[1:Int(sens["seeds"])]
sens_horizon = Int(sens["horizon"])
sens_runs = SolverRun[]
for (i, target) in enumerate(sens["targets"])
    n_m = Int(sens["mcts_iterations"][i])
    n_d = Int(sens["dpw_iterations"][i])
    push!(sens_runs, run_solver("mcts@$(target)", "discrete online planning",
        mcts_mdp, solve(mcts_solver(n_m), mcts_mdp); seeds=sens_seeds,
        horizon=sens_horizon, config="n=$n_m"))
    push!(sens_runs, run_solver("dpw@$(target)", "continuous online planning",
        dpw_mdp, solve(dpw_solver(n_d), dpw_mdp); seeds=sens_seeds,
        horizon=sens_horizon, config="n=$n_d"))
end

# --- artefacts -------------------------------------------------------------
outdir = joinpath(ROOT, "artifacts", "fj8")
mkpath(outdir)

write(joinpath(outdir, "six_solver_episodes.csv"), episode_csv(runs))
write(joinpath(outdir, "planner_sensitivity_episodes.csv"),
    episode_csv(sens_runs))

planners = filter(r -> occursin("planning", r.family), runs)
learned = filter(r -> occursin("learned", r.family), runs)
pairs = PairedDifference[]
for p in planners, l in learned
    push!(pairs, paired_difference(p.name, p.episodes, l.name, l.episodes, :ret))
end
length(planners) == 2 && push!(pairs, paired_difference(
    planners[1].name, planners[1].episodes, planners[2].name,
    planners[2].episodes, :ret))

open(joinpath(outdir, "six_solver_comparison.md"), "w") do io
    println(io, "# FJ8.4b — six-solver comparison\n")
    println(io, "Frozen protocol: `configs/planning/evaluation.yaml`. ",
        length(EPISODES), " evaluation seeds (disjoint from the development ",
        "seeds used for calibration), horizon ", HORIZON,
        ", one shared evaluator, planner RNG ", PLANNER_RNG, ".\n")
    println(io, "**There is no combined ranking.** Task performance and ",
        "computational cost are separate blocks and are not comparable to ",
        "each other.\n")
    println(io, "## Task performance\n\n```")
    print(io, task_table(runs))
    println(io, "```\n")
    println(io, "## Safety, stops and ducks\n")
    println(io, "`compliance` = (stop encounters − violations) / stop ",
        "encounters, where an encounter is a `passed_stop` event. `n/a` ",
        "means no stop sign was ever reached, which is not the same as ",
        "perfect compliance.\n\n```")
    print(io, safety_table(runs))
    println(io, "```\n")
    println(io, "## Computational cost\n")
    println(io, "`gen/act = 0` is measured and means the policy performs no ",
        "generative planning; `n/a` would mean unmeasured. Learned policies ",
        "do tabular or network inference, planners run hundreds to thousands ",
        "of model simulations — the latency columns are not like-for-like ",
        "and must not be read as a single efficiency axis.\n\n```")
    print(io, cost_table(runs))
    println(io, "```\n")
    println(io, "## Paired per-seed differences in return\n")
    println(io, "Every solver ran the same seeds, so these are differences on ",
        "identical initial conditions rather than a comparison of independent ",
        "group means. With n = ", length(EPISODES),
        " these are descriptive; no superiority claim is made.\n\n```")
    print(io, paired_table(pairs))
    println(io, "```\n")
    println(io, "## Planning cost by position in the episode\n")
    println(io, "FJ8.4a measured a 3.8x spread in generative cost across ",
        "states. This is where a high p95 comes from.\n")
    for r in planners
        println(io, "### ", r.name, "\n\n```")
        print(io, position_table(cost_by_episode_position(r.decisions)))
        println(io, "```\n")
    end
    println(io, "## Planner budget sensitivity (planner-only)\n")
    println(io, "Not additional rows of the six-solver table: a separate ",
        "study on ", length(sens_seeds), " seeds at horizon ", sens_horizon,
        ".\n\n```")
    print(io, task_table(sens_runs))
    println(io, "```\n\n```")
    print(io, cost_table(sens_runs))
    println(io, "```")
end

@info "FJ8.4b artefacts written" dir = outdir solvers = length(runs) sensitivity_runs =
    length(sens_runs)
println("FJ8.4B_SOLVERS=", length(runs))
println("FJ8.4B_SEEDS=", length(EPISODES))
println("FJ8.4B_OK=true")
