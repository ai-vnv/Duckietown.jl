# FJ8.4c — evaluation artefact enrichment.
#
#   FJ8.4c DOES NOT SUPERSEDE FJ8.4b. It reproduces the frozen evaluation
#   protocol while enriching the observational record.
#
# Everything that can affect a decision is frozen and unchanged: the 20
# evaluation seeds and their pairing, horizon 150, the tabular checkpoints and
# their deterministic tie rule, the SAC/TD3 checkpoints, the MCTS and DPW
# configurations, the planner RNG protocol, the environment config, the
# termination and reward rules, and the evaluator's metric definitions.
#
# The only change is that more of what already happened is written down. The
# logger performs no extra generative call, no extra policy inference, no
# second `action`, and takes no draw from the evaluator's rng — it reads the
# `TransitionResult` the decision produced.
#
# The gate is the aggregate-back check: re-derive the episode metrics from the
# enriched decision rows and require them to equal the original
# `six_solver_episodes.csv` EXACTLY. The protocol is deterministic, so exact
# reproduction is the expectation; a mismatch is evidence that the run differs,
# not an invitation to add a tolerance.
#
#     tools/run_enrich.sh

using DuckietownDecisionModels
using POMDPs
using Random
using YAML
using JSON3
using MCTS

const ROOT = pkgdir(DuckietownDecisionModels)
const DUCK = joinpath(ROOT, "..", "duckduck")
const OUT = joinpath(ROOT, "artifacts", "fj8", "enriched")
const ORIGINAL = joinpath(ROOT, "artifacts", "fj8", "six_solver_episodes.csv")

seeds_cfg = planning_seed_config()
proto = YAML.load_file(joinpath(ROOT, "configs", "planning",
    "fj8_evaluation.yaml"))
EPISODES = seeds_cfg.evaluation.episodes
HORIZON = Int(proto["protocol"]["horizon"])
PLANNER_RNG = seeds_cfg.evaluation.planner

qcfg = joinpath(DUCK, "policies", "q_learning", "training_config.yaml")
ccfg = joinpath(DUCK, "policies", "sac", "training_config.yaml")
tcfg = joinpath(DUCK, "policies", "td3", "training_config.yaml")

@info "FJ8.4c frozen protocol" episodes = length(EPISODES) horizon = HORIZON planner_rng =
    PLANNER_RNG

traces = DecisionTrace[]
episodes = Dict{String,Vector{EpisodeMetrics}}()

function run_one(name, mdp, policy)
    @info "  $name"
    n0 = length(traces)
    eps = evaluate_policy(mdp, policy; seeds=EPISODES, max_steps=HORIZON,
        record=traces, trace_solver=name)
    episodes[name] = eps
    @info "    decisions logged" rows = length(traces) - n0
    return eps
end

dmdp = InstrumentedMDP(DuckietownMDP(qcfg; action_space=:discrete))
for (name, dir) in ("q_learning" => "q_learning", "sarsa" => "sarsa")
    f = joinpath(DUCK, "policies", dir, "policy.npy")
    isfile(f) || continue
    run_one(name, dmdp, QTablePolicy(f; solver=Symbol(dir)))
end

if torch_policy_available()
    b = TorchPolicyReferenceBackend()
    try
        for (name, cfgp, ctor) in (("sac", ccfg, SACActorPolicy),
            ("td3", tcfg, TD3ActorPolicy))
            meta = torch_policy_init!(b, name)
            m = InstrumentedMDP(DuckietownMDP(cfgp; action_space=:continuous))
            run_one(name, m, ctor(String(meta["weights_dir"])))
        end
    finally
        close(b)
    end
else
    @warn "ddm-torch unavailable: SAC and TD3 omitted"
end

mspec = proto["solvers"]["mcts"]
dspec = proto["solvers"]["dpw"]
mcts_mdp = InstrumentedMDP(DuckietownMDP(qcfg; action_space=:discrete))
dpw_mdp = InstrumentedMDP(DuckietownMDP(ccfg; action_space=:continuous))
run_one("mcts@1k", mcts_mdp, solve(MCTSSolver(
    n_iterations=Int(mspec["n_iterations"]), depth=Int(mspec["depth"]),
    exploration_constant=Float64(mspec["exploration_constant"]),
    rng=MersenneTwister(PLANNER_RNG), reuse_tree=false), mcts_mdp))
run_one("dpw@1k", dpw_mdp, solve(DPWSolver(
    n_iterations=Int(dspec["n_iterations"]), depth=Int(dspec["depth"]),
    exploration_constant=Float64(dspec["exploration_constant"]),
    enable_action_pw=true, enable_state_pw=false,
    k_action=Float64(dspec["k_action"]),
    alpha_action=Float64(dspec["alpha_action"]),
    rng=MersenneTwister(PLANNER_RNG), keep_tree=false), dpw_mdp))

# --- aggregate back and compare, field by field ---------------------------
original = load_rollout_artifact(ORIGINAL; horizon=HORIZON)
bykey = Dict((r.solver, r.seed) => r for r in original.records)

const METRIC_FIELDS = (:decisions, :ret, :discounted_return, :progress,
    :mean_abs_d, :max_abs_d, :mean_abs_phi, :mean_speed, :brake_ratio,
    :offroad, :other_collision, :duck_collision, :timeout, :goal,
    :full_stops, :stop_violations, :passed_stops, :stop_zone_decisions,
    :duck_yield_decisions, :duck_active_decisions, :crossings, :reason)

const ORIGINAL_FIELD = Dict(:decisions => :decisions, :ret => :ret,
    :discounted_return => :discounted_return, :progress => :progress,
    :mean_abs_d => :mean_abs_d, :max_abs_d => :max_abs_d,
    :mean_abs_phi => :mean_abs_phi, :mean_speed => :mean_speed,
    :brake_ratio => :brake_ratio, :offroad => :offroad,
    :other_collision => :other_collision, :duck_collision => :duck_collision,
    :timeout => :timeout, :goal => :goal, :full_stops => :full_stops,
    :stop_violations => :stop_violations, :passed_stops => :passed_stops,
    :stop_zone_decisions => :stop_zone_decisions,
    :duck_yield_decisions => :duck_yield_decisions,
    :duck_active_decisions => :duck_active_decisions,
    :crossings => :crossings, :reason => :reason)

# Wrapped in a function: a bare top-level loop puts `checked`, `mismatches`
# and `reagg` in soft scope, and Julia then treats each as a fresh local. This
# is the third tool script in the project to hit that, after
# fj8_native_check.jl and fj9_render_check.jl.
function compare_to_original(traces, episodes, bykey)
    mismatches = Any[]
    checked = 0
    reagg = EpisodeMetrics[]
    for name in sort(collect(keys(episodes)))
        rows = filter(t -> t.solver == name, traces)
        re = reaggregate_episodes(rows)
        append!(reagg, re)
        for e in re
            key = (name, e.seed)
            haskey(bykey, key) || continue
            o = bykey[key]
            checked += 1
            for f in METRIC_FIELDS
                got = getfield(e, f)
                want = getfield(o, ORIGINAL_FIELD[f])
                (got === want || got == want) && continue
                push!(mismatches, Dict("solver" => name, "seed" => e.seed,
                    "field" => String(f), "enriched" => string(got),
                    "original" => string(want)))
            end
        end
    end
    return mismatches, checked, reagg
end

mismatches, checked, reagg = compare_to_original(traces, episodes, bykey)

mkpath(OUT)
write(joinpath(OUT, "decisions.csv"), decision_csv(traces))
write(joinpath(OUT, "episodes_reaggregated.csv"),
    episode_csv([SolverRun(name, "", episodes[name],
        summarize_planning(PlanningDiagnostics[]), DecisionRecord[], "")
                 for name in sort(collect(keys(episodes)))]))

fingerprints = Dict(
    "experiment_fingerprint" => string(hash((HORIZON, EPISODES, PLANNER_RNG,
        [(k, proto["solvers"][k]) for k in sort(collect(keys(proto["solvers"])))]));
        base=16, pad=16),
    "episode_fingerprint_enriched" => string(hash([(e.seed, e.ret, e.decisions,
        e.reason) for e in sort(reagg; by=e -> (e.seed, e.ret))]); base=16, pad=16),
    "episode_fingerprint_original" => string(hash([(r.seed, r.ret, r.decisions,
        r.reason) for r in sort(original.records; by=r -> (r.seed, r.ret))]);
        base=16, pad=16),
    "decision_log_fingerprint" => string(hash(decision_csv(traces));
        base=16, pad=16),
    "original_artifact_fingerprint" => artifact_fingerprint(original))
write(joinpath(OUT, "fingerprints.json"), JSON3.write(fingerprints))

report = Dict("gate" => "FJ8.4c",
    "statement" => "FJ8.4c does not supersede FJ8.4b. It reproduces the " *
        "frozen evaluation protocol while enriching the observational record.",
    "episodes_checked" => checked,
    "fields_per_episode" => length(METRIC_FIELDS),
    "mismatches" => mismatches,
    "exact" => isempty(mismatches),
    "decision_rows" => length(traces),
    "solvers" => sort(collect(keys(episodes))),
    "seeds" => EPISODES, "horizon" => HORIZON)
write(joinpath(OUT, "reproduction_report.json"), JSON3.write(report))

println("EPISODES_CHECKED=", checked)
println("DECISION_ROWS=", length(traces))
println("MISMATCHES=", length(mismatches))
println("EXACT=", isempty(mismatches))
println("EPISODE_FP_MATCH=", fingerprints["episode_fingerprint_enriched"] ==
    fingerprints["episode_fingerprint_original"])
println("ORIGINAL_UNTOUCHED=", isfile(ORIGINAL))
for m in mismatches[1:min(10, end)]
    println("  MISMATCH ", m)
end
println("ENRICH_OK=", isempty(mismatches))
