# FJ9.5b — capture what the frozen planners actually searched, once.
#
# This is a DIAGNOSTIC capture, not a benchmark run. It does not touch FJ8.4b
# and produces no performance claim; it answers one question:
#
#     what did this planner simulate from this one identified state?
#
# The state is chosen by a rule fixed BEFORE any tree is inspected, so the
# picture cannot be selected for looking interesting:
#
#     seed 1001 (the first frozen evaluation seed)
#     driven by the shipped Q-learning policy
#     decision 30
#
# The same latent state is used for both planners; only the action
# representation differs (MacroAction vs DuckieAction), which is exactly the
# distinction FJ4 kept explicit.
#
#     tools/run_capture_search.sh

using DuckietownDecisionModels
using POMDPs
using Random
using MCTS

const ROOT = pkgdir(DuckietownDecisionModels)
const DUCK = joinpath(ROOT, "..", "duckduck")
const OUT = joinpath(ROOT, "artifacts", "fj9")

const STATE_SEED = 1001
const STATE_DECISION = 30
const PLANNER_SEED = 2026
const STATE_RULE = "seed $STATE_SEED, shipped Q-learning policy, decision " *
    "$STATE_DECISION (fixed before any tree was inspected)"

qcfg = joinpath(DUCK, "policies", "q_learning", "training_config.yaml")
ccfg = joinpath(DUCK, "policies", "sac", "training_config.yaml")

"""The frozen state, by the rule above. Deterministic and reconstructible."""
function frozen_state()
    mdp = DuckietownMDP(qcfg; action_space=:discrete)
    pol = QTablePolicy(joinpath(DUCK, "policies", "q_learning", "policy.npy");
        solver=:q_learning)
    s = rand(MersenneTwister(STATE_SEED), initialstate(mdp))
    rng = MersenneTwister(STATE_SEED)
    for _ in 1:STATE_DECISION
        a = policy_action(pol, mdp, s)
        r = simulate_decision(mdp.transition, s, a, rng)
        s = r.sp
        (r.terminated || r.truncated) &&
            error("the state rule terminated early; it must reach decision " *
                  "$STATE_DECISION")
    end
    return s
end

state = frozen_state()
println("STATE_RULE=", STATE_RULE)
println("STATE_FINGERPRINT=", state_fingerprint(state))

dmdp = DuckietownMDP(qcfg; action_space=:discrete)
cmdp = DuckietownMDP(ccfg; action_space=:continuous)

# --- MCTS, discrete -------------------------------------------------------
# enable_tree_vis is required for capture: without it MCTSTree records no
# action -> next-state edges and the structure is genuinely absent.
mcts_solver = MCTSSolver(n_iterations=36, depth=10, exploration_constant=5.0,
    rng=MersenneTwister(PLANNER_SEED), reuse_tree=false, enable_tree_vis=true)
mcts_planner = solve(mcts_solver, dmdp)
mcts_choice = action(mcts_planner, state)
mcts_snap = capture_search(mcts_planner, state; id="fj95b-mcts",
    planner_seed=PLANNER_SEED, selected_action=mcts_choice)
chk_m = check_snapshot(mcts_snap; action_space=actions(dmdp))
println("MCTS_NODES=", length(mcts_snap.nodes))
println("MCTS_ROOT_CHILDREN=", length(root_children(mcts_snap)))
println("MCTS_MAX_DEPTH=", search_max_depth(mcts_snap))
println("MCTS_VALID=", chk_m.ok, " issues=", chk_m.issues)
save_snapshot(joinpath(OUT, "search_snapshot_mcts.json"), mcts_snap)

# --- DPW, continuous ------------------------------------------------------
dpw_solver = DPWSolver(n_iterations=35, depth=10, exploration_constant=5.0,
    enable_action_pw=true, enable_state_pw=false, k_action=4.0,
    alpha_action=0.5, rng=MersenneTwister(PLANNER_SEED), keep_tree=false)
dpw_planner = solve(dpw_solver, cmdp)
dpw_choice = action(dpw_planner, state)
dpw_snap = capture_search(dpw_planner, state; id="fj95b-dpw",
    planner_seed=PLANNER_SEED, selected_action=dpw_choice)
chk_d = check_snapshot(dpw_snap; action_space=actions(cmdp))
println("DPW_NODES=", length(dpw_snap.nodes))
println("DPW_ROOT_CHILDREN=", length(root_children(dpw_snap)))
println("DPW_MAX_DEPTH=", search_max_depth(dpw_snap))
println("DPW_VALID=", chk_d.ok, " issues=", chk_d.issues)
save_snapshot(joinpath(OUT, "search_snapshot_dpw.json"), dpw_snap)

# both snapshots must name the SAME latent state
println("SAME_STATE=",
    mcts_snap.state_fingerprint == dpw_snap.state_fingerprint)
println("MCTS_FINGERPRINT=", snapshot_fingerprint(mcts_snap))
println("DPW_FINGERPRINT=", snapshot_fingerprint(dpw_snap))
println("CAPTURE_OK=", chk_m.ok && chk_d.ok)
