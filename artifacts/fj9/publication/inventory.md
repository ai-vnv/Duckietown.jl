# FJ9.8a — publication figure inventory

| Artifact | Role | Figure | Source | Present |
|---|---|---|---|---|
| world_scene — top-down latent world state | MAIN_FIGURE | Figure 1 | `artifacts/fj9/world.png` | yes |
| projection_panel — privileged ContinuousState projection | MAIN_FIGURE | Figure 1 | `artifacts/fj9/projection.png` | yes |
| tabular_slice — Q-learning action / value / tie / margin slice | MAIN_FIGURE | Figure 2 | `computed` | yes |
| continuous_slice — TD3 v_cmd and omega_cmd surfaces | MAIN_FIGURE | Figure 2 | `computed` | yes |
| search_tree_mcts — MCTS tree from the frozen state | MAIN_FIGURE | Figure 3 | `artifacts/fj9/search_snapshot_mcts.json` | yes |
| search_tree_dpw — DPW tree from the same frozen state | MAIN_FIGURE | Figure 3 | `artifacts/fj9/search_snapshot_dpw.json` | yes |
| action_plane_dpw — DPW sampled continuous root actions | MAIN_FIGURE | Figure 3 | `artifacts/fj9/search_snapshot_dpw.json` | yes |
| six_solver_table — frozen six-solver outcome summary | MAIN_FIGURE | Figure 4 | `artifacts/fj8/six_solver_episodes.csv` | yes |
| diagnostics_td3 — TD3 stagnation time series | MAIN_FIGURE | Figure 4 | `artifacts/fj8/enriched/decisions.csv` | yes |
| diagnostics_dpw — DPW deterioration time series | MAIN_FIGURE | Figure 4 | `artifacts/fj8/enriched/decisions.csv` | yes |
| compute_by_progress — realised model calls by episode progress | MAIN_FIGURE | Figure 4 | `artifacts/fj8/enriched/decisions.csv` | yes |
| rollout_comparison — aggregate return / length comparison | SUPPLEMENTARY | Supplementary | `artifacts/fj9/rollout_comparison.png` | yes |
| video_td3 — TD3 stagnation playback | SUPPLEMENTARY | Supplementary Video 1 | `artifacts/fj9/anim_td3_stagnation.mp4` | yes |
| video_dpw — DPW termination playback | SUPPLEMENTARY | Supplementary Video 2 | `artifacts/fj9/anim_dpw_terminal.mp4` | yes |
| video_mcts — MCTS reference playback | SUPPLEMENTARY | Supplementary Video 3 | `artifacts/fj9/anim_mcts_reference.mp4` | yes |
| video_paired — paired TD3 / DPW playback | SUPPLEMENTARY | Supplementary Video 4 | `artifacts/fj9/anim_paired_td3_dpw_1001.mp4` | yes |
| decision_contract — FJ9.6a decision-artifact contract | DIAGNOSTIC_ONLY | — | `artifacts/fj9/decision_contract.md` | yes |
| budget_study — FJ8.4a planner cost curve | DIAGNOSTIC_ONLY | — | `artifacts/fj8/budget_study.md` | yes |
| lap_completion — FJ8 lap analysis | DIAGNOSTIC_ONLY | — | `artifacts/fj8/lap_completion.txt` | yes |
