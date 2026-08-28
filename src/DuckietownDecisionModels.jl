"""
    DuckietownDecisionModels

Solver-independent decision-process formulation package for the Duckietown MDP
described by the Python repository at `aivnv/duckduck` (FJ0 audit:
`duckduck/docs/FJ0_repository_audit.md`).

Three-level state hierarchy (canonical):
- [`DuckieWorldState`](@ref): full branchable latent dynamics state (delayed
  DB18 motor model, duckie objects, stop memory, map). This is the canonical
  state type for generative planning (MCTS/DPW). It is *not* an MDP state for
  the tabular/continuous solvers.
- [`RawState`](@ref): 7-component lane-relative projection used by
  Q-learning/SARSA (after discretization) and by the reward function.
- [`ContinuousState`](@ref): 15-component privileged projection used by
  SAC/TD3 observation encodings.

Gate status: FJ1 (skeleton, typed config hierarchy, YAML loaders, data model,
interface boundaries) — implemented. Dynamics, reward computation, transitions
and solvers land in later gates (FJ2+).
"""
module DuckietownDecisionModels

using POMDPs
using Random
using YAML
using LinearAlgebra
# JSON3 is used only by the FJ5 reference-backend line protocol; it carries no
# Python dependency — `using DuckietownDecisionModels` stays pure Julia.
using JSON3

include("rng/ziggurat_constants.jl")
include("rng/numpy_rng.jl")
include("model/tabular_state.jl")
include("model/continuous_state.jl")
include("model/actions.jl")
include("model/discretizer.jl")
include("model/encoding.jl")
include("model/state_projection.jl")
include("reward/events.jl")
include("reward/stop_tracker.jl")
include("reward/reward.jl")
include("dynamics/world_state.jl")
include("config/config.jl")
include("config/yaml_loader.jl")
include("backends/abstract_backend.jl")
include("interfaces/policies.jl")

include("dynamics/ego.jl")
include("dynamics/lane_geometry.jl")
include("dynamics/collision.jl")
include("dynamics/map_loading.jl")
include("dynamics/pedestrian.jl")
include("model/observers.jl")
include("generative/transition.jl")
include("generative/initial_state.jl")
include("backends/native_julia.jl")
include("backends/gym_duckietown.jl")
include("backends/torch_policy.jl")
include("interfaces/pomdps.jl")
# FJ8.1: the solver-facing contract. Deliberately mentions no solver — solver
# integrations live in ext/, never here.
include("interfaces/planning.jl")
# FJ10: POMDP readiness audit. An audit, not an implementation.
include("interfaces/pomdp_readiness.jl")
include("interfaces/rl_environment.jl")
include("solvers/adapters.jl")
include("solvers/actor_adapters.jl")
include("evaluation/rollout.jl")
include("evaluation/parity.jl")
include("evaluation/metrics.jl")
include("evaluation/benchmark.jl")
# FJ8.4a: the cost-search curve, solver-agnostic.
include("evaluation/budget.jl")
# FJ8.4b: cross-family comparison statistics — paired, two blocks, no ranking.
include("evaluation/comparison.jl")
# FJ9.0: the visualisation contract. All geometry is computed here, in the
# core; a backend extension only draws it. No plotting package is a
# dependency of this package.
include("visualization/scene.jl")
include("visualization/native_render.jl")
include("visualization/world_view.jl")
# FJ9.3: slices. Included after the solver adapters it dispatches on.
include("visualization/policy_slice.jl")
include("visualization/rollout.jl")
include("visualization/search_tree.jl")
include("visualization/diagnostics.jl")
# FJ9.7: playback. Included after diagnostics.jl — its signatures annotate
# `DecisionLog`, which is resolved when the method is defined, not when called.
include("visualization/animation.jl")
include("visualization/paper_figure.jl")
# FJ9.9: reproducibility closure — artefact ledger, documentation audit,
# core-formulation fingerprint, source-import lint, known limitations.
include("interfaces/reproducibility.jl")

export AbstractBackend, AbstractPolicy, act
export ActionConfig, ActionSpec, MacroAction, FAST_LEFT, FAST_STRAIGHT, FAST_RIGHT,
    SLOW_LEFT, SLOW_STRAIGHT, SLOW_RIGHT, BRAKE, DuckieAction
export build_action_table, build_continuous_state
export classify_tile, compute_reward, ContinuousState, ContinuousStateConfig
export continuous_observation_space, curve_signed_curvature, D_BINS
export DuckControllerConfig, DuckietownConfig, DuckRelativeState, EnvironmentConfig
export DuckThreat, NONE, SIDE_FAR, SIDE_NEAR, CROSSING_FAR, CROSSING_NEAR
export DuckieEgoState, DuckieState, DuckieWorldState, branch
export digitize, discretize, ego_relative_curve
export encode_continuous_state, EventFlags, EvaluationConfig
export gate_duck_visibility, LaneTeacherConfig
export QLearningConfig, SarsaConfig, SacConfig, Td3Config
export RawState, RoadMap, RewardBreakdown, RewardConfig
export StateConfig, StopMemory, StopSignState, StopTracker, hold_progress, reset_tracker
export terminal_lane_fallback, TileSpec, TileType, STRAIGHT, CURVE_LEFT, CURVE_RIGHT
export TrainingConfig, TransitionModelConfig, update!
export vw_to_wheels, action_to_wheels, bezier_point, bezier_tangent, curve_matrix
export bezier_closest, closest_curve_point, get_lane_pos2, LanePosition, NotInLane
export get_dir_vec, get_right_vec, heading_vec
export generate_corners, generate_norm, agent_boundbox, intersects, intersects_single_obj
export _valid_pose, _collision, _inconvenient_spawn, tile_corners
export _get_tile, _drivable_pos, get_agent_corners, calculate_safety_radius
export MapObjectData, interpret_object_desc, small_loop_map, parse_map_tiles
export sample_spawn_pose, get_grid_coords
export DB18Parameters, db18_nominal, db18_model, DelayedCommand, get_commands_at
export EGO_DELAY, EGO_DT, ego_tick, initial_ego, axis_observed_ticks
export cartesian_from_weird, weird_from_cartesian, SE2_from_se2, se2_multiply
export se2_from_linear_angular, translation_angle_from_SE2
export duck_step, before_step, activate_duck, replace_duck
export lane_frame_tabular, lane_frame_continuous, tile_ahead
export next_stop_candidate, distance_to_next_stop, classify_duck
export get_raw_state, signed_curvature_ahead, duck_relative_state
export get_continuous_state
export DuckieTransitionModel, TransitionResult, simulate_decision
export NumpyMT19937, random_sample, mt_next_uint32!, mt_state
export NumpySeedSequence, seedseq_generate_state
export NumpyPCG64, pcg64_next_uint64, pcg64_next_uint32, np_random_double,
    np_uniform, np_integers, np_standard_normal, np_normal
export TerminationReason, DUCK_COLLISION, OTHER_COLLISION, TIMEOUT, OFFROAD,
    GOAL, IN_PROGRESS
export termination_reason, is_terminated, is_truncated
export DuckietownMDP, DuckieActionSpace, DuckieInitialStateDistribution
export ALL_MACRO_ACTIONS, spawn_accepted, build_world
export initial_duckie, initial_map, drivable_tiles, object_world_pose
export route_circulation_score, position_in_bounds_xz
export AbstractReferenceBackend, ProcessReferenceBackend,
    PythonCallReferenceBackend
export ReferenceBackend, reference_backend_available, ref_call, ref_reset!,
    ref_get_state, ref_set_state!, ref_step!, ref_probe_stop,
    ref_to_world, world_to_ref
export QTablePolicy, QDecision, decide, read_npy, all_state_indices,
    greedy_action_table, tie_statistics, TIE_ATOL
export SACActorPolicy, TD3ActorPolicy, LinearLayer, forward
export TorchPolicyReferenceBackend, torch_policy_available, torch_call,
    torch_policy_init!, torch_policy_infer, torch_policy_infer_batch
export ReadinessStatus, READY, NEEDS_REFACTOR, NOT_READY, ReadinessItem,
    pomdp_readiness, readiness_table, readiness_counts,
    ObservabilityClass, SENSOR_ESTIMABLE, TEMPORALLY_DERIVED,
    MAP_PRIVILEGED, SIMULATOR_PRIVILEGED, AGENT_MEMORY,
    ComponentObservability, continuous_state_observability,
    observability_table, observability_counts,
    VISUALIZATION_EXTENSION_POINTS
export PlanningDiagnostics, plan_action, policy_action, InstrumentedMDP, MDPLike, AnyMDPLike,
    model_calls, reset_model_calls!, model_capabilities, capability_report
export EpisodeMetrics, evaluate_policy, summarize_evaluation, compare_policies
export PlannerCost, evaluate_planner, summarize_planning, DecisionRecord
export DecisionTrace, DECISION_TRACE_SCHEMA, decision_csv,
    reaggregate_episodes
export SummaryStats, summary_stats, PairedDifference, paired_difference,
    SolverRun, stop_compliance, task_table, safety_table, cost_table,
    paired_table, cost_by_episode_position, position_table, episode_csv,
    check_paired_protocol
export BudgetPoint, budget_study, budget_table, compute_matched_budget,
    estimate_budget_for_calls, matched_operating_points,
    operating_point_table, planning_seed_config
export WorldMismatch, world_differences, worlds_identical,
    shared_mutable_arrays, shared_by_design, rng_frozen
export GenBenchmark, measure, benchmark_gen, gen_scaling, gen_stage_profile,
    benchmark_table, per_call_us, bytes_per_call, allocs_per_call,
    calls_per_second, planning_budget_estimate
export RolloutRecord, rollout_native, rollout_reference, DriftReport,
    compare_rollouts, drift_summary, event_timing, event_timing_diff,
    rollout_table, three_lane_table, libm_hypothesis_check
export FieldDiff, StepParityReport, compare_worlds, compare_step,
    matched_state_sweep, parity_summary, worst, exact, parity_accepted,
    nonzero_fields, LIBM_1ULP_FIELDS, LIBM_DERIVED_FIELDS, LIBM_MAX_ULPS,
    LIBM_MAX_ABSDIFF, bitwise_only_fields, SIGNED_ZERO_FIELDS
export OBSERVATION_NAMES, Q_SHAPE, STATE_SHAPE, TRACKING_ERROR_BINS, V_BINS
export render_native, native_world, NativeWorld, NativeObject,
    NativeMeshGroup, load_obj_groups, place_object, tile_texture_file,
    duckietown_assets_root, NATIVE_RENDER_NOTE, NATIVE_CAMERA_FOV_Y,
    NATIVE_CAMERA_FLOOR_DIST, NATIVE_CAMERA_FORWARD_DIST,
    NATIVE_CAMERA_ANGLE
export render_world, render_projection, render_policy, render_search,
    render_search_action_plane,
    render_rollout, render_diagnostics, render_diagnostics_aggregate,
    render_frame, render_animation, render_paired_animation, render_composite
export ArtifactStatus, REBUILT, PERSISTED_SOURCE, PROVISIONED_FROZEN_INPUT,
    ArtifactRecord, artifact_ledger, STALE_CLAIMS, DocIssue,
    documentation_audit, core_fingerprint, SOURCE_IMPORT_BAN,
    source_import_audit, KNOWN_LIMITATIONS
export FigureRole, MAIN_FIGURE, SUPPLEMENTARY, DIAGNOSTIC_ONLY,
    PublicationArtifact, publication_inventory, inventory_table,
    PanelSpec, PublicationComposite, panel_ids, PUBLICATION_LAYOUT_VERSION,
    grid_layout, wrap_text,
    CaptionRule, CAPTION_RULES, caption_rule, check_caption,
    provenance_block, figure_model, figure_policy, figure_search,
    figure_episode
export AnimationFrame, AnimationSequence, animation_sequence,
    ANIMATION_TIMELINE_LABEL, ANIMATION_LAYOUT_VERSION, ANIMATION_ABSENT,
    animation_absent_lines, trajectory_through, events_through,
    series_through, model_time, MODEL_TIME_LABEL, frame_caption,
    animation_provenance, EpisodeSelection, select_episode, paired_frames,
    frame_index, is_frozen, StaticWorld, static_world, FrameScene, frame_scene
export TilePatch, WorldScene, world_scene, tile_patches, lane_centrelines,
    stop_line_segment, trajectory_points, projection_rows,
    PROJECTION_PANEL_TITLE, ProjectionCategory, LANE_GEOMETRY, EGO_MOTION,
    STOP_SUBSYSTEM, DUCK_SUBSYSTEM, ProjectionEntry, ProjectionScene,
    projection_scene
export capture_search, search_statistics, visible_nodes, search_summary
export state_fingerprint, snapshot_fingerprint, save_snapshot,
    load_snapshot
export SearchNode, SearchSnapshot, root_children, search_max_depth,
    check_snapshot
export SearchDataStatus, PERSISTED, AGGREGATE_ONLY, ABSENT, SearchDataItem,
    search_artifact_audit, search_audit_table, search_visualisation_supported
export FieldAvailability, LOGGED, DERIVED_IDENTITY, FIELD_ABSENT,
    availability_label, DecisionFieldItem, DECISION_QUANTITY_CONTRACT,
    decision_log_audit, decision_audit_table
export DecisionLog, DECISION_LOG_REQUIRED, load_decision_log,
    episode_diagnostics, episode_lengths, progress_bins
export SeriesCategory, NAVIGATION, MOTION_COMMAND, STOP_SUBSYS, DUCK_SUBSYS,
    REWARD, COMPUTE
export SeriesKind, INSTANTANEOUS, CUMULATIVE, FLAG
export AxisMode, ABSOLUTE_DECISION, NORMALIZED_PROGRESS
export DiagnosticSeries, EpisodeDiagnostics, DIAGNOSTIC_SERIES_SPEC,
    DIAGNOSTIC_EVENT_COLUMNS, series_named, series_in, n_missing,
    diagnostics_fingerprint, diagnostics_provenance
export EpisodeOutcome, ENV_TERMINATED, HORIZON_REACHED, EpisodeRecord,
    ROLLOUT_ARTIFACT_SCHEMA, outcome, ArtifactProvenance, RolloutAggregate,
    RolloutComparison, load_rollout_artifact, artifact_fingerprint,
    comparison_at_seed, median_return_seed, paired_metric,
    stop_compliance_of, solver_summary, comparison_table, provenance_lines
export SliceMode, FEATURE_SPACE, SLICE_FEATURE_SPACE_CAVEAT, SliceAxis,
    TabularSliceCell, ContinuousSliceCell, PolicySlice,
    TabularPolicySlice, ContinuousPolicySlice, policy_slice,
    slice_fingerprint, fixed_context_lines, slice_summary,
    raw_state_grid, continuous_state_grid, value_surface, action_surface,
    tie_surface, margin_surface, v_surface, omega_surface,
    TABULAR_SLICE_FIELDS
export default_config, load_config, SCENARIOS, scenario_config

end