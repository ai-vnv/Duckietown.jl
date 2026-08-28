using DuckietownDecisionModels
using Test

const DUCKDUCK_POLICIES = joinpath(
    dirname(pkgdir(DuckietownDecisionModels)), "duckduck", "policies")

function config_path(algorithm::String)
    path = joinpath(DUCKDUCK_POLICIES, algorithm, "training_config.yaml")
    isfile(path) || error("reference config not found: $path")
    return path
end

@testset "load_config parses all four reference configs" begin
  if !(@isdefined(HAVE_REFERENCE)) || !HAVE_REFERENCE
    @test_skip "reference package not checked out beside this repository"
  else
    ql = load_config(config_path("q_learning"))
    sarsa = load_config(config_path("sarsa"))
    sac = load_config(config_path("sac"))
    td3 = load_config(config_path("td3"))

    @testset "q_learning" begin
        @test ql.algorithm === :q_learning
        @test ql.stage == "full"
        @test ql.seed == 53
        @test ql.continuous_state === nothing

        e = ql.environment
        @test e.map_name == "small_loop"
        @test !e.domain_rand
        @test e.max_steps == 1500
        @test e.frame_skip == 6
        @test !e.render_observations
        @test e.accept_start_angle_deg == 10
        @test e.spawn_max_abs_d == 0.08
        @test e.spawn_max_abs_phi == 0.175
        @test e.spawn_attempts == 50
        @test e.spawn_route_direction === nothing
        @test e.spawn_route_center === nothing
        @test e.spawn_min_route_alignment == 0.50
        @test e.spawn_position_bounds_xz === nothing
        @test e.user_tile_start === nothing
        @test e.goal_tile === nothing

        s = ql.state
        @test s.stop_lateral_limit == 0.40
        @test s.stop_orientation_cos == 0.70710678
        @test s.sign_to_line_offset == 0.20
        @test s.stop_max_distance == 3.0
        @test s.stop_zone == 0.45
        @test s.stop_pass_distance == 0.55
        @test s.stop_speed == 0.02
        @test s.stop_hold_steps == 1
        @test s.tile_lookahead == 0.30
        @test s.curvature_threshold == 0.05
        @test s.duck_max_distance == 1.20
        @test s.duck_near_distance == 0.60
        @test s.duck_corridor_width == 0.60

        a = ql.actions
        @test a.v_fast == 0.41
        @test a.v_slow == 0.17
        @test a.w0 == 1.50
        @test a.wheel_base == 0.102

        d = ql.duck_controller
        @test d.p_cross == 1.0
        @test d.make_dynamic
        @test d.require_duck && d.inject_if_missing
        @test d.spawn_pos == (1.62, 0.50)
        @test d.spawn_rotate == 0.0
        @test d.spawn_height == 0.08
        @test d.walk_distance == 0.90
        @test d.trigger_min_ego_distance == 0.35
        @test d.trigger_max_ego_distance == 0.45
        @test !d.spawn_on_ego_proximity
        @test d.max_crossings_per_episode == 1
        @test d.repeat_rearm_distance == 0.0
        @test d.inject_stop_if_missing && d.require_stop
        @test d.stop_spawn_pos == (1.20, 2.10)
        @test d.stop_spawn_rotate == 180.0
        @test d.stop_spawn_height == 0.18

        r = ql.reward
        @test r.alpha_progress == 1.0
        @test r.alpha_lateral == 2.0
        @test r.alpha_heading == 0.5
        @test r.step_cost == 0.01
        @test r.collision_duck == -200.0
        @test r.other_collision == -200.0
        @test r.offroad == -200.0
        @test r.stop_violation == -40.0
        @test r.full_stop == 15.0
        @test r.goal == 50.0
        @test r.duck_yield == 0.0
        @test r.duck_unsafe == 0.0
        @test r.unnecessary_stop == 0.0
        @test r.straight_steer_penalty == 0.0
        @test r.stop_approach_distance == 0.0

        q = ql.solver::QLearningConfig
        @test q.gamma == 0.99
        @test q.alpha_lr == 0.10
        @test q.epsilon_start == 0.20
        @test q.epsilon_end == 0.01
        @test q.epsilon_decay_steps == 60000
        @test q.allowed_actions == collect(0:6)
        @test ql.training.initial_q_table == "artifacts/ablation/lane/q_learning_teacher/q_table_best.npy"

        lt = ql.lane_teacher
        @test lt !== nothing
        @test lt.enabled
        @test lt.full_control_episodes == 100
        @test lt.decay_episodes == 200
        @test lt.min_probability == 0.0
        @test lt.d_gain == 1.0
        @test lt.error_threshold == 0.10
        @test lt.brake_for_stop
        @test lt.stop_brake_distance == 0.45
        @test lt.brake_for_duck

        @test !ql.transition_model.enabled
        @test ql.training.episodes == 400
        @test ql.training.output_dir == "runs/full_q_teacher"
        @test ql.evaluation.episodes == 30
        @test ql.evaluation.seeds == [101, 202, 303, 404, 505]
        @test ql.evaluation.success_min_progress_m == 5.0
        @test ql.evaluation.success_max_brake_ratio == 0.25
    end

    @testset "sarsa" begin
        @test sarsa.algorithm === :sarsa
        @test sarsa.seed == 53
        @test sarsa.environment.max_steps == 1500
        @test sarsa.state.stop_hold_steps == 1
        @test sarsa.reward.stop_violation == -40.0
        q = sarsa.solver::SarsaConfig
        @test q.gamma == 0.99
        @test q.epsilon_start == 0.03
        @test q.epsilon_end == 0.00
        @test q.allowed_actions == collect(0:6)
        @test sarsa.training.initial_q_table == "artifacts/ablation/lane/sarsa_teacher/q_table_best.npy"
        @test sarsa.lane_teacher !== nothing && sarsa.lane_teacher.enabled
    end

    @testset "sac" begin
        @test sac.algorithm === :sac
        @test sac.seed == 73
        @test sac.continuous_state !== nothing

        e = sac.environment
        @test e.max_steps == 9000
        @test e.spawn_attempts == 100
        @test e.spawn_route_direction === :clockwise
        @test e.spawn_route_center === nothing
        @test e.spawn_min_route_alignment == 0.50
        @test e.spawn_position_bounds_xz == (0.10, 0.45, 0.45, 1.30)
        @test e.user_tile_start == (0, 1)
        @test e.goal_tile === nothing

        @test sac.state.stop_hold_steps == 3

        c = sac.continuous_state
        @test c.max_speed == 0.41
        @test c.max_abs_curvature == 8.0
        @test c.max_stop_distance == 3.0
        @test c.max_duck_distance == 2.0
        @test c.max_relative_speed == 0.50
        @test c.curvature_samples == 33
        @test c.duck_detection_range == 1.20
        @test c.duck_detection_corridor_width == 0.60
        @test c.duck_detection_forward_only

        r = sac.reward
        @test r.duck_yield == 0.0
        @test r.duck_unsafe == -5.0
        @test r.unnecessary_stop == -2.0
        @test r.straight_steer_penalty == 0.5
        @test r.stop_approach_distance == 0.0
        @test r.stop_approach_yield == 0.0
        @test r.stop_approach_unsafe == 0.0

        q = sac.solver::SacConfig
        @test q.gamma == 0.99
        @test q.tau == 0.005
        @test q.actor_lr == 0.0003
        @test q.critic_lr == 0.0003
        @test q.alpha_lr == 0.0003
        @test q.initial_alpha == 0.2
        @test q.batch_size == 256
        @test q.replay_capacity == 300000
        @test q.hidden_size == 256
        @test q.target_entropy == -2.0

        @test sac.training.total_steps == 30000
        @test sac.training.random_steps == 0
        @test sac.training.gradient_steps == 1
        @test sac.training.checkpoint_interval == 5000
        @test sac.training.log_interval == 1000
        @test sac.training.output_dir == "runs/sac_full_gated_duck_ft"
        @test sac.training.initial_checkpoint == "artifacts/sac/full_repeat_duck_5min/sac_best.pt"
        @test sac.training.save_initial_checkpoint === true
        @test sac.training.device == "cuda"

        @test sac.evaluation.development_episodes == 10
        @test sac.evaluation.final_episodes == 30
        @test sac.evaluation.development_seeds == [2101, 2202, 2303, 2404, 2505]
        @test sac.evaluation.final_seeds == [20101, 20202, 20303, 20404, 20505]
        @test sac.evaluation.success_min_progress_m == 18.0
        @test sac.evaluation.success_max_brake_ratio == 0.25
        @test sac.evaluation.brake_command_threshold == 0.04
        @test sac.evaluation.move_command_threshold == 0.10
        @test sac.evaluation.spin_omega_threshold == 0.75
        @test sac.evaluation.resume_window_steps == 20
    end

    @testset "td3" begin
        @test td3.algorithm === :td3
        @test td3.seed == 73
        @test td3.environment.max_steps == 9000
        @test td3.state.stop_hold_steps == 3
        @test td3.continuous_state.duck_detection_range == 1.20

        r = td3.reward
        @test r.stop_violation == -80.0
        @test r.full_stop == 40.0
        @test r.stop_approach_distance == 0.60
        @test r.stop_approach_speed == 0.02
        @test r.stop_approach_yield == 1.0
        @test r.stop_approach_unsafe == -5.0
        @test r.straight_steer_penalty == 0.5

        q = td3.solver::Td3Config
        @test q.gamma == 0.99
        @test q.tau == 0.005
        @test q.exploration_noise == 0.10
        @test q.target_policy_noise == 0.20
        @test q.target_noise_clip == 0.50
        @test q.policy_delay == 2
        @test q.actor_update_start == 3000
        @test q.batch_size == 256
        @test q.replay_capacity == 300000
        @test q.hidden_size == 256

        @test td3.training.total_steps == 60000
        @test td3.training.checkpoint_interval == 10000
        @test td3.training.log_interval == 500
        @test td3.training.initial_checkpoint == "runs/td3_stop_curriculum/checkpoints/td3_step_000100000.pt"
        @test td3.training.save_initial_checkpoint === nothing
    end
  end
end

@testset "default_config matches Python source defaults" begin
    d = default_config(:q_learning)
    @test d.environment.accept_start_angle_deg == 60
    @test d.environment.spawn_attempts == 50
    @test d.state.stop_hold_steps == 1
    @test d.state.duck_max_distance == 2.0
    @test d.state.duck_corridor_width == 0.35
    @test d.state.stop_orientation_cos == 0.70710678
    @test d.actions.v_fast == 0.40
    @test d.actions.v_slow == 0.15
    @test d.reward.alpha_lateral == 10.0
    @test d.reward.alpha_heading == 2.0
    @test d.reward.collision_duck == -100.0
    @test d.reward.other_collision == -50.0
    @test d.reward.offroad == -50.0
    @test d.reward.stop_violation == -20.0
    @test d.reward.full_stop == 10.0
    @test d.reward.goal == 50.0
    @test d.reward.stop_approach_yield == 0.0
    @test d.reward.stop_approach_unsafe == 0.0
    @test d.reward.max_steer_command == 1.5
    @test d.duck_controller.p_cross == 0.02
    @test d.duck_controller.trigger_min_ego_distance == 0.55
    @test d.duck_controller.max_crossings_per_episode == 0
    @test d.duck_controller.inject_if_missing == false
    @test d.duck_controller.inject_stop_if_missing == false
    @test d.continuous_state === nothing
    @test d.lane_teacher === nothing
end

@testset "validation mirrors Python constructors" begin
    @test_throws ArgumentError EnvironmentConfig(spawn_route_direction=:north)
    @test_throws ArgumentError EnvironmentConfig(spawn_min_route_alignment=1.5)
    @test_throws ArgumentError EnvironmentConfig(spawn_position_bounds_xz=(0.5, 0.1, 0.0, 1.0))
    @test_throws ArgumentError default_config(:ppo)
end
# FJ-post: the notebook-facing scenarios. `default_config` must keep meaning
# "the Python source defaults" — a corrected or altered world gets its own
# name, which has been the rule for every config change in this project.
@testset "named scenarios are new configs, not edits to the defaults" begin
    using POMDPs, Random

    @test SCENARIOS == (:lane_following, :stop_and_duck, :stop_and_duck_safe)
    @test_throws ArgumentError scenario_config(:nonsense)

    # the defaults are untouched, including the two switches that matter
    d = default_config(:q_learning)
    @test d.duck_controller.inject_stop_if_missing == false
    @test d.duck_controller.require_stop == false
    @test d.duck_controller.p_cross == 0.02
    # `==` on these structs is object identity (they carry a Vector), so
    # compare field by field
    same(a, b) = all(getfield(a, f) == getfield(b, f)
                     for f in fieldnames(typeof(a)))
    lf = scenario_config(:lane_following)
    @test same(lf.environment, d.environment)
    @test same(lf.duck_controller, d.duck_controller)
    @test same(lf.reward, d.reward)

    # ...and the world built from them genuinely has no stop sign, which is
    # why a notebook user must not be pointed at it
    m0 = DuckietownMDP(d; action_space=:discrete)
    s0 = rand(MersenneTwister(1001), initialstate(m0))
    @test isempty(s0.stop_signs)

    # the scenario turns the task on
    sc = scenario_config(:stop_and_duck)
    @test sc.duck_controller.inject_stop_if_missing
    @test sc.duck_controller.require_stop
    @test sc.duck_controller.p_cross == 1.0
    @test sc.duck_controller.max_crossings_per_episode == 1
    # and constrains the spawn, or a straight rollout leaves the road at once
    @test sc.environment.spawn_max_abs_d == 0.08
    @test sc.environment.spawn_max_abs_phi == 0.175

    m1 = DuckietownMDP(sc; action_space=:discrete)
    s1 = rand(MersenneTwister(1001), initialstate(m1))
    @test length(s1.stop_signs) == 1
    @test length(s1.ducks) == 1
    raw1, _ = get_raw_state(s1, m1.transition.state_cfg)
    @test abs(raw1.d) <= 0.08 + 1e-9
    @test abs(raw1.phi) <= 0.175 + 1e-9

    # reward and state semantics are NOT touched by the scenario: it changes
    # where an episode starts and what is on the road, never how it is scored
    @test same(sc.reward, d.reward)
    @test same(sc.state, d.state)
    @test same(sc.actions, d.actions)
    @test same(sc.solver, d.solver)

    # every algorithm, both action spaces
    for alg in (:q_learning, :sarsa, :sac, :td3)
        c = scenario_config(:stop_and_duck; algorithm=alg)
        @test c.algorithm === alg
        @test c.duck_controller.require_stop
    end
    mc = DuckietownMDP(scenario_config(:stop_and_duck; algorithm=:td3);
        action_space=:continuous)
    @test length(rand(MersenneTwister(4), initialstate(mc)).stop_signs) == 1

    # `:stop_and_duck_safe` is the same rule applied again: a NEW name for a
    # changed world, never an edit to `:stop_and_duck`. It may differ from
    # `:stop_and_duck` in exactly the documented places — four safety reward
    # weights, and the stop sign turned to face the actual traffic on a
    # straight with enough detection runway — and nowhere else.
    safe = scenario_config(:stop_and_duck_safe)
    reward_delta = [f for f in fieldnames(typeof(safe.reward))
                    if getfield(safe.reward, f) != getfield(sc.reward, f)]
    @test sort(reward_delta) == sort([:duck_unsafe, :stop_approach_distance,
        :stop_approach_yield, :stop_approach_unsafe])
    @test safe.reward.duck_unsafe == -5.0
    @test safe.reward.stop_approach_distance == 0.60
    duck_delta = [f for f in fieldnames(typeof(safe.duck_controller))
                  if getfield(safe.duck_controller, f) !=
                     getfield(sc.duck_controller, f)]
    @test sort(duck_delta) == sort([:stop_spawn_pos, :stop_spawn_rotate])
    @test safe.duck_controller.stop_spawn_rotate == 0.0
    @test same(safe.environment, sc.environment)
    @test same(safe.state, sc.state)
    @test same(safe.actions, sc.actions)
end
