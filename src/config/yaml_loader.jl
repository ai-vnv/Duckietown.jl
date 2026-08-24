"""
    load_config(path) -> DuckietownConfig

Parse one experiment `training_config.yaml` into the fully typed
[`DuckietownConfig`](@ref), applying the authoritative config hierarchy:

> experiment YAML > Python source defaults, per missing key

i.e. `StateConfig(**config["state"])` semantics: keys absent from the YAML
fall back to the Python defaults encoded in the config structs. Values are
validated exactly like the Python constructors (`spawn_route_direction`,
`spawn_min_route_alignment`, spawn bounds). Unknown keys are ignored, mirroring
the Python loader (only the keys each section declares are consumed).
"""
function load_config(path::AbstractString)
    raw = YAML.load_file(String(path))
    raw isa AbstractDict || throw(ArgumentError("config root must be a mapping"))
    data = _string_dict(raw)

    algorithm = _symbol(data, "algorithm")
    algorithm in (:q_learning, :sarsa, :sac, :td3) ||
        throw(ArgumentError("unsupported algorithm: $algorithm"))

    env = _environment(data)
    state = _state(data)
    continuous = _optional_continuous_state(data)
    actions = _actions(data)
    duck = _duck_controller(data)
    reward = _reward(data)
    solver = _solver(algorithm, data)
    lane_teacher = haskey(data, "lane_teacher") ? _lane_teacher(data["lane_teacher"]) : nothing
    transition = haskey(data, "transition_model") ?
        TransitionModelConfig(_bool(_string_dict(data["transition_model"]), "enabled", false)) :
        TransitionModelConfig(false)
    training = haskey(data, "training") ? _training(data["training"]) : TrainingConfig()
    evaluation = haskey(data, "evaluation") ? _evaluation(data["evaluation"]) : EvaluationConfig()
    wandb = get(data, "wandb", Dict{String,Any}())

    return DuckietownConfig(
        algorithm, _string(data, "stage", "full"), _int(data, "seed", 0),
        env, state, continuous, actions, duck, reward, solver,
        lane_teacher, transition, training, evaluation, wandb,
    )
end

"""
    default_config(algorithm::Symbol) -> DuckietownConfig

Config with every MDP parameter at its Python source default. Useful for
provenance tests and for experiments that only override a handful of keys.
"""
function default_config(algorithm::Symbol)
    algorithm in (:q_learning, :sarsa, :sac, :td3) ||
        throw(ArgumentError("unsupported algorithm: $algorithm"))
    solver = algorithm === :q_learning ? QLearningConfig(0.99, 0.10, 0.20, 0.01, 60000, collect(0:6)) :
        algorithm === :sarsa ? SarsaConfig(0.99, 0.10, 0.03, 0.00, 60000, collect(0:6)) :
        algorithm === :sac ? SacConfig(0.99, 0.005, 0.0003, 0.0003, 0.0003, 0.2, 256, 300000, 256, -2.0) :
        Td3Config(0.99, 0.005, 0.0003, 0.0003, 256, 300000, 256, 0.10, 0.20, 0.50, 2, 3000)
    DuckietownConfig(
        algorithm, "full", 0, EnvironmentConfig(), StateConfig(),
        nothing, ActionConfig(), DuckControllerConfig(), RewardConfig(),
        solver, nothing, TransitionModelConfig(false), TrainingConfig(),
        EvaluationConfig(), Dict{String,Any}(),
    )
end

# --- section parsers ---------------------------------------------------------

function _environment(data::Dict{String,Any})
    d = _section(data, "environment")
    direction = haskey(d, "spawn_route_direction") ?
        Symbol(_string(d, "spawn_route_direction", "")) : nothing
    if direction !== nothing && direction ∉ (:clockwise, :counterclockwise)
        throw(ArgumentError("spawn_route_direction must be clockwise, counterclockwise, or null"))
    end
    alignment = _float(d, "spawn_min_route_alignment", 0.50)
    if !(0.0 <= alignment <= 1.0)
        throw(ArgumentError("spawn_min_route_alignment must be in [0, 1]"))
    end
    bounds = _opt_tuple4(d, "spawn_position_bounds_xz")
    if bounds !== nothing && (bounds[1] > bounds[2] || bounds[3] > bounds[4])
        throw(ArgumentError("spawn position bounds must satisfy min <= max"))
    end
    return EnvironmentConfig(
        map_name=_string(d, "map_name", "small_loop"),
        domain_rand=_bool(d, "domain_rand", false),
        max_steps=_int(d, "max_steps", 1500),
        frame_skip=_int(d, "frame_skip", 6),
        render_observations=_bool(d, "render_observations", true),
        accept_start_angle_deg=_float(d, "accept_start_angle_deg", 60),
        spawn_max_abs_d=_opt_float(d, "spawn_max_abs_d"),
        spawn_max_abs_phi=_opt_float(d, "spawn_max_abs_phi"),
        spawn_attempts=_int(d, "spawn_attempts", 50),
        spawn_route_direction=direction,
        spawn_route_center=_opt_tuple2(d, "spawn_route_center"),
        spawn_min_route_alignment=alignment,
        spawn_position_bounds_xz=bounds,
        user_tile_start=_opt_tuple2i(d, "user_tile_start"),
        goal_tile=_opt_tuple2i(d, "goal_tile"),
    )
end

function _state(data::Dict{String,Any})
    d = _section(data, "state")
    return StateConfig(
        stop_lateral_limit=_float(d, "stop_lateral_limit", 0.40),
        stop_orientation_cos=_float(d, "stop_orientation_cos", 0.70710678),
        sign_to_line_offset=_float(d, "sign_to_line_offset", 0.20),
        stop_max_distance=_float(d, "stop_max_distance", 3.0),
        stop_zone=_float(d, "stop_zone", 0.45),
        stop_pass_distance=_float(d, "stop_pass_distance", 0.55),
        stop_speed=_float(d, "stop_speed", 0.02),
        stop_hold_steps=_int(d, "stop_hold_steps", 1),
        tile_lookahead=_float(d, "tile_lookahead", 0.30),
        curvature_threshold=_float(d, "curvature_threshold", 0.05),
        duck_max_distance=_float(d, "duck_max_distance", 2.0),
        duck_near_distance=_float(d, "duck_near_distance", 0.60),
        duck_corridor_width=_float(d, "duck_corridor_width", 0.35),
    )
end

function _optional_continuous_state(data::Dict{String,Any})
    haskey(data, "continuous_state") || return nothing
    d = _section(data, "continuous_state")
    return ContinuousStateConfig(
        max_speed=_float(d, "max_speed", 0.41),
        max_abs_curvature=_float(d, "max_abs_curvature", 8.0),
        max_stop_distance=_float(d, "max_stop_distance", 3.0),
        max_duck_distance=_float(d, "max_duck_distance", 2.0),
        max_relative_speed=_float(d, "max_relative_speed", 0.50),
        curvature_samples=_int(d, "curvature_samples", 33),
        duck_detection_range=_opt_float(d, "duck_detection_range"),
        duck_detection_corridor_width=_opt_float(d, "duck_detection_corridor_width"),
        duck_detection_forward_only=_bool(d, "duck_detection_forward_only", false),
    )
end

function _actions(data::Dict{String,Any})
    d = _section(data, "actions")
    return ActionConfig(
        v_fast=_float(d, "v_fast", 0.40),
        v_slow=_float(d, "v_slow", 0.15),
        w0=_float(d, "w0", 1.50),
        wheel_base=_float(d, "wheel_base", 0.102),
    )
end

function _duck_controller(data::Dict{String,Any})
    d = _section(data, "duck_controller")
    return DuckControllerConfig(
        p_cross=_float(d, "p_cross", 0.02),
        make_dynamic=_bool(d, "make_dynamic", true),
        require_duck=_bool(d, "require_duck", true),
        inject_if_missing=_bool(d, "inject_if_missing", false),
        spawn_pos=_opt_tuple2(d, "spawn_pos", (1.62, 0.50)),
        spawn_rotate=_float(d, "spawn_rotate", 0.0),
        spawn_height=_float(d, "spawn_height", 0.08),
        walk_distance=_float(d, "walk_distance", 0.90),
        trigger_min_ego_distance=_float(d, "trigger_min_ego_distance", 0.55),
        trigger_max_ego_distance=_float(d, "trigger_max_ego_distance", 1.10),
        spawn_on_ego_proximity=_bool(d, "spawn_on_ego_proximity", false),
        max_crossings_per_episode=_int(d, "max_crossings_per_episode", 0),
        repeat_rearm_distance=_float(d, "repeat_rearm_distance", 0.0),
        inject_stop_if_missing=_bool(d, "inject_stop_if_missing", false),
        require_stop=_bool(d, "require_stop", false),
        stop_spawn_pos=_opt_tuple2(d, "stop_spawn_pos", (1.20, 2.10)),
        stop_spawn_rotate=_float(d, "stop_spawn_rotate", 180.0),
        stop_spawn_height=_float(d, "stop_spawn_height", 0.18),
    )
end

function _reward(data::Dict{String,Any})
    d = _section(data, "reward")
    return RewardConfig(
        alpha_progress=_float(d, "alpha_progress", 1.0),
        alpha_lateral=_float(d, "alpha_lateral", 10.0),
        alpha_heading=_float(d, "alpha_heading", 2.0),
        step_cost=_float(d, "step_cost", 0.01),
        collision_duck=_float(d, "collision_duck", -100.0),
        other_collision=_float(d, "other_collision", -50.0),
        offroad=_float(d, "offroad", -50.0),
        stop_violation=_float(d, "stop_violation", -20.0),
        full_stop=_float(d, "full_stop", 10.0),
        duck_yield=_float(d, "duck_yield", 0.0),
        duck_unsafe=_float(d, "duck_unsafe", 0.0),
        duck_yield_speed=_float(d, "duck_yield_speed", 0.04),
        unnecessary_stop=_float(d, "unnecessary_stop", 0.0),
        idle_speed=_float(d, "idle_speed", 0.04),
        stop_exemption_distance=_float(d, "stop_exemption_distance", 0.45),
        stop_approach_distance=_float(d, "stop_approach_distance", 0.0),
        stop_approach_speed=_float(d, "stop_approach_speed", 0.02),
        stop_approach_yield=_float(d, "stop_approach_yield", 0.0),
        stop_approach_unsafe=_float(d, "stop_approach_unsafe", 0.0),
        straight_steer_penalty=_float(d, "straight_steer_penalty", 0.0),
        straight_curvature_threshold=_float(d, "straight_curvature_threshold", 0.05),
        max_steer_command=_float(d, "max_steer_command", 1.5),
        goal=_float(d, "goal", 50.0),
    )
end

function _solver(algorithm::Symbol, data::Dict{String,Any})
    d = _section(data, String(algorithm))
    if algorithm === :q_learning || algorithm === :sarsa
        allowed = [Int(x) for x in _vector(d, "allowed_actions")]
        if algorithm === :q_learning
            return QLearningConfig(
                _float(d, "gamma", 0.99), _float(d, "alpha_lr", 0.10),
                _float(d, "epsilon_start", 0.20), _float(d, "epsilon_end", 0.01),
                _int(d, "epsilon_decay_steps", 60000), allowed,
            )
        end
        return SarsaConfig(
            _float(d, "gamma", 0.99), _float(d, "alpha_lr", 0.10),
            _float(d, "epsilon_start", 0.03), _float(d, "epsilon_end", 0.00),
            _int(d, "epsilon_decay_steps", 60000), allowed,
        )
    elseif algorithm === :sac
        return SacConfig(
            _float(d, "gamma", 0.99), _float(d, "tau", 0.005),
            _float(d, "actor_lr", 0.0003), _float(d, "critic_lr", 0.0003),
            _float(d, "alpha_lr", 0.0003), _float(d, "initial_alpha", 0.2),
            _int(d, "batch_size", 256), _int(d, "replay_capacity", 300000),
            _int(d, "hidden_size", 256), _float(d, "target_entropy", -2.0),
        )
    end
    return Td3Config(
        _float(d, "gamma", 0.99), _float(d, "tau", 0.005),
        _float(d, "actor_lr", 0.0003), _float(d, "critic_lr", 0.0003),
        _int(d, "batch_size", 256), _int(d, "replay_capacity", 300000),
        _int(d, "hidden_size", 256), _float(d, "exploration_noise", 0.10),
        _float(d, "target_policy_noise", 0.20), _float(d, "target_noise_clip", 0.50),
        _int(d, "policy_delay", 2), _int(d, "actor_update_start", 3000),
    )
end

function _lane_teacher(d::Any)
    d isa AbstractDict || throw(ArgumentError("lane_teacher must be a mapping"))
    d = _string_dict(d)
    return LaneTeacherConfig(
        _bool(d, "enabled", true), _int(d, "full_control_episodes", 100),
        _int(d, "decay_episodes", 200), _float(d, "min_probability", 0.0),
        _float(d, "d_gain", 1.0), _float(d, "error_threshold", 0.10),
        _bool(d, "brake_for_stop", true), _float(d, "stop_brake_distance", 0.45),
        _bool(d, "brake_for_duck", true),
    )
end

function _training(d::Any)
    d isa AbstractDict || throw(ArgumentError("training must be a mapping"))
    d = _string_dict(d)
    TrainingConfig(
        _opt_int(d, "episodes"), _opt_int(d, "log_every"),
        _opt_int(d, "moving_average_window"), _opt_int(d, "checkpoint_every"),
        haskey(d, "milestone_episodes") ? [Int(x) for x in _vector(d, "milestone_episodes")] : nothing,
        _opt_int(d, "total_steps"), _opt_int(d, "random_steps"),
        _opt_int(d, "gradient_steps"), _opt_int(d, "checkpoint_interval"),
        _opt_int(d, "log_interval"),
        haskey(d, "output_dir") ? String(d["output_dir"]) : nothing,
        haskey(d, "initial_q_table") ? String(d["initial_q_table"]) : nothing,
        haskey(d, "initial_checkpoint") ? String(d["initial_checkpoint"]) : nothing,
        _opt_bool(d, "save_initial_checkpoint"),
        haskey(d, "device") ? String(d["device"]) : nothing,
    )
end

function _evaluation(d::Any)
    d isa AbstractDict || throw(ArgumentError("evaluation must be a mapping"))
    d = _string_dict(d)
    EvaluationConfig(
        _opt_int(d, "episodes"),
        haskey(d, "seeds") ? [Int(x) for x in _vector(d, "seeds")] : nothing,
        _opt_int(d, "development_episodes"), _opt_int(d, "final_episodes"),
        haskey(d, "development_seeds") ? [Int(x) for x in _vector(d, "development_seeds")] : nothing,
        haskey(d, "final_seeds") ? [Int(x) for x in _vector(d, "final_seeds")] : nothing,
        _float(d, "success_min_progress_m", 5.0),
        _float(d, "success_max_brake_ratio", 0.25),
        _opt_float(d, "brake_command_threshold"), _opt_float(d, "move_command_threshold"),
        _opt_float(d, "spin_omega_threshold"), _opt_int(d, "resume_window_steps"),
    )
end

# --- low-level helpers --------------------------------------------------------

_string_dict(d::AbstractDict) = Dict{String,Any}(string(k) => v for (k, v) in d)

_section(data::Dict{String,Any}, key::String) =
    _string_dict(_mapping(get(data, key, nothing), "missing section: $key"))

_mapping(x, what) = x isa AbstractDict ? x : throw(ArgumentError(what))

_string(d::Dict{String,Any}, key::String, default::String) =
    get(d, key, nothing) === nothing ? default : String(d[key])

_symbol(d::Dict{String,Any}, key::String) = Symbol(_string(d, key, ""))

_bool(d::Dict{String,Any}, key::String, default::Bool) =
    get(d, key, nothing) === nothing ? default : Bool(d[key])

_int(d::Dict{String,Any}, key::String, default::Int) =
    get(d, key, nothing) === nothing ? default : Int(d[key])

_float(d::Dict{String,Any}, key::String, default::Real) =
    get(d, key, nothing) === nothing ? Float64(default) : Float64(d[key])

_opt_int(d::Dict{String,Any}, key::String) =
    get(d, key, nothing) === nothing ? nothing : Int(d[key])

_opt_bool(d::Dict{String,Any}, key::String) =
    get(d, key, nothing) === nothing ? nothing : Bool(d[key])

_opt_float(d::Dict{String,Any}, key::String) =
    get(d, key, nothing) === nothing ? nothing : Float64(d[key])

_opt_tuple2(d::Dict{String,Any}, key::String) =
    get(d, key, nothing) === nothing ? nothing :
        (Float64(d[key][1]), Float64(d[key][2]))

_opt_tuple2(d::Dict{String,Any}, key::String, default::Tuple) =
    get(d, key, nothing) === nothing ? default :
        (Float64(d[key][1]), Float64(d[key][2]))

_opt_tuple2i(d::Dict{String,Any}, key::String) =
    get(d, key, nothing) === nothing ? nothing :
        (Int(d[key][1]), Int(d[key][2]))

_opt_tuple4(d::Dict{String,Any}, key::String) =
    get(d, key, nothing) === nothing ? nothing :
        (Float64(d[key][1]), Float64(d[key][2]), Float64(d[key][3]), Float64(d[key][4]))

_vector(d::Dict{String,Any}, key::String) =
    (v = get(d, key, nothing); v === nothing ? throw(ArgumentError("missing key: $key")) : v)