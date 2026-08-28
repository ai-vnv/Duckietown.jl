"""
    EnvironmentConfig

Simulator/environment section of the experiment YAML (`environment:`).
Defaults mirror `build_env` in `src/env_wrapper.py` (note the source default
`accept_start_angle_deg = 60`; the experiments set 10).

- `spawn_route_direction`: `:clockwise` or `:counterclockwise` (or `nothing`),
  validated like Python.
- `spawn_route_center`: world `(x, z)` route centre; `nothing` means the map
  centre computed at spawn time.
- `spawn_position_bounds_xz`: `(xmin, xmax, zmin, zmax)` spawn rectangle.
- `user_tile_start`, `goal_tile`: fixed spawn/goal tiles, `nothing` disables.
"""
struct EnvironmentConfig
    map_name::String
    domain_rand::Bool
    max_steps::Int
    frame_skip::Int
    render_observations::Bool
    accept_start_angle_deg::Float64
    spawn_max_abs_d::Union{Nothing,Float64}
    spawn_max_abs_phi::Union{Nothing,Float64}
    spawn_attempts::Int
    spawn_route_direction::Union{Nothing,Symbol}
    spawn_route_center::Union{Nothing,NTuple{2,Float64}}
    spawn_min_route_alignment::Float64
    spawn_position_bounds_xz::Union{Nothing,NTuple{4,Float64}}
    user_tile_start::Union{Nothing,NTuple{2,Int}}
    goal_tile::Union{Nothing,NTuple{2,Int}}
end

function EnvironmentConfig(;
    map_name="small_loop",
    domain_rand=false,
    max_steps=1500,
    frame_skip=6,
    render_observations=true,
    accept_start_angle_deg=60,
    spawn_max_abs_d=nothing,
    spawn_max_abs_phi=nothing,
    spawn_attempts=50,
    spawn_route_direction=nothing,
    spawn_route_center=nothing,
    spawn_min_route_alignment=0.50,
    spawn_position_bounds_xz=nothing,
    user_tile_start=nothing,
    goal_tile=nothing,
)
    direction = spawn_route_direction === nothing ? nothing : Symbol(spawn_route_direction)
    if direction !== nothing && direction ∉ (:clockwise, :counterclockwise)
        throw(ArgumentError(
            "spawn_route_direction must be :clockwise, :counterclockwise, or nothing"))
    end
    alignment = Float64(spawn_min_route_alignment)
    if !(0.0 <= alignment <= 1.0)
        throw(ArgumentError("spawn_min_route_alignment must be in [0, 1]"))
    end
    if spawn_position_bounds_xz !== nothing
        xmin, xmax, zmin, zmax = spawn_position_bounds_xz
        if xmin > xmax || zmin > zmax
            throw(ArgumentError("spawn position bounds must satisfy min <= max"))
        end
    end
    EnvironmentConfig(
        String(map_name), Bool(domain_rand), Int(max_steps), Int(frame_skip),
        Bool(render_observations), Float64(accept_start_angle_deg),
        spawn_max_abs_d === nothing ? nothing : Float64(spawn_max_abs_d),
        spawn_max_abs_phi === nothing ? nothing : Float64(spawn_max_abs_phi),
        Int(spawn_attempts), direction,
        spawn_route_center === nothing ? nothing :
            (Float64(spawn_route_center[1]), Float64(spawn_route_center[2])),
        alignment,
        spawn_position_bounds_xz === nothing ? nothing :
            (Float64(spawn_position_bounds_xz[1]), Float64(spawn_position_bounds_xz[2]),
             Float64(spawn_position_bounds_xz[3]), Float64(spawn_position_bounds_xz[4])),
        user_tile_start === nothing ? nothing :
            (Int(user_tile_start[1]), Int(user_tile_start[2])),
        goal_tile === nothing ? nothing : (Int(goal_tile[1]), Int(goal_tile[2])),
    )
end

"""
    DuckControllerConfig

Pedestrian crossing controller parameters (`src/duck_controller.py`). The
defaults mirror the Python dataclass; experiments override `p_cross 1.0`,
triggers `0.35/0.45`, `max_crossings_per_episode 1`, and enable stop-sign
injection.

`spawn_pos`/`stop_spawn_pos` are YAML-frame tile coordinates, converted to
world poses by `get_transform` at map-preparation time.
"""
struct DuckControllerConfig
    p_cross::Float64
    make_dynamic::Bool
    require_duck::Bool
    inject_if_missing::Bool
    spawn_pos::NTuple{2,Float64}
    spawn_rotate::Float64
    spawn_height::Float64
    walk_distance::Float64
    trigger_min_ego_distance::Float64
    trigger_max_ego_distance::Float64
    spawn_on_ego_proximity::Bool
    max_crossings_per_episode::Int
    repeat_rearm_distance::Float64
    inject_stop_if_missing::Bool
    require_stop::Bool
    stop_spawn_pos::NTuple{2,Float64}
    stop_spawn_rotate::Float64
    stop_spawn_height::Float64
end

function DuckControllerConfig(;
    p_cross=0.02,
    make_dynamic=true,
    require_duck=true,
    inject_if_missing=false,
    spawn_pos=(1.62, 0.50),
    spawn_rotate=0.0,
    spawn_height=0.08,
    walk_distance=0.90,
    trigger_min_ego_distance=0.55,
    trigger_max_ego_distance=1.10,
    spawn_on_ego_proximity=false,
    max_crossings_per_episode=0,
    repeat_rearm_distance=0.0,
    inject_stop_if_missing=false,
    require_stop=false,
    stop_spawn_pos=(1.20, 2.10),
    stop_spawn_rotate=180.0,
    stop_spawn_height=0.18,
)
    DuckControllerConfig(
        Float64(p_cross), Bool(make_dynamic), Bool(require_duck),
        Bool(inject_if_missing), (Float64(spawn_pos[1]), Float64(spawn_pos[2])),
        Float64(spawn_rotate), Float64(spawn_height), Float64(walk_distance),
        Float64(trigger_min_ego_distance), Float64(trigger_max_ego_distance),
        Bool(spawn_on_ego_proximity), Int(max_crossings_per_episode),
        Float64(repeat_rearm_distance), Bool(inject_stop_if_missing),
        Bool(require_stop), (Float64(stop_spawn_pos[1]), Float64(stop_spawn_pos[2])),
        Float64(stop_spawn_rotate), Float64(stop_spawn_height),
    )
end

"""
    AbstractSolverConfig

Common supertype of the per-algorithm solver blocks (`q_learning:`, `sarsa:`,
`sac:`, `td3:`). Tabular schemas are reconstructed from the YAML because the
training code is not shipped in the duckduck repository.
"""
abstract type AbstractSolverConfig end

"""
    QLearningConfig

`q_learning:` block. `allowed_actions` indexes [`MacroAction`](@ref)
(experiments: `0:6`).
"""
struct QLearningConfig <: AbstractSolverConfig
    gamma::Float64
    alpha_lr::Float64
    epsilon_start::Float64
    epsilon_end::Float64
    epsilon_decay_steps::Int
    allowed_actions::Vector{Int}
end

"""
    SarsaConfig

`sarsa:` block (on-policy; epsilon decayed to zero so late episodes measure
the stable behaviour policy).
"""
struct SarsaConfig <: AbstractSolverConfig
    gamma::Float64
    alpha_lr::Float64
    epsilon_start::Float64
    epsilon_end::Float64
    epsilon_decay_steps::Int
    allowed_actions::Vector{Int}
end

"""
    SacConfig

`sac:` block.
"""
struct SacConfig <: AbstractSolverConfig
    gamma::Float64
    tau::Float64
    actor_lr::Float64
    critic_lr::Float64
    alpha_lr::Float64
    initial_alpha::Float64
    batch_size::Int
    replay_capacity::Int
    hidden_size::Int
    target_entropy::Float64
end

"""
    Td3Config

`td3:` block. `actor_update_start` restarts the critic warm-up relative to the
resumed update count when a checkpoint is loaded.
"""
struct Td3Config <: AbstractSolverConfig
    gamma::Float64
    tau::Float64
    actor_lr::Float64
    critic_lr::Float64
    batch_size::Int
    replay_capacity::Int
    hidden_size::Int
    exploration_noise::Float64
    target_policy_noise::Float64
    target_noise_clip::Float64
    policy_delay::Int
    actor_update_start::Int
end

"""
    LaneTeacherConfig

`lane_teacher:` block (tabular configs only; the teacher is training-time
infrastructure that is *not shipped* — Q-tables in `policies/` are the final
teacher-free greedy policies).
"""
struct LaneTeacherConfig
    enabled::Bool
    full_control_episodes::Int
    decay_episodes::Int
    min_probability::Float64
    d_gain::Float64
    error_threshold::Float64
    brake_for_stop::Bool
    stop_brake_distance::Float64
    brake_for_duck::Bool
end

"""
    TransitionModelConfig

`transition_model:` block (present in tabular configs; disabled).
"""
struct TransitionModelConfig
    enabled::Bool
end

"""
    TrainingConfig

`training:` block. The tabular and SAC/TD3 schemas differ; optional fields
carry `nothing` when absent in the given experiment.
"""
struct TrainingConfig
    episodes::Union{Nothing,Int}
    log_every::Union{Nothing,Int}
    moving_average_window::Union{Nothing,Int}
    checkpoint_every::Union{Nothing,Int}
    milestone_episodes::Union{Nothing,Vector{Int}}
    total_steps::Union{Nothing,Int}
    random_steps::Union{Nothing,Int}
    gradient_steps::Union{Nothing,Int}
    checkpoint_interval::Union{Nothing,Int}
    log_interval::Union{Nothing,Int}
    output_dir::Union{Nothing,String}
    initial_q_table::Union{Nothing,String}
    initial_checkpoint::Union{Nothing,String}
    save_initial_checkpoint::Union{Nothing,Bool}
    device::Union{Nothing,String}
end

TrainingConfig() = TrainingConfig(nothing, nothing, nothing, nothing, nothing,
    nothing, nothing, nothing, nothing, nothing,
    nothing, nothing, nothing, nothing, nothing)

"""
    EvaluationConfig

`evaluation:` block (tabular: `episodes`/`seeds`; SAC/TD3:
`development_*/final_*`).
"""
struct EvaluationConfig
    episodes::Union{Nothing,Int}
    seeds::Union{Nothing,Vector{Int}}
    development_episodes::Union{Nothing,Int}
    final_episodes::Union{Nothing,Int}
    development_seeds::Union{Nothing,Vector{Int}}
    final_seeds::Union{Nothing,Vector{Int}}
    success_min_progress_m::Float64
    success_max_brake_ratio::Float64
    brake_command_threshold::Union{Nothing,Float64}
    move_command_threshold::Union{Nothing,Float64}
    spin_omega_threshold::Union{Nothing,Float64}
    resume_window_steps::Union{Nothing,Int}
end

EvaluationConfig() = EvaluationConfig(nothing, nothing, nothing, nothing, nothing,
    nothing, 0.25, 0.25, nothing, nothing, nothing, nothing)

"""
    DuckietownConfig

The complete typed view of one `training_config.yaml`. The solver, training,
evaluation, and wandb blocks are preserved for provenance and solver
adapters; the MDP is fully determined by `environment`, `state`,
`continuous_state`, `actions`, `duck_controller`, and `reward`.

The solver block is stored as `solver::AbstractSolverConfig` (one of
[`QLearningConfig`](@ref), [`SarsaConfig`](@ref), [`SacConfig`](@ref),
[`Td3Config`](@ref) selected by `algorithm`); `continuous_state` and
`lane_teacher` are `nothing` when the experiment omits them.
"""
struct DuckietownConfig
    algorithm::Symbol
    stage::String
    seed::Int
    environment::EnvironmentConfig
    state::StateConfig
    continuous_state::Union{Nothing,ContinuousStateConfig}
    actions::ActionConfig
    duck_controller::DuckControllerConfig
    reward::RewardConfig
    solver::AbstractSolverConfig
    lane_teacher::Union{Nothing,LaneTeacherConfig}
    transition_model::TransitionModelConfig
    training::TrainingConfig
    evaluation::EvaluationConfig
    wandb::Dict{String,Any}
end