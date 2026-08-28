"""
    DuckieEgoState

Minimal Markov-sufficient ego state of the simulator (FJ0 audit section C).

The DB18 motor model runs *delayed* dynamics (`ApplyDelay`, 0.15 s): the
command applied at time `t` is the wheel command issued at `t - 0.15`. The
delay window spans roughly 5 physics ticks (dt = 1/30 s), and therefore
reaches into the *previous* decision when `frame_skip = 6`. Without the
command history, the generative process would not be Markov — this field is
the reason `DuckieWorldState` is branchable for MCTS/DPW.

- `pos`: simulator `cur_pos`, `(x, 0, z)` in world metres.
- `angle`: simulator `cur_angle`, world heading (rad).
- `v_long`, `omega`: DB18 body-state linear/angular velocity of the
  underlying `DynamicModel.v0`.
- `speed`: simulator `self.speed = norm(cur_pos - prev_pos) / delta_time`
  (displacement speed of the last tick, NOT `v_long`) — this is the `v`
  the raw-state extraction observes (`env.speed` in `get_raw_state`).
- `step_count`: physics ticks elapsed; `timestamp`: seconds elapsed.
- `command_history`: `(t, u_L, u_R)` wheel-command log over the delay window.
- `q0`: 3×3 SE(2) pose in cartesian coordinates (DB18 axis), i.e. the
  reference `DynamicModel.q0`.
- `v0`: 3×3 se(2) body velocity of the last tick (the reference `v0`).
- `axis_left_rad`/`axis_right_rad`: accumulated wheel angles in radians.
"""
struct DuckieEgoState
    pos::NTuple{3,Float64}
    angle::Float64
    v_long::Float64
    omega::Float64
    speed::Float64
    step_count::Int
    timestamp::Float64
    command_history::Vector{Tuple{Float64,Float64,Float64}}
    q0::Matrix{Float64}
    v0::Matrix{Float64}
    axis_left_rad::Float64
    axis_right_rad::Float64
end

"""
    MapObjectData

Static object descriptor resolved during initial-state construction (FJ3.1).

Mirrors the fields the simulator carries per collidable `WorldObj`:
- `pos`, `angle`, `scale`: object anchor pose (world frame).
- `static`, `optional`, `visible`: map-level flags (visible flips only under
  domain randomization).
- `min_coords`, `max_coords`: mesh bounding-box corners; `safety_radius`:
  `SAFETY_RAD_MULT · norm(max(|min|, |max|) on x/z) · scale`.
- `corners`: 4×2 SAT corners (`objects.generate_corners` order); `norm`: 2×2
  axis normals; `heading`: forward vector `(cos, 0, -sin)`.
"""
struct MapObjectData
    kind::Symbol
    pos::NTuple{3,Float64}
    angle::Float64
    scale::Float64
    static::Bool
    optional::Bool
    visible::Bool
    min_coords::NTuple{3,Float64}
    max_coords::NTuple{3,Float64}
    safety_radius::Float64
    corners::Matrix{Float64}
    norm::Matrix{Float64}
    heading::NTuple{3,Float64}
end

"""
    DuckieState

Full pedestrian (DuckieObj) sub-state.

- `pos`, `center`, `start`: object anchor, walking position, walking origin.
- `angle`, `heading`: object yaw and forward vector `(cos, 0, -sin)`.
- `vel`: walking speed magnitude (0.02 m/s in this setup); `walk_distance`:
  distance after which `finish_walk` flips heading and velocity.
- `pedestrian_active`: walking flag; `pedestrian_wait_time`: countdown that
  the DuckController pins to `Inf` before every decision (no autonomous
  activation); `time`: seconds since activation.
- `obj_corners`: 2D (x, z) SAT corners (shifted during walking and restored
  by the controller reset — hence state, not derived data); `obj_norm`: 2×2
  SAT axis normals; `scale`/`safety_radius`/`min_coords`/`max_coords`: mesh
  extent used by spawn checks and proximity rewards.
"""
struct DuckieState
    pos::NTuple{3,Float64}
    center::NTuple{3,Float64}
    start::NTuple{3,Float64}
    angle::Float64
    heading::NTuple{3,Float64}
    vel::Float64
    visible::Bool
    pedestrian_active::Bool
    pedestrian_wait_time::Float64
    time::Float64
    walk_distance::Float64
    scale::Float64
    safety_radius::Float64
    min_coords::NTuple{3,Float64}
    max_coords::NTuple{3,Float64}
    obj_corners::Vector{NTuple{2,Float64}}
    obj_norm::Matrix{Float64}
end

"""
    StopSignState

Pose of one static `sign_stop` object. Collision corners derive from the
static mesh bounding box (FJ2 collision module).
"""
struct StopSignState
    pos::NTuple{3,Float64}
    angle::Float64
end

"""
    StopMemory

Wrapper-side stop memory needed by the reward process:

- `sigma_stop`, `hold_steps`: the StopTracker dwell state (mirrored here so
  the world state is self-contained for branching).
- `last_stop_id`, `last_d_stop`: previous decision's stop-candidate identity
  and distance, required by `StopTracker.update(prev, curr, prev_id, curr_id)`
  for the "passed" test.
"""
struct StopMemory
    sigma_stop::Bool
    hold_steps::Int
    last_stop_id::Union{Nothing,Int}
    last_d_stop::Union{Nothing,Float64}
end

"""
    TileSpec

One map tile: kind (Python tile `kind`, lowercased; e.g. `:straight`,
`:curve_left`, `:curve_right`, `:3way_left`, `:4way`, `:asphalt`), rotation in
degrees, drivable flag, and the static directed lane curves (each a cubic
Bézier control polygon of 4×3 points, in world coordinates).
"""
struct TileSpec
    kind::Symbol
    angle_deg::Float64
    drivable::Bool
    curves::Vector{Vector{NTuple{3,Float64}}}
end

classify_tile(tile::TileSpec) = classify_tile(tile.drivable, String(tile.kind))

"""
    RoadMap

Static map data of one world (`small_loop` in all four experiments): tile
size, grid of [`TileSpec`](@ref), and the static collidable objects
(`sign_stop`; duckies are dynamic state in [`DuckieWorldState`](@ref)). The
map is immutable and shared across branch copies.
"""
struct RoadMap
    name::String
    tile_size::Float64
    grid::Matrix{TileSpec}
    static_objects::Vector{MapObjectData}
end

RoadMap(name::AbstractString, tile_size::Real, grid::AbstractMatrix{TileSpec}) =
    RoadMap(String(name), Float64(tile_size), grid, MapObjectData[])

"""
    DuckieWorldState

The canonical branchable latent dynamics state for generative planning
(FJ0 audit section C; user constraint #2). It contains *everything* needed to
reproduce the next decision deterministically:

- the ego (including the delayed-command window),
- all duckie objects and the crossing controller counters,
- the stop-sign poses, the stop memory, and the lane-fallback memory,
- the static map, and the controller RNG stream.

Projections onto the solver-facing states are pure functions:
`RawState = f_tab(world)` and `ContinuousState = f_cont(world)` (FJ2/FJ3).

`controller_rng` is stored as a native `MersenneTwister` for now. Note: the
Python stream is `np.random.RandomState` (MT19937 with legacy seeding); the
exact stream compatibility needed for seed-exact parity is scheduled for FJ3
dynamics work.
"""
mutable struct DuckieWorldState
    ego::DuckieEgoState
    ducks::Vector{DuckieState}
    stop_signs::Vector{StopSignState}
    map::RoadMap
    stop_memory::StopMemory
    lane_fallback::NTuple{2,Float64}
    crossings_started::Vector{Int}
    crossing_armed::Vector{Bool}
    controller_rng::MersenneTwister
end

"""
    branch(world, ego, ducks) -> DuckieWorldState

Copy a world state for simulated branching, deep-copying every mutable field
(command history, duckie corners, counters, RNG stream) so that branch
rollouts never alias shared mutation.
"""
function branch(world::DuckieWorldState;
    ego=world.ego,
    ducks=world.ducks,
    stop_signs=world.stop_signs,
    map=world.map,
    stop_memory=world.stop_memory,
    lane_fallback=world.lane_fallback,
    crossings_started=world.crossings_started,
    crossing_armed=world.crossing_armed,
    controller_rng=world.controller_rng,
)
    DuckieWorldState(
        DuckieEgoState(ego.pos, ego.angle, ego.v_long, ego.omega, ego.speed,
            ego.step_count, ego.timestamp, copy(ego.command_history),
            copy(ego.q0), copy(ego.v0), ego.axis_left_rad, ego.axis_right_rad),
        [DuckieState(d.pos, d.center, d.start, d.angle, d.heading, d.vel,
            d.visible, d.pedestrian_active, d.pedestrian_wait_time, d.time,
            d.walk_distance, d.scale, d.safety_radius, d.min_coords,
            d.max_coords, copy(d.obj_corners), copy(d.obj_norm)) for d in ducks],
        stop_signs,
        map,
        StopMemory(stop_memory.sigma_stop, stop_memory.hold_steps,
            stop_memory.last_stop_id, stop_memory.last_d_stop),
        lane_fallback,
        copy(crossings_started),
        copy(crossing_armed),
        copy(controller_rng),
    )
end