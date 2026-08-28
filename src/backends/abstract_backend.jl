"""
    AbstractBackend

Interface boundary between the decision-process formulation and the two
simulation backends (FJ0 audit section E/F):

- `native_julia.jl` — full generative reproduction (FJ3 dynamics);
- `gym_duckietown.jl` — PythonCall bridge to the unmodified duckduck
  `DuckieMDPEnv`/`ContinuousDuckieMDPEnv` (reference, for parity).

Backends translate between [`DuckieWorldState`](@ref) (canonical branchable
dynamics state) and their internal representation, and expose the decision
step with the *locked* transition order:

    1. duck controller `before_step` (activation draw)
    2. action → wheel commands
    3. `frame_skip` delayed-DB18 physics ticks (duckie step included)
    4. raw-state extraction (lane frame, stop candidate, duck threat)
    5. StopTracker update (sigma, events)
    6. collision / termination classification (duck > other > timeout >
       offroad > goal > in_progress)
    7. reward evaluation on the post-transition state

The projections `get_raw_state`/`get_continuous_state` are pure functions of
the world state (user constraint #2: the 7-D/15-D states are projections, not
the canonical MDP state).
"""
abstract type AbstractBackend end

"""
    reset!(backend, seed) -> DuckieWorldState

Sample `s0` from ρ0 (spawn curriculum, duck controller reset, stop memory
reset) and return the canonical world state.
"""
function reset! end

"""
    step!(backend, world::DuckieWorldState, action, rng) ->
        (world′, reward, terminated, truncated, info)

One decision: `frame_skip` physics ticks under the locked transition order.
`info` carries events, reward breakdown, raw/continuous projections, and
termination reason.
"""
function step! end

"""
    get_raw_state(world) -> RawState

Projection `f_tab`: the 7-component lane-relative state.
"""
function get_raw_state end

"""
    get_continuous_state(world, continuous_cfg) -> ContinuousState

Projection `f_cont`: the 15-component privileged state.
"""
function get_continuous_state end