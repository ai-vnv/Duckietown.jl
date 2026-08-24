# MASTER IMPLEMENTATION PROMPT  
## Python Duckietown MDP → Julia `DuckietownDecisionModels.jl`

You are a senior researcher and software engineer specializing in:

- Markov Decision Processes (MDP)
- Partially Observable Markov Decision Processes (POMDP)
- Reinforcement Learning
- Online planning
- Generative models for sequential decision making
- Autonomous mobile robotics
- Differential-drive vehicle dynamics
- Julia scientific computing
- `POMDPs.jl`
- Monte Carlo Tree Search
- Gym-Duckietown
- Reproducible computational research
- Scientific visualization

Your task is **not merely to translate Python syntax into Julia**.

Your task is to reconstruct an existing Duckietown MDP implementation as a clean, reusable, scientifically defensible **Julia decision-model package**, conceptually similar to how benchmark environments such as `SimpleGridWorld`, `MountainCar`, or other MDP/POMDP models can be instantiated independently of their solvers.

The intended package name is:

```text
DuckietownDecisionModels.jl
```

The package must ultimately support:

```julia
using DuckietownDecisionModels
using POMDPs

mdp = DuckietownMDP(...)
```

and allow multiple solver families to operate on the **same underlying problem formulation**.

---

# 1. FIRST RULE: READ THE EXISTING REPOSITORY BEFORE WRITING CODE

You are given an existing Python repository containing the completed MDP formulation and policies.

Do not immediately rewrite anything.

First inspect the repository systematically.

Treat the **actual Python source code and experiment YAML files** as the authoritative reference.

Comments and README documentation may help interpretation, but if documentation conflicts with executable code or experiment configuration, report the discrepancy and use the executable implementation as the source of truth.

Do not silently "improve" the mathematical formulation.

Do not redesign the reward, state, action, or scenario because another formulation appears more elegant.

The initial Julia implementation must reproduce the existing Python formulation as faithfully as possible.

---

# 2. IMPORTANT: IGNORE EXPLAINABILITY WORK

This repository also contains a substantial explainability pipeline.

That work is **outside the scope of this project**.

Do not attempt to port, redesign, interpret, or reproduce the explainability framework.

In particular, ignore implementation content under directories such as:

```text
configs/explainability/
collection/
docs/primitive_lexicon_v1.freeze.json
src/explainability/
artifacts/explainability/
runs/explanations/
```

and explainability-specific scripts/tests.

They may be inspected only if absolutely necessary to understand a dependency of the base MDP, but they must not influence the Julia architecture.

The current task concerns:

```text
MDP formulation
state representation
actions
transition dynamics
reward
environment logic
pedestrian dynamics
stop-sign logic
RL policies
generative models
online planning
visualization
validation
```

not explanation generation.

---

# 3. UNDERSTAND THE REPOSITORY STRUCTURE

Before implementation, produce an **MDP Repository Audit**.

At minimum inspect the following files.

## Core state representation

```text
src/state.py
src/continuous_state.py
src/discretizer.py
```

## Actions and robot control

```text
src/actions.py
```

## Reward

```text
src/reward.py
```

## MDP environment

```text
src/env_wrapper.py
src/continuous_env.py
```

## Dynamic pedestrian/task controller

```text
src/duck_controller.py
```

## Existing deep-RL agents

```text
src/agents/sac.py
src/agents/td3.py
```

## Existing trained policies and experimental configurations

```text
policies/q_learning/
policies/sarsa/
policies/sac/
policies/td3/
```

especially:

```text
training_config.yaml
```

for every solver.

The policy/checkpoint files are reference artefacts and must not be overwritten.

---

# 4. CURRENT PYTHON ARCHITECTURE THAT MUST BE UNDERSTOOD

The existing project already represents an MDP over Gym-Duckietown.

Conceptually:

```text
             Gym-Duckietown
                  │
           latent world state
                  │
        ┌─────────┴─────────┐
        │                   │
  tabular feature      continuous feature
     extraction            extraction
        │                   │
   RawState            ContinuousState
        │                   │
 discretization             │
        │                   │
 Q-learning/SARSA        SAC/TD3
```

The Julia implementation must **preserve this conceptual distinction**.

---

# 5. TABULAR POLICY STATE

Inspect `src/state.py`.

The current tabular-oriented policy state is:

```python
RawState(
    d,
    phi,
    v,
    tile,
    d_stop,
    sigma_stop,
    duck
)
```

Interpret the fields exactly from the existing implementation.

Broadly:

```text
d
    lateral lane error

phi
    heading error relative to lane

v
    actual ego speed

tile
    ego-relative lane geometry / TileType
    STRAIGHT
    CURVE_LEFT
    CURVE_RIGHT

d_stop
    distance to the next valid stop line,
    or no-stop state

sigma_stop
    one-bit stop-compliance memory

duck
    pedestrian threat class
```

The threat categories currently include:

```text
NONE
SIDE_FAR
SIDE_NEAR
CROSSING_FAR
CROSSING_NEAR
```

Do not simplify this representation.

---

# 6. TABULAR DISCRETIZATION

Inspect:

```text
src/discretizer.py
```

The tabular state does **not** bin `phi` independently.

It defines:

```text
tracking_error = phi + d
```

and discretizes approximately:

```text
d
tracking_error
v
tile
d_stop
sigma_stop
duck
```

Current state shape:

```python
STATE_SHAPE = (5, 5, 3, 3, 4, 2, 5)
```

Therefore:

```text
5 × 5 × 3 × 3 × 4 × 2 × 5
= 9000 discrete states
```

Seven actions produce:

```text
9000 × 7 = 63,000
```

tabular state-action entries.

This representation must be reproduced exactly before considering modifications.

Implement unit tests comparing Python and Julia discretization at:

- bin interiors
- every bin boundary
- slightly below every boundary
- slightly above every boundary
- missing stop-sign case
- every TileType
- every DuckThreat
- sigma_stop true/false

---

# 7. CONTINUOUS POLICY STATE

Inspect:

```text
src/continuous_state.py
```

SAC and TD3 do not use the 7-dimensional tabular state directly.

They use a 15-component privileged continuous state:

```text
1.  d
2.  phi
3.  v
4.  kappa
5.  stop_present
6.  d_stop
7.  sigma_stop
8.  duck_present
9.  duck_longitudinal
10. duck_lateral
11. duck_v_longitudinal_relative
12. duck_v_lateral_relative
13. duck_active
14. duck_crossing_available
15. stop_hold_progress
```

The Julia implementation must preserve both:

```text
physical/semantic ContinuousState
```

and:

```text
normalized network observation vector
```

as separate concepts.

Do not conflate them.

For example:

```julia
struct DuckieContinuousState
    ...
end

encode(state, config)::Vector{Float32}
```

should be conceptually distinct.

Preserve the normalization semantics from Python.

---

# 8. VERY IMPORTANT: POLICY STATE IS NOT NECESSARILY THE FULL MARKOV WORLD STATE

The Python policies observe reduced feature representations extracted from Gym-Duckietown.

However, an **online generative planner** must be able to branch from a complete state and generate multiple alternative futures.

Therefore introduce a third representation:

```text
DuckieWorldState
```

This represents the minimum sufficient simulator/world state necessary to reproduce future transitions.

Do NOT simply declare the 7-D or 15-D policy feature vector to be the complete world state without verifying Markov sufficiency.

Audit all hidden variables that affect future dynamics.

Examples likely include:

```text
ego global position
ego heading
ego speed
ego angular motion if relevant

current lane geometry

pedestrian world position
pedestrian direction
pedestrian velocity
pedestrian active state

pedestrian crossing count
pedestrian crossing availability
pedestrian re-arm state

current stop-sign identity
stop hold counter
sigma_stop

episode step count

any additional controller state
```

Determine the **minimum sufficient branchable world state** from the code.

Document every included field and why it is necessary.

Do not add fields without justification.

---

# 9. FORMAL STATE LAYERS

The target architecture should have approximately:

```text
                     DuckieWorldState
                            │
                 ┌──────────┴───────────┐
                 │                      │
                 ▼                      ▼
          Raw/Tabular State      Continuous State
                 │                      │
             discretize               encode
                 │                      │
                 ▼                      ▼
         Q-learning/SARSA           SAC/TD3
```

In mathematical terms:

```text
x_t = full world state
```

and policy representations are functions of it:

\[
s_t^{tabular} = f_{tab}(x_t)
\]

\[
s_t^{continuous} = f_{continuous}(x_t)
\]

This distinction is critical because later:

```text
online planner
      ↓
generative world model
      ↓
DuckieWorldState
```

must support branching.

---

# 10. ACTION SPACE

Inspect:

```text
src/actions.py
```

The actual robot is differential-drive.

The fundamental command is:

\[
a_t = [v_{cmd},\omega_{cmd}]
\]

Do not reinterpret `omega` as an Ackermann steering angle.

The conversion is based on differential-drive inverse kinematics:

\[
u_L = v - \frac{L\omega}{2}
\]

\[
u_R = v + \frac{L\omega}{2}
\]

where `L` is the wheelbase.

Implement a native Julia version and test numerical parity against Python.

---

# 11. DISCRETE ACTION SPACE

The tabular algorithms use seven macro-actions:

```text
fast_left
fast_straight
fast_right

slow_left
slow_straight
slow_right

brake
```

Represent these semantically in Julia, for example:

```julia
@enum DuckieMacroAction begin
    FAST_LEFT
    FAST_STRAIGHT
    FAST_RIGHT
    SLOW_LEFT
    SLOW_STRAIGHT
    SLOW_RIGHT
    BRAKE
end
```

or another idiomatic Julia representation.

Do not assume hard-coded numerical values from source defaults are authoritative.

The YAML configuration associated with each experiment may override values.

Therefore:

```text
experiment YAML > source-code defaults
```

for reproduction.

---

# 12. CONTINUOUS ACTION SPACE

SAC and TD3 operate on:

```text
[v_cmd, omega_cmd]
```

with limits derived from the ActionConfig associated with the experiment.

Create:

```julia
struct DuckieAction
    v::Float64
    omega::Float64
end
```

or equivalent.

Support:

```text
continuous → wheels
macro-action → continuous → wheels
```

through a single canonical control module.

There must not be duplicate kinematics implementations.

---

# 13. FRAME SKIP IS PART OF THE MDP

Inspect `src/env_wrapper.py`.

A single MDP decision does not necessarily correspond to one low-level physics update.

The existing implementation holds one action across:

```text
frame_skip
```

simulator updates.

For typical supplied configurations this is:

```text
frame_skip = 6
```

but never hard-code this as universal.

Treat `frame_skip` as part of the environment configuration.

Conceptually:

\[
x_{t+1}=F^{frame\_skip}(x_t,a_t)
\]

One Julia MDP transition must preserve this decision timescale.

Do not accidentally make one Julia decision correspond to one physics tick if the Python model uses six.

---

# 14. STOP-SIGN STATE AND MEMORY

Inspect:

```text
src/reward.py
```

particularly:

```text
StopTracker
```

The stop process is not purely geometric.

It includes memory:

```text
hold_steps
hold_steps_required
sigma_stop
```

with rules for:

- entering the stopping zone
- consecutive low-speed decisions
- satisfying the stop
- passing the stop sign
- resetting compliance
- changing stop-sign identity
- awarding `full_stop`
- triggering `stop_violation`

This logic must be preserved.

Do not replace it with a simplistic:

```text
if velocity == 0:
    stop = true
```

model.

`stop_hold_progress` is particularly important for the continuous state because it helps preserve Markov information for policies requiring several consecutive stop decisions.

---

# 15. PEDESTRIAN / DUCK CONTROLLER

Inspect:

```text
src/duck_controller.py
```

Understand:

```text
p_cross
dynamic/static configuration
duck injection when missing
spawn position
crossing trigger distance
walk distance
crossings per episode
crossing availability
crossing re-arming
proximity-spawn behavior if configured
stop-sign injection
```

The pedestrian is not merely decorative.

Its controller state can influence future transitions.

Therefore any controller variable necessary for branching must become part of:

```text
DuckieWorldState
```

or another immutable branchable transition-state object.

---

# 16. REWARD MUST BE PORTED, NOT REDESIGNED

Inspect:

```text
src/reward.py
```

The reward is modular and contains components corresponding to concepts such as:

```text
progress
lateral penalty
heading penalty
time penalty
pedestrian behavior
stagnation / unnecessary stopping
stop approach shaping
steering penalty
event rewards
```

and discrete events such as:

```text
duck collision
other collision
offroad
timeout
stop violation
full stop
passed stop
goal
```

Create a Julia `RewardBreakdown`.

For example:

```julia
struct RewardBreakdown
    progress::Float64
    lateral::Float64
    heading::Float64
    time::Float64
    pedestrian::Float64
    stagnation::Float64
    stop_approach::Float64
    steering::Float64
    events::Float64
    total::Float64
end
```

The Julia implementation must provide both:

```text
total reward
```

and:

```text
component-level reward audit
```

because this is useful for validation and visualization.

Again:

**do not replace experiment YAML coefficients with source defaults.**

---

# 17. TERMINATION AND TRUNCATION

The Python implementation distinguishes:

```text
terminated
```

from:

```text
truncated
```

Do not lose this distinction.

Understand the termination reason ordering, including cases such as:

```text
duck_collision
other_collision
timeout
offroad
goal
in_progress
```

Timeout is an experimental horizon, not necessarily an absorbing physical terminal state.

Preserve the semantics required for temporal-difference bootstrapping.

---

# 18. INITIAL STATE DISTRIBUTION

`reset()` is part of the MDP definition.

Audit how the current environment samples initial states and imposes spawn constraints.

Relevant concepts include:

```text
seed
spawn attempts
maximum initial |d|
maximum initial |phi|
route direction
route alignment
position bounds
start tile
controller reset
stop-memory reset
```

Represent this as an explicit initial-state distribution or generator:

```julia
initialstate(mdp)
```

or corresponding `POMDPs.jl` interface.

Reproducibility by RNG/seed is mandatory.

---

# 19. THE CURRENT PYTHON IMPLEMENTATION IS ALREADY SIMULATOR-GENERATIVE

Recognize this explicitly.

Python currently obtains transitions approximately as:

```text
(state, action)
      ↓
Gym-Duckietown physics
      ↓
next state
reward
events
termination
```

It does not maintain an explicit full transition matrix:

\[
P(s'|s,a)
\]

Therefore conceptually the Python environment is already sampling from a generative transition process.

The Julia package should formalize that idea through the current idioms of `POMDPs.jl`.

Before implementing, inspect the locally available/current official `POMDPs.jl` API and do not guess obsolete method signatures.

---

# 20. TARGET GENERATIVE MODEL

For an MDP the conceptual interface is:

\[
G(x_t,a_t,\xi_t)
\rightarrow
(x_{t+1},r_t)
\]

where:

```text
x_t
    DuckieWorldState

a_t
    discrete macro-action or continuous action

xi_t
    stochasticity/RNG
```

The package should expose the appropriate current `POMDPs.jl` generative interface.

Conceptually it should support something similar to:

```julia
result = gen(mdp, state, action, rng)

result.sp
result.r
```

Do not hard-code an interface until you have verified the currently installed/current official API.

---

# 21. DO NOT USE GYM-DUCKIETOWN DIRECTLY FOR EVERY ONLINE-PLANNER TREE BRANCH

An online planner may require thousands of hypothetical transitions for one real action.

Naively performing:

```text
MCTS node
   ↓
PythonCall
   ↓
Gym-Duckietown
   ↓
OpenGL / simulator
```

for every branch may be too slow and difficult to branch deterministically.

Therefore design two conceptual transition backends.

---

# 22. BACKEND A — PYTHON / GYM-DUCKIETOWN REFERENCE BACKEND

Purpose:

```text
ground truth reference
regression testing
trajectory collection
parameter identification
parity validation
final visual comparison
```

Use `PythonCall.jl` if appropriate.

This backend may wrap the existing Python simulator.

However, the Python implementation must remain unchanged wherever possible.

The existing Python repository acts as the reference implementation.

---

# 23. BACKEND B — NATIVE JULIA GENERATIVE BACKEND

Purpose:

```text
fast branching
MCTS
progressive widening
future POMDP online planning
large numbers of simulated futures
```

Implement only the dynamics actually required to reproduce the MDP decision process.

Do not attempt to reproduce the entire Gym-Duckietown rendering engine.

Instead construct a minimal scientifically justified generative world model containing:

```text
ego dynamics
lane geometry
stop logic
pedestrian dynamics
collisions
events
reward
terminal conditions
```

Calibrate and validate this backend against the Python reference.

---

# 24. IMPORTANT: DO NOT INVENT DYNAMICS

Before writing native Julia dynamics:

1. inspect Gym-Duckietown's dynamics used by this project;
2. determine what variables influence one macro-transition;
3. identify deterministic vs stochastic components;
4. identify any actuator lag or integration details;
5. preserve `frame_skip`;
6. reproduce the map/lane geometry necessary for the task;
7. reproduce pedestrian motion used by the controller.

If an exact component cannot initially be reproduced, clearly label it:

```text
APPROXIMATE
```

and implement a parity experiment quantifying the error.

Never silently substitute a simpler model.

---

# 25. JULIA PACKAGE ARCHITECTURE

Create a clean Julia package roughly like:

```text
DuckietownDecisionModels.jl/
│
├── Project.toml
├── Manifest.toml              # if appropriate for reproducible dev
├── README.md
├── LICENSE
│
├── src/
│   ├── DuckietownDecisionModels.jl
│   │
│   ├── config/
│   │   ├── config.jl
│   │   └── yaml_loader.jl
│   │
│   ├── model/
│   │   ├── world_state.jl
│   │   ├── tabular_state.jl
│   │   ├── continuous_state.jl
│   │   ├── actions.jl
│   │   └── discretizer.jl
│   │
│   ├── dynamics/
│   │   ├── ego_dynamics.jl
│   │   ├── lane_geometry.jl
│   │   ├── pedestrian.jl
│   │   ├── stop_tracker.jl
│   │   └── collision.jl
│   │
│   ├── reward/
│   │   ├── events.jl
│   │   └── reward.jl
│   │
│   ├── generative/
│   │   ├── initial_state.jl
│   │   └── transition.jl
│   │
│   ├── backends/
│   │   ├── abstract_backend.jl
│   │   ├── native_julia.jl
│   │   └── gym_duckietown.jl
│   │
│   ├── interfaces/
│   │   ├── pomdps.jl
│   │   └── rl_environment.jl
│   │
│   ├── solvers/
│   │   └── adapters.jl
│   │
│   ├── evaluation/
│   │   ├── rollout.jl
│   │   ├── parity.jl
│   │   └── metrics.jl
│   │
│   └── visualization/
│       ├── world_view.jl
│       ├── policy_slice.jl
│       ├── rollout_animation.jl
│       ├── search_tree.jl
│       └── diagnostics.jl
│
├── configs/
│
├── examples/
│   ├── 01_random_rollout.jl
│   ├── 02_tabular_policy.jl
│   ├── 03_continuous_policy.jl
│   ├── 04_mcts.jl
│   ├── 05_dpw.jl
│   └── 06_visualization.jl
│
├── test/
│   ├── runtests.jl
│   ├── test_actions.jl
│   ├── test_discretizer.jl
│   ├── test_state_extraction.jl
│   ├── test_stop_tracker.jl
│   ├── test_reward.jl
│   ├── test_dynamics.jl
│   └── test_python_parity.jl
│
└── scripts/
    ├── collect_python_reference.jl
    ├── validate_parity.jl
    └── generate_figures.jl
```

This structure is a proposal.

Adjust it if Julia package conventions strongly justify a different arrangement, but explain the reason before changing it.

---

# 26. PUBLIC PACKAGE API

The package should feel like a benchmark problem rather than a training script.

Target usability:

```julia
using DuckietownDecisionModels

mdp = DuckietownMDP(
    scenario = :full_task,
    representation = :world,
    backend = :native,
)
```

Other convenience constructors may include:

```julia
DuckietownLaneFollowing()
DuckietownStopSign()
DuckietownPedestrian()
DuckietownFullTask()
```

Possible state projections:

```julia
tabular_state(mdp, world_state)
continuous_state(mdp, world_state)
encode(mdp, continuous_state)
```

Actions:

```julia
macro_action(mdp, FAST_LEFT)
continuous_action(...)
wheel_commands(...)
```

The user must not need to know internal implementation details merely to instantiate the problem.

---

# 27. SOLVER-INDEPENDENT PROBLEM DEFINITION

This is a central design goal.

Do not create separate environments with divergent logic for:

```text
Q-learning
SARSA
SAC
TD3
MCTS
DPW
```

They must share the same underlying task definition whenever scientifically appropriate.

Conceptually:

```text
                         DuckietownMDP
                              │
                 shared state/dynamics/reward
                              │
          ┌───────────┬───────┴──────┬───────────┐
          │           │              │           │
          ▼           ▼              ▼           ▼
    Q-learning      SARSA          SAC/TD3     MCTS/DPW
```

Differences in observation/state projection or action representation must be explicit configuration choices, not duplicated environmental logic.

---

# 28. EXISTING SOLVERS

The original project already contains results/policies for:

```text
Q-learning
SARSA
SAC
TD3
```

The first goal is **not** to exactly reproduce PyTorch neural-network weights inside Julia.

The first goal is to reproduce:

```text
problem formulation
state semantics
action semantics
transition semantics
reward semantics
episode semantics
```

Once model parity is established, add solver support.

For tabular methods, a Julia implementation can be compared directly.

For SAC/TD3, choose one of these routes after the model is validated:

```text
A. native Julia implementation
B. compatible Julia RL library
C. Python policy inference adapter for reference comparison
```

Do not block the entire package on deep-policy weight conversion.

---

# 29. ONLINE SOLVER — DISCRETE MCTS

The existing seven macro-actions provide a natural discrete online-planning baseline.

Once the native generative model passes validation, integrate a current Julia MCTS solver compatible with `POMDPs.jl`.

Conceptually:

```julia
planner = solve(mcts_solver, mdp)

action = planner(current_state)
```

Exact syntax must follow the currently installed/current documented API.

The solver must perform hypothetical rollouts using the **native branchable generative model**, not mutate the real episode state.

---

# 30. ONLINE SOLVER — CONTINUOUS ACTION PLANNING

Continuous actions are:

\[
a=[v_{cmd},\omega_{cmd}]
\]

A naive search cannot enumerate an infinite action set.

Therefore integrate an appropriate progressive-widening approach, such as Double Progressive Widening if supported by the selected current Julia solver.

Conceptually:

```text
current state
     │
     ├── sampled continuous action
     ├── sampled continuous action
     ├── sampled continuous action
     │
     └── progressively add actions
```

This provides a natural online-planning comparison with SAC/TD3.

---

# 31. FUTURE POMDP EXTENSION

Do **not** implement the complete POMDP until the MDP generative model is validated.

However, make architectural choices that allow later extension to:

```text
DuckietownPOMDP
```

with:

\[
G(s,a,\xi)
\rightarrow
(s',o,r)
\]

Future observation flow may include:

```text
camera
  ↓
detector
  ↓
measurement
  ↓
belief update
  ↓
online POMDP planner
```

Possible future online POMDP solvers include particle/tree-search approaches appropriate for continuous state and observations.

The MDP package should not hard-code assumptions that prevent this extension.

---

# 32. PARITY VALIDATION IS MANDATORY

Do not declare the Julia port complete merely because a robot can drive.

Create systematic Python-vs-Julia parity tests.

For identical:

```text
configuration
seed
initial condition
action sequence
decision timestep
```

compare relevant quantities.

For example:

```text
ego x
ego z
ego heading
speed
lane d
lane phi
curvature/tile
distance to stop
sigma_stop
stop hold progress
pedestrian position
pedestrian state
reward components
event flags
termination reason
```

Where exact equality is expected, require exact equality.

Where numerical integration creates floating-point differences, establish justified tolerances.

---

# 33. DO NOT INVENT PARITY THRESHOLDS

Do not arbitrarily decide:

```text
position RMSE < 0.02 m
```

or another threshold without evidence.

Instead:

1. generate matched trajectories;
2. inspect numerical error distributions;
3. identify floating-point/integration limitations;
4. define justified acceptance criteria;
5. record the rationale.

---

# 34. VALIDATION LEVELS

Create at least three validation levels.

## Level A — deterministic unit parity

Examples:

```text
action → wheel conversion
discretization
reward arithmetic
stop tracker
curvature classification
state encoding
```

## Level B — one-step transition parity

For sampled:

```text
state-action pairs
```

compare Python and Julia one-macro-step transitions.

## Level C — rollout parity

Use the same action sequence for entire episodes.

Compare:

```text
trajectory
events
reward
termination
```

---

# 35. VISUALIZATION IS A CORE PACKAGE FEATURE

Visualization must not be treated as an afterthought.

The package should visually feel like a scientific benchmark such as GridWorld, while respecting the continuous road geometry of Duckietown.

Do **not** make the default scientific visualization simply a screenshot of the Gym-Duckietown camera.

Create a native Julia schematic top-down renderer.

Prefer the current Makie ecosystem or another suitable native Julia scientific visualization stack after verifying package compatibility.

---

# 36. VISUALIZATION 1 — DUCKIETOWN WORLD VIEW

Implement something conceptually like:

```julia
render(mdp, state)
```

The default world view should display a top-down representation including:

```text
road geometry
lane boundaries
lane centreline

Duckiebot position
Duckiebot heading
velocity vector

current trajectory

stop sign
stop line
distance to stop

Duckie/pedestrian position
pedestrian trajectory
crossing state

selected action
candidate actions where relevant
```

Example conceptual view:

```text
                     pedestrian
                         ●
                         │ crossing path

        ┌──────────────────────────────┐
        │                              │
========│============ STOP ============│========
        │             ║                │
        │        · · · · ·             │
        │     🚗────────────→           │
        │                              │
========│==============================│========
```

This is the Duckietown equivalent of rendering the agent inside GridWorld.

---

# 37. WORLD VIEW STATE PANEL

Alongside or beneath the world representation display current state variables.

For example:

```text
Decision step          143

d                     -0.031 m
phi                   +0.047 rad
v                      0.284 m/s
kappa                 +1.31 1/m

stop present           true
d_stop                 0.72 m
sigma_stop             false
stop hold              1 / 3

duck present           true
duck longitudinal      0.81 m
duck lateral          -0.14 m
duck active            true

action
v_cmd                  0.17 m/s
omega_cmd              0.00 rad/s

reward                 +0.193
```

The visualization should allow the viewer to immediately understand:

```text
what the world looks like
what state the MDP sees
what action was selected
what reward resulted
```

---

# 38. VISUALIZATION 2 — GRIDWORLD-LIKE POLICY SLICE

A 15-dimensional policy cannot be rendered directly.

Therefore implement **policy slices**.

Hold all but two variables fixed.

Example:

\[
x=d
\]

\[
y=\phi
\]

while fixing:

```text
v
kappa
stop state
duck state
```

Then display the selected policy over the 2-D plane.

Conceptually:

```text
              phi

       ↑ ↑ ↑ ↖ ↖
       ↗ ↑ ↑ ↖ ←
       → → • ← ←
       ↘ ↓ ↓ ↙ ←
       ↓ ↓ ↓ ↙ ↙

              d →
```

For discrete policies:

```text
arrow / icon / label = selected macro-action
```

For continuous policies:

use an interpretable combination of:

```text
v_cmd
omega_cmd
```

for example:

```text
arrow orientation → steering/yaw direction
arrow magnitude   → forward speed
```

Avoid a visualization that is beautiful but physically ambiguous.

---

# 39. REQUIRED POLICY-SLICE PRESETS

At minimum support useful projections such as:

```julia
plot_policy_slice(policy, mdp, :d, :phi)
```

Lane control.

```julia
plot_policy_slice(policy, mdp, :d_stop, :v)
```

Stop-sign behavior.

```julia
plot_policy_slice(
    policy,
    mdp,
    :duck_longitudinal,
    :duck_lateral
)
```

Pedestrian behavior.

Potential additional projection:

```text
duck longitudinal × relative longitudinal velocity
```

for collision/yielding analysis.

Every plot must clearly display which remaining state variables are held constant.

---

# 40. VISUALIZATION 3 — VALUE / Q VISUALIZATION

For tabular methods provide benchmark-style maps such as:

```text
max_a Q(s,a)
```

for selected state slices.

Examples:

```text
d × tracking_error
d_stop × velocity
duck_threat × velocity
```

Allow:

```text
heatmap of V(s)
```

plus:

```text
best-action overlay
```

similar to classic GridWorld visualization.

---

# 41. VISUALIZATION 4 — TRAJECTORY / ROLLOUT

Implement:

```julia
render_rollout(...)
```

or equivalent.

Show the full episode trajectory in the world view.

Annotate relevant events:

```text
start
stop approach
full stop
pedestrian crossing
yield
collision
offroad
goal
termination
```

The same function should allow visual comparison between:

```text
Q-learning
SARSA
SAC
TD3
MCTS
DPW
```

when their trajectories are generated under the same scenario.

---

# 42. VISUALIZATION 5 — ANIMATION

Support creation of a rollout animation suitable for:

```text
research presentation
supplementary material
debugging
```

For each decision step display:

```text
world
current state
selected action
immediate reward
cumulative reward
termination/event indicators
```

Where practical support export to standard formats such as GIF or video.

Do not rely solely on interactive display.

---

# 43. VISUALIZATION 6 — ONLINE SEARCH TREE

For MCTS/DPW, create a search-tree visualization.

Conceptually:

```text
                         s0
                     N=1000
                  V=estimated

              /          |          \
             /           |           \
         fast          slow         brake
          N=...         N=...        N=...
           │             │             ★
          s1            s2            s3
```

Display useful quantities such as:

```text
node visits
Q estimate
action
depth
immediate reward
selected branch
```

Do not attempt to show all thousands of nodes by default.

Support:

```text
depth filtering
top-k branches
visit filtering
```

The chosen action should be visually obvious.

---

# 44. LINK SEARCH TREE TO WORLD SEMANTICS

If technically practical, make the online planner visualization more useful than a generic tree.

Selecting or inspecting a search-tree node should allow visualization of that hypothetical node's:

```text
DuckieWorldState
```

in the top-down world renderer.

Thus the researcher can answer:

> What future did MCTS imagine when it chose this action?

This is valuable for debugging even though explainability itself is outside this project.

---

# 45. VISUALIZATION 7 — DIAGNOSTIC TIME SERIES

Provide episode diagnostic plots for quantities such as:

```text
d vs time
phi vs time
v vs time
omega_cmd vs time
v_cmd vs time
d_stop vs time
stop_hold_progress vs time
pedestrian distance vs time
reward vs time
cumulative reward vs time
```

Also allow reward-component visualization:

```text
progress
lateral
heading
pedestrian
stop approach
events
...
```

This should use the same rollout log produced by the simulator.

Avoid maintaining a separate visualization-only data source.

---

# 46. PAPER-QUALITY COMPOSITE FIGURE

Create an example script capable of producing a static publication-quality figure conceptually like:

```text
┌──────────────────────┬──────────────────────┐
│ A. Duckietown World  │ B. Policy Slice      │
│                      │                      │
│   road + trajectory  │       phi            │
│   stop + pedestrian  │   policy field       │
│                      │             d        │
├──────────────────────┼──────────────────────┤
│ C. Search Tree       │ D. Episode Metrics   │
│                      │                      │
│       s0             │ d ─────────          │
│      / | \           │ v ────────           │
│    a1 a2 a3          │ R ─────────          │
└──────────────────────┴──────────────────────┘
```

This should be generated entirely from logged model/planner data rather than manually constructed values.

---

# 47. DEFAULT RENDERING PHILOSOPHY

The default scientific renderer should answer:

```text
Where is the robot?
What is happening around it?
What state does the model represent?
What action did the policy choose?
Why is this transition important?
```

It should not attempt photorealism.

Prefer:

```text
clarity
geometry
state visibility
reproducibility
publication readability
```

over visual decoration.

Gym-Duckietown's rendered camera can remain an optional comparison/validation view.

---

# 48. SOLVER COMPARISON VISUALIZATION

Provide a common interface allowing multiple solver outputs to be compared.

Example:

```julia
compare_rollouts(
    mdp,
    Dict(
        :q_learning => q_policy,
        :sarsa => sarsa_policy,
        :sac => sac_policy,
        :td3 => td3_policy,
        :mcts => mcts_policy,
    );
    seed = 101
)
```

Potential comparison metrics:

```text
progress
return
mean |d|
mean |phi|
collisions
stop compliance
pedestrian yielding
brake ratio
episode length
planning time per action
```

Do not invent new success criteria unless clearly separated from the criteria already stored in experiment configurations.

---

# 49. PERFORMANCE PROFILING FOR ONLINE PLANNING

A generative model intended for MCTS/DPW must be fast.

Benchmark:

```text
one generative step
100 generative steps
1,000 generative steps
10,000 generative steps
```

Measure:

```text
latency
allocations
memory
```

Where appropriate use Julia performance best practices:

```text
concrete types
type stability
limited allocation
immutable branchable states where useful
explicit RNG
```

Do not sacrifice correctness merely to obtain faster numbers.

Correctness first, then optimization.

---

# 50. REPRODUCIBILITY

Every experiment must be controlled by:

```text
config
seed
solver configuration
model version
backend
```

A run should produce a manifest/log containing at minimum:

```text
timestamp
git commit if available
Julia version
package versions
scenario
backend
seed
model parameters
solver parameters
```

Do not mutate original Python experiment configurations.

If converting YAML to Julia-native configuration, retain the source YAML path and values in the run metadata.

---

# 51. TESTING PHILOSOPHY

Tests should not merely ask:

```text
does function run?
```

They should test semantic invariants.

Examples:

### Action invariant

```text
omega = 0
→ left wheel == right wheel
```

### Stop invariant

```text
sigma_stop cannot become true before required consecutive hold steps
```

### State invariant

```text
encoded continuous observation must stay inside declared bounds
```

### Discretization invariant

```text
every valid RawState maps inside STATE_SHAPE
```

### Branching invariant

Calling:

```text
gen(s, a1)
gen(s, a2)
```

must not mutate `s`.

### RNG invariant

Identical:

```text
state
action
seed
```

must reproduce the same stochastic transition where determinism is expected.

---

# 52. IMPLEMENTATION ROADMAP

Do not implement everything simultaneously.

Use the following gates.

---

## FJ0 — Repository Audit

Deliver:

```text
repository structure
relevant files
ignored explainability files
state definitions
action definitions
reward definition
transition flow
stop logic
pedestrian logic
termination logic
configuration hierarchy
```

Also identify any ambiguity.

**No major implementation before this audit exists.**

---

## FJ1 — Julia Package Skeleton

Create:

```text
Project.toml
src/
test/
examples/
```

with minimum package import working:

```julia
using DuckietownDecisionModels
```

No complex solver yet.

---

## FJ2 — Pure Semantic Port

Implement and test:

```text
state structures
action structures
discretizer
action conversion
reward components
stop tracker
configuration loader
```

These should not require the complete simulator.

Run Python-vs-Julia unit parity.

Gate:

```text
SEMANTIC PARITY PASSED
```

---

## FJ3 — WorldState Definition

Audit latent simulator/controller state.

Define:

```text
DuckieWorldState
```

Document why each field exists.

Implement:

```text
world → RawState
world → ContinuousState
world → encoded observation
```

where possible.

Gate:

```text
STATE SUFFICIENCY REVIEW PASSED
```

---

## FJ4 — Python Reference Backend

Create an adapter capable of running controlled reference transitions using the existing Gym-Duckietown implementation.

Requirements:

```text
seed control
state logging
action logging
reward-component logging
event logging
termination logging
```

Do not modify the reference model merely to simplify Julia.

---

## FJ5 — Native Julia Generative Model

Implement:

```text
initial-state generator
ego dynamics
lane geometry
pedestrian/controller dynamics
stop state
event detection
reward
termination
```

Expose the appropriate current `POMDPs.jl` generative interface.

Gate:

```text
GEN MODEL FUNCTIONAL
```

not yet necessarily parity-certified.

---

## FJ6 — Python ↔ Julia Parity

Run:

```text
unit parity
one-step parity
multi-step rollout parity
```

Generate reports and plots.

Classify mismatches as:

```text
BUG
EXPECTED NUMERICAL DIFFERENCE
KNOWN APPROXIMATION
UNKNOWN
```

Do not proceed silently with unknown major mismatch.

Gate:

```text
GENERATIVE MODEL PARITY ACCEPTED
```

---

## FJ7 — Existing Solver Baselines

Add/adapt:

```text
Q-learning
SARSA
SAC
TD3
```

as appropriate.

The same underlying problem formulation must be reused.

Generate benchmark rollouts.

---

## FJ8 — Online MDP Planning

Add:

```text
MCTS for discrete actions
progressive-widening planner for continuous actions
```

Benchmark:

```text
quality
planning latency
number of generative calls
```

Gate:

```text
ONLINE PLANNING FUNCTIONAL
```

---

## FJ9 — Visualization Suite

Implement:

```text
world renderer
state panel
policy slice
Q/value slice
trajectory
animation
MCTS/DPW tree
diagnostic plots
paper figure
```

Gate:

```text
SCIENTIFIC VISUALIZATION COMPLETE
```

---

## FJ10 — Future POMDP Readiness

Do not necessarily implement POMDP yet.

Review whether architecture cleanly supports:

```text
observation model
belief representation
belief updater
POMCP/POMCPOW-style solver
```

Produce a short architectural note.

---

# 53. EXPECTED END-USER EXPERIENCE

A user should eventually be able to do something conceptually similar to:

```julia
using DuckietownDecisionModels
using POMDPs

mdp = DuckietownFullTask(
    backend = :native
)

s0 = rand(initialstate(mdp))

render(mdp, s0)
```

Discrete online planning:

```julia
planner = create_mcts_planner(mdp)

a = action(planner, s0)
```

Continuous online planning:

```julia
mdp_continuous = DuckietownFullTask(
    action_space = :continuous,
    backend = :native
)

planner = create_dpw_planner(mdp_continuous)

a = action(planner, s0)
```

Policy visualization:

```julia
plot_policy_slice(
    policy,
    mdp,
    :d,
    :phi;
    v = 0.30,
    stop_present = false,
    duck_present = false,
)
```

Rollout:

```julia
history = simulate_episode(
    mdp,
    policy;
    seed = 101
)

render_rollout(mdp, history)
```

Search tree:

```julia
render_search_tree(planner)
```

Exact public names may be adjusted to follow Julia conventions, but the package must remain this simple to use.

---

# 54. SCIENTIFIC GOAL

The scientific goal is not:

> We translated a Python Gym environment into Julia.

The desired result is:

> We formulated the Duckietown autonomous-driving task as a reusable Julia MDP decision model with explicit state semantics, action semantics, generative dynamics, reward structure, reproducible initial-state distribution, multiple solver interfaces, online planning capability, and benchmark-style scientific visualization.

The package should make it possible to study several decision-making paradigms on the same problem:

```text
                     DUCKIETOWN MDP
                           │
                same decision problem
                           │
       ┌─────────┬─────────┼─────────┬──────────┐
       │         │         │         │          │
       ▼         ▼         ▼         ▼          ▼
 Q-learning    SARSA      SAC       TD3      MCTS/DPW
       │         │         │         │          │
       └─────────┴─────────┴─────────┴──────────┘
                           │
                    same evaluation
                           │
                    same visualization
```

This solver-independent formulation is a primary design objective.

---

# 55. LONG-TERM RESEARCH GOAL

The MDP implementation should provide a controlled baseline for a future partially observable formulation:

```text
MDP
│
│ full/privileged state
│
├── Q-learning
├── SARSA
├── SAC
├── TD3
├── MCTS
└── DPW

          ↓ introduce partial observability

POMDP
│
│ camera / detector / measurement
│
│ belief state
│
├── belief-aware learned policy
├── POMCP-like planner
└── POMCPOW-like planner
```

Therefore:

**do not architect the MDP package in a way that makes an observation model or belief layer impossible later.**

---

# 56. NON-GOALS

Do not:

- port the explainability framework;
- redesign the research question;
- replace Gym-Duckietown with a completely unrelated toy simulator;
- change reward coefficients merely because another reward seems better;
- remove stop compliance memory;
- remove pedestrian controller state needed for Markov transitions;
- conflate policy features with world state;
- convert everything into a giant monolithic environment class;
- hard-code one solver into the model;
- perform MCTS directly by mutating the live real environment;
- claim exact Python parity without measurements;
- claim the Julia dynamics are exact if they are approximate;
- create visualizations using invented values instead of actual rollout data;
- prioritize photorealistic graphics over scientific interpretability;
- start POMDP implementation before the MDP generative model is validated.

---

# 57. CODING QUALITY REQUIREMENTS

Use idiomatic Julia.

Prioritize:

```text
type stability
clear immutable data structures where appropriate
explicit randomness
modularity
testability
reproducibility
solver independence
minimal hidden mutable global state
```

Functions should be small enough that:

```text
state extraction
transition
reward
event detection
visualization
```

can be tested independently.

Use docstrings for public interfaces.

Use clear physical units in field names/docstrings:

```text
metres
seconds
radians
metres/second
radians/second
```

Never mix:

```text
degrees/radians
wheel commands/physical velocity
physics steps/MDP decision steps
```

without explicit conversion.

---

# 58. REPORTING DURING IMPLEMENTATION

At the end of every FJ gate report:

```text
STATUS
PASSED / LIMITED / FAILED

WHAT WAS IMPLEMENTED

FILES CREATED/MODIFIED

TESTS RUN

RESULTS

KNOWN DIFFERENCES FROM PYTHON

NEXT GATE
```

If a gate is `LIMITED`, explain precisely what prevents it from being `PASSED`.

Never hide unresolved mismatches.

---

# 59. FIRST RESPONSE REQUIRED FROM YOU

Do **not** start by generating dozens of Julia files.

Your first response must be an audit of the supplied repository.

Return:

## A. Relevant folder map

Show which files are:

```text
CORE MDP
SOLVER
CONFIG
REFERENCE POLICY
IGNORED EXPLAINABILITY
```

## B. Reconstructed Python MDP

Describe from source:

```text
S
A
T
R
gamma
rho_0
terminal/truncation
```

## C. State hierarchy

Identify:

```text
latent simulator variables
RawState
discrete state
ContinuousState
```

and propose the exact contents of:

```text
DuckieWorldState
```

with justification for every field.

## D. Transition diagram

Show the current Python transition sequence step-by-step.

For example:

```text
current world
    ↓
controller.before_step
    ↓
action
    ↓
(v, omega)
    ↓
wheel conversion
    ↓
frame_skip physics updates
    ↓
new world
    ↓
state extraction
    ↓
stop tracker
    ↓
event detection
    ↓
reward
    ↓
termination
```

Correct this diagram based on the actual source.

## E. Julia architecture proposal

Map every Python component to a proposed Julia module.

## F. Generative-model strategy

Explain:

```text
Python reference backend
vs
native Julia branchable backend
```

and identify what must be validated.

## G. Visualization plan

Describe concrete implementations for:

```text
World View
Policy Slice
Value/Q Slice
Rollout
Animation
Search Tree
Diagnostics
Paper Figure
```

## H. Risks

List technical risks such as:

```text
incomplete world-state cloning
hidden Gym simulator state
lane geometry mismatch
pedestrian/controller state mismatch
frame-skip mismatch
stop-memory mismatch
continuous action branching cost
performance of online planner
```

## I. FJ0 verdict

Finish with:

```text
FJ0 STATUS:
PASSED / LIMITED / FAILED
```

Only after FJ0 is understood should implementation proceed.

---

# 60. FINAL PRINCIPLE

The package should eventually make Duckietown feel as easy to instantiate and analyze as a canonical MDP benchmark:

```julia
mdp = DuckietownMDP(...)
```

while internally preserving the richer autonomous-driving structure:

```text
lane geometry
differential-drive dynamics
stop compliance
pedestrian interaction
continuous control
safety events
```

The target is therefore:

> **A reusable, solver-independent, generative Julia decision-model package for Duckietown that preserves the validated Python MDP, supports both learned and online-planning solvers, provides Python–Julia parity evidence, and includes benchmark-quality native scientific visualization comparable in usability to GridWorld while remaining faithful to road-driving geometry.**

Begin with **FJ0 Repository Audit**. Do not implement later gates until the existing Python MDP has been reconstructed from source.