# FJ3.2 fixture generator: ego DB18 delayed dynamics, layered trace.
# Run in the REFERENCE venv: conda activate ddm-ref
#   (Python 3.9, numpy 1.20.0, gym 0.23.1, duckietown-gym-daffy 6.1.34,
#    duckietown-world-daffy 6.4.3, PyGeometry-z6 2.1.5)
# Usage: python tools/parity/gen_fj3_ego_fixtures.py
import json, math, sys, os
out_path = sys.argv[1] if len(sys.argv) > 1 else (
    "/home/pannntastic/aivnv/DuckietownDecisionModels.jl/test/fixtures/fj3_ego.json")
os.environ.setdefault("PYGLET_HEADLESS", "1")
import numpy as np
import geometry as geo
from gym_duckietown.envs import DuckietownEnv
import gym_duckietown.simulator as sim
from duckietown_world.world_duckietown.pwm_dynamics import (
    DynamicModel, DynamicModelParameters, PWMCommands)
from duckietown_world.world_duckietown.dynamics_delay import DelayedDynamics

sys.path.insert(0, "/home/pannntastic/aivnv/duckduck")
from src.duck_controller import DuckControllerConfig, prepare_task_map_data

def enc(x):
    if isinstance(x, np.ndarray):
        return [enc(v) for v in x]
    if isinstance(x, (list, tuple)):
        return [enc(v) for v in x]
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        f = float(x)
        if math.isnan(f):
            return {"nonfinite": "nan"}
        if math.isinf(f):
            return {"nonfinite": "inf" if f > 0 else "-inf"}
        if f == 0.0 and math.copysign(1.0, f) < 0.0:
            return {"nonfinite": "-0.0"}
        return f
    if x is None:
        return None
    raise TypeError(type(x))

def v3(a):
    return enc(np.asarray(a, dtype=float))

def m3(a):
    return enc(np.asarray(a, dtype=float))

out = {}

env = DuckietownEnv(
    map_name="small_loop", domain_rand=False, max_steps=9000, frame_skip=6,
    user_tile_start=(0, 1), accept_start_angle_deg=10, seed=53,
)
cfg = DuckControllerConfig(
    p_cross=1.0, make_dynamic=True, require_duck=True, inject_if_missing=True,
    spawn_pos=[1.62, 0.50], spawn_rotate=0.0, spawn_height=0.08, walk_distance=0.90,
    trigger_min_ego_distance=0.35, trigger_max_ego_distance=0.45,
    max_crossings_per_episode=1, inject_stop_if_missing=True, require_stop=True,
    stop_spawn_pos=[1.20, 2.10], stop_spawn_rotate=180.0, stop_spawn_height=0.18,
)
prepared, n_ducks, n_stops = prepare_task_map_data(env.map_data, cfg)
env.map_data = prepared
env._interpret_map(prepared)
env.seed(53)
env.reset()

out["delta_time"] = enc(float(env.delta_time))
out["grid_height"] = int(env.grid_height)
out["tile_size"] = enc(float(env.road_tile_size))
out["init_pos"] = v3(env.cur_pos)
out["init_angle"] = enc(float(env.cur_angle))
out["init_q0"] = m3(env.cartesian_from_weird(env.cur_pos, env.cur_angle))
out["init_v0"] = m3(env.state.TSE2_from_state()[1])

# DB18 nominal parameters (as constructed by get_DB18_nominal)
p = env.state.state.parameters
out["params"] = {
    "u1": enc(float(p.u1)), "u2": enc(float(p.u2)), "u3": enc(float(p.u3)),
    "w1": enc(float(p.w1)), "w2": enc(float(p.w2)), "w3": enc(float(p.w3)),
    "u_alpha_r": enc(float(p.u_alpha_r)), "u_alpha_l": enc(float(p.u_alpha_l)),
    "w_alpha_r": enc(float(p.w_alpha_r)), "w_alpha_l": enc(float(p.w_alpha_l)),
    "wheel_radius_left": enc(float(p.wheel_radius_left)),
    "wheel_radius_right": enc(float(p.wheel_radius_right)),
    "wheel_distance": enc(float(p.wheel_distance)),
    "encoder_resolution_rad": enc(float(p.encoder_resolution_rad)),
}
out["delay"] = enc(float(env.state.delay))

# --- SE2 primitive samples (PyGeometry-z6 reference) ---
se2_cases = [
    ([0.0, 0.0], 0.0),
    ([0.1, 0.0], 0.0),
    ([0.0, 0.0], 0.05),
    ([0.02, -0.01], -0.3),
    ([0.14732, 0.0], 0.00234),
    ([-0.05, 0.03], 1.2),
]
out["se2_cases"] = []
for (lin, ang) in se2_cases:
    v = geo.se2_from_linear_angular(list(lin), ang)
    dt = 1.0 / 30.0
    diff = geo.SE2.group_from_algebra(dt * v)
    out["se2_cases"].append({
        "lin": [enc(lin[0]), enc(lin[1])], "ang": enc(ang),
        "dt": enc(dt),
        "se2": m3(v),
        "diff": m3(diff),
    })

q0 = out["init_q0"]
out["mult_cases"] = []
for (lin, ang) in [(0.12, 0.0), (-0.03, 0.04), (0.0, -0.02)]:
    v = geo.se2_from_linear_angular([lin, 0.0], ang)
    diff = geo.SE2.group_from_algebra((1.0 / 30.0) * v)
    q1 = geo.SE2.multiply(np.array(q0, dtype=float), diff)
    t, a = geo.translation_angle_from_SE2(q1)
    out["mult_cases"].append({
        "lin": enc(lin), "ang": enc(ang), "diff": m3(diff),
        "q1": m3(q1), "t": v3(t), "angle": enc(float(a)),
    })

out["cart_cases"] = []
for (pos, angle) in [(out["init_pos"], out["init_angle"]),
                     ([0.10, 0.0, 0.60], 1.57),
                     ([1.20, 0.0, 1.70], -0.9)]:
    q = env.cartesian_from_weird(pos, angle)
    pos2, angle2 = env.weird_from_cartesian(q)
    out["cart_cases"].append({
        "pos": v3(pos), "angle": enc(float(angle)),
        "q": m3(q), "pos_back": v3(pos2), "angle_back": enc(float(angle2)),
    })

# --- DB18 model cases (u, w, motor commands -> acceleration) ---
out["model_cases"] = []
for u in (0.0, 0.5, 2.0, 10.0):
    for w in (0.0, 0.5, 2.0, -1.5):
        for (mL, mR) in [(-1.0, -1.0), (-0.5, 0.5), (0.0, 0.0), (0.35, 0.1), (1.0, -1.0), (1.0, 1.0)]:
            xdd = DynamicModel.model(PWMCommands(mL, mR), p, u=u, w=w)
            out["model_cases"].append({
                "u": enc(u), "w": enc(w),
                "mL": enc(mL), "mR": enc(mR),
                "x_dot_dot": [enc(float(xdd[0, 0])), enc(float(xdd[1, 0]))],
            })

# --- Delayed command selection unit cases (get_commands_at) ---
def make_dd(ts, cs, delay, t0=0.0):
    dd = object.__new__(DelayedDynamics)
    dd.commands = list(cs)
    dd.timestamps = list(ts)
    dd.delay = delay
    dd.u0 = PWMCommands(0.0, 0.0)
    dd.state = None
    dd.t = t0
    return dd

ts_base = [i / 30.0 for i in range(10)]
cs_base = [PWMCommands(0.5, 0.2), PWMCommands(-0.3, 0.7), PWMCommands(0.9, 0.1),
           PWMCommands(-0.8, -0.4), PWMCommands(0.2, 0.6), PWMCommands(0.0, 0.0),
           PWMCommands(0.7, -0.9), PWMCommands(-0.1, 0.3), PWMCommands(0.6, -0.6),
           PWMCommands(0.4, 0.4)]
queries = [-0.5, 0.0, 1e-9, 1.0 / 30.0, 0.03333333333333333, 0.033333333333333335,
           0.05, 0.099999, 0.1, 0.100001, 0.16666666666666666, 0.2, 0.2833333333333333,
           0.2999999999999999, 0.3]
out["delay_cases"] = []
for q in queries:
    dd = make_dd(ts_base, cs_base, 0.15)
    i, told, uc = dd.get_commands_at(q)
    out["delay_cases"].append({
        "t": enc(float(q)),
        "i": int(i),
        "told": enc(float(told)),
        "used": [enc(float(uc.motor_left)), enc(float(uc.motor_right))],
    })
# variant with a non-regular timestamp grid
ts_irr = [0.0, 0.023, 0.059, 0.101, 0.148, 0.197, 0.250, 0.301]
cs_irr = [PWMCommands(0.1, 0.1), PWMCommands(-0.2, 0.2), PWMCommands(0.3, -0.3),
          PWMCommands(-0.4, 0.4), PWMCommands(0.5, -0.5), PWMCommands(-0.6, 0.6),
          PWMCommands(0.7, -0.7), PWMCommands(-0.8, 0.8)]
for q in [0.0, 0.031, 0.07, 0.101, 0.102, 0.16, 0.251, 0.3001]:
    dd = make_dd(ts_irr, cs_irr, 0.15)
    i, told, uc = dd.get_commands_at(q)
    out["delay_cases"].append({
        "grid": "irregular",
        "t": enc(float(q)),
        "i": int(i),
        "told": enc(float(told)),
        "used": [enc(float(uc.motor_left)), enc(float(uc.motor_right))],
    })

# --- 18-tick layered trace via patched DynamicModel.integrate ---
out["ticks"] = []
trace = []
orig_dm_integrate = DynamicModel.integrate

def patched_dm_integrate(self, dt, commands):
    la = geo.linear_angular_from_se2(self.v0)
    longit_prev = float(la[0][0])
    angular_prev = float(la[1])
    xdd = self.model(commands, self.parameters, u=longit_prev, w=angular_prev)
    longitudinal = longit_prev + dt * float(xdd[0, 0])
    angular = angular_prev + dt * float(xdd[1, 0])
    commands_se2 = geo.se2_from_linear_angular([longitudinal, 0.0], angular)
    res = orig_dm_integrate(self, dt, commands)
    d = self.parameters.wheel_distance
    Rr = self.parameters.wheel_radius_right
    Rl = self.parameters.wheel_radius_left
    M = np.array([[Rr / d, -Rl / d], [Rr / 2, Rl / 2]])
    MInv = np.linalg.inv(M)
    wRL = MInv @ np.array([[angular], [longitudinal]])
    wR = float(wRL[0, 0])
    wL = float(wRL[1, 0])
    trace.append({
        "used": [enc(float(commands.motor_left)), enc(float(commands.motor_right))],
        "longit_prev": enc(longit_prev), "angular_prev": enc(angular_prev),
        "x_dot_dot": [enc(float(xdd[0, 0])), enc(float(xdd[1, 0]))],
        "longitudinal": enc(float(longitudinal)), "angular": enc(float(angular)),
        "commands_se2": m3(commands_se2),
        "q1": m3(res.q0), "v1": m3(res.v0),
        "wL": enc(wL), "wR": enc(wR),
        "d_axis_left": enc(float(res.axis_left_rad - self.axis_left_rad)),
        "d_axis_right": enc(float(res.axis_right_rad - self.axis_right_rad)),
        "axis_left_rad": enc(float(res.axis_left_rad)),
        "axis_right_rad": enc(float(res.axis_right_rad)),
        "axis_left_obs": enc(float(res.axis_left_obs_rad)),
        "axis_right_obs": enc(float(res.axis_right_obs_rad)),
        # reference `DynamicModel.__init__` computes ticks from the RAW axis:
        #   left_ticks = int(np.round(axis_left_rad / resolution))
        # (obs rad = ticks * resolution is quantized afterwards)
        "left_ticks": int(np.round(res.axis_left_rad / p.encoder_resolution_rad)),
        "right_ticks": int(np.round(res.axis_right_rad / p.encoder_resolution_rad)),
    })
    return res

DynamicModel.integrate = patched_dm_integrate

actions = [
    [0.70, 0.40], [0.70, 0.40], [0.70, 0.40], [0.70, 0.40],
    [0.50, 0.80], [0.50, 0.80],
    [0.30, 0.30], [0.30, 0.30], [0.30, 0.30],
    [-0.20, 0.60], [-0.20, 0.60],
    [0.10, -0.50], [0.10, -0.50],
    [0.90, 0.90], [0.90, 0.90],
    [0.60, 0.20], [0.00, 0.00], [0.40, 0.10],
]
assert len(actions) == 18
for k, a in enumerate(actions):
    act = np.clip(np.array(a), -1.0, 1.0)
    env.update_physics(act)
    dd = env.state
    i, told, uc = dd.get_commands_at(dd.t - dd.delay)
    row = trace[-1]
    row.update({
        "tick": k + 1,
        "cmd": [enc(float(a[0])), enc(float(a[1]))],
        "t": enc(float(dd.t)),
        "delay_i": int(i),
        "delay_told": enc(float(told)),
        "delay_used": [enc(float(uc.motor_left)), enc(float(uc.motor_right))],
        "world_pos": v3(env.cur_pos),
        "world_angle": enc(float(env.cur_angle)),
        # FJ3.3: the trimmed delay-buffer window (timestamps + commands)
        "buf_ts": [enc(float(x)) for x in dd.timestamps],
        "buf_cmds": [[enc(float(c.motor_left)), enc(float(c.motor_right))]
                     for c in dd.commands],
    })
    out["ticks"].append(row)

with open(out_path, "w") as f:
    json.dump(out, f, indent=1)
print("WROTE", out_path, "ticks:", len(out["ticks"]),
      "model_cases:", len(out["model_cases"]),
      "delay_cases:", len(out["delay_cases"]))
