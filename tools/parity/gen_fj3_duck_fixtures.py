# FJ3.4/FJ3.5 fixture generator: DuckieObj walk dynamics and DuckController
# before_step trigger semantics.
# Run in the REFERENCE venv: conda activate ddm-ref
#   (Python 3.9, numpy 1.20.0, gym 0.23.1, duckietown-gym-daffy 6.1.34)
# Usage: python tools/parity/gen_fj3_duck_fixtures.py [out_path]
import json, math, os, sys
out_path = sys.argv[1] if len(sys.argv) > 1 else (
    "/home/pannntastic/aivnv/DuckietownDecisionModels.jl/test/fixtures/fj3_duck.json")
os.environ.setdefault("PYGLET_HEADLESS", "1")
import numpy as np
from gym_duckietown.envs import DuckietownEnv

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

def unf(v):
    if isinstance(v, dict):
        tag = v["nonfinite"]
        return {"nan": float("nan"), "inf": float("inf"),
                "-inf": float("-inf"), "-0.0": -0.0}[tag]
    return v

def duck_state(duck):
    return {
        "pos": enc(duck.pos), "center": enc(duck.center),
        "start": enc(duck.start), "angle": enc(float(duck.angle)),
        "heading": enc(duck.heading), "vel": enc(float(duck.vel)),
        "active": bool(duck.pedestrian_active),
        "visible": bool(duck.visible),
        "wait": enc(float(duck.pedestrian_wait_time)),
        "time": enc(float(duck.time)),
        "walk_distance": enc(float(duck.walk_distance)),
        "corners": enc(duck.obj_corners), "norm": enc(duck.obj_norm),
    }

env = DuckietownEnv(
    map_name="small_loop", domain_rand=False, max_steps=9000, frame_skip=6,
    user_tile_start=(0, 1), accept_start_angle_deg=10, seed=53,
)
cfg = DuckControllerConfig(
    p_cross=1.0, make_dynamic=True, require_duck=True, inject_if_missing=True,
    spawn_pos=[1.62, 0.50], spawn_rotate=0.0, spawn_height=0.08,
    walk_distance=0.90, trigger_min_ego_distance=0.35,
    trigger_max_ego_distance=0.45, max_crossings_per_episode=1,
    inject_stop_if_missing=True, require_stop=True,
    stop_spawn_pos=[1.20, 2.10], stop_spawn_rotate=180.0, stop_spawn_height=0.18,
)
prepared, n_ducks, n_stops = prepare_task_map_data(env.map_data, cfg)
env.map_data = prepared
env._interpret_map(prepared)
env.seed(53)
env.reset()
duck = env.objects[0]
assert str(duck.kind).lower() == "duckie"
out = {}
out["dt"] = enc(float(env.delta_time))
out["duck_init"] = duck_state(duck)
out["cfg"] = {
    "p_cross": enc(float(cfg.p_cross)),
    "walk_distance": enc(float(cfg.walk_distance)),
    "trigger_min_ego_distance": enc(float(cfg.trigger_min_ego_distance)),
    "trigger_max_ego_distance": enc(float(cfg.trigger_max_ego_distance)),
    "max_crossings_per_episode": int(cfg.max_crossings_per_episode),
    "repeat_rearm_distance": enc(float(cfg.repeat_rearm_distance)),
    "spawn_on_ego_proximity": bool(cfg.spawn_on_ego_proximity),
}

# --- Part A: DuckieObj.step walk dynamics, no controller (auto activation) ---
# A1: duck starts inactive with wait_time = 8 -> auto-activates at t = 8 s,
# walks 0.9 m (45 ticks at 0.02 m/s), finish_walk, waits 8 s, walks back.
out["walk_scenarios"] = []
for scenario, start_active in (("auto_wait", False), ("active", True)):
    duck.pedestrian_active = start_active
    duck.pedestrian_wait_time = 8.0 if not start_active else 0.0
    duck.time = 0.0
    duck.center = np.array([unf(v) for v in out["duck_init"]["center"]], copy=True)
    duck.pos = np.array([unf(v) for v in out["duck_init"]["pos"]], copy=True)
    duck.start = np.array([unf(v) for v in out["duck_init"]["start"]], copy=True)
    duck.angle = unf(out["duck_init"]["angle"])
    duck.heading = np.array([unf(v) for v in out["duck_init"]["heading"]], copy=True)
    duck.vel = unf(out["duck_init"]["vel"])
    duck.obj_corners = np.array([[unf(v) for v in c] for c in out["duck_init"]["corners"]], copy=True)
    duck.obj_norm = np.array([[unf(v) for v in c] for c in out["duck_init"]["norm"]], copy=True)
    rows = []
    for _ in range(600):
        duck.step(float(env.delta_time))
        rows.append(duck_state(duck))
    out["walk_scenarios"].append({"name": scenario, "ticks": rows})

# --- Part B: before_step trigger semantics, real rollout to the crossing ---
# Drive the ego around the loop with a heading P-controller on waypoints;
# the crossing approach drifts LEFT (z ~ 1.23) so the ego passes the duck
# crossing midpoint (1.3977, 0.8775) at ~0.36 m, inside the [0.35, 0.45]
# trigger window, with the crossing ahead. Actions are (v, omega).
duck.pedestrian_active = False
duck.pedestrian_wait_time = float("inf")
duck.time = 0.0
duck.center = np.array([unf(v) for v in out["duck_init"]["center"]], copy=True)
duck.pos = np.array([unf(v) for v in out["duck_init"]["pos"]], copy=True)
duck.start = np.array([unf(v) for v in out["duck_init"]["start"]], copy=True)
duck.angle = unf(out["duck_init"]["angle"])
duck.heading = np.array([unf(v) for v in out["duck_init"]["heading"]], copy=True)
duck.vel = unf(out["duck_init"]["vel"])
duck.obj_corners = np.array([[unf(v) for v in c] for c in out["duck_init"]["corners"]], copy=True)
duck.obj_norm = np.array([[unf(v) for v in c] for c in out["duck_init"]["norm"]], copy=True)

env.seed(53)
env.reset()
from src.duck_controller import DuckController
from src.actions import vw_to_wheels
controller = DuckController(env, cfg, 53)

def wrap_angle(a):
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a

# waypoints along the loop; the approach wpt sits left of the centerline so
# the pass distance to the crossing midpoint lands inside [0.35, 0.45]
wpts = [(0.0, 0.0)]
GAIN = 1.5
LAT_GAIN = 3.0
LAT_REF = 0.05   # drive 5 cm right of the lane centerline
SPEED = 0.25
def command(pos, angle, _wpt):
    pt, tangent = env.closest_curve_point(pos, angle)
    if tangent is None:
        return [SPEED, 0.0]
    theta_d = math.atan2(-tangent[2], tangent[0])
    right = np.array([-math.sin(angle), 0.0, -math.cos(angle)])
    lateral = float(np.dot(np.asarray(pos, dtype=float)
                           - np.asarray(pt, dtype=float), right))
    # d(theta)/dt = +omega (measured): to decrease theta (turn right) omega < 0
    omega = -(GAIN * wrap_angle(angle - theta_d) + LAT_GAIN * (lateral - LAT_REF))
    return [SPEED, float(np.clip(omega, -0.6, 0.6))]

def switch_waypoint(pos, idx):
    return idx

out["rollout"] = []
activated_at = None
wp = 0
for decision in range(200):
    pos_pre = list(env.cur_pos)
    angle_pre = float(env.cur_angle)
    controller.before_step()
    duck_pre = env.objects[0]
    rel_p = (duck_pre.start[0] + 0.5 * float(duck_pre.walk_distance)
             * float(duck_pre.heading[0]) * (np.sign(float(duck_pre.vel)) or 1.0),
             duck_pre.start[2] + 0.5 * float(duck_pre.walk_distance)
             * float(duck_pre.heading[2]) * (np.sign(float(duck_pre.vel)) or 1.0))
    dist_pre = float(np.linalg.norm(np.array(rel_p) - np.array(pos_pre)[[0, 2]]))
    a = command(env.cur_pos, env.cur_angle, wpts[wp])
    wheels = vw_to_wheels(a[0], a[1], 0.102)
    for _ in range(env.frame_skip):
        env.update_physics(wheels)
    wp = switch_waypoint(env.cur_pos, wp)
    duck = env.objects[0]
    row = {
        "decision": decision,
        "action": [enc(float(a[0])), enc(float(a[1]))],
        "wheels": [enc(float(wheels[0])), enc(float(wheels[1]))],
        "pos_pre": enc(pos_pre), "angle_pre": enc(float(angle_pre)),
        "dist_pre": enc(float(dist_pre)),
        "pos": enc(env.cur_pos), "angle": enc(float(env.cur_angle)),
        "step_count": int(env.step_count),
        "crossings_started": list(controller.crossings_started),
        "crossing_armed": list(controller.crossing_armed),
        "duck": duck_state(duck),
    }
    out["rollout"].append(row)
    if activated_at is None and duck.pedestrian_active:
        activated_at = decision
    if activated_at is not None and decision > activated_at + 60:
        break

with open(out_path, "w") as f:
    json.dump(out, f, indent=1)
if activated_at is None:
    best = min((math.hypot(r["pos"][0] - (out["duck_init"]["center"][0]
        + 0.5 * out["cfg"]["walk_distance"] * out["duck_init"]["heading"][0]),
        r["pos"][2] - (out["duck_init"]["center"][2]
        + 0.5 * out["cfg"]["walk_distance"] * out["duck_init"]["heading"][2])),
        r["decision"]) for r in out["rollout"])
    print("never activated; closest approach dist", round(best[0], 3),
          "at decision", best[1])
    raise SystemExit(1)
assert activated_at is not None, "duck was never activated"
print("ACTIVATED at decision", activated_at, "of", len(out["rollout"]))

print("WROTE", out_path, "scenarios:", len(out["walk_scenarios"]),
      "rollout decisions:", len(out["rollout"]))