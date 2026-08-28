# FJ3.7 fixture generator: full one-decision transition chain, pinned against
# the REAL wrapper step functions (`DuckieMDPEnv.step`,
# `ContinuousDuckieMDPEnv.step`) built by the reference factories
# (`build_env` / `build_continuous_env`) from the actual experiment YAMLs —
# NOT against a hand-reconstructed chain, so a chain-order bug cannot be
# copied into both sides.
#
# Scenarios (each: fresh reset(seed) -> recorded action sequence -> per
# decision, the full latent + wrapper outputs):
#   discrete (q_learning YAML, its seed):
#     macro_<name> x7  - constant macro action until done (BRAKE capped high
#                        enough to reach the 1500-tick timeout truncation)
#     mixed            - scripted heuristic lane-following macro policy
#   continuous (sac YAML, its seed):
#     cont_point_<k>   - constant [v, omega] at the action-space key points
#                        (0, extremes, interior, one out-of-range clip case)
#     cont_follow      - scripted continuous P-controller rollout
#
# Run in the REFERENCE venv: conda activate ddm-ref
# Usage: python tools/parity/gen_fj37_fixtures.py [out_path]
import json, math, os, sys
out_path = sys.argv[1] if len(sys.argv) > 1 else (
    "/home/pannntastic/aivnv/DuckietownDecisionModels.jl/test/fixtures/fj37_transition.json")
os.environ.setdefault("PYGLET_HEADLESS", "1")
import numpy as np
import yaml

sys.path.insert(0, "/home/pannntastic/aivnv/duckduck")
from src.env_wrapper import build_env
from src.continuous_env import build_continuous_env

DUCKDUCK = "/home/pannntastic/aivnv/duckduck"


def enc(x):
    if isinstance(x, np.ndarray):
        return [enc(v) for v in x]
    if isinstance(x, (list, tuple)):
        return [enc(v) for v in x]
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
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


def kindstr(o):
    return str(getattr(o.kind, "value", o.kind)).lower()


def duck_state(duck):
    return {
        "pos": enc(np.asarray(duck.pos, dtype=float)),
        "center": enc(np.asarray(duck.center, dtype=float)),
        "start": enc(np.asarray(duck.start, dtype=float)),
        "angle": enc(float(duck.angle)),
        "heading": enc(np.asarray(duck.heading, dtype=float)),
        "vel": enc(float(duck.vel)),
        "active": bool(duck.pedestrian_active),
        "wait": enc(float(duck.pedestrian_wait_time)),
        "time": enc(float(duck.time)),
        "walk_distance": enc(float(duck.walk_distance)),
        "visible": bool(duck.visible),
        "corners": enc(np.asarray(duck.obj_corners, dtype=float)),
        "norm": enc(np.asarray(duck.obj_norm, dtype=float)),
        "scale": enc(float(getattr(duck, "scale", 1.0))),
        "safety_radius": enc(float(getattr(duck, "safety_radius", 0.1))),
        "min_coords": enc(np.asarray(getattr(duck, "min_coords",
            np.zeros(3)), dtype=float)),
        "max_coords": enc(np.asarray(getattr(duck, "max_coords",
            np.zeros(3)), dtype=float)),
    }


def scenario_init(mdp, sign_indices):
    base = mdp.unwrapped
    ctrl = mdp.duck_controller
    last = mdp._last_state
    lid = mdp._last_stop_id
    return {
        "pos": enc(np.asarray(base.cur_pos, dtype=float)),
        "angle": enc(float(base.cur_angle)),
        "speed": enc(float(base.speed)),
        "step_count": int(base.step_count),
        "fallback": enc([float(v) for v in base._mdp_last_lane_position]),
        "last_d_stop": enc(last.d_stop),
        "last_stop_id": (None if lid is None else sign_indices.index(lid)),
        "ducks": [duck_state(base.objects[i]) for i, o in
                  enumerate(base.objects) if kindstr(base.objects[i]) == "duckie"],
        "signs": [{"pos": enc(np.asarray(base.objects[i].pos, dtype=float)),
                   "angle": enc(float(base.objects[i].angle))}
                  for i in sign_indices],
        "crossings_started": [int(v) for v in ctrl.crossings_started],
        "crossing_armed": [bool(v) for v in ctrl.crossing_armed],
    }


def decision_out(mdp, info, sign_indices):
    base = mdp.unwrapped
    ctrl = mdp.duck_controller
    lid = mdp._last_stop_id
    raw = info["raw_state"]
    duck = next(base.objects[i] for i in range(len(base.objects))
                if kindstr(base.objects[i]) == "duckie")
    return {
        "wheels": enc(np.asarray(info["wheel_commands"], dtype=float)),
        "pos": enc(np.asarray(base.cur_pos, dtype=float)),
        "angle": enc(float(base.cur_angle)),
        "speed": enc(float(base.speed)),
        "step_count": int(base.step_count),
        "raw": {
            "d": enc(raw["d"]), "phi": enc(raw["phi"]), "v": enc(raw["v"]),
            "tile": int(raw["tile"]), "d_stop": enc(raw["d_stop"]),
            "sigma_stop": bool(raw["sigma_stop"]), "duck": int(raw["duck"]),
        },
        "stop_id": (None if lid is None else sign_indices.index(lid)),
        "sigma": bool(mdp.stop_tracker.sigma_stop),
        "hold_steps": int(mdp.stop_tracker.hold_steps),
        "events": {k: bool(v) for k, v in info["events"].items()},
        "reward": {k: enc(float(v)) for k, v in info["reward_terms"].items()},
        "reason": info["termination_reason"],
        "terminated": bool(info["terminated"]),
        "truncated": bool(info["truncated"]),
        "duck": {"center": enc(np.asarray(duck.center, dtype=float)),
                 "active": bool(duck.pedestrian_active),
                 "vel": enc(float(duck.vel))},
        "crossings_started": [int(v) for v in ctrl.crossings_started],
        "crossing_armed": [bool(v) for v in ctrl.crossing_armed],
    }


def wrap_angle(a):
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def desired_omega(base):
    pt, tangent = base.closest_curve_point(base.cur_pos, base.cur_angle)
    if tangent is None:
        return 0.0
    theta_d = math.atan2(-tangent[2], tangent[0])
    right = np.array([-math.sin(base.cur_angle), 0.0, -math.cos(base.cur_angle)])
    lateral = float(np.dot(np.asarray(base.cur_pos, dtype=float)
                           - np.asarray(pt, dtype=float), right))
    return -(1.5 * wrap_angle(base.cur_angle - theta_d) + 3.0 * (lateral - 0.05))


def omega_towards(base, target_xz):
    rel = np.asarray(target_xz, dtype=float) - np.asarray(
        base.cur_pos, dtype=float)[[0, 2]]
    theta_d = math.atan2(-rel[1], rel[0])
    return -1.8 * wrap_angle(base.cur_angle - theta_d)


MACRO_NAMES = ["fast_left", "fast_straight", "fast_right",
               "slow_left", "slow_straight", "slow_right", "brake"]

out = {"scenarios": []}

# --- discrete scenarios (q_learning config) ---------------------------------
q_yaml = os.path.join(DUCKDUCK, "policies", "q_learning", "training_config.yaml")
with open(q_yaml) as f:
    q_config = yaml.safe_load(f)
q_seed = int(q_config["seed"])
env = build_env(q_config, q_seed)
base = env.unwrapped
sign_indices = [i for i in range(len(base.objects))
                if kindstr(base.objects[i]) == "sign_stop"]


def run_discrete(name, policy, cap, denv=None, dsigns=None, meta=None):
    denv = env if denv is None else denv
    dsigns = sign_indices if dsigns is None else dsigns
    denv.reset(q_seed)
    init = scenario_init(denv, dsigns)
    decisions = []
    for k in range(cap):
        a = policy(k)
        state, r, done, info = denv.step(a)
        decisions.append({"action": int(a),
                          "out": decision_out(denv, info, dsigns)})
        if done:
            break
    scenario = {
        "name": name, "kind": "discrete", "config": "q_learning",
        "seed": q_seed, "init": init, "decisions": decisions,
    }
    if meta:
        scenario["meta"] = meta
    out["scenarios"].append(scenario)
    return decisions


reasons = set()
for aid, aname in enumerate(MACRO_NAMES):
    cap = 260 if aname == "brake" else 30
    ds = run_discrete(f"macro_{aname}", lambda k, a=aid: a, cap)
    reasons.add(ds[-1]["out"]["reason"])

def mixed_policy_for(b, _k):
    om = desired_omega(b)
    if om > 0.30:
        return 3   # slow_left
    if om < -0.30:
        return 5   # slow_right
    return 1 if abs(om) < 0.10 else 4   # fast_straight / slow_straight

def mixed_policy(k):
    return mixed_policy_for(base, k)

mixed_ds = run_discrete("mixed", mixed_policy, 170)
reasons.add(mixed_ds[-1]["out"]["reason"])

# --- variant scenarios: goal, reachable stop sign, sign crash, duck crash ----
# These exercise chain paths the baseline scenario cannot reach (FJ3.6
# scenario observation: the baseline sign never passes the lateral filter).
# Each variant is an explicit config-space point recorded in the scenario
# meta — the baseline configs are never modified.
from copy import deepcopy

TS = float(base.road_tile_size)

# goal: the tile the mixed rollout occupies at decision ~25
goal_pos = [unv for unv in mixed_ds[min(25, len(mixed_ds) - 1)]["out"]["pos"]]
goal_tile = [int(math.floor(float(goal_pos[0]) / TS)),
             int(math.floor(float(goal_pos[2]) / TS))]
q_goal = deepcopy(q_config)
q_goal["environment"]["goal_tile"] = goal_tile
env_goal = build_env(q_goal, q_seed)
base_goal = env_goal.unwrapped
gsigns = [i for i in range(len(base_goal.objects))
          if kindstr(base_goal.objects[i]) == "sign_stop"]
ds = run_discrete("goal", lambda k: mixed_policy_for(base_goal, k), 60,
                  denv=env_goal, dsigns=gsigns,
                  meta={"goal_tile": goal_tile})
reasons.add(ds[-1]["out"]["reason"])

# reachable stop sign: placed ON the recorded route ring (a straight-segment
# pose from the mixed rollout), laterally offset 0.28 m (inside the 0.40
# lateral filter, outside the collision path) and facing against the route
# direction. The route ring is the same lane cycle regardless of the spawn
# the moved sign induces, and a 170-decision lane-follow covers a full lap,
# so the sign is always reached.
def sign_from_route(ds_list, lateral_offset):
    for j in range(20, len(ds_list)):
        a1 = float(ds_list[j - 6]["out"]["angle"])
        a2 = float(ds_list[j]["out"]["angle"])
        if abs(wrap_angle(a1 - a2)) < 0.05:
            p = [float(v) for v in ds_list[j]["out"]["pos"]]
            wx = p[0] + lateral_offset * math.sin(a2)
            wz = p[2] + lateral_offset * math.cos(a2)
            rotate = -math.degrees(wrap_angle(a2 + math.pi))
            return (wx, wz), rotate
    raise RuntimeError("no straight route segment found")

stop_world, stop_rot = sign_from_route(mixed_ds, 0.28)
stop_tiles = [stop_world[0] / TS, stop_world[1] / TS - 1.0]
q_stop = deepcopy(q_config)
q_stop["duck_controller"]["stop_spawn_pos"] = stop_tiles
q_stop["duck_controller"]["stop_spawn_rotate"] = stop_rot
env_stop = build_env(q_stop, q_seed)
base_stop = env_stop.unwrapped
ssigns = [i for i in range(len(base_stop.objects))
          if kindstr(base_stop.objects[i]) == "sign_stop"]
sign_obj = base_stop.objects[ssigns[0]]
stop_meta = {"sign_desc": {"pos": enc(np.asarray(sign_obj.pos, dtype=float)),
                           "rotate_deg": float(stop_rot), "height": float(
                               q_stop["duck_controller"]["stop_spawn_height"])}}

def stop_comply_policy(k):
    raw = env_stop._last_state
    if (raw.d_stop is not None and raw.d_stop <= 0.45
            and not env_stop.stop_tracker.sigma_stop):
        return 6   # brake until sigma latches
    return mixed_policy_for(base_stop, k)

ds = run_discrete("stop_comply", stop_comply_policy, 170,
                  denv=env_stop, dsigns=ssigns, meta=stop_meta)
assert any(d["out"]["events"]["full_stop"] for d in ds), "no full_stop"
assert any(d["out"]["events"]["passed_stop"] for d in ds), "no passed_stop"

ds = run_discrete("stop_violate",
                  lambda k: mixed_policy_for(base_stop, k), 170,
                  denv=env_stop, dsigns=ssigns, meta=stop_meta)
assert any(d["out"]["events"]["stop_violation"] for d in ds), "no violation"

# sign crash: the sign ON the lane center -> other_collision
crash_world, crash_rot = sign_from_route(mixed_ds, 0.0)
q_crash = deepcopy(q_config)
q_crash["duck_controller"]["stop_spawn_pos"] = [crash_world[0] / TS,
                                                crash_world[1] / TS - 1.0]
q_crash["duck_controller"]["stop_spawn_rotate"] = crash_rot
env_crash = build_env(q_crash, q_seed)
base_crash = env_crash.unwrapped
csigns2 = [i for i in range(len(base_crash.objects))
           if kindstr(base_crash.objects[i]) == "sign_stop"]
crash_obj = base_crash.objects[csigns2[0]]
crash_meta = {"sign_desc": {"pos": enc(np.asarray(crash_obj.pos, dtype=float)),
                            "rotate_deg": float(crash_rot), "height": float(
                                q_crash["duck_controller"]["stop_spawn_height"])}}
ds = run_discrete("sign_crash",
                  lambda k: mixed_policy_for(base_crash, k), 170,
                  denv=env_crash, dsigns=csigns2, meta=crash_meta)
reasons.add(ds[-1]["out"]["reason"])

# duck crash: the duckie idles on the non-drivable centre island, so a
# collision is only reachable while it is CROSSING the road. Try policy /
# trigger-window variants until one run ends in duck_collision; the winning
# variant's overrides are recorded in the scenario meta.
def find_duck(b):
    return next(b.objects[i] for i in range(len(b.objects))
                if kindstr(b.objects[i]) == "duckie")


def crossing_midpoint(duck):
    s = math.copysign(1.0, float(duck.vel)) if float(duck.vel) != 0.0 else 1.0
    h = np.asarray(duck.heading, dtype=float)
    st = np.asarray(duck.start, dtype=float)
    mid = st + 0.5 * float(duck.walk_distance) * h * s
    return mid[[0, 2]]


# Timing-based intercepts are hopeless (the duckie crosses the ego lane in
# ~3 decisions while the delayed ego needs ~5), so make the collision
# geometric instead: spawn the DUCKIE on the lane itself, heading back along
# the lane. Lane-following then either meets it head-on when the trigger
# fires (closing 0.2 m/decision) or simply drives into the resting duckie.
duck_world, duck_rot = sign_from_route(mixed_ds, 0.0)
duck_attempts = [
    ("headon", {"spawn_pos": [duck_world[0] / TS, duck_world[1] / TS - 1.0],
                "spawn_rotate": duck_rot,
                "max_crossings_per_episode": 0}),
    ("headon_single", {"spawn_pos": [duck_world[0] / TS,
                                     duck_world[1] / TS - 1.0],
                       "spawn_rotate": duck_rot}),
]
crashed = False
for label, overrides in duck_attempts:
    q_duck = deepcopy(q_config)
    q_duck["duck_controller"].update(overrides)
    env_d = build_env(q_duck, q_seed)
    base_d = env_d.unwrapped
    dsigns_d = [i for i in range(len(base_d.objects))
                if kindstr(base_d.objects[i]) == "sign_stop"]
    ds = run_discrete("duck_crash",
                      lambda k: mixed_policy_for(base_d, k), 250,
                      denv=env_d, dsigns=dsigns_d,
                      meta={"duck_overrides": overrides})
    if ds[-1]["out"]["reason"] == "duck_collision":
        crashed = True
        print("duck_crash succeeded with variant:", label)
        break
    out["scenarios"].pop()   # discard the failed attempt
    print("duck_crash attempt", label, "ended:", ds[-1]["out"]["reason"])
assert crashed, "no duck_crash variant produced duck_collision"
reasons.add("duck_collision")

# --- continuous scenarios (sac config) ---------------------------------------
sac_yaml = os.path.join(DUCKDUCK, "policies", "sac", "training_config.yaml")
with open(sac_yaml) as f:
    sac_config = yaml.safe_load(f)
sac_seed = int(sac_config["seed"])
cenv = build_continuous_env(sac_config, sac_seed)
cmdp = cenv.mdp_env
cbase = cenv.unwrapped
csign_indices = [i for i in range(len(cbase.objects))
                 if kindstr(cbase.objects[i]) == "sign_stop"]


def cdecision_out(info):
    row = decision_out(cmdp, info, csign_indices)
    cont = info["continuous_state"]
    row["cont"] = {k: enc(cont[k]) for k in cont}
    row["command_v_omega"] = enc([float(v) for v in info["command_v_omega"]])
    return row


def run_continuous(name, policy, cap):
    cenv.reset(sac_seed)
    init = scenario_init(cmdp, csign_indices)
    decisions = []
    for k in range(cap):
        a = policy(k)
        obs, r, done, info = cenv.step(np.asarray(a, dtype=np.float32))
        decisions.append({"action": enc([float(a[0]), float(a[1])]),
                          "obs": enc([float(v) for v in obs]),
                          "out": cdecision_out(info)})
        if done:
            break
    out["scenarios"].append({
        "name": name, "kind": "continuous", "config": "sac",
        "seed": sac_seed, "init": init, "decisions": decisions,
    })
    return decisions


CONT_POINTS = [
    (0.00, 0.00),
    (0.17, 0.00), (0.41, 0.00),
    (0.17, 1.50), (0.17, -1.50),
    (0.41, 1.50), (0.41, -1.50),
    (0.30, 0.70), (0.10, -0.40),
    (0.60, 2.50),   # out of range: clips to (0.41, 1.50)
]
for k, pt in enumerate(CONT_POINTS):
    ds = run_continuous(f"cont_point_{k}", lambda _i, p=pt: p, 25)
    reasons.add(ds[-1]["out"]["reason"])

def cont_follow_policy(_k):
    om = desired_omega(cbase)
    return (0.25, float(np.clip(om, -1.5, 1.5)))

ds = run_continuous("cont_follow", cont_follow_policy, 170)
reasons.add(ds[-1]["out"]["reason"])

with open(out_path, "w") as f:
    json.dump(out, f, indent=1)
n_dec = sum(len(s["decisions"]) for s in out["scenarios"])
print("WROTE", out_path, "scenarios:", len(out["scenarios"]),
      "decisions:", n_dec)
print("terminal/last reasons seen:", sorted(reasons))
all_reasons = sorted({d["out"]["reason"] for s in out["scenarios"]
                      for d in s["decisions"]})
print("all reasons across decisions:", all_reasons)
events_seen = sorted({k for s in out["scenarios"] for d in s["decisions"]
                      for k, v in d["out"]["events"].items() if v})
print("events raised somewhere:", events_seen)
