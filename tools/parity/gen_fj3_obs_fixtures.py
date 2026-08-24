# FJ3.6 fixture generator: state-extraction observers (state.py get_raw_state
# and friends, continuous_state.py duck_relative_state / signed_curvature_ahead
# / build_continuous_state) evaluated on recorded latent states.
#
# Rows are latent-in -> outputs-out: each row records the full latent input
# (ego pose/speed, duck state, controller counters, sigma/fallback memory) and
# every observer output, so the Julia side rebuilds the DuckieWorldState from
# the row and never accumulates dynamics drift (chain-level integration parity
# is FJ3.7).
#
# Run in the REFERENCE venv: conda activate ddm-ref
#   (Python 3.9, numpy 1.20.0, gym 0.23.1, duckietown-gym-daffy 6.1.34)
# Usage: python tools/parity/gen_fj3_obs_fixtures.py [out_path]
import json, math, os, sys
out_path = sys.argv[1] if len(sys.argv) > 1 else (
    "/home/pannntastic/aivnv/DuckietownDecisionModels.jl/test/fixtures/fj3_obs.json")
os.environ.setdefault("PYGLET_HEADLESS", "1")
import numpy as np
from gym_duckietown.envs import DuckietownEnv

sys.path.insert(0, "/home/pannntastic/aivnv/duckduck")
from src.duck_controller import DuckController, DuckControllerConfig, prepare_task_map_data
from src.actions import vw_to_wheels
from src.state import (StateConfig, get_raw_state, next_stop_candidate,
                       classify_duck, tile_ahead, _lane_frame as lane_frame_tab)
from src.continuous_state import (ContinuousStateConfig, build_continuous_state,
                                  duck_relative_state, signed_curvature_ahead,
                                  encode_continuous_state,
                                  _lane_frame as lane_frame_cont)


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
controller = DuckController(env, cfg, 53)

s_cfg = StateConfig()
c_cfg = ContinuousStateConfig()
c_cfg_gated = ContinuousStateConfig(
    duck_detection_range=2.0, duck_detection_corridor_width=0.35,
    duck_detection_forward_only=True)

duck = env.objects[0]
assert str(duck.kind).lower() == "duckie"
sign_indices = [i for i, o in enumerate(env.objects)
                if str(getattr(o.kind, "value", o.kind)).lower() == "sign_stop"]
assert len(sign_indices) == 1
sign = env.objects[sign_indices[0]]

out = {
    "stop_signs": [{"pos": enc(np.asarray(sign.pos, dtype=float)),
                    "angle": enc(float(sign.angle))}],
    "duck_static": {
        "walk_distance": enc(float(duck.walk_distance)),
        "start": enc(np.asarray(duck.start, dtype=float)),
    },
    "cfg": {"max_crossings_per_episode": int(cfg.max_crossings_per_episode)},
    "rows": [],
}


def snapshot_row(sigma_stop, gated, stop_hold_progress):
    """Record inputs (current env/duck/controller latent) and all outputs."""
    ccfg = c_cfg_gated if gated else c_cfg
    fwd_t, right_t = lane_frame_tab(env)
    fwd_c, right_c = lane_frame_cont(env)
    lane_ok = True
    lane_dist = lane_angle = None
    try:
        lp = env.get_lane_pos2(env.cur_pos, env.cur_angle)
        lane_dist, lane_angle = float(lp.dist), float(lp.angle_rad)
    except Exception:
        lane_ok = False
    raw = get_raw_state(env, sigma_stop=sigma_stop, config=s_cfg)
    dist, obj_index = next_stop_candidate(env, s_cfg)
    stop_rel_index = (None if obj_index is None
                      else sign_indices.index(obj_index))
    drel = duck_relative_state(env, raw.v, controller)
    kappa = signed_curvature_ahead(env, s_cfg, ccfg)
    cont = build_continuous_state(env, raw, s_cfg, ccfg, controller,
                                  stop_hold_progress)
    encoded = encode_continuous_state(cont, ccfg)
    return {
        "in": {
            "pos": enc(np.asarray(env.cur_pos, dtype=float)),
            "angle": enc(float(env.cur_angle)),
            "speed": enc(float(env.speed)),
            "fallback": enc([float(v) for v in
                             getattr(env, "_mdp_last_lane_position", (1.0, 1.0))]),
            "sigma_stop": bool(sigma_stop),
            "gated": bool(gated),
            "stop_hold_progress": enc(float(stop_hold_progress)),
            "duck": {
                "center": enc(np.asarray(duck.center, dtype=float)),
                "heading": enc(np.asarray(duck.heading, dtype=float)),
                "vel": enc(float(duck.vel)),
                "active": bool(duck.pedestrian_active),
                "visible": bool(duck.visible),
            },
            "crossings_started": [int(v) for v in controller.crossings_started],
            "crossing_armed": [bool(v) for v in controller.crossing_armed],
        },
        "out": {
            "fwd_tab": enc(fwd_t), "right_tab": enc(right_t),
            "fwd_cont": enc(fwd_c), "right_cont": enc(right_c),
            "lane_ok": lane_ok,
            "lane_dist": enc(lane_dist), "lane_angle": enc(lane_angle),
            "raw": {
                "d": enc(raw.d), "phi": enc(raw.phi), "v": enc(raw.v),
                "tile": int(raw.tile),
                "d_stop": enc(raw.d_stop),
                "sigma_stop": bool(raw.sigma_stop),
                "duck": int(raw.duck),
            },
            "stop_index": stop_rel_index,
            "duck_rel": {
                "present": bool(drel.present),
                "longitudinal": enc(drel.longitudinal),
                "lateral": enc(drel.lateral),
                "v_long": enc(drel.v_longitudinal_relative),
                "v_lat": enc(drel.v_lateral_relative),
                "active": bool(drel.active),
                "crossing_available": bool(drel.crossing_available),
            },
            "kappa": enc(kappa),
            "cont": {
                "kappa": enc(cont.kappa),
                "stop_present": bool(cont.stop_present),
                "d_stop": enc(cont.d_stop),
                "duck_present": bool(cont.duck_present),
                "duck_longitudinal": enc(cont.duck_longitudinal),
                "duck_lateral": enc(cont.duck_lateral),
                "duck_v_long": enc(cont.duck_v_longitudinal_relative),
                "duck_v_lat": enc(cont.duck_v_lateral_relative),
                "duck_active": bool(cont.duck_active),
                "duck_crossing_available": bool(cont.duck_crossing_available),
                "stop_hold_progress": enc(cont.stop_hold_progress),
            },
            "encoded": [enc(float(v)) for v in encoded],
        },
    }


# --- Part A: driven rollout over the loop (3 laps, crossing + sign passes) ---
def wrap_angle(a):
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


GAIN = 1.5
LAT_GAIN = 3.0
LAT_REF = 0.05
SPEED = 0.25


def command():
    pt, tangent = env.closest_curve_point(env.cur_pos, env.cur_angle)
    if tangent is None:
        return [SPEED, 0.0]
    theta_d = math.atan2(-tangent[2], tangent[0])
    right = np.array([-math.sin(env.cur_angle), 0.0, -math.cos(env.cur_angle)])
    lateral = float(np.dot(np.asarray(env.cur_pos, dtype=float)
                           - np.asarray(pt, dtype=float), right))
    omega = -(GAIN * wrap_angle(env.cur_angle - theta_d)
              + LAT_GAIN * (lateral - LAT_REF))
    return [SPEED, float(np.clip(omega, -0.6, 0.6))]


env._mdp_last_lane_position = (1.0, 1.0)
curve_pose = None
for decision in range(300):
    controller.before_step()
    a = command()
    wheels = vw_to_wheels(a[0], a[1], 0.102)
    for _ in range(env.frame_skip):
        env.update_physics(wheels)
    sigma = decision % 7 == 3
    gated = decision % 5 == 2
    hold = min(1.0, decision * 0.01)
    row = snapshot_row(sigma, gated, hold)
    if curve_pose is None and row["out"]["raw"]["tile"] != 0:
        curve_pose = (np.asarray(env.cur_pos, dtype=float).copy(),
                      float(env.cur_angle))
    out["rows"].append(row)
n_rollout = len(out["rows"])

# --- Part B: synthetic sweeps on the non-drivable centre tile --------------
# closest_curve_point returns (None, None) there, so forward == heading_vec
# (deterministic), get_lane_pos2 raises NotInLane (fallback path), and
# tile_ahead falls back / raises (STRAIGHT path).
sign_pos = np.asarray(sign.pos, dtype=float)
sign_facing = np.array([math.cos(float(sign.angle)), 0.0,
                        -math.sin(float(sign.angle))])
centre = np.array([1.5 * env.road_tile_size, 0.0, 1.5 * env.road_tile_size])

synthetic = []
# stop-candidate filters: distance sweep along -facing (ego faces the sign)
forward = -sign_facing
angle_facing = math.atan2(-forward[2], forward[0])
for dist in (0.30, 0.55, 1.20, 2.95, 3.15, 3.60):
    synthetic.append((sign_pos + dist * sign_facing, angle_facing))
# behind the sign (ahead <= 0)
synthetic.append((sign_pos - 0.4 * sign_facing, angle_facing))
# lateral offsets at 1.0 m (0.35 passes, 0.55 fails the 0.40 limit)
right = np.array([-forward[2], 0.0, forward[0]])
for lat in (0.35, 0.55):
    synthetic.append((sign_pos + 1.0 * sign_facing + lat * right, angle_facing))
# orientation filter: ego perpendicular / same-facing as the sign
synthetic.append((sign_pos + 1.0 * sign_facing, angle_facing + math.pi / 2))
synthetic.append((sign_pos + 1.0 * sign_facing, angle_facing + math.pi))
# off-map pose (tiles are None everywhere around it)
synthetic.append((np.array([-0.5, 0.0, -0.5]), 1.0))
# reversed direction on a curve tile: the ego-relative class flips to
# CURVE_LEFT (the clockwise rollout only ever sees CURVE_RIGHT)
assert curve_pose is not None, "rollout never crossed a curve tile"
synthetic.append((curve_pose[0], curve_pose[1] + math.pi))
# corner-tile centre in all four headings: the direction opposite the
# rollout's lane direction classifies the directed curve as CURVE_LEFT
corner = np.array([0.5 * env.road_tile_size, 0.0, 0.5 * env.road_tile_size])
for k in range(4):
    synthetic.append((corner, k * math.pi / 2))

duck_base_center = np.asarray(duck.center, dtype=float).copy()
duck_base_active = bool(duck.pedestrian_active)

rows_b = []
for pos, ang in synthetic:
    env.cur_pos = np.asarray(pos, dtype=float)
    env.cur_angle = float(ang)
    env.speed = 0.3
    env._mdp_last_lane_position = (-0.07, 0.12)
    rows_b.append(snapshot_row(False, False, 0.0))

# duck-threat classes at a fixed pose on the centre tile, heading +x
env.cur_pos = centre
env.cur_angle = 0.0
env.speed = 0.25
fwd = np.array([1.0, 0.0, 0.0])
rgt = np.array([0.0, 0.0, 1.0])
for (a_long, a_lat, active, name) in (
    (0.30, 0.10, False, "side_near"),
    (1.20, 0.10, False, "side_far"),
    (0.30, 0.10, True, "crossing_near"),
    (1.20, 0.10, True, "crossing_far"),
    (1.20, 0.50, True, "outside_corridor"),
    (2.40, 0.10, True, "beyond_range"),
    (-0.40, 0.10, True, "behind"),
):
    duck.center = centre + a_long * fwd + a_lat * rgt
    duck.pedestrian_active = active
    rows_b.append(snapshot_row(False, False, 0.0))
    rows_b.append(snapshot_row(False, True, 0.0))  # gated variant

# controller-counter variants (crossing_available)
duck.center = centre + 0.5 * fwd
duck.pedestrian_active = False
controller.crossings_started[0] = 1
rows_b.append(snapshot_row(False, False, 0.0))
controller.crossings_started[0] = 0
controller.crossing_armed[0] = False
rows_b.append(snapshot_row(False, False, 0.0))
controller.crossing_armed[0] = True

# invisible duck -> absent
duck.visible = False
rows_b.append(snapshot_row(False, False, 0.0))
duck.visible = True

duck.center = duck_base_center
duck.pedestrian_active = duck_base_active
out["rows"].extend(rows_b)

with open(out_path, "w") as f:
    json.dump(out, f, indent=1)
n_stop_rows = sum(1 for r in out["rows"] if r["out"]["raw"]["d_stop"] is not None)
n_threats = sorted({r["out"]["raw"]["duck"] for r in out["rows"]})
print("WROTE", out_path, "rows:", len(out["rows"]),
      "(rollout:", n_rollout, "synthetic:", len(out["rows"]) - n_rollout, ")")
print("rows with stop candidate:", n_stop_rows, "| duck threat classes:", n_threats)
