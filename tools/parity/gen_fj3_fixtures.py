# FJ3 fixture generator: map reconstruction + lane geometry + spawn sampling.
# Run in the REFERENCE venv: conda activate ddm-ref
#   (Python 3.9, numpy 1.20.0, gym 0.23.1, duckietown-gym-daffy 6.1.34,
#    duckietown-world-daffy 6.4.3, PyGeometry-z6 2.1.5)
# Usage: python tools/parity/gen_fj3_fixtures.py > ../../test/fixtures/fj3_map.json
import json, math, sys, os
out_path = sys.argv[1] if len(sys.argv) > 1 else (
    "/home/pannntastic/aivnv/DuckietownDecisionModels.jl/test/fixtures/fj3_map.json")
os.environ.setdefault("PYGLET_HEADLESS", "1")
import numpy as np
from gym_duckietown.envs import DuckietownEnv
import gym_duckietown.simulator as sim
from gym_duckietown.simulator import NotInLane
from gym_duckietown.graphics import bezier_closest, bezier_point, bezier_tangent

sys.path.insert(0, "/home/pannntastic/aivnv/duckduck")
from src.duck_controller import DuckControllerConfig, prepare_task_map_data

def enc(x):
    """JSON-safe exact float encoding (FJ2 convention)."""
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
out["tile_size"] = float(env.road_tile_size)
out["grid_shape"] = [int(env.grid_width), int(env.grid_height)]
out["grid"] = []
for j in range(env.grid_height):
    row = []
    for i in range(env.grid_width):
        t = env._get_tile(i, j)
        row.append(None if t is None else {
            "kind": t["kind"], "angle": int(t["angle"]),
            "drivable": bool(t["drivable"]),
            "curves": [m3(c) for c in t["curves"]] if "curves" in t else None,
        })
    out["grid"].append(row)

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

out["objects"] = []
for o in env.objects:
    out["objects"].append({
        "kind": str(o.kind),
        "pos": v3(o.pos), "angle": enc(float(o.angle)),
        "scale": enc(float(o.scale)),
        "static": bool(o.static), "optional": bool(o.optional),
        "visible": bool(o.visible),
        "min_coords": v3(o.mesh.min_coords),
        "max_coords": v3(o.mesh.max_coords),
        "safety_radius": enc(float(o.safety_radius)),
        "corners": m3(o.obj_corners),
        "norm": m3(o.obj_norm),
        "heading": v3(getattr(o, "heading", [0.0, 0.0, 0.0])),
        "walk_distance": enc(getattr(o, "walk_distance", float("nan"))),
        "vel": enc(getattr(o, "vel", float("nan"))),
        "pedestrian_wait_time": enc(getattr(o, "pedestrian_wait_time", float("nan"))),
    })

# lane geometry samples: (x, z, angle) -> closest_curve_point + get_lane_pos2
out["lane_samples"] = []
samples = [
    (0.20, 0.80, 1.6), (0.35, 0.60, 1.6), (0.10, 0.78, 1.61),
    (0.45, 1.00, 1.59), (0.30, 0.90, 1.60), (0.15, 0.50, 1.58),
    (0.20, 1.20, 1.61), (0.40, 0.70, 1.62), (1.05, 0.70, 3.1),
    (1.20, 0.20, 3.2), (0.60, 1.50, 0.8), (1.00, 1.20, 0.9),
    (1.40, 1.60, 4.7), (0.30, 1.60, 4.7), (1.55, 0.40, 4.75),
]
for (x, z, ang) in samples:
    pos = np.array([x, 0.0, z])
    i, j = env.get_grid_coords(pos)
    tile = env._get_tile(i, j)
    p, tng = env.closest_curve_point(pos, ang)
    lane = None
    if p is not None:
        lp = env.get_lane_pos2(pos, ang)
        lane = {"dist": enc(float(lp.dist)),
                "dot_dir": enc(float(lp.dot_dir)),
                "angle_rad": enc(float(lp.angle_rad)),
                "angle_deg": enc(float(lp.angle_deg))}
    out["lane_samples"].append({
        "pos": [x, z], "angle": ang,
        "coords": [int(i), int(j)],
        "tile": None if tile is None else tile["kind"],
        "drivable": bool(tile["drivable"]) if tile else False,
        "closest_point": None if p is None else v3(p),
        "tangent": None if tng is None else v3(tng),
        "lane_pos": lane,
        "valid_pose": bool(env._valid_pose(pos, ang)),
        "valid_pose_13": bool(env._valid_pose(pos, ang, safety_factor=1.3)),
        "inconvenient": bool(env._inconvenient_spawn(pos)),
        "agent_corners": m3(sim.get_agent_corners(pos, ang)),
    })

out["tile_corners"] = []
for (i, j) in [(0, 1), (1, 0), (2, 2), (1, 1), (0, 2)]:
    out["tile_corners"].append({
        "coords": [i, j], "corners": m3(sim.tile_corners(np.array([i, j]), env.road_tile_size))})

out["collidable"] = {
    "corners": m3(env.collidable_corners),
    "norms": m3(env.collidable_norms),
    "centers": m3(env.collidable_centers),
    "safety_radii": [enc(float(r)) for r in env.collidable_safety_radii],
}

# bezier_closest samples on two reference curves (right lane of tile (0,1) straight/E)
def tile_curves(i, j):
    t = env._get_tile(i, j)
    return [np.array(c) for c in t["curves"]]

curve_a = tile_curves(0, 1)[1]   # straight tile (0,1)
curve_b = tile_curves(2, 2)[0]   # curve_left/E tile (2,2)
out["bezier_closest"] = []
for cps, pts in [
    (curve_a, [(0.15, 0.0, 0.85), (0.25, 0.0, 1.10), (0.10, 0.0, 0.70)]),
    (curve_b, [(1.35, 0.0, 1.10), (1.40, 0.0, 1.30), (1.60, 0.0, 1.45)]),
]:
    for p in pts:
        t = bezier_closest(cps, np.array(p))
        out["bezier_closest"].append({
            "cps": m3(cps), "p": [p[0], p[2]], "t": enc(float(t)),
            "point": v3(bezier_point(cps, t)),
        })

# spawn sampling: raw env.reset() for the eval seeds (no curriculum)
out["spawn_samples"] = []
for seed in (53, 73, 2101, 20101):
    env.seed(seed)
    env.reset()
    out["spawn_samples"].append({
        "seed": int(seed),
        "pos": v3(env.cur_pos),
        "angle": enc(float(env.cur_angle)),
    })

# env.reset() repeated draws for one seed (RNG stream order check)
env.seed(53)
out["reset_draws"] = []
for _ in range(5):
    env.reset()
    out["reset_draws"].append({
        "pos": v3(env.cur_pos), "angle": enc(float(env.cur_angle)),
    })

with open(out_path, "w") as f:
    json.dump(out, f, indent=1)
print("WROTE", out_path)