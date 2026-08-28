# FJ3.8 fixture generator: exact NumPy RNG stream identity (RNG-C).
#
# Two reference streams:
#   controller — np.random.RandomState(seed): MT19937 with legacy seeding,
#                draws via random_sample() (two 32-bit outputs -> 53-bit
#                double). Pinned by: the full 624-word state vector after
#                seeding, raw tempered uint32s, random_sample sequences, and
#                a p_cross=0.5 trigger chain whose OUTCOMES depend on the
#                exact draw values.
#   spawn      — gym.utils.seeding.np_random(seed):
#                Generator(PCG64(SeedSequence(seed))). Pinned by:
#                SeedSequence.generate_state words, raw uint64s, random()
#                doubles, uniform(a, b), integers(0, n) (Lemire), and the
#                verbatim call log of one Simulator.reset spawn.
#
# Run in the REFERENCE venv: conda activate ddm-ref (numpy 1.20.0)
# Usage: python tools/parity/gen_fj38_fixtures.py [out_path]
import json, math, os, sys
out_path = sys.argv[1] if len(sys.argv) > 1 else (
    "/home/pannntastic/aivnv/DuckietownDecisionModels.jl/test/fixtures/fj38_rng.json")
os.environ.setdefault("PYGLET_HEADLESS", "1")
import numpy as np


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


out = {"numpy_version": np.__version__}

# --- Part A: RandomState / MT19937 -------------------------------------------
SEEDS = [0, 1, 2, 53, 73, 101, 123456789, 2**31 - 1, 2**32 - 1]
out["mt19937"] = []
for seed in SEEDS:
    rs = np.random.RandomState(seed)
    kind, key, pos, has_gauss, cached = rs.get_state()
    assert kind == "MT19937" and pos == 624
    doubles = [float(rs.random_sample()) for _ in range(64)]
    # raw tempered outputs: tomaxint uses genrand >> 1; use randint on the
    # full uint32 range instead (one tempered word per draw)
    rs2 = np.random.RandomState(seed)
    raws = [int(v) for v in rs2.randint(0, 2**32, size=32, dtype=np.uint32)]
    out["mt19937"].append({
        "seed": int(seed),
        "state": [int(v) for v in key],
        "random_sample": [enc(v) for v in doubles],
        "raw_uint32": raws,
    })

# --- Part B: SeedSequence + PCG64 + Generator ---------------------------------
out["pcg64"] = []
for seed in SEEDS:
    # uint64 values are stored as DECIMAL STRINGS: JSON numbers above 2^53
    # lose bits through the Float64 path in the Julia JSON reader
    ss = np.random.SeedSequence(seed)
    words = [str(int(v)) for v in ss.generate_state(8, np.uint64)]
    bg = np.random.PCG64(np.random.SeedSequence(seed))
    raw64 = [str(int(bg.random_raw())) for _ in range(32)]
    g = np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed)))
    doubles = [float(g.random()) for _ in range(32)]
    g = np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed)))
    uniforms = [float(g.uniform(-2.5, 7.25)) for _ in range(16)]
    ints = {}
    for n in (2, 5, 7, 16, 37, 1000):
        g = np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed)))
        ints[str(n)] = [int(g.integers(0, n)) for _ in range(24)]
    g = np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed)))
    normals = [float(g.standard_normal()) for _ in range(24)]
    out["pcg64"].append({
        "seed": int(seed),
        "seedseq_state8": words,
        "raw_uint64": raw64,
        "random": [enc(v) for v in doubles],
        "uniform_m2p5_7p25": [enc(v) for v in uniforms],
        "integers": ints,
        "standard_normal": [enc(v) for v in normals],
    })

# --- Part C: trigger chain with p_cross = 0.5 (outcomes depend on draws) ------
# Reuse the FJ3.7 head-on duck variant so triggers recur; with p_cross = 0.5
# some eligible decisions do NOT activate — the activation pattern encodes the
# exact MT19937 stream through the full transition chain.
import yaml
sys.path.insert(0, "/home/pannntastic/aivnv/duckduck")
from src.env_wrapper import build_env
from copy import deepcopy

# baseline duckie (island crossing) + unlimited crossings: the lane-follow
# route passes the trigger window every lap, so eligible decisions recur and
# with p_cross = 0.5 the activation pattern encodes the exact draw values
DUCK_OVERRIDES = {"max_crossings_per_episode": 0, "p_cross": 0.5}
with open("/home/pannntastic/aivnv/duckduck/policies/q_learning/training_config.yaml") as f:
    q = yaml.safe_load(f)
q["duck_controller"].update(DUCK_OVERRIDES)

CTRL_SEED = 53
env = build_env(q, CTRL_SEED)
base = env.unwrapped


class LoggingRandomState(np.random.RandomState):
    def __init__(self, seed):
        super().__init__(seed)
        self.log = []

    def random_sample(self, *args, **kwargs):
        v = super().random_sample(*args, **kwargs)
        self.log.append(float(v))
        return v


def kindstr(o):
    return str(getattr(o.kind, "value", o.kind)).lower()


def wrap_angle(a):
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def desired_omega(b):
    pt, tg = b.closest_curve_point(b.cur_pos, b.cur_angle)
    if tg is None:
        return 0.0
    th = math.atan2(-tg[2], tg[0])
    right = np.array([-math.sin(b.cur_angle), 0.0, -math.cos(b.cur_angle)])
    lat = float(np.dot(np.asarray(b.cur_pos, dtype=float)
                       - np.asarray(pt, dtype=float), right))
    return -(1.5 * wrap_angle(b.cur_angle - th) + 3.0 * (lat - 0.05))


def lane_policy(k):
    om = desired_omega(base)
    if om > 0.30:
        return 3
    if om < -0.30:
        return 5
    return 1 if abs(om) < 0.10 else 4


env.reset(CTRL_SEED)
rng_log = LoggingRandomState(CTRL_SEED)
env.duck_controller.rng = rng_log
duck = next(o for o in base.objects if kindstr(o) == "duckie")

sign_indices = [i for i in range(len(base.objects))
                if kindstr(base.objects[i]) == "sign_stop"]
lid = env._last_stop_id
init = {
    "pos": enc(np.asarray(base.cur_pos, dtype=float)),
    "angle": enc(float(base.cur_angle)),
    "fallback": enc([float(v) for v in base._mdp_last_lane_position]),
    "last_d_stop": enc(env._last_state.d_stop),
    "last_stop_id": (None if lid is None else sign_indices.index(lid)),
    "ducks": [{
        "pos": enc(np.asarray(d.pos, dtype=float)),
        "center": enc(np.asarray(d.center, dtype=float)),
        "start": enc(np.asarray(d.start, dtype=float)),
        "angle": enc(float(d.angle)),
        "heading": enc(np.asarray(d.heading, dtype=float)),
        "vel": enc(float(d.vel)), "active": bool(d.pedestrian_active),
        "wait": enc(float(d.pedestrian_wait_time)), "time": enc(float(d.time)),
        "walk_distance": enc(float(d.walk_distance)),
        "visible": bool(d.visible),
        "corners": enc(np.asarray(d.obj_corners, dtype=float)),
        "norm": enc(np.asarray(d.obj_norm, dtype=float)),
        "scale": enc(float(getattr(d, "scale", 1.0))),
        "safety_radius": enc(float(getattr(d, "safety_radius", 0.1))),
        "min_coords": enc(np.asarray(getattr(d, "min_coords", np.zeros(3)), dtype=float)),
        "max_coords": enc(np.asarray(getattr(d, "max_coords", np.zeros(3)), dtype=float)),
    } for d in base.objects if kindstr(d) == "duckie"],
    "signs": [{"pos": enc(np.asarray(base.objects[i].pos, dtype=float)),
               "angle": enc(float(base.objects[i].angle))} for i in sign_indices],
    "crossings_started": [int(v) for v in env.duck_controller.crossings_started],
    "crossing_armed": [bool(v) for v in env.duck_controller.crossing_armed],
}
decisions = []
for k in range(200):
    n_before = len(rng_log.log)
    a = lane_policy(k)
    state, r, done, info = env.step(a)
    decisions.append({
        "action": int(a),
        "draws": [enc(v) for v in rng_log.log[n_before:]],
        "duck_active": bool(duck.pedestrian_active),
        "crossings_started": [int(v) for v in env.duck_controller.crossings_started],
        "pos": enc(np.asarray(base.cur_pos, dtype=float)),
        "reason": info["termination_reason"],
    })
    if done:
        break
out["trigger_chain"] = {
    "seed": CTRL_SEED, "p_cross": 0.5,
    "duck_overrides": DUCK_OVERRIDES,
    "init": init, "decisions": decisions,
    "total_draws": len(rng_log.log),
}
assert len(rng_log.log) >= 3, "trigger chain drew too few samples"
assert any(len(d["draws"]) > 0 and d["draws"][0] >= 0.5
           for d in decisions), "no rejected draw in the chain"
assert any(len(d["draws"]) > 0 and d["draws"][0] < 0.5
           for d in decisions), "no accepted draw in the chain"

# --- Part D: verbatim spawn call log of one Simulator.reset -------------------
from gym.utils.seeding import RandomNumberGenerator

call_log = []


def _enc_result(v):
    if isinstance(v, np.ndarray):
        return [enc(float(x)) for x in np.ravel(v)]
    return enc(v)


def _bg_state(self):
    st = self.bit_generator.state
    return {"state": str(st["state"]["state"]),
            "inc": str(st["state"]["inc"]),
            "has_uint32": int(st["has_uint32"]),
            "uinteger": int(st["uinteger"])}


def _log_method(name):
    def wrapper(self, *args, **kwargs):
        pre = _bg_state(self)
        v = getattr(RandomNumberGenerator, name)(self, *args, **kwargs)
        call_log.append((name, [_enc_result(a) for a in args],
                         {k: _enc_result(w) for k, w in kwargs.items()},
                         _enc_result(v), pre))
        return v
    return wrapper


class LoggingGenerator(RandomNumberGenerator):
    integers = _log_method("integers")
    uniform = _log_method("uniform")
    random = _log_method("random")
    normal = _log_method("normal")
    standard_normal = _log_method("standard_normal")
    choice = _log_method("choice")


SPAWN_SEED = 53
env.seed(SPAWN_SEED)
env.unwrapped.np_random = LoggingGenerator(np.random.PCG64(np.random.SeedSequence(SPAWN_SEED)))
base.reset()
out["spawn_log"] = {
    "seed": SPAWN_SEED,
    "n_drivable": len(base.drivable_tiles),
    "calls": [{"fn": c[0], "args": c[1], "kwargs": c[2], "result": c[3],
               "pre": c[4]} for c in call_log],
    "final_pos": enc(np.asarray(base.cur_pos, dtype=float)),
    "final_angle": enc(float(base.cur_angle)),
}

with open(out_path, "w") as f:
    json.dump(out, f, indent=1)
acts = sum(1 for d in out["trigger_chain"]["decisions"] if d["crossings_started"][0] > 0)
n_draws = out["trigger_chain"]["total_draws"]
print("WROTE", out_path)
print("trigger chain:", len(decisions), "decisions,", n_draws, "draws, last reason:",
      decisions[-1]["reason"])
print("spawn log calls:", len(call_log))
