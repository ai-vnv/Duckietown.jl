# FJ5.1: the Python reference backend, driven as a JSON-lines subprocess.
#
# Why a subprocess and not PythonCall.jl: this workstation runs a NATIVE
# WINDOWS Julia while the pinned reference stack (`ddm-ref`: Python 3.9,
# numpy 1.20.0, gym 0.23.1, duckietown-gym-daffy 6.1.34) is an ELF/Linux
# conda env inside WSL. PythonCall loads libpython in-process, which cannot
# cross that boundary, and no Linux Julia exists here. A line protocol gives
# the same capability (live side-by-side execution with matched states),
# keeps the Julia package free of ANY Python dependency, and leaves the
# door open for a PythonCall implementation of the same interface later.
#
# The reference implementation itself is NEVER modified: this server only
# constructs the real `DuckieMDPEnv` / `ContinuousDuckieMDPEnv` through the
# real factories, reads their attributes, and (for matched-state parity)
# writes the documented mutable latent attributes back.
#
# Protocol: one JSON object per line on stdin, one JSON object per line on
# stdout ({"ok": true, ...} or {"ok": false, "error": "..."}).
#
# Run:  conda activate ddm-ref && python tools/parity/reference_server.py
import json
import math
import os
import sys
import traceback

os.environ.setdefault("PYGLET_HEADLESS", "1")

import logging
logging.disable(logging.CRITICAL)

# The protocol channel must stay clean, but the reference stack prints to
# stdout on import (pyglet dumps its options dict) and logs freely. When run
# as a SERVER we therefore duplicate the real stdout onto a private fd used
# only for responses and point fd 1 at stderr, so library chatter can never
# desynchronise the channel.
#
# This must NOT happen at import time: FJ5-R imports this module in-process
# through PythonCall, where hijacking fd 1 would redirect Julia's own stdout.
PROTOCOL = None


def claim_protocol_channel():
    """Take exclusive ownership of stdout for the line protocol (server mode)."""
    global PROTOCOL
    if PROTOCOL is None:
        fd = os.dup(1)
        os.dup2(2, 1)
        PROTOCOL = os.fdopen(fd, "w")
    return PROTOCOL


def respond(obj):
    PROTOCOL.write(json.dumps(obj) + "\n")
    PROTOCOL.flush()

import numpy as np
import yaml

sys.path.insert(0, "/home/pannntastic/aivnv/duckduck")

from src.env_wrapper import build_env, _any_collision, _duck_collision
from src.continuous_env import build_continuous_env
from src.state import RawState, TileType, DuckThreat, next_stop_candidate
from src.continuous_state import continuous_state_to_dict
from duckietown_world.world_duckietown.dynamics_delay import DelayedDynamics
from duckietown_world.world_duckietown.pwm_dynamics import DynamicModel

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
    raise TypeError(str(type(x)))


def dec(v):
    if isinstance(v, dict) and "nonfinite" in v:
        return {"nan": float("nan"), "inf": float("inf"),
                "-inf": float("-inf"), "-0.0": -0.0}[v["nonfinite"]]
    if isinstance(v, list):
        return [dec(x) for x in v]
    return v


def kindstr(o):
    return str(getattr(o.kind, "value", o.kind)).lower()


class Session:
    def __init__(self):
        self.env = None        # the wrapper that owns step(): discrete or continuous
        self.mdp = None        # the DuckieMDPEnv
        self.base = None       # the raw Simulator
        self.continuous = False
        self.sign_indices = []

    # --- construction -----------------------------------------------------
    def init(self, config, seed, action_space="discrete", overrides=None):
        path = os.path.join(DUCKDUCK, "policies", config, "training_config.yaml")
        with open(path) as f:
            cfg = yaml.safe_load(f)
        for section, values in (overrides or {}).items():
            cfg.setdefault(section, {}).update(values)
        self.continuous = action_space == "continuous"
        if self.continuous:
            self.env = build_continuous_env(cfg, int(seed))
            self.mdp = self.env.mdp_env
        else:
            self.env = build_env(cfg, int(seed))
            self.mdp = self.env
        self.base = self.mdp.unwrapped
        self.sign_indices = [i for i in range(len(self.base.objects))
                             if kindstr(self.base.objects[i]) == "sign_stop"]
        return {"config": config, "seed": int(seed),
                "action_space": action_space,
                "frame_skip": int(self.base.frame_skip),
                "max_steps": int(self.base.max_steps),
                "grid": [len(self.base.grid) // self.base.grid_width,
                         self.base.grid_width],
                "tile_size": enc(float(self.base.road_tile_size)),
                "n_signs": len(self.sign_indices)}

    # --- state export -----------------------------------------------------
    def ducks(self):
        return [o for o in self.base.objects if kindstr(o) == "duckie"]

    def export_state(self):
        b = self.base
        st = b.state                    # DelayedDynamics
        inner = st.state                # DynamicModel
        ctrl = self.mdp.duck_controller
        tracker = self.mdp.stop_tracker
        last = self.mdp._last_state
        lid = self.mdp._last_stop_id
        out = {
            "ego": {
                "pos": enc(np.asarray(b.cur_pos, dtype=float)),
                "angle": enc(float(b.cur_angle)),
                "speed": enc(float(b.speed)),
                "step_count": int(b.step_count),
                "timestamp": enc(float(b.timestamp)),
                "state_t": enc(float(st.t)),
                "inner_t0": enc(float(inner.t0)),
                "q0": enc(np.asarray(inner.q0, dtype=float)),
                "v0": enc(np.asarray(inner.v0, dtype=float)),
                "axis_left_rad": enc(float(inner.axis_left_rad)),
                "axis_right_rad": enc(float(inner.axis_right_rad)),
                "commands": [[enc(float(t)), enc(float(c.motor_left)),
                              enc(float(c.motor_right))]
                             for t, c in zip(st.timestamps, st.commands)],
            },
            "ducks": [{
                "pos": enc(np.asarray(d.pos, dtype=float)),
                "center": enc(np.asarray(d.center, dtype=float)),
                "start": enc(np.asarray(d.start, dtype=float)),
                "angle": enc(float(d.angle)),
                "heading": enc(np.asarray(d.heading, dtype=float)),
                "vel": enc(float(d.vel)),
                "visible": bool(d.visible),
                "active": bool(d.pedestrian_active),
                "wait": enc(float(d.pedestrian_wait_time)),
                "time": enc(float(d.time)),
                "walk_distance": enc(float(d.walk_distance)),
                "scale": enc(float(d.scale)),
                "safety_radius": enc(float(d.safety_radius)),
                "min_coords": enc(np.asarray(d.min_coords, dtype=float)),
                "max_coords": enc(np.asarray(d.max_coords, dtype=float)),
                "corners": enc(np.asarray(d.obj_corners, dtype=float)),
                "norm": enc(np.asarray(d.obj_norm, dtype=float)),
            } for d in self.ducks()],
            "signs": [{"pos": enc(np.asarray(self.base.objects[i].pos, dtype=float)),
                       "angle": enc(float(self.base.objects[i].angle))}
                      for i in self.sign_indices],
            "controller": {
                "crossings_started": [int(v) for v in ctrl.crossings_started],
                "crossing_armed": [bool(v) for v in ctrl.crossing_armed],
            },
            "memory": {
                "sigma_stop": bool(tracker.sigma_stop),
                "hold_steps": int(tracker.hold_steps),
                "hold_steps_required": int(tracker.hold_steps_required),
                "last_stop_id": (None if lid is None
                                 else self.sign_indices.index(lid)),
                "last_d_stop": enc(None if last is None else last.d_stop),
                "lane_fallback": enc([float(v) for v in
                                      getattr(b, "_mdp_last_lane_position", (1.0, 1.0))]),
                "env_sigma_stop": bool(getattr(b, "_mdp_sigma_stop", False)),
            },
        }
        if last is not None:
            out["raw_state"] = {
                "d": enc(float(last.d)), "phi": enc(float(last.phi)),
                "v": enc(float(last.v)), "tile": int(last.tile),
                "d_stop": enc(last.d_stop),
                "sigma_stop": bool(last.sigma_stop), "duck": int(last.duck),
            }
        if self.continuous and self.env.current_state is not None:
            out["continuous_state"] = {
                k: enc(v) for k, v in
                continuous_state_to_dict(self.env.current_state).items()}
        return out

    # --- state import (matched-state parity) ------------------------------
    def import_state(self, s):
        b = self.base
        ego = s["ego"]
        q0 = np.array([[dec(v) for v in row] for row in ego["q0"]], dtype=float)
        v0 = np.array([[dec(v) for v in row] for row in ego["v0"]], dtype=float)
        t0 = dec(ego["inner_t0"])
        inner = DynamicModel(b.state.state.parameters, (q0, v0), t0,
                             dec(ego["axis_left_rad"]),
                             dec(ego["axis_right_rad"]))
        # DelayedDynamics(state, delay, t0, u0, commands, timestamps): the
        # constructor re-applies the SAME window trim the simulator applies
        # after each tick, so injecting an already-trimmed window is a no-op
        from duckietown_world.world_duckietown.pwm_dynamics import PWMCommands
        commands = [PWMCommands(motor_left=dec(c[1]), motor_right=dec(c[2]))
                    for c in ego["commands"]]
        timestamps = [dec(c[0]) for c in ego["commands"]]
        delayed = DelayedDynamics(inner, b.state.delay, dec(ego["state_t"]),
                                  b.state.u0, commands, timestamps)
        delayed.t = dec(ego["state_t"])
        b.state = delayed
        b.cur_pos = np.array([dec(v) for v in ego["pos"]], dtype=float)
        b.cur_angle = dec(ego["angle"])
        b.speed = dec(ego["speed"])
        b.step_count = int(ego["step_count"])
        b.timestamp = dec(ego["timestamp"])

        for duck, ds in zip(self.ducks(), s["ducks"]):
            duck.pos = np.array([dec(v) for v in ds["pos"]], dtype=float)
            duck.center = np.array([dec(v) for v in ds["center"]], dtype=float)
            duck.start = np.array([dec(v) for v in ds["start"]], dtype=float)
            duck.angle = dec(ds["angle"])
            duck.heading = np.array([dec(v) for v in ds["heading"]], dtype=float)
            duck.vel = dec(ds["vel"])
            duck.visible = bool(ds["visible"])
            duck.pedestrian_active = bool(ds["active"])
            duck.pedestrian_wait_time = dec(ds["wait"])
            duck.time = dec(ds["time"])
            duck.walk_distance = dec(ds["walk_distance"])
            duck.obj_corners = np.array(
                [[dec(v) for v in c] for c in ds["corners"]], dtype=float)
            duck.obj_norm = np.array(
                [[dec(v) for v in c] for c in ds["norm"]], dtype=float)

        ctrl = self.mdp.duck_controller
        ctrl.crossings_started = [int(v) for v in s["controller"]["crossings_started"]]
        ctrl.crossing_armed = [bool(v) for v in s["controller"]["crossing_armed"]]

        mem = s["memory"]
        tracker = self.mdp.stop_tracker
        tracker.sigma_stop = bool(mem["sigma_stop"])
        tracker.hold_steps = int(mem["hold_steps"])
        b._mdp_sigma_stop = bool(mem.get("env_sigma_stop", mem["sigma_stop"]))
        b._mdp_last_lane_position = tuple(dec(v) for v in mem["lane_fallback"])
        last_d_stop = dec(mem["last_d_stop"])
        # StopTracker.update reads only `previous.d_stop` off the previous
        # RawState, exactly like the Julia StopMemory
        self.mdp._last_state = RawState(0.0, 0.0, 0.0, TileType.STRAIGHT,
                                        last_d_stop, bool(mem["sigma_stop"]),
                                        DuckThreat.NONE)
        lid = mem["last_stop_id"]
        self.mdp._last_stop_id = (None if lid is None
                                  else self.sign_indices[int(lid)])
        if self.continuous:
            # the continuous wrapper reads self.current_state.kappa (pre-action)
            from src.continuous_state import build_continuous_state
            from src.state import get_raw_state
            raw = get_raw_state(self.mdp, tracker.sigma_stop, self.mdp.state_cfg)
            self.env.current_state = build_continuous_state(
                self.env, raw, self.mdp.state_cfg, self.env.continuous_cfg,
                ctrl, tracker.hold_progress)
        return {"imported": True}

    # --- stepping ---------------------------------------------------------
    def reset(self, seed=None):
        if seed is None:
            self.env.reset()
        else:
            self.env.reset(int(seed))
        return self.export_state()

    def step(self, action):
        if self.continuous:
            obs, r, done, info = self.env.step(
                np.asarray([float(action[0]), float(action[1])], dtype=np.float32))
        else:
            state, r, done, info = self.env.step(int(action))
        out = self.export_state()
        out["result"] = {
            "reward": enc(float(r)),
            "done": bool(done),
            "reward_terms": {k: enc(float(v)) for k, v in info["reward_terms"].items()},
            "events": {k: bool(v) for k, v in info["events"].items()},
            "reason": info["termination_reason"],
            "terminated": bool(info["terminated"]),
            "truncated": bool(info["truncated"]),
            "wheels": [enc(float(v)) for v in info["wheel_commands"]],
            "raw_state": {k: (enc(v) if not isinstance(v, bool) else v)
                          for k, v in info["raw_state"].items()},
        }
        if self.continuous:
            out["result"]["continuous_state"] = {
                k: enc(v) for k, v in info["continuous_state"].items()}
            out["result"]["observation"] = [enc(float(v)) for v in obs]
        return out

    # --- FJ5.4 stop-sign reachability probe --------------------------------
    def probe_stop(self, decisions=400, policy="lane_follow"):
        """Record every stop-candidate filter quantity per decision.

        The baseline configuration is NOT modified; this only observes.
        """
        from src.state import _lane_frame, _heading_vector, _kind
        b = self.base
        cfg = self.mdp.state_cfg
        rows = []
        if self.mdp._last_state is None:
            self.env.reset()

        def wrap(a):
            while a > math.pi:
                a -= 2 * math.pi
            while a < -math.pi:
                a += 2 * math.pi
            return a

        def lane_action():
            pt, tg = b.closest_curve_point(b.cur_pos, b.cur_angle)
            if tg is None:
                return 1
            th = math.atan2(-tg[2], tg[0])
            right = np.array([-math.sin(b.cur_angle), 0.0, -math.cos(b.cur_angle)])
            lat = float(np.dot(np.asarray(b.cur_pos, dtype=float)
                               - np.asarray(pt, dtype=float), right))
            om = -(1.5 * wrap(b.cur_angle - th) + 3.0 * (lat - 0.05))
            if om > 0.30:
                return 3
            if om < -0.30:
                return 5
            return 1 if abs(om) < 0.10 else 4

        for k in range(int(decisions)):
            forward, right = _lane_frame(b)
            per_sign = []
            for idx in self.sign_indices:
                obj = b.objects[idx]
                facing = _heading_vector(float(obj.angle))
                rel = (np.asarray(obj.pos, dtype=float)
                       - np.asarray(b.cur_pos, dtype=float))
                ahead = float(np.dot(rel, forward))
                lateral = abs(float(np.dot(rel, right)))
                orient = float(np.dot(facing, forward))
                distance = max(0.0, ahead - cfg.sign_to_line_offset)
                accepted = (orient <= -cfg.stop_orientation_cos and ahead > 0.0
                            and lateral <= cfg.stop_lateral_limit
                            and distance <= cfg.stop_max_distance)
                per_sign.append({
                    "ahead": enc(ahead), "lateral": enc(lateral),
                    "orient_dot": enc(orient), "distance": enc(distance),
                    "geometric": enc(float(np.linalg.norm(rel[[0, 2]]))),
                    "accepted": accepted,
                })
            d_stop, _ = next_stop_candidate(b, cfg)
            rows.append({
                "decision": k,
                "pos": enc(np.asarray(b.cur_pos, dtype=float)),
                "angle": enc(float(b.cur_angle)),
                "signs": per_sign,
                "d_stop": enc(d_stop),
            })
            a = lane_action() if policy == "lane_follow" else int(policy)
            _, _, done, info = self.env.step(a)
            if done:
                rows[-1]["terminated_after"] = info["termination_reason"]
                self.env.reset()
        return {"rows": rows}


def main():
    claim_protocol_channel()
    session = Session()
    handlers = {
        "init": lambda p: session.init(p["config"], p["seed"],
                                       p.get("action_space", "discrete"),
                                       p.get("overrides")),
        "reset": lambda p: session.reset(p.get("seed")),
        "get_state": lambda p: session.export_state(),
        "set_state": lambda p: session.import_state(p["state"]),
        "step": lambda p: session.step(p["action"]),
        "probe_stop": lambda p: session.probe_stop(p.get("decisions", 400),
                                                   p.get("policy", "lane_follow")),
        "ping": lambda p: {"pong": True},
    }
    sys.stderr.write("reference_server ready\n")
    sys.stderr.flush()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except Exception as e:
            respond({"ok": False, "error": f"bad json: {e}"})
            continue
        cmd = msg.get("cmd")
        if cmd == "quit":
            respond({"ok": True, "result": {"bye": True}})
            return
        handler = handlers.get(cmd)
        if handler is None:
            respond({"ok": False, "error": f"unknown cmd {cmd}"})
            continue
        try:
            respond({"ok": True, "result": handler(msg)})
        except Exception:
            respond({"ok": False, "error": traceback.format_exc()})


if __name__ == "__main__":
    main()
