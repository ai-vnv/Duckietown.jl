"""FJ2 parity fixture generator.

Runs the REAL duckduck/src Python functions (actions, discretizer, state
helpers, StopTracker, compute_reward, continuous encoding) against a
deterministic case list and dumps expected outputs to
test/fixtures/fj2_parity.json.

The duckduck modules import from `gym` and `gym_duckietown` at module level,
which are not installed here; we inject stub packages whose `simulator` module
carries the VERBATIM bezier_point/bezier_tangent implementations from the
pinned duckietown-gym-daffy-6.1.34 source (src/gym_duckietown/graphics.py) so
curvature fixtures are computed with the real functions.

Run (WSL, numpy available):
    python3 tools/parity/gen_fj2_fixtures.py
"""
import importlib.util
import json
import math
import sys
import types
from pathlib import Path

import numpy as np

PACKAGE = Path(__file__).resolve().parents[2]
DUCKDUCK = PACKAGE.parent / "duckduck"
OUT = PACKAGE / "test" / "fixtures" / "fj2_parity.json"

SEED = 20260818


def enc(x):
    """JSON-safe float encoding; non-finite values become markers."""
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
            # JSON cannot round-trip the sign of -0.0 (JSON3 parses it as +0.0).
            return {"nonfinite": "-0.0"}
        return f
    if x is None:
        return None
    raise TypeError(type(x))


def enc32(x):
    """float32 value as its exact float64 representation."""
    return enc(np.float32(x))


# --------------------------------------------------------------------------
# Stub gym / gym_duckietown with verbatim bezier from the pinned source
# --------------------------------------------------------------------------

def bezier_point(cps, t):
    p = ((1 - t) ** 3) * cps[0, :]
    p += 3 * t * ((1 - t) ** 2) * cps[1, :]
    p += 3 * (t**2) * (1 - t) * cps[2, :]
    p += (t**3) * cps[3, :]
    return p


def bezier_tangent(cps, t):
    p = 3 * ((1 - t) ** 2) * (cps[1, :] - cps[0, :])
    p += 6 * (1 - t) * t * (cps[2, :] - cps[1, :])
    p += 3 * (t**2) * (cps[3, :] - cps[2, :])
    norm = np.linalg.norm(p)
    p /= norm
    return p


class _Box:
    def __init__(self, low, high, dtype):
        self.low, self.high, self.dtype = low, high, dtype


class _NotInLane(Exception):
    pass


spaces = types.ModuleType("gym.spaces")
spaces.Box = _Box
gym = types.ModuleType("gym")
gym.spaces = spaces

simulator = types.ModuleType("gym_duckietown.simulator")
simulator.bezier_point = bezier_point
simulator.bezier_tangent = bezier_tangent
simulator.NotInLane = _NotInLane
gymd = types.ModuleType("gym_duckietown")
gymd.simulator = simulator

sys.modules["gym"] = gym
sys.modules["gym.spaces"] = spaces
sys.modules["gym_duckietown"] = gymd
sys.modules["gym_duckietown.simulator"] = simulator

sys.path.insert(0, str(DUCKDUCK))
import src.actions as actions_mod
import src.discretizer as discretizer_mod
import src.reward as reward_mod
import src.state as state_mod
import src.continuous_state as cont_mod

ActionConfig = actions_mod.ActionConfig
ActionSpec = actions_mod.ActionSpec
build_action_table = actions_mod.build_action_table
vw_to_wheels = actions_mod.vw_to_wheels
action_to_wheels = actions_mod.action_to_wheels

discretize = discretizer_mod.discretize
STATE_SHAPE = discretizer_mod.STATE_SHAPE

RawState = state_mod.RawState
TileType = state_mod.TileType
DuckThreat = state_mod.DuckThreat
classify_tile = state_mod.classify_tile
_terminal_lane_fallback = state_mod._terminal_lane_fallback

StopTracker = reward_mod.StopTracker
EventFlags = reward_mod.EventFlags
RewardConfig = reward_mod.RewardConfig
compute_reward = reward_mod.compute_reward

ContinuousState = cont_mod.ContinuousState
ContinuousStateConfig = cont_mod.ContinuousStateConfig
DuckRelativeState = cont_mod.DuckRelativeState
encode_continuous_state = cont_mod.encode_continuous_state
continuous_observation_space = cont_mod.continuous_observation_space
gate_duck_visibility = cont_mod.gate_duck_visibility
curve_signed_curvature = cont_mod.curve_signed_curvature

# --------------------------------------------------------------------------
# Case lists
# --------------------------------------------------------------------------

out = {"meta": {"python": sys.version.split()[0], "numpy": np.__version__,
                "duckduck": str(DUCKDUCK), "seed": SEED}}

# --- FJ2.1 actions --------------------------------------------------------

vw_cases = []
for v in (-1.2, -1.0, -0.5, -0.15, 0.0, 0.15, 0.4, 1.0, 1.2):
    for omega in (-3.0, -2.0, -1.5, -1.0, 0.0, 1.0, 1.5, 2.0, 3.0):
        for wb in (0.102, 0.2):
            wheels = vw_to_wheels(v, omega, wb)
            vw_cases.append([v, omega, wb, [enc32(wheels[0]), enc32(wheels[1])]])

table = build_action_table(ActionConfig())
table_case = [[spec.name, spec.v, spec.omega] for spec in table]
atw_cases = []
for aid in range(7):
    wheels = action_to_wheels(aid)
    atw_cases.append([aid, [enc32(wheels[0]), enc32(wheels[1])]])
atw_alt = []
for cfg in (ActionConfig(v_fast=0.41, v_slow=0.17), ActionConfig(v_fast=0.30, v_slow=0.10, w0=2.0, wheel_base=0.12)):
    for aid in (0, 3, 6):
        wheels = action_to_wheels(aid, cfg)
        atw_alt.append([aid, cfg.v_fast, cfg.v_slow, cfg.w0, cfg.wheel_base,
                        [enc32(wheels[0]), enc32(wheels[1])]])

out["actions"] = {"table": table_case, "vw_to_wheels": vw_cases,
                  "action_to_wheels": atw_cases, "action_to_wheels_alt": atw_alt,
                  "bad_ids": [7, -1, 100]}

# --- FJ2.2 discretizer ----------------------------------------------------

DS = {"d": (-0.3, -0.25, -0.16, -0.15, -0.14, -0.06, -0.05, -0.04, 0.0, 0.04,
            0.05, 0.06, 0.14, 0.15, 0.16, 0.24, 0.25, 0.3),
      "phi": (-0.6, -0.51, -0.5, -0.49, -0.11, -0.1, -0.09, 0.0, 0.09, 0.1,
              0.11, 0.49, 0.5, 0.51, 0.6),
      "v": (0.0, 0.03, 0.04, 0.05, 0.15, 0.16, 0.17, 0.41, 0.5)}

disc_cases = []
for d in DS["d"]:
    for phi in DS["phi"]:
        for v in DS["v"]:
            st = RawState(d, phi, v, TileType.STRAIGHT, None, False, DuckThreat.NONE)
            idx = list(discretize(st))
            disc_cases.append([d, phi, v, 0, None, False, 0, idx])

for tile in range(3):
    for d_stop in (None, 0.1, 0.29, 0.3, 0.31, 0.99, 1.0, 1.01, 2.5):
        for sigma in (False, True):
            for duck in range(5):
                st = RawState(0.0, 0.0, 0.1, TileType(tile), d_stop, sigma, DuckThreat(duck))
                idx = list(discretize(st))
                disc_cases.append([0.0, 0.0, 0.1, tile, d_stop, sigma, duck, idx])

disc_errors = []
# The Python guard `any(i < 0 or i >= n for ...)` is defensive: with the four
# bin edges of D_BINS/TRACKING_ERROR_BINS/V_BINS the digitized index can never
# leave STATE_SHAPE for valid enums (max digitize = len(bins) < shape). The
# Julia port keeps the same defensive IndexError.

out["discretize"] = {"cases": disc_cases, "errors": disc_errors,
                     "state_shape": list(STATE_SHAPE)}

# --- FJ2.3 continuous encoding --------------------------------------------

CONT_CFG = (
    ContinuousStateConfig(),
    ContinuousStateConfig(max_speed=0.5, max_abs_curvature=4.0,
                          max_stop_distance=1.5, max_duck_distance=1.5,
                          max_relative_speed=1.0),
)
enc_cases = []
for cfg_id, cfg in enumerate(CONT_CFG):
    for d in (-0.3, -0.25, 0.0, 0.25, 0.3):
        for phi in (-1.6, -1.571, 0.0, 1.571, 1.6):
            for v in (0.0, 0.2, 0.41, 0.5, 1.2):
                for kappa in (-9.0, -8.0, 0.0, 8.0, 9.0):
                    st = ContinuousState(d, phi, v, kappa, True, 1.2, False,
                                         True, 1.5, -0.5, 0.3, -0.2, True, True, 0.5)
                    enc_cases.append([d, phi, v, kappa, True, 1.2, False,
                                      True, 1.5, -0.5, 0.3, -0.2, True, True, 0.5,
                                      cfg_id, [enc32(x) for x in encode_continuous_state(st, cfg)]])
for cfg_id, cfg in enumerate(CONT_CFG):
    for stop_present, d_stop in ((False, None), (True, None), (True, -0.4),
                                 (True, 0.0), (True, 1.5), (True, 3.0), (True, 9.0)):
        for duck_present, dlong, dlat, vlong, vlat, active, crossing in (
                (False, 1.0, 0.0, 0.0, 0.0, False, False),
                (True, -2.0, -2.0, -1.0, -1.0, False, False),
                (True, 2.0, 2.0, 1.0, 1.0, True, True),
                (True, 0.0, 0.0, 0.0, 0.0, True, False)):
            for sigma in (False, True):
                for hold in (-0.1, 0.0, 0.33, 1.0, 1.5):
                    st = ContinuousState(0.0, 0.0, 0.2, 0.0, stop_present, d_stop,
                                         sigma, duck_present, dlong, dlat, vlong,
                                         vlat, active, crossing, hold)
                    enc_cases.append([0.0, 0.0, 0.2, 0.0, stop_present, d_stop,
                                      sigma, duck_present, dlong, dlat, vlong,
                                      vlat, active, crossing, hold, cfg_id,
                                      [enc32(x) for x in encode_continuous_state(st, cfg)]])

space = continuous_observation_space()
out["encoding"] = {
    "cases": enc_cases,
    "low": [enc32(x) for x in np.asarray(space.low, dtype=np.float32)],
    "high": [enc32(x) for x in np.asarray(space.high, dtype=np.float32)],
}

# --- gate_duck_visibility -------------------------------------------------

GATE_CFG = (
    ContinuousStateConfig(),
    ContinuousStateConfig(duck_detection_range=1.2, duck_detection_corridor_width=0.6,
                          duck_detection_forward_only=True),
    ContinuousStateConfig(duck_detection_range=1.2),
    ContinuousStateConfig(duck_detection_corridor_width=0.6),
    ContinuousStateConfig(duck_detection_forward_only=True),
)
gate_cases = []
for cfg_id, cfg in enumerate(GATE_CFG):
    for present in (False, True):
        for long, lat in ((-0.5, 0.0), (0.0, 0.0), (0.5, 0.3), (1.0, 0.6),
                          (1.3, 0.0), (0.5, 0.7), (1.1, 0.55)):
            duck = DuckRelativeState(present, long, lat, 0.2, -0.1, True, True)
            g = gate_duck_visibility(duck, cfg)
            gate_cases.append([present, long, lat, 0.2, -0.1, True, True, cfg_id,
                               [g.present, g.longitudinal, g.lateral,
                                g.v_longitudinal_relative, g.v_lateral_relative,
                                g.active, g.crossing_available]])

out["gate"] = {"cases": gate_cases}

# --- classify_tile --------------------------------------------------------

kind_cases = []
for drivable, kind in ((True, "straight"), (True, "curve_left"),
                       (True, "curve_right"), (True, "3way"),
                       (True, "3way_left"), (True, "4way"),
                       (True, "Curve_Left")):
    kind_cases.append([drivable, kind, int(classify_tile({"drivable": drivable,
                                                          "kind": kind}))])
kind_errors = []
for drivable, kind in ((False, "straight"), (True, "oval"), (True, ""),
                       (True, None)):
    try:
        classify_tile({"drivable": drivable, "kind": kind})
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    kind_errors.append([drivable, kind])
try:
    classify_tile(None)
    raise AssertionError("expected ValueError")
except ValueError:
    pass
kind_errors.append([None, None])
out["classify_tile"] = {"cases": kind_cases, "errors": kind_errors}

# --- FJ2.4 StopTracker ----------------------------------------------------

def raw(d, phi, v, d_stop, sigma=False, duck=0, tile=0):
    return RawState(d, phi, v, TileType(tile), d_stop, sigma, DuckThreat(duck))


ST_SEQ = []
def raw_case(st):
    return [st.d, st.phi, st.v, int(st.tile), st.d_stop, st.sigma_stop, int(st.duck)]


def run_seq(cfg, states_events):
    tr = StopTracker(*cfg)
    results = []
    for (prev, curr, prev_id, curr_id) in states_events:
        sigma, events = tr.update(prev, curr, prev_id, curr_id)
        results.append([raw_case(prev), raw_case(curr), prev_id, curr_id, sigma,
                        [events.collision_duck, events.other_collision,
                         events.offroad, events.timeout, events.stop_violation,
                         events.full_stop, events.passed_stop, events.goal],
                        tr.sigma_stop, tr.hold_steps])
    return results

CFG_1 = (0.45, 0.02, 0.55, 1)
CFG_3 = (0.45, 0.02, 0.55, 3)
CFG_CLAMP = (0.45, 0.02, 0.30, 0)
CFG_ZONE2 = (0.60, 0.05, 0.70, 2)

S1 = run_seq(CFG_1, [
    (raw(0.1, 0.0, 0.3, 0.40), raw(0.1, 0.0, 0.01, 0.35), 3, 3),
    (raw(0.1, 0.0, 0.01, 0.35), raw(0.1, 0.0, 0.30, 0.30), 3, 3),
    (raw(0.1, 0.0, 0.30, 0.30), raw(0.1, 0.0, 0.01, 0.25), 3, 4),
])
S2 = run_seq(CFG_1, [
    (raw(0.1, 0.0, 0.3, 0.50), raw(0.1, 0.0, 0.01, 0.50), None, None),
    (raw(0.1, 0.0, 0.01, 0.50), raw(0.1, 0.0, 0.01, 0.50), None, None),
])
S3 = run_seq(CFG_3, [
    (raw(0.1, 0.0, 0.02, 0.30), raw(0.1, 0.0, 0.01, 0.30), 3, 3),
    (raw(0.1, 0.0, 0.01, 0.30), raw(0.1, 0.0, 0.01, 0.30), 3, 3),
    (raw(0.1, 0.0, 0.01, 0.30), raw(0.1, 0.0, 0.01, 0.30), 3, 3),
    (raw(0.1, 0.0, 0.01, 0.30), raw(0.1, 0.0, 0.30, 0.30), 3, 3),
])
S4 = run_seq(CFG_3, [
    (raw(0.1, 0.0, 0.01, 0.30), raw(0.1, 0.0, 0.20, 0.30), 3, 3),
    (raw(0.1, 0.0, 0.20, 0.30), raw(0.1, 0.0, 0.01, 0.30), 3, 3),
])
S5 = run_seq(CFG_1, [
    (raw(0.1, 0.0, 0.01, 0.30), raw(0.1, 0.0, 0.01, 0.30), 3, 4),
])
S6 = run_seq(CFG_1, [
    (raw(0.1, 0.0, 0.01, 0.30), raw(0.1, 0.0, 0.01, 0.30), 3, 3),
    (raw(0.1, 0.0, 0.01, 0.30), raw(0.1, 0.0, 0.01, 0.30), 3, 3),
])
S7 = run_seq(CFG_1, [
    (raw(0.1, 0.0, 0.01, 0.30), raw(0.1, 0.0, 0.02, 0.10), 3, 3),
])
S8 = run_seq(CFG_1, [
    (raw(0.1, 0.0, 0.01, 0.70), raw(0.1, 0.0, 0.01, 0.30), 3, 3),
    (raw(0.1, 0.0, 0.01, 0.30), raw(0.1, 0.0, 0.30, 0.30), 3, 3),
])
S9 = run_seq(CFG_1, [
    (raw(0.1, 0.0, 0.01, 0.10), raw(0.1, 0.0, 0.01, 0.30), None, None),
])
S10 = run_seq(CFG_1, [
    (raw(0.1, 0.0, 0.01, 0.60), raw(0.1, 0.0, 0.30, 0.30), None, None),
])
S11 = run_seq(CFG_1, [
    (raw(0.1, 0.0, 0.01, 0.40), raw(0.1, 0.0, 0.01, 0.40), None, None),
    (raw(0.1, 0.0, 0.01, 0.40), raw(0.1, 0.0, 0.01, 0.40), 5, 5),
])
S12 = run_seq(CFG_1, [
    (raw(0.1, 0.0, 0.01, 0.40), raw(0.1, 0.0, 0.01, 0.40), None, 5),
])
S13 = run_seq(CFG_1, [
    (raw(0.1, 0.0, 0.01, 0.40), raw(0.1, 0.0, 0.01, 0.40), None, 5),
    (raw(0.1, 0.0, 0.01, 0.40), raw(0.1, 0.0, 0.01, 0.40), 5, 5),
])
S14 = run_seq(CFG_CLAMP, [
    (raw(0.1, 0.0, 0.01, 0.40), raw(0.1, 0.0, 0.01, 0.40), None, None),
])
S15 = run_seq(CFG_ZONE2, [
    (raw(0.1, 0.0, 0.04, 0.55), raw(0.1, 0.0, 0.03, 0.55), None, None),
    (raw(0.1, 0.0, 0.03, 0.55), raw(0.1, 0.0, 0.03, 0.55), None, None),
    (raw(0.1, 0.0, 0.03, 0.55), raw(0.1, 0.0, 0.03, 0.55), None, None),
])
S16 = run_seq(CFG_ZONE2, [
    (raw(0.1, 0.0, 0.03, 0.55), raw(0.1, 0.0, 0.02, 0.55), None, None),
    (raw(0.1, 0.0, 0.02, 0.55), raw(0.1, 0.0, 0.04, 0.55), None, None),
    (raw(0.1, 0.0, 0.04, 0.55), raw(0.1, 0.0, 0.02, 0.55), None, None),
    (raw(0.1, 0.0, 0.02, 0.55), raw(0.1, 0.0, 0.02, 0.55), None, None),
])

ST_CASES = [
    ["hold1", list(CFG_1), S1],
    ["noids-pass", list(CFG_1), S2],
    ["hold3", list(CFG_3), S3],
    ["nonconsecutive", list(CFG_3), S4],
    ["changed-id", list(CFG_1), S5],
    ["already-stopped", list(CFG_1), S6],
    ["slow-but-far", list(CFG_1), S7],
    ["near-but-fast", list(CFG_1), S8],
    ["passed-noids-far", list(CFG_1), S9],
    ["passed-noids-discontinuity", list(CFG_1), S10],
    ["id-flip", list(CFG_1), S11],
    ["id-appears", list(CFG_1), S12],
    ["id-appears-then-stable", list(CFG_1), S13],
    ["clamped-pass", list(CFG_CLAMP), S14],
    ["zone2-accumulate", list(CFG_ZONE2), S15],
    ["zone2-reset-on-speed", list(CFG_ZONE2), S16],
]
out["stop_tracker"] = {"cases": ST_CASES}

# --- FJ2.5 reward ---------------------------------------------------------

RCFG = (
    RewardConfig(),
    RewardConfig(duck_yield=1.0, duck_unsafe=-5.0, unnecessary_stop=-2.0),
    RewardConfig(duck_unsafe=-5.0, unnecessary_stop=-2.0,
                 stop_approach_distance=0.60, stop_approach_speed=0.02,
                 stop_approach_yield=1.0, stop_approach_unsafe=-5.0,
                 straight_steer_penalty=0.5),
    RewardConfig(alpha_progress=1.0, alpha_lateral=2.0, alpha_heading=0.5,
                 step_cost=0.01, collision_duck=-200.0, other_collision=-50.0,
                 offroad=-200.0, stop_violation=-40.0, full_stop=15.0,
                 goal=50.0),
)
EVENTS = (
    EventFlags(),
    EventFlags(collision_duck=True),
    EventFlags(offroad=True),
    EventFlags(collision_duck=True, offroad=True),
    EventFlags(stop_violation=True),
    EventFlags(full_stop=True),
    EventFlags(stop_violation=True, full_stop=True),
    EventFlags(goal=True),
    EventFlags(collision_duck=True, other_collision=True, offroad=True,
               timeout=True, stop_violation=True, full_stop=True,
               passed_stop=True, goal=True),
)
EVENT_LISTS = [[e.collision_duck, e.other_collision, e.offroad, e.timeout,
                e.stop_violation, e.full_stop, e.passed_stop, e.goal]
               for e in EVENTS]


def reward_case(st, omega, curvature, cfg_id, ev_id):
    bd = compute_reward(st, EVENTS[ev_id], RCFG[cfg_id], omega, curvature)
    return [st.d, st.phi, st.v, int(st.tile), st.d_stop, st.sigma_stop,
            int(st.duck), omega, enc(curvature), cfg_id, ev_id,
            [enc(x) for x in (bd.progress, bd.lateral, bd.heading, bd.time,
                              bd.pedestrian, bd.stagnation, bd.stop_approach,
                              bd.steering, bd.events, bd.total)]]

R = []
# dense sweep: v x duck x d_stop x sigma, default cfg, events 0/4/5/7
for v in (0.0, 0.02, 0.03, 0.04, 0.05, 0.2, 0.41):
    for duck in range(5):
        for d_stop in (None, 0.2, 0.45, 0.46, 1.0):
            for sigma in (False, True):
                for ev_id in (0, 4, 5, 7):
                    st = raw(0.0, 0.0, v, d_stop, sigma, duck)
                    R.append(reward_case(st, 0.0, None, 0, ev_id))
# heading / lateral sweep
for d in (-0.25, -0.1, 0.0, 0.1, 0.25):
    for phi in (-1.4, -0.5, 0.0, 0.5, 1.4):
        st = raw(d, phi, 0.2, None, False, 0)
        R.append(reward_case(st, 0.0, None, 0, 0))
# crossing shaping with teacher-free config
for v in (0.0, 0.03, 0.04, 0.05, 0.06, 0.2):
    for duck in (3, 4):
        st = raw(0.0, 0.0, v, None, False, duck)
        R.append(reward_case(st, 0.0, None, 1, 0))
# stagnation vs must_stop interplay
for v in (0.0, 0.02, 0.03, 0.04, 0.05, 0.2):
    for d_stop in (None, 0.3, 0.45, 0.46, 0.7):
        for sigma in (False, True):
            st = raw(0.0, 0.0, v, d_stop, sigma, 0)
            R.append(reward_case(st, 0.0, None, 1, 0))
# td3 stop-approach + steering
for v in (0.0, 0.01, 0.02, 0.03, 0.04, 0.2):
    for d_stop in (None, 0.59, 0.6, 0.61, 0.9, 1.5):
        for sigma in (False, True):
            st = raw(0.0, 0.0, v, d_stop, sigma, 0)
            R.append(reward_case(st, 0.0, None, 2, 0))
for curvature in (None, 0.0, 0.049, 0.05, 0.051, 2.0):
    for omega in (0.0, 0.5, 1.5, 1.6, 2.0, -1.5):
        st = raw(0.0, 0.0, 0.2, None, False, 0)
        R.append(reward_case(st, omega, curvature, 2, 0))
# tabular config: full event combination
for ev_id in range(len(EVENTS)):
    st = raw(0.0, 0.0, 0.2, 0.3, False, 0)
    R.append(reward_case(st, 0.0, None, 3, ev_id))
# misc boundary: v at yield-speed edges with crossing
for v in (0.0399, 0.04, 0.0401):
    st = raw(0.0, 0.0, v, None, False, 4)
    R.append(reward_case(st, 0.0, None, 1, 0))

out["reward"] = {"cases": R}

# --- FJ2.6 geometry + projection helpers ----------------------------------

straight_curve = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0],
                           [1.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
left_curve = np.array([[0.0, 0.0, 0.0], [0.4, 0.0, 0.2],
                       [0.8, 0.0, 0.6], [1.2, 0.0, 1.2]])
right_curve = np.array([[0.0, 0.0, 0.0], [0.4, 0.0, -0.2],
                        [0.8, 0.0, -0.6], [1.2, 0.0, -1.2]])
tiny_curve = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.005],
                       [1.0, 0.0, 0.010], [1.5, 0.0, 0.015]])
degenerate = np.array([[0.5, 0.0, 0.5]] * 4)

CURVES = {"straight": straight_curve, "left": left_curve,
          "right": right_curve, "tiny": tiny_curve, "degenerate": degenerate}

bezier_cases = []
for name, curve in CURVES.items():
    for t in (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0):
        bezier_cases.append([name, t, "point", enc(bezier_point(curve, t))])
        bezier_cases.append([name, t, "tangent", enc(bezier_tangent(curve, t))])

curv_cases = []
for name, curve in CURVES.items():
    curv_cases.append([name, 33, 0.05, enc(curve_signed_curvature(curve, 33, 0.05))])
    curv_cases.append([name, 5, 0.05, enc(curve_signed_curvature(curve, 5, 0.05))])
for th in (0.01, 0.05, 0.10):
    curv_cases.append(["tiny", 33, th, enc(curve_signed_curvature(tiny_curve, 33, th))])
curv_errors = [["straight", 2, 0.05]]

out["geometry"] = {"curves": {k: enc(v) for k, v in CURVES.items()},
                   "bezier": bezier_cases,
                   "curvature": curv_cases,
                   "curvature_errors": curv_errors}

tf_cases = []
for last_d, last_phi in ((None, None), (0.1, 0.2), (0.0, 0.3), (0.3, 0.0),
                         (0.0, 0.0), (1e-10, 0.5), (0.5, 1e-10), (-0.1, -0.2),
                         (-0.0, -0.0), (-0.001, 0.3)):
    if last_d is None:
        d, phi = _terminal_lane_fallback(object())
    else:
        d, phi = _terminal_lane_fallback(type("E", (), {"_mdp_last_lane_position": (last_d, last_phi)})())
    tf_cases.append([last_d, last_phi, [d, phi]])
out["terminal_fallback"] = {"cases": tf_cases}

# --------------------------------------------------------------------------

OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w") as fh:
    json.dump(out, fh, indent=1, sort_keys=True)
print(f"wrote {OUT}")
print(f"actions.vw_to_wheels={len(vw_cases)}  discretize={len(disc_cases)}  "
      f"encoding={len(enc_cases)}  reward={len(R)}  stop_tracker={len(ST_CASES)}  "
      f"bezier={len(bezier_cases)}  curvature={len(curv_cases)}")