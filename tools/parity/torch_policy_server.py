# FJ7.4a / FJ7.5a: the PyTorch policy oracle.
#
# Runs in the SEPARATE `ddm-torch` env (python 3.11 + CPU-only torch). It is
# deliberately NOT the simulator reference backend and not the PythonCall
# interpreter: the validated `ddm-ref` (python 3.9) stays exactly as it is,
# and this process only ever
#
#     loads a checkpoint  ->  runs the actor forward  ->  returns activations
#
# The actor classes are the REAL reference ones imported from
# `duckduck/src/agents/` (they depend on nothing but numpy + torch), so this
# is the reference network, not a re-implementation. Architecture parameters
# come from the checkpoint itself (`obs_dim`, `config.hidden_size`,
# `action_low`, `action_high`) — nothing is guessed.
#
# Protocol: one JSON object per line in, one per line out (same convention as
# the simulator reference server).
#
# Run:  conda activate ddm-torch && python tools/parity/torch_policy_server.py
import json
import math
import os
import sys
import traceback
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch

sys.path.insert(0, "/home/pannntastic/aivnv/duckduck")
from src.agents.sac import SquashedGaussianActor
from src.agents.td3 import DeterministicActor

POLICIES = "/home/pannntastic/aivnv/duckduck/policies"

PROTOCOL = None


def claim_protocol_channel():
    """Own stdout for the line protocol; library chatter goes to stderr."""
    global PROTOCOL
    if PROTOCOL is None:
        fd = os.dup(1)
        os.dup2(2, 1)
        PROTOCOL = os.fdopen(fd, "w")
    return PROTOCOL


def respond(obj):
    PROTOCOL.write(json.dumps(obj) + "\n")
    PROTOCOL.flush()


def enc(x):
    if isinstance(x, (np.ndarray, list, tuple)):
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


class PolicySession:
    """One loaded reference actor, in eval mode, with grad disabled."""

    def __init__(self, name, weights_dir):
        path = os.path.join(POLICIES, name, "policy.pt")
        # weights_only=False is required: the checkpoint stores numpy arrays
        # and the training config alongside the tensors. It is the project's
        # own frozen artefact.
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.name = name
        self.obs_dim = int(ck["obs_dim"])
        self.action_low = np.asarray(ck["action_low"], dtype=np.float32)
        self.action_high = np.asarray(ck["action_high"], dtype=np.float32)
        hidden = int(ck["config"]["hidden_size"])
        cls = SquashedGaussianActor if name == "sac" else DeterministicActor
        self.actor = cls(self.obs_dim, self.action_low, self.action_high, hidden)
        missing, unexpected = self.actor.load_state_dict(ck["actor"], strict=True)
        self.actor.eval()
        for p in self.actor.parameters():
            p.requires_grad_(False)
        self.hidden = hidden
        self.checkpoint_path = path

        # export every actor parameter as a .npy file so the Julia side can
        # read the EXACT bits with its own native reader (no float formatting
        # in the path, and no PyTorch dependency in Julia)
        self.weights_dir = os.path.join(weights_dir, name)
        os.makedirs(self.weights_dir, exist_ok=True)
        self.param_files = {}
        for key, tensor in ck["actor"].items():
            fname = key.replace(".", "_") + ".npy"
            np.save(os.path.join(self.weights_dir, fname),
                    tensor.detach().cpu().numpy())
            self.param_files[key] = os.path.join(self.weights_dir, fname)

    def meta(self):
        return {
            "policy": self.name,
            "obs_dim": self.obs_dim,
            "hidden": self.hidden,
            "action_low": enc(self.action_low),
            "action_high": enc(self.action_high),
            "checkpoint": self.checkpoint_path,
            "weights_dir": self.weights_dir,
            "param_files": self.param_files,
            "param_shapes": {k: list(v.shape)
                             for k, v in self.actor.state_dict().items()},
            "torch": torch.__version__,
            "numpy": np.__version__,
            "python": sys.version.split()[0],
            "eval_mode": not self.actor.training,
            # the evaluation rule, read from the reference agent source:
            #   SAC : select_action(deterministic=True) returns
            #         tanh(mean) * action_scale + action_bias  and does NOT clip
            #   TD3 : actor(obs) = tanh(net(obs)) * scale + bias, then the
            #         agent ALWAYS applies np.clip(action, low, high)
            "eval_rule": ("tanh(mean)*scale+bias, no clip" if self.name == "sac"
                          else "tanh(net)*scale+bias, then np.clip"),
        }

    def infer(self, obs):
        x = torch.as_tensor(np.asarray(obs, dtype=np.float32),
                            dtype=torch.float32).unsqueeze(0)
        out = {"obs": enc(np.asarray(obs, dtype=np.float32))}
        with torch.no_grad():
            if self.name == "sac":
                b = self.actor.backbone
                h0 = b[0](x)                      # Linear(15, hidden)
                h1 = b[1](h0)                     # ReLU
                h2 = b[2](h1)                     # Linear(hidden, hidden)
                h3 = b[3](h2)                     # ReLU
                mean = self.actor.mean(h3)
                log_std = self.actor.log_std(h3)
                log_std_clamped = torch.clamp(log_std,
                    self.actor.LOG_STD_MIN, self.actor.LOG_STD_MAX)
                squashed = torch.tanh(mean)
                scaled = squashed * self.actor.action_scale + self.actor.action_bias
                # what the agent returns for deterministic evaluation
                _, _, deterministic = self.actor.sample(x)
                out.update({
                    "layer0_linear": enc(h0.squeeze(0).numpy()),
                    "layer1_relu": enc(h1.squeeze(0).numpy()),
                    "layer2_linear": enc(h2.squeeze(0).numpy()),
                    "layer3_relu": enc(h3.squeeze(0).numpy()),
                    "mean": enc(mean.squeeze(0).numpy()),
                    "log_std": enc(log_std.squeeze(0).numpy()),
                    "log_std_clamped": enc(log_std_clamped.squeeze(0).numpy()),
                    "tanh_mean": enc(squashed.squeeze(0).numpy()),
                    "action_scale": enc(self.actor.action_scale.numpy()),
                    "action_bias": enc(self.actor.action_bias.numpy()),
                    "scaled_action": enc(scaled.squeeze(0).numpy()),
                    "agent_action": enc(
                        deterministic.squeeze(0).numpy().astype(np.float32)),
                    "clipped_action": enc(np.clip(
                        deterministic.squeeze(0).numpy().astype(np.float32),
                        self.action_low, self.action_high)),
                })
            else:
                n = self.actor.net
                h0 = n[0](x)
                h1 = n[1](h0)
                h2 = n[2](h1)
                h3 = n[3](h2)
                pre_tanh = n[4](h3)
                squashed = torch.tanh(pre_tanh)
                scaled = squashed * self.actor.action_scale + self.actor.action_bias
                forward = self.actor(x)
                agent = forward.squeeze(0).numpy().astype(np.float32)
                out.update({
                    "layer0_linear": enc(h0.squeeze(0).numpy()),
                    "layer1_relu": enc(h1.squeeze(0).numpy()),
                    "layer2_linear": enc(h2.squeeze(0).numpy()),
                    "layer3_relu": enc(h3.squeeze(0).numpy()),
                    "layer4_linear": enc(pre_tanh.squeeze(0).numpy()),
                    "pre_tanh": enc(pre_tanh.squeeze(0).numpy()),
                    "tanh": enc(squashed.squeeze(0).numpy()),
                    "action_scale": enc(self.actor.action_scale.numpy()),
                    "action_bias": enc(self.actor.action_bias.numpy()),
                    "scaled_action": enc(scaled.squeeze(0).numpy()),
                    "agent_action": enc(agent),
                    "clipped_action": enc(np.clip(agent, self.action_low,
                                                  self.action_high)),
                })
        return out


def main():
    claim_protocol_channel()
    weights_dir = os.environ.get("DDM_TORCH_WEIGHTS", "/tmp/ddm_torch_weights")
    os.makedirs(weights_dir, exist_ok=True)
    sessions = {}

    def get(name):
        if name not in sessions:
            sessions[name] = PolicySession(name, weights_dir)
        return sessions[name]

    handlers = {
        "ping": lambda p: {"pong": True, "torch": torch.__version__},
        "init": lambda p: get(p["policy"]).meta(),
        "infer": lambda p: get(p["policy"]).infer(p["obs"]),
        "infer_batch": lambda p: {
            "rows": [get(p["policy"]).infer(o) for o in p["obs"]]},
    }

    sys.stderr.write("torch_policy_server ready\n")
    sys.stderr.flush()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except Exception as e:
            respond({"ok": False, "error": "bad json: %s" % e})
            continue
        if msg.get("cmd") == "quit":
            respond({"ok": True, "result": {"bye": True}})
            return
        handler = handlers.get(msg.get("cmd"))
        if handler is None:
            respond({"ok": False, "error": "unknown cmd %s" % msg.get("cmd")})
            continue
        try:
            respond({"ok": True, "result": handler(msg)})
        except Exception:
            respond({"ok": False, "error": traceback.format_exc()})


if __name__ == "__main__":
    main()
