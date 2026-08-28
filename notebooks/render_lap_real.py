"""Render the recorded DORA lap with gym-duckietown's own top-down view.

Each state was produced by the Julia model and exported with the package's
validated `world_to_ref` encoder. This script pushes them into the real
simulator through the SAME `import_state` the FJ5 reference server uses, so
the pictures are the reference implementation's, not a redrawing of it.

Nothing is re-simulated: the trajectory is replayed, frame by frame.

    conda activate ddm-ref
    python notebooks/render_lap_real.py
"""
import json
import os
import sys

DUCK = os.path.expanduser("~/aivnv/duckduck")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, DUCK)
JLREPO = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(JLREPO, "tools", "parity"))
os.chdir(DUCK)

import logging
logging.disable(logging.CRITICAL)

import numpy as np
import yaml
from PIL import Image, ImageDraw

from reference_server import Session

payload = json.load(open(os.path.join(HERE, "lap_states.json")))
states = payload["states"]
print("%d states, outcome %s, cost %.2f (model %.2f)" %
      (len(states), payload["outcome"], payload["cost"],
       payload["cost_model"]))

# Session.init takes the POLICY NAME and resolves the path itself
session = Session()
session.init("q_learning", 1, "discrete", None)
# reset before rendering: the simulator fills in per-tile render attributes
# (tile["color"]) during reset, and _render_img reads them
session.reset(1)
print("reference simulator initialised")

sim = session.mdp
while not hasattr(sim, "_render_img") and hasattr(sim, "env"):
    sim = sim.env
print("renderer:", type(sim).__name__)

OUT = os.path.join(HERE, "bev_frames")
os.makedirs(OUT, exist_ok=True)
for f in os.listdir(OUT):
    f.endswith(".png") and os.remove(os.path.join(OUT, f))

W, H = 800, 600
n = len(states)
for rec in states:
    i = rec["i"]
    session.import_state(rec["state"])
    img = sim._render_img(W, H, sim.multi_fbo_human, sim.final_fbo_human,
                          sim.img_array_human, top_down=True)
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    prog = rec["progress"]
    d.rectangle([0, H - 26, W, H], fill=(20, 20, 20))
    d.text((8, H - 20),
           "DORA lap of small_loop   decision %3d/%d   lap %d/8   %s"
           % (i - 1, n - 1, prog, "LAP COMPLETE" if prog >= 8 else ""),
           fill=(240, 240, 240))
    # progress bar
    d.rectangle([0, H - 30, int(W * prog / 8), H - 26], fill=(230, 200, 40))
    im.save(os.path.join(OUT, "f%04d.png" % i))
    if i % 20 == 0:
        print("  rendered %d / %d" % (i, n))

print("wrote %d frames to %s" % (n, os.path.relpath(OUT, DUCK)))
