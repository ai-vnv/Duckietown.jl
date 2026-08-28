"""Render the safety-shaped DORA lap: ego front camera beside the top-down view.

Both panels come from gym-duckietown's own renderer via the FJ5 reference
server's `import_state`, so neither is a redrawing. Nothing is re-simulated —
the recorded physics substeps are replayed one per frame (30 fps = real time).

`import_state` moves the ego and the ducks but NOT the static stop sign, and
the Python session builds its map with the Python-side sign placement — which
`:stop_and_duck_safe` deliberately corrects (the source's sign shows its back
to the actual traffic). So the sign object is relocated once, from the pose
recorded in the exported states themselves.

    conda activate ddm-ref
    python notebooks/render_lap_safe.py
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
from PIL import Image, ImageDraw

from reference_server import Session, dec
from gym_duckietown.objects import generate_corners, generate_norm

payload = json.load(open(os.path.join(HERE, "lap_states_safe.json")))
states = payload["states"]
print("%d substeps, scenario %s, outcome %s, cost %.2f (first plan %.2f)" %
      (len(states), payload["scenario"], payload["outcome"],
       payload["cost"], payload["cost_model"]))

session = Session()
session.init("q_learning", 1, "discrete", None)
# reset before rendering: per-tile render attributes (tile["color"]) are filled
# in during reset and _render_img reads them
session.reset(1)

sim = session.mdp
while not hasattr(sim, "_render_img") and hasattr(sim, "env"):
    sim = sim.env
print("renderer:", type(sim).__name__)

# relocate the static sign to the recorded scenario pose
sign_rec = states[0]["state"]["signs"][0]
spos = np.array([dec(v) for v in sign_rec["pos"]], dtype=float)
sangle = dec(sign_rec["angle"])
signs = [o for o in sim.objects if "sign" in str(getattr(o, "kind", "")).lower()]
assert len(signs) == 1, [o.kind for o in sim.objects]
sg = signs[0]
print("sign: python map had pos %s y_rot %.1f" % (np.round(sg.pos, 3), sg.y_rot))
sg.pos = spos
sg.angle = sangle
sg.y_rot = np.rad2deg(sangle)
sg.obj_corners = generate_corners(sg.pos, sg.min_coords, sg.max_coords,
                                  sg.angle, sg.scale)
sg.obj_norm = generate_norm(sg.obj_corners)
print("sign: relocated to pos %s y_rot %.1f (recorded scenario pose)" %
      (np.round(sg.pos, 3), sg.y_rot))

OUT = os.path.join(HERE, "safe_frames")
os.makedirs(OUT, exist_ok=True)
for f in os.listdir(OUT):
    f.endswith(".png") and os.remove(os.path.join(OUT, f))

# Render at the size the human FBO and img_array were allocated for — asking
# _render_img for any other size reads the buffer back with the wrong stride
# and produces interlaced garbage. Downscale afterwards instead.
RW, RH = 800, 600
W, H = 640, 480          # per panel; composite is 1280 x 480 + a caption strip
BAR = 46


def shot(top_down):
    img = sim._render_img(RW, RH, sim.multi_fbo_human, sim.final_fbo_human,
                          sim.img_array_human, top_down=top_down)
    return Image.fromarray(img).resize((W, H), Image.LANCZOS)


def status(rec):
    """One line of live model state, colour-coded. Numbers, not impressions."""
    if rec["event"] == "FULL_STOP":
        return "FULL STOP AT SIGN", (120, 220, 255)
    if rec["event"] == "STOP_VIOLATION":
        return "STOP VIOLATION", (255, 90, 90)
    if rec["event"] == "PASSED_STOP":
        return "PASSED SIGN (stop satisfied)", (140, 230, 140)
    if rec["sigma"]:
        return "STOPPED - cleared to proceed", (120, 220, 255)
    if rec["d_stop"] is not None:
        return "APPROACHING STOP SIGN  d=%.2f m" % rec["d_stop"], (240, 200, 90)
    if rec["pedestrian"] < 0:
        return "UNSAFE PASS", (255, 90, 90)
    if rec["duck"].startswith("CROSSING") and rec["v"] < 0.05:
        return "YIELDING TO DUCK", (120, 220, 255)
    if rec["duck"] != "NONE":
        return "duck in view: %s" % rec["duck"], (200, 200, 200)
    return "", (200, 200, 200)


n = len(states)
ndec = states[-1]["dec"]
for rec in states:
    i = rec["i"]
    session.import_state(rec["state"])
    front = shot(False)
    bev = shot(True)

    im = Image.new("RGB", (2 * W, H + BAR), (18, 18, 18))
    im.paste(front, (0, 0))
    im.paste(bev, (W, 0))
    d = ImageDraw.Draw(im)
    d.line([(W, 0), (W, H)], fill=(60, 60, 60), width=2)
    d.text((10, 8), "EGO FRONT CAMERA", fill=(245, 245, 245))
    d.text((W + 10, 8), "TOP-DOWN (BEV)", fill=(245, 245, 245))

    prog = rec["progress"]
    d.text((10, H + 8),
           "DORA  small_loop  :stop_and_duck_safe   decision %3d/%d   lap %d/8"
           % (rec["dec"], ndec, prog), fill=(240, 240, 240))
    msg, colour = status(rec)
    d.text((10, H + 26), "speed %.3f m/s    %s" % (rec["v"], msg), fill=colour)
    d.text((2 * W - 200, H + 26),
           "LAP COMPLETE" if prog >= 8 else "30 fps = real time",
           fill=(230, 200, 40) if prog >= 8 else (150, 150, 150))
    d.rectangle([0, H + BAR - 4, int(2 * W * prog / 8), H + BAR],
                fill=(230, 200, 40))

    im.save(os.path.join(OUT, "f%04d.png" % i))
    if i % 100 == 0:
        print("  rendered %d / %d" % (i, n))

print("wrote %d frames to %s" % (n, os.path.relpath(OUT, DUCK)))
