"""Render the recorded zigzag lap: ego front camera beside the top-down view.

Frames come from gym-duckietown's own renderer for the real zigzag_dists map.
Only the ego pose is replayed per recorded physics substep — this map has no
ducks and no stop sign, so there is no other dynamic state to import. Nothing
is re-simulated.

    conda activate ddm-ref
    python notebooks/render_zigzag.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

import logging
logging.disable(logging.CRITICAL)

import numpy as np
from PIL import Image, ImageDraw

from gym_duckietown.simulator import Simulator

payload = json.load(open(os.path.join(HERE, "zigzag_lap.json")))
states = payload["states"]
nring = payload["nring"]
print("%d substeps, outcome %s, cost %.2f (first plan %.2f)" %
      (len(states), payload["outcome"], payload["cost"],
       payload["cost_model"]))

env = Simulator(map_name="zigzag_dists", domain_rand=False, seed=1)
env.reset()
print("renderer:", type(env).__name__)

OUT = os.path.join(HERE, "zigzag_frames")
os.makedirs(OUT, exist_ok=True)
for f in os.listdir(OUT):
    f.endswith(".png") and os.remove(os.path.join(OUT, f))

# render at the FBO's allocated size, downscale after (wrong sizes read the
# buffer back with the wrong stride -> interlaced garbage)
RW, RH = 800, 600
W, H = 640, 480
BAR = 46


def shot(top_down):
    img = env._render_img(RW, RH, env.multi_fbo_human, env.final_fbo_human,
                          env.img_array_human, top_down=top_down)
    return Image.fromarray(img).resize((W, H), Image.LANCZOS)


n = len(states)
ndec = states[-1]["dec"]
for rec in states:
    i = rec["i"]
    env.cur_pos = np.array(rec["pos"], dtype=float)
    env.cur_angle = float(rec["angle"])
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
           "DORA  zigzag_dists  lane following   decision %3d/%d   lap %d/%d"
           % (rec["dec"], ndec, prog, nring), fill=(240, 240, 240))
    d.text((10, H + 26), "speed %.3f m/s" % rec["v"], fill=(200, 200, 200))
    d.text((2 * W - 200, H + 26),
           "LAP COMPLETE" if prog >= nring else "30 fps = real time",
           fill=(230, 200, 40) if prog >= nring else (150, 150, 150))
    d.rectangle([0, H + BAR - 4, int(2 * W * prog / nring), H + BAR],
                fill=(230, 200, 40))

    im.save(os.path.join(OUT, "f%04d.png" % i))
    if i % 200 == 0:
        print("  rendered %d / %d" % (i, n))

print("wrote %d frames to %s" % (n, OUT))
