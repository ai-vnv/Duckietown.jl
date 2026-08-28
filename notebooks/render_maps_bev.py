"""Render a labelled BEV contact sheet of the built-in gym-duckietown maps,
so a target map for the next DORA experiment can be chosen by looking at the
real scenes rather than at YAML names.

    conda activate ddm-ref
    python notebooks/render_maps_bev.py
"""
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))

import logging
logging.disable(logging.CRITICAL)

import numpy as np
from PIL import Image, ImageDraw

from gym_duckietown.simulator import Simulator

# the classic set, ordered simple -> complex; a curated subset of the ~40
# YAML files so the sheet stays readable
MAPS = [
    "straight_road",
    "small_loop",
    "small_loop_cw",
    "loop_empty",
    "zigzag_dists",
    "loop_obstacles",
    "loop_pedestrians",
    "loop_dyn_duckiebots",
    "4way",
    "udem1",
    "ETHZ_autolab_technical_track",
    "TTIC_large_loop",
]

RW, RH = 800, 600
W, H = 400, 300
LABEL = 24

tiles = []
for name in MAPS:
    try:
        env = Simulator(map_name=name, domain_rand=False, draw_curve=False,
                        seed=1)
        env.reset()
        img = env._render_img(RW, RH, env.multi_fbo_human, env.final_fbo_human,
                              env.img_array_human, top_down=True)
        im = Image.fromarray(img).resize((W, H), Image.LANCZOS)
        note = "%dx%d tiles, %d objects" % (
            env.grid_width, env.grid_height, len(env.objects))
        env.window and env.window.close()
        tiles.append((name, note, im))
        print("ok  %-32s %s" % (name, note))
    except Exception as e:
        print("FAIL %-31s %s" % (name, e))
        traceback.print_exc(limit=1)

cols = 3
rows = (len(tiles) + cols - 1) // cols
sheet = Image.new("RGB", (cols * W, rows * (H + LABEL)), (15, 15, 15))
d = ImageDraw.Draw(sheet)
for k, (name, note, im) in enumerate(tiles):
    x = (k % cols) * W
    y = (k // cols) * (H + LABEL)
    sheet.paste(im, (x, y + LABEL))
    d.rectangle([x, y, x + W, y + LABEL], fill=(30, 30, 30))
    d.text((x + 6, y + 5), "%d. %s   (%s)" % (k + 1, name, note),
           fill=(240, 240, 240))

out = os.path.join(HERE, "maps_bev.png")
sheet.save(out)
print("wrote %s  (%d maps)" % (out, len(tiles)))
