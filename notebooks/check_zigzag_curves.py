"""Parity check: Julia zigzag tile curves vs the reference simulator's own.

    conda activate ddm-ref
    python notebooks/check_zigzag_curves.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

import logging
logging.disable(logging.CRITICAL)

import numpy as np
from gym_duckietown.simulator import Simulator

jl = json.load(open(os.path.join(HERE, "zigzag_curves_julia.json")))
jtiles = {(t["i"], t["j"]): t for t in jl["tiles"]}

env = Simulator(map_name="zigzag_dists", domain_rand=False, seed=1)
env.reset()

worst = 0.0
worst_at = None
n = 0
for j in range(env.grid_height):
    for i in range(env.grid_width):
        tile = env._get_tile(i, j)
        if tile is None or not tile["drivable"]:
            continue
        n += 1
        key = (i, j)
        if key not in jtiles:
            print("MISSING in julia: tile %s kind %s" % (key, tile["kind"]))
            continue
        ref = np.array(tile["curves"], dtype=float)      # (n,4,3)
        jt = np.array(jtiles[key]["curves"], dtype=float)
        if ref.shape != jt.shape:
            print("SHAPE %s: ref %s julia %s" % (key, ref.shape, jt.shape))
            continue
        # curve ORDER may differ; match each ref curve to its nearest julia one
        d = 0.0
        for rc in ref:
            best = min(np.abs(rc - jc).max() for jc in jt)
            d = max(d, best)
        if d > worst:
            worst, worst_at = d, (key, tile["kind"])
        if d > 1e-9:
            print("DIFF %-8s %-12s max|delta| = %.6f" %
                  (key, tile["kind"], d))
print("\nchecked %d drivable tiles" % n)
print("worst max|delta| = %.9f at %s" % (worst, worst_at))
