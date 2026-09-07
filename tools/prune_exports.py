#!/usr/bin/env python3
"""Remove curated generic/internal names from the export statements of
src/Duckietown.jl (registry review: 301 exports, some very generic, five
underscore-prefixed). The names stay callable as Duckietown.<name>."""
import re
import pathlib

PRUNE = {
    # underscore-prefixed internals (reviewer's list)
    "_collision", "_drivable_pos", "_get_tile", "_inconvenient_spawn",
    "_valid_pose",
    # generic lowercase (reviewer's list + same class)
    "act", "decide", "forward", "measure", "branch", "update!",
    "intersects", "intersects_single_obj", "random_sample", "exact",
    "worst", "outcome", "digitize", "wrap_text", "n_missing", "is_frozen",
    "model_time", "grid_layout", "select_episode", "root_children",
    "panel_ids", "series_in", "hold_progress",
    # generic uppercase enum members / constants (reviewer named NONE,
    # STRAIGHT, READY; the rest are the same collision class)
    "NONE", "STRAIGHT", "READY", "NOT_READY", "NEEDS_REFACTOR", "GOAL",
    "REWARD", "COMPUTE", "FLAG", "ABSENT", "PERSISTED", "LOGGED",
    "CUMULATIVE", "INSTANTANEOUS", "TIMEOUT", "OFFROAD", "IN_PROGRESS",
    "MAIN_FIGURE", "SUPPLEMENTARY", "AGGREGATE_ONLY", "NAVIGATION",
}

p = pathlib.Path("src/Duckietown.jl")
lines = p.read_text().split("\n")
out = []
in_export = False
removed = 0
for line in lines:
    stripped = line.strip()
    starts = stripped.startswith("export ")
    if starts or in_export:
        for name in PRUNE:
            pat = r"(?<![\w!.])" + re.escape(name) + r"(?![\w!])"
            new, k = re.subn(pat, "\x00", line)
            if k:
                line = new
                removed += k
        # drop the placeholders and tidy commas/space
        line = re.sub(r"\x00,?\s*", "", line).rstrip()
        line = re.sub(r",\s*,", ",", line)
        line = re.sub(r",\s*$", ",", line)
        in_export = line.rstrip().endswith(",")
        # a line reduced to nothing (or a bare "export") is dropped
        if line.strip() in ("", "export", "export,"):
            continue
        line = line.rstrip()
        if line.endswith("export"):
            continue
    out.append(line)

text = "\n".join(out)
# heal statements whose final line lost its last name: "...,\n\n" endings
text = re.sub(r",\n(\n|(?=[^ ]))", r"\n\1", text)
p.write_text(text)
print(f"removed {removed} occurrences of {len(PRUNE)} names")
