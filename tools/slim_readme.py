#!/usr/bin/env python3
"""Registry review: the README is on the long side. Replace the detail
sections (The model ... Known limitations) with pointers to the Documenter
site, where docs/src/guide.md now carries them in full."""
import pathlib

p = pathlib.Path("README.md")
t = p.read_text()
start = t.index("## The model")
end = t.index("## Acknowledgments")

replacement = """## Documentation

Everything below this point used to live here and now has a page of its own
on the [documentation site](https://ai-vnv.github.io/Duckietown.jl/dev/):

- [Guide](https://ai-vnv.github.io/Duckietown.jl/dev/guide/) — the model and
  its two privileged projections, scenarios, drawing (including the native
  lookalike renderer), running planners, the DORA case study, reproducing
  the recorded experiments, and the known limitations.
- [How it was built](https://ai-vnv.github.io/Duckietown.jl/dev/building/) —
  what every file is, why it exists, and where each claim's evidence lives.
- [Validation records](https://ai-vnv.github.io/Duckietown.jl/dev/validation/) —
  one status document per gate (FJ2–FJ10): what was measured, what
  deviated, what was left undone.
- [API](https://ai-vnv.github.io/Duckietown.jl/dev/api/model/) — per-layer
  reference.

The recorded media (rendered laps, animations, diagnostics, publication
figures) live in the
[companion repository](https://github.com/ai-vnv/Duckietown-artifacts), so
`Pkg.add` downloads source, tests and fixtures only.

---

"""
t = t[:start] + replacement + t[end:]

old_bullet = (
    "- The two GIFs in `docs/assets/` are renders of Duckietown environments and\n"
    "  carry attribution in their captions."
)
new_bullet = (
    "- The animated media (in the\n"
    "  [companion repository](https://github.com/ai-vnv/Duckietown-artifacts))\n"
    "  are renders of Duckietown environments and carry attribution in their\n"
    "  captions."
)
assert old_bullet in t
t = t.replace(old_bullet, new_bullet, 1)
p.write_text(t)
print("README:", len(t.splitlines()), "lines")
