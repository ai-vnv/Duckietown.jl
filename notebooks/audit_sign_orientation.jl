# Is the stop sign facing the traffic, or its back?
#
# The Python source injects the sign at YAML (1.20, 2.10) with rotate = 180
# and its docstring claims that faces "vehicles travelling east". Measure,
# on the recorded lap:
#   - which way the ego actually travels when it passes the sign's tile
#   - the sign's heading vector
#   - what rotate WOULD satisfy the observer's facing test there
# so the fix is derived from data, not from another guess about conventions.

using DuckietownDecisionModels, LinearAlgebra, Serialization, Printf

const CFG = scenario_config(:stop_and_duck_safe)
const SCFG = DuckietownMDP(CFG; action_space = :discrete).transition.state_cfg
const D = deserialize(joinpath(@__DIR__, "lap_frames_safe.jls"))

sg = D.frames[1][1].stop_signs[1]
@printf("sign pos (%.3f, %.3f)   angle %.3f rad = %.1f deg\n",
        sg.pos[1], sg.pos[3], sg.angle, rad2deg(sg.angle))
hv = collect(heading_vec(sg.angle))
@printf("sign heading vector (%.3f, %.3f, %.3f)\n", hv[1], hv[2], hv[3])

# ego travel near the sign: use frames where the sign is within 1 m planar
println("\nego motion within 1.0 m of the sign:")
@printf("  %4s %8s %8s %10s %10s %9s\n",
        "dec", "x", "z", "fwd_x", "fwd_z", "facing")
for (i, f) in enumerate(D.frames)
    w = f[1]
    dist = hypot(sg.pos[1] - w.ego.pos[1], sg.pos[3] - w.ego.pos[3])
    dist > 1.0 && continue
    fwd, _ = lane_frame_tabular(w)
    @printf("  %4d %8.3f %8.3f %10.3f %10.3f %9.3f\n",
            i - 1, w.ego.pos[1], w.ego.pos[3], fwd[1], fwd[3],
            dot(hv, fwd))
end

# What rotate would pass the facing test at the closest approach?
# The observer needs dot(heading_vec(sign_angle), forward) <= -cos(45 deg).
best = argmin([hypot(sg.pos[1] - f[1].ego.pos[1], sg.pos[3] - f[1].ego.pos[3])
               for f in D.frames])
fwd, _ = lane_frame_tabular(D.frames[best][1])
println("\nfacing dot by candidate rotate at the closest-approach frame (dec ",
        best - 1, "):")
for rot in 0.0:45.0:315.0
    a = deg2rad(rot)
    v = collect(heading_vec(a))
    mark = dot(v, fwd) <= -SCFG.stop_orientation_cos ? "  <- passes" : ""
    @printf("  rotate %5.1f deg: dot %+.3f%s\n", rot, dot(v, fwd), mark)
end
