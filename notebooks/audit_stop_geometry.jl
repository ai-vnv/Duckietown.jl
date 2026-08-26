# Why does the stop sign never become a candidate?
#
# `next_stop_candidate` accepts a sign only if all FOUR hold:
#     the sign faces the ego:  dot(sign_facing, forward) <= -stop_orientation_cos
#     ahead > 0,  |lateral| <= stop_lateral_limit,  distance <= stop_max_distance
# Rather than guess which one fails, measure all four on every frame of the
# lap. A weight that is switched on but whose subsystem never fires changes
# nothing, and that has to be reported, not assumed away.

using DuckietownDecisionModels, LinearAlgebra, Serialization, Printf

const CFG = scenario_config(:stop_and_duck_safe)
const SCFG = DuckietownMDP(CFG; action_space = :discrete).transition.state_cfg
const D = deserialize(joinpath(@__DIR__, "lap_frames_safe.jls"))

@printf("stop_lateral_limit %.3f   stop_max_distance %.3f   sign_to_line_offset %.3f\n",
        SCFG.stop_lateral_limit, SCFG.stop_max_distance, SCFG.sign_to_line_offset)

function probe()
    best = (ahead = -Inf, lat = Inf, dist = Inf, i = 0, facing = 0.0)
    n_face = 0; n_ahead = 0; n_lat = 0; n_range = 0
    faces = Float64[]
    for (i, f) in enumerate(D.frames)
        w = f[1]
        forward, right = lane_frame_tabular(w)
        pos = collect(w.ego.pos)
        for sg in w.stop_signs
            facing = dot(collect(heading_vec(sg.angle)), forward)
            push!(faces, facing)
            ok_face = facing <= -SCFG.stop_orientation_cos
            rel = collect(sg.pos) .- pos
            ahead = dot(rel, forward)
            lat = abs(dot(rel, right))
            dist = max(0.0, ahead - SCFG.sign_to_line_offset)
            ok_face && (n_face += 1)
            (ok_face && ahead > 0.0) && (n_ahead += 1)
            (ok_face && ahead > 0.0 && lat <= SCFG.stop_lateral_limit) && (n_lat += 1)
            (ok_face && ahead > 0.0 && lat <= SCFG.stop_lateral_limit &&
                dist <= SCFG.stop_max_distance) && (n_range += 1)
            if lat < best.lat
                best = (ahead = ahead, lat = lat, dist = dist, i = i - 1,
                        facing = facing)
            end
        end
    end
    n = length(D.frames)
    @printf("\nof %d frames, filters applied cumulatively:\n", n)
    @printf("  sign faces the ego                   : %d\n", n_face)
    @printf("  ... and is ahead of the ego          : %d\n", n_ahead)
    @printf("  ... and within lateral limit %.2f m   : %d\n",
            SCFG.stop_lateral_limit, n_lat)
    @printf("  ... and within max distance %.2f m    : %d  <- candidates\n",
            SCFG.stop_max_distance, n_range)
    @printf("\nfacing dot over the lap: min %+.3f  max %+.3f  (needs <= %+.3f)\n",
            minimum(faces), maximum(faces), -SCFG.stop_orientation_cos)
    @printf("closest the sign came to the corridor centre line:\n")
    @printf("  decision %d: ahead %+.3f m, lateral %.3f m, dist %.3f m, facing %+.3f\n",
            best.i, best.ahead, best.lat, best.dist, best.facing)
end

probe()

println("""
Read the counts top to bottom: whichever line drops to zero is the test that
rejects the sign. Whatever the cause, `d_stop` is nothing on every frame, and
both stop terms are gated on `d_stop !== nothing`, so switching on
stop_approach_* cannot change behaviour on this map.""")
