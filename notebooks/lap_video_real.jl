# The DORA lap, rendered by gym-duckietown itself.
#
# The previous video was my own Makie approximation of a Duckietown look. This
# one pushes each recorded world state into the real simulator through the FJ5
# reference backend and asks IT to draw the top-down view — real textures, the
# real Duckiebot mesh, the real duckie.
#
# The states come from the recorded lap; nothing is re-simulated. The
# reference backend is used purely as a renderer here, and `set_state` is the
# same validated path FJ5 uses for parity.

using DuckietownDecisionModels
using POMDPs, Serialization, Printf, Random

const ROOT = pkgdir(DuckietownDecisionModels)
const CFG = joinpath(ROOT, "..", "duckduck", "policies", "q_learning",
                     "training_config.yaml")
const OUT = joinpath(@__DIR__, "bev_frames")
mkpath(OUT)

D = deserialize(joinpath(@__DIR__, "lap_frames.jls"))
frames = D.frames
@printf("%d recorded states, outcome %s\n", length(frames), D.outcome)

reference_backend_available() ||
    error("the ddm-ref reference environment is not available")

b = ProcessReferenceBackend()
try
    ref_reset!(b; config = CFG, seed = 1, action_space = "discrete")
    println("reference simulator up")

    ok = 0
    for (i, (s, prog)) in enumerate(frames)
        ref_set_state!(b, s)
        path = joinpath(OUT, @sprintf("f%04d.png", i))
        r = ref_call(b, Dict("cmd" => "render", "path" => path,
                             "width" => 800, "height" => 600))
        (r isa AbstractDict && get(r, "ok", true) === false) &&
            error("render failed at frame $i: $(get(r, "error", ""))")
        isfile(path) && (ok += 1)
        i % 20 == 0 && @printf("  rendered %d / %d\n", i, length(frames))
    end
    @printf("rendered %d / %d frames into %s\n", ok, length(frames),
            relpath(OUT, ROOT))
finally
    close(b)
end
