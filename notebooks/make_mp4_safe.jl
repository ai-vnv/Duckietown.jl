# Assemble the two-panel lap at REAL TIME.
#
# The recording now holds every PHYSICS SUBSTEP (frame_skip = 6 ticks per
# decision at 30 Hz), each replayed exactly and asserted bit-identical to
# `simulate_decision`'s outcome — so real time is simply the simulator's own
# 30 fps, and the motion is as smooth as the physics itself. The earlier
# 5 fps per-decision cut is why the video looked like a slideshow.

using FFMPEG, JSON3, Printf

const SIM_FPS = 30.0
const FPS = SIM_FPS

d = joinpath(@__DIR__, "safe_frames")
out = joinpath(@__DIR__, "dora_lap_safe_front_bev.mp4")
pat = joinpath(d, "f%04d.png")
n = count(f -> endswith(f, ".png"), readdir(d))

FFMPEG.exe(`-y -loglevel error -framerate $FPS -start_number 1 -i $pat
            -c:v libx264 -pix_fmt yuv420p -crf 20 $out`)

# summary from the JSON export — the .jls holds package types and would force
# loading the whole package just to print three numbers
D = JSON3.read(read(joinpath(@__DIR__, "lap_states_safe.json")))
@printf("frames %d   %.1f fps   %.1f s of video = %.1f s of model time\n",
        n, FPS, n / FPS, (n - 1) / SIM_FPS)
@printf("mp4 %.1f kB -> %s\n", filesize(out) / 1024, basename(out))
@printf("lap %s, cost %.2f (model %.2f)\n", D.outcome, D.cost, D.cost_model)
