# Assemble the zigzag lap at real time: every recorded frame is one physics
# substep at the simulator's own 30 Hz.

using FFMPEG, JSON3, Printf

const FPS = 30.0

d = joinpath(@__DIR__, "zigzag_frames")
out = joinpath(@__DIR__, "dora_zigzag_front_bev.mp4")
pat = joinpath(d, "f%04d.png")
n = count(f -> endswith(f, ".png"), readdir(d))

FFMPEG.exe(`-y -loglevel error -framerate $FPS -start_number 1 -i $pat
            -c:v libx264 -pix_fmt yuv420p -crf 20 $out`)

D = JSON3.read(read(joinpath(@__DIR__, "zigzag_lap.json")))
@printf("frames %d   %.1f fps   %.1f s of video = %.1f s of model time\n",
        n, FPS, n / FPS, (n - 1) / FPS)
@printf("mp4 %.1f kB -> %s\n", filesize(out) / 1024, basename(out))
@printf("outcome %s, cost %.2f (first plan %.2f)\n",
        D.outcome, D.cost, D.cost_model)
