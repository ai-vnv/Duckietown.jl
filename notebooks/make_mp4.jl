using FFMPEG

d = joinpath(@__DIR__, "bev_frames")
out = joinpath(@__DIR__, "dora_lap_real_bev.mp4")
pat = joinpath(d, "f%04d.png")
n = count(f -> endswith(f, ".png"), readdir(d))

FFMPEG.exe(`-y -loglevel error -framerate 12 -start_number 1 -i $pat
            -c:v libx264 -pix_fmt yuv420p -crf 20 $out`)

println("frames: ", n)
println("mp4   : ", round(filesize(out) / 1024; digits = 1), " kB  -> ",
        basename(out))
