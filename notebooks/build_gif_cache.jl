# Build the per-decision GIF cache for both recorded laps, so `pluto_gifs/`
# can be committed and the Pluto notebook works on a fresh clone without the
# rendered frame PNGs. Uses the same segmentation and encoder as the notebook.

using JSON3, FFMPEG, Printf

const LAPS = Dict(
    "safe" => (json = "lap_states_safe.json", frames = "safe_frames", nring = 8),
    "zigzag" => (json = "zigzag_lap.json", frames = "zigzag_frames", nring = 26),
)

function tile_segments(states, nring; per_decision = true)
    keep = if per_decision
        last_of_dec = Dict{Int,Int}()
        for s in states
            last_of_dec[s.dec] = max(get(last_of_dec, s.dec, 0), s.i)
        end
        Set(values(last_of_dec))
    else
        Set(s.i for s in states)
    end
    segs = [Int[] for _ in 1:nring]
    for s in states
        s.i in keep || continue
        p = clamp(s.progress, 0, nring - 1)
        s.progress >= nring && (p = nring - 1)
        push!(segs[p + 1], s.i)
    end
    return segs
end

function build_gif(framesdir, idxs, out; width = 560, dt = 0.1)
    isfile(out) && return out
    mkpath(dirname(out))
    list = tempname() * ".txt"
    open(list, "w") do io
        for i in idxs
            println(io, "file '", joinpath(framesdir, "f" * lpad(i, 4, '0') * ".png"), "'")
            println(io, "duration ", dt)
        end
    end
    FFMPEG.exe(`-y -loglevel error -f concat -safe 0 -i $list
        -vf "scale=$(width):-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse"
        $out`)
    rm(list; force = true)
    return out
end

gifdir = joinpath(@__DIR__, "pluto_gifs")
total = 0.0
for (name, lap) in LAPS
    states = JSON3.read(read(joinpath(@__DIR__, lap.json))).states
    segs = tile_segments(states, lap.nring)
    for (t, seg) in enumerate(segs)
        out = build_gif(joinpath(@__DIR__, lap.frames), seg,
            joinpath(gifdir, "$(name)_d_tile$(lpad(t, 2, '0')).gif"))
        global total += filesize(out)
    end
    @printf("%-8s %2d GIFs\n", name, lap.nring)
end
@printf("cache total %.1f MB in %s\n", total / 1e6, gifdir)
