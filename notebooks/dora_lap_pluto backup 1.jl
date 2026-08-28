### A Pluto.jl notebook ###
# v1.0.3

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ a1b20002-0002-4002-8002-000000000002
begin
    using PlutoUI
    using JSON3
    using FFMPEG
    using Base64
end

# ╔═╡ a1b20001-0001-4001-8001-000000000001
md"""
# DORA di Duckietown.jl — playback per tile

Notebook ini **memutar rekaman** lap yang sudah dieksekusi dan divalidasi —
tidak menjalankan ulang solver maupun simulator. Setiap frame adalah satu
substep fisika (30 Hz) yang direplay eksak dan di-assert identik-bit dengan
`simulate_decision` saat perekaman; gambarnya dirender oleh gym-duckietown
asli lewat protokol referensi FJ5.

**Resep DORA** (dijelaskan lengkap di percakapan/dokumen eksperimen):

| DORA butuh | Diisi dengan |
|---|---|
| kernel transisi diketahui | determinisme terukur → `Deterministic` eksak |
| state berhingga & hashable | `key = (progress ring, discretize(d, φ, v, …))` |
| goal / crash | penghitung ring monoton + `isterminal` |
| biaya positif | `max(c_min, 1 − reward)` — bentuk terdokumentasi |
| percabangan bermakna | macro action K = 8 decision (1.6 s) |
| tahan aliasing | receding horizon: replan tiap macro action |
"""

# ╔═╡ a1b20003-0003-4003-8003-000000000003
md"""
## Pilih lap rekaman
"""

# ╔═╡ a1b20004-0004-4004-8004-000000000004
@bind lapchoice Select([
    "safe" => "small_loop — :stop_and_duck_safe (yield bebek + full stop di rambu)",
    "zigzag" => "zigzag_dists — lap 26 tile, kelokan kiri-kanan",
])

# ╔═╡ a1b20005-0005-4005-8005-000000000005
LAPS = Dict(
    "safe" => (json = "lap_states_safe.json", frames = "safe_frames",
               nring = 8, title = "small_loop :stop_and_duck_safe"),
    "zigzag" => (json = "zigzag_lap.json", frames = "zigzag_frames",
                 nring = 26, title = "zigzag_dists lane following"),
)

# ╔═╡ a1b20006-0006-4006-8006-000000000006
begin
    LAP = LAPS[lapchoice]
    jsonpath = joinpath(@__DIR__, LAP.json)
    framesdir = joinpath(@__DIR__, LAP.frames)
    if !isfile(jsonpath) || !isdir(framesdir)
        error("""artefak rekaman tidak ditemukan:
                 $(jsonpath)
                 $(framesdir)
              Jalankan dulu skrip eksperimen + render yang menghasilkannya
              (dora_lap_safe.jl / dora_zigzag.jl lalu render_*.py).""")
    end
    payload = JSON3.read(read(jsonpath))
    states = payload.states
    md"""
    **$(LAP.title)** — hasil: `$(payload.outcome)`,
    biaya eksekusi $(round(payload.cost; digits = 2))
    vs rencana awal $(round(payload.cost_model; digits = 2)),
    $(length(states)) substep fisika = $(round((length(states) - 1) / 30; digits = 1)) s waktu model.
    """
end

# ╔═╡ a1b20007-0007-4007-8007-000000000007
md"""
## GIF per tile

Lap dipotong menurut penghitung kemajuan ring: segmen ke-``t`` berisi semua
substep selama `progress == t − 1` (perjalanan menuju tile ring ke-``t``).
GIF dibangun sekali dari frame PNG hasil render asli, lalu di-cache di
`pluto_gifs/`.

**Mode frame** — per decision memakai hanya substep terakhir tiap decision
(hasil keputusannya): 6× lebih sedikit frame, jadi jauh lebih cepat dibangun.
Centang untuk mode halus 30 Hz (semua substep):
"""

# ╔═╡ a1b20010-0010-4010-8010-000000000010
@bind smooth CheckBox(default = false)

# ╔═╡ a1b20008-0008-4008-8008-000000000008
"""Indeks frame per segmen tile: segmen t = substep dengan progress t-1;
frame penutup lap ikut segmen terakhir. Bila `per_decision`, hanya substep
TERAKHIR tiap decision yang dipakai (frame ke-1 tetap ikut sebagai awal)."""
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
        # frame terakhir (progress == nring) menutup segmen terakhir
        s.progress >= nring && (p = nring - 1)
        push!(segs[p + 1], s.i)
    end
    return segs
end

# ╔═╡ a1b20009-0009-4009-8009-000000000009
"""GIF dari daftar frame PNG (concat demuxer + palettegen). `dt` adalah durasi
tampil per frame; default di pemanggil menjaga playback 2x kecepatan nyata
untuk kedua mode."""
function build_gif(framesdir, idxs, out; width = 560, dt = 0.0667)
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

# ╔═╡ a1b2000a-000a-400a-800a-00000000000a
begin
    segs = tile_segments(states, LAP.nring; per_decision = !smooth)
    gifdir = joinpath(@__DIR__, "pluto_gifs")
    mode = smooth ? "s" : "d"
    # 2x kecepatan nyata di kedua mode: substep nyata 1/30 s -> tampil 1/15 s;
    # decision nyata 0.2 s -> tampil 0.1 s
    dt = smooth ? 0.0667 : 0.1
    gifs = [build_gif(framesdir, seg,
                joinpath(gifdir, "$(lapchoice)_$(mode)_tile$(lpad(t, 2, '0')).gif");
                dt = dt)
            for (t, seg) in enumerate(segs)]
    md"**$(length(gifs)) GIF** ($(smooth ? "per substep, halus" : "per decision, cepat")) siap di `pluto_gifs/` (di-cache; hapus foldernya untuk membangun ulang)."
end

# ╔═╡ a1b2000b-000b-400b-800b-00000000000b
md"""
## Jelajah per tile
"""

# ╔═╡ a1b2000c-000c-400c-800c-00000000000c
@bind tile Slider(1:length(gifs); default = 1, show_value = true)

# ╔═╡ a1b2000d-000d-400d-800d-00000000000d
"""Ringkasan terukur dari satu segmen: kecepatan, dan (bila terekam) event
stop/bebek — angka, bukan kesan."""
function segment_caption(states, idxs)
    ss = [s for s in states if s.i in Set(idxs)]
    vs = [s.v for s in ss]
    parts = ["decision $(minimum(s.dec for s in ss))–$(maximum(s.dec for s in ss))",
             "v $(round(minimum(vs); digits = 3))–$(round(maximum(vs); digits = 3)) m/s"]
    if haskey(first(ss), :event)
        evs = unique(s.event for s in ss if s.event != "")
        isempty(evs) || push!(parts, "event: " * join(evs, ", "))
        ducks = unique(s.duck for s in ss if s.duck != "NONE")
        isempty(ducks) || push!(parts, "bebek: " * join(ducks, ", "))
        any(s.sigma for s in ss) && push!(parts, "sigma_stop aktif")
    end
    return join(parts, " · ")
end

# ╔═╡ a1b2000e-000e-400e-800e-00000000000e
begin
    cap = segment_caption(states, segs[tile])
    b64 = base64encode(read(gifs[tile]))
    HTML("""
    <div style="max-width:600px">
      <h4 style="margin:0 0 4px 0">Tile $(tile) / $(LAP.nring) — $(LAP.title)</h4>
      <p style="margin:0 0 8px 0;color:#888;font-size:0.9em">$(cap)</p>
      <img src="data:image/gif;base64,$(b64)" width="560"
           style="border:1px solid #444;border-radius:6px"/>
      <p style="margin:6px 0 0 0;color:#888;font-size:0.8em">
        playback 2× · $(smooth ? "per substep (30 Hz)" : "1 frame per decision (0.2 s)") ·
        panel kiri: kamera depan ego · panel kanan: BEV ·
        keduanya dari renderer gym-duckietown asli</p>
    </div>
    """)
end

# ╔═╡ a1b2000f-000f-400f-800f-00000000000f
md"""
---
### Catatan kejujuran artefak

- Frame **tidak pernah** dihasilkan ulang oleh notebook ini — ia hanya
  memotong dan mengemas rekaman yang assertion-nya lulus saat eksperimen.
- `progress` adalah penghitung ring monoton dari eksperimen, bukan hasil
  hitung ulang di sini.
- Kalau artefaknya belum ada, notebook berhenti dengan instruksi — ia tidak
  akan diam-diam menjalankan solver.
"""

# ╔═╡ Cell order:
# ╟─a1b20001-0001-4001-8001-000000000001
# ╠═a1b20002-0002-4002-8002-000000000002
# ╟─a1b20003-0003-4003-8003-000000000003
# ╟─a1b20004-0004-4004-8004-000000000004
# ╠═a1b20005-0005-4005-8005-000000000005
# ╟─a1b20006-0006-4006-8006-000000000006
# ╟─a1b20007-0007-4007-8007-000000000007
# ╟─a1b20010-0010-4010-8010-000000000010
# ╠═a1b20008-0008-4008-8008-000000000008
# ╠═a1b20009-0009-4009-8009-000000000009
# ╟─a1b2000a-000a-400a-800a-00000000000a
# ╟─a1b2000b-000b-400b-800b-00000000000b
# ╟─a1b2000c-000c-400c-800c-00000000000c
# ╠═a1b2000d-000d-400d-800d-00000000000d
# ╟─a1b2000e-000e-400e-800e-00000000000e
# ╟─a1b2000f-000f-400f-800f-00000000000f
