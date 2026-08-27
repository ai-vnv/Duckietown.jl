# The README hero GIF: a built-in solver driving, drawn by the package's own
# native renderer — no Python anywhere in the loop.
#
# Solver setup is the FJ8 lap study's, replicated exactly
# (tools/fj8_lap_analysis.jl; results in artifacts/fj8/lap_completion.txt):
# MCTSSolver(n_iterations = 36, depth = 10, exploration_constant = 5.0) —
# about 1000 generative calls per decision — on the frozen q_learning
# evaluation config, where it completed >= 1 lap on 5/5 seeds with 0
# crashes. The episode protocol (seed list, per-episode rng, decision limit)
# is the study's own.
#
# Rendering streams STRAIGHT INTO THE GIF with `Makie.record`: ONE GL scene
# (two child scenes — ego camera and top-down — sharing the parent), built
# once; per frame only the ego camera and the two moving meshes (duckiebot,
# duck) are updated. The previous per-frame-scene pipeline created ~300 GL
# scenes and wedged WSLg. Every frame is one solver decision (0.2 s of model
# time) shown at 2x. `render_native` assets and camera constants; a
# lookalike, never parity evidence.

using DuckietownDecisionModels
using POMDPs, POMDPTools, MCTS
using GLMakie
using Random, Printf

const DDM = DuckietownDecisionModels
const QCFG = joinpath(homedir(), "aivnv", "duckduck", "policies",
    "q_learning", "training_config.yaml")
const CFG = load_config(QCFG)
const BASE = DuckietownMDP(QCFG; action_space = :discrete)
const TR = BASE.transition
const MAXDEC = BASE.transition.max_steps ÷ 6      # the study's tab_limit

tile_of(s) = get_grid_coords(s.map, collect(s.ego.pos))

const RING = let m = initial_map(CFG)
    t = collect(drivable_tiles(m))
    cx = sum(first.(t)) / length(t); cy = sum(last.(t)) / length(t)
    sort(t; by = p -> atan(p[2] - cy, p[1] - cx))
end
const RIX = Dict(t => i for (i, t) in enumerate(RING))
const NRING = length(RING)

function advance(prog, prev, new)
    new == prev && return prog
    a = get(RIX, prev, 0); b = get(RIX, new, 0)
    (a == 0 || b == 0) && return prog
    return mod(b - a, NRING) == 1 ? prog + 1 : prog
end

"""One episode under the study's exact protocol; every post-decision world
state recorded."""
function drive(seed)
    planner = solve(MCTSSolver(n_iterations = 36, depth = 10,
        exploration_constant = 5.0, rng = MersenneTwister(2026),
        reuse_tree = false), BASE)
    rng = MersenneTwister(seed)
    s = rand(MersenneTwister(seed), initialstate(BASE))
    states = [s]
    prog = 0
    tile = tile_of(s)
    for t in 1:MAXDEC
        a = action(planner, s)
        r = simulate_decision(TR, s, a, rng)
        s = r.sp
        push!(states, s)
        prog = advance(prog, tile, tile_of(s)); tile = tile_of(s)
        prog >= NRING && return (states, :lap, prog)
        (r.terminated || r.truncated) &&
            return (states, r.terminated ? :crash : :timeout, prog)
    end
    return (states, :horizon, prog)
end

function pick_lap_episode(seeds)
    for sd in seeds
        t0 = time()
        st, out, pg = drive(sd)
        @printf("seed %d -> %s after %d decisions (prog %d/%d) in %.0f s\n",
                sd, out, length(st) - 1, pg, NRING, time() - t0)
        out === :lap && return (st, sd)
    end
    error("no study seed completed a lap — investigate before making a GIF")
end

seeds = planning_seed_config().evaluation.episodes[1:5]
states, used_seed = pick_lap_episode(seeds)
println("using seed ", used_seed)

# --- ONE scene, two child views, streamed straight into the GIF -------------
const ASSETS = duckietown_assets_root()
const PW, PH = 380, 285

"""Static scene content (tiles + stop signs) into one child scene."""
function populate_static!(scene, nw)
    for (i, j, texpath, rot) in nw.tiles
        x0, x1 = i * nw.tile_size, (i + 1) * nw.tile_size
        z0, z1 = j * nw.tile_size, (j + 1) * nw.tile_size
        pts = Point3f[(x0, 0, z0), (x1, 0, z0), (x1, 0, z1), (x0, 0, z1)]
        uv0 = Vec2f[(0, 1), (1, 1), (1, 0), (0, 0)]
        uv = [uv0[mod1(k + rot, 4)] for k in 1:4]
        m = Makie.GeometryBasics.Mesh(pts,
            [Makie.GeometryBasics.TriangleFace(1, 2, 3),
             Makie.GeometryBasics.TriangleFace(1, 3, 4)]; uv = uv)
        mesh!(scene, m; color = Makie.FileIO.load(texpath),
              shading = NoShading)
    end
end

"""Add one NativeObject's mesh groups; returns the plots so the object can be
moved per frame with `apply_pose!`."""
function add_object!(scene, o::DDM.NativeObject)
    plots = []
    for g in o.groups
        pts = [Point3f(p...) for p in g.points]
        fcs = [Makie.GeometryBasics.TriangleFace(f...) for f in g.faces]
        uvs = [Vec2f(u...) for u in g.uvs]
        m = Makie.GeometryBasics.Mesh(pts, fcs; uv = uvs)
        col = g.texture === nothing ? RGBf(g.color...) :
              Makie.FileIO.load(g.texture)
        push!(plots, mesh!(scene, m; color = col, shading = NoShading))
    end
    return plots
end

function apply_pose!(plots, o::DDM.NativeObject)
    for plt in plots
        Makie.scale!(plt, o.scale, o.scale, o.scale)
        Makie.rotate!(plt, Makie.qrotation(Vec3f(0, 1, 0), o.angle))
        Makie.translate!(plt, o.offset...)
    end
end

# scene descriptions: static parts from the first state; per-frame poses via
# the same pure-layer transform (place_object re-reads the small OBJs; the
# expensive part — GL scenes and textures — is built exactly once)
nw0 = DDM.native_world(states[1]; assets = ASSETS)

parent = Scene(size = (2PW, PH), backgroundcolor = RGBf(0.45, 0.82, 0.98))
s_ego = Scene(parent, viewport = Makie.Observables.Observable(Makie.Rect2i(0, 0, PW, PH)),
              backgroundcolor = RGBf(0.45, 0.82, 0.98), clear = true)
s_bev = Scene(parent, viewport = Makie.Observables.Observable(Makie.Rect2i(PW, 0, PW, PH)),
              backgroundcolor = RGBf(0.45, 0.82, 0.98), clear = true)
cam3d!(s_ego); cam3d!(s_bev)

for sc in (s_ego, s_bev)
    populate_static!(sc, nw0)
end
# stop signs are static objects; ducks and the ego robot move
sign_objs = [o for o in nw0.objects if !any(g -> g.texture !== nothing &&
              endswith(g.texture, "duckie.png"), o.groups)]
duck_objs0 = [o for o in nw0.objects if any(g -> g.texture !== nothing &&
              endswith(g.texture, "duckie.png"), o.groups)]
for sc in (s_ego, s_bev), o in sign_objs
    apply_pose!(add_object!(sc, o), o)
end
duck_plots = [(add_object!(s_ego, o), add_object!(s_bev, o)) for o in duck_objs0]
ego_plots = add_object!(s_bev, nw0.ego)

# fixed BEV camera
cameracontrols(s_bev).fov[] = 35.0
let cx = nw0.extent[1] / 2, cz = nw0.extent[2] / 2, ht = 1.6 * max(nw0.extent...)
    update_cam!(s_bev, Vec3f(cx, ht, cz + 1e-3), Vec3f(cx, 0, cz), Vec3f(0, 1, 0))
end
cameracontrols(s_ego).fov[] = nw0.fov

"""Per-frame update: ego camera, ego mesh in the BEV, duck poses."""
function set_frame!(w)
    nw = DDM.native_world(w; assets = ASSETS)
    update_cam!(s_ego, Vec3f(nw.ego_eye...), Vec3f(nw.ego_lookat...),
                Vec3f(0, 1, 0))
    apply_pose!(ego_plots, nw.ego)
    ducks = [o for o in nw.objects if any(g -> g.texture !== nothing &&
             endswith(g.texture, "duckie.png"), o.groups)]
    for (k, o) in enumerate(ducks)
        k > length(duck_plots) && break
        apply_pose!(duck_plots[k][1], o)
        apply_pose!(duck_plots[k][2], o)
    end
end

out = joinpath(dirname(@__DIR__), "docs", "assets", "native_mcts_lap.gif")
mkpath(dirname(out))
println("recording ", length(states), " frames straight to GIF...")
t0 = time()
record(parent, out, states; framerate = 10) do w
    set_frame!(w)
end
@printf("gif: %.2f MB in %.0f s -> %s\n", filesize(out) / 1e6, time() - t0, out)
