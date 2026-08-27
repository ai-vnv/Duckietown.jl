# Two README GIFs, both drawn by the native renderer, both streamed straight
# into the GIF with Makie.record on ONE reused GL scene (no intermediate
# frames on disk):
#
#   1. docs/assets/native_dora_lap.gif — the hero: DORA (receding horizon,
#      the case-study formulation) completing a full :stop_and_duck_safe lap,
#      yielding to the crossing duck and stopping at the sign. The safe
#      scenario is the one whose stop sign actually faces the traffic — the
#      frozen source config shows the board's back to the route (measured in
#      notebooks/audit_sign_orientation.jl), which is why the earlier
#      MCTS-on-frozen-config GIF showed a reversed sign.
#
#   2. docs/assets/native_solver_zoo.gif — every built-in driver, one labeled
#      segment each, same scenario and same spawn seed, first 50 decisions
#      (10 s of model time) or until the episode ends: random, the four
#      frozen reference policies (Q-learning, SARSA, SAC, TD3), MCTS (the
#      FJ8 lap-study solver settings), and DORA.
#
# Frozen tabular policies load from the sibling reference checkpoints; SAC
# and TD3 from the weight exports committed in artifacts/fj9/weights.
# Lookalike renders, never parity evidence.

using DuckietownDecisionModels
using POMDPs, POMDPTools, MCTS, DORASolvers
using GLMakie
using Random, Printf

const DDM = DuckietownDecisionModels
const SCEN = :stop_and_duck_safe
const SEED = 1001
const ZOO_CAP = 40

const CFG = scenario_config(SCEN)
const BASE = DuckietownMDP(CFG; action_space = :discrete)
const TR = BASE.transition
const ACTS = collect(POMDPs.actions(BASE))

tile_of(s) = get_grid_coords(s.map, collect(s.ego.pos))
raw_of(s) = first(get_raw_state(s, TR.state_cfg))

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

# ── per-decision drivers (discrete) ─────────────────────────────────────────
"""Roll one episode with `pick(s, t)`; returns (states, outcome)."""
function run_discrete(pick; cap, seed = SEED)
    rng = MersenneTwister(1)
    s = rand(MersenneTwister(seed), initialstate(BASE))
    states = [s]
    prog = 0; tile = tile_of(s)
    for t in 1:cap
        a = pick(s, t)
        r = simulate_decision(TR, s, a, rng)
        s = r.sp
        push!(states, s)
        prog = advance(prog, tile, tile_of(s)); tile = tile_of(s)
        prog >= NRING && return (states, :lap)
        (r.terminated || r.truncated) &&
            return (states, Symbol(lowercase(string(r.reason))))
    end
    return (states, :running)
end

# ── continuous frozen actors ────────────────────────────────────────────────
function run_actor(name; cap, seed = SEED)
    cfg = scenario_config(SCEN; algorithm = Symbol(name))
    mdp = DuckietownMDP(cfg; action_space = :continuous)
    tr = mdp.transition
    wdir = normpath(joinpath(dirname(@__DIR__), "artifacts", "fj9", "weights", name))
    pol = name == "sac" ? SACActorPolicy(wdir) : TD3ActorPolicy(wdir)
    rng = MersenneTwister(1)
    s = rand(MersenneTwister(seed), initialstate(mdp))
    raw, _ = get_raw_state(s, tr.state_cfg)
    cs = get_continuous_state(s, raw, tr.state_cfg, tr.continuous_cfg;
        controller_cfg = cfg.duck_controller, stop_hold_progress = 0.0)
    obs = encode_continuous_state(cs, tr.continuous_cfg)
    states = [s]
    for t in 1:cap
        a = act(pol, obs)
        r = simulate_decision(tr, s, a, rng)
        s = r.sp
        push!(states, s)
        obs = encode_continuous_state(r.continuous_state, tr.continuous_cfg)
        (r.terminated || r.truncated) &&
            return (states, Symbol(lowercase(string(r.reason))))
    end
    return (states, :running)
end

# ── DORA: the case-study receding-horizon adaptor, compact ──────────────────
const K = 8
const C_MIN = 0.05
step_cost(r) = max(C_MIN, 1.0 - r.reward.total)
const GOAL = Ref(NRING)

struct LapState
    s::DuckieWorldState
    prog::Int
end

function macro_step(ls::LapState, a)
    s = ls.s; prog = ls.prog; tile = tile_of(s); c = 0.0
    for _ in 1:K
        r = simulate_decision(TR, s, a, MersenneTwister(1))
        c += step_cost(r)
        prog = advance(prog, tile, tile_of(r.sp))
        tile = tile_of(r.sp); s = r.sp
        (r.terminated || r.truncated || prog >= GOAL[]) &&
            return (LapState(s, prog), c, true)
    end
    return (LapState(s, prog), c, false)
end

struct LapMDP <: MDP{LapState,MacroAction} end
POMDPs.actions(::LapMDP) = ACTS
POMDPs.discount(::LapMDP) = 1.0
POMDPs.isterminal(::LapMDP, ls) = POMDPs.isterminal(BASE, ls.s)
POMDPs.transition(::LapMDP, ls, a) = Deterministic(first(macro_step(ls, a)))
lapkey(ls) = (min(ls.prog, GOAL[]), discretize(raw_of(ls.s)))
lapclassify(ls) = POMDPs.isterminal(BASE, ls.s) ? :crash :
                  ls.prog >= GOAL[] ? :goal : :normal
plan_from(from) = solve(DORASolver(
    start = from, classify = lapclassify,
    cost = (ls, a, lsp) -> macro_step(ls, a)[2],
    key = lapkey,
    c_min = K * C_MIN, c_to = 2000.0, c_crash = 1000.0, horizon = 60,
), LapMDP())

"""DORA under receding horizon until `goal` ring tiles or `cap` decisions."""
function run_dora(; goal = NRING, cap = 200, seed = SEED)
    GOAL[] = goal
    rng = MersenneTwister(1)
    ls = LapState(rand(MersenneTwister(seed), initialstate(BASE)), 0)
    states = [ls.s]
    dec = 0
    while dec < cap
        POMDPs.isterminal(BASE, ls.s) && return (states, :crash)
        ls.prog >= goal && return (states, :lap)
        pl = plan_from(ls)
        a = action(pl, ls)
        s = ls.s; prog = ls.prog; tile = tile_of(s)
        for _ in 1:K
            dec += 1
            r = simulate_decision(TR, s, a, rng)
            s = r.sp
            push!(states, s)
            prog = advance(prog, tile, tile_of(s)); tile = tile_of(s)
            prog >= goal && return (states, :lap)
            (r.terminated || r.truncated) &&
                return (states, r.terminated ? :crash : :timeout)
            dec >= cap && break
        end
        ls = LapState(s, prog)
    end
    return (states, :running)
end

# ── collect the episodes ─────────────────────────────────────────────────────
qpath(n) = normpath(joinpath(homedir(), "aivnv", "duckduck", "policies",
    n, "policy.npy"))

episodes = Tuple{String,Vector,Symbol}[]

println("random...")
let rrng = MersenneTwister(42)
    st, out = run_discrete((s, t) -> rand(rrng, ACTS); cap = ZOO_CAP)
    push!(episodes, ("random", st, out))
end

for name in ("q_learning", "sarsa")
    println(name, "...")
    pol = QTablePolicy(qpath(name))
    st, out = run_discrete((s, t) -> act(pol, raw_of(s)); cap = ZOO_CAP)
    push!(episodes, (name == "q_learning" ? "Q-learning (frozen)" :
                     "SARSA (frozen)", st, out))
end

for name in ("sac", "td3")
    println(name, "...")
    st, out = run_actor(name; cap = ZOO_CAP)
    push!(episodes, (uppercase(name) * " (frozen actor)", st, out))
end

println("MCTS...")
let planner = solve(MCTSSolver(n_iterations = 36, depth = 10,
        exploration_constant = 5.0, rng = MersenneTwister(2026),
        reuse_tree = false), BASE)
    st, out = run_discrete((s, t) -> action(planner, s); cap = ZOO_CAP)
    push!(episodes, ("MCTS (FJ8 settings)", st, out))
end

println("DORA (zoo segment)...")
let (st, out) = run_dora(; cap = ZOO_CAP)
    push!(episodes, ("DORA (receding horizon)", st, out))
end

for (n, st, out) in episodes
    @printf("  %-24s %3d decisions  %s\n", n, length(st) - 1, out)
end

println("DORA full lap for the hero GIF...")
t0 = time()
hero_states, hero_out = run_dora(; goal = NRING, cap = 200)
@printf("  hero: %s after %d decisions (%.0f s)\n",
        hero_out, length(hero_states) - 1, time() - t0)
hero_out === :lap || error("hero DORA episode did not complete a lap")

# ── ONE scene, two child views, a pixel-space label overlay ─────────────────
const ASSETS = duckietown_assets_root()
const PW, PH = 380, 285

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

is_duck(o) = any(g -> g.texture !== nothing &&
                 endswith(g.texture, "duckie.png"), o.groups)

nw0 = DDM.native_world(hero_states[1]; assets = ASSETS)

parent = Scene(size = (2PW, PH), backgroundcolor = RGBf(0.45, 0.82, 0.98))
s_ego = Scene(parent, viewport = Makie.Observables.Observable(Makie.Rect2i(0, 0, PW, PH)),
              backgroundcolor = RGBf(0.45, 0.82, 0.98), clear = true)
s_bev = Scene(parent, viewport = Makie.Observables.Observable(Makie.Rect2i(PW, 0, PW, PH)),
              backgroundcolor = RGBf(0.45, 0.82, 0.98), clear = true)
cam3d!(s_ego); cam3d!(s_bev)

for sc in (s_ego, s_bev)
    populate_static!(sc, nw0)
end
for sc in (s_ego, s_bev), o in filter(!is_duck, nw0.objects)
    apply_pose!(add_object!(sc, o), o)
end
duck_plots = [(add_object!(s_ego, o), add_object!(s_bev, o))
              for o in filter(is_duck, nw0.objects)]
ego_plots = add_object!(s_bev, nw0.ego)

overlay = Scene(parent, viewport = Makie.Observables.Observable(Makie.Rect2i(0, 0, 2PW, PH)),
                clear = false)
Makie.campixel!(overlay)
label = Makie.Observables.Observable(" ")
text!(overlay, Point2f(11, PH - 11); text = label, color = :black, fontsize = 17)
text!(overlay, Point2f(10, PH - 10); text = label, color = :white, fontsize = 17)

cameracontrols(s_bev).fov[] = 35.0
let cx = nw0.extent[1] / 2, cz = nw0.extent[2] / 2, ht = 1.6 * max(nw0.extent...)
    update_cam!(s_bev, Vec3f(cx, ht, cz + 1e-3), Vec3f(cx, 0, cz), Vec3f(0, 1, 0))
end
cameracontrols(s_ego).fov[] = nw0.fov

function set_frame!(w)
    nw = DDM.native_world(w; assets = ASSETS)
    update_cam!(s_ego, Vec3f(nw.ego_eye...), Vec3f(nw.ego_lookat...),
                Vec3f(0, 1, 0))
    apply_pose!(ego_plots, nw.ego)
    for (k, o) in enumerate(filter(is_duck, nw.objects))
        k > length(duck_plots) && break
        apply_pose!(duck_plots[k][1], o)
        apply_pose!(duck_plots[k][2], o)
    end
end

outdir = joinpath(dirname(@__DIR__), "docs", "assets")
mkpath(outdir)

# hero: the DORA lap
hero = joinpath(outdir, "native_dora_lap.gif")
println("recording hero (", length(hero_states), " frames)...")
label[] = "DORA - receding horizon, full lap"
record(parent, hero, hero_states; framerate = 10) do w
    set_frame!(w)
end
@printf("hero gif: %.2f MB -> %s\n", filesize(hero) / 1e6, hero)

# zoo: one labeled segment per solver, at a smaller panel size so the GIF
# stays under GitHub's ~10 MB image-render ceiling. GLMakie scenes do not
# resize cleanly, so rebuild the (cheap, one-time) scene at zoo size.
const ZW, ZH = 320, 240
parent = Scene(size = (2ZW, ZH), backgroundcolor = RGBf(0.45, 0.82, 0.98))
s_ego = Scene(parent, viewport = Makie.Observables.Observable(Makie.Rect2i(0, 0, ZW, ZH)),
              backgroundcolor = RGBf(0.45, 0.82, 0.98), clear = true)
s_bev = Scene(parent, viewport = Makie.Observables.Observable(Makie.Rect2i(ZW, 0, ZW, ZH)),
              backgroundcolor = RGBf(0.45, 0.82, 0.98), clear = true)
cam3d!(s_ego); cam3d!(s_bev)
for sc in (s_ego, s_bev)
    populate_static!(sc, nw0)
end
for sc in (s_ego, s_bev), o in filter(!is_duck, nw0.objects)
    apply_pose!(add_object!(sc, o), o)
end
duck_plots = [(add_object!(s_ego, o), add_object!(s_bev, o))
              for o in filter(is_duck, nw0.objects)]
ego_plots = add_object!(s_bev, nw0.ego)
overlay = Scene(parent, viewport = Makie.Observables.Observable(Makie.Rect2i(0, 0, 2ZW, ZH)),
                clear = false)
Makie.campixel!(overlay)
text!(overlay, Point2f(11, ZH - 11); text = label, color = :black, fontsize = 15)
text!(overlay, Point2f(10, ZH - 10); text = label, color = :white, fontsize = 15)
cameracontrols(s_bev).fov[] = 35.0
let cx = nw0.extent[1] / 2, cz = nw0.extent[2] / 2, ht = 1.6 * max(nw0.extent...)
    update_cam!(s_bev, Vec3f(cx, ht, cz + 1e-3), Vec3f(cx, 0, cz), Vec3f(0, 1, 0))
end
cameracontrols(s_ego).fov[] = nw0.fov

zoo = joinpath(outdir, "native_solver_zoo.gif")
println("recording zoo...")
record(parent, zoo; framerate = 10) do io
    for (name, st, out) in episodes
        tag = out === :running ? "" :
              out === :lap ? "  -> lap" : "  -> " * replace(string(out), "_" => " ")
        label[] = name * tag
        for w in st
            set_frame!(w)
            recordframe!(io)
        end
        # hold the last frame briefly so the outcome is readable
        for _ in 1:4
            recordframe!(io)
        end
    end
end
@printf("zoo gif: %.2f MB -> %s\n", filesize(zoo) / 1e6, zoo)
