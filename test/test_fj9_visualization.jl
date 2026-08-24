# FJ9.0 / FJ9.1 — the visualisation contract and the world renderer.
#
# These tests run with NO plotting backend installed. That is the point of the
# contract: every coordinate is computed in the core, so the geometry is
# testable on its own and the backend extension only draws it.
#
# The tests are geometric, not pixel-based. "Is this pixel #4C72B0" is brittle
# and proves nothing about correctness; "the drawn ego footprint IS
# get_agent_corners of this state" proves that what is drawn is what the
# physics uses.

using DuckietownDecisionModels
using POMDPs
using Test
using Random
using LinearAlgebra

const FJ9_QCFG = joinpath(pkgdir(DuckietownDecisionModels), "..", "duckduck",
    "policies", "q_learning", "training_config.yaml")

fj9_mdp() = DuckietownMDP(FJ9_QCFG; action_space=:discrete)
fj9_state(m, seed=11) = rand(MersenneTwister(seed), initialstate(m))

@testset "FJ9.0 the contract exists and needs no backend" begin
    # the four entry points FJ10 marked buildable are declared ...
    for f in (:render_world, :render_projection, :render_policy, :render_search)
        @test isdefined(DuckietownDecisionModels, f)
    end
    # ... with no methods, because no backend is loaded here. Calling one is a
    # MethodError, which is the correct behaviour for a package that does not
    # depend on a plotting library.
    @test isempty(methods(render_world))
    @test isempty(methods(render_projection))
    mdp = fj9_mdp()
    @test_throws MethodError render_world(mdp, fj9_state(mdp))

    # the FJ10 reservation still holds, and FJ9 must not quietly break it
    @test !isdefined(DuckietownDecisionModels, :render_observation)
    @test !isdefined(DuckietownDecisionModels, :render_belief)

    # no plotting package is a dependency
    proj = read(joinpath(pkgdir(DuckietownDecisionModels), "Project.toml"),
        String)
    deps = split(split(proj, "[deps]")[2], "\n[")[1]
    for pkg in ("Makie", "CairoMakie", "GLMakie", "Plots")
        @test !occursin(pkg, deps)
    end
    @test occursin("Makie", split(split(proj, "[weakdeps]")[2], "\n[")[1])
    @test occursin("DuckietownMakieExt = \"Makie\"", proj)
    @test isfile(joinpath(pkgdir(DuckietownDecisionModels), "ext",
        "DuckietownMakieExt.jl"))
end

@testset "FJ9.1 the drawn world IS the model's geometry" begin
    mdp = fj9_mdp()
    s = fj9_state(mdp)
    sc = world_scene(mdp, s)

    # --- ego --------------------------------------------------------------
    @test sc.ego_position === (s.ego.pos[1], s.ego.pos[3])
    @test sc.ego_angle === s.ego.angle
    dir = get_dir_vec(s.ego.angle)
    @test sc.ego_heading === (dir[1], dir[3])
    @test isapprox(hypot(sc.ego_heading...), 1.0; atol=1e-12)
    @test sc.ego_speed === s.ego.speed
    @test sc.ego_velocity ===
        (sc.ego_heading[1] * s.ego.speed, sc.ego_heading[2] * s.ego.speed)

    # the footprint is the TRUE collision polygon, not a decorative box
    corners = get_agent_corners(collect(s.ego.pos), s.ego.angle)
    @test size(corners) == (4, 2)
    @test length(sc.ego_footprint) == 5
    @test sc.ego_footprint[1] === sc.ego_footprint[end]      # closed ring
    for k in 1:4
        @test sc.ego_footprint[k] === (corners[k, 1], corners[k, 2])
    end
    # its centroid is the collision box centre, which is deliberately NOT the
    # pose: get_agent_corners builds the box around `_actual_center`
    cen = sum(collect.(sc.ego_footprint[1:4])) ./ 4
    @test !isapprox(cen[1], sc.ego_position[1]; atol=1e-6) ||
          !isapprox(cen[2], sc.ego_position[2]; atol=1e-6)
    @test hypot(cen[1] - sc.ego_position[1], cen[2] - sc.ego_position[2]) < 0.1

    # --- tiles ------------------------------------------------------------
    @test length(sc.tiles) == 9
    @test count(t -> t.drivable, sc.tiles) == 8
    ts = s.map.tile_size
    @test sc.tile_size === ts
    for t in sc.tiles
        @test length(t.corners) == 5
        @test t.corners[1] === t.corners[end]
        @test t.corners[1] === (t.i * ts, t.j * ts)
        @test t.corners[3] === ((t.i + 1) * ts, (t.j + 1) * ts)
    end
    @test sc.extent === (0.0, 3ts, 0.0, 3ts)

    # --- lane centrelines come from the model's own Bezier curves ---------
    @test !isempty(sc.lane_centrelines)
    drivable = drivable_tiles(s.map)
    @test length(sc.lane_centrelines) ==
        sum(length(_get_tile(s.map, i, j).curves) for (i, j) in drivable)
    tile = _get_tile(s.map, drivable[1]...)
    cm = curve_matrix(tile.curves[1])
    first_curve = sc.lane_centrelines[1]
    p0 = bezier_point(cm, 0.0)
    p1 = bezier_point(cm, 1.0)
    @test first_curve[1] === (p0[1], p0[3])
    @test first_curve[end] === (p1[1], p1[3])

    # --- ducks ------------------------------------------------------------
    @test length(sc.ducks) == length(s.ducks)
    for (k, d) in enumerate(s.ducks)
        @test sc.ducks[k] === (d.pos[1], d.pos[3])
        @test sc.duck_visible[k] === d.visible
        @test sc.duck_active[k] === d.pedestrian_active
        @test length(sc.duck_footprints[k]) == length(d.obj_corners) + 1
    end

    # --- purity: extracting a scene must not touch the state --------------
    snapshot = branch(s)
    world_scene(mdp, s)
    @test worlds_identical(s, snapshot)
end

@testset "FJ9.1 the stop line is the model's measurement, not a decoration" begin
    # The model has no stop-line object. `next_stop_candidate` measures
    #     ahead  = dot(sign.pos - ego.pos, forward)
    #     d_stop = max(0, ahead - sign_to_line_offset)
    # along the EGO's lane frame. The drawn line must satisfy that identity;
    # an earlier version offset along the sign's own facing and looked
    # perfectly plausible while corresponding to nothing.
    mdp = fj9_mdp()
    cfg = mdp.transition.state_cfg

    # A stop sign is only an accepted candidate from states the vehicle
    # actually reaches while driving — never from a spawn pose — so the states
    # are found by driving, exactly as FJ8.3c located the trigger region.
    function states_with_stop_candidate(n)
        qpath = joinpath(pkgdir(DuckietownDecisionModels), "..", "duckduck",
            "policies", "q_learning", "policy.npy")
        pol = isfile(qpath) ? QTablePolicy(qpath; solver=:q_learning) : nothing
        out = DuckieWorldState[]
        for seed in 1:4
            s = fj9_state(mdp, seed)
            rng = MersenneTwister(90 + seed)
            for _ in 1:200
                distance_to_next_stop(s, cfg) === nothing || push!(out, s)
                length(out) >= n && return out
                a = pol === nothing ? FAST_STRAIGHT : policy_action(pol, mdp, s)
                r = simulate_decision(mdp.transition, s, a, rng)
                s = r.sp
                (r.terminated || r.truncated) && break
            end
        end
        return out
    end

    checked = 0
    for s in states_with_stop_candidate(12)
        isempty(s.stop_signs) && continue
        sc = world_scene(mdp, s)
        forward, right = lane_frame_tabular(s)
        f = [forward[1], forward[3]]
        f ./= hypot(f...)

        for (k, sg) in enumerate(s.stop_signs)
            x1, z1, x2, z2 = sc.stop_lines[k]
            centre = [(x1 + x2) / 2, (z1 + z2) / 2]
            expected = [sg.pos[1] - cfg.sign_to_line_offset * f[1],
                sg.pos[3] - cfg.sign_to_line_offset * f[2]]
            @test isapprox(centre, expected; atol=1e-12)

            # the segment is perpendicular to the ego's forward ...
            seg = [x2 - x1, z2 - z1]
            @test isapprox(dot(seg, f), 0.0; atol=1e-12)
            # ... and as wide as the model's own lateral acceptance gate
            @test isapprox(hypot(seg...), 2 * cfg.stop_lateral_limit; atol=1e-12)

            # THE identity: along-track distance to the drawn line reproduces
            # exactly what the observer reports as d_stop
            along = dot(centre .- [s.ego.pos[1], s.ego.pos[3]], f)
            d_stop = distance_to_next_stop(s, cfg)
            if d_stop !== nothing
                _, idx = next_stop_candidate(s, cfg)
                if idx == k - 1
                    @test isapprox(max(0.0, along), d_stop; atol=1e-9)
                    checked += 1
                end
            end
        end
    end
    @test checked >= 1      # the identity was actually exercised
    @info "FJ9.1 stop-line identity checked on $checked (state, sign) pairs"
end

@testset "FJ9.1 the view shows what lies outside the tile grid" begin
    # On small_loop the injected stop sign sits at z = 1.8135 while the 3x3
    # grid ends at 1.755 — the reference's own placement, validated in FJ3 and
    # exercised live in FJ5/FJ6. A view clipped to the map would hide it.
    mdp = fj9_mdp()
    s = fj9_state(mdp)
    sc = world_scene(mdp, s)
    ts = s.map.tile_size

    @test any(sg -> sg[2] > 3ts, sc.stop_signs)     # genuinely outside
    xmin, xmax, zmin, zmax = sc.view_extent
    @test xmin <= sc.extent[1] && xmax >= sc.extent[2]
    @test zmin <= sc.extent[3] && zmax >= sc.extent[4]
    for sg in sc.stop_signs
        @test xmin <= sg[1] <= xmax
        @test zmin <= sg[2] <= zmax
    end
    for (x1, z1, x2, z2) in sc.stop_lines
        @test xmin <= min(x1, x2) && max(x1, x2) <= xmax
        @test zmin <= min(z1, z2) && max(z1, z2) <= zmax
    end
    for p in sc.ego_footprint
        @test xmin <= p[1] <= xmax && zmin <= p[2] <= zmax
    end
    @info "FJ9.1 map vs view extent" map = round.(sc.extent; digits=4) view =
        round.(sc.view_extent; digits=4)
end

@testset "FJ9.1 trajectory overlay" begin
    mdp = fj9_mdp()
    s = fj9_state(mdp)
    rng = MersenneTwister(3)
    states = DuckieWorldState[s]
    cur = s
    for _ in 1:10
        r = simulate_decision(mdp.transition, cur, FAST_STRAIGHT, rng)
        cur = r.sp
        push!(states, cur)
    end
    pts = trajectory_points(states)
    @test length(pts) == length(states)
    @test pts[1] === (s.ego.pos[1], s.ego.pos[3])
    @test pts[end] === (cur.ego.pos[1], cur.ego.pos[3])

    sc = world_scene(mdp, cur; trajectory=pts)
    @test sc.trajectory == pts
    @test world_scene(mdp, cur).trajectory == NTuple{2,Float64}[]
end

@testset "FJ9.2 the panel's semantics come from the core, not the backend" begin
    mdp = DuckietownMDP(joinpath(pkgdir(DuckietownDecisionModels), "..",
        "duckduck", "policies", "sac", "training_config.yaml");
        action_space=:continuous)
    s = fj9_state(mdp, 11)
    raw, _ = get_raw_state(s, mdp.transition.state_cfg)
    cont = get_continuous_state(s, raw, mdp.transition.state_cfg,
        mdp.transition.continuous_cfg; controller_cfg=mdp.transition.duck_cfg)
    sc = projection_scene(raw, cont)

    # 1. every component exactly once, in a deterministic order
    @test length(sc.entries) == fieldcount(ContinuousState) == 15
    @test [e.field for e in sc.entries] == collect(fieldnames(ContinuousState))
    @test [e.order for e in sc.entries] == collect(1:15)
    @test length(unique(e.field for e in sc.entries)) == 15
    @test projection_scene(raw, cont).entries == sc.entries   # reproducible

    # 2. the values ARE the policy's input, read off the same object rather
    #    than recomputed. Row k is component k of the encoded vector.
    for e in sc.entries
        @test e.value === getfield(cont, e.field)
    end
    encoded = encode_continuous_state(cont, mdp.transition.continuous_cfg)
    @test length(encoded) == length(sc.entries)
    @test sc.source === :continuous_state

    # 3. labels, units and privilege classes are core metadata
    cls = Dict(r.name => r.class for r in continuous_state_observability())
    for e in sc.entries
        @test e.privilege === cls[e.field]
        @test e.category isa ProjectionCategory
        @test !isempty(e.display)
    end
    @test only(filter(e -> e.field === :d, sc.entries)).unit == "m"
    @test only(filter(e -> e.field === :phi, sc.entries)).unit == "rad"
    @test only(filter(e -> e.field === :v, sc.entries)).unit == "m/s"
    @test only(filter(e -> e.field === :kappa, sc.entries)).unit == "1/m"
    @test only(filter(e -> e.field === :sigma_stop, sc.entries)).unit == ""
    # category and privilege are orthogonal: a stop-subsystem field can be
    # agent memory
    sigma = only(filter(e -> e.field === :sigma_stop, sc.entries))
    @test sigma.category === STOP_SUBSYSTEM
    @test sigma.privilege === AGENT_MEMORY

    # 4. labelled privileged, never "observation"
    @test sc.title === PROJECTION_PANEL_TITLE
    @test occursin("Privileged", sc.title)
    @test !occursin("bservation", sc.title)

    # 5. tabular-only facts are CONTEXT, not presented as policy inputs
    @test !isempty(sc.context)
    @test all(c -> c.first ∉ String.(fieldnames(ContinuousState)), sc.context)
    @test any(c -> c.first == "tile", sc.context)

    # 6. a different state gives a different panel, and building one does not
    #    touch the world model
    snapshot = branch(s)
    s2 = fj9_state(mdp, 12)
    raw2, _ = get_raw_state(s2, mdp.transition.state_cfg)
    cont2 = get_continuous_state(s2, raw2, mdp.transition.state_cfg,
        mdp.transition.continuous_cfg; controller_cfg=mdp.transition.duck_cfg)
    sc2 = projection_scene(raw2, cont2)
    @test [e.field for e in sc2.entries] == [e.field for e in sc.entries]
    @test [e.value for e in sc2.entries] != [e.value for e in sc.entries]
    @test worlds_identical(s, snapshot)

    # a `nothing` field renders as an em dash rather than "nothing"
    @test all(e -> !occursin("nothing", e.display), sc.entries)
    nothings = filter(e -> e.value === nothing, sc.entries)
    isempty(nothings) || @test all(e -> e.display == "—", nothings)

    @info "FJ9.2 panel" entries = length(sc.entries) context = length(sc.context) categories =
        unique(e.category for e in sc.entries)
end

@testset "FJ9.2 the projection panel is labelled privileged" begin
    mdp = fj9_mdp()
    s = fj9_state(mdp)
    raw, _ = get_raw_state(s, mdp.transition.state_cfg)
    cont = get_continuous_state(s, raw, mdp.transition.state_cfg,
        mdp.transition.continuous_cfg; controller_cfg=mdp.transition.duck_cfg)

    @test occursin("Privileged", PROJECTION_PANEL_TITLE)
    @test !occursin("bservation", PROJECTION_PANEL_TITLE)

    rows = projection_rows(raw, cont)
    # exactly the 15 policy inputs: `tile` is tabular-projection CONTEXT and
    # lives in ProjectionScene.context, not in the rows
    @test length(rows) == fieldcount(ContinuousState)
    # every continuous component carries its FJ10 observability class
    cls = Dict(r.name => r.class for r in continuous_state_observability())
    for f in fieldnames(ContinuousState)
        row = only(filter(r -> r.label == String(f), rows))
        @test row.observability === cls[f]
        @test !isempty(row.value)
    end
    # a `nothing` field renders as an em dash rather than "nothing"
    @test all(r -> !occursin("nothing", r.value), rows)
end

@testset "FJ9.0 the search snapshot is solver-neutral and validated" begin
    nodes = [SearchNode(1, 0, 0, 30, 0.0, nothing),
        SearchNode(2, 1, 1, 17, 1.25, FAST_STRAIGHT),
        SearchNode(3, 1, 1, 13, -0.5, BRAKE),
        SearchNode(4, 2, 2, 5, 2.0, SLOW_LEFT)]
    snap = SearchSnapshot("test-planner", 30, nodes)
    @test snap.solver == "test-planner"
    @test snap.extra === NamedTuple()
    @test length(root_children(snap)) == 2
    @test search_max_depth(snap) == 2
    chk = check_snapshot(snap)
    @test chk.ok
    @test chk.nodes == 4 && chk.roots == 1
    @test isempty(chk.issues)

    # a continuous-action snapshot needs no different type
    csnap = SearchSnapshot("dpw", 10,
        [SearchNode(1, 0, 0, 10, 0.0, nothing),
         SearchNode(2, 1, 1, 10, 0.3, DuckieAction(0.2, -0.4))])
    @test check_snapshot(csnap).ok
    @test root_children(csnap)[1].action isa DuckieAction

    # malformed snapshots are caught where they are built
    bad_depth = SearchSnapshot("x", 1, [SearchNode(1, 0, 0, 1, 0.0, nothing),
        SearchNode(2, 1, 5, 1, 0.0, BRAKE)])
    @test !check_snapshot(bad_depth).ok
    forward_parent = SearchSnapshot("x", 1,
        [SearchNode(1, 0, 0, 1, 0.0, nothing),
         SearchNode(2, 3, 1, 1, 0.0, BRAKE),
         SearchNode(3, 1, 1, 1, 0.0, BRAKE)])
    @test !check_snapshot(forward_parent).ok
    @test check_snapshot(SearchSnapshot("empty", 0, SearchNode[])).ok
end

# ---------------------------------------------------------------------------
# FJ9.3 — policy / value / ambiguity slices
# ---------------------------------------------------------------------------

fj9_qpolicy() = QTablePolicy(joinpath(pkgdir(DuckietownDecisionModels), "..",
    "duckduck", "policies", "q_learning", "policy.npy"); solver=:q_learning)

@testset "FJ9.3a the slice contract is core data, not a figure" begin
    pol = fj9_qpolicy()
    sl = policy_slice(pol, :d, :phi; xs=range(-0.2, 0.2; length=21),
        ys=range(-0.5, 0.5; length=21), name="q_learning")

    @test sl isa PolicySlice
    @test sl isa TabularPolicySlice
    @test !(sl isa ContinuousPolicySlice)
    @test size(sl.cells) == (21, 21)
    @test sl.x.field === :d && sl.y.field === :phi
    @test sl.x.unit == "m" && sl.y.unit == "rad"
    @test length(sl.x.values) == 21 && length(sl.y.values) == 21

    # no backend anywhere near this
    @test isempty(methods(render_policy))

    # the surfaces are plain arrays the backend can draw without thinking
    for f in (value_surface, action_surface, tie_surface, margin_surface)
        @test size(f(sl)) == size(sl.cells)
    end
    @test eltype(action_surface(sl)) === Int
    @test eltype(value_surface(sl)) === Float64
end

@testset "FJ9.3b the tabular action comes from the validated decide()" begin
    pol = fj9_qpolicy()
    sl = policy_slice(pol, :d, :phi; xs=range(-0.2, 0.2; length=15),
        ys=range(-0.5, 0.5; length=15))

    # every cell reproduces `decide` exactly — the visualisation inherits
    # FJ7's near-tie rule rather than re-deriving it
    grid = raw_state_grid(:d, :phi, sl.x.values, sl.y.values,
        Dict{Symbol,Any}())
    for k in eachindex(grid)
        dec = decide(pol, discretize(grid[k]))
        @test sl.cells[k].selected_action === dec.action
        @test sl.cells[k].action_id == dec.action_id
        @test sl.cells[k].tie_count == length(dec.ties)
        @test sl.cells[k].q_margin === dec.q_margin
        @test sl.cells[k].index === discretize(grid[k])
    end
end

@testset "FJ9.3b near-tie: the slice must not fall back to argmax" begin
    # A synthetic table where a plain first-maximum argmax and the reference
    # near-tie rule are exercised together. On the shipped checkpoints they
    # happen to agree everywhere (FJ7 measured argmax_differs = 0), so without
    # a synthetic case this property would pass for the wrong reason.
    table = zeros(Float32, Q_SHAPE...)
    table[:, :, :, :, :, :, :, 4] .= 1.0f0    # action id 3
    table[:, :, :, :, :, :, :, 2] .= 1.0f0    # action id 1, tied with it
    pol = QTablePolicy(table, collect(0:6), build_action_table(ActionConfig()),
        :synthetic, "<synthetic>")
    idx = (0, 0, 0, 0, 0, 0, 0)
    dec = decide(pol, idx)
    @test dec.action_id == 1              # reference rule: lowest tied id
    @test 3 in dec.ties
    @test length(dec.ties) == 2

    sl = policy_slice(pol, :d, :phi; xs=range(-0.01, 0.01; length=3),
        ys=range(-0.01, 0.01; length=3), name="synthetic")
    @test all(c -> c.action_id == 1, sl.cells)
    @test all(c -> c.tie_count == 2, sl.cells)
    # the VALUE surface is the raw maximum, untouched by tie-breaking
    @test all(c -> c.qmax == 1.0, sl.cells)
    @test all(c -> c.q_margin == 0.0, sl.cells)
end

@testset "FJ9.3c value and ambiguity are separate layers" begin
    pol = fj9_qpolicy()
    sl = policy_slice(pol, :d, :phi)

    V = value_surface(sl)
    ties = tie_surface(sl)
    margins = margin_surface(sl)
    @test all(isfinite, V)
    @test all(t -> 1 <= t <= 7, ties)
    @test all(m -> m >= 0, margins)

    # tied cells have zero margin: the two diagnostics agree
    for k in eachindex(sl.cells)
        if sl.cells[k].tie_count > 1
            @test margins[k] == 0.0
        end
    end
    # the ambiguity layer carries information a one-action-per-cell map cannot
    @test length(unique(ties)) > 1
    @info "FJ9.3c ambiguity" tie_range = extrema(ties) margin_range =
        round.(extrema(margins); digits=3) decisive_cells =
        count(==(1), ties) tied_cells = count(>(1), ties)
end

@testset "FJ9.3e honesty: coordinates, mode and fixed context" begin
    pol = fj9_qpolicy()
    sl = policy_slice(pol, :d, :phi)

    # the mode is DATA, not a caption
    @test sl.mode === FEATURE_SPACE
    @test occursin("not guaranteed", SLICE_FEATURE_SPACE_CAVEAT)
    @test occursin("feature-space", SLICE_FEATURE_SPACE_CAVEAT)
    @test occursin(string(sl.mode), slice_summary(sl))
    @test occursin(SLICE_FEATURE_SPACE_CAVEAT, slice_summary(sl))

    # d and phi are NOT two independent tabular axes, and the slice says so
    @test occursin("bin(phi + d)", sl.coordinate_note)
    @test sl.distinct_states < length(sl.cells)
    @test sl.distinct_states == length(unique(c.index for c in sl.cells))
    @info "FJ9.3e coordinate collapse" cells = length(sl.cells) distinct_tabular_states =
        sl.distinct_states

    # every dimension not on an axis is recorded, and rendered readably
    @test length(sl.fixed) == length(TABULAR_SLICE_FIELDS) - 2
    fixed_fields = [k for (k, _) in sl.fixed]
    @test !(:d in fixed_fields) && !(:phi in fixed_fields)
    @test Set(fixed_fields) == Set(setdiff(TABULAR_SLICE_FIELDS, (:d, :phi)))
    lines = fixed_context_lines(sl)
    @test any(l -> occursin("sigma_stop = false", l), lines)   # not "0.0"
    @test any(l -> occursin("d_stop = none", l), lines)        # not "nothing"

    # fixed context is part of IDENTITY: same grid, different context,
    # different fingerprint
    other = policy_slice(pol, :d, :phi; fixed=Dict{Symbol,Any}(:v => 0.30))
    @test slice_fingerprint(sl) != slice_fingerprint(other)
    @test slice_fingerprint(sl) == slice_fingerprint(policy_slice(pol, :d, :phi))
    @test length(slice_fingerprint(sl)) == 16
end

@testset "FJ9.3d continuous slices keep v and omega separate" begin
    if !torch_policy_available()
        @test_skip "ddm-torch unavailable"
    else
        b = TorchPolicyReferenceBackend()
        try
            mdp = DuckietownMDP(joinpath(pkgdir(DuckietownDecisionModels), "..",
                "duckduck", "policies", "sac", "training_config.yaml");
                action_space=:continuous)
            cfg = mdp.transition.continuous_cfg
            meta = torch_policy_init!(b, "sac")
            pol = SACActorPolicy(String(meta["weights_dir"]))

            sl = policy_slice(pol, cfg, :d, :phi;
                xs=range(-0.2, 0.2; length=13), ys=range(-0.5, 0.5; length=13))
            @test sl isa ContinuousPolicySlice
            @test size(sl.cells) == (13, 13)
            @test sl.policy_name == "sac"
            @test sl.mode === FEATURE_SPACE

            # two separate surfaces, never one ambiguous symbol
            V = v_surface(sl)
            W = omega_surface(sl)
            @test size(V) == size(W) == size(sl.cells)
            @test all(0.0 .<= V .<= 0.41)
            @test all(-1.5 .<= W .<= 1.5)
            @test V != W

            # the cells come from the FJ7-validated actor pipeline
            grid = continuous_state_grid(:d, :phi, sl.x.values, sl.y.values,
                Dict{Symbol,Any}())
            for k in (1, length(grid) ÷ 2, length(grid))
                a = act(pol, encode_continuous_state(grid[k], cfg))
                @test sl.cells[k].v_cmd === a.v
                @test sl.cells[k].omega_cmd === a.omega
            end

            # all 13 other features are recorded as fixed context
            @test length(sl.fixed) == fieldcount(ContinuousState) - 2
            @test Set(k for (k, _) in sl.fixed) ==
                Set(setdiff(fieldnames(ContinuousState), (:d, :phi)))

            # the same pipeline serves TD3 with no new function
            meta3 = torch_policy_init!(b, "td3")
            pol3 = TD3ActorPolicy(String(meta3["weights_dir"]))
            sl3 = policy_slice(pol3, cfg, :d_stop, :v;
                xs=range(0.0, 1.0; length=9), ys=range(0.0, 0.41; length=9))
            @test sl3.policy_name == "td3"
            @test sl3.x.field === :d_stop && sl3.y.field === :v
            @test all(0.0 .<= v_surface(sl3) .<= 0.41)

            @info "FJ9.3d continuous slices" sac_v = round.(extrema(V); digits=4) sac_omega =
                round.(extrema(W); digits=4) td3_v =
                round.(extrema(v_surface(sl3)); digits=4)
        finally
            close(b)
        end
    end
end

@testset "FJ9.3e the fixed context is not a neutral choice" begin
    # The strongest argument for recording `mode` and `fixed` as data rather
    # than as a caption: a slice's default context can be off the manifold of
    # states the policy actually visits, and the picture changes qualitatively
    # because of it.
    #
    # Measured for TD3: on states reached while driving it commands
    # v in [0.034, 0.39], varied. On this module's default slice context it
    # saturates at v = 0.41 everywhere. The encoded inputs differ in exactly
    # two of fifteen components — `kappa` and `duck_crossing_available`.
    if !torch_policy_available()
        @test_skip "ddm-torch unavailable"
    else
        b = TorchPolicyReferenceBackend()
        try
            mdp = DuckietownMDP(joinpath(pkgdir(DuckietownDecisionModels), "..",
                "duckduck", "policies", "td3", "training_config.yaml");
                action_space=:continuous)
            m = mdp.transition
            meta = torch_policy_init!(b, "td3")
            pol = TD3ActorPolicy(String(meta["weights_dir"]))

            # what the policy commands on states it actually reaches
            s = rand(MersenneTwister(1001), initialstate(mdp))
            rng = MersenneTwister(7)
            visited = Float64[]
            for _ in 1:12
                raw, _ = get_raw_state(s, m.state_cfg)
                cont = get_continuous_state(s, raw, m.state_cfg,
                    m.continuous_cfg; controller_cfg=m.duck_cfg)
                a = act(pol, encode_continuous_state(cont, m.continuous_cfg))
                push!(visited, a.v)
                r = simulate_decision(m, s, a, rng)
                s = r.sp
                (r.terminated || r.truncated) && break
            end
            @test length(visited) >= 5
            @test minimum(visited) < 0.2          # varied, not saturated
            @test maximum(visited) < 0.41

            # what the default slice context produces
            sl = policy_slice(pol, m.continuous_cfg, :d, :phi;
                xs=range(-0.1, 0.1; length=5), ys=range(-0.2, 0.2; length=5))
            V = v_surface(sl)
            # near-saturated across the whole grid: measured [0.4034, 0.41]
            @test all(v -> v > 0.40, V)
            @test maximum(V) <= 0.41

            # the two pictures disagree, and the difference is the context
            @test maximum(visited) < minimum(V)

            # a context closer to what is visited restores the variation,
            # which is the point: `fixed` determines the figure
            realistic = policy_slice(pol, m.continuous_cfg, :d, :phi;
                xs=range(-0.1, 0.1; length=5), ys=range(-0.2, 0.2; length=5),
                fixed=Dict{Symbol,Any}(:kappa => 2.2,
                    :duck_crossing_available => false))
            @test slice_fingerprint(realistic) != slice_fingerprint(sl)
            @test v_surface(realistic) != V

            @info "FJ9.3e context sensitivity" visited_v_range =
                round.(extrema(visited); digits=4) default_context_v =
                round.(extrema(V); digits=4) realistic_context_v =
                round.(extrema(v_surface(realistic)); digits=4)
        finally
            close(b)
        end
    end
end

# ---------------------------------------------------------------------------
# FJ9.4 — rollout comparison from the frozen FJ8.4b artefacts
# ---------------------------------------------------------------------------

const FJ94_ARTIFACT = joinpath(pkgdir(DuckietownDecisionModels), "artifacts",
    "fj8", "six_solver_episodes.csv")

@testset "FJ9.4 the artefact is the only source of data" begin
    if !isfile(FJ94_ARTIFACT)
        @test_skip "FJ8.4b artefact not present"
    else
        a = load_rollout_artifact(FJ94_ARTIFACT)
        @test a isa RolloutAggregate
        @test a.provenance.rows == 120
        @test length(a.provenance.solvers) == 6
        @test length(a.provenance.seeds) == 20
        @test a.provenance.horizon == 150
        @test a.provenance.experiment_id == "FJ8.4b"
        @test !isempty(a.provenance.content_fingerprint)

        # the loader constructs no model and runs nothing: the source file's
        # own content is what every number traces back to
        @test length(a.records) == 120
        @test Set(r.solver for r in a.records) == Set(a.provenance.solvers)

        # provenance travels with the data
        lines = provenance_lines(a)
        @test any(l -> occursin("FJ8.4b", l), lines)
        @test any(l -> occursin("six_solver_episodes.csv", l), lines)
        @test any(l -> occursin(a.provenance.content_fingerprint, l), lines)
        @test any(l -> occursin("150 decisions", l), lines)
    end
end

@testset "FJ9.4 the loader validates rather than forgives" begin
    if !isfile(FJ94_ARTIFACT)
        @test_skip "FJ8.4b artefact not present"
    else
        text = read(FJ94_ARTIFACT, String)
        lines = split(strip(text), "\n")
        dir = mktempdir()

        # a renamed column is a schema error, not something to guess around
        bad = joinpath(dir, "bad_header.csv")
        write(bad, replace(text, "mean_abs_d" => "meanAbsD"))
        @test_throws ArgumentError load_rollout_artifact(bad)

        # an episode longer than the declared horizon is refused
        long = joinpath(dir, "too_long.csv")
        row = split(lines[2], ",")
        row[4] = "999"
        write(long, join(vcat(lines[1], join(row, ","), lines[3:end]), "\n"))
        @test_throws ArgumentError load_rollout_artifact(long)

        # a solver missing a seed breaks pairing and must be refused
        dropped = joinpath(dir, "unpaired.csv")
        write(dropped, join(vcat(lines[1], lines[3:end]), "\n"))
        @test_throws ArgumentError load_rollout_artifact(dropped)

        @test_throws ArgumentError load_rollout_artifact(joinpath(dir, "nope.csv"))
    end
end

@testset "FJ9.4 negative control: perturbed evidence is different evidence" begin
    if !isfile(FJ94_ARTIFACT)
        @test_skip "FJ8.4b artefact not present"
    else
        a = load_rollout_artifact(FJ94_ARTIFACT)
        text = read(FJ94_ARTIFACT, String)
        lines = split(strip(text), "\n")
        dir = mktempdir()

        # copying it unchanged must give identical evidence ...
        same = joinpath(dir, "copy.csv")
        write(same, text)
        b = load_rollout_artifact(same)
        @test artifact_fingerprint(b) == artifact_fingerprint(a)
        @test comparison_table(b) == comparison_table(a)

        # ... and changing ONE number must not be forgiven anywhere
        row = split(lines[2], ",")
        original_return = row[5]
        row[5] = string(parse(Float64, original_return) + 1.0)
        perturbed = joinpath(dir, "perturbed.csv")
        write(perturbed, join(vcat(lines[1], join(row, ","), lines[3:end]), "\n"))
        c = load_rollout_artifact(perturbed)
        @test artifact_fingerprint(c) != artifact_fingerprint(a)

        solver = split(lines[2], ",")[1]
        @test solver_summary(c, solver).mean_return !=
              solver_summary(a, solver).mean_return
        # and only that solver's summary moved
        for s in a.provenance.solvers
            s == solver && continue
            @test solver_summary(c, s).mean_return ==
                  solver_summary(a, s).mean_return
        end

        # perturbing a termination reason changes the outcome classification
        row2 = split(lines[2], ",")
        row2[25] = "offroad"
        term = joinpath(dir, "term.csv")
        write(term, join(vcat(lines[1], join(row2, ","), lines[3:end]), "\n"))
        d = load_rollout_artifact(term)
        @test artifact_fingerprint(d) != artifact_fingerprint(a)
        @test solver_summary(d, solver).env_terminated !=
              solver_summary(a, solver).env_terminated
    end
end

@testset "FJ9.4 pairing is preserved and seeds are explicit" begin
    if !isfile(FJ94_ARTIFACT)
        @test_skip "FJ8.4b artefact not present"
    else
        a = load_rollout_artifact(FJ94_ARTIFACT)
        seeds, ret = paired_metric(a, :ret)
        @test length(seeds) == 20
        @test issorted(seeds)
        @test length(ret) == 6
        @test all(v -> length(v) == 20, values(ret))

        # column k of every solver's vector is the SAME initial condition
        for (k, sd) in enumerate(seeds)
            for s in a.provenance.solvers
                row = only(filter(r -> r.solver == s && r.seed == sd, a.records))
                @test ret[s][k] == row.ret
            end
        end

        # a single-seed view keeps the six solvers together
        c = comparison_at_seed(a, seeds[1])
        @test c.seed == seeds[1]
        @test length(c.records) == 6
        @test Set(r.solver for r in c.records) == Set(a.provenance.solvers)
        @test all(r -> r.seed == seeds[1], c.records)
        @test artifact_fingerprint(c) == artifact_fingerprint(a)

        # a seed that does not exist is an error, not an empty figure
        @test_throws ArgumentError comparison_at_seed(a, 999_999)

        # a "representative" seed comes from a stated rule
        for s in a.provenance.solvers
            sd = median_return_seed(a, s)
            @test sd in seeds
        end
    end
end

@testset "FJ9.4 horizon is distinguished from environment termination" begin
    if !isfile(FJ94_ARTIFACT)
        @test_skip "FJ8.4b artefact not present"
    else
        a = load_rollout_artifact(FJ94_ARTIFACT)
        for r in a.records
            if r.reason == "in_progress"
                @test outcome(r) === HORIZON_REACHED
                @test r.decisions == a.provenance.horizon || r.decisions >= 1
            else
                @test outcome(r) === ENV_TERMINATED
            end
        end
        # both kinds are present, so the distinction is exercised
        @test any(r -> outcome(r) === HORIZON_REACHED, a.records)
        @test any(r -> outcome(r) === ENV_TERMINATED, a.records)

        # the two failure modes FJ8.4b found are legible as different things
        td3 = solver_summary(a, "td3")
        dpw = solver_summary(a, "dpw@1k")
        @test td3.env_terminated == 0 && td3.horizon_reached == 20
        @test dpw.env_terminated > 0
        @test dpw.offroad > 0
        @test td3.offroad == 0 && td3.collisions == 0
        # similar returns, entirely different behaviour
        @test abs(td3.mean_return - dpw.mean_return) < 10
        @test dpw.mean_length < td3.mean_length
        @info "FJ9.4 same score, different failure" td3 =
            (td3.mean_return, td3.env_terminated, td3.offroad, td3.mean_speed) dpw =
            (dpw.mean_return, dpw.env_terminated, dpw.offroad, dpw.mean_speed)
    end
end

@testset "FJ9.4 not-applicable survives to the renderer" begin
    if !isfile(FJ94_ARTIFACT)
        @test_skip "FJ8.4b artefact not present"
    else
        a = load_rollout_artifact(FJ94_ARTIFACT)
        td3 = solver_summary(a, "td3")
        # TD3 never reached a stop sign: the rate is missing, not zero
        @test td3.stop_encounters == 0
        @test td3.stop_compliance === nothing
        @test stop_compliance_of(filter(r -> r.solver == "td3", a.records)) ===
            nothing

        # a solver that did encounter signs has a real rate
        sac = solver_summary(a, "sac")
        @test sac.stop_encounters > 0
        @test sac.stop_compliance isa Float64
        @test 0.0 <= sac.stop_compliance <= 1.0

        # and the rendered table says N/A, never 0 % or 100 %
        tbl = comparison_table(a)
        td3_line = only(filter(l -> startswith(l, "td3"), split(tbl, "\n")))
        @test occursin("N/A", td3_line)
        @test !occursin("0.0%", td3_line)
        @test !occursin("100.0%", td3_line)
    end
end

# ---------------------------------------------------------------------------
# FJ9.5a — search-artifact availability audit
# ---------------------------------------------------------------------------

@testset "FJ9.5a search-data availability, tracked not assumed" begin
    dir = joinpath(pkgdir(DuckietownDecisionModels), "artifacts", "fj8")
    items = search_artifact_audit(dir)
    @test length(items) == 8
    @test all(i -> !isempty(i.quantity) && !isempty(i.evidence), items)
    @test all(i -> i.status in (PERSISTED, AGGREGATE_ONLY, ABSENT), items)

    byname(q) = only(filter(i -> occursin(q, i.quantity), items))

    # FJ9.5a found all five ABSENT. FJ9.5b then captured them, and the audit
    # tracks that rather than staying stale — which is the whole point of it
    # being executable. What must never change is that the five move together
    # and only ever by an actual capture.
    tree_fields = ("node ids", "visit counts", "Q estimates", "depths",
        "root action labels")
    statuses = unique(byname(q).status for q in tree_fields)
    @test length(statuses) == 1
    captured = only(statuses) === PERSISTED
    @test only(statuses) in (ABSENT, PERSISTED)
    for q in tree_fields
        if captured
            @test occursin("captured in", byname(q).evidence)
        else
            @test occursin("no search snapshot", byname(q).evidence)
        end
    end

    # what does exist is aggregate, and aggregates cannot determine a tree
    @test byname("aggregate tree counters").status === AGGREGATE_ONLY
    @test byname("PlanningDiagnostics.extra").status === AGGREGATE_ONLY
    @test byname("planner configuration").status === PERSISTED

    # a faithful search figure is possible exactly when a capture exists
    @test search_visualisation_supported(items) == captured

    # the episode artefacts genuinely carry no search columns — checked
    # directly, not just asserted by the audit
    for f in ("six_solver_episodes.csv", "planner_sensitivity_episodes.csv")
        path = joinpath(dir, f)
        isfile(path) || continue
        header = strip.(split(first(eachline(path)), ","))
        for col in ("visits", "q", "value", "depth", "parent", "node",
                    "tree_nodes", "action_nodes", "v_cmd", "omega_cmd")
            @test !(col in header)
        end
    end

    # the aggregate-only verdicts are unaffected by any capture: means over
    # decisions still do not determine a tree
    @test byname("aggregate tree counters").status === AGGREGATE_ONLY
    @test byname("PlanningDiagnostics.extra").status === AGGREGATE_ONLY

    @info "FJ9.5a search data availability\n" * search_audit_table(items)
end

@testset "FJ9.5a a snapshot cannot be reconstructed from aggregates" begin
    # The concrete reason the audit blocks FJ9.5c: aggregate counters are
    # consistent with many different trees, so building one from them would be
    # invention rather than depiction.
    #
    # Two genuinely different trees with IDENTICAL summary statistics:
    a = SearchSnapshot("A", 3, [
        SearchNode(1, 0, 0, 3, 0.0, nothing),
        SearchNode(2, 1, 1, 2, 1.0, FAST_STRAIGHT),
        SearchNode(3, 1, 1, 1, 0.5, BRAKE),
        SearchNode(4, 2, 2, 1, 2.0, SLOW_LEFT)])
    b = SearchSnapshot("B", 3, [
        SearchNode(1, 0, 0, 3, 0.0, nothing),
        SearchNode(2, 1, 1, 2, 1.0, FAST_LEFT),
        SearchNode(3, 1, 1, 1, 0.5, FAST_RIGHT),
        SearchNode(4, 3, 2, 1, 2.0, SLOW_RIGHT)])

    @test check_snapshot(a).ok && check_snapshot(b).ok
    # identical on every aggregate a figure could be built from ...
    @test length(a.nodes) == length(b.nodes)
    @test a.root_visits == b.root_visits
    @test length(root_children(a)) == length(root_children(b))
    @test search_max_depth(a) == search_max_depth(b)
    @test sort([n.visits for n in a.nodes]) == sort([n.visits for n in b.nodes])
    # ... and yet different trees: the depth-2 node hangs off a different parent
    @test a.nodes[4].parent != b.nodes[4].parent
    @test [n.action for n in root_children(a)] != [n.action for n in root_children(b)]
end

# ---------------------------------------------------------------------------
# FJ9.5b — the captured search snapshots
# ---------------------------------------------------------------------------
#
# The capture itself needs MCTS.jl and runs in tools/run_capture_search.sh.
# These tests read the ARTEFACTS, with no planning library installed — which
# is the property that makes FJ9.5c possible: a search figure must be
# reproducible without the solver that produced it.

const FJ95_SNAPSHOTS = joinpath(pkgdir(DuckietownDecisionModels), "artifacts",
    "fj9")

@testset "FJ9.5b snapshots load and validate with no solver installed" begin
    paths = [joinpath(FJ95_SNAPSHOTS, f) for f in
             ("search_snapshot_mcts.json", "search_snapshot_dpw.json")]
    if !all(isfile, paths)
        @test_skip "search snapshots not captured yet"
    else
        # NOTE: "loads without a solver" cannot be asserted here — Pkg.test
        # installs MCTS and FJ8.2 loads it earlier in this same session. The
        # property is proved in a fresh process by tools/fj9_render_check.jl,
        # which reports MCTS_LOADED=false alongside SNAPSHOTS_LOADED=true.

        snaps = load_snapshot.(paths)
        for s in snaps
            chk = check_snapshot(s)
            @test chk.ok
            isempty(chk.issues) || @info "snapshot invalid" s.solver chk.issues
            @test chk.roots == 1
            @test length(s.nodes) == chk.nodes
            @test s.nodes[1].parent == 0
            @test s.nodes[1].action === nothing
            @test s.nodes[1].value === missing      # not back-computed
            @test all(n -> n.visits >= 0, s.nodes)
            @test s.planner_seed == 2026
            @test !isempty(s.state_fingerprint)
            @test !isempty(s.config_fingerprint)
            @test s.selected_action !== nothing
            # the selected action really is one the search considered
            @test any(c -> c.action == s.selected_action, root_children(s))
        end

        # both planners searched the SAME latent state; only the action
        # representation differs
        @test snaps[1].state_fingerprint == snaps[2].state_fingerprint
        @test snaps[1].config_fingerprint != snaps[2].config_fingerprint
        @test snapshot_fingerprint(snaps[1]) != snapshot_fingerprint(snaps[2])

        # action types match their models
        mcts, dpw = snaps
        @test all(c -> c.action isa MacroAction, root_children(mcts))
        @test all(c -> c.action isa DuckieAction, root_children(dpw))
        @test all(c -> 0.0 <= c.action.v <= 0.41 &&
                       -1.5 <= c.action.omega <= 1.5, root_children(dpw))
    end
end

@testset "FJ9.5b what the two searches actually did" begin
    paths = [joinpath(FJ95_SNAPSHOTS, f) for f in
             ("search_snapshot_mcts.json", "search_snapshot_dpw.json")]
    if !all(isfile, paths)
        @test_skip "search snapshots not captured yet"
    else
        mcts, dpw = load_snapshot.(paths)
        m_children = root_children(mcts)
        d_children = root_children(dpw)

        # MCTS searches SEVEN distinct actions and concentrates on some
        @test mcts.extra.distinct_root_actions == 7
        @test length(unique(c.action for c in m_children)) == 7
        @test maximum(c.visits for c in m_children) > 10

        # DPW samples ~4*sqrt(N) distinct continuous actions and spreads thin
        @test dpw.extra.distinct_root_actions == length(d_children)
        @test length(unique((c.action.v, c.action.omega) for c in d_children)) ==
            length(d_children)
        @test maximum(c.visits for c in d_children) <= 5

        # the quantitative reason DPW behaves near-randomly at this budget:
        # no action accumulates evidence
        @info "FJ9.5b root evidence per action" mcts_actions =
            length(unique(c.action for c in m_children)) mcts_visits =
            extrema(c.visits for c in m_children) dpw_actions =
            length(d_children) dpw_visits = extrema(c.visits for c in d_children)
        @test maximum(c.visits for c in m_children) >
              maximum(c.visits for c in d_children)
    end
end

@testset "FJ9.5b a modified snapshot is refused" begin
    path = joinpath(FJ95_SNAPSHOTS, "search_snapshot_dpw.json")
    if !isfile(path)
        @test_skip "search snapshots not captured yet"
    else
        text = read(path, String)
        dir = mktempdir()

        good = joinpath(dir, "good.json")
        write(good, text)
        @test load_snapshot(good) isa SearchSnapshot

        # change one visit count without updating the stored fingerprint
        tampered = joinpath(dir, "tampered.json")
        write(tampered, replace(text, "\"visits\": 1" => "\"visits\": 99";
            count=1))
        @test_throws ArgumentError load_snapshot(tampered)

        # an unknown schema is refused rather than guessed at
        wrong = joinpath(dir, "wrong.json")
        write(wrong, replace(text, "fj9.5b-search-snapshot-1" => "something-else"))
        @test_throws ArgumentError load_snapshot(wrong)

        @test_throws ArgumentError load_snapshot(joinpath(dir, "absent.json"))
    end
end

# ---------------------------------------------------------------------------
# FJ9.5c / FJ9.5d — search statistics and display filtering
# ---------------------------------------------------------------------------
#
# The renderers themselves need Makie and are exercised by
# tools/fj9_render_check.jl in a fresh process (which also proves they run
# with MCTS_LOADED=false). What is tested here is everything the figures are
# built from, with no backend installed.

@testset "FJ9.5c statistics are per action, not per child node" begin
    # A vanilla MCTS tree has one root child per simulation because states
    # never merge (FJ8.2), so several children repeat the same action and the
    # same action-node visit count. Counting children would report that count
    # many times over.
    nodes = [SearchNode(1, 0, 0, 10, missing, nothing),
        SearchNode(2, 1, 1, 8, 1.0, FAST_STRAIGHT),
        SearchNode(3, 1, 1, 8, 1.0, FAST_STRAIGHT),   # same action, repeated
        SearchNode(4, 1, 1, 8, 1.0, FAST_STRAIGHT),
        SearchNode(5, 1, 1, 1, -2.0, BRAKE),
        SearchNode(6, 1, 1, 1, -3.0, FAST_LEFT)]
    s = SearchSnapshot("t", 10, nodes; selected_action=FAST_STRAIGHT)
    st = search_statistics(s)

    @test st.root_children == 5          # five child nodes ...
    @test st.root_actions == 3           # ... but three actions
    @test st.distinct_actions == 3
    @test st.max_visits == 8
    @test st.median_visits == 1
    @test st.mean_visits ≈ (8 + 1 + 1) / 3
    @test st.single_visit_fraction ≈ 2 / 3

    # a degenerate snapshot does not throw
    empty_ = SearchSnapshot("t", 0, SearchNode[])
    @test search_statistics(empty_).root_actions == 0
end

@testset "FJ9.5c display filters never mutate the snapshot" begin
    paths = [joinpath(FJ95_SNAPSHOTS, f) for f in
             ("search_snapshot_mcts.json", "search_snapshot_dpw.json")]
    if !all(isfile, paths)
        @test_skip "search snapshots not captured yet"
    else
        s = load_snapshot(paths[1])
        before = snapshot_fingerprint(s)
        n_before = length(s.nodes)

        all_ids = visible_nodes(s)
        @test length(all_ids) == n_before
        @test all_ids == collect(1:n_before)

        shallow = visible_nodes(s; max_depth=1)
        @test length(shallow) < n_before
        @test all(id -> s.nodes[id].depth <= 1, shallow)
        @test 1 in shallow                      # the root is always kept

        busy = visible_nodes(s; min_visits=5)
        @test all(id -> id == 1 || s.nodes[id].visits >= 5, busy)

        top = visible_nodes(s; top_k=3, max_depth=1)
        @test length(top) <= 4                  # root + at most 3 children

        # a filtered subtree stays connected: every kept non-root node has a
        # kept parent
        for ids in (shallow, busy, top)
            kept = Set(ids)
            for id in ids
                s.nodes[id].parent == 0 && continue
                @test s.nodes[id].parent in kept
            end
        end

        # and the evidence is untouched by any of it
        @test snapshot_fingerprint(s) == before
        @test length(s.nodes) == n_before
    end
end

@testset "FJ9.5d the value label matches what MCTS.jl computes" begin
    paths = [joinpath(FJ95_SNAPSHOTS, f) for f in
             ("search_snapshot_mcts.json", "search_snapshot_dpw.json")]
    if !all(isfile, paths)
        @test_skip "search snapshots not captured yet"
    else
        for p in paths
            s = load_snapshot(p)
            # the semantics were read out of MCTS.jl's source before any
            # colorbar was labelled: vanilla.jl:274 and dpw.jl:151 both do
            # q += (return - q)/n, an incremental sample mean
            @test haskey(s.extra, :value_semantics)
            sem = String(s.extra.value_semantics)
            @test occursin("mean", sem)
            @test occursin("init_Q", sem)      # the seeding is disclosed
            @test !occursin("total", sem)
            @test occursin(sem, search_summary(s)) ||
                  occursin("mean of backed-up returns", search_summary(s))
        end
    end
end

@testset "FJ9.5d the continuous action plane draws only sampled actions" begin
    path = joinpath(FJ95_SNAPSHOTS, "search_snapshot_dpw.json")
    if !isfile(path)
        @test_skip "search snapshots not captured yet"
    else
        s = load_snapshot(path)
        rc = root_children(s)
        st = search_statistics(s)

        # every plotted point is an action the search really sampled
        @test all(c -> c.action isa DuckieAction, rc)
        @test length(rc) == st.root_actions
        @test length(unique((c.action.v, c.action.omega) for c in rc)) ==
            length(rc)

        # inside the reference box, and the box is not a rescaling of the data
        @test all(c -> 0.0 <= c.action.v <= 0.41, rc)
        @test all(c -> -1.5 <= c.action.omega <= 1.5, rc)
        # the samples do not fill the box: a smoothed surface would imply they
        # did, which is why the renderer does not interpolate
        vs = [c.action.v for c in rc]
        @test length(vs) < 100

        # visits come straight from the snapshot
        @test all(c -> c.visits >= 1, rc)
        @test maximum(c.visits for c in rc) == st.max_visits

        # the highlighted action is one the captured search selected
        @test s.selected_action isa DuckieAction
        @test any(c -> c.action == s.selected_action, rc)

        # the concentration statistic the figure reports
        @test 0.0 <= st.single_visit_fraction <= 1.0
        @info "FJ9.5d action-plane statistics" actions = st.root_actions visits_mean =
            round(st.mean_visits; digits=2) visits_max = st.max_visits once_only =
            round(100 * st.single_visit_fraction; digits=1)
    end
end
