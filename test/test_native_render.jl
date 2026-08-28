# Native lookalike renderer — the PURE half (src/visualization/native_render.jl).
#
# Everything here runs without a plotting backend and without a display. The
# asset-dependent parts guard themselves on the reference asset tree being
# present, the same honest-degradation pattern the reference-dependent suite
# files use.

@testset "native render: pure layer" begin
    @testset "the honesty note is load-bearing" begin
        # captions and docs quote this constant; the words that make it a
        # disclaimer must not drift
        @test occursin("lookalike", NATIVE_RENDER_NOTE)
        @test occursin("not parity evidence", NATIVE_RENDER_NOTE)
    end

    @testset "camera constants match simulator.py" begin
        @test NATIVE_CAMERA_FOV_Y == 75.0
        @test NATIVE_CAMERA_FLOOR_DIST == 0.108
        @test NATIVE_CAMERA_FORWARD_DIST == 0.066
    end

    @testset "tile texture mapping" begin
        spec(kind, deg) = TileSpec(kind, deg, kind in
            (:straight, :curve_left, :curve_right), Vector{NTuple{3,Float64}}[])
        # straights follow the map angle directly
        for deg in (0.0, 90.0, 180.0, 270.0)
            f, rot = tile_texture_file(spec(:straight, deg))
            @test f == "straight_1.jpg"
            @test rot == mod(round(Int, deg / 90), 4)
        end
        # curves carry the calibrated half-turn
        for deg in (0.0, 90.0, 180.0, 270.0)
            f, rot = tile_texture_file(spec(:curve_left, deg))
            @test f == "curve_left_1.png"
            @test rot == mod(round(Int, deg / 90) + 2, 4)
        end
        @test tile_texture_file(spec(:asphalt, 0.0))[1] == "asphalt_1.jpg"
        @test tile_texture_file(spec(:grass, 0.0))[1] == "grass_1.jpg"
    end

    @testset "missing assets fail loudly, with instructions" begin
        # an explicit bogus override must not fall through to a guess
        withenv("DUCKIETOWN_ASSETS" => joinpath(tempdir(), "nope")) do
            if !isdir(joinpath(homedir(),
                "miniconda3/envs/ddm-ref/lib/python3.9/site-packages",
                "duckietown_world/data/gd1"))
                err = try
                    duckietown_assets_root()
                    nothing
                catch e
                    e
                end
                @test err isa ArgumentError
                @test occursin("DUCKIETOWN_ASSETS", err.msg)
            else
                # the conventional location exists on this machine, so the
                # resolver legitimately falls back to it
                @test isdir(duckietown_assets_root())
            end
        end
    end

    assets = try
        duckietown_assets_root()
    catch
        nothing
    end
    if assets === nothing
        @info "native render: reference assets not present — skipping the OBJ/scene tests (expected off the development machine)"
    else
        @testset "OBJ/MTL reader on the reference meshes" begin
            # duckie.obj: no MTL, partial uv indexing — the case MeshIO rejects
            duck = load_obj_groups(joinpath(assets, "meshes", "duckie", "duckie.obj"))
            @test length(duck) == 1
            @test duck[1].texture === nothing        # bare OBJ: no material file
            @test !isempty(duck[1].faces)
            @test length(duck[1].points) == length(duck[1].uvs)
            @test all(f -> all(1 .<= f .<= length(duck[1].points)), duck[1].faces)

            # sign_stop.obj: three materials, two textured (one via ../ path)
            sign = load_obj_groups(joinpath(assets, "meshes", "signs",
                "sign_stop", "sign_stop.obj"))
            @test length(sign) == 3
            @test count(g -> g.texture !== nothing, sign) == 2
            @test any(g -> g.texture !== nothing &&
                           endswith(g.texture, "sign_stop.png"), sign)
            @test any(g -> g.texture !== nothing &&
                           endswith(g.texture, "wood_osb.jpg"), sign)
        end

        @testset "native_world of a stop-and-duck spawn" begin
            mdp = DuckietownMDP(scenario_config(:stop_and_duck_safe);
                action_space = :discrete)
            w = rand(MersenneTwister(1002), initialstate(mdp))
            nw = native_world(w; assets = assets)
            @test nw.tile_size == w.map.tile_size
            @test length(nw.tiles) == 9                 # the full 3x3 grid
            @test all(isfile(t[3]) for t in nw.tiles)   # every texture resolves
            # one visible duck + one stop sign
            @test length(nw.objects) == 2
            # the duck fallback texture was attached (bare OBJ)
            @test any(o -> any(g -> g.texture !== nothing &&
                                    endswith(g.texture, "duckie.png"), o.groups),
                      nw.objects)
            # ego camera at the reference height, ahead of the axle
            @test nw.ego_eye[2] == NATIVE_CAMERA_FLOOR_DIST
            @test nw.fov == NATIVE_CAMERA_FOV_Y
            dx = nw.ego_eye[1] - w.ego.pos[1]
            dz = nw.ego_eye[3] - w.ego.pos[3]
            @test hypot(dx, dz) ≈ NATIVE_CAMERA_FORWARD_DIST atol = 1e-9
        end
    end
end
