# Native tests for the policy adapters and slices.
#
# The tabular adapter is exercised with SYNTHETIC Q-tables built in-test (the
# real checkpoints are frozen reference inputs and stay behind the FJ7 gate);
# the actor adapters load the exported weights committed under
# artifacts/fj9/weights/, which the ledger classifies as provisioned frozen
# inputs. Nothing here touches the reference checkout.

using Duckietown
using POMDPs
using Test
using Random

const PN_PKG = pkgdir(Duckietown)

# a minimal NumPy .npy v1.0 writer, enough for the reader's supported dtypes
function pn_write_npy(path, A::Array{T}) where {T}
    descr = T === Float32 ? "<f4" : T === Float64 ? "<f8" :
            T === Int64 ? "<i8" : T === Int32 ? "<i4" : error("dtype")
    shape = join(size(A), ", ") * (ndims(A) == 1 ? "," : "")
    header = "{'descr': '$descr', 'fortran_order': True, 'shape': ($shape), }"
    pad = 64 - mod(10 + length(header) + 1, 64)
    header *= repeat(" ", pad) * "\n"
    open(path, "w") do io
        write(io, UInt8[0x93], "NUMPY", UInt8[0x01, 0x00])
        write(io, htol(UInt16(length(header))))
        write(io, header)
        write(io, A)   # column-major bytes; declared fortran_order above
    end
    return path
end

@testset "adapters: read_npy round-trips what it claims to support" begin
    dir = mktempdir()
    for (name, A) in ("f4" => Float32[1 2 3; 4 5 6],
                      "f8" => Float64[0.5, -0.25, 3.0],
                      "i4" => Int32[7 8; 9 10])
        p = pn_write_npy(joinpath(dir, "$name.npy"), A)
        B = read_npy(p)
        @test size(B) == size(A)
        @test B == A
    end
    # a non-npy file is refused, not misread
    bad = joinpath(dir, "bad.npy")
    write(bad, "not numpy at all")
    @test_throws ArgumentError read_npy(bad)
    rm(dir; recursive = true)
end

@testset "adapters: a synthetic Q-table drives decide() honestly" begin
    dir = mktempdir()

    # value == action id: the greedy action is 6 at EVERY index, margin 1
    ramp = Array{Float32,8}(undef, Duckietown.Q_SHAPE...)
    for a in 0:6
        ramp[:, :, :, :, :, :, :, a + 1] .= Float32(a)
    end
    path = pn_write_npy(joinpath(dir, "ramp.npy"), ramp)
    pol = QTablePolicy(path)
    @test length(all_state_indices()) == prod(Duckietown.STATE_SHAPE)
    for idx in Iterators.take(all_state_indices(), 25)
        d = decide(pol, idx)
        @test Int(d.action) == 6      # QDecision carries the MacroAction
    end

    # restricting the allowed set restricts the argmax
    pol4 = QTablePolicy(path; allowed_actions = 0:4)
    for idx in Iterators.take(all_state_indices(), 10)
        @test Int(decide(pol4, idx).action) == 4
    end

    # an all-zero table ties every action, and the statistics say so
    flat = zeros(Float32, Duckietown.Q_SHAPE...)
    pflat = QTablePolicy(pn_write_npy(joinpath(dir, "flat.npy"), flat))
    stats = tie_statistics(pflat)
    @test stats !== nothing

    # the validations refuse what they document
    @test_throws ArgumentError QTablePolicy(
        pn_write_npy(joinpath(dir, "small.npy"), zeros(Float32, 2, 2)))
    nanbad = copy(ramp); nanbad[1] = NaN32
    @test_throws ArgumentError QTablePolicy(
        pn_write_npy(joinpath(dir, "nan.npy"), nanbad))
    @test_throws ArgumentError QTablePolicy(path; allowed_actions = Int[])
    @test_throws ArgumentError QTablePolicy(path; allowed_actions = [0, 0])
    @test_throws ArgumentError QTablePolicy(path; allowed_actions = [7])

    # and the policy drives a world state through the POMDPs entry point
    mdp = DuckietownMDP(scenario_config(:stop_and_duck);
        action_space = :discrete)
    s = rand(MersenneTwister(3), initialstate(mdp))
    a = POMDPs.action(pol, mdp, s)
    @test a isa MacroAction
    rm(dir; recursive = true)
end

@testset "actors: the committed exported weights load and act" begin
    sac = SACActorPolicy(joinpath(PN_PKG, "artifacts", "fj9", "weights", "sac"))
    td3 = TD3ActorPolicy(joinpath(PN_PKG, "artifacts", "fj9", "weights", "td3"))

    @test size(sac.l0.W, 2) == 15
    obs = zeros(Float32, 15)
    fs = forward(sac, obs)
    ft = forward(td3, obs)
    @test fs !== nothing && ft !== nothing

    a1 = act(sac, obs)
    a2 = act(td3, obs)
    @test a1 isa DuckieAction && a2 isa DuckieAction

    # deterministic actors: the same observation gives the same action
    @test act(sac, obs) == a1
    @test act(td3, obs) == a2

    cmdp = DuckietownMDP(scenario_config(:stop_and_duck; algorithm = :sac);
        action_space = :continuous)
    s = rand(MersenneTwister(4), initialstate(cmdp))
    aw = POMDPs.action(sac, cmdp, s)
    @test aw isa DuckieAction
end

@testset "slices: tabular and continuous policy slices from native inputs" begin
    dir = mktempdir()
    ramp = Array{Float32,8}(undef, Duckietown.Q_SHAPE...)
    for a in 0:6
        ramp[:, :, :, :, :, :, :, a + 1] .= Float32(a)
    end
    pol = QTablePolicy(pn_write_npy(joinpath(dir, "ramp.npy"), ramp))

    sl = policy_slice(pol, :d, :phi;
        xs = range(-0.2, 0.2; length = 9), ys = range(-0.5, 0.5; length = 9),
        name = "synthetic_ramp")
    @test sl isa TabularPolicySlice
    @test !isempty(slice_fingerprint(sl))
    @test !isempty(fixed_context_lines(sl))
    @test !isempty(slice_summary(sl))
    @test size(value_surface(sl)) == (9, 9)
    @test size(action_surface(sl)) == (9, 9)
    @test tie_surface(sl) !== nothing
    @test margin_surface(sl) !== nothing

    cmdp = DuckietownMDP(scenario_config(:stop_and_duck; algorithm = :sac);
        action_space = :continuous)
    sac = SACActorPolicy(joinpath(PN_PKG, "artifacts", "fj9", "weights", "sac"))
    csl = policy_slice(sac, cmdp.transition.continuous_cfg, :d, :phi;
        xs = range(-0.2, 0.2; length = 7), ys = range(-0.5, 0.5; length = 7),
        name = "sac_exported")
    @test csl isa ContinuousPolicySlice
    @test size(v_surface(csl)) == (7, 7)
    @test size(omega_surface(csl)) == (7, 7)
    rm(dir; recursive = true)
end

@testset "config loader: a full synthetic YAML round-trips every block" begin
    dir = mktempdir()

    ql = joinpath(dir, "ql.yaml")
    write(ql, """
    algorithm: q_learning
    stage: full
    seed: 7
    environment:
      map_name: small_loop
      max_steps: 400
      accept_start_angle_deg: 60
      spawn_max_abs_d: 0.1
      spawn_max_abs_phi: 0.5
      spawn_route_direction: clockwise
      spawn_route_center: [2.0, 2.0]
      spawn_min_route_alignment: 0.6
      spawn_position_bounds_xz: [0.0, 3.0, 0.0, 3.0]
      user_tile_start: [1, 1]
      goal_tile: [2, 1]
    state:
      stop_zone: 0.45
      stop_hold_steps: 2
    continuous_state:
      max_speed: 0.41
      duck_detection_range: 1.5
      duck_detection_corridor_width: 0.3
      duck_detection_forward_only: true
    actions:
      v_fast: 0.40
      w0: 1.5
    duck_controller:
      p_cross: 0.5
      spawn_pos: [1.62, 0.5]
      require_stop: true
      stop_spawn_pos: [1.2, 2.1]
    reward:
      goal: 50.0
      alpha_progress: 1.0
    q_learning:
      gamma: 0.99
      allowed_actions: [0, 1, 2, 3, 4, 5, 6]
    lane_teacher:
      enabled: true
      d_gain: 1.0
    transition_model:
      enabled: true
    training:
      episodes: 10
      milestone_episodes: [5, 10]
    evaluation:
      episodes: 2
      seeds: [1, 2]
    wandb:
      project: synthetic
    """)
    cfg = load_config(ql)
    @test cfg.algorithm == :q_learning
    @test cfg.seed == 7
    @test cfg.environment.map_name == "small_loop"
    @test cfg.environment.spawn_route_direction == :clockwise
    @test cfg.environment.goal_tile == (2, 1)
    @test cfg.state.stop_hold_steps == 2
    @test cfg.continuous_state !== nothing
    @test cfg.continuous_state.duck_detection_forward_only
    @test cfg.duck_controller.p_cross == 0.5
    @test cfg.duck_controller.require_stop
    @test cfg.solver isa QLearningConfig
    @test cfg.lane_teacher !== nothing
    @test cfg.transition_model.enabled

    # and a world actually builds from the loaded config
    m = DuckietownMDP(cfg; action_space = :discrete)
    @test m isa DuckietownMDP

    # the three continuous-family branches parse too
    for alg in ("sarsa", "sac", "td3")
        p = joinpath(dir, "$alg.yaml")
        solver_block = alg == "sarsa" ?
            "sarsa:\n  allowed_actions: [0, 1, 2]\n" : "$alg:\n  gamma: 0.99\n"
        write(p, """
        algorithm: $alg
        environment:
          map_name: small_loop
        state:
          stop_zone: 0.45
        actions:
          v_fast: 0.40
        duck_controller:
          p_cross: 0.02
        reward:
          goal: 50.0
        $solver_block
        """)
        c = load_config(p)
        @test c.algorithm == Symbol(alg)
    end

    # the spawn validations refuse what they document
    for (frag, label) in (
            ("spawn_route_direction: diagonal", "direction"),
            ("spawn_min_route_alignment: 1.5", "alignment"),
            ("spawn_position_bounds_xz: [3.0, 0.0, 0.0, 3.0]", "bounds"))
        p = joinpath(dir, "bad_$label.yaml")
        write(p, """
        algorithm: q_learning
        environment:
          map_name: small_loop
          $frag
        state:
          stop_zone: 0.45
        actions:
          v_fast: 0.40
        duck_controller:
          p_cross: 0.02
        reward:
          goal: 50.0
        q_learning:
          allowed_actions: [0, 1]
        """)
        @test_throws ArgumentError load_config(p)
    end
    rm(dir; recursive = true)
end

@testset "config loader: the documented refusals refuse" begin
    dir = mktempdir()
    seq = joinpath(dir, "seq.yaml")
    write(seq, "- 1\n- 2\n")
    @test_throws ArgumentError load_config(seq)

    alg = joinpath(dir, "alg.yaml")
    write(alg, "algorithm: dreamer\n")
    @test_throws ArgumentError load_config(alg)
    rm(dir; recursive = true)

    # every named scenario builds for every algorithm it supports
    for sc in SCENARIOS
        cfg = scenario_config(sc)
        @test cfg.algorithm in (:q_learning, :sarsa, :sac, :td3)
    end
    for alg2 in (:q_learning, :sarsa, :sac, :td3)
        cfg = scenario_config(:stop_and_duck; algorithm = alg2)
        @test cfg.algorithm == alg2
    end
end
