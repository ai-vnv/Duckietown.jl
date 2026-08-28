# FJ5: live reference-runtime integration and matched-state one-step parity.
#
# Unlike the FJ3 fixtures (recorded offline), these tests run the REAL Python
# reference stack (`ddm-ref`) and the native Julia model side by side in the
# same test process, with the reference simulator LOADED with exactly the
# state the Julia model starts from — so no initial-condition or accumulation
# confound remains. Accumulated drift over a free-running episode is FJ6.
#
# The whole set skips (never fails) when the reference environment is absent,
# so the package remains testable on machines without WSL + `ddm-ref`.

using DuckietownDecisionModels
using POMDPs
using Test
using Random

# Julia 1.10.11 on Windows crashes the RUNTIME while compiling this test set:
# `EXCEPTION_ACCESS_VIOLATION` in `gc_mark_stack`, raised from the compiler's
# own inlining pass. It reproduces in a standalone script as well as under
# `@testset`, survived both a process-handle fix and rewriting the only
# closure involved, and disappears entirely on 1.11.3 (where the whole set
# passes). It is a toolchain defect, not a parity result, so the set is gated
# rather than silently weakened — run these tests on Julia >= 1.11.
const FJ5_JULIA_OK = VERSION >= v"1.11"
const FJ5_AVAILABLE = FJ5_JULIA_OK && reference_backend_available()

if !FJ5_JULIA_OK
    @info "FJ5: skipped on Julia $(VERSION) — the live parity set needs " *
        "Julia >= 1.11 (1.10.11 Windows GC crash, see docs/validation/FJ5_STATUS.md)"
elseif !FJ5_AVAILABLE
    @info "FJ5: reference backend unavailable (needs WSL + ddm-ref) — skipping"
end

FJ5_AVAILABLE && @testset "FJ5.1 reference backend lifecycle" begin
    ref = ReferenceBackend("q_learning"; seed=53)
    try
        @test ref.info[:config] == "q_learning"
        @test ref.info[:frame_skip] == 6
        @test ref.info[:max_steps] == 1500
        @test ref.info[:tile_size] == 0.585
        @test ref.info[:n_signs] == 1
        @test ref_call(ref, Dict("cmd" => "ping")).pong == true

        w, dump = ref_reset!(ref, 53)
        @test w isa DuckieWorldState
        @test w.ego.step_count == 0
        @test w.ego.timestamp == 0.0
        @test isempty(w.ego.command_history)
        @test length(w.ducks) == 1
        @test length(w.stop_signs) == 1
        @test !w.stop_memory.sigma_stop && w.stop_memory.hold_steps == 0
        @test w.crossings_started == [0] && w.crossing_armed == [true]
        # the reference reset is seed-reproducible through the backend
        w2, _ = ref_reset!(ref, 53)
        @test w2.ego.pos == w.ego.pos && w2.ego.angle == w.ego.angle
    finally
        close(ref)
    end
end

FJ5_AVAILABLE && @testset "FJ5.2 state bridge round-trips exactly" begin
    ref = ReferenceBackend("q_learning"; seed=53)
    try
        w, _ = ref_reset!(ref, 53)
        # drive a few decisions so the delay window, duckie and memories are
        # non-trivial, then export -> inject -> export again
        for a in (FAST_STRAIGHT, SLOW_LEFT, BRAKE, FAST_RIGHT, SLOW_STRAIGHT)
            w, _ = ref_step!(ref, a)
        end
        @test w.ego.step_count == 30
        @test !isempty(w.ego.command_history)

        ref_set_state!(ref, w)
        w2, _ = ref_get_state(ref)
        @test w2.ego.pos == w.ego.pos
        @test w2.ego.angle == w.ego.angle
        @test w2.ego.speed == w.ego.speed
        @test w2.ego.step_count == w.ego.step_count
        @test w2.ego.timestamp == w.ego.timestamp
        @test w2.ego.q0 == w.ego.q0
        @test w2.ego.v0 == w.ego.v0
        @test w2.ego.axis_left_rad == w.ego.axis_left_rad
        @test w2.ego.axis_right_rad == w.ego.axis_right_rad
        @test w2.ego.command_history == w.ego.command_history
        @test w2.ducks[1].center == w.ducks[1].center
        @test w2.ducks[1].obj_corners == w.ducks[1].obj_corners
        @test w2.ducks[1].obj_norm == w.ducks[1].obj_norm
        @test w2.ducks[1].pedestrian_active == w.ducks[1].pedestrian_active
        @test w2.ducks[1].pedestrian_wait_time == w.ducks[1].pedestrian_wait_time
        @test w2.crossings_started == w.crossings_started
        @test w2.crossing_armed == w.crossing_armed
        @test w2.stop_memory.sigma_stop == w.stop_memory.sigma_stop
        @test w2.stop_memory.hold_steps == w.stop_memory.hold_steps
        @test w2.stop_memory.last_stop_id == w.stop_memory.last_stop_id
        @test w2.lane_fallback == w.lane_fallback
        # zero diffs through the comparison machinery as well
        @test all(d -> d.ulps == 0, compare_worlds(w, w2))

        # injecting a state the reference never produced (a Julia-sampled one)
        mdp = DuckietownMDP(joinpath(pkgdir(DuckietownDecisionModels), "..",
            "duckduck", "policies", "q_learning", "training_config.yaml"))
        js = rand(MersenneTwister(7), initialstate(mdp))
        ref_set_state!(ref, js)
        back, _ = ref_get_state(ref)
        @test back.ego.pos == js.ego.pos
        @test back.ego.angle == js.ego.angle
        @test back.ego.q0 == js.ego.q0
        @test back.ducks[1].center == js.ducks[1].center
    finally
        close(ref)
    end
end

FJ5_AVAILABLE && @testset "FJ5.3 matched-state one-step parity (discrete)" begin
    cfg_path = joinpath(pkgdir(DuckietownDecisionModels), "..", "duckduck",
        "policies", "q_learning", "training_config.yaml")
    mdp = DuckietownMDP(cfg_path)
    ref = ReferenceBackend("q_learning"; seed=53, map=mdp.map)
    try
        w, _ = ref_reset!(ref, 53)

        # A. branching: all 7 macro actions from the SAME latent state
        reports = matched_state_sweep(ref, mdp.transition, w,
            ALL_MACRO_ACTIONS)
        s = parity_summary(reports)
        @test s.steps == 7
        @test s.accepted == 7
        @test isempty(s.discrete_mismatches)
        # structural criterion: nothing outside the libm-derived chain rooted
        # at the atan2 pose readback may deviate at all, and the deviations
        # that do occur stay inside the measured bounds
        # (see LIBM_DERIVED_FIELDS)
        @test issubset(Set(s.nonzero_fields), LIBM_DERIVED_FIELDS)
        @test s.max_ulps <= LIBM_MAX_ULPS
        @test s.max_absdiff <= LIBM_MAX_ABSDIFF
        # numerically-equal-but-different-bits entries are confined to the
        # inert se(2) diagonal signed zeros
        @test issubset(Set(s.bitwise_only_fields), SIGNED_ZERO_FIELDS)
        for r in reports
            @test parity_accepted(r)
            @test length(r.diffs) >= 80
        end
        # the dynamical state itself is bit-identical for every action:
        # q0/v0 carry the DB18 state and never differ
        for r in reports
            @test all(d -> d.ulps == 0,
                filter(d -> startswith(d.name, "ego.q0") ||
                    startswith(d.name, "ego.v0") ||
                    startswith(d.name, "ego.pos") ||
                    startswith(d.name, "duck1."), r.diffs))
        end

        # B. walking a trajectory, every step still matched-state
        rng = MersenneTwister(4)
        traj = [rand(rng, ALL_MACRO_ACTIONS) for _ in 1:40]
        reports = matched_state_sweep(ref, mdp.transition, w, traj;
            rng=MersenneTwister(4), advance=true)
        s = parity_summary(reports)
        @test s.steps == 40
        @test s.accepted == 40
        @test issubset(Set(s.nonzero_fields), LIBM_DERIVED_FIELDS)
        @test s.max_ulps <= LIBM_MAX_ULPS
        @test s.max_absdiff <= LIBM_MAX_ABSDIFF
        @test isempty(s.discrete_mismatches)

        # C. a Julia-sampled initial state (never seen by the reference)
        js = rand(MersenneTwister(21), initialstate(mdp))
        reports = matched_state_sweep(ref, mdp.transition, js,
            [FAST_STRAIGHT, BRAKE, SLOW_LEFT, FAST_RIGHT]; advance=true)
        s = parity_summary(reports)
        @test s.accepted == s.steps == 4
        @test issubset(Set(s.nonzero_fields), LIBM_DERIVED_FIELDS)

        # D. attribution of the one tolerated field: on turning actions the
        # dynamical state (q0/v0) is bit-identical and recomputing atan2 in
        # Julia on the reference's own q0 reproduces the Julia angle exactly
        ref_set_state!(ref, w)
        refw, _ = ref_step!(ref, FAST_LEFT)
        jl = simulate_decision(mdp.transition, w, FAST_LEFT, MersenneTwister(0))
        @test jl.sp.ego.q0 == refw.ego.q0
        @test jl.sp.ego.v0 == refw.ego.v0
        @test atan(refw.ego.q0[2, 1], refw.ego.q0[1, 1]) == jl.sp.ego.angle
        @test abs(reinterpret(Int64, jl.sp.ego.angle) -
                  reinterpret(Int64, refw.ego.angle)) <= 1
    finally
        close(ref)
    end
end

FJ5_AVAILABLE && @testset "FJ5.3 matched-state one-step parity (continuous)" begin
    cfg_path = joinpath(pkgdir(DuckietownDecisionModels), "..", "duckduck",
        "policies", "sac", "training_config.yaml")
    mdp = DuckietownMDP(cfg_path; action_space=:continuous)
    ref = ReferenceBackend("sac"; seed=73, action_space=:continuous,
        map=mdp.map)
    try
        w, _ = ref_reset!(ref, 73)
        # action-space key points, including an out-of-range clip case
        acts = [DuckieAction(0.0, 0.0), DuckieAction(0.17, 0.0),
            DuckieAction(0.41, 0.0), DuckieAction(0.17, 1.5),
            DuckieAction(0.17, -1.5), DuckieAction(0.41, 1.5),
            DuckieAction(0.30, 0.7), DuckieAction(0.60, 2.5)]
        reports = matched_state_sweep(ref, mdp.transition, w, acts)
        s = parity_summary(reports)
        @test s.steps == length(acts)
        @test s.accepted == length(acts)
        @test issubset(Set(s.nonzero_fields), LIBM_DERIVED_FIELDS)
        @test s.max_ulps <= LIBM_MAX_ULPS
        @test s.max_absdiff <= LIBM_MAX_ABSDIFF
        @test isempty(s.discrete_mismatches)
        # the continuous projection was actually compared
        @test any(d -> startswith(d.name, "cont."), reports[1].diffs)

        # a driven trajectory (steering reward uses pre-action kappa)
        rng = MersenneTwister(9)
        traj = [DuckieAction(0.25, 1.2 * (rand(rng) - 0.5)) for _ in 1:30]
        reports = matched_state_sweep(ref, mdp.transition, w, traj;
            advance=true)
        s = parity_summary(reports)
        @test s.steps == 30
        @test s.accepted == 30
        @test issubset(Set(s.nonzero_fields), LIBM_DERIVED_FIELDS)
        @test s.max_ulps <= LIBM_MAX_ULPS
    finally
        close(ref)
    end
end
