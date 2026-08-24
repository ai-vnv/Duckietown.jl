# FJ6: free-running rollout parity (regression guard).
#
# The full experiment lives in `tools/parity/run_fj6_rollout.jl` and writes
# `artifacts/fj6/`. This test set is the short regression version: same
# machinery, fewer decisions, asserting the properties that define the gate.
#
# What must hold (and what deliberately need not):
#   MUST  no discrete/semantic divergence at any decision
#   MUST  identical termination decision and reason
#   MUST  identical event timing (difference 0 for every event that fires)
#   MUST  returns agree to the measured libm-scale bound
#   MUST  any q0/v0 divergence stay at libm scale (never a real drift)
#   MAY   pose/observer/reward values differ at ~1e-16 (Type 1 readback)
#
# Uses the PROCESS backend: after the FJ5-R interposition finding, that is the
# numerical reference of record.

using DuckietownDecisionModels
using POMDPs
using Test
using Random

const FJ6_OK = VERSION >= v"1.11" && reference_backend_available()

if !FJ6_OK
    @info "FJ6: skipped (needs Julia >= 1.11 + the ddm-ref reference env)"
end

# Bounds are the FJ5/FJ6 measured libm scale, not invented tolerances:
# worst observed |Δq0| = 1.11e-16, |Δreturn| = 3.55e-15 over 539 free-running
# decisions across five trajectories.
const FJ6_MAX_DQ = 1e-14
const FJ6_MAX_DRETURN = 1e-12

FJ6_OK && @testset "FJ6 free-running rollout parity (discrete)" begin
    mdp = DuckietownMDP(joinpath(pkgdir(DuckietownDecisionModels), "..",
        "duckduck", "policies", "q_learning", "training_config.yaml"))
    model = mdp.transition
    ref = ProcessReferenceBackend("q_learning"; seed=53, map=mdp.map)
    try
        x0, _ = ref_reset!(ref, 53)

        # a fixed action sequence, replayed identically by both runs
        rng = MersenneTwister(11)
        actions = [rand(rng, ALL_MACRO_ACTIONS) for _ in 1:60]

        rec_ref = rollout_reference(ref, x0, actions)
        rec_jl = rollout_native(model, x0, actions)

        @test !isempty(rec_ref)
        @test length(rec_ref) == length(rec_jl)     # same episode length

        r = compare_rollouts(rec_ref, rec_jl)
        s = drift_summary(r)

        # the gate's core assertions
        @test s.D_discrete === nothing
        @test s.D_terminal === nothing
        @test s.reason_reference == s.reason_test
        @test s.max_dq0 <= FJ6_MAX_DQ
        @test s.max_dv0 <= FJ6_MAX_DQ
        @test s.max_dreturn <= FJ6_MAX_DRETURN
        @test s.divergence_kind in
            ("IDENTICAL", "TYPE1_READBACK_ONLY", "TYPE2_DYNAMICAL")

        # event timing must agree exactly wherever an event fires at all
        for (name, t) in event_timing_diff(rec_ref, rec_jl)
            if t["reference"] !== nothing && t["test"] !== nothing
                @test t["difference"] == 0
            else
                # an event may fire in neither run, but never in only one
                @test t["reference"] === t["test"]
            end
        end

        # the per-decision log is complete enough to audit
        csv = rollout_table(rec_jl)
        @test count(==('\n'), csv) == length(rec_jl) + 1
        @test occursin("cumulative_return", csv)
    finally
        close(ref)
    end
end

FJ6_OK && @testset "FJ6 free-running rollout parity (continuous)" begin
    mdp = DuckietownMDP(joinpath(pkgdir(DuckietownDecisionModels), "..",
        "duckduck", "policies", "sac", "training_config.yaml");
        action_space=:continuous)
    model = mdp.transition
    ref = ProcessReferenceBackend("sac"; seed=73, action_space=:continuous,
        map=mdp.map)
    try
        x0, _ = ref_reset!(ref, 73)
        rng = MersenneTwister(5)
        actions = [DuckieAction(0.2, 0.8 * (rand(rng) - 0.5)) for _ in 1:40]

        rec_ref = rollout_reference(ref, x0, actions)
        rec_jl = rollout_native(model, x0, actions)
        @test length(rec_ref) == length(rec_jl)

        s = drift_summary(compare_rollouts(rec_ref, rec_jl))
        @test s.D_discrete === nothing
        @test s.D_terminal === nothing
        @test s.max_dq0 <= FJ6_MAX_DQ
        @test s.max_dv0 <= FJ6_MAX_DQ
        @test s.max_dreturn <= FJ6_MAX_DRETURN
    finally
        close(ref)
    end
end

FJ6_OK && @testset "FJ6 harness invariants" begin
    mdp = DuckietownMDP(joinpath(pkgdir(DuckietownDecisionModels), "..",
        "duckduck", "policies", "q_learning", "training_config.yaml"))
    model = mdp.transition
    x0 = rand(MersenneTwister(53), initialstate(mdp))

    # a rollout against ITSELF must be exactly identical: this guards the
    # comparison machinery from silently passing everything
    acts = [FAST_STRAIGHT, SLOW_LEFT, BRAKE, FAST_RIGHT, SLOW_STRAIGHT]
    a = rollout_native(model, x0, acts)
    b = rollout_native(model, x0, acts)
    s = drift_summary(compare_rollouts(a, b))
    @test s.divergence_kind == "IDENTICAL"
    @test s.D_readback === nothing && s.D_dynamic === nothing
    @test s.D_discrete === nothing && s.D_terminal === nothing

    # and a deliberately perturbed run must be DETECTED
    perturbed = [FAST_STRAIGHT, SLOW_LEFT, BRAKE, FAST_RIGHT, FAST_LEFT]
    c = rollout_native(model, x0, perturbed)
    s2 = drift_summary(compare_rollouts(a, c))
    @test s2.divergence_kind == "TYPE2_DYNAMICAL"
    @test s2.D_dynamic !== nothing

    # rollouts stop at the episode boundary
    long = fill(FAST_STRAIGHT, 2000)
    rec = rollout_native(model, x0, long)
    @test length(rec) < 2000
    @test rec[end].terminated || rec[end].truncated
end
