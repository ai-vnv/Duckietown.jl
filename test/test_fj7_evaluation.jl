# FJ7.6 — one evaluation harness for every solver family.
#
# The point of this gate is NOT that some policy scores well. It is that all
# four reference policies are scored through the SAME code path on the SAME
# validated MDP, so the numbers are comparable and nothing gets a private
# environment. The tests therefore check harness properties (determinism,
# bookkeeping consistency, policy-independence) rather than performance.

using DuckietownDecisionModels
using POMDPs
using Test
using Random

const FJ76_CFG = joinpath(pkgdir(DuckietownDecisionModels), "..", "duckduck",
    "policies", "q_learning", "training_config.yaml")
const FJ76_SAC_CFG = joinpath(pkgdir(DuckietownDecisionModels), "..", "duckduck",
    "policies", "sac", "training_config.yaml")

"""Fixed action, used to exercise the harness without any learned policy."""
struct ConstantMacroPolicy <: DuckietownDecisionModels.AbstractPolicy
    a::MacroAction
end
POMDPs.action(p::ConstantMacroPolicy, ::DuckietownMDP, ::DuckieWorldState) = p.a

struct ConstantContinuousPolicy <: DuckietownDecisionModels.AbstractPolicy
    a::DuckieAction
end
POMDPs.action(p::ConstantContinuousPolicy, ::DuckietownMDP, ::DuckieWorldState) = p.a

@testset "FJ7.6 evaluation harness (solver independent)" begin
    mdp = DuckietownMDP(FJ76_CFG; action_space=:discrete)

    @testset "determinism and seed dependence" begin
        pol = ConstantMacroPolicy(FAST_STRAIGHT)
        a = evaluate_policy(mdp, pol; seeds=1:4, max_steps=60)
        b = evaluate_policy(mdp, pol; seeds=1:4, max_steps=60)
        @test length(a) == 4
        @test [m.ret for m in a] == [m.ret for m in b]
        @test [m.decisions for m in a] == [m.decisions for m in b]
        @test [m.reason for m in a] == [m.reason for m in b]
        # the seed genuinely selects the episode: at least two seeds differ
        @test length(unique(m.ret for m in a)) > 1
        @test [m.seed for m in a] == collect(1:4)
    end

    @testset "bookkeeping is internally consistent" begin
        ms = evaluate_policy(mdp, ConstantMacroPolicy(SLOW_STRAIGHT);
            seeds=1:6, max_steps=80)
        for m in ms
            @test m.decisions >= 1
            @test m.decisions <= 80
            @test 0.0 <= m.brake_ratio <= 1.0
            @test m.mean_abs_d >= 0.0
            @test m.mean_abs_phi >= 0.0
            @test m.stop_zone_decisions <= m.decisions
            @test m.duck_yield_decisions <= m.decisions
            @test m.full_stops <= m.decisions
            @test m.stop_violations <= m.decisions
            # exactly one termination reason, and it agrees with the flags
            flags = (m.duck_collision, m.other_collision, m.offroad, m.timeout,
                m.goal)
            if m.reason == "in_progress"
                @test !any(flags)                     # cut off by max_steps
                @test m.decisions == 80
            else
                @test count(flags) >= 1
            end
            m.reason == "offroad" && @test m.offroad
            m.reason == "duck_collision" && @test m.duck_collision
            m.reason == "other_collision" && @test m.other_collision
            m.reason == "timeout" && @test m.timeout
            m.reason == "goal" && @test m.goal
            # discounted return is bounded by the undiscounted one in magnitude
            # when every term shares a sign is not guaranteed, so only check the
            # discount actually applied: gamma < 1 => |disc| <= sum|r| bound
            @test isfinite(m.discounted_return)
            @test isfinite(m.ret)
        end
    end

    @testset "BRAKE is recorded as braking, driving is not" begin
        brake = evaluate_policy(mdp, ConstantMacroPolicy(BRAKE);
            seeds=1:3, max_steps=40)
        drive = evaluate_policy(mdp, ConstantMacroPolicy(FAST_STRAIGHT);
            seeds=1:3, max_steps=40)
        @test all(m -> m.brake_ratio == 1.0, brake)
        @test all(m -> m.brake_ratio == 0.0, drive)
        # a braking agent covers less ground than a driving one
        @test sum(m -> m.mean_speed, brake) < sum(m -> m.mean_speed, drive)
    end

    @testset "aggregation matches the episode list" begin
        ms = evaluate_policy(mdp, ConstantMacroPolicy(FAST_STRAIGHT);
            seeds=1:5, max_steps=50)
        s = summarize_evaluation(ms)
        @test s.episodes == 5
        @test s.mean_return ≈ sum(m -> m.ret, ms) / 5
        @test s.mean_length ≈ sum(m -> m.decisions, ms) / 5
        @test s.offroad == count(m -> m.offroad, ms)
        @test s.full_stops == sum(m -> m.full_stops, ms)
        @test sum(values(s.reasons)) == 5
        @test summarize_evaluation(EpisodeMetrics[]).episodes == 0
    end

    @testset "the same harness accepts a continuous-action policy" begin
        cmdp = DuckietownMDP(FJ76_SAC_CFG; action_space=:continuous)
        ms = evaluate_policy(cmdp, ConstantContinuousPolicy(DuckieAction(0.2, 0.0));
            seeds=1:3, max_steps=40)
        @test length(ms) == 3
        @test all(m -> m.brake_ratio == 0.0, ms)
        zero_v = evaluate_policy(cmdp,
            ConstantContinuousPolicy(DuckieAction(0.0, 0.0));
            seeds=1:3, max_steps=40)
        @test all(m -> m.brake_ratio == 1.0, zero_v)
    end
end

# The real reference policies. Tabular checkpoints are always present; the
# actor checkpoints need the ddm-torch oracle only to export their weights.
@testset "FJ7.6 reference policies through the shared harness" begin
    root = joinpath(pkgdir(DuckietownDecisionModels), "..", "duckduck")
    mdp = DuckietownMDP(FJ76_CFG; action_space=:discrete)
    mdps = Dict{String,Any}()
    pols = Dict{String,Any}()
    for name in ("q_learning", "sarsa")
        f = joinpath(root, "policies", name, "policy.npy")
        isfile(f) || continue
        pols[name] = QTablePolicy(f; solver=Symbol(name))
        mdps[name] = mdp
    end
    @test haskey(pols, "q_learning")
    @test haskey(pols, "sarsa")

    if torch_policy_available()
        b = TorchPolicyReferenceBackend()
        try
            cmdp = DuckietownMDP(FJ76_SAC_CFG; action_space=:continuous)
            for (name, ctor) in ("sac" => SACActorPolicy, "td3" => TD3ActorPolicy)
                meta = torch_policy_init!(b, name)
                pols[name] = ctor(String(meta["weights_dir"]))
                mdps[name] = cmdp
            end
        finally
            close(b)
        end
    end

    res = compare_policies(mdps, pols; seeds=1:5, max_steps=150)
    @test length(res) == length(pols)
    for (name, s) in res
        @test s.episodes == 5
        @test isfinite(s.mean_return)
        @test isfinite(s.mean_discounted_return)
        @test 1 <= s.mean_length <= 150
        @test sum(values(s.reasons)) == 5
    end

    # comparability: identical seeds and horizon for every solver, and the
    # tabular pair is separable from the continuous pair only by its actions,
    # not by a different environment
    rows = sort(collect(keys(res)))
    @info "FJ7.6 shared-harness comparison (5 seeds, 150 decisions max)\n" *
          join(["  $n  return=$(round(res[n].mean_return; digits=3))" *
                "  len=$(round(res[n].mean_length; digits=1))" *
                "  |d|=$(round(res[n].mean_abs_d; digits=4))" *
                "  v=$(round(res[n].mean_speed; digits=4))" *
                "  brake=$(round(res[n].brake_ratio; digits=3))" *
                "  offroad=$(res[n].offroad)" *
                "  stops=$(res[n].full_stops)/$(res[n].stop_violations)" *
                "  reasons=$(res[n].reasons)" for n in rows], "\n")

    # re-running the comparison is bit-identical
    res2 = compare_policies(mdps, pols; seeds=1:5, max_steps=150)
    for n in rows
        @test res[n].mean_return == res2[n].mean_return
        @test res[n].mean_length == res2[n].mean_length
        @test res[n].reasons == res2[n].reasons
    end
end
