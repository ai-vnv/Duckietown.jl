# FJ8.0 — the generative-model contract, established BEFORE any planner.
#
# A planner is only as trustworthy as `gen`. These tests pin the four
# properties MCTS/DPW silently depend on:
#
#   determinism      same (s, a, rng) -> bitwise identical successor
#   root immutability expanding a node never touches the node's own state
#   branch purity     siblings share no mutable array, in either direction
#   native execution  the tree is grown by Julia, never by the Python reference
#
# and then measures what one `gen` costs, since a planner multiplies that by
# its node budget. No performance threshold is asserted here: the cost is
# reported so the planner gates can choose a budget from measurement.

using DuckietownDecisionModels
using POMDPs
using Test
using Random

const FJ8_QCFG = joinpath(pkgdir(DuckietownDecisionModels), "..", "duckduck",
    "policies", "q_learning", "training_config.yaml")
const FJ8_CCFG = joinpath(pkgdir(DuckietownDecisionModels), "..", "duckduck",
    "policies", "sac", "training_config.yaml")

fj8_mdp() = DuckietownMDP(FJ8_QCFG; action_space=:discrete)
fj8_cmdp() = DuckietownMDP(FJ8_CCFG; action_space=:continuous)
fj8_state(mdp, seed=11) = rand(MersenneTwister(seed), initialstate(mdp))

"""A few states spread along a driven trajectory, not just the spawn."""
function fj8_states(mdp, n=5; seed=11)
    s = fj8_state(mdp, seed)
    out = [s]
    rng = MersenneTwister(seed + 1)
    while length(out) < n
        r = simulate_decision(mdp.transition, s, FAST_STRAIGHT, rng)
        s = r.sp
        push!(out, s)
        (r.terminated || r.truncated) && break
    end
    return out
end

@testset "FJ8.0 gen is deterministic" begin
    mdp = fj8_mdp()
    for s in fj8_states(mdp, 4)
        for a in actions(mdp)
            x1 = POMDPs.gen(mdp, s, a, MersenneTwister(7))
            x2 = POMDPs.gen(mdp, s, a, MersenneTwister(7))
            diffs = world_differences(x1.sp, x2.sp)
            @test isempty(diffs)
            isempty(diffs) || @info "gen nondeterminism" a diffs[1:min(3, end)]
            @test x1.r === x2.r          # bitwise, not approximately
        end
    end

    cmdp = fj8_cmdp()
    cs = fj8_state(cmdp, 23)
    for a in (DuckieAction(0.0, 0.0), DuckieAction(0.2, 0.5),
        DuckieAction(0.41, -1.5))
        x1 = POMDPs.gen(cmdp, cs, a, MersenneTwister(3))
        x2 = POMDPs.gen(cmdp, cs, a, MersenneTwister(3))
        @test worlds_identical(x1.sp, x2.sp)
        @test x1.r === x2.r
    end

    # a different rng seed is allowed to differ, but must still be reproducible
    # for that seed — this is what makes a planner's branches replayable
    y1 = POMDPs.gen(mdp, fj8_state(mdp), FAST_LEFT, MersenneTwister(99))
    y2 = POMDPs.gen(mdp, fj8_state(mdp), FAST_LEFT, MersenneTwister(99))
    @test worlds_identical(y1.sp, y2.sp)
end

@testset "FJ8.0 gen never mutates the state it is called on" begin
    mdp = fj8_mdp()
    for s in fj8_states(mdp, 4)
        before = branch(s)                       # independent snapshot
        @test worlds_identical(s, before)        # the snapshot itself is faithful
        rng = MersenneTwister(5)
        for a in actions(mdp), _ in 1:3
            POMDPs.gen(mdp, s, a, rng)
        end
        diffs = world_differences(s, before)
        @test isempty(diffs)
        isempty(diffs) || @info "root mutated by gen" n = length(diffs) diffs[1:min(5, end)]
    end

    # the user-facing form of the same property
    cmdp = fj8_cmdp()
    s = fj8_state(cmdp, 31)
    root_before = branch(s)
    POMDPs.gen(cmdp, s, DuckieAction(0.3, 0.9), MersenneTwister(2))
    @test worlds_identical(s, root_before)

    # and through the richer entry point the planner may also use
    simulate_decision(cmdp.transition, s, DuckieAction(0.3, 0.9),
        MersenneTwister(2))
    @test worlds_identical(s, root_before)
end

@testset "FJ8.0 sibling branches do not alias" begin
    mdp = fj8_mdp()
    s = fj8_state(mdp, 17)
    kids = [POMDPs.gen(mdp, s, a, MersenneTwister(4)).sp for a in actions(mdp)]

    # no successor shares a mutable container with the root ...
    for (i, k) in enumerate(kids)
        shared = shared_mutable_arrays(s, k)
        @test isempty(shared)
        isempty(shared) || @info "child aliases root" child = i shared
    end
    # ... nor with any sibling
    for i in eachindex(kids), j in eachindex(kids)
        i < j || continue
        shared = shared_mutable_arrays(kids[i], kids[j])
        @test isempty(shared)
        isempty(shared) || @info "siblings alias" i j shared
    end

    # behavioural proof, not just identity: writing through one successor must
    # leave the root and the other successors untouched
    root_snapshot = branch(s)
    sib_snapshot = branch(kids[2])
    push!(kids[1].ego.command_history, (999.0, 9.0, 9.0))
    kids[1].ego.q0[1, 1] = -12345.0
    kids[1].crossings_started[1] += 77
    kids[1].crossing_armed[1] = !kids[1].crossing_armed[1]
    isempty(kids[1].ducks) || (kids[1].ducks[1].obj_corners[1] = (9.0, 9.0))
    @test worlds_identical(s, root_snapshot)
    @test worlds_identical(kids[2], sib_snapshot)

    # the track description is shared on purpose and must stay shared: copying
    # the map per node would dominate a planner's memory
    @test kids[1].map === s.map
    @test kids[1].stop_signs === s.stop_signs
    @test Set(shared_by_design(s, kids[1])) ==
        Set(["map", "stop_signs", "controller_rng"])
end

@testset "FJ8.0 the shared legacy RNG is frozen, not merely shared" begin
    # `controller_rng` is one mutable object held by every node in the tree.
    # That is only as safe as sharing the map IF nothing ever draws from it,
    # so the freeze is measured rather than assumed. The alternative —
    # copying it per decision — was measured at ~9 % of one gen call's total
    # allocation for a field with no semantic role, and rejected on that basis.
    mdp = fj8_mdp()
    s = fj8_state(mdp, 17)
    reference = copy(s.controller_rng)
    @test s.controller_rng == reference

    # build a wide, deep tree from the same root
    frontier = [s]
    all_nodes = DuckieWorldState[]
    rng = MersenneTwister(21)
    for _ in 1:3
        next = DuckieWorldState[]
        for node in frontier, a in actions(mdp)
            sp = POMDPs.gen(mdp, node, a, rng).sp
            push!(next, sp)
            push!(all_nodes, sp)
        end
        frontier = next[1:min(3, end)]      # keep the breadth bounded
    end
    @test length(all_nodes) >= 7

    # every node still sits at the root's stream position ...
    @test rng_frozen(all_nodes, reference)
    @test s.controller_rng == reference
    # ... and every node literally holds the same object
    @test all(n -> n.controller_rng === s.controller_rng, all_nodes)

    # behavioural proof: the shared stream produces the same next draw it would
    # have produced before the tree existed
    @test rand(copy(s.controller_rng)) === rand(copy(reference))

    # and the transition's own stochasticity really does come from the caller:
    # two different external rngs from the same state can differ, while the
    # state's stream stays put either way
    x = POMDPs.gen(mdp, s, FAST_STRAIGHT, MersenneTwister(1)).sp
    y = POMDPs.gen(mdp, s, FAST_STRAIGHT, MersenneTwister(2)).sp
    @test x.controller_rng == reference
    @test y.controller_rng == reference
end

@testset "FJ8.0 branch order does not change the result" begin
    # expanding a node's actions in any order must give the same children —
    # otherwise a planner's result would depend on its expansion schedule
    mdp = fj8_mdp()
    s = fj8_state(mdp, 41)
    acts = collect(actions(mdp))
    forward = Dict(a => POMDPs.gen(mdp, s, a, MersenneTwister(6)).sp for a in acts)
    reverse_order = Dict{MacroAction,DuckieWorldState}()
    for a in reverse(acts)
        reverse_order[a] = POMDPs.gen(mdp, s, a, MersenneTwister(6)).sp
    end
    for a in acts
        @test worlds_identical(forward[a], reverse_order[a])
    end

    # interleaving two branch lines must not couple them
    rngA, rngB = MersenneTwister(8), MersenneTwister(8)
    a1 = POMDPs.gen(mdp, s, FAST_LEFT, rngA).sp
    b1 = POMDPs.gen(mdp, s, BRAKE, rngB).sp
    a2 = POMDPs.gen(mdp, a1, FAST_LEFT, rngA).sp
    b2 = POMDPs.gen(mdp, b1, BRAKE, rngB).sp
    rngC, rngD = MersenneTwister(8), MersenneTwister(8)
    a1c = POMDPs.gen(mdp, s, FAST_LEFT, rngC).sp
    a2c = POMDPs.gen(mdp, a1c, FAST_LEFT, rngC).sp
    b1c = POMDPs.gen(mdp, s, BRAKE, rngD).sp
    b2c = POMDPs.gen(mdp, b1c, BRAKE, rngD).sp
    @test worlds_identical(a2, a2c)
    @test worlds_identical(b2, b2c)
end

@testset "FJ8.0 terminal semantics survive branching" begin
    mdp = fj8_mdp()
    s = fj8_state(mdp, 11)
    rng = MersenneTwister(1)
    seen_terminal = false
    for _ in 1:400
        r = simulate_decision(mdp.transition, s, FAST_LEFT, rng)
        # gen's successor and isterminal must agree with the chain's own reason
        @test isterminal(mdp, r.sp) == r.terminated
        @test is_truncated(mdp, r.sp) == r.truncated
        s = r.sp
        if r.terminated || r.truncated
            seen_terminal = true
            break
        end
    end
    @test seen_terminal          # the horizon is reachable, so the test is real

    # a planner may still expand a terminal node; that must not throw, and the
    # node must stay terminal
    if isterminal(mdp, s)
        x = POMDPs.gen(mdp, s, BRAKE, MersenneTwister(1))
        @test x.sp isa DuckieWorldState
        @test isterminal(mdp, x.sp)
    end
end

@testset "FJ8.0 the action sets a planner will search" begin
    mdp = fj8_mdp()
    acts = actions(mdp)
    @test length(acts) == 7
    @test Set(acts) == Set(ALL_MACRO_ACTIONS)
    @test all(a -> actionindex(mdp, a) isa Int, acts)
    s = fj8_state(mdp)
    @test actions(mdp, s) === acts          # state-independent, so cacheable

    # the continuous problem must NOT be enumerable: a planner has to widen
    cmdp = fj8_cmdp()
    space = actions(cmdp)
    @test space isa DuckieActionSpace
    @test_throws MethodError length(space)
    draws = [rand(MersenneTwister(k), space) for k in 1:200]
    @test all(a -> a in space, draws)
    @test length(unique(a -> (a.v, a.omega), draws)) == 200
    @test rand(MersenneTwister(5), space) == rand(MersenneTwister(5), space)
end

@testset "FJ8.0 gen cost profile (measured, not asserted)" begin
    mdp = fj8_mdp()
    s = fj8_state(mdp, 11)

    scaling = gen_scaling(mdp, s, FAST_STRAIGHT; calls=(1, 100, 1_000, 10_000))
    @test length(scaling) == 4
    @test [b.calls for b in scaling] == [1, 100, 1_000, 10_000]
    @test all(b -> b.seconds > 0 && b.bytes > 0 && b.allocs > 0, scaling)

    chain = benchmark_gen(mdp, s, FAST_STRAIGHT; calls=10_000, mode=:chain)
    cont = benchmark_gen(fj8_cmdp(), fj8_state(fj8_cmdp(), 23),
        DuckieAction(0.2, 0.3); calls=10_000, mode=:branch,
        label="gen(:branch) continuous")
    stages = gen_stage_profile(mdp, s, FAST_STRAIGHT; calls=2_000)

    @info "FJ8.0 gen scaling\n" * benchmark_table(scaling)
    @info "FJ8.0 gen variants\n" * benchmark_table([scaling[end], chain, cont])
    @info "FJ8.0 where one gen goes\n" * benchmark_table(stages)

    full = scaling[end]
    for nodes in (100, 1_000, 10_000)
        e = planning_budget_estimate(full, nodes)
        @info "FJ8.0 planning budget" nodes = e.nodes  seconds_per_decision =
            round(e.seconds_per_decision; digits=4)  mib_per_decision =
            round(e.mib_per_decision; digits=2)
    end

    # Stage accounting. `ego_tick` and `duck_step` run once per physics tick,
    # so their per-call figures must be weighted by frame_skip before being
    # compared with the whole decision; `branch(world)` is a reference point,
    # not a stage of the chain.
    fs = mdp.transition.frame_skip
    per_tick = ("ego_tick", "duck_step")
    weight(b) = b.label in per_tick ? fs : 1
    accounted = filter(b -> b.label != "FULL gen" && b.label != "branch(world)",
        stages)
    stage_us = sum(weight(b) * per_call_us(b) for b in accounted)
    stage_kib = sum(weight(b) * bytes_per_call(b) / 1024 for b in accounted)
    full = stages[end]
    @test full.label == "FULL gen"
    @info "FJ8.0 stage accounting (per-tick stages x frame_skip=$fs)" summed_us =
        round(stage_us; digits=2)  full_us = round(per_call_us(full); digits=2) time_ratio =
        round(stage_us / per_call_us(full); digits=3)  summed_kib =
        round(stage_kib; digits=1)  full_kib = round(bytes_per_call(full) / 1024; digits=1) mem_ratio =
        round(stage_kib / (bytes_per_call(full) / 1024); digits=3)

    # the single most expensive stage, named explicitly so the optimisation
    # target is recorded rather than rediscovered later
    worst = argmax(b -> weight(b) * per_call_us(b), accounted)
    @info "FJ8.0 dominant stage" stage = worst.label share_of_gen_time =
        round(weight(worst) * per_call_us(worst) / per_call_us(full); digits=3) share_of_gen_bytes =
        round(weight(worst) * bytes_per_call(worst) / bytes_per_call(full); digits=3)
end
