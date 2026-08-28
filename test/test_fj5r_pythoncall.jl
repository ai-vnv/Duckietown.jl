# FJ5-R: the in-process PythonCall reference backend.
#
# Three-way validation from ONE latent state and ONE action:
#
#                          same x_t, same a_t
#              ┌─────────────────┼─────────────────┐
#              ▼                 ▼                 ▼
#     ProcessReferenceBackend  PythonCallRef.    native Julia
#        (JSON-lines, FJ5)     (in-process)      (simulate_decision)
#              └──── must be EXACTLY equal ──┘         │
#                          └──── FJ5 structural rule ──┘
#
# The two Python transports drive the SAME `Session` class, so the state
# bridge and every discrete/semantic field must be EXACTLY equal. They are
# NOT bit-identical on libm-sensitive quantities, and the cause is measured
# here rather than assumed: running Python inside the Julia process rebinds
# its `atan2` to Julia's libm (see the interposition testset). Against native
# Julia the existing FJ5 criterion applies unchanged (Rule A–E: no discrete
# mismatch, nothing outside LIBM_DERIVED_FIELDS, bounded ULP/absdiff, signed
# zeros tracked separately).
#
# Requires: Linux Julia >= 1.11 in WSL, `using PythonCall` bound to the
# validated `ddm-ref` interpreter. Skips cleanly everywhere else.

using DuckietownDecisionModels
using POMDPs
using Test
using Random

const FJ5R_PYCALL_OK = try
    # the extension only exists once PythonCall is loaded by the test env
    @eval using PythonCall
    sys = pyimport("sys")
    exe = pyconvert(String, sys.executable)
    ver = pyconvert(String, sys.version)
    occursin("ddm-ref", exe) && startswith(ver, "3.9")
catch err
    @info "FJ5-R: PythonCall unavailable or not bound to ddm-ref — skipping" err
    false
end

if !FJ5R_PYCALL_OK
    @info "FJ5-R: skipped (needs Linux Julia >= 1.11 + PythonCall on ddm-ref)"
end

FJ5R_PYCALL_OK && @testset "FJ5-R environment identification" begin
    sys = pyimport("sys")
    @test occursin("ddm-ref", pyconvert(String, sys.executable))
    @test startswith(pyconvert(String, sys.version), "3.9")
    @test pyconvert(String, pyimport("numpy").__version__) == "1.20.0"
    @test pyconvert(String, pyimport("gym").__version__) == "0.23.1"
    @test pyconvert(String, pyimport("gym_duckietown").__version__) == "6.1.34"
    @test VERSION >= v"1.11"
    @test Sys.islinux()
end

FJ5R_PYCALL_OK && @testset "FJ5-R backend construction + interface" begin
    ref = PythonCallReferenceBackend("q_learning"; seed=53)
    try
        @test ref isa AbstractReferenceBackend
        @test ref.info[:config] == "q_learning"
        @test ref.info[:frame_skip] == 6
        @test ref.info[:max_steps] == 1500
        @test ref.info[:tile_size] == 0.585
        @test ref.info[:n_signs] == 1

        w, dump = ref_reset!(ref, 53)
        @test w isa DuckieWorldState
        @test w.ego.step_count == 0
        @test isempty(w.ego.command_history)
        @test length(w.ducks) == 1
        @test length(w.stop_signs) == 1

        # state bridge round-trip through the in-process path
        for a in (FAST_STRAIGHT, SLOW_LEFT, BRAKE)
            w, _ = ref_step!(ref, a)
        end
        @test w.ego.step_count == 18
        ref_set_state!(ref, w)
        w2, _ = ref_get_state(ref)
        @test w2.ego.pos == w.ego.pos
        @test w2.ego.q0 == w.ego.q0
        @test w2.ego.v0 == w.ego.v0
        @test w2.ego.command_history == w.ego.command_history
        @test w2.ducks[1].center == w.ducks[1].center
        @test w2.ducks[1].obj_norm == w.ducks[1].obj_norm
        @test all(d -> d.ulps == 0, compare_worlds(w, w2))
    finally
        close(ref)
    end
end

"""
Compare two reference backends field by field from the same injected state.
Returns the list of differing field names.

Expected result is NOT "empty": see the libm-interposition testset below.
Everything outside `LIBM_DERIVED_FIELDS` must be exactly equal.
"""
function transport_diff(a, b, world, action)
    ref_set_state!(a, world)
    ref_set_state!(b, world)
    wa, da = ref_step!(a, action)
    wb, db = ref_step!(b, action)
    names = String[]
    for d in compare_worlds(wa, wb)
        d.ulps == 0 && !d.bitdiff && continue
        push!(names, d.name)
    end
    ra, rb = da.result, db.result
    for f in (:progress, :lateral, :heading, :time, :pedestrian, :stagnation,
        :stop_approach, :steering, :events, :total)
        Float64(ra.reward_terms[f]) == Float64(rb.reward_terms[f]) ||
            push!(names, "reward.$f")
    end
    String(ra.reason) == String(rb.reason) || push!(names, "reason")
    Bool(ra.terminated) == Bool(rb.terminated) || push!(names, "terminated")
    Bool(ra.truncated) == Bool(rb.truncated) || push!(names, "truncated")
    for k in keys(ra.events)
        Bool(ra.events[k]) == Bool(rb.events[k]) || push!(names, "events.$k")
    end
    return names
end

FJ5R_PYCALL_OK && @testset "FJ5-R libm interposition (why the transports differ)" begin
    # Running Python INSIDE the Julia process changes which libm its math
    # calls bind to: Julia loads its own libm (OpenLibm) with global scope, so
    # CPython's and NumPy's `atan2` resolve to Julia's implementation instead
    # of glibc's. Demonstrated here on the exact q0 entries the pose readback
    # uses. Standalone Python returns 0.4677396694940821 for the same inputs
    # (measured; see docs/src/validation/FJ5R_STATUS.md) — one ULP away.
    math = pyimport("math")
    np = pyimport("numpy")
    q21, q11 = 0.45086989117635035, 0.8925896824580856
    @test pyconvert(Float64, math.atan2(q21, q11)) === atan(q21, q11)
    @test pyconvert(Float64, np.arctan2(q21, q11)) === atan(q21, q11)
    # consequence: the in-process reference is NOT bit-faithful for
    # libm-sensitive quantities; ProcessReferenceBackend stays the numerical
    # reference of record.
end

FJ5R_PYCALL_OK && @testset "FJ5-R transport equivalence (process vs pythoncall)" begin
    reference_backend_available() ||
        (@info "FJ5-R: process backend unavailable, skipping transport test";
         return)
    mdp = DuckietownMDP(joinpath(pkgdir(DuckietownDecisionModels), "..",
        "duckduck", "policies", "q_learning", "training_config.yaml"))
    pc = PythonCallReferenceBackend("q_learning"; seed=53, map=mdp.map)
    pr = ProcessReferenceBackend("q_learning"; seed=53, map=mdp.map)
    try
        # both reset identically from the same seed: reset does not go
        # through a pose readback difference, so this IS exact
        wp, _ = ref_reset!(pc, 53)
        wq, _ = ref_reset!(pr, 53)
        @test wp.ego.pos == wq.ego.pos
        @test wp.ego.angle == wq.ego.angle
        @test wp.ducks[1].center == wq.ducks[1].center
        @test all(d -> d.ulps == 0, compare_worlds(wp, wq))

        # injecting a state and reading it back is EXACT on both transports
        ref_set_state!(pc, wp)
        ref_set_state!(pr, wp)
        ip, _ = ref_get_state(pc)
        iq, _ = ref_get_state(pr)
        @test all(d -> d.ulps == 0 && !d.bitdiff, compare_worlds(ip, iq))

        # stepping: everything is exactly equal EXCEPT the libm-derived chain,
        # because in-process Python binds atan2 to Julia's libm (see the
        # interposition testset above)
        for a in ALL_MACRO_ACTIONS
            names = transport_diff(pr, pc, wp, a)
            @test issubset(Set(names), LIBM_DERIVED_FIELDS)
        end

        # a stateful sequence, advancing the Julia-side state each step
        w = wp
        rng = MersenneTwister(4)
        offenders = String[]
        for _ in 1:30
            a = rand(rng, ALL_MACRO_ACTIONS)
            append!(offenders, transport_diff(pr, pc, w, a))
            w = simulate_decision(mdp.transition, w, a, MersenneTwister(0)).sp
        end
        @test issubset(Set(offenders), LIBM_DERIVED_FIELDS)
    finally
        close(pc)
        close(pr)
    end
end

FJ5R_PYCALL_OK && @testset "FJ5-R matched-state parity vs native Julia (discrete)" begin
    mdp = DuckietownMDP(joinpath(pkgdir(DuckietownDecisionModels), "..",
        "duckduck", "policies", "q_learning", "training_config.yaml"))
    ref = PythonCallReferenceBackend("q_learning"; seed=53, map=mdp.map)
    try
        w, _ = ref_reset!(ref, 53)

        reports = matched_state_sweep(ref, mdp.transition, w, ALL_MACRO_ACTIONS)
        s = parity_summary(reports)
        @test s.steps == 7
        @test s.accepted == 7
        @test isempty(s.discrete_mismatches)
        @test issubset(Set(s.nonzero_fields), LIBM_DERIVED_FIELDS)
        @test issubset(Set(s.bitwise_only_fields), SIGNED_ZERO_FIELDS)
        @test s.max_ulps <= LIBM_MAX_ULPS
        @test s.max_absdiff <= LIBM_MAX_ABSDIFF

        rng = MersenneTwister(4)
        traj = [rand(rng, ALL_MACRO_ACTIONS) for _ in 1:30]
        reports = matched_state_sweep(ref, mdp.transition, w, traj;
            rng=MersenneTwister(4), advance=true)
        s = parity_summary(reports)
        @test s.steps == 30
        @test s.accepted == 30
        @test isempty(s.discrete_mismatches)
        @test issubset(Set(s.nonzero_fields), LIBM_DERIVED_FIELDS)
        @test s.max_ulps <= LIBM_MAX_ULPS

        # branch purity is unaffected by which reference backend is attached
        snap = deepcopy(w)
        sp1 = simulate_decision(mdp.transition, w, FAST_LEFT, MersenneTwister(3)).sp
        sp2 = simulate_decision(mdp.transition, w, FAST_RIGHT, MersenneTwister(3)).sp
        @test w.ego.pos == snap.ego.pos
        @test w.ego.command_history == snap.ego.command_history
        @test sp1.ego.pos != sp2.ego.pos
        @test sp1.ducks !== w.ducks
    finally
        close(ref)
    end
end

FJ5R_PYCALL_OK && @testset "FJ5-R matched-state parity vs native Julia (continuous)" begin
    mdp = DuckietownMDP(joinpath(pkgdir(DuckietownDecisionModels), "..",
        "duckduck", "policies", "sac", "training_config.yaml");
        action_space=:continuous)
    ref = PythonCallReferenceBackend("sac"; seed=73, action_space=:continuous,
        map=mdp.map)
    try
        w, _ = ref_reset!(ref, 73)
        acts = [DuckieAction(0.0, 0.0), DuckieAction(0.17, 0.0),
            DuckieAction(0.41, 0.0), DuckieAction(0.17, 1.5),
            DuckieAction(0.17, -1.5), DuckieAction(0.41, 1.5),
            DuckieAction(0.41, -1.5), DuckieAction(0.30, 0.7),
            DuckieAction(0.60, 2.5)]     # last one is out of range -> clipped
        reports = matched_state_sweep(ref, mdp.transition, w, acts)
        s = parity_summary(reports)
        @test s.steps == length(acts)
        @test s.accepted == length(acts)
        @test isempty(s.discrete_mismatches)
        @test issubset(Set(s.nonzero_fields), LIBM_DERIVED_FIELDS)
        @test s.max_ulps <= LIBM_MAX_ULPS
        @test any(d -> startswith(d.name, "cont."), reports[1].diffs)

        rng = MersenneTwister(9)
        traj = [DuckieAction(0.25, 1.2 * (rand(rng) - 0.5)) for _ in 1:30]
        reports = matched_state_sweep(ref, mdp.transition, w, traj; advance=true)
        s = parity_summary(reports)
        @test s.steps == 30
        @test s.accepted == 30
        @test issubset(Set(s.nonzero_fields), LIBM_DERIVED_FIELDS)
    finally
        close(ref)
    end
end
