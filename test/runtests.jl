using DuckietownDecisionModels
using Test

# FJ9.9b — the structured reporter. Every include below registers its
# testsets into `SUITE`, which is walked afterwards and written as JSON. The
# terminal parsers stay, and must agree with it.
include("reporter.jl")
include("reference_guard.jl")

const SUITE = Test.DefaultTestSet("DuckietownDecisionModels"; verbose=true)
const REPORT_PATH = get(ENV, "DDM_TEST_REPORT",
    joinpath(pkgdir(DuckietownDecisionModels), "artifacts", "fj9",
        "test_report.json"))
Test.push_testset(SUITE)

try

include("test_configs.jl")
include("test_data_model.jl")
include("test_native_render.jl")
# Fixture-based parity needs NOTHING from the reference checkout: the FJ2/FJ3
# fixtures are committed in test/fixtures/, generated once from the reference
# and frozen. These ran only under HAVE_REFERENCE for years by accident of
# grouping, which is why CI coverage read 14% — the beefiest tests never
# executed there. Only the files that read the reference's frozen configs,
# checkpoints, or its live Python stay behind the guard.
include("test_fj2_parity.jl")
include("test_fj3_map.jl")
include("test_fj3_ego.jl")
include("test_fj3_delay.jl")
include("test_fj3_duck.jl")
include("test_fj3_obs.jl")
include("test_fj3_rng.jl")

if HAVE_REFERENCE
    include("test_fj3_transition.jl")
    include("test_fj4_pomdps.jl")
# FJ5 runs the real Python reference stack side by side; it skips itself when
# the reference environment (WSL + ddm-ref) is not present.
    include("test_fj5_reference.jl")
# FJ5-R: in-process PythonCall backend; skips unless Linux Julia >= 1.11 with
# PythonCall bound to the validated ddm-ref interpreter.
    include("test_fj5r_pythoncall.jl")
    include("test_fj6_rollout.jl")
    include("test_fj7_tabular.jl")
    include("test_fj7_actors.jl")
    include("test_fj7_evaluation.jl")
# FJ8.0: the generative-model contract a planner depends on, and its cost.
    include("test_fj8_gen.jl")
    include("test_fj8_gen_native.jl")
# FJ8.1: the solver-facing contract, tested with no solver installed.
    include("test_fj8_planning.jl")
# FJ8.2 / FJ8.5: the first external solver integration, and the proof that
# adding or removing it changes nothing about the model.
    include("test_fj8_mcts.jl")
# FJ8.3: continuous-action DPW, action widening on, state widening off.
    include("test_fj8_dpw.jl")
# FJ8.4a: the cost-search curve and compute-matched operating points. This
# is a real planner benchmark (MCTS and DPW driven across budget curves) and
# is the single most expensive file in the suite — measured at 3m23s of an
# 8-minute run. DDM_SKIP_BENCH=1 skips it for day-to-day runs; leave it
# unset for release-grade runs. The skip is announced, never silent.
    if get(ENV, "DDM_SKIP_BENCH", "") == "1"
        @info "DDM_SKIP_BENCH=1 — skipping test_fj8_budget.jl (the planner benchmark, ~3.5 min); unset it for a release-grade run"
    else
        include("test_fj8_budget.jl")
    end
# FJ8.4b: the cross-family comparison machinery (the experiment itself is
# tools/solver_comparison.jl, which is far too slow for a regression suite).
    include("test_fj8_comparison.jl")
# FJ10: POMDP readiness audit — an audit, not an implementation.
# FJ9.0/9.1: the visualisation contract and world geometry. Runs with NO
# plotting backend — the geometry is core, the drawing is an extension.
# FJ8.4c: frozen-protocol replication with per-decision logging.
    include("test_fj84c_enrichment.jl")
    include("test_fj9_visualization.jl")
# FJ9.6: diagnostic time series read from the FJ8.4c log — no execution.
include("test_fj96_diagnostics.jl")
# FJ9.7: animation as playback of the same log — still no execution.
include("test_fj97_animation.jl")
# FJ9.8: publication composites — captions validated like data.
include("test_fj98_publication.jl")
    include("test_fj10_readiness.jl")
    include("test_fj8_solver_independence.jl")
else
    @warn "SKIPPING 26 reference-dependent test files" files=26
end
# FJ9.9: reproducibility closure — audits that fail when reality drifts.
include("test_fj99_closure.jl")

finally
    Test.pop_testset()
end

let r = SuiteReporter.write_report(SUITE, REPORT_PATH)
    @info "FJ9.9b structured test report" path=REPORT_PATH
    println("REPORT_TESTSETS=", r["testsets"])
    println("REPORT_ASSERTIONS=", r["assertions"])
    println("REPORT_FAILURES=", r["failures"])
    println("REPORT_ERRORS=", r["errors"])
    println("REPORT_BROKEN=", r["broken"])
end

# print the usual summary and fail the run if anything failed
Test.finish(SUITE)
