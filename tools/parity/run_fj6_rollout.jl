# FJ6: free-running full-episode rollout parity.
#
# Objective (as agreed after the FJ5-R libm finding):
#   validate free-running rollout parity between the ISOLATED Python/glibc
#   reference runtime and the native Julia model, using the in-process
#   PythonCall backend as a secondary semantic oracle to distinguish true
#   model divergence from process-level math-library effects.
#
# Three lanes:
#   A  process reference  vs  native Julia    -> true cross-runtime drift
#   B  pycall reference   vs  native Julia    -> semantic/event parity
#   C  process reference  vs  pycall          -> libm isolation diagnostic
#
# x0 is matched by construction: it is exported from the reference reset and
# injected into every run exactly once, at t0. No per-step re-injection.
#
# Usage (WSL, ddm-ref active, Julia 1.11.3):
#   julia --project=/tmp/fj5rdev tools/parity/run_fj6_rollout.jl

using DuckietownDecisionModels
using PythonCall            # enables the in-process oracle (lane B/C)
using JSON3
using Random
using Printf

const OUT = joinpath(pkgdir(DuckietownDecisionModels), "artifacts", "fj6")
mkpath(OUT)

const QCFG = joinpath(pkgdir(DuckietownDecisionModels), "..", "duckduck",
    "policies", "q_learning", "training_config.yaml")
const SACCFG = joinpath(pkgdir(DuckietownDecisionModels), "..", "duckduck",
    "policies", "sac", "training_config.yaml")

wrap_angle(a) = (while a > pi; a -= 2pi; end; while a < -pi; a += 2pi; end; a)

"""
    lane_omega(world) -> Float64

The same lane-following P-controller used by the FJ3/FJ5 fixture generators
(heading error to the lane tangent plus a lateral term, driving 5 cm right of
the centreline). It keeps the ego on the road for hundreds of decisions,
which is what a free-running rollout comparison needs.
"""
function lane_omega(world)
    pos = collect(world.ego.pos)
    angle = world.ego.angle
    pt, tangent = closest_curve_point(world.map, pos, angle)
    (pt === nothing || tangent === nothing) && return 0.0
    theta_d = atan(-tangent[3], tangent[1])
    right = [-sin(angle), 0.0, -cos(angle)]
    lateral = sum((pos .- pt) .* right)
    return -(1.5 * wrap_angle(angle - theta_d) + 3.0 * (lateral - 0.05))
end

"""Scripted lane-following macro policy, replayed identically by every run."""
function scripted_discrete(model, x0, n)
    acts = MacroAction[]
    w = x0
    rng = MersenneTwister(0)
    for _ in 1:n
        om = lane_omega(w)
        a = om > 0.30 ? SLOW_LEFT :
            om < -0.30 ? SLOW_RIGHT :
            abs(om) < 0.10 ? FAST_STRAIGHT : SLOW_STRAIGHT
        push!(acts, a)
        r = simulate_decision(model, w, a, rng)
        w = r.sp
        (r.terminated || r.truncated) && break
    end
    return acts
end

function scripted_continuous(model, x0, n)
    acts = DuckieAction[]
    w = x0
    rng = MersenneTwister(0)
    for _ in 1:n
        om = clamp(lane_omega(w), -1.5, 1.5)
        # slow down when steering hard, otherwise the constant-speed
        # controller leaves the road within ~40 decisions
        a = DuckieAction(abs(om) > 0.5 ? 0.17 : 0.30, om)
        push!(acts, a)
        r = simulate_decision(model, w, a, rng)
        w = r.sp
        (r.terminated || r.truncated) && break
    end
    return acts
end

"""
Lane following plus stop compliance: brake inside the stop zone until the
dwell latches `sigma_stop`. Uses the UNMODIFIED baseline config — only the
action sequence differs, which is what FJ5.4 showed is needed to reach the
baseline sign.
"""
function scripted_stop_compliance(model, x0, n)
    acts = DuckieAction[]
    w = x0
    rng = MersenneTwister(0)
    for _ in 1:n
        raw, _ = get_raw_state(w, model.state_cfg)
        must_stop = raw.d_stop !== nothing &&
            raw.d_stop <= model.state_cfg.stop_zone &&
            !w.stop_memory.sigma_stop
        a = if must_stop
            DuckieAction(0.0, 0.0)          # hold still until sigma latches
        else
            # slow and steady after the stop, so the ego stays on the road
            # long enough to actually PASS the sign (passed_stop event)
            DuckieAction(0.17, clamp(lane_omega(w), -1.5, 1.5))
        end
        push!(acts, a)
        r = simulate_decision(model, w, a, rng)
        w = r.sp
        (r.terminated || r.truncated) && break
    end
    return acts
end

"""
Stop compliance with a gentler resume: after `sigma_stop` latches, keep the
speed low and the steering soft so the ego stays on the road long enough to
PASS the sign (which is what raises `passed_stop`).
"""
function scripted_stop_pass(model, x0, n)
    acts = DuckieAction[]
    w = x0
    rng = MersenneTwister(0)
    for _ in 1:n
        raw, _ = get_raw_state(w, model.state_cfg)
        must_stop = raw.d_stop !== nothing &&
            raw.d_stop <= model.state_cfg.stop_zone &&
            !w.stop_memory.sigma_stop
        a = if must_stop
            DuckieAction(0.0, 0.0)
        else
            DuckieAction(0.17, clamp(0.6 * lane_omega(w), -0.8, 0.8))
        end
        push!(acts, a)
        r = simulate_decision(model, w, a, rng)
        w = r.sp
        (r.terminated || r.truncated) && break
    end
    return acts
end

function run_case(name, cfgpath, action_space, config_name, seed, nsteps,
    make_actions; overrides=Dict{String,Any}())
    @info "FJ6 case" name action_space nsteps
    mdp = DuckietownMDP(cfgpath; action_space=action_space)
    model = mdp.transition

    # x0 from the reference reset (the reference's own rho_0), exported once
    pr = ProcessReferenceBackend(config_name; seed=seed,
        action_space=action_space, map=mdp.map, overrides=overrides)
    pc = PythonCallReferenceBackend(config_name; seed=seed,
        action_space=action_space, map=mdp.map, overrides=overrides)
    result = Dict{String,Any}()
    try
        x0, _ = ref_reset!(pr, seed)
        actions = make_actions(model, x0, nsteps)
        @info "  action sequence" n = length(actions)

        rec_pr = rollout_reference(pr, x0, actions)
        rec_pc = rollout_reference(pc, x0, actions)
        rec_jl = rollout_native(model, x0, actions)

        lane_a = compare_rollouts(rec_pr, rec_jl)   # numerical reference
        lane_b = compare_rollouts(rec_pc, rec_jl)   # semantic oracle
        lane_c = compare_rollouts(rec_pr, rec_pc)   # libm isolation

        three = three_lane_table(rec_pr, rec_pc, rec_jl)
        hyp = libm_hypothesis_check(three)

        write(joinpath(OUT, "rollout_$(name)_process.csv"), rollout_table(rec_pr))
        write(joinpath(OUT, "rollout_$(name)_pycall.csv"), rollout_table(rec_pc))
        write(joinpath(OUT, "rollout_$(name)_julia.csv"), rollout_table(rec_jl))

        result["name"] = name
        result["config"] = config_name
        result["action_space"] = String(action_space)
        result["seed"] = seed
        result["n_actions"] = length(actions)
        result["lane_A_process_vs_julia"] = drift_summary(lane_a)
        result["lane_B_pycall_vs_julia"] = drift_summary(lane_b)
        result["lane_C_process_vs_pycall"] = drift_summary(lane_c)
        result["event_timing_A"] = event_timing_diff(rec_pr, rec_jl)
        result["event_timing_C"] = event_timing_diff(rec_pr, rec_pc)
        result["libm_hypothesis"] = Dict(
            "fields" => hyp.fields, "stats" => hyp.stats,
            "max_abs_d_CJ" => hyp.max_abs_d_CJ,
            "max_abs_dPJ_minus_dPP" => hyp.max_abs_dPJ_minus_dPP,
            "d_CJ_exactly_zero" => hyp.d_CJ_exactly_zero)
        result["returns"] = Dict(
            "process" => isempty(rec_pr) ? 0.0 : rec_pr[end].cumulative_return,
            "pycall" => isempty(rec_pc) ? 0.0 : rec_pc[end].cumulative_return,
            "julia" => isempty(rec_jl) ? 0.0 : rec_jl[end].cumulative_return)
        result["lengths"] = Dict("process" => length(rec_pr),
            "pycall" => length(rec_pc), "julia" => length(rec_jl))
        result["terminal"] = Dict(
            "process" => isempty(rec_pr) ? "" : rec_pr[end].reason,
            "pycall" => isempty(rec_pc) ? "" : rec_pc[end].reason,
            "julia" => isempty(rec_jl) ? "" : rec_jl[end].reason)
        # stop / duck exercise evidence
        result["stop"] = Dict(
            "process_d_stop_seen" => count(r -> r.d_stop !== nothing, rec_pr),
            "julia_d_stop_seen" => count(r -> r.d_stop !== nothing, rec_jl),
            "process_min_d_stop" => let v = [r.d_stop for r in rec_pr if r.d_stop !== nothing]
                isempty(v) ? nothing : minimum(v) end,
            "julia_min_d_stop" => let v = [r.d_stop for r in rec_jl if r.d_stop !== nothing]
                isempty(v) ? nothing : minimum(v) end)
        result["duck"] = Dict(
            "process_active_decisions" => count(r -> r.duck_active, rec_pr),
            "julia_active_decisions" => count(r -> r.duck_active, rec_jl),
            "process_crossings" => isempty(rec_pr) ? 0 : rec_pr[end].crossings_started,
            "julia_crossings" => isempty(rec_jl) ? 0 : rec_jl[end].crossings_started)

        @info "  lane A" kind = result["lane_A_process_vs_julia"].divergence_kind D_readback =
            result["lane_A_process_vs_julia"].D_readback D_dynamic =
            result["lane_A_process_vs_julia"].D_dynamic D_discrete =
            result["lane_A_process_vs_julia"].D_discrete
        @info "  lane C" kind = result["lane_C_process_vs_pycall"].divergence_kind
        @info "  libm hypothesis" max_abs_d_CJ = hyp.max_abs_d_CJ exactly_zero =
            hyp.d_CJ_exactly_zero
    finally
        close(pr)
        close(pc)
    end
    return result
end

report = Dict{String,Any}()
report["environment"] = Dict("julia" => string(VERSION),
    "os" => string(Sys.KERNEL), "pythoncall" => "0.9.25")

cases = Any[]
push!(cases, run_case("discrete_lanefollow", QCFG, :discrete, "q_learning",
    53, 120, scripted_discrete))
push!(cases, run_case("continuous_lanefollow", SACCFG, :continuous, "sac",
    73, 120, scripted_continuous))
# a long discrete run to reach the baseline stop sign (FJ5.4 showed it is
# reachable, just far along the route)
push!(cases, run_case("discrete_long", QCFG, :discrete, "q_learning",
    53, 400, scripted_discrete))
# stop-compliance trajectory on the UNMODIFIED baseline config: the sign is
# reachable (FJ5.4), it just needs a long enough route and a policy that
# actually brakes for it
push!(cases, run_case("continuous_stop_compliance", SACCFG, :continuous,
    "sac", 73, 400, scripted_stop_compliance))
push!(cases, run_case("continuous_stop_pass", SACCFG, :continuous, "sac",
    73, 400, scripted_stop_pass))

report["cases"] = cases

open(joinpath(OUT, "drift_summary.json"), "w") do io
    JSON3.pretty(io, report)
end
open(joinpath(OUT, "event_timing.json"), "w") do io
    JSON3.pretty(io, Dict(c["name"] => Dict(
        "lane_A_process_vs_julia" => c["event_timing_A"],
        "lane_C_process_vs_pycall" => c["event_timing_C"]) for c in cases))
end
open(joinpath(OUT, "stop_parity.json"), "w") do io
    JSON3.pretty(io, Dict(c["name"] => c["stop"] for c in cases))
end
open(joinpath(OUT, "duck_parity.json"), "w") do io
    JSON3.pretty(io, Dict(c["name"] => c["duck"] for c in cases))
end
@info "FJ6 artifacts written" OUT
