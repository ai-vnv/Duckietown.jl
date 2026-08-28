# FJ5.4: stop-sign reachability probe against the LIVE reference runtime.
#
# Question: in the baseline configuration, does `next_stop_candidate` ever
# return a candidate along the legal route? The FJ3.6 offline observation said
# no (`d_stop = None` across a 300-decision rollout). This probe answers it
# from the real Python runtime and classifies the outcome.
#
# The baseline config is NEVER modified. The probe only observes; the one
# control condition it runs (a corrected placement) is an explicitly separate
# scenario used to prove the filters themselves work.
#
# Usage:  julia --project=. tools/parity/run_fj54_stop_probe.jl [out.json]

using DuckietownDecisionModels
using JSON3
using Printf

out_path = length(ARGS) >= 1 ? ARGS[1] :
    joinpath(pkgdir(DuckietownDecisionModels), "docs", "validation", "fj54_stop_probe.json")

reference_backend_available() ||
    error("FJ5.4 probe needs the reference backend (WSL + ddm-ref)")

num(x) = x === nothing ? nothing :
    x isa AbstractDict ? (x["nonfinite"] == "nan" ? NaN :
        x["nonfinite"] == "inf" ? Inf :
        x["nonfinite"] == "-inf" ? -Inf : -0.0) : Float64(x)

"""Summarise one probe run: how close each filter ever came to passing."""
function summarise(rows)
    n = length(rows)
    min_lateral = Inf
    min_geometric = Inf
    max_ahead = -Inf
    min_distance = Inf
    orient_pass = 0
    lateral_pass = 0
    ahead_pass = 0
    range_pass = 0
    all_pass = 0
    d_stop_seen = 0
    for r in rows
        for s in r.signs
            lat = num(s.lateral)
            geo = num(s.geometric)
            ahead = num(s.ahead)
            dist = num(s.distance)
            orient = num(s.orient_dot)
            min_lateral = min(min_lateral, lat)
            min_geometric = min(min_geometric, geo)
            max_ahead = max(max_ahead, ahead)
            ahead > 0.0 && (min_distance = min(min_distance, dist))
            orient <= -0.70710678 && (orient_pass += 1)
            lat <= 0.40 && (lateral_pass += 1)
            ahead > 0.0 && (ahead_pass += 1)
            (ahead > 0.0 && dist <= 3.0) && (range_pass += 1)
            Bool(s.accepted) && (all_pass += 1)
        end
        r.d_stop === nothing || (d_stop_seen += 1)
    end
    return (decisions=n,
        min_lateral=min_lateral, min_geometric=min_geometric,
        max_ahead=max_ahead, min_distance_ahead=min_distance,
        orientation_condition_passes=orient_pass,
        lateral_condition_passes=lateral_pass,
        ahead_condition_passes=ahead_pass,
        range_condition_passes=range_pass,
        all_conditions_pass=all_pass,
        decisions_with_d_stop=d_stop_seen)
end

function classify(s)
    s.decisions_with_d_stop > 0 && return "A_REACHABLE"
    if s.orientation_condition_passes == 0
        return "B_UNREACHABLE_BY_CONFIGURATION (orientation filter never passes)"
    elseif s.lateral_condition_passes == 0
        return "B_UNREACHABLE_BY_CONFIGURATION (lateral filter never passes)"
    elseif s.ahead_condition_passes == 0
        return "B_UNREACHABLE_BY_CONFIGURATION (sign never ahead)"
    elseif s.all_conditions_pass == 0
        return "B_UNREACHABLE_BY_CONFIGURATION (no decision satisfies all filters simultaneously)"
    end
    return "D_UNKNOWN (filters passed but no candidate returned — investigate)"
end

"""Rows where the candidate filter actually returned a stop distance."""
function hits(rows)
    out = []
    for r in rows
        r.d_stop === nothing && continue
        s = r.signs[1]
        push!(out, Dict(
            "decision" => Int(r.decision),
            "pos" => [num(r.pos[1]), num(r.pos[2]), num(r.pos[3])],
            "angle" => num(r.angle),
            "d_stop" => num(r.d_stop),
            "ahead" => num(s.ahead), "lateral" => num(s.lateral),
            "orient_dot" => num(s.orient_dot),
            "geometric" => num(s.geometric)))
    end
    return out
end

report = Dict{String,Any}()

@info "FJ5.4: probing the BASELINE configuration (unmodified)"
ref = ReferenceBackend("q_learning"; seed=53)
baseline_rows = try
    ref_probe_stop(ref; decisions=400)
finally
    close(ref)
end
baseline = summarise(baseline_rows)
baseline_hits = hits(baseline_rows)
report["baseline"] = Dict("summary" => baseline,
    "verdict" => classify(baseline),
    "config" => "q_learning (unmodified)",
    "sign_world_pos" => [0.702, 0.0, 1.8135],
    "hits" => baseline_hits,
    "n_resets" => count(r -> haskey(r, :terminated_after), baseline_rows))

@info "FJ5.4 baseline" baseline classify(baseline)

# Control condition: the SAME filters with a placement on the route ring.
# This is a separate scenario, never a change to the baseline — it exists to
# show whether an unreachable verdict is about the placement or the filters.
@info "FJ5.4: control condition (corrected placement, separate scenario)"
overrides = Dict("duck_controller" => Dict(
    "stop_spawn_pos" => [0.28, -0.52], "stop_spawn_rotate" => 90.0))
ref2 = ReferenceBackend("q_learning"; seed=53, overrides=overrides)
control_rows = try
    ref_probe_stop(ref2; decisions=400)
finally
    close(ref2)
end
control = summarise(control_rows)
report["control_corrected_placement"] = Dict("summary" => control,
    "verdict" => classify(control), "overrides" => overrides,
    "hits" => hits(control_rows))

@info "FJ5.4 control" control classify(control)

open(out_path, "w") do io
    JSON3.pretty(io, report)
end
@info "FJ5.4 probe written" out_path
