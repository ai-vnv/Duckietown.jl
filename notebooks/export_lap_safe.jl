# Audit the stop subsystem on the recorded lap and export every physics
# substep in the reference protocol's encoding for the real renderer.

using DuckietownDecisionModels
using Serialization, JSON3, Printf

const CFG = scenario_config(:stop_and_duck_safe)
const BASE = DuckietownMDP(CFG; action_space = :discrete)
const SCFG = BASE.transition.state_cfg
const D = deserialize(joinpath(@__DIR__, "lap_frames_safe.jls"))
const SUBS = D.subs

function stop_audit()
    seen = count(f -> f.d_stop !== nothing, SUBS)
    sigma = count(f -> f.sigma, SUBS)
    evs = unique((f.dec, f.event) for f in SUBS if !isempty(f.event))
    @printf("stop sign: d_stop available %d/%d substeps, sigma_stop %d substeps\n",
            seen, length(SUBS), sigma)
    println("           events: ", isempty(evs) ? "none" :
            join(("$(e[2]) at dec $(e[1])" for e in evs), ", "))
    sg = SUBS[1].w.stop_signs[1]
    @printf("           sign at (%.3f, %.3f) angle %.1f deg\n",
            sg.pos[1], sg.pos[3], rad2deg(sg.angle))
end

stop_audit()

@printf("\n%d substeps, outcome %s, cost %.2f (first plan %.2f)\n",
        length(SUBS), D.outcome, D.total, D.cost_model)

payload = Dict(
    "outcome" => string(D.outcome),
    "cost" => D.total,
    "cost_model" => D.cost_model,
    "scenario" => string(D.scenario),
    "frame_skip" => D.frame_skip,
    "states" => [begin
        raw, _ = get_raw_state(f.w, SCFG)
        Dict("i" => i, "dec" => f.dec, "progress" => f.prog,
             "action" => f.a === nothing ? "-" : string(f.a),
             "v" => f.v,
             "pedestrian" => f.ped,
             "stop_approach" => f.stop_approach,
             "d_stop" => f.d_stop,
             "sigma" => f.sigma,
             "event" => f.event,
             "duck" => string(raw.duck),
             "state" => world_to_ref(f.w))
    end for (i, f) in enumerate(SUBS)])

out = joinpath(@__DIR__, "lap_states_safe.json")
open(out, "w") do io
    JSON3.write(io, payload)
end
@printf("wrote %s (%.1f MB)\n", basename(out), filesize(out) / 1e6)
