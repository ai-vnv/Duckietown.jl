# Does the shipped reward actually penalise not yielding and not stopping?
#
# Before inventing a cost, check what the model's own reward already says on
# the frames where the duck was crossing. If the weights are zero, the fix is
# a NEW named scenario, never an edit to a shipped one.

using DuckietownDecisionModels, POMDPs, Serialization, Printf, Random

const CFG = scenario_config(:stop_and_duck)
const BASE = DuckietownMDP(CFG; action_space = :discrete)
const TR = BASE.transition
const SCFG = TR.state_cfg
const ACTS = collect(POMDPs.actions(BASE))
const RNG = MersenneTwister(1)

println("RewardConfig fields and values in :stop_and_duck")
rc = CFG.reward
for f in fieldnames(typeof(rc))
    @printf("  %-26s %s\n", string(f), string(getfield(rc, f)))
end

D = deserialize(joinpath(@__DIR__, "lap_frames.jls"))
frames = D.frames

"""Replay each recorded state, take FAST_STRAIGHT, and report the reward
breakdown on the frames where the duck is a threat."""
function probe_duck_frames()
    rows = NamedTuple[]
    for (i, (s, _)) in enumerate(frames)
        raw, _ = get_raw_state(s, SCFG)
        raw.duck == NONE && continue
        r = simulate_decision(TR, s, FAST_STRAIGHT, RNG)
        push!(rows, (i = i, threat = raw.duck, v = round(raw.v; digits = 3),
            pedestrian = round(r.reward.pedestrian; digits = 3),
            events = round(r.reward.events; digits = 3),
            total = round(r.reward.total; digits = 3)))
    end
    return rows
end

rows = probe_duck_frames()
println("\nframes where the duck is a threat, driving straight through it:")
if isempty(rows)
    println("  none")
else
    @printf("  %5s %-14s %7s %11s %9s %8s\n",
            "frame", "threat", "v", "pedestrian", "events", "total")
    for r in rows
        @printf("  %5d %-14s %7.3f %11.3f %9.3f %8.3f\n",
                r.i, string(r.threat), r.v, r.pedestrian, r.events, r.total)
    end
end

"""Would slowing down actually be rewarded? Compare BRAKE against
FAST_STRAIGHT on the same states."""
function compare_yield()
    out = NamedTuple[]
    for (i, (s, _)) in enumerate(frames)
        raw, _ = get_raw_state(s, SCFG)
        raw.duck == NONE && continue
        rf = simulate_decision(TR, s, FAST_STRAIGHT, RNG)
        rb = simulate_decision(TR, s, BRAKE, RNG)
        push!(out, (i = i,
            straight = round(rf.reward.total; digits = 3),
            brake = round(rb.reward.total; digits = 3),
            ped_straight = round(rf.reward.pedestrian; digits = 3),
            ped_brake = round(rb.reward.pedestrian; digits = 3)))
    end
    return out
end

cmp = compare_yield()
println("\nstraight vs brake on those same frames:")
if !isempty(cmp)
    @printf("  %5s %10s %10s %13s %11s\n",
            "frame", "straight", "brake", "ped straight", "ped brake")
    for r in cmp
        @printf("  %5d %10.3f %10.3f %13.3f %11.3f\n",
                r.i, r.straight, r.brake, r.ped_straight, r.ped_brake)
    end
end
