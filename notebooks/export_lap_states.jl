# Export the recorded lap states in the reference protocol's own encoding, so
# the real simulator can be driven from Python without crossing the
# Windows/WSL process boundary the ProcessReferenceBackend assumes.
#
# `world_to_ref` is the package's validated encoder — the same one FJ5 uses
# for parity — so nothing about the state representation is improvised here.

using DuckietownDecisionModels
using Serialization, JSON3, Printf

D = deserialize(joinpath(@__DIR__, "lap_frames.jls"))
frames = D.frames
@printf("%d states, outcome %s, cost %.2f\n", length(frames), D.outcome, D.total)

payload = Dict(
    "outcome" => string(D.outcome),
    "cost" => D.total,
    "cost_model" => D.cost_model,
    "states" => [Dict("i" => i, "progress" => p,
                      "state" => world_to_ref(s))
                 for (i, (s, p)) in enumerate(frames)])

out = joinpath(@__DIR__, "lap_states.json")
open(out, "w") do io
    JSON3.write(io, payload)
end
@printf("wrote %s (%.1f MB)\n", basename(out), filesize(out) / 1e6)
