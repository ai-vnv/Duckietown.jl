# What the lap video does and does not show.
#
# Accumulators live inside functions: a bare top-level loop makes them fresh
# locals. That trap has now cost four debugging cycles in this session alone.

using DuckietownDecisionModels, POMDPs, Serialization, Printf

const CFG = scenario_config(:stop_and_duck)
const BASE = DuckietownMDP(CFG; action_space = :discrete)
const SCFG = BASE.transition.state_cfg
const D = deserialize(joinpath(@__DIR__, "lap_frames.jls"))
const FRAMES = D.frames

function timing()
    dt = 6 / 30
    n = length(FRAMES) - 1
    @printf("decisions %d   model time %.1f s   video at 12 fps %.1f s -> %.1fx faster\n",
            n, n * dt, length(FRAMES) / 12, (n * dt) / (length(FRAMES) / 12))
    @printf("real time needs %.1f fps\n", 1 / dt)
end

function duck_audit()
    threat = 0; act = 0; nearest = Inf; classes = Set{Any}()
    for (s, _) in FRAMES
        raw, _ = get_raw_state(s, SCFG)
        push!(classes, raw.duck)
        raw.duck == NONE || (threat += 1)
        d = s.ducks[1]
        d.pedestrian_active && (act += 1)
        nearest = min(nearest, hypot(d.pos[1] - s.ego.pos[1],
                                     d.pos[3] - s.ego.pos[3]))
    end
    @printf("\nduck: threat classes seen %s\n", collect(classes))
    @printf("      frames with a threat %d/%d   pedestrian_active %d   closest %.3f m\n",
            threat, length(FRAMES), act, nearest)
end

function stop_audit()
    seen = 0; zone = 0; minds = Inf
    for (s, _) in FRAMES
        raw, _ = get_raw_state(s, SCFG)
        raw.sigma_stop && (zone += 1)
        if raw.d_stop !== nothing
            seen += 1
            minds = min(minds, raw.d_stop)
        end
    end
    @printf("\nstop sign: d_stop available %d/%d frames   in stop zone %d frames\n",
            seen, length(FRAMES), zone)
    println("           min d_stop: ",
            minds === Inf ? "never a candidate" : @sprintf("%.3f m", minds))
    sg = FRAMES[1][1].stop_signs
    println("           objects in the world state: ", length(sg), " at ",
            [(round(x.pos[1]; digits = 3), round(x.pos[3]; digits = 3)) for x in sg])
    println("           map extent is 0 .. 1.755, so the sign sits OUTSIDE the tile grid")
end

timing()
duck_audit()
stop_audit()

println("""

cost handed to DORA:  1.0 + 4|d| + 0.5|phi|
  duck term:       none
  stop-sign term:  none
So nothing in the objective asks the vehicle to yield or to stop. It was
asked to complete a lap while staying near the lane centre, and that is
exactly what it did.""")
