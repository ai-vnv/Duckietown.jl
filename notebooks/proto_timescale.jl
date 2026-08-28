# Why no state discretization both closes and covers.
#
# Hypothesis: one decision moves the vehicle far less than any cell size whose
# state count stays manageable, so a coarse key gives self-transitions (BFS
# closes at a handful of states) and a fine key gives a new state every step
# (BFS never closes). If that is right, the missing ingredient is not a finer
# grid but a coarser DECISION TIMESCALE.
#
# Measured, not assumed.

using DuckietownDecisionModels
using POMDPs, Random, Printf, Statistics

const BASE = DuckietownMDP(scenario_config(:stop_and_duck); action_space = :discrete)
const TR = BASE.transition
const SCFG = TR.state_cfg
const ACTS = collect(POMDPs.actions(BASE))
const RNG = MersenneTwister(1)

raw_of(s) = first(get_raw_state(s, SCFG))
tile_of(s) = get_grid_coords(s.map, collect(s.ego.pos))
dkey(s) = discretize(raw_of(s))

# 1. how far does one decision actually move the vehicle?
function displacement(; n = 400)
    rng = MersenneTwister(3)
    d = Float64[]
    for _ in 1:n
        s = rand(rng, initialstate(BASE))
        for _ in 1:rand(rng, 1:20)
            r = simulate_decision(TR, s, rand(rng, ACTS), RNG)
            (r.terminated || r.truncated) && break
            s = r.sp
        end
        POMDPs.isterminal(BASE, s) && continue
        sp = simulate_decision(TR, s, FAST_STRAIGHT, RNG).sp
        push!(d, hypot(sp.ego.pos[1] - s.ego.pos[1], sp.ego.pos[3] - s.ego.pos[3]))
    end
    return d
end

disp = displacement()
@printf("displacement of one decision: median %.4f m   max %.4f m\n",
        median(disp), maximum(disp))
@printf("  tile size            %.3f m  -> %5.1f decisions to cross a tile\n",
        0.585, 0.585 / median(disp))
@printf("  d bin of 0.05 m              -> %5.1f decisions to cross a bin\n",
        0.05 / median(disp))
@printf("  d bin of 0.02 m              -> %5.1f decisions to cross a bin\n",
        0.02 / median(disp))

# 2. holding one action, how many decisions until the key changes?
function dwell(key; n = 200, horizon = 120)
    rng = MersenneTwister(5)
    out = Int[]
    for _ in 1:n
        s = rand(rng, initialstate(BASE))
        a = rand(rng, ACTS)
        k0 = key(s)
        t = 0
        for i in 1:horizon
            r = simulate_decision(TR, s, a, RNG)
            t = i
            (r.terminated || r.truncated) && break
            s = r.sp
            key(s) == k0 || break
        end
        push!(out, t)
    end
    return out
end

println("\ndecisions held before the key changes (same action repeated):")
for (name, k) in (("tabular 7-tuple", dkey), ("tile", tile_of),
                  ("tile+d/0.02+phi/0.05",
                   s -> (tile_of(s)..., round(Int, raw_of(s).d / 0.02),
                         round(Int, raw_of(s).phi / 0.05))))
    w = dwell(k)
    @printf("  %-24s median %5.1f   mean %6.1f   share changing in 1 step %.2f\n",
            name, median(w), mean(w), count(==(1), w) / length(w))
end

# 3. if one ACTION were k decisions of the same command, how often does the
#    tabular key then change? that is the option-duration question.
function change_rate(key, repeat; n = 300)
    rng = MersenneTwister(9)
    changed = alive = 0
    for _ in 1:n
        s = rand(rng, initialstate(BASE))
        for _ in 1:rand(rng, 0:15)
            r = simulate_decision(TR, s, rand(rng, ACTS), RNG)
            (r.terminated || r.truncated) && break
            s = r.sp
        end
        POMDPs.isterminal(BASE, s) && continue
        k0 = key(s)
        ok = true
        a = rand(rng, ACTS)
        for _ in 1:repeat
            r = simulate_decision(TR, s, a, RNG)
            if r.terminated || r.truncated
                ok = false; break
            end
            s = r.sp
        end
        ok || continue
        alive += 1
        key(s) == k0 || (changed += 1)
    end
    return changed / max(alive, 1)
end

println("\nprobability the tabular key changes, if one action = k decisions:")
for k in (1, 2, 4, 8, 16, 32)
    @printf("  k = %2d   P(key changes) = %.2f\n", k, change_rate(dkey, k))
end
