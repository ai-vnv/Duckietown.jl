# DORA given the model AS IS, under the FJ8.2 protocol.
#
# FJ8 established the rule: a solver is handed `DuckietownMDP` unchanged, and
# the compatibility claim rests on the ecosystem's own checker rather than on
# ours. MCTS.jl was validated that way. DORA gets the identical treatment —
# no abstraction, no discretization, no bridge.
#
# Everything I built before this was me reshaping the model to fit the solver,
# which is precisely what FJ8 forbade.

using DuckietownDecisionModels
using POMDPs, POMDPTools, DORASolvers
using Random, Printf

mdp = DuckietownMDP(scenario_config(:stop_and_duck); action_space = :discrete)
s0 = rand(MersenneTwister(1001), initialstate(mdp))

println("="^72)
println("1. What the model declares (FJ8.1 capability report)")
println("="^72)
caps = model_capabilities(mdp)
for (k, v) in pairs(caps)
    @printf("  %-28s %s\n", string(k), string(v))
end

println()
println("="^72)
println("2. What DORA needs, checked against the model")
println("="^72)
checks = (
    ("explicit transition distribution",
     hasmethod(POMDPs.transition, Tuple{typeof(mdp),DuckieWorldState,MacroAction})),
    ("enumerable state space (states)",
     hasmethod(POMDPs.states, Tuple{typeof(mdp)})),
    ("state index (stateindex)",
     hasmethod(POMDPs.stateindex, Tuple{typeof(mdp),DuckieWorldState})),
    ("discrete action space (actions)",
     hasmethod(POMDPs.actions, Tuple{typeof(mdp)})),
    ("generative model (gen)",
     hasmethod(POMDPs.gen, Tuple{typeof(mdp),DuckieWorldState,MacroAction,
                                 AbstractRNG})),
)
for (name, ok) in checks
    @printf("  %-34s %s\n", name, ok ? "[OK]" : "[NO]")
end
@printf("  %-34s %s (DORA treats the model as an undiscounted SSP)\n",
        "discount == 1", POMDPs.discount(mdp) == 1.0 ? "[OK]" : "[NO]")

println()
println("="^72)
println("3. Hand DORA the model unchanged, and record what happens")
println("="^72)
verdict = try
    planner = solve(DORASolver(start = s0), mdp)
    "solve returned a planner with $(planner.tab.S) tabular states"
catch e
    msg = sprint(showerror, e)
    "REFUSED: " * first(split(msg, "\n")[1], 200)
end
println("  ", verdict)

println()
println("="^72)
println("4. The ecosystem's own verdict (POMDPLinter), as in FJ8.2")
println("="^72)
ok = try
    @eval using POMDPLinter
    true
catch err
    println("  POMDPLinter unavailable: ", first(sprint(showerror, err), 100))
    false
end

if ok
    # MCTS declares its requirements on `action`; check whether DORA declares
    # any at all, and on which entry point.
    for (label, f, args) in (("solve", POMDPs.solve, (DORASolver(start = s0), mdp)),)
        r = try
            POMDPLinter.get_requirements(f, map(typeof, args) |> Tuple)
        catch e
            e
        end
        @printf("  requirements declared on %-6s : %s\n", label,
                r isa POMDPLinter.RequirementSet ? "RequirementSet" : string(typeof(r)))
    end

    # And the check FJ8.2 actually cares about: which POMDPs functions this
    # model implements, of those an SSP solver would need.
    println()
    println("  implemented(f, types) for the interface DORA relies on:")
    for (f, types) in ((POMDPs.transition, Tuple{typeof(mdp),DuckieWorldState,MacroAction}),
                       (POMDPs.states, Tuple{typeof(mdp)}),
                       (POMDPs.stateindex, Tuple{typeof(mdp),DuckieWorldState}),
                       (POMDPs.actions, Tuple{typeof(mdp)}),
                       (POMDPs.reward, Tuple{typeof(mdp),DuckieWorldState,MacroAction}),
                       (POMDPs.isterminal, Tuple{typeof(mdp),DuckieWorldState}))
        impl = try
            POMDPLinter.implemented(f, types)
        catch
            false
        end
        @printf("    %-6s %s\n", impl ? "[OK]" : "[NO]",
                string(f) * "(" * join(collect(types.parameters), ", ") * ")")
    end
end

println()
println("="^72)
println("5. For contrast: the same three questions for MCTS")
println("="^72)
mok = try
    @eval using MCTS
    true
catch
    false
end
if !mok
    println("  MCTS unavailable in this environment")
else
    p = solve(MCTSSolver(n_iterations = 20, depth = 10), mdp)
    im = InstrumentedMDP(mdp)
    pm = solve(MCTSSolver(n_iterations = 20, depth = 10), im)
    reset_model_calls!(im)
    a = POMDPs.action(pm, s0)
    @printf("  MCTS solved the model unchanged: action %s after %d gen calls\n",
            string(a), model_calls(im))
    println("  MCTS needs only the GENERATIVE interface; it samples the model.")
end
