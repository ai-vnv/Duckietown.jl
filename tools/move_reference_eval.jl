# One-shot refactor: move the functions that REQUIRE a live reference backend
# (compare_step, matched_state_sweep, rollout_reference) out of
# src/evaluation/ into src/backends/reference_parity.jl, where the rest of
# the reference-bridge code lives. Pure text move — no line is edited.

function docstring_open(lines, fnline)
    # the docstring closes on the line immediately above the function; scan
    # up for its opening `"""` line
    fnline >= 2 && strip(lines[fnline - 1]) == "\"\"\"" ||
        error("no docstring directly above line $fnline")
    i = fnline - 2
    while i >= 1
        startswith(lines[i], "\"\"\"") && return i
        i -= 1
    end
    error("unterminated docstring above line $fnline")
end

function cut_block!(lines, fn_pattern, stop_pattern)
    fnline = findfirst(l -> occursin(fn_pattern, l), lines)
    fnline === nothing && error("pattern not found: $fn_pattern")
    from = docstring_open(lines, fnline)
    stopline = findfirst(l -> occursin(stop_pattern, l), lines)
    stopline === nothing && error("stop pattern not found: $stop_pattern")
    to = docstring_open(lines, stopline) - 1
    block = lines[from:to]
    deleteat!(lines, from:to)
    return block
end

root = normpath(joinpath(@__DIR__, ".."))

par = readlines(joinpath(root, "src/evaluation/parity.jl"))
par_block = cut_block!(par, r"^function compare_step\(",
    r"^function parity_summary\(")
write(joinpath(root, "src/evaluation/parity.jl"), join(par, "\n") * "\n")

rol = readlines(joinpath(root, "src/evaluation/rollout.jl"))
rol_block = cut_block!(rol, r"^function rollout_reference\(",
    r"^struct DriftReport$")
write(joinpath(root, "src/evaluation/rollout.jl"), join(rol, "\n") * "\n")

header = """
# The evaluation functions that REQUIRE a live reference backend: matched
# one-step comparison (FJ5) and the reference side of episode rollouts (FJ6).
# They live in backends/ because not one of their lines can execute without
# the reference installation — the same reason the transport bridges do.
# Moved verbatim from src/evaluation/{parity,rollout}.jl; behaviour, names
# and exports are unchanged, and the gated FJ5/FJ6 suites still run them.
"""

open(joinpath(root, "src/backends/reference_parity.jl"), "w") do io
    print(io, header, "\n")
    print(io, join(par_block, "\n"), "\n\n")
    print(io, join(rol_block, "\n"), "\n")
end

println("moved: ", length(par_block), " + ", length(rol_block), " lines")
