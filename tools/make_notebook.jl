# Generate examples/quickstart.ipynb FROM examples/quickstart.jl.
#
#     julia --project=. tools/make_notebook.jl
#
# The .jl is the single source of truth: it is what CI executes, so it cannot
# rot. The notebook is derived from it by splitting on jupytext "percent"
# markers, which means the two can never drift apart.
#
#   # %%              -> code cell
#   # %% [markdown]   -> markdown cell (leading "# " stripped)

using JSON3

const SRC = joinpath(@__DIR__, "..", "examples", "quickstart.jl")
const OUT = joinpath(@__DIR__, "..", "examples", "quickstart.ipynb")

function cells(path)
    out = Vector{Any}()
    kind, buf = :code, String[]

    function flush!()
        # drop leading/trailing blank lines without touching interior ones
        while !isempty(buf) && isempty(strip(first(buf)))
            popfirst!(buf)
        end
        while !isempty(buf) && isempty(strip(last(buf)))
            pop!(buf)
        end
        isempty(buf) && return
        src = [line * "\n" for line in buf]
        src[end] = rstrip(src[end], '\n')
        if kind === :markdown
            push!(out, Dict("cell_type" => "markdown", "metadata" => Dict(),
                "source" => src))
        else
            push!(out, Dict("cell_type" => "code", "metadata" => Dict(),
                "execution_count" => nothing, "outputs" => Any[],
                "source" => src))
        end
        empty!(buf)
    end

    for line in eachline(path)
        if startswith(line, "# %% [markdown]")
            flush!()
            kind = :markdown
        elseif startswith(line, "# %%")
            flush!()
            kind = :code
        else
            push!(buf, kind === :markdown ?
                (startswith(line, "# ") ? line[3:end] :
                 line == "#" ? "" : line) : line)
        end
    end
    flush!()
    return out
end

nb = Dict(
    "cells" => cells(SRC),
    "metadata" => Dict(
        "kernelspec" => Dict("display_name" => "Julia 1.11",
            "language" => "julia", "name" => "julia-1.11"),
        "language_info" => Dict("file_extension" => ".jl",
            "mimetype" => "application/julia", "name" => "julia")),
    "nbformat" => 4, "nbformat_minor" => 5)

open(OUT, "w") do io
    JSON3.pretty(io, nb)
end

n = length(nb["cells"])
println("wrote ", relpath(OUT, joinpath(@__DIR__, "..")), " (", n, " cells, ",
    count(c -> c["cell_type"] == "code", nb["cells"]), " code)")
