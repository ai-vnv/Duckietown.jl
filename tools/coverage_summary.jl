# Summarize .cov files left by tools/coverage_local.sh. Function-wrapped:
# bare top-level accumulator loops hit Julia's soft-scope rule (see the
# repo's own history with tools/*.jl).
using Pkg
Pkg.activate(mktempdir(); io = devnull)
Pkg.add("Coverage"; io = devnull)
using Coverage

function main()
    fcs = process_folder("src")
    rows = Tuple{String,Int,Int}[]
    tot_h = 0
    tot_l = 0
    for f in fcs
        h = count(x -> x !== nothing && x > 0, f.coverage)
        l = count(x -> x !== nothing, f.coverage)
        tot_h += h
        tot_l += l
        push!(rows, (f.filename, h, l))
    end
    sort!(rows; by = r -> r[3] - r[2], rev = true)
    println("== per file (worst absolute gap first) ==")
    for (fn, h, l) in rows
        l == 0 && continue
        println(rpad(fn, 46), lpad(h, 5), "/", rpad(l, 5), "  ",
            round(100h / l; digits = 1), "%")
    end
    println("TOTAL ", tot_h, "/", tot_l, " = ",
        round(100tot_h / tot_l; digits = 2), "%")
end

main()
