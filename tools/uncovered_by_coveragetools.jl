# Print the line numbers CoverageTools itself counts as executable-but-unhit
# for the given files — the exact accounting Codecov receives.
using Pkg
Pkg.activate(mktempdir(); io = devnull)
Pkg.add("Coverage"; io = devnull)
using Coverage

function main(files)
    root = normpath(joinpath(@__DIR__, ".."))
    for f in files
        fc = Coverage.process_file(joinpath(root, f), joinpath(root, dirname(f)))
        src = readlines(joinpath(root, f))
        println("== ", f)
        for (i, c) in enumerate(fc.coverage)
            c === nothing && continue
            c > 0 && continue
            println(lpad(i, 5), "  ", i <= length(src) ? src[i] : "")
        end
    end
end

main(ARGS)
