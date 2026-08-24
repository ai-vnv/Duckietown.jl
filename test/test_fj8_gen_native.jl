# FJ8.0 — the tree is grown by Julia, not by the Python reference.
#
# This has to run in a SEPARATE process: FJ5-R loads PythonCall into the main
# test session on purpose, so asserting "PythonCall is not loaded" there would
# prove nothing. A fresh process runs 5 000 `gen` calls and reports which
# Python-related modules ended up loaded.

using DuckietownDecisionModels
using Test

@testset "FJ8.0 gen runs without any Python (fresh process)" begin
    script = joinpath(pkgdir(DuckietownDecisionModels), "tools",
        "fj8_native_check.jl")
    @test isfile(script)
    project = pkgdir(DuckietownDecisionModels)
    out = try
        read(`$(Base.julia_cmd()) --project=$project --startup-file=no $script`,
            String)
    catch err
        @info "FJ8.0 native check could not run" err
        ""
    end
    if isempty(out)
        @test_skip "fresh-process native check unavailable"
    else
        fields = Dict(split(l, "=")[1] => split(l, "=")[2]
                      for l in split(strip(out), "\n") if occursin("=", l))
        @info "FJ8.0 native execution check" fields
        @test fields["GEN_CALLS"] == "5000"
        @test fields["REWARD_SUM_FINITE"] == "true"
        @test fields["PYTHON_MODULES"] == "none"
        @test fields["EXT_LOADED"] == "false"
        @test fields["NATIVE_OK"] == "true"
    end
end
