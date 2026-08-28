# Registry-readiness audit: the General/AutoMerge requirements, checked
# mechanically against this repository. Run from anywhere:
#     julia tools/registry_audit.jl

import Pkg
import TOML

root = normpath(joinpath(@__DIR__, ".."))
proj = TOML.parsefile(joinpath(root, "Project.toml"))

fails = String[]
warns = String[]
oks = String[]
check(cond, okmsg, failmsg; warn = false) =
    cond ? push!(oks, okmsg) : push!(warn ? warns : fails, failmsg)

# ── name rules ───────────────────────────────────────────────────────────────
name = proj["name"]
check(length(name) >= 5, "name length $(length(name)) >= 5",
      "name shorter than 5 characters")
check(isuppercase(name[1]), "name starts uppercase", "name must start uppercase")
check(all(c -> isletter(c) || isdigit(c), name) && isascii(name),
      "name is ASCII letters/digits", "name has non-ASCII or punctuation")

# ── version / uuid / authors ────────────────────────────────────────────────
v = proj["version"]
check(v in ("0.0.1", "0.1.0", "1.0.0"),
      "initial version $v is a standard choice",
      "version $v is not a standard initial version (0.0.1 / 0.1.0 / 1.0.0)";
      warn = true)
check(haskey(proj, "uuid"), "uuid present", "uuid missing")
check(haskey(proj, "authors"), "authors present", "authors missing"; warn = true)

# ── compat coverage: every dep and weakdep, with bounded entries ────────────
compat = get(proj, "compat", Dict{String,Any}())
isbounded(spec) = !occursin(">=", string(spec)) && strip(string(spec)) != "*"
for section in ("deps", "weakdeps")
    for d in keys(get(proj, section, Dict{String,Any}()))
        if haskey(compat, d)
            check(isbounded(compat[d]), "compat $d = $(compat[d]) (bounded)",
                  "compat for $d is unbounded: $(compat[d])")
        else
            push!(fails, "no [compat] entry for $section dependency $d")
        end
    end
end
check(haskey(compat, "julia"), "compat julia = $(get(compat, "julia", "?"))",
      "no [compat] entry for julia")

# ── dependencies must be registered in General ───────────────────────────────
general = only(filter(r -> r.name == "General", Pkg.Registry.reachable_registries())
    )
regnames = Set(x.name for x in values(general.pkgs))
for section in ("deps", "weakdeps")
    for (d, u) in get(proj, section, Dict{String,Any}())
        d in ("LinearAlgebra", "Random", "Test", "Printf", "Statistics",
              "Serialization") && continue          # stdlib
        check(d in regnames, "$d is registered in General",
              "$section dependency $d is NOT in the General registry")
    end
end

# ── name similarity / collision with General ────────────────────────────────
check(!(name in regnames), "no exact name collision in General",
      "a package named $name already exists in General")
close = [r for r in regnames if startswith(lowercase(r), "duckietown")]
isempty(close) ? push!(oks, "no Duckietown* names in General") :
    push!(warns, "similar names in General: $(join(close, ", "))")

# ── license ──────────────────────────────────────────────────────────────────
lic = filter(f -> startswith(uppercase(f), "LICENSE"), readdir(root))
check(!isempty(lic), "license file present: $(join(lic, ", "))",
      "NO LICENSE FILE — AutoMerge requires an OSI-approved license")

# ── package layout ───────────────────────────────────────────────────────────
check(isfile(joinpath(root, "src", name * ".jl")),
      "src/$name.jl exists (root package layout)",
      "src/$name.jl not found")
check(isfile(joinpath(root, "test", "runtests.jl")),
      "test/runtests.jl exists", "no test/runtests.jl"; warn = true)

# ── fresh-environment load test (what a new user's Pkg resolution sees) ─────
mktempdir() do tmp
    code = """
        import Pkg
        Pkg.activate("$(escape_string(tmp))")
        Pkg.develop(path="$(escape_string(root))"; io=devnull)
        Pkg.precompile(; io=devnull)
        using DuckietownDecisionModels
        c = scenario_config(:stop_and_duck_safe)
        @assert c.reward.duck_unsafe == -5.0
        println("FRESH_ENV_OK")
    """
    out = read(pipeline(`$(Base.julia_cmd()) --startup-file=no -e $code`;
                        stderr = devnull), String)
    check(occursin("FRESH_ENV_OK", out),
          "loads and runs from a fresh environment (resolver-clean)",
          "failed to load from a fresh environment")
end

# ── report ───────────────────────────────────────────────────────────────────
println("── PASS ($(length(oks))) ──")
foreach(m -> println("  ✓ ", m), oks)
println("── WARN ($(length(warns))) ──")
foreach(m -> println("  ~ ", m), warns)
println("── FAIL ($(length(fails))) ──")
foreach(m -> println("  ✗ ", m), fails)
println(isempty(fails) ? "REGISTRY AUDIT: READY" :
        "REGISTRY AUDIT: NOT READY ($(length(fails)) blocker(s))")
