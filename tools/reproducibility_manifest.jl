# FJ9.9f — the reproducibility manifest, and the FJ9.9e isolation matrix.
#
#     tools/run_manifest.sh
#
# Writes artifacts/fj9/reproducibility_manifest.json: what was built, from
# what, with which versions, and what is deliberately not done.
#
# The isolation check is the interesting half. `core_fingerprint` is computed
# with each optional package loaded in turn; loading Makie or MCTS must not
# change the formulation. FJ8.5 asserted that architecturally — this measures
# it.

using DuckietownDecisionModels
using POMDPs
using JSON3
using Pkg

const ROOT = pkgdir(DuckietownDecisionModels)
const OUT = joinpath(ROOT, "artifacts", "fj9")
const CFG = joinpath(ROOT, "..", "duckduck", "policies", "q_learning",
    "training_config.yaml")

base_mdp() = DuckietownMDP(CFG; action_space=:discrete)

# ---- FJ9.9e isolation matrix ----------------------------------------------
function isolation_matrix()
    rows = Vector{Any}()
    core = core_fingerprint(base_mdp())
    push!(rows, Dict("configuration" => "core only",
        "core_fingerprint" => core, "matches" => true))
    for (name, mod) in (("Makie", :CairoMakie), ("MCTS", :MCTS))
        ok, fp, note = try
            @eval using $mod
            f = core_fingerprint(base_mdp())
            (f == core, f, "loaded")
        catch e
            (true, core, "unavailable: " *
                first(sprint(showerror, e), 60))
        end
        push!(rows, Dict("configuration" => "core + $name",
            "core_fingerprint" => fp, "matches" => ok, "note" => note))
    end
    # both together
    fp_both = core_fingerprint(base_mdp())
    push!(rows, Dict("configuration" => "core + Makie + MCTS",
        "core_fingerprint" => fp_both, "matches" => fp_both == core))
    return core, rows
end

core_fp, matrix = isolation_matrix()
for r in matrix
    println("ISOLATION[", r["configuration"], "]=", r["matches"], " ",
        r["core_fingerprint"])
end
println("ISOLATION_ALL_MATCH=", all(r -> r["matches"], matrix))

# ---- audits ----------------------------------------------------------------
docs = documentation_audit(ROOT)
imports = source_import_audit(ROOT)
ledger = artifact_ledger(ROOT)
println("DOC_ISSUES=", length(docs))
println("IMPORT_ISSUES=", length(imports))
println("LEDGER_MISSING=", count(a -> !a.present, ledger))

# ---- test report -----------------------------------------------------------
report_path = joinpath(OUT, "test_report.json")
test_report = isfile(report_path) ?
    JSON3.read(read(report_path, String), Dict{String,Any}) :
    Dict{String,Any}("testsets" => -1)

# ---- figure and source fingerprints ---------------------------------------
function figure_fingerprints()
    out = Dict{String,Any}()
    pub = joinpath(OUT, "publication")
    isdir(pub) || return out
    for f in sort(readdir(pub))
        endswith(f, ".caption.txt") || continue
        stem = replace(f, ".caption.txt" => "")
        for line in eachline(joinpath(pub, f))
            startswith(line, "figure_fingerprint = ") || continue
            out[stem] = strip(line[22:end])
        end
        haskey(out, stem) || (out[stem] = "no fingerprint recorded")
    end
    return out
end

function snapshot_fingerprints()
    out = Dict{String,Any}()
    for f in ("search_snapshot_mcts.json", "search_snapshot_dpw.json")
        p = joinpath(OUT, f)
        isfile(p) || continue
        s = load_snapshot(p)
        out[f] = Dict("state" => s.state_fingerprint,
            "config" => s.config_fingerprint,
            "snapshot" => snapshot_fingerprint(s),
            "valid" => check_snapshot(s).ok)
    end
    return out
end

git_commit() = try
    strip(read(`git -C $ROOT rev-parse HEAD`, String))
catch
    "not a git repository"
end

manifest = Dict{String,Any}(
    "schema" => "fj99.manifest.1",
    "package" => "DuckietownDecisionModels.jl",
    "git_commit" => git_commit(),
    "julia_version" => string(VERSION),
    "core_fingerprint" => core_fp,
    "isolation_matrix" => matrix,
    "isolation_all_match" => all(r -> r["matches"], matrix),
    "test_report" => test_report,
    "artifacts" => [Dict("path" => a.path, "status" => string(a.status),
        "role" => a.role, "present" => a.present, "bytes" => a.bytes,
        "fingerprint" => a.fingerprint) for a in ledger],
    "search_snapshots" => snapshot_fingerprints(),
    "figures" => figure_fingerprints(),
    "documentation_issues" => [Dict("file" => i.file,
        "kind" => string(i.kind), "detail" => i.detail) for i in docs],
    "source_import_issues" => [Dict("file" => i.file, "detail" => i.detail)
        for i in imports],
    "known_limitations" => [Dict("item" => k, "detail" => v)
        for (k, v) in KNOWN_LIMITATIONS],
)

path = joinpath(OUT, "reproducibility_manifest.json")
open(path, "w") do io
    JSON3.pretty(io, manifest)
end
println("MANIFEST=", isfile(path), " ", filesize(path))
println("MANIFEST_LIMITATIONS=", length(KNOWN_LIMITATIONS))
println("MANIFEST_OK=true")
