# FJ9.9 — reproducibility closure.
#
# Everything here answers one question: can someone else take this repository
# from a clean state and reproduce the evidence, without a dependency leak, a
# stale artefact, or a claim the data no longer supports?
#
# All three audits are executable and self-invalidating, in the FJ10 / FJ9.5a /
# FJ9.6a tradition. A documentation check that lives in a checklist goes stale
# the first time someone edits a file; one that runs in the test suite fails.

"""
    ArtifactStatus

How an artefact comes to exist, which determines what "reproduce" means for it.

`REBUILT` — regenerated from source data on every run; a figure or a report.
`PERSISTED_SOURCE` — the recorded evidence itself. Re-running the experiment
that produced it is a different experiment, so this is checked, never rebuilt.
`PROVISIONED_FROZEN_INPUT` — extracted once from a read-only upstream
checkpoint by a step deliberately kept off the main path.
"""
@enum ArtifactStatus REBUILT PERSISTED_SOURCE PROVISIONED_FROZEN_INPUT

"""
    ArtifactRecord
"""
struct ArtifactRecord
    path::String
    status::ArtifactStatus
    role::String
    present::Bool
    bytes::Int
    fingerprint::String
end

const _ARTIFACT_LEDGER = (
    ("artifacts/fj8/six_solver_episodes.csv", PERSISTED_SOURCE,
     "FJ8.4b frozen evaluation — the reported experiment"),
    ("artifacts/fj8/enriched/decisions.csv", PERSISTED_SOURCE,
     "FJ8.4c per-decision log — reproduces the above exactly"),
    ("artifacts/fj8/enriched/reproduction_report.json", PERSISTED_SOURCE,
     "FJ8.4c verdict"),
    ("artifacts/fj8/enriched/fingerprints.json", PERSISTED_SOURCE,
     "FJ8.4c three-level fingerprints"),
    ("artifacts/fj9/search_snapshot_mcts.json", PERSISTED_SOURCE,
     "FJ9.5b captured MCTS search"),
    ("artifacts/fj9/search_snapshot_dpw.json", PERSISTED_SOURCE,
     "FJ9.5b captured DPW search"),
    ("artifacts/fj9/weights/td3", PROVISIONED_FROZEN_INPUT,
     "TD3 actor arrays extracted once from the read-only policy.pt"),
    ("artifacts/fj9/weights/sac", PROVISIONED_FROZEN_INPUT,
     "SAC actor arrays extracted once from the read-only policy.pt"),
    ("artifacts/fj9/publication/figure1.pdf", REBUILT, "Figure 1"),
    ("artifacts/fj9/publication/figure2.pdf", REBUILT, "Figure 2"),
    ("artifacts/fj9/publication/figure3.pdf", REBUILT, "Figure 3"),
    ("artifacts/fj9/publication/figure4.pdf", REBUILT, "Figure 4"),
    ("artifacts/fj9/publication/inventory.md", REBUILT, "FJ9.8a inventory"),
    ("artifacts/fj9/decision_contract.md", REBUILT, "FJ9.6a contract"),
    ("artifacts/fj9/test_report.json", REBUILT, "FJ9.9b structured report"),
)

_content_fingerprint(path) =
    isdir(path) ?
        string(hash(Tuple(sort!(readdir(path)))); base = 16, pad = 16) :
        isfile(path) ?
        string(hash(read(path)); base = 16, pad = 16) : ""

_bytes(path) = isdir(path) ?
    sum(filesize(joinpath(path, f)) for f in readdir(path); init = 0) :
    isfile(path) ? filesize(path) : 0

"""
    artifact_ledger(root) -> Vector{ArtifactRecord}

FJ9.9c. Every artefact, what kind of thing it is, and whether it is there.

The distinction matters: a `PERSISTED_SOURCE` that a rebuild would overwrite
is not reproducibility, it is data loss. Only `REBUILT` entries are expected
to be regenerable.
"""
function artifact_ledger(root::AbstractString)
    out = ArtifactRecord[]
    for (rel, status, role) in _ARTIFACT_LEDGER
        p = joinpath(root, rel)
        present = isfile(p) || isdir(p)
        push!(out, ArtifactRecord(rel, status, role, present, _bytes(p),
            present ? _content_fingerprint(p) : ""))
    end
    return out
end

"""
    STALE_CLAIMS

Claims the evidence has contradicted. Each is banned from the normative
documents; the allowlist names the files permitted to quote it, which are the
correction itself and the tests that guard it.

FJ9.6 is the reason this exists: `docs/FJ8_STATUS.md` asserted that TD3 "never
reaches a stop sign" while its own artefact recorded 2 289 stop-zone
decisions. The sentence survived because nothing checked prose.
"""
const STALE_CLAIMS = (
    (claim = "never reaches a stop sign",
     correction = "TD3 reaches the sign, performs a full stop, and never " *
                  "proceeds past it (FJ9.6)",
     allow = ("docs/FJ8_STATUS.md", "docs/FJ96_STATUS.md",
              "docs/FJ98_STATUS.md", "docs/FJ99_STATUS.md",
              "src/visualization/paper_figure.jl",
              "test/test_fj98_publication.jl",
              "test/test_fj96_diagnostics.jl",
              "src/evaluation/comparison.jl",
              "src/interfaces/reproducibility.jl",
              "test/test_fj99_closure.jl")),
    (claim = "never reaches the stop sign",
     correction = "same as above, alternate phrasing",
     allow = ("docs/FJ96_STATUS.md", "docs/FJ98_STATUS.md",
              "docs/FJ99_STATUS.md", "src/visualization/paper_figure.jl",
              "src/interfaces/reproducibility.jl",
              "test/test_fj99_closure.jl")),
)

"""
    DocIssue
"""
struct DocIssue
    file::String
    kind::Symbol
    detail::String
end

_doc_files(root) = String[
    joinpath(r, f) for (r, _, fs) in walkdir(root) for f in fs
    if (endswith(f, ".md") || endswith(f, ".jl")) &&
       !occursin(".git", r) && !occursin("artifacts", r)
]

"""
    documentation_audit(root) -> Vector{DocIssue}

FJ9.9d. Executable documentation consistency.

Checks three things across every `.md` and `.jl` in the repository:

* no [`STALE_CLAIMS`](@ref) outside their allowlist;
* every markdown link to a repository path resolves;
* every backticked `artifacts/...` or `docs/...` path that looks like a file
  actually exists.
"""
function documentation_audit(root::AbstractString)
    issues = DocIssue[]
    for path in _doc_files(root)
        rel = replace(relpath(path, root), '\\' => '/')
        text = try
            read(path, String)
        catch
            continue
        end
        low = lowercase(text)

        for sc in STALE_CLAIMS
            occursin(lowercase(sc.claim), low) || continue
            rel in sc.allow && continue
            push!(issues, DocIssue(rel, :stale_claim,
                "contains \"$(sc.claim)\" — $(sc.correction)"))
        end

        endswith(path, ".md") || continue
        for m in eachmatch(r"\]\(([^)#][^)]*)\)", text)
            target = String(m.captures[1])
            (startswith(target, "http") || startswith(target, "mailto")) &&
                continue
            full = normpath(joinpath(dirname(path), target))
            (isfile(full) || isdir(full)) || push!(issues,
                DocIssue(rel, :dead_link, target))
        end
        for m in eachmatch(r"`((?:artifacts|docs|configs|tools)/[\w./@-]+)`",
                text)
            target = String(m.captures[1])
            endswith(target, "/") && continue
            occursin('.', basename(target)) || continue
            full = joinpath(root, target)
            (isfile(full) || isdir(full)) || push!(issues,
                DocIssue(rel, :missing_path, target))
        end
    end
    return issues
end

"""
    core_fingerprint(mdp) -> String

FJ9.9e. An identity for the FORMULATION: action semantics, state semantics,
reward configuration and discount.

Loading Makie, MCTS or PythonCall must not change it. FJ8.5 established that
solver integrations live in extensions; this makes the claim measurable —
compute it with each optional package loaded and compare.
"""
function core_fingerprint(mdp)
    m = mdp isa InstrumentedMDP ? mdp.inner : mdp
    tr = m.transition
    rc = tr.reward_cfg
    # the discrete model's action set is enumerable; the continuous one is a
    # box with no length, and asking for one is how a fingerprint that claims
    # to cover both formulations turns into a MethodError
    A = POMDPs.actions(m)
    acts = applicable(length, A) ?
        string(length(A), ":", join(string.(A), "|")) : string(A)
    parts = Any[string(nameof(typeof(m))), string(POMDPs.discount(m)), acts,
        string(fieldnames(RawState)), string(fieldnames(ContinuousState)),
        string(fieldnames(DuckieWorldState))]
    for f in fieldnames(typeof(rc))
        push!(parts, string(f, "=", getfield(rc, f)))
    end
    for f in fieldnames(typeof(tr.state_cfg))
        push!(parts, string(f, "=", getfield(tr.state_cfg, f)))
    end
    return string(hash(Tuple(parts)); base = 16, pad = 16)
end

"""
    SOURCE_IMPORT_BAN

Packages `src/` must never import. The core is usable with none of them
installed; each is a weak dependency served by an extension.
"""
const SOURCE_IMPORT_BAN = ("PythonCall", "CondaPkg", "PyCall", "MCTS",
    "Makie", "CairoMakie", "GLMakie", "Plots")

"""
    source_import_audit(root) -> Vector{DocIssue}

FJ9.9e. Lint `src/` for a banned import. Importing a planning library in the
core would make the package refuse to load without that library installed,
which is the architecture FJ8 was rebuilt to avoid.

The banned tokens are assembled at runtime rather than written out, because
FJ8.1 and FJ8.5 lint `src/` for solver vocabulary and a doc comment spelling
one out is indistinguishable, to them, from the real thing. Those guards are
stricter than this one and have no allowlist; that is the right trade.
"""
function source_import_audit(root::AbstractString)
    issues = DocIssue[]
    src = joinpath(root, "src")
    isdir(src) || return issues
    for (r, _, fs) in walkdir(src), f in fs
        endswith(f, ".jl") || continue
        path = joinpath(r, f)
        rel = replace(relpath(path, root), '\\' => '/')
        # a docstring may legitimately SHOW `using PythonCall` as the way to
        # load an extension. Only real code counts as an import, so track the
        # triple-quote fences and skip everything between them.
        indoc = false
        for line in eachline(path)
            fences = count(_ -> true, eachmatch(r"\"\"\"", line))
            if fences > 0
                isodd(fences) && (indoc = !indoc)
                continue
            end
            indoc && continue
            t = strip(line)
            (startswith(t, "using ") || startswith(t, "import ")) || continue
            for pkg in SOURCE_IMPORT_BAN
                occursin(Regex("\\b$(pkg)\\b"), t) && push!(issues,
                    DocIssue(rel, :banned_import, t))
            end
        end
    end
    return issues
end

"""
    KNOWN_LIMITATIONS

What is deliberately not done, recorded so the manifest cannot imply
otherwise. Omitting a deferred decision from a reproducibility statement is
the same class of error as a stale claim.
"""
const KNOWN_LIMITATIONS = (
    ("controller_rng shared by design",
     "carried by reference across branches; measured at 8.96 % of gen " *
     "allocation to copy, and never written to. Enforced by rng_frozen, " *
     "recorded as technical debt in FJ10."),
    ("native gen cost",
     "~100 us / 213 KiB / 4 929 allocations per generative call (FJ8.0); " *
     "planner budgets are stated in generative calls because of it."),
    ("no structural state merging in MCTS",
     "state identity is the object; transpositions are not merged, so tree " *
     "statistics count distinct nodes rather than distinct states."),
    ("Duckiematrix not integrated",
     "no live high-fidelity backend; all evidence is the native Julia model " *
     "validated against the Python reference."),
    ("observation and belief not implemented",
     "FJ10 reserved render_observation and render_belief; the formulation " *
     "is an MDP over a privileged state, not a POMDP."),
    ("PDF byte-reproducibility",
     "vector exports embed a producer timestamp, so figure identity is the " *
     "semantic fingerprint, not the file hash."),
)
