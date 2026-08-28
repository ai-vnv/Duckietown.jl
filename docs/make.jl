using Documenter
using DuckietownDecisionModels

DocMeta.setdocmeta!(DuckietownDecisionModels, :DocTestSetup,
    :(using DuckietownDecisionModels); recursive = true)

makedocs(
    sitename = "Duckietown.jl",
    modules = [DuckietownDecisionModels],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://pannntastic.github.io/Duckietown.jl",
        assets = String[],
    ),
    pages = [
        "Home" => "index.md",
        "How it was built" => "building.md",
        "Validation record" => [
            "Overview" => "validation/README.md",
            "FJ1 — skeleton & data model" => "validation/FJ1_STATUS.md",
            "FJ2 — semantic parity" => "validation/FJ2_STATUS.md",
            "FJ3 — dynamics & RNG" => "validation/FJ3_STATUS.md",
            "FJ4 — POMDPs.jl interface" => "validation/FJ4_STATUS.md",
            "FJ5 — live reference parity" => "validation/FJ5_STATUS.md",
            "FJ5-R — in-process backend" => "validation/FJ5R_STATUS.md",
            "FJ6 — episode parity" => "validation/FJ6_STATUS.md",
            "FJ7 — trained baselines" => "validation/FJ7_STATUS.md",
            "FJ8 — online planning" => "validation/FJ8_STATUS.md",
            "FJ8.4c — artefact enrichment" => "validation/FJ84C_STATUS.md",
            "FJ9 — visualization" => "validation/FJ9_STATUS.md",
            "FJ9.6 — diagnostics" => "validation/FJ96_STATUS.md",
            "FJ9.7 — animation" => "validation/FJ97_STATUS.md",
            "FJ9.8 — publication figures" => "validation/FJ98_STATUS.md",
            "FJ9.9 — reproducibility closure" => "validation/FJ99_STATUS.md",
            "FJ10 — POMDP readiness" => "validation/FJ10_STATUS.md",
        ],
        "API" => [
            "Configuration" => "api/config.md",
            "Model & observers" => "api/model.md",
            "Dynamics & world" => "api/dynamics.md",
            "Reward & events" => "api/reward.md",
            "Generative transition" => "api/generative.md",
            "Exact RNG" => "api/rng.md",
            "Interfaces" => "api/interfaces.md",
            "Backends" => "api/backends.md",
            "Frozen policies" => "api/solvers.md",
            "Evaluation" => "api/evaluation.md",
            "Visualization" => "api/visualization.md",
        ],
    ],
    # The validation records are working documents that reference repository
    # files (artifacts/, test/) which are not site pages, and not every
    # exported symbol carries a docstring yet — warn, do not fail.
    warnonly = true,
)

deploydocs(
    repo = "github.com/PannnTastic/Duckietown.jl",
    devbranch = "main",
)
