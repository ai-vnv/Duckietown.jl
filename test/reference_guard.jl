# Where the reference material lives, and whether it is here.
#
# This package is a PORT. Its parity gates (FJ2-FJ7) compare against recorded
# outputs from the Python `duckduck` supplementary package, and its evaluation
# gates load that package's frozen `training_config.yaml` and policy
# checkpoints. None of that is distributable from here, so on any machine
# without a sibling `duckduck` checkout those test sets cannot run.
#
# They must SKIP, loudly and by name — never error, and never quietly pass.
# Before this guard existed the suite errored 69 times on a clean checkout,
# which meant it had only ever been runnable by its author. FJ9.9a missed it
# because that check used a clean *depot* with the reference still on disk.

const REFERENCE_ROOT = normpath(joinpath(pkgdir(DuckietownDecisionModels),
    "..", "duckduck"))
const REFERENCE_POLICIES = joinpath(REFERENCE_ROOT, "policies")

"""
    have_reference() -> Bool

True when the `duckduck` supplementary package sits beside this one.
"""
have_reference() = isdir(REFERENCE_POLICIES) &&
    isfile(joinpath(REFERENCE_POLICIES, "q_learning", "training_config.yaml"))

const HAVE_REFERENCE = have_reference()

"""
    reference_config(algorithm) -> String

Path to a frozen reference config. Callers must check [`HAVE_REFERENCE`](@ref)
first; this throws a message that says what is missing rather than a bare
`SystemError` from deep inside a loader.
"""
function reference_config(algorithm::AbstractString)
    p = joinpath(REFERENCE_POLICIES, algorithm, "training_config.yaml")
    isfile(p) || error("""
        reference config not available: $p

        This test needs the `duckduck` supplementary package checked out
        beside this repository. It is not distributed here.""")
    return p
end

if !HAVE_REFERENCE
    @info """
    Reference package not found at $REFERENCE_ROOT.

    Parity and evaluation test sets will SKIP. What still runs: the model
    itself, the POMDPs interface, the visualisation geometry, the audits, and
    everything built on `scenario_config`.
    """
end
