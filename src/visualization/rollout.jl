# FJ9.4 — rollout comparison driven by the FROZEN FJ8.4b artefacts.
#
#     frozen artifact  ->  validated loader  ->  RolloutAggregate  ->  backend
#
# Never:
#
#     visualisation  ->  load policy  ->  run the environment again
#
# A figure must depict the experiment that was actually reported, not a fresh
# run that happens to share a config. Nothing in this file constructs an MDP,
# loads a checkpoint or steps a transition; the loader reads a CSV and
# validates it, and every number in a figure is traceable to a row of it.
#
# SCOPE, stated because the artefact does not contain everything a rollout
# figure could want: `six_solver_episodes.csv` is EPISODE-LEVEL — 6 solvers x
# 20 seeds x 25 columns. It carries outcomes, event COUNTS and summary
# statistics, but no ego positions, no per-decision series and no event
# timestamps. World-trajectory overlays and speed/phi-versus-decision panels
# therefore cannot be built from it, and are not faked here. Adding them
# requires extending what the experiment records, which is a decision about
# the experiment rather than about the renderer.

"""
    EpisodeOutcome

How an episode ended. The distinction is load-bearing: FJ8.4b's `in_progress`
means **the evaluation horizon was reached while the environment was still
running**, which is not the same event as the environment terminating, and
must not share a marker with it.
"""
@enum EpisodeOutcome ENV_TERMINATED HORIZON_REACHED

"""
    EpisodeRecord

One row of the frozen artefact, typed. Field names match the CSV header, so a
schema change is a load error rather than a silent shift.
"""
struct EpisodeRecord
    solver::String
    family::String
    seed::Int
    decisions::Int
    ret::Float64
    discounted_return::Float64
    progress::Float64
    mean_abs_d::Float64
    max_abs_d::Float64
    mean_abs_phi::Float64
    mean_speed::Float64
    brake_ratio::Float64
    offroad::Bool
    other_collision::Bool
    duck_collision::Bool
    timeout::Bool
    goal::Bool
    full_stops::Int
    stop_violations::Int
    passed_stops::Int
    stop_zone_decisions::Int
    duck_yield_decisions::Int
    duck_active_decisions::Int
    crossings::Int
    reason::String
end

const ROLLOUT_ARTIFACT_SCHEMA = ("solver", "family", "seed", "decisions",
    "return", "discounted_return", "progress", "mean_abs_d", "max_abs_d",
    "mean_abs_phi", "mean_speed", "brake_ratio", "offroad", "other_collision",
    "duck_collision", "timeout", "goal", "full_stops", "stop_violations",
    "passed_stops", "stop_zone_decisions", "duck_yield_decisions",
    "duck_active_decisions", "crossings", "reason")

"""
    outcome(record) -> EpisodeOutcome

`HORIZON_REACHED` when the run stopped because the evaluation horizon expired
with the environment still in progress; `ENV_TERMINATED` when the environment
itself ended the episode.
"""
outcome(r::EpisodeRecord) =
    r.reason == "in_progress" ? HORIZON_REACHED : ENV_TERMINATED

"""
    ArtifactProvenance

Where the data came from and what it is. Two figures that look alike but come
from different experiments must not be interchangeable, so provenance travels
with the data rather than in a filename.
"""
struct ArtifactProvenance
    path::String
    experiment_id::String
    schema_version::String
    rows::Int
    solvers::Vector{String}
    seeds::Vector{Int}
    horizon::Int
    content_fingerprint::String
end

"""
    RolloutAggregate

Every episode of the frozen experiment, with its provenance.
"""
struct RolloutAggregate
    records::Vector{EpisodeRecord}
    provenance::ArtifactProvenance
end

"""
    RolloutComparison

The six solvers at ONE seed — a paired comparison on one initial condition,
which is the structure FJ8.4b was designed around.
"""
struct RolloutComparison
    seed::Int
    records::Vector{EpisodeRecord}
    provenance::ArtifactProvenance
end

_parse_bool(s) = s in ("true", "True", "1")

function _parse_row(fields)
    length(fields) == length(ROLLOUT_ARTIFACT_SCHEMA) ||
        throw(ArgumentError("row has $(length(fields)) fields, expected " *
                            "$(length(ROLLOUT_ARTIFACT_SCHEMA))"))
    f = fields
    return EpisodeRecord(f[1], f[2], parse(Int, f[3]), parse(Int, f[4]),
        parse(Float64, f[5]), parse(Float64, f[6]), parse(Float64, f[7]),
        parse(Float64, f[8]), parse(Float64, f[9]), parse(Float64, f[10]),
        parse(Float64, f[11]), parse(Float64, f[12]),
        _parse_bool(f[13]), _parse_bool(f[14]), _parse_bool(f[15]),
        _parse_bool(f[16]), _parse_bool(f[17]),
        parse(Int, f[18]), parse(Int, f[19]), parse(Int, f[20]),
        parse(Int, f[21]), parse(Int, f[22]), parse(Int, f[23]),
        parse(Int, f[24]), f[25])
end

"""
    load_rollout_artifact(path; experiment_id, horizon) -> RolloutAggregate

Read and **validate** the frozen episode artefact.

Validation is strict on purpose — a renderer must not be the layer that
forgives malformed evidence: the header must match
[`ROLLOUT_ARTIFACT_SCHEMA`](@ref) exactly, every solver must have run the same
seed set, and no episode may exceed the declared horizon.
"""
function load_rollout_artifact(path::AbstractString;
    experiment_id::AbstractString="FJ8.4b", horizon::Integer=150,
    schema_version::AbstractString="fj8.4b-episodes-1")
    isfile(path) || throw(ArgumentError("artifact not found: $path"))
    text = read(path, String)
    lines = filter(!isempty, split(strip(text), "\n"))
    length(lines) >= 2 || throw(ArgumentError("artifact has no rows: $path"))

    header = Tuple(strip.(split(lines[1], ",")))
    header == ROLLOUT_ARTIFACT_SCHEMA || throw(ArgumentError(
        "artifact schema mismatch in $path:\n  got      $(header)\n" *
        "  expected $(ROLLOUT_ARTIFACT_SCHEMA)"))

    records = [_parse_row(strip.(split(l, ","))) for l in lines[2:end]]

    solvers = sort(unique(r.solver for r in records))
    seeds = sort(unique(r.seed for r in records))
    for s in solvers
        rs = sort([r.seed for r in records if r.solver == s])
        rs == seeds || throw(ArgumentError(
            "solver $s ran seeds $rs, but the artefact's seed set is $seeds; " *
            "a paired comparison requires identical seed sets"))
    end
    for r in records
        1 <= r.decisions <= horizon || throw(ArgumentError(
            "$(r.solver) seed $(r.seed) has $(r.decisions) decisions, " *
            "outside 1:$horizon"))
    end

    prov = ArtifactProvenance(String(path), String(experiment_id),
        String(schema_version), length(records), solvers, seeds,
        Int(horizon), string(hash(text); base=16, pad=16))
    return RolloutAggregate(records, prov)
end

"""
    artifact_fingerprint(x) -> String

Content fingerprint of the loaded artefact. Perturbing any value in the file
changes it, so a figure cannot be silently re-attributed to different evidence.
"""
artifact_fingerprint(a::RolloutAggregate) = a.provenance.content_fingerprint
artifact_fingerprint(c::RolloutComparison) = c.provenance.content_fingerprint

"""
    comparison_at_seed(aggregate, seed) -> RolloutComparison

The paired set of episodes at one seed. Throws if the seed is absent — the
caller must name a seed that exists rather than receive a quietly empty figure.
"""
function comparison_at_seed(a::RolloutAggregate, seed::Integer)
    rs = filter(r -> r.seed == seed, a.records)
    isempty(rs) && throw(ArgumentError(
        "seed $seed is not in this artefact; it has $(a.provenance.seeds)"))
    return RolloutComparison(Int(seed), sort(rs; by=r -> r.solver),
        a.provenance)
end

"""
    median_return_seed(aggregate, solver) -> Int

The seed whose return is closest to that solver's median. Offered so a
"representative seed" can be chosen by a stated rule rather than by which
picture looks most dramatic; the rule used must be recorded with the figure.
"""
function median_return_seed(a::RolloutAggregate, solver::AbstractString)
    rs = filter(r -> r.solver == solver, a.records)
    isempty(rs) && throw(ArgumentError("no records for solver $solver"))
    vals = sort([r.ret for r in rs])
    med = vals[cld(length(vals), 2)]
    return rs[argmin(abs(r.ret - med) for r in rs)].seed
end

"""
    paired_metric(aggregate, metric) -> (seeds, Dict{String,Vector{Float64}})

One vector per solver, aligned to a shared, sorted seed list. Column *k* of
every vector is the same initial condition, which is what makes a paired
difference meaningful.
"""
function paired_metric(a::RolloutAggregate, metric::Symbol)
    seeds = a.provenance.seeds
    out = Dict{String,Vector{Float64}}()
    for s in a.provenance.solvers
        bysd = Dict(r.seed => r for r in a.records if r.solver == s)
        out[s] = [Float64(getfield(bysd[sd], metric)) for sd in seeds]
    end
    return seeds, out
end

"""
    stop_compliance_of(records) -> Union{Nothing,Float64}

Compliant stop encounters over stop **encounters**. `nothing` when no stop sign
was ever reached — TD3 in FJ8.4b — and the renderer must carry that through as
`N/A`. Turning a missing rate into `0.0` would invent a failure; turning it
into `1.0` would invent a success.
"""
function stop_compliance_of(rs::AbstractVector{EpisodeRecord})
    enc = sum(r -> r.passed_stops, rs; init=0)
    enc == 0 && return nothing
    return (enc - sum(r -> r.stop_violations, rs; init=0)) / enc
end

"""
    solver_summary(aggregate, solver) -> NamedTuple

Everything the episode-level artefact supports for one solver, with
not-applicable preserved.
"""
function solver_summary(a::RolloutAggregate, solver::AbstractString)
    rs = filter(r -> r.solver == solver, a.records)
    n = length(rs)
    mean_(f) = sum(f, rs) / n
    return (solver=solver, family=rs[1].family, episodes=n,
        mean_return=mean_(r -> r.ret),
        median_return=sort([r.ret for r in rs])[cld(n, 2)],
        mean_length=mean_(r -> r.decisions),
        mean_abs_d=mean_(r -> r.mean_abs_d),
        max_abs_d=maximum(r -> r.max_abs_d, rs),
        mean_abs_phi=mean_(r -> r.mean_abs_phi),
        mean_speed=mean_(r -> r.mean_speed),
        env_terminated=count(r -> outcome(r) === ENV_TERMINATED, rs),
        horizon_reached=count(r -> outcome(r) === HORIZON_REACHED, rs),
        offroad=count(r -> r.offroad, rs),
        collisions=count(r -> r.other_collision || r.duck_collision, rs),
        stop_encounters=sum(r -> r.passed_stops, rs; init=0),
        stop_compliance=stop_compliance_of(rs),
        crossings=sum(r -> r.crossings, rs; init=0))
end

"""
    comparison_table(aggregate) -> String
"""
function comparison_table(a::RolloutAggregate)
    w = maximum(length(s) for s in a.provenance.solvers)
    io = IOBuffer()
    println(io, rpad("solver", w), "  ", lpad("eps", 4), "  ",
        lpad("mean ret", 9), "  ", lpad("med ret", 9), "  ",
        lpad("len", 6), "  ", lpad("env end", 8), "  ", lpad("horizon", 8),
        "  ", lpad("offroad", 8), "  ", lpad("collide", 8), "  ",
        lpad("stop enc", 9), "  ", lpad("compliance", 11))
    println(io, "-"^(w + 92))
    for s in a.provenance.solvers
        t = solver_summary(a, s)
        println(io, rpad(s, w), "  ", lpad(t.episodes, 4), "  ",
            lpad(round(t.mean_return; digits=2), 9), "  ",
            lpad(round(t.median_return; digits=2), 9), "  ",
            lpad(round(t.mean_length; digits=1), 6), "  ",
            lpad(t.env_terminated, 8), "  ", lpad(t.horizon_reached, 8), "  ",
            lpad(t.offroad, 8), "  ", lpad(t.collisions, 8), "  ",
            lpad(t.stop_encounters, 9), "  ",
            lpad(t.stop_compliance === nothing ? "N/A" :
                 string(round(100 * t.stop_compliance; digits=1), "%"), 11))
    end
    return String(take!(io))
end

"""
    provenance_lines(p) -> Vector{String}

The provenance block a figure must carry.
"""
provenance_lines(p::ArtifactProvenance) = [
    "experiment: $(p.experiment_id)",
    "artifact:   $(basename(p.path))",
    "schema:     $(p.schema_version)",
    "content:    $(p.content_fingerprint)",
    "solvers:    $(join(p.solvers, ", "))",
    "seeds:      $(length(p.seeds)) frozen evaluation seeds",
    "horizon:    $(p.horizon) decisions",
]
provenance_lines(a::RolloutAggregate) = provenance_lines(a.provenance)
provenance_lines(c::RolloutComparison) =
    vcat(provenance_lines(c.provenance), ["seed:       $(c.seed)"])
