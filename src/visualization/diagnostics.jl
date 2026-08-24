# FJ9.6 — diagnostic time series from the frozen FJ8.4c decision log.
#
#     decisions.csv -> validated loader -> EpisodeDiagnostics -> backend
#
# Nothing here runs a policy, a planner, a transition or an environment. The
# enriched log is the sole evidence, and FJ8.4c showed it re-aggregates to the
# FJ8.4b episode artefact exactly, so a figure drawn from it depicts the
# experiment that was reported rather than a fresh one wearing its name.
#
# Units, categories and semantics are decided HERE, as in FJ9.2. A backend
# left to decide for itself whether a column is metres, radians, a flag or a
# cumulative quantity can quietly disagree with the model.
#
# `missing` stays missing. `d_stop` is absent on 12 814 of 16 522 rows because
# no stop sign was a candidate, which is a different statement from `d_stop`
# being zero, and the difference must survive all the way to the axis.

"""
    FieldAvailability

`LOGGED` — the artefact records the quantity directly.
`DERIVED_IDENTITY` — computed here from logged columns by an exact identity
that is part of the schema (a running sum), never by inference.
`FIELD_ABSENT` — the experiment never stored it. Reported as ABSENT and left
absent; FJ9.4 and FJ9.5a set the precedent.
"""
@enum FieldAvailability LOGGED DERIVED_IDENTITY FIELD_ABSENT

availability_label(a::FieldAvailability) =
    a === LOGGED ? "LOGGED" :
    a === DERIVED_IDENTITY ? "DERIVED (exact identity)" : "ABSENT"

"""
    DecisionFieldItem

One line of the FJ9.6a contract: a quantity a diagnostic figure would want,
the column that supplies it, and what the artefact actually holds.

`blanks` is the count of rows where the column is empty — a number rather
than a sentence, so a consumer never has to read the prose to find out how
much of a column is missing.
"""
struct DecisionFieldItem
    quantity::String
    column::String
    status::FieldAvailability
    blanks::Int
    evidence::String
end

"""
    SeriesCategory

Which panel a series belongs on. Quantities with different units must not
share an axis: normalising heading, speed, reward and model calls onto one
axis looks compact and destroys every physical scale on it.
"""
@enum SeriesCategory NAVIGATION MOTION_COMMAND STOP_SUBSYS DUCK_SUBSYS REWARD COMPUTE

"""
    SeriesKind

`INSTANTANEOUS` — the value at that decision.
`CUMULATIVE` — a running total, derived here and labelled as derived.
`FLAG` — 0/1 state, to be shaded rather than drawn as a line.
"""
@enum SeriesKind INSTANTANEOUS CUMULATIVE FLAG

"""
    AxisMode

`ABSOLUTE_DECISION` — x is the decision index, 1..T.
`NORMALIZED_PROGRESS` — x is `(k-1)/T`, episode progress.

Which one a figure uses changes what it claims, so it is a stated property of
the figure, never a rendering detail. Normalised progress is *not* time: two
episodes at progress 0.5 have taken different numbers of decisions.
"""
@enum AxisMode ABSOLUTE_DECISION NORMALIZED_PROGRESS

"""
    DiagnosticSeries

One named quantity over one episode, carrying everything a renderer needs and
nothing it has to infer.
"""
struct DiagnosticSeries
    name::String
    values::Vector{Union{Float64,Missing}}
    unit::String
    category::SeriesCategory
    kind::SeriesKind
    semantics::String
end

Base.length(s::DiagnosticSeries) = length(s.values)
n_missing(s::DiagnosticSeries) = count(ismissing, s.values)

"""
    EpisodeDiagnostics

One episode's series, its logged events, its outcome and the provenance that
ties a figure back to the artefact it came from.
"""
struct EpisodeDiagnostics
    solver::String
    seed::Int
    decisions::Vector{Int}
    progress::Vector{Float64}
    series::Vector{DiagnosticSeries}
    events::Vector{Pair{String,Vector{Int}}}
    outcome::EpisodeOutcome
    reason::String
    source_fingerprint::String
end

"""
    DECISION_LOG_REQUIRED

Columns without which the log cannot be interpreted. A renderer must not be
the layer that tolerates a missing column.
"""
const DECISION_LOG_REQUIRED = ("solver", "seed", "decision", "ego_x", "ego_z",
    "ego_angle", "ego_speed", "action_kind", "action_id", "v_cmd", "omega_cmd",
    "d", "phi", "v", "kappa", "d_stop", "sigma_stop", "stop_hold_progress",
    "duck_present", "duck_longitudinal", "duck_lateral", "duck_active",
    "reward_total", "reward_progress", "reward_lateral", "reward_heading",
    "reward_events", "full_stop", "passed_stop", "stop_violation", "offroad", "other_collision",
    "duck_collision", "goal", "timeout", "terminated", "truncated", "reason",
    "planning_time", "model_calls")

"""
    DecisionLog

The enriched log, parsed, validated, and fingerprinted by content.
"""
struct DecisionLog
    header::Vector{String}
    rows::Vector{Vector{String}}
    index::Dict{String,Int}
    episodes::Vector{Tuple{String,Int}}
    fingerprint::String
    path::String
end

Base.length(l::DecisionLog) = length(l.rows)

"""
    load_decision_log(path) -> DecisionLog

Read and validate `decisions.csv`.

Validation is strict, as in FJ9.4. Every required column must be present;
every row must have the full width; and within each `(solver, seed)` episode
the decision indices must run contiguously from 1, with the terminal flag —
if it appears at all — appearing only on the final decision. An episode with
a gap in it is a corrupt record, not a short episode.
"""
function load_decision_log(path::AbstractString)
    isfile(path) || throw(ArgumentError("decision log not found: $path"))
    text = read(path, String)
    lines = filter(!isempty, split(strip(text), "\n"))
    length(lines) >= 2 || throw(ArgumentError("decision log has no data rows"))
    header = String.(strip.(split(lines[1], ",")))
    idx = Dict(h => k for (k, h) in enumerate(header))
    for c in DECISION_LOG_REQUIRED
        haskey(idx, c) || throw(ArgumentError(
            "decision log is missing the required column '$c'"))
    end
    rows = Vector{String}[]
    for (k, l) in enumerate(lines[2:end])
        r = String.(split(l, ","))
        length(r) == length(header) || throw(ArgumentError(
            "row $k has $(length(r)) fields, expected $(length(header))"))
        push!(rows, r)
    end

    order = Tuple{String,Int}[]
    byep = Dict{Tuple{String,Int},Vector{Int}}()
    for (k, r) in enumerate(rows)
        key = (r[idx["solver"]], parse(Int, r[idx["seed"]]))
        haskey(byep, key) || push!(order, key)
        push!(get!(byep, key, Int[]), k)
    end
    for key in order
        ks = byep[key]
        ds = [parse(Int, rows[k][idx["decision"]]) for k in ks]
        sort(ds) == collect(1:length(ds)) || throw(ArgumentError(
            "episode $(key[1]) seed $(key[2]) has non-contiguous decision " *
            "indices ($(length(ds)) rows spanning $(extrema(ds)))"))
        term = [rows[k][idx["terminated"]] == "true" for k in ks[sortperm(ds)]]
        any(view(term, 1:length(term)-1)) && throw(ArgumentError(
            "episode $(key[1]) seed $(key[2]) is terminated before its " *
            "final decision"))
    end

    return DecisionLog(header, rows, idx, order,
        string(hash(text); base = 16, pad = 16), String(path))
end

_num(x) = isempty(x) ? missing : parse(Float64, x)
_flag(x) = x == "true" || x == "1"
_flagf(x) = _flag(x) ? 1.0 : 0.0

# name, column, unit, category, kind, semantics
const DIAGNOSTIC_SERIES_SPEC = (
    ("d", "d", "m", NAVIGATION, INSTANTANEOUS,
     "lateral offset from the lane centre, after the transition"),
    ("phi", "phi", "rad", NAVIGATION, INSTANTANEOUS,
     "heading error relative to the lane, after the transition; a different " *
     "coordinate space from d and never to be summed with it"),
    ("kappa", "kappa", "1/m", NAVIGATION, INSTANTANEOUS,
     "signed curvature of the lane ahead (map-privileged)"),
    ("ego_speed", "ego_speed", "m/s", MOTION_COMMAND, INSTANTANEOUS,
     "speed of the state the decision was taken FROM (pre-transition)"),
    ("v", "v", "m/s", MOTION_COMMAND, INSTANTANEOUS,
     "realised lane-frame speed after the transition"),
    ("v_cmd", "v_cmd", "m/s", MOTION_COMMAND, INSTANTANEOUS,
     "commanded forward speed"),
    ("omega_cmd", "omega_cmd", "rad/s", MOTION_COMMAND, INSTANTANEOUS,
     "commanded angular rate"),
    ("d_stop", "d_stop", "m", STOP_SUBSYS, INSTANTANEOUS,
     "distance to the stop line; MISSING when no sign is a candidate, which " *
     "is not the same as zero and must not be drawn as zero"),
    ("sigma_stop", "sigma_stop", "", STOP_SUBSYS, FLAG,
     "the stop-zone indicator the reward and events used"),
    ("stop_hold_progress", "stop_hold_progress", "", STOP_SUBSYS,
     INSTANTANEOUS, "the StopTracker's hold counter — agent memory, not a " *
     "world quantity"),
    ("duck_present", "duck_present", "", DUCK_SUBSYS, FLAG,
     "a duck is in the observation"),
    ("duck_active", "duck_active", "", DUCK_SUBSYS, FLAG,
     "the duck is crossing, as the projections saw it"),
    ("duck_longitudinal", "duck_longitudinal", "m", DUCK_SUBSYS,
     INSTANTANEOUS, "duck position along the lane, relative to the ego"),
    ("duck_lateral", "duck_lateral", "m", DUCK_SUBSYS, INSTANTANEOUS,
     "duck position across the lane, relative to the ego"),
    ("reward_total", "reward_total", "", REWARD, INSTANTANEOUS,
     "reward for this decision"),
    ("reward_progress", "reward_progress", "", REWARD, INSTANTANEOUS,
     "the progress term"),
    ("reward_lateral", "reward_lateral", "", REWARD, INSTANTANEOUS,
     "the lateral-error term"),
    ("reward_heading", "reward_heading", "", REWARD, INSTANTANEOUS,
     "the heading-error term"),
    ("reward_events", "reward_events", "", REWARD, INSTANTANEOUS,
     "the terminal/event term"),
    ("model_calls", "model_calls", "calls", COMPUTE, INSTANTANEOUS,
     "generative-model calls consumed by this decision; 0 is a measurement " *
     "meaning the policy planned nothing, not a missing value"),
    ("planning_time", "planning_time", "s", COMPUTE, INSTANTANEOUS,
     "wall-clock seconds spent producing this decision"),
)

"""
    DIAGNOSTIC_EVENT_COLUMNS

Point events. Markers come from these logged indices and from nothing else —
FJ9.1 showed how convincing an inferred marker looks while corresponding to
nothing the model computes.
"""
const DIAGNOSTIC_EVENT_COLUMNS = ("full_stop", "passed_stop", "stop_violation",
    "offroad", "other_collision", "duck_collision", "goal", "timeout")

"""
    episode_diagnostics(log, solver, seed) -> EpisodeDiagnostics

Extract one episode. Values are copied from the artefact, never recomputed,
and `d_stop` keeps its `missing` where the log recorded no candidate.

The outcome is read from the record rather than assumed: a terminal row means
`ENV_TERMINATED`; a final row still reading `in_progress` means the evaluator
stopped at its horizon, `HORIZON_REACHED`. In this artefact `truncated` is
`false` on every one of the 16 522 rows, so it carries no signal and is not
used to make the distinction.
"""
function episode_diagnostics(log::DecisionLog, solver::AbstractString,
        seed::Integer)
    i = log.index
    rows = filter(r -> r[i["solver"]] == solver &&
                       parse(Int, r[i["seed"]]) == seed, log.rows)
    isempty(rows) && throw(ArgumentError("no episode for solver '$solver' " *
        "seed $seed in $(basename(log.path))"))
    sort!(rows; by = r -> parse(Int, r[i["decision"]]))
    ds = [parse(Int, r[i["decision"]]) for r in rows]
    T = length(ds)
    progress = [(k - 1) / T for k in 1:T]

    series = DiagnosticSeries[]
    for (name, col, unit, cat, kind, sem) in DIAGNOSTIC_SERIES_SPEC
        haskey(i, col) || continue
        vals = kind === FLAG ?
            Union{Float64,Missing}[_flagf(r[i[col]]) for r in rows] :
            Union{Float64,Missing}[_num(r[i[col]]) for r in rows]
        push!(series, DiagnosticSeries(name, vals, unit, cat, kind, sem))
    end

    # the one derived series, labelled as derived
    running = 0.0
    cum = Union{Float64,Missing}[]
    for r in rows
        running += parse(Float64, r[i["reward_total"]])
        push!(cum, running)
    end
    push!(series, DiagnosticSeries("cumulative_return", cum, "", REWARD,
        CUMULATIVE, "running sum of reward_total; derived here by an exact " *
        "identity, and its final value is the episode return FJ8.4b reported"))

    events = Pair{String,Vector{Int}}[]
    for c in DIAGNOSTIC_EVENT_COLUMNS
        haskey(i, c) || continue
        ks = [ds[k] for k in 1:T if _flag(rows[k][i[c]])]
        isempty(ks) || push!(events, c => ks)
    end

    final = rows[end]
    reason = final[i["reason"]]
    outcome = _flag(final[i["terminated"]]) ? ENV_TERMINATED : HORIZON_REACHED
    return EpisodeDiagnostics(String(solver), Int(seed), ds, progress, series,
        events, outcome, reason, log.fingerprint)
end

"""
    series_named(ep, name) -> DiagnosticSeries
"""
series_named(ep::EpisodeDiagnostics, name::AbstractString) =
    only(filter(s -> s.name == name, ep.series))

"""
    series_in(ep, category) -> Vector{DiagnosticSeries}
"""
series_in(ep::EpisodeDiagnostics, c::SeriesCategory) =
    filter(s -> s.category === c, ep.series)

"""
    diagnostics_fingerprint(ep; fields, mode) -> String

Figure identity: the source artefact, the episode, the series selected, and
the x-axis mode. The same selection over a different artefact is a different
figure, and so is a different selection over the same one.
"""
diagnostics_fingerprint(ep::EpisodeDiagnostics;
        fields::AbstractVector{<:AbstractString} = String[],
        mode::AxisMode = ABSOLUTE_DECISION) =
    string(hash((ep.source_fingerprint, ep.solver, ep.seed,
        sort(String[String(f) for f in fields]), Int(mode)));
        base = 16, pad = 16)

"""
    diagnostics_provenance(ep) -> Vector{String}

The lines a figure must carry to be re-derivable.
"""
function diagnostics_provenance(ep::EpisodeDiagnostics)
    parts = String[]
    for (k, v) in ep.events
        push!(parts, k * " @ " * join(v, ","))
    end
    ev = isempty(parts) ? "none logged" : join(parts, "; ")
    return ["solver: $(ep.solver)   seed: $(ep.seed)",
        "decisions: $(length(ep.decisions))   outcome: $(ep.outcome)" *
        "   reason: $(ep.reason)",
        "events: $ev",
        "source: FJ8.4c decisions.csv  fingerprint $(ep.source_fingerprint)"]
end

"""
    progress_bins(log, solver, column; bins) -> Vector{NamedTuple}

Aggregate one column across every episode of a solver by **normalised episode
progress**, using bin membership only.

There is no interpolation: stretching a 42-decision DPW episode onto a
150-point grid would manufacture values that never occurred. Each bin reports
its own `n`, so a bin only short episodes reach is visibly thinner evidence
rather than an equally confident line.
"""
function progress_bins(log::DecisionLog, solver::AbstractString,
        column::AbstractString; bins::Integer = 5)
    i = log.index
    haskey(i, column) || throw(ArgumentError("no column '$column'"))
    bins >= 1 || throw(ArgumentError("bins must be positive"))
    rows = filter(r -> r[i["solver"]] == solver, log.rows)
    isempty(rows) && throw(ArgumentError("no rows for solver '$solver'"))
    lens = Dict{Int,Int}()
    for r in rows
        s = parse(Int, r[i["seed"]])
        lens[s] = max(get(lens, s, 0), parse(Int, r[i["decision"]]))
    end
    buckets = [Float64[] for _ in 1:bins]
    absent = 0
    for r in rows
        v = _num(r[i[column]])
        if v === missing
            absent += 1
            continue
        end
        s = parse(Int, r[i["seed"]])
        f = (parse(Int, r[i["decision"]]) - 1) / max(lens[s], 1)
        push!(buckets[clamp(floor(Int, f * bins) + 1, 1, bins)], v)
    end
    out = NamedTuple[]
    for (k, b) in enumerate(buckets)
        isempty(b) && continue
        sb = sort(b)
        n = length(sb)
        push!(out, (bin = k, from = (k - 1) / bins, to = k / bins, n = n,
            mean = sum(sb) / n, median = sb[cld(n, 2)],
            q25 = sb[clamp(cld(n, 4), 1, n)], q75 = sb[clamp(cld(3n, 4), 1, n)],
            min = sb[1], max = sb[end], absent = absent))
    end
    return out
end

"""
    episode_lengths(log, solver) -> Vector{Int}
"""
function episode_lengths(log::DecisionLog, solver::AbstractString)
    i = log.index
    lens = Dict{Int,Int}()
    for r in log.rows
        r[i["solver"]] == solver || continue
        s = parse(Int, r[i["seed"]])
        lens[s] = max(get(lens, s, 0), parse(Int, r[i["decision"]]))
    end
    return [lens[k] for k in sort(collect(keys(lens)))]
end

"""
    DECISION_QUANTITY_CONTRACT

The FJ9.6a wish list: what a diagnostic figure would want, and the column
expected to supply it. `decision_log_audit` probes the artefact against this
rather than trusting it.
"""
const DECISION_QUANTITY_CONTRACT = (
    ("episode identity", ("solver", "seed", "decision")),
    ("ego world pose", ("ego_x", "ego_z", "ego_angle", "ego_speed")),
    ("lane projection", ("d", "phi", "kappa")),
    ("action identity and command", ("action_kind", "action_id", "v_cmd",
        "omega_cmd")),
    ("reward total and components", ("reward_total", "reward_progress",
        "reward_lateral", "reward_heading", "reward_time", "reward_pedestrian",
        "reward_stagnation", "reward_stop_approach", "reward_steering",
        "reward_events")),
    ("stop subsystem", ("d_stop", "sigma_stop", "stop_hold_progress",
        "full_stop", "passed_stop", "stop_violation")),
    ("duck subsystem", ("duck_present", "duck_longitudinal", "duck_lateral",
        "duck_active", "duck_class", "duck_active_state", "crossings_started")),
    ("episode outcome", ("terminated", "truncated", "reason")),
    ("planning cost", ("planning_time", "model_calls")),
)

"""
    decision_log_audit(log) -> Vector{DecisionFieldItem}

FJ9.6a. What the enriched log actually supports, probed rather than assumed,
one line per wanted quantity plus the derived and absent entries.

The audit is executable and therefore self-invalidating: if a future artefact
drops a column, its line changes from LOGGED to ABSENT on its own.
"""
function decision_log_audit(log::DecisionLog)
    items = DecisionFieldItem[]
    n = length(log.rows)
    for (group, cols) in DECISION_QUANTITY_CONTRACT
        for c in cols
            if !haskey(log.index, c)
                push!(items, DecisionFieldItem(group, c, FIELD_ABSENT, n,
                    "not a column of $(basename(log.path))"))
                continue
            end
            k = log.index[c]
            blank = count(r -> isempty(r[k]), log.rows)
            vals = length(unique(r[k] for r in log.rows))
            ev = blank == 0 ?
                "$n rows, $vals distinct, none empty" :
                "$n rows, $blank empty ($(round(100blank / n; digits = 1))%) " *
                "— MISSING, never zero"
            push!(items, DecisionFieldItem(group, c, LOGGED, blank, ev))
        end
    end
    push!(items, DecisionFieldItem("reward", "cumulative_return",
        DERIVED_IDENTITY, 0,
        "running sum of reward_total; FJ8.4c showed the final value equals " *
        "the FJ8.4b episode return exactly"))
    push!(items, DecisionFieldItem("episode outcome", "horizon_reached",
        DERIVED_IDENTITY, 0,
        "terminated == false on the final row; 'truncated' is false on all " *
        "$n rows and carries no signal on its own"))
    for q in ("observation as the policy saw it", "belief state",
              "per-decision search tree", "wall-clock timestamp")
        push!(items, DecisionFieldItem("not recorded", q, FIELD_ABSENT, n,
            "the evaluation never stored it; FJ9.5 persists search trees " *
            "only for the two explicitly captured snapshots"))
    end
    return items
end

"""
    decision_audit_table(items) -> String
"""
function decision_audit_table(items::AbstractVector{DecisionFieldItem})
    io = IOBuffer()
    println(io, "| Quantity | Column | Status | Empty rows | Evidence |")
    println(io, "|---|---|---|---|---|")
    for it in items
        println(io, "| $(it.quantity) | `$(it.column)` | ",
            availability_label(it.status), " | ",
            it.status === LOGGED ? string(it.blanks) : "—",
            " | $(it.evidence) |")
    end
    return String(take!(io))
end
