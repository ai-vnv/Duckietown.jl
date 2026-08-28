# FJ9.7 — artifact-driven animation.
#
#     decisions.csv -> AnimationSequence -> frame_scene(seq, t) -> backend
#
# Animation is playback of recorded evidence. No policy, no planner, no `gen`,
# no environment step, no Python and no solver appears on this path, and the
# render check asserts all of it in a fresh process.
#
# The one thing an animation can do that a static figure cannot is imply
# TIME. This artefact has no wall-clock timestamp — FJ9.6a established that —
# so the canonical timeline is the decision index and the figure says so.
# `planning_time` is computational latency, not elapsed world time, and is
# never used for pacing.
#
# Two rules carry most of the weight, because both failure modes look
# perfectly convincing on screen:
#
#   * the trajectory at frame t is rows 1..t, never the whole episode;
#   * an event marker appears at the frame the event was logged, never before.
#
# A renderer that drew the finished trajectory from frame 1 would show the
# viewer the future and look better doing it.

"""
    ANIMATION_TIMELINE_LABEL

What the x-axis and the caption must say. The artefact records no wall-clock
time, so this is not real-time playback and must not be presented as such.
"""
const ANIMATION_TIMELINE_LABEL = "Decision-index playback"

"""
    ANIMATION_LAYOUT_VERSION

Part of every frame's identity. A layout change makes different frames from
the same evidence, and the fingerprint should say so.
"""
const ANIMATION_LAYOUT_VERSION = "fj97.1"

"""
    ANIMATION_ABSENT

What the decision log cannot support, reported rather than reconstructed.

`duck world position` is the one that would be easy to fake: the reset duck
pose is seed-invariant and available, but ducks move during an episode and the
log stores only the lane-relative pair. Drawing a stationary duck for 150
frames would be a fabrication that looks like data, so the world view omits it
and the lane-relative panel shows what was actually recorded.
"""
const ANIMATION_ABSENT = (
    ("duck world position", "the log records duck_longitudinal / duck_lateral " *
        "in the lane frame only; the world-frame track is not stored"),
    ("wall-clock timestamp", "never recorded; the timeline is the decision " *
        "index, and planning_time is latency, not elapsed world time"),
    ("observation as the policy saw it", "never recorded (FJ9.6a)"),
    ("belief state", "no partially observable formulation exists (FJ10)"),
    ("per-decision search tree", "FJ9.5 persists trees only for the two " *
        "explicitly captured snapshots"),
)

"""
    AnimationFrame

One logged decision, in playback form. Every dynamic value here is a copy of
one row of the decision log; nothing is recomputed from a model.

`events` holds the flags that fired **at this decision**, so a consumer cannot
accidentally reveal a later one.
"""
struct AnimationFrame
    decision::Int
    # world pose the decision was taken from
    ego_position::NTuple{2,Float64}
    ego_angle::Float64
    ego_speed::Float64
    # lane projection after the transition
    d::Float64
    phi::Float64
    v::Float64
    kappa::Float64
    # stop subsystem
    d_stop::Union{Float64,Missing}
    sigma_stop::Bool
    stop_hold_progress::Float64
    # duck subsystem, lane-relative: the world-frame position is ABSENT
    duck_present::Bool
    duck_active::Bool
    duck_longitudinal::Float64
    duck_lateral::Float64
    # action
    action_kind::String
    action_id::Int
    v_cmd::Float64
    omega_cmd::Float64
    # reward
    reward_total::Float64
    cumulative_return::Float64
    # events logged at THIS decision
    events::Vector{String}
    # planning cost
    model_calls::Int
    planning_time::Float64
    # episode
    terminated::Bool
    reason::String
    fingerprint::String
end

"""
    AnimationSequence

One episode's playback, with the provenance that ties it to the artefact.

`fingerprint` is a function of the ordered frame identities, so changing one
row of evidence changes the identity of the animation.
"""
struct AnimationSequence
    solver::String
    seed::Int
    frames::Vector{AnimationFrame}
    horizon::Int
    outcome::EpisodeOutcome
    reason::String
    timeline::String
    source_fingerprint::String
    fingerprint::String
end

Base.length(s::AnimationSequence) = length(s.frames)
Base.getindex(s::AnimationSequence, t::Integer) = s.frames[t]

"""
    animation_sequence(log, solver, seed) -> AnimationSequence

Build the playback for one episode from the validated decision log.

`horizon` is the episode's own length, not a protocol constant: an episode
that terminated at 86 decisions has 86 frames and no more. Padding it out to
150 would invent stationary frames that never happened.
"""
function animation_sequence(log::DecisionLog, solver::AbstractString,
        seed::Integer)
    i = log.index
    rows = filter(r -> r[i["solver"]] == solver &&
                       parse(Int, r[i["seed"]]) == seed, log.rows)
    isempty(rows) && throw(ArgumentError("no episode for solver '$solver' " *
        "seed $seed in $(basename(log.path))"))
    sort!(rows; by = r -> parse(Int, r[i["decision"]]))

    frames = AnimationFrame[]
    running = 0.0
    for r in rows
        k = parse(Int, r[i["decision"]])
        running += parse(Float64, r[i["reward_total"]])
        ev = [c for c in DIAGNOSTIC_EVENT_COLUMNS
              if haskey(i, c) && _flag(r[i[c]])]
        fp = string(hash((log.fingerprint, solver, Int(seed), k,
            ANIMATION_LAYOUT_VERSION)); base = 16, pad = 16)
        push!(frames, AnimationFrame(k,
            (parse(Float64, r[i["ego_x"]]), parse(Float64, r[i["ego_z"]])),
            parse(Float64, r[i["ego_angle"]]),
            parse(Float64, r[i["ego_speed"]]),
            parse(Float64, r[i["d"]]), parse(Float64, r[i["phi"]]),
            parse(Float64, r[i["v"]]), parse(Float64, r[i["kappa"]]),
            _num(r[i["d_stop"]]), _flag(r[i["sigma_stop"]]),
            parse(Float64, r[i["stop_hold_progress"]]),
            _flag(r[i["duck_present"]]), _flag(r[i["duck_active"]]),
            parse(Float64, r[i["duck_longitudinal"]]),
            parse(Float64, r[i["duck_lateral"]]),
            r[i["action_kind"]], parse(Int, r[i["action_id"]]),
            parse(Float64, r[i["v_cmd"]]), parse(Float64, r[i["omega_cmd"]]),
            parse(Float64, r[i["reward_total"]]), running, ev,
            parse(Int, r[i["model_calls"]]),
            parse(Float64, r[i["planning_time"]]),
            _flag(r[i["terminated"]]), r[i["reason"]], fp))
    end

    final = frames[end]
    outcome = final.terminated ? ENV_TERMINATED : HORIZON_REACHED
    return AnimationSequence(String(solver), Int(seed), frames, length(frames),
        outcome, final.reason, ANIMATION_TIMELINE_LABEL, log.fingerprint,
        string(hash(Tuple(f.fingerprint for f in frames)); base = 16, pad = 16))
end

"""
    trajectory_through(seq, t) -> Vector{NTuple{2,Float64}}

The ground track **as far as frame `t`**, and no further. This is the whole
point: at frame 20 the viewer must not see where the vehicle ends up.
"""
function trajectory_through(seq::AnimationSequence, t::Integer)
    1 <= t <= length(seq) || throw(BoundsError(seq, t))
    return [f.ego_position for f in view(seq.frames, 1:t)]
end

"""
    events_through(seq, t) -> Vector{Pair{String,Int}}

Every event logged at or before frame `t`, as `name => decision`. An event at
decision 28 is invisible at frame 27 and visible from frame 28 onward.
"""
function events_through(seq::AnimationSequence, t::Integer)
    1 <= t <= length(seq) || throw(BoundsError(seq, t))
    out = Pair{String,Int}[]
    for f in view(seq.frames, 1:t), e in f.events
        push!(out, e => f.decision)
    end
    return out
end

"""
    series_through(seq, t, name) -> Vector{Union{Float64,Missing}}

The history of one quantity up to frame `t`. `d_stop` keeps its `missing`, so
a frame before the sign became a candidate draws a gap rather than a zero.
"""
function series_through(seq::AnimationSequence, t::Integer,
        name::AbstractString)
    1 <= t <= length(seq) || throw(BoundsError(seq, t))
    g = name == "d" ? (f -> f.d) : name == "phi" ? (f -> f.phi) :
        name == "v" ? (f -> f.v) : name == "kappa" ? (f -> f.kappa) :
        name == "ego_speed" ? (f -> f.ego_speed) :
        name == "v_cmd" ? (f -> f.v_cmd) :
        name == "omega_cmd" ? (f -> f.omega_cmd) :
        name == "d_stop" ? (f -> f.d_stop) :
        name == "stop_hold_progress" ? (f -> f.stop_hold_progress) :
        name == "duck_longitudinal" ? (f -> f.duck_longitudinal) :
        name == "duck_lateral" ? (f -> f.duck_lateral) :
        name == "reward_total" ? (f -> f.reward_total) :
        name == "cumulative_return" ? (f -> f.cumulative_return) :
        name == "model_calls" ? (f -> Float64(f.model_calls)) :
        name == "planning_time" ? (f -> f.planning_time) :
        throw(ArgumentError("no animation series '$name'"))
    return Union{Float64,Missing}[g(f) for f in view(seq.frames, 1:t)]
end

"""
    model_time(seq, t; frame_skip, dt) -> Float64

Model time at frame `t`, in seconds, by the exact identity
`(t - 1) * frame_skip * dt`.

This is **model time**, not recorded wall-clock time — the artefact has none.
`frame_skip` and `dt` are not in the log either, so the caller must supply
them from the configuration the protocol used; there is no default that could
silently be wrong.
"""
model_time(seq::AnimationSequence, t::Integer; frame_skip::Integer,
    dt::Real) = (t - 1) * frame_skip * float(dt)

const MODEL_TIME_LABEL = "model time (derived from decision × frame_skip × " *
    "dt; NOT recorded wall-clock time)"

"""
    frame_caption(seq, t) -> Vector{String}

The lines a frame must carry to be re-derivable, including the terminal
banner once the episode has ended.
"""
function frame_caption(seq::AnimationSequence, t::Integer)
    f = seq.frames[t]
    act = f.action_kind == "continuous" ?
        "v_cmd $(round(f.v_cmd; digits = 3))  omega_cmd " *
        "$(round(f.omega_cmd; digits = 3))" :
        "macro #$(f.action_id)  (v $(round(f.v_cmd; digits = 3)), omega " *
        "$(round(f.omega_cmd; digits = 3)))"
    lines = ["decision $(f.decision) / $(seq.horizon)   ·   " *
             ANIMATION_TIMELINE_LABEL,
        "action: $act",
        "reward $(round(f.reward_total; digits = 3))   return " *
        "$(round(f.cumulative_return; digits = 2))",
        # significant digits, not decimal places: a tabular policy's 4e-5 s
        # rounds to "0.0" at four decimals, which reads as "not measured"
        "model_calls $(f.model_calls)   planning_time " *
        "$(round(f.planning_time; sigdigits = 3)) s"]
    ev = events_through(seq, t)
    isempty(ev) || push!(lines, "events so far: " *
        join(("$k@$v" for (k, v) in ev), ", "))
    if t == length(seq)
        push!(lines, seq.outcome === ENV_TERMINATED ?
            "TERMINATED at decision $(f.decision) — $(seq.reason)" :
            "HORIZON REACHED at decision $(f.decision) — $(seq.reason)")
    end
    return lines
end

"""
    animation_provenance(seq) -> Vector{String}
"""
animation_provenance(seq::AnimationSequence) =
    ["solver: $(seq.solver)   seed: $(seq.seed)   frames: $(length(seq))",
     "outcome: $(seq.outcome) ($(seq.reason))   timeline: $(seq.timeline)",
     "source: FJ8.4c decisions.csv  fingerprint $(seq.source_fingerprint)",
     "animation $(seq.fingerprint)  layout $(ANIMATION_LAYOUT_VERSION)"]

"""
    EpisodeSelection

A stated selection rule and the episode it picked. FJ9.4 forbade choosing a
representative episode by eye; the rule is recorded so the choice is
reproducible and arguable.
"""
struct EpisodeSelection
    solver::String
    seed::Int
    rule::Symbol
    justification::String
end

"""
    select_episode(log, solver; rule) -> EpisodeSelection

Pick one episode of a solver by an explicit rule, never by inspection.

* `:median_return` — the seed whose episode return is the lower median.
* `:median_length_terminating` — among episodes the environment terminated,
  the lower-median length. Errors if the solver never terminates.
* `:first_stagnation` — the lowest seed that performs a full stop and never
  passes the sign. Errors if no episode does.
"""
function select_episode(log::DecisionLog, solver::AbstractString;
        rule::Symbol = :median_return)
    seeds = sort!([sd for (sv, sd) in log.episodes if sv == solver])
    isempty(seeds) && throw(ArgumentError("no episodes for solver '$solver'"))
    seqs = [animation_sequence(log, solver, sd) for sd in seeds]

    if rule === :median_return
        rets = [s.frames[end].cumulative_return for s in seqs]
        p = sortperm(rets)[cld(length(rets), 2)]
        return EpisodeSelection(String(solver), seeds[p], rule,
            "lower-median return over $(length(seeds)) seeds " *
            "($(round(rets[p]; digits = 2)))")
    elseif rule === :median_length_terminating
        term = [s for s in seqs if s.outcome === ENV_TERMINATED]
        isempty(term) && throw(ArgumentError(
            "solver '$solver' has no environment-terminated episode"))
        lens = [length(s) for s in term]
        p = sortperm(lens)[cld(length(lens), 2)]
        return EpisodeSelection(String(solver), term[p].seed, rule,
            "lower-median length among the $(length(term)) terminating " *
            "episodes ($(lens[p]) decisions, $(term[p].reason))")
    elseif rule === :first_stagnation
        for s in seqs
            ev = Set(k for (k, _) in events_through(s, length(s)))
            if "full_stop" in ev && !("passed_stop" in ev)
                n = count(f -> f.sigma_stop, s.frames)
                return EpisodeSelection(String(solver), s.seed, rule,
                    "lowest seed that performs a full stop and never passes " *
                    "the sign ($n of $(length(s)) decisions in the stop zone)")
            end
        end
        throw(ArgumentError("solver '$solver' has no stagnation episode"))
    end
    throw(ArgumentError("unknown selection rule :$rule"))
end

"""
    paired_frames(a, b) -> Int

Frame count for a side-by-side playback: the longer of the two episodes.

The timeline is the **absolute decision index**, so decision 40 in one panel
is decision 40 in the other. A panel whose episode has ended freezes on its
terminal frame with the banner `frame_caption` produced; it is never looped,
restarted, or stretched to match. Normalised progress would align DPW's
decision 40 of 80 with TD3's decision 75 of 150, which are not the same
decision and were not taken under the same conditions.
"""
paired_frames(a::AnimationSequence, b::AnimationSequence) =
    max(length(a), length(b))

"""
    frame_index(seq, t) -> Int

The frame of `seq` to show at absolute decision `t`: `t` while the episode is
running, and its final frame once it has ended.
"""
frame_index(seq::AnimationSequence, t::Integer) = min(t, length(seq))

"""
    is_frozen(seq, t) -> Bool

Whether `seq` has already ended at absolute decision `t`, so the panel is
showing a held terminal state rather than a live one.
"""
is_frozen(seq::AnimationSequence, t::Integer) = t > length(seq)

"""
    animation_absent_lines() -> Vector{String}
"""
animation_absent_lines() =
    ["ABSENT — $(q): $(why)" for (q, why) in ANIMATION_ABSENT]

# ---------------------------------------------------------------------------
# World geometry for playback
# ---------------------------------------------------------------------------

"""
    StaticWorld

The read-only track description: tiles, lane centrelines, stop-sign positions.

FJ8.0 certified `map` and `stop_signs` as the shared-by-design, never-written
part of a world state, and they are seed-invariant — a test asserts that two
different reset seeds produce the same `StaticWorld`. So this is map loading,
not simulation, and nothing episode-specific can leak in through it.

Everything that *moves* comes from the decision log instead.
"""
struct StaticWorld
    tiles::Vector{TilePatch}
    lane_centrelines::Vector{Vector{NTuple{2,Float64}}}
    map_extent::NTuple{4,Float64}
    view_extent::NTuple{4,Float64}
    tile_size::Float64
    roadmap::RoadMap
    signs::Vector{StopSignState}
    sign_positions::Vector{NTuple{2,Float64}}
    sign_to_line_offset::Float64
    stop_lateral_limit::Float64
end

"""
    static_world(mdp, reference_state; lane_samples) -> StaticWorld

Extract the track description. Only `map` and `stop_signs` are read from
`reference_state`; its ego pose, ducks, stop memory and RNG are ignored.

`view_extent` is fixed here, from the map and the signs, and never from the
trajectory. A camera fitted to the episode's full extent would hold the
vehicle's future in the empty space around it, and a camera refitted per
frame would jitter; neither is what the viewer should be reading.
"""
function static_world(m::AnyMDPLike, s::DuckieWorldState;
        lane_samples::Integer = 24)
    map_ = s.map
    ts = map_.tile_size
    h, w = size(map_.grid)
    cfg = m.transition.state_cfg
    extent = (0.0, w * ts, 0.0, h * ts)
    signpos = [(sg.pos[1], sg.pos[3]) for sg in s.stop_signs]
    return StaticWorld(tile_patches(map_),
        lane_centrelines(map_; samples = lane_samples), extent,
        _view_extent(extent, signpos, 0.08 * ts), ts, map_,
        collect(s.stop_signs), signpos,
        cfg.sign_to_line_offset, cfg.stop_lateral_limit)
end

"""
    FrameScene

The drawable world at one frame. Every dynamic quantity is computed from the
logged pose by the same geometry the physics uses — `get_agent_corners` for
the footprint, `closest_curve_point` for the lane frame — and from nothing
else.

`ducks` is deliberately absent; see [`ANIMATION_ABSENT`](@ref).
"""
struct FrameScene
    decision::Int
    ego_position::NTuple{2,Float64}
    ego_angle::Float64
    ego_speed::Float64
    ego_footprint::Vector{NTuple{2,Float64}}
    ego_heading::NTuple{2,Float64}
    ego_velocity::NTuple{2,Float64}
    stop_lines::Vector{NTuple{4,Float64}}
    trajectory::Vector{NTuple{2,Float64}}
    view_extent::NTuple{4,Float64}
end

"""
    frame_scene(static, seq, t; roadmap) -> FrameScene

The world at frame `t`, with the ground track grown to exactly `t` points.

The stop line is drawn in the ego's lane frame, which is what the model
measures `d_stop` against — the FJ9.2 correction, applied per frame because
the frame moves with the vehicle.
"""
function frame_scene(sw::StaticWorld, seq::AnimationSequence, t::Integer)
    1 <= t <= length(seq) || throw(BoundsError(seq, t))
    f = seq.frames[t]
    pos3 = [f.ego_position[1], 0.0, f.ego_position[2]]
    corner_rows = get_agent_corners(pos3, f.ego_angle)
    footprint = _close_ring([(corner_rows[k, 1], corner_rows[k, 2])
                             for k in axes(corner_rows, 1)])
    dir = get_dir_vec(f.ego_angle)
    heading = (dir[1], dir[3])

    _, tangent = closest_curve_point(sw.roadmap, pos3, f.ego_angle)
    forward = tangent === nothing ? collect(heading_vec(f.ego_angle)) :
        _normalize_gt0(tangent)
    any(x -> x != 0.0, forward) || (forward = collect(heading_vec(f.ego_angle)))
    lines = [stop_line_segment(sg, forward, sw.sign_to_line_offset,
                 sw.stop_lateral_limit) for sg in sw.signs]

    return FrameScene(f.decision, f.ego_position, f.ego_angle, f.ego_speed,
        footprint, heading,
        (heading[1] * f.ego_speed, heading[2] * f.ego_speed),
        lines, trajectory_through(seq, t), sw.view_extent)
end
