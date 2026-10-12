# Frozen views, finite state catalogues, and explicitly supported live resumes.
# No callback, cache, source text or module name is executable session data.

"""
    VisualSessionState

A frozen view captured by `visual_state`. Its selected readouts and drawing data
are separate from an optional mathematical source artifact. Runtime metadata
which has no owned declarative representation is listed explicitly in
`describe(state).omissions`; callbacks and caches are never restored.
"""
struct VisualSessionState
    snapshot::VisualizationSpec
    selection::NamedTuple
    request::NamedTuple
    omissions::Vector{NamedTuple}
end

"""
    SavedVisualSession

A validated finite catalogue returned by `load_visual_session`. Use
`visual_session_states`, `inspection_selection`, `inspection_snapshot` and
`visualize` with `state=:name`. Opening a file does not run a mathematical query.
`inspection_session(saved)` explicitly resumes supported queries in Julia when
the matching owned source is available. The browser can select only saved views.
"""
struct SavedVisualSession
    states::NamedTuple
    selected::Symbol
    style::VisualStyle
    size::Union{Nothing,Tuple{Int,Int}}
    title::String
    caption::String
    provenance::NamedTuple
    source::NamedTuple
    base_dir::String
end

struct _OmittedVisualValue end
const _OMITTED_VISUAL_VALUE = _OmittedVisualValue()

function _session_omit!(omissions, x, path, reason)
    push!(omissions, (; path=String(path), type=string(typeof(x)), reason=String(reason)))
    return _OMITTED_VISUAL_VALUE
end

function _session_project(x, omissions, path, active=IdDict{Any,Nothing}(); depth=0)
    Base.@nospecialize x
    depth <= 64 || throw(ArgumentError("Saved metadata exceeds the nesting budget at $path"))
    x isa VisualizationSpec && return _session_snapshot(x, omissions, path, active; depth=depth+1)
    container = x isa Union{AbstractArray,AbstractDict}
    container && haskey(active, x) && return _session_omit!(omissions, x, path, "cyclic runtime metadata")
    container && (active[x] = nothing)
    try
        if x isa NamedTuple
            entries = Pair{Symbol,Any}[]
            for (key, value) in pairs(x)
                projected = _session_project(value, omissions, "$path.$key", active; depth=depth+1)
                projected isa _OmittedVisualValue || push!(entries, key=>projected)
            end
            return (; entries...)
        end
        try
            Serialization.visual_data_to_json(x)
            return x
        catch err
            err isa InterruptException && rethrow()
            err isa ArgumentError || rethrow()
        end
        if x isa Tuple
            return map(enumerate(x)) do (index, value)
                projected = _session_project(value, omissions, "$path[$index]", active; depth=depth+1)
                projected isa _OmittedVisualValue ? nothing : projected
            end |> Tuple
        elseif x isa AbstractArray
            values = map(enumerate(x)) do (index, value)
                projected = _session_project(value, omissions, "$path[$index]", active; depth=depth+1)
                projected isa _OmittedVisualValue ? nothing : projected
            end
            return reshape(Any[values...], size(x))
        elseif x isa AbstractDict
            result = Dict{Any,Any}()
            for (key, value) in pairs(x)
                k = _session_project(key, omissions, "$path.key", active; depth=depth+1)
                v = _session_project(value, omissions, "$path[$(repr(key))]", active; depth=depth+1)
                k isa _OmittedVisualValue || v isa _OmittedVisualValue || (result[k] = v)
            end
            return result
        end
        return _session_omit!(omissions, x, path, "runtime object; attach a supported owned mathematical source for new queries")
    finally
        container && delete!(active, x)
    end
end

function _session_snapshot(spec, omissions, path="snapshot", active=IdDict{Any,Nothing}(); depth=0)
    depth <= 64 || throw(ArgumentError("Saved specification exceeds the nesting budget at $path"))
    haskey(active,spec) && throw(ArgumentError("Saved specification contains cyclic view metadata at $path"))
    spec.kind === :linked_inspector && throw(ArgumentError("Capture the session, not its live browser wrapper"))
    active[spec]=nothing
    try
        return VisualizationSpec(spec.kind; title=spec.title, subtitle=spec.subtitle,
            layers=spec.layers,
            panels=VisualizationSpec[_session_snapshot(panel, omissions, "$path.panels[$i]",active; depth=depth+1) for (i,panel) in enumerate(spec.panels)],
            axes=spec.axes, legend=spec.legend,
            interaction=(; hover=false, clicks=false, widgets=(), requires_live_julia=false,
                saved_snapshot=true, new_queries=:require_julia),
            metadata=_session_project(spec.metadata, omissions, "$path.metadata",active; depth=depth+1))
    finally
        delete!(active,spec)
    end
end

"""
    visual_state(session)
    visual_state(spec::VisualizationSpec; selection=(;))

Freeze the current selected figure without retaining callbacks or caches.
Selections, exact available readouts, axes, scales and layout are copied;
Float64 drawing coordinates keep their existing display precision. Inspect
`describe(state)` for omitted runtime-only metadata. Capture each desired state
explicitly before changing a live selection. This does not serialize the full
mathematical object; `save_visual_session(...; object=...)` is a separate opt-in.
For resumable finite/ordinary inspectors, capture computes the owned source
identity. This may materialize a lazy module and validate its cover paths. Use
`visual_state(inspection_snapshot(session))` for a view-only capture without that cost.
"""
function visual_state(spec::VisualizationSpec; selection::NamedTuple=(;))
    return _visual_state(spec, selection, (; family=:snapshot, options=(;), selection=(;), source_identity=nothing))
end

function _visual_state(spec, selection, request)
    check_visual_spec(spec; throw=true)
    omissions = NamedTuple[]
    frozen = _session_snapshot(_freeze_visual_composition(spec), omissions)
    # The closed codec gives owned arrays instead of aliases into mutable sessions.
    data = Serialization.visual_data_from_json(Serialization.visual_data_to_json((; frozen, selection, request)))
    check_visual_spec(data.frozen; throw=true)
    return VisualSessionState(data.frozen, data.selection, data.request, omissions)
end

_session_request(s::_AbstractInspectionSession) =
    (; family=:snapshot, options=(;), selection=(;), source_identity=nothing)

function _session_request(s::InspectionSession)
    s.closed && return (; family=:snapshot, options=(;), selection=(;), source_identity=nothing)
    state = s.state
    selector = state.selector === :point ? (; point=only(state.query_points)) :
        state.selector === :parameter_pair ? (; parameter_pair=state.query_points) :
        state.selector === :vertex ? (; vertex=state.vertex) :
        state.selector === :pair ? (; pair=state.pair) : (;)
    return (; family=:finite_module,
        options=(; view=state.view, box=s.prepared.geometry === nothing ? nothing : s.prepared.geometry.box,
            matrix_limit=s.matrix_limit, cache_limit=s.cache_limit, upset=state.upset, downset=state.downset,
            slice=state.slice, slice_scope=state.slice_scope, slice_limit=s.slice_limit),
        selection=merge(selector, (; basis=state.basis, input=state.input, interval=state.interval)),
        source_identity=Serialization.visual_source_identity(s.object))
end

function _session_request(s::IntervalInspectionSession)
    (s.closed || !(s.object isa OrdinaryPersistence.PersistenceDiagram)) &&
        return (; family=:snapshot, options=(;), selection=(;), source_identity=nothing)
    return (; family=:ordinary_intervals,
        options=(; dim=s.payload.metadata.homological_dimension, window=s.window, max_intervals=s.max_intervals),
        selection=(; interval=something(s.state.interval,0), member=something(s.state.member,0), representative=s.state.representative),
        source_identity=Serialization.visual_source_identity(s.object))
end

function visual_state(session::_AbstractInspectionSession)
    return lock(session.lock) do
        session.busy && throw(ArgumentError("Cannot capture a session during a selection transaction"))
        _visual_state(inspection_snapshot(session), inspection_selection(session), _session_request(session))
    end
end

describe(frame::VisualSessionState) = (; kind=:visual_session_state, visual_kind=frame.snapshot.kind,
    selection=frame.selection, resume_family=frame.request.family, omissions=Tuple(frame.omissions))
inspection_snapshot(frame::VisualSessionState) = frame.snapshot
inspection_selection(frame::VisualSessionState) = frame.selection
inspection_summary(frame::VisualSessionState) = describe(frame)
available_visuals(frame::VisualSessionState) = available_visuals(frame.snapshot)
visual_spec(frame::VisualSessionState; kwargs...) = visual_spec(frame.snapshot; kwargs...)
Base.show(io::IO, frame::VisualSessionState) = print(io, "VisualSessionState(", frame.snapshot.kind,
    ", resume=", frame.request.family, ", omissions=", length(frame.omissions), ")")

"""Return the named finite states available without Julia-side recomputation."""
visual_session_states(saved::SavedVisualSession) = keys(saved.states)

function _saved_frame(saved::SavedVisualSession, state)
    state isa Symbol && haskey(saved.states, state) || throw(ArgumentError(
        "Unknown saved state $(repr(state)); choose one of $(keys(saved.states))"))
    return saved.states[state]
end

inspection_snapshot(saved::SavedVisualSession; state=saved.selected) = _saved_frame(saved,state).snapshot
inspection_selection(saved::SavedVisualSession; state=saved.selected) = _saved_frame(saved,state).selection
inspection_summary(saved::SavedVisualSession) = (; kind=:saved_visual_session,
    states=keys(saved.states), selected=saved.selected, source_mode=saved.source.mode,
    source_identity=saved.source.identity, caption=saved.caption,
    resume_families=Tuple(frame.request.family for frame in values(saved.states)),
    omissions=sum(length(frame.omissions) for frame in values(saved.states)),
    offline=:precomputed_states_only, new_queries=:require_julia_and_matching_owned_source)
describe(saved::SavedVisualSession) = inspection_summary(saved)
Base.show(io::IO, saved::SavedVisualSession) = print(io, "SavedVisualSession(states=", keys(saved.states),
    ", selected=", saved.selected, ", source=", saved.source.mode, ")")

function _check_saved_request(request)
    request isa NamedTuple && keys(request) == (:family,:options,:selection,:source_identity) ||
        throw(ArgumentError("Invalid declarative inspection request"))
    request.options isa NamedTuple && request.selection isa NamedTuple ||
        throw(ArgumentError("Inspection options and selectors must be named records"))
    if request.family === :snapshot
        request.source_identity === nothing || throw(ArgumentError("A generic snapshot has no implicit source identity"))
    else
        request.source_identity isa String && occursin(r"^[0-9a-f]{64}$",request.source_identity) ||
            throw(ArgumentError("A resumable state needs its captured source identity"))
    end
    allowed_options, allowed_selection = if request.family === :finite_module
        ((:view,:box,:matrix_limit,:cache_limit,:upset,:downset,:slice,:slice_scope,:slice_limit),
            (:point,:parameter_pair,:vertex,:pair,:basis,:input,:interval))
    elseif request.family === :ordinary_intervals
        ((:dim,:window,:max_intervals), (:interval,:member,:representative))
    elseif request.family === :snapshot
        ((), ())
    else
        throw(ArgumentError("Unknown saved inspection family"))
    end
    Set(keys(request.options)) == Set(allowed_options) &&
        all(in(allowed_selection), keys(request.selection)) ||
        throw(ArgumentError("Unsupported saved inspection options or selectors"))
    return nothing
end

"""
    check_visual_session(saved; throw=false)

Validate saved state IDs, declarative requests, snapshots and source-reference
structure. It does not read a relative source or run queries. The report is
accepted by `VisualizationValidationSummary`; source identity is checked when
`inspection_session(saved)` explicitly resumes a supported Julia computation.
"""
function check_visual_session(saved::SavedVisualSession; throw::Bool=false)
    issues = String[]
    try
        1 <= length(saved.states) <= 128 || Base.throw(ArgumentError("Save between 1 and 128 states"))
        saved.selected in keys(saved.states) || Base.throw(ArgumentError("Initial state is missing"))
        all(key -> !isempty(String(key)) && sizeof(String(key)) <= 256, keys(saved.states)) ||
            Base.throw(ArgumentError("State names must contain 1 to 256 UTF-8 bytes"))
        _check_visual_render_options(; style=saved.style, size=saved.size)
        for (id, frame) in pairs(saved.states)
            frame isa VisualSessionState || Base.throw(ArgumentError("State $id must be a VisualSessionState"))
            _check_saved_request(frame.request)
            all(item -> keys(item) == (:path,:type,:reason) &&
                all(v -> v isa String,values(item)),frame.omissions) ||
                Base.throw(ArgumentError("Invalid omitted-metadata records"))
            append!(issues, ["$id: $issue" for issue in check_visual_spec(frame.snapshot).issues])
            Serialization.visual_data_to_json((; snapshot=frame.snapshot, selection=frame.selection,
                request=frame.request, omissions=frame.omissions))
        end
        ref = saved.source
        allowed = get(ref,:mode,nothing) === :none ? (:mode,:identity) :
            get(ref,:mode,nothing) === :embedded ? (:mode,:identity,:payload) :
            get(ref,:mode,nothing) === :relative ? (:mode,:identity,:path,:sha256) : ()
        Set(keys(ref)) == Set(allowed) && !isempty(allowed) ||
            Base.throw(ArgumentError("Invalid saved source reference"))
        ref.mode === :none && ref.identity !== nothing && Base.throw(ArgumentError("A source-free view has no mathematical source identity"))
        if ref.mode !== :none
            ref.identity isa String && occursin(r"^[0-9a-f]{64}$",ref.identity) ||
                Base.throw(ArgumentError("Invalid mathematical source identity"))
        end
        ref.mode === :embedded && !(ref.payload isa String) && Base.throw(ArgumentError("Invalid source payload"))
        ref.mode === :relative && !(ref.path isa String && ref.sha256 isa String &&
            occursin(r"^[0-9a-f]{64}$",ref.sha256)) && Base.throw(ArgumentError("Invalid relative source reference"))
        ref.mode === :relative && Serialization.visual_session_resource_path(ref.path)
    catch err
        err isa InterruptException && rethrow()
        push!(issues, sprint(showerror,err))
    end
    valid = isempty(issues)
    throw && !valid && _throw_invalid_visual(:check_visual_session,issues)
    return (; kind=:visual_session, valid, issues, states=keys(saved.states))
end

function _saved_record(saved)
    states = NamedTuple{keys(saved.states)}(Tuple((; snapshot=f.snapshot, selection=f.selection,
        request=f.request, omissions=f.omissions) for f in values(saved.states)))
    return (; states, selected=saved.selected, style=saved.style, size=saved.size,
        title=saved.title, caption=saved.caption, provenance=saved.provenance, source=saved.source)
end

function _saved_from_record(record, base_dir)
    keys(record) == (:states,:selected,:style,:size,:title,:caption,:provenance,:source) ||
        throw(ArgumentError("Invalid visual-session record fields"))
    record.states isa NamedTuple && 1 <= length(record.states) <= 128 ||
        throw(ArgumentError("Saved states must be a bounded named catalogue"))
    frames = map(values(record.states)) do frame
        frame isa NamedTuple && keys(frame) == (:snapshot,:selection,:request,:omissions) ||
            throw(ArgumentError("Invalid saved visual state"))
        VisualSessionState(frame.snapshot, frame.selection, frame.request, NamedTuple[frame.omissions...])
    end
    saved = SavedVisualSession(NamedTuple{keys(record.states)}(frames), record.selected, record.style,
        record.size, record.title, record.caption, record.provenance, record.source, String(base_dir))
    check_visual_session(saved; throw=true)
    return saved
end

"""
    load_visual_session(path; max_bytes=268435456)

Open a versioned `.json` or self-contained `.html` saved view without executing
callbacks, loading its full source or recomputing invariants. The content digest
detects changed data; it is not an authenticated signature. Missing or changed
relative sources do not prevent viewing the retained states, but are rejected
on live resume. Unknown schemas/types, invalid requests and malformed drawing
data are rejected. Resource references remain relative to the containing file.
"""
function load_visual_session(path::AbstractString; max_bytes=268435456)
    try
        return _saved_from_record(Serialization.read_visual_session(path; max_bytes), dirname(abspath(path)))
    catch err
        err isa InterruptException && rethrow()
        err isa ArgumentError && rethrow()
        throw(ArgumentError("Invalid saved visual session: $(sprint(showerror,err))"))
    end
end

function _saved_resume(saved, frame; object=nothing)
    request = frame.request
    request.family === :snapshot && throw(ArgumentError(
        "This state is a precomputed snapshot. Its inspector has no supported owned live-resume contract."))
    saved.source.mode === :none && throw(ArgumentError(
        "This file contains precomputed views only; no owned source was attached for live queries."))
    request.source_identity == saved.source.identity || throw(ArgumentError(
        "The attached mathematical source differs from the object used to capture this state"))
    source = Serialization.visual_session_source(saved.source, saved.base_dir; object)
    if request.family === :finite_module
        source isa Union{Modules.PModule,EncodingResult} || throw(ArgumentError("A finite-module view needs its saved module or encoding"))
    elseif request.family === :ordinary_intervals
        source isa OrdinaryPersistence.PersistenceDiagram || throw(ArgumentError("An ordinary interval view needs its saved diagram"))
    end
    session = inspection_session(source; request.options...)
    try
        isempty(request.selection) || select_inspection!(session; request.selection...)
        _session_same_view(frame, session) || throw(ArgumentError(
            "The attached source and declarative request do not reproduce the saved selected view"))
    catch
        close_inspection!(session)
        rethrow()
    end
    return session
end

function _session_comparison_value(x)
    Base.@nospecialize x
    if x isa VisualizationSpec
        return (; kind=x.kind, title=x.title, subtitle=x.subtitle, layers=x.layers,
            axes=x.axes, legend=x.legend, panels=Tuple(_session_comparison_value(p) for p in x.panels),
            metadata=_session_comparison_value(x.metadata))
    elseif x isa NamedTuple
        return (; (k=>_session_comparison_value(v) for (k,v) in pairs(x)
            if k !== :revision && k !== :module_materialized_before && k !== :module_materialized_after)...)
    elseif x isa Tuple
        return map(_session_comparison_value,x)
    elseif x isa AbstractArray && eltype(x) <: NamedTuple
        return reshape(Any[_session_comparison_value(v) for v in x],size(x))
    end
    return x
end

function _session_same_view(frame, session)
    # Revision counts and cache provenance change on a fresh process; the actual
    # selected IDs, parameter values, layer values, scales and layout must agree.
    without_revision(s) = (; (k=>v for (k,v) in pairs(s) if k !== :revision)...)
    isequal(without_revision(frame.selection), without_revision(inspection_selection(session))) || return false
    stored = Serialization.visual_data_to_json(_session_comparison_value(frame.snapshot))
    rebuilt = Serialization.visual_data_to_json(_session_comparison_value(_session_snapshot(inspection_snapshot(session), NamedTuple[])))
    return isequal(stored, rebuilt)
end

"""
    inspection_session(saved::SavedVisualSession; state=saved.selected, object=nothing)

Explicitly reconstruct a fresh live finite-module/encoding or ordinary-diagram
inspector, with its saved selection and limits. Source identity must match;
`object=...` can supply that exact object when an external file is unavailable.
This computes in Julia. Browser callbacks, listeners and caches start afresh.
Other view families still reopen as the exact retained static states.
"""
function inspection_session(saved::SavedVisualSession; state=saved.selected, object=nothing)
    check_visual_session(saved; throw=true)
    return _saved_resume(saved, _saved_frame(saved,state); object)
end

function _session_atomic_write(path, writer)
    directory = dirname(abspath(path))
    isdir(directory) || throw(ArgumentError("Create the destination directory before saving a session"))
    temporary, io = mktemp(directory)
    close(io)
    try
        writer(temporary)
        mv(temporary, path; force=true)
    finally
        isfile(temporary) && rm(temporary)
    end
    return String(path)
end

"""
    save_visual_session(path, session_or_state; object=nothing, object_path=nothing,
        selected=:view, caption="", title="Saved mathematical views", style=VisualStyle(),
        size=nothing, provenance=(;), max_bytes=268435456)
    save_visual_session(path, (name=state, ...); selected=first_state, ...)

Save a captured view or up to 128 explicitly named states. JSON retains exact
available readouts and declarative drawing data; HTML embeds the same record,
Cairo-rendered SVGs and finite browser controls, with no server or network assets.
Import CairoMakie for HTML. Downloadable records distinguish selected values
from Float64 drawing coordinates. A caption, provenance, style and size travel
with the catalogue. No arbitrary callbacks or Julia expressions are stored.

Full source retention is explicit: `object` supports finite modules, encodings
(including their original module coordinates), and owned ordinary diagrams.
Alternatively `object_path` references an owned JSON resource inside the bundle
directory; its relative path and content identity are checked on live resume.
Without a source, all saved states remain viewable but new queries are unavailable.
Writing is atomic: a failed encoding or rendering leaves an existing file intact.
"""
function save_visual_session(path::AbstractString, input; object=nothing, object_path=nothing,
        selected=nothing, caption::AbstractString="", title::AbstractString="Saved mathematical views",
        style=VisualStyle(), size=nothing, provenance::NamedTuple=(;), max_bytes=268435456)
    extension = lowercase(splitext(path)[2])
    extension in (".json",".html") || throw(ArgumentError("Save a visual session as .json or .html; use save_visual for SVG/PNG/PDF"))
    _check_visual_render_options(;style,size)
    selected === nothing || selected isa Symbol || throw(ArgumentError("selected must name a saved state with a Symbol"))
    input isa NamedTuple && !(1 <= length(input) <= 128) && throw(ArgumentError("Save between 1 and 128 named states"))
    frames = if input isa NamedTuple
        NamedTuple{keys(input)}(map(value -> value isa VisualSessionState ? value : visual_state(value), values(input)))
    else
        (; view=input isa VisualSessionState ? input : visual_state(input))
    end
    isempty(frames) && throw(ArgumentError("A saved session needs at least one state"))
    selected === nothing && (selected = first(keys(frames)))
    source = Serialization.visual_session_reference(; object, object_path,
        base_dir=dirname(abspath(path)), max_bytes)
    saved = SavedVisualSession(frames, selected, style, size === nothing ? nothing : Tuple(size),
        String(title), String(caption), provenance, source, dirname(abspath(path)))
    check_visual_session(saved; throw=true)
    if source.mode !== :none
        checked_source = Serialization.visual_session_source(source, saved.base_dir; object)
        for frame in values(frames)
            frame.request.family === :snapshot && continue
            resumed = _saved_resume(saved, frame; object=checked_source)
            close_inspection!(resumed)
        end
    end
    json = Serialization.visual_session_to_json(_saved_record(saved); max_bytes)
    if extension == ".json"
        return _session_atomic_write(path, temporary -> write(temporary, json))
    end
    states = map(collect(pairs(frames))) do (id, frame)
        (; id=String(id), label=String(id), svg=_portable_svg(frame.snapshot; style, size),
            selection=sprint(show, frame.selection),
            context=string("Caption: ", caption, "\nProvenance: ", provenance,
                "\nSource identity: ", source.identity, "\nResume family: ", frame.request.family,
                "\nRuntime metadata omitted: ", frame.omissions),
            plotted_json=String(JSON3.write(Serialization.visual_data_to_json(
                (; snapshot=frame.snapshot, selection=frame.selection, omissions=frame.omissions)))))
    end
    return _session_atomic_write(path, temporary -> begin
        _write_portable_visual_html(temporary; title=String(title), caption=String(caption), states,
            initial=String(selected), session_json=json)
        filesize(temporary) <= max_bytes || throw(ArgumentError("Portable HTML exceeds max_bytes"))
    end)
end

available_visuals(::SavedVisualSession) = (:saved_view,)
_visual_request_keywords(::SavedVisualSession, ::Symbol) = (:state,)
_visual_request_cost(::SavedVisualSession, ::Symbol) = (; construction=:retained_snapshot, algebra=:none)
function _append_visual_request_issues!(issues::Vector{String}, saved::SavedVisualSession, kind::Symbol; state=saved.selected)
    append!(issues, check_visual_session(saved).issues)
    state in keys(saved.states) || push!(issues,"Unknown saved state")
    return nothing
end
function visual_spec(saved::SavedVisualSession; state=saved.selected, kind::Symbol=:auto, cache=:auto)
    kind in (:auto,:saved_view) || throw(ArgumentError("Saved sessions support kind=:saved_view"))
    cache === :auto || throw(ArgumentError("Saved states do not accept a computation cache"))
    check_visual_session(saved; throw=true)
    return inspection_snapshot(saved; state)
end
function visualize(saved::SavedVisualSession; state=saved.selected, style=saved.style,
        size=saved.size, kind::Symbol=:auto, cache=:auto, kwargs...)
    return render(visual_spec(saved; state, kind, cache); style, size, kwargs...)
end
"""
    save_visual(path, saved::SavedVisualSession; state=saved.selected, sidecar=true, ...)
    save_visual(outdir, stem, saved::SavedVisualSession; state=saved.selected, sidecar=true, ...)

Export one saved figure using its stored style/size unless overridden. Static
SVG/PNG/PDF exports also write `path * ".json"`: a reopenable snapshot with exact
available plotted values, caption, provenance, original source identity and the
figure's SHA-256 digest. The sidecar contains no full mathematical source.
`sidecar=false` requests just the figure. HTML uses a self-contained saved-view export.
Figure and record are prepared before replacing either destination;
each file replacement is atomic, but two files are not a filesystem transaction.
"""
function save_visual(path::AbstractString, saved::SavedVisualSession; state=saved.selected,
        style=saved.style, size=saved.size, kind::Symbol=:auto, cache=:auto,
        sidecar::Bool=true, backend::Symbol=:auto, kwargs...)
    spec = visual_spec(saved; state, kind, cache)
    source_frame = _saved_frame(saved,state)
    frame = VisualSessionState(source_frame.snapshot,source_frame.selection,
        (;family=:snapshot,options=(;),selection=(;),source_identity=nothing),source_frame.omissions)
    format = _visual_export_format_from_path(path)
    if format === :html
        isempty(kwargs) || throw(ArgumentError("Saved HTML accepts style and size, not renderer-owned controls"))
        backend in (:auto,:cairomakie) || throw(ArgumentError("Portable saved HTML renders with CairoMakie"))
        return save_visual_session(path, NamedTuple{(state,)}((frame,)); selected=state,
            style,size,title=saved.title,caption=saved.caption,
            provenance=(;original=saved.provenance,source_identity=saved.source.identity))
    end
    format in (:svg,:png,:pdf) || throw(ArgumentError("Saved figures support SVG, PNG, PDF or portable HTML"))
    directory = dirname(abspath(path))
    isdir(directory) || throw(ArgumentError("Create the destination directory before exporting a saved figure"))
    return mktempdir(directory) do temporary
        figure_path = joinpath(temporary,basename(path))
        save_visual(figure_path,spec;style,size,backend,kwargs...)
        if sidecar
            provenance = (; original=saved.provenance, source_identity=saved.source.identity,
                figure=(; filename=basename(path),format,sha256=bytes2hex(SHA.sha256(read(figure_path)))))
            record_path = figure_path * ".json"
            save_visual_session(record_path,NamedTuple{(state,)}((frame,));selected=state,
                style,size,title=saved.title,caption=saved.caption,provenance)
            mv(figure_path,path;force=true)
            mv(record_path,path * ".json";force=true)
        else
            mv(figure_path,path;force=true)
        end
        String(path)
    end
end

function save_visual(outdir::AbstractString, stem::AbstractString, saved::SavedVisualSession;
        state=saved.selected,style=saved.style,size=saved.size,kind::Symbol=:auto,cache=:auto,
        prefer::Symbol=:static,format::Symbol=:auto,backend::Symbol=:auto,kwargs...)
    if format === :html || (prefer === :interactive && format === :auto)
        backend in (:auto,:cairomakie) || throw(ArgumentError("Portable saved HTML renders with CairoMakie"))
        backend=:cairomakie
        format=:html
    end
    target = _choose_visual_export_target(outdir,stem;prefer,format,backend)
    save_visual(target.path,saved;state,style,size,kind,cache,backend=target.backend,kwargs...)
    return VisualExportResult(target.path,target.backend,target.format,inspection_snapshot(saved;state).kind,target.stem)
end
