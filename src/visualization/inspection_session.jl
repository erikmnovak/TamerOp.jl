# Stateful selection orchestration over the existing static inspectors. Input
# mathematics is read-only; renderers subscribe without owning algebra caches.

const _InspectionCacheKey = Tuple{Symbol,Union{Nothing,Int},Union{Nothing,Tuple{Int,Int}},Symbol,Bool}

"""
    InspectionSession

A linked inspection of one fixed finite module or encoding. Construct with
[`inspection_session`](@ref), inspect [`inspection_selection`](@ref), and obtain
an ordinary static figure specification with [`inspection_snapshot`](@ref).

Geometry, module dimensions and the schematic Hasse layout are prepared once.
Selected algebra is retained in a bounded cache. No structure maps or image
bases are requested until the selected view needs them. The input mathematics
and returned specifications must be treated as read-only; reopen the session
after modifying the underlying module, presentation, or encoding map.

Updates are synchronous and transactional. Invalid requests leave selection,
snapshot and session caches unchanged. A backend callback observes the committed
snapshot; reentrant updates are rejected. Use [`close_inspection!`](@ref) to
release callbacks and caches while retaining the final static snapshot.
"""
mutable struct InspectionSession{T,P} <: _AbstractInspectionSession
    object::Union{Nothing,T}
    prepared::Union{Nothing,P}
    state::NamedTuple
    snapshot::VisualizationSpec
    matrix_limit::Tuple{Int,Int}
    cache_limit::Int
    cache::Dict{_InspectionCacheKey,NamedTuple}
    cache_order::Vector{_InspectionCacheKey}
    cache_hits::Int
    cache_misses::Int
    slice_limit::Int
    slice_result::Union{Nothing,NamedTuple}
    slice_cache::Dict{Any,NamedTuple}
    slice_cache_order::Vector{Any}
    slice_cache_hits::Int
    slice_cache_misses::Int
    snapshot_builds::Int
    # Callbacks are heterogeneous callables, including renderer-owned functors.
    listeners::Dict{Int,Any}
    next_listener::Int
    listener_errors::Vector{String}
    capabilities::NamedTuple
    closed::Bool
    busy::Bool
    lock::ReentrantLock
end

function _inspection_integer(value, name; lower=0)
    value isa Integer && !(value isa Bool) && lower <= value <= typemax(Int) ||
        throw(ArgumentError("$name must be an integer in $lower:$(typemax(Int))."))
    return Int(value)
end

function _inspection_support_id(value, count::Int, name::Symbol)
    value === nothing && return nothing
    q = _inspection_integer(value, name; lower=1)
    q <= count || throw(ArgumentError("$name must be a retained support ID in 1:$count."))
    return q
end

function _inspection_prepare(obj, box)
    P = _inspection_poset(obj)
    dims = Int[d for d in _inspection_dimensions(obj)]
    length(dims) == nvertices(P) || throw(ArgumentError("Module dimensions do not match its poset."))
    hasse = _prepare_hasse(P)
    geometry_available = obj isa EncodingResult && _inspection_has_geometry(obj)
    box === nothing || geometry_available ||
        throw(ArgumentError("A viewing box requires a supported planar encoding."))
    pi = geometry_available ? _inspection_classifier(encoding_map(obj)) : nothing
    geometry = geometry_available ? _region_geometry_2d(pi; box) : nothing
    empty_selection = (; vertex=nothing, pair=nothing, query_points=(), parameter_relation=:not_applicable)
    region = geometry_available ? _inspection_region_panel(obj, empty_selection; prepared=geometry) : nothing
    graph = _hasse_spec(P; dims, prepared=hasse, field_label=_inspection_field_label(_inspection_field(obj)))
    H = obj isa EncodingResult ? Results.encoding_presentation(obj) : nothing
    slice_available, slice_unavailable_reason = _inspection_slice_capability(obj, geometry_available)
    capabilities = (; nvertices=nvertices(P), geometry_available,
        slice_available, slice_unavailable_reason,
        presentation_available=H !== nothing,
        nupsets=H === nothing ? 0 : length(FiniteFringe.birth_upsets(H)),
        ndownsets=H === nothing ? 0 : length(FiniteFringe.death_downsets(H)),
        supported_views=H === nothing ? (:module,) : (:module, :presentation),
        geometry_preparations=geometry_available ? 1 : 0, hasse_preparations=1)
    return (; dims, hasse, geometry, region, graph, presentation=H), capabilities
end

function _inspection_initial_state(capabilities; view, upset, downset, slice=nothing, slice_scope=:window)
    view in capabilities.supported_views || throw(ArgumentError(
        "view=$view is unavailable; supported views are $(capabilities.supported_views)."))
    u = upset === nothing ? (capabilities.nupsets == 0 ? nothing : 1) :
        _inspection_support_id(upset, capabilities.nupsets, :upset)
    d = downset === nothing ? (capabilities.ndownsets == 0 ? nothing : 1) :
        _inspection_support_id(downset, capabilities.ndownsets, :downset)
    slice_scope in (:window, :global) || throw(ArgumentError("slice_scope must be :window or :global."))
    chosen_slice = slice === nothing || slice === false ? nothing : _inspection_slice_config(slice)
    chosen_slice === nothing || capabilities.slice_available ||
        throw(ArgumentError(capabilities.slice_unavailable_reason))
    return (; vertex=nothing, pair=nothing, query_points=(), parameter_relation=:not_applicable,
        selector=:none, input=:provided, revision=0, view, basis=false, upset=u, downset=d,
        slice=chosen_slice, slice_scope, interval=nothing)
end

function _inspection_graph(obj, prepared, state)
    vertex = state.vertex === 0 ? nothing : state.vertex
    pair = state.pair === nothing || any(iszero, state.pair) ? nothing : state.pair
    return _hasse_spec(_inspection_poset(obj); dims=prepared.dims, vertex, pair,
        prepared=prepared.hasse, field_label=_inspection_field_label(_inspection_field(obj)))
end

function _inspection_algebra(obj, prepared, state, graph, matrix_limit)
    if state.view === :module
        return (; readout=_inspection_readout(obj, prepared.dims, state,
            graph.metadata.relation, matrix_limit))
    end
    return _presentation_fibers(prepared.presentation, state; basis=state.basis)
end

function _inspection_build_snapshot(obj, prepared, state, graph, algebra, matrix_limit;
                                    materialized_before=nothing, slice_result=nothing)
    spec = if state.view === :module
        _module_visual_spec(obj, :module_inspector; prepared, selection=state, graph,
            inspection_data=algebra.readout, matrix_limit, materialized_before)
    else
        _presentation_visual_spec(obj; prepared, selection=state, graph,
            presentation_data=algebra, matrix_limit, basis=state.basis,
            upset=state.upset, downset=state.downset)
    end
    subtitle = spec.subtitle
    state.input === :pointer && !isempty(state.query_points) &&
        (subtitle *= "\nPointer coordinates are approximate; exact text entry is available.")
    materialized_after = obj isa EncodingResult ? Results.result_summary(obj).materialized : true
    metadata = merge(spec.metadata, (; inspection_selection=state,
        module_materialized_before=materialized_before === nothing ? materialized_after : materialized_before,
        module_materialized_after=materialized_after,
        snapshot_semantics=:static, input_origin=state.input, slice_result,
        coordinate_input=state.input === :pointer && !isempty(state.query_points) ? :approximate_pointer : :provided))
    result = VisualizationSpec(spec.kind; title=spec.title, subtitle, layers=spec.layers,
        panels=spec.panels, axes=spec.axes, legend=spec.legend,
        interaction=spec.interaction, metadata)
    slice_result === nothing || (result = _inspection_slice_snapshot(result, slice_result, state.interval))
    check_visual_spec(result; throw=true)
    return result
end

"""
    inspection_session(obj; view=:module, box=nothing, matrix_limit=(12,12),
                       cache_limit=16, upset=nothing, downset=nothing,
                       slice=nothing, slice_scope=:window, slice_limit=512)

Open a linked inspector for a `PModule` or `EncodingResult`. The initial view
shows finite labels and stored module dimensions without querying structure
maps. Supported planar encodings also show their actual classifier geometry.
`view=:presentation` requires a retained current-poset fringe witness; its
image coordinates are distinct from the stored module's coordinates.

The optional finite planar `box` fixes the viewport for the session. The
`matrix_limit` limits displayed entries, not mathematical matrix construction.
`cache_limit` bounds each of the selected-algebra and selected-slice caches;
zero disables caching.
`upset` and `downset` choose displayed presentation supports, defaulting to the
first support in each nonempty family. They do not restrict the algebra.

Use `select_inspection!` to select a stalk or pair, `inspection_snapshot` for
static export, and `close_inspection!` when finished. Reopen after mutating the
underlying mathematics. Construction computes dimensions and layout once;
later geometry selections retain the supplied exact coordinates.

Opt in to linked slice/barcode/diagram panels with
`slice=(basepoint=(0,0), direction=(1,1))`. The line is `basepoint + t*direction`;
its nonnegative, nonzero direction is not normalized. Exact event points and
the open intervals between them preserve open/closed ends and singleton bars.
With `slice_scope=:window` (default), only the restriction to the finite
viewport is computed: an interval reaching its boundary is censored. Opt in
to `slice_scope=:global` to enumerate the complete line through a globally
represented classifier. Its outer constant strata certify infinite endpoints;
the viewport then clips only the drawing. An uncovered stratum rejects a
global request rather than being inferred zero or essential. Unrepresented labels are
rejected. Slice work may materialize the module. `slice_limit` bounds event
strata before the quadratic barcode rank calculation. Reopen with a larger
limit or a smaller box if that budget is exceeded.
Polyhedral encodings currently require rational line and viewport coordinates;
integer and finite floating inputs are preserved as exact rationals. Irrational
algebraic coordinates remain supported for box and positively oriented grid
encodings, whose classifiers retain them exactly.
"""
function inspection_session(obj::Union{Modules.PModule,EncodingResult}; view=:module,
                            box=nothing, matrix_limit=(12,12), cache_limit=16,
                            upset=nothing, downset=nothing, slice=nothing, slice_scope=:window, slice_limit=512)
    view isa Symbol || throw(ArgumentError("view must be :module or :presentation."))
    (matrix_limit isa Tuple || matrix_limit isa AbstractVector) && length(matrix_limit) == 2 ||
        throw(ArgumentError("matrix_limit must contain row and column counts."))
    limit = (_inspection_integer(matrix_limit[1], :matrix_rows; lower=1),
             _inspection_integer(matrix_limit[2], :matrix_columns; lower=1))
    capacity = _inspection_integer(cache_limit, :cache_limit)
    stratum_limit = _inspection_integer(slice_limit, :slice_limit; lower=1)
    prepared, capabilities = _inspection_prepare(obj, box)
    state = _inspection_initial_state(capabilities; view, upset, downset, slice, slice_scope)
    if state.slice !== nothing
        state = merge(state, (; slice=_inspection_slice_coordinates(obj, state.slice, prepared.geometry.box)))
    end
    materialized_before = obj isa EncodingResult ? Results.result_summary(obj).materialized : true
    slice_result = state.slice === nothing ? nothing :
        _inspection_slice_result(obj, prepared, state.slice, stratum_limit; scope=state.slice_scope)
    algebra = _inspection_algebra(obj, prepared, state, prepared.graph, limit)
    snapshot = _inspection_build_snapshot(obj, prepared, state, prepared.graph, algebra, limit;
        materialized_before, slice_result)
    session = InspectionSession{typeof(obj),typeof(prepared)}(obj, prepared, state, snapshot,
        limit, capacity, Dict{_InspectionCacheKey,NamedTuple}(), _InspectionCacheKey[],
        0, 0, stratum_limit, slice_result, Dict{Any,NamedTuple}(), Any[], 0, 0,
        1, Dict{Int,Any}(), 1, String[], capabilities, false, false, ReentrantLock())
    slice_result === nothing || _inspection_commit_slice_cache!(session, (state.slice, state.slice_scope), slice_result, false)
    return session
end

"""
    inspection_selection(session)

Return the immutable current selection: actual `vertex` or `pair`, immutable
`query_points`, original `parameter_relation`, selector kind, input origin,
revision, view, single-stalk basis opt-in, and displayed upset/downset IDs.
Label `0` from a parameter query means outside, not a represented zero space.
`slice` is the independent exact line configuration; `interval` is a selected
multiplicity-group ID in the current slice, not an identity across moving lines.
"""
inspection_selection(s::InspectionSession) = lock(s.lock) do
    s.state
end

"""
    inspection_snapshot(session)

Return the current ordinary static `VisualizationSpec`, suitable for existing
rendering/export functions. It contains no live callbacks and remains available
after closing. Treat its arrays and retained algebra as read-only.
"""
inspection_snapshot(s::InspectionSession) = lock(s.lock) do
    s.snapshot
end

"""
    inspection_summary(session)

Return a cheap summary of selection, available views, preparation counts,
bounded algebra-cache use, listener failures, and closure status. It performs
no rank, basis, map, geometry or layout calculation.
"""
function inspection_summary(s::InspectionSession)
    return lock(s.lock) do
        merge(s.capabilities, (; kind=:inspection_session, closed=s.closed,
            view=s.state.view, basis=s.state.basis, revision=s.state.revision,
            selection=s.state, matrix_limit=s.matrix_limit,
            cache_entries=length(s.cache), cache_limit=s.cache_limit,
            cache_hits=s.cache_hits, cache_misses=s.cache_misses,
            slice_limit=s.slice_limit, slice_cache_entries=length(s.slice_cache),
            slice_cache_hits=s.slice_cache_hits, slice_cache_misses=s.slice_cache_misses,
            slice_active=s.state.slice !== nothing, slice_scope=s.state.slice_scope,
            slice_intervals=s.slice_result === nothing ? 0 : length(s.slice_result.intervals),
            snapshot_builds=s.snapshot_builds, listener_count=length(s.listeners),
            listener_errors=Tuple(s.listener_errors), updating=s.busy,
            module_materialized=get(s.snapshot.metadata, :module_materialized_after, true),
            cost=(; dimensions=:prepared_once, geometry=:prepared_once_when_available,
                hasse_layout=:prepared_once, selected_algebra=:bounded_cache,
                slices=:exact_event_strata_with_bounded_cache,
                snapshot=:static, updates=:synchronous)))
    end
end

describe(s::InspectionSession) = inspection_summary(s)

function Base.show(io::IO, s::InspectionSession)
    d = inspection_summary(s)
    print(io, "InspectionSession(view=", d.view, ", revision=", d.revision,
        ", cache=", d.cache_entries, "/", d.cache_limit, ", closed=", d.closed, ")")
end

function Base.show(io::IO, ::MIME"text/plain", s::InspectionSession)
    show(io, s)
    d = inspection_summary(s)
    print(io, "\n  selection: ", d.selection, "\n  supported views: ", d.supported_views,
        "\n  geometry: ", d.geometry_available, "\n  cache hits/misses: ", d.cache_hits, "/", d.cache_misses)
end

function _inspection_require_open(s)
    s.closed && throw(ArgumentError("This inspection session is closed; open a new session."))
    return nothing
end

function _inspection_candidate(s; vertex=nothing, pair=nothing, point=nothing,
                               parameter_pair=nothing, view=nothing, basis=nothing,
                               upset=nothing, downset=nothing, input=:provided,
                               slice=nothing, slice_scope=nothing, interval=nothing)
    _inspection_require_open(s)
    input isa Symbol && input in (:provided, :pointer) || throw(ArgumentError("input must be :provided or :pointer."))
    selected_view = view === nothing ? s.state.view : view
    selected_view isa Symbol && selected_view in s.capabilities.supported_views || throw(ArgumentError(
        "view=$selected_view is unavailable; supported views are $(s.capabilities.supported_views)."))
    selectors = (; vertex, pair, point, parameter_pair)
    supplied = Tuple(k for k in keys(selectors) if selectors[k] !== nothing)
    length(supplied) <= 1 || throw(ArgumentError("Select one of vertex, pair, point, or parameter_pair."))
    if !s.capabilities.geometry_available && (point !== nothing || parameter_pair !== nothing)
        throw(ArgumentError("Parameter selection requires a supported planar encoding."))
    end
    issues = String[]
    _check_module_selection!(issues, s.object, :module_inspector; vertex, pair, point, parameter_pair)
    isempty(issues) || throw(ArgumentError(join(issues, " ")))
    chosen = isempty(supplied) ? s.state : _inspection_selection(s.object; vertex, pair, point, parameter_pair)
    selector = isempty(supplied) ? s.state.selector : only(supplied)
    single = selector in (:vertex, :point)
    chosen_basis = basis === nothing ? (selected_view === :presentation && single && s.state.basis) : basis
    chosen_basis isa Bool || throw(ArgumentError("basis must be true or false."))
    chosen_basis && !(selected_view === :presentation && single) &&
        throw(ArgumentError("basis=true requires a single selected stalk in presentation view."))
    chosen_upset = upset === nothing ? s.state.upset :
        _inspection_support_id(upset, s.capabilities.nupsets, :upset)
    chosen_downset = downset === nothing ? s.state.downset :
        _inspection_support_id(downset, s.capabilities.ndownsets, :downset)
    chosen_slice = slice === nothing ? s.state.slice :
        (slice === false ? nothing : _inspection_slice_config(slice))
    chosen_slice === nothing || s.capabilities.slice_available ||
        throw(ArgumentError(s.capabilities.slice_unavailable_reason))
    chosen_slice === nothing || (chosen_slice =
        _inspection_slice_coordinates(s.object, chosen_slice, s.prepared.geometry.box))
    chosen_scope = slice_scope === nothing ? s.state.slice_scope : slice_scope
    chosen_scope in (:window, :global) || throw(ArgumentError("slice_scope must be :window or :global."))
    changed_slice = chosen_slice != s.state.slice || chosen_scope != s.state.slice_scope
    chosen_interval = if interval === nothing
        changed_slice ? nothing : s.state.interval
    else
        id = _inspection_integer(interval, :interval)
        iszero(id) ? nothing : id
    end
    chosen_interval === nothing || chosen_slice !== nothing ||
        throw(ArgumentError("An interval selection requires an active slice."))
    !changed_slice && _inspection_slice_validate_interval(s.slice_result, chosen_interval)
    return (; vertex=chosen.vertex, pair=chosen.pair, query_points=Tuple(chosen.query_points),
        parameter_relation=chosen.parameter_relation, selector,
        input=isempty(supplied) ? s.state.input : input,
        revision=s.state.revision + 1, view=selected_view, basis=chosen_basis,
        upset=chosen_upset, downset=chosen_downset, slice=chosen_slice, slice_scope=chosen_scope, interval=chosen_interval)
end

"""
    check_inspection_selection(session; throw=false, kwargs...)

Validate a proposed `select_inspection!` request without computing ranks, maps
or bases, and without changing selection or caches. Parameter queries may call
the classifier. The report includes the proposed immutable selection when
valid. Wrap reports with `VisualizationValidationSummary(report)` for display.
For a changed line, interval IDs are checked after the new restriction is
computed by `select_inspection!`; this cheap check validates their format only.
"""
function check_inspection_selection(s::InspectionSession; throw::Bool=false, kwargs...)
    return lock(s.lock) do
        issues = String[]
        candidate = nothing
        try
            candidate = _inspection_candidate(s; kwargs...)
        catch err
            err isa InterruptException && rethrow()
            push!(issues, sprint(showerror, err))
        end
        valid = isempty(issues)
        throw && !valid && _throw_invalid_visual(:check_inspection_selection, issues)
        return (; kind=:inspection_selection, valid, issues, selection=candidate)
    end
end

function _inspection_cache_key(state)
    return (state.view, state.vertex, state.pair, state.parameter_relation, state.basis)
end

function _inspection_commit_cache!(s, key, algebra, hit)
    hit ? (s.cache_hits += 1) : (s.cache_misses += 1)
    s.cache_limit == 0 && return nothing
    position = findfirst(isequal(key), s.cache_order)
    position === nothing || deleteat!(s.cache_order, position)
    s.cache[key] = algebra
    push!(s.cache_order, key)
    while length(s.cache_order) > s.cache_limit
        delete!(s.cache, popfirst!(s.cache_order))
    end
    return nothing
end

function _inspection_commit_slice_cache!(s, key, result, hit)
    hit ? (s.slice_cache_hits += 1) : (s.slice_cache_misses += 1)
    s.cache_limit == 0 && return nothing
    position = findfirst(isequal(key), s.slice_cache_order)
    position === nothing || deleteat!(s.slice_cache_order, position)
    s.slice_cache[key] = result
    push!(s.slice_cache_order, key)
    while length(s.slice_cache_order) > s.cache_limit
        delete!(s.slice_cache, popfirst!(s.slice_cache_order))
    end
    return nothing
end

function _inspection_notify(s)
    callbacks = lock(s.lock) do
        [(token, s.listeners[token]) for token in sort!(collect(keys(s.listeners)))]
    end
    for (token, callback) in callbacks
        active = lock(s.lock) do
            haskey(s.listeners, token)
        end
        active || continue
        try
            callback(s)
        catch err
            err isa InterruptException && rethrow()
            lock(s.lock) do
                push!(s.listener_errors, "listener $token: " * sprint(showerror, err))
                length(s.listener_errors) <= 16 || popfirst!(s.listener_errors)
            end
        end
    end
    return nothing
end

function _inspection_update!(s, candidate_builder)
    lock(s.lock)
    try
        _inspection_require_open(s)
        s.busy && throw(ArgumentError("An inspection update or callback is already in progress."))
        s.busy = true
        try
            state = candidate_builder()
            key = _inspection_cache_key(state)
            selected = state.selector !== :none
            cached = selected ? get(s.cache, key, nothing) : nothing
            materialized_before = s.object isa EncodingResult ? Results.result_summary(s.object).materialized : true
            # Keeping a selected line while inspecting a stalk or a bar reuses
            # the retained restriction even when cache_limit is zero. Cache
            # accounting records only changes to the active line, not
            # interval/readout-only interaction.
            slice_key = (state.slice, state.slice_scope)
            previous_slice_key = (s.state.slice, s.state.slice_scope)
            slice_requested = state.slice !== nothing && slice_key != previous_slice_key
            cached_slice = state.slice === nothing ? nothing :
                (slice_key == previous_slice_key ? s.slice_result : get(s.slice_cache, slice_key, nothing))
            slice_result = state.slice === nothing ? nothing :
                (cached_slice === nothing ?
                    _inspection_slice_result(s.object, s.prepared, state.slice, s.slice_limit; scope=state.slice_scope) : cached_slice)
            _inspection_slice_validate_interval(slice_result, state.interval)
            graph = _inspection_graph(s.object, s.prepared, state)
            algebra = cached === nothing ?
                _inspection_algebra(s.object, s.prepared, state, graph, s.matrix_limit) : cached
            snapshot = _inspection_build_snapshot(s.object, s.prepared, state, graph, algebra, s.matrix_limit;
                materialized_before, slice_result)
            # Commit only after every mathematical and display operation has
            # succeeded. Underlying owners may maintain their own benign caches.
            selected && _inspection_commit_cache!(s, key, algebra, cached !== nothing)
            slice_requested && _inspection_commit_slice_cache!(s, slice_key, slice_result, cached_slice !== nothing)
            s.slice_result = slice_result
            s.state, s.snapshot = state, snapshot
            s.snapshot_builds += 1
        catch
            s.busy = false
            rethrow()
        end
    finally
        unlock(s.lock)
    end
    try
        _inspection_notify(s)
    finally
        lock(s.lock) do
            s.busy = false
        end
    end
    return s
end

"""
    select_inspection!(session; vertex=nothing, pair=nothing, point=nothing,
        parameter_pair=nothing, view=nothing, basis=nothing,
        upset=nothing, downset=nothing, input=:provided,
        slice=nothing, slice_scope=nothing, interval=nothing)

Select one finite vertex/pair or one planar point/parameter pair. Selectors are
mutually exclusive; changing coordinate mode replaces both old endpoints.
With no selector, view/basis/support changes preserve the current selection.
Parameter pairs retain original order information even when labels coincide.

`view=:module` reads stored module coordinates. `view=:presentation` reads the
retained fringe image. A defined presentation pair explicitly computes its
endpoint bases; `basis=true` is an opt-in for a single presentation stalk.
Switching to pair/module mode clears that single-stalk opt-in. Support IDs only
change displayed membership. `input=:pointer` identifies approximate pointer
coordinates; supplied text/numeric values use `:provided`. View-only changes
retain the origin of the selected coordinates.

`slice=(basepoint=(x,y), direction=(dx,dy))` changes the independent line while
preserving selected stalks/maps; `slice=false` removes the slice panels.
Changing the line or `slice_scope=:window|:global` clears its interval selection. `interval=i` selects a
displayed multiplicity group, shared by barcode and diagram; `interval=0`
clears it. These IDs identify intervals only within the current slice and do
not track classes across line changes. Changing just the interval reuses the
existing restriction without recomputing ranks. Values of `nothing` preserve
the existing line/interval. See `inspection_session` for finite-window,
endpoint, missing-label and budget semantics.

Invalid requests leave session state, snapshot and session cache unchanged.
Updates are synchronous; concurrent/reentrant changes during notification are
rejected. Backend listener errors are recorded in `inspection_summary` after
the valid selection has committed, rather than undoing that selection.
"""
function select_inspection!(s::InspectionSession; kwargs...)
    return _inspection_update!(s, () -> _inspection_candidate(s; kwargs...))
end

"""
    reset_inspection!(session)

Clear both endpoints and the single-stalk basis opt-in, retaining the current
view, viewport, displayed supports and slice line. The slice interval selection
is also cleared. Prepared geometry/layout and the bounded caches remain
reusable. The revision increases and listeners are notified.
"""
function reset_inspection!(s::InspectionSession)
    return _inspection_update!(s, () -> merge(s.state,
        (; vertex=nothing, pair=nothing, query_points=(), parameter_relation=:not_applicable,
            selector=:none, input=:provided, basis=false, interval=nothing,
            revision=s.state.revision + 1)))
end

"""
    close_inspection!(session)

Close the session, notify listeners once so backends can dispose their event
handlers, and release callbacks, prepared navigation data and algebra caches.
The final static snapshot and selection remain inspectable/exportable. Closing
an already closed session is harmless; updates require opening a new session.
"""
function close_inspection!(s::InspectionSession)
    lock(s.lock)
    try
        s.closed && return s
        s.busy && throw(ArgumentError("Cannot close during an inspection update or callback."))
        s.busy = true
        s.closed = true
    finally
        unlock(s.lock)
    end
    try
        _inspection_notify(s)
    finally
        lock(s.lock) do
            empty!(s.listeners)
            empty!(s.cache)
            empty!(s.cache_order)
            empty!(s.slice_cache)
            empty!(s.slice_cache_order)
            s.prepared = nothing
            s.object = nothing
            s.busy = false
        end
    end
    return s
end

"""
    check_inspection_session(session; throw=false)

Check the session's selection/snapshot revision, bounded cache bookkeeping and
closed-session cleanup, then validate its current static specification. This
does not recompute the mathematical input or certify it was left unchanged.
Returns a report accepted by `VisualizationValidationSummary`.
"""
function check_inspection_session(s::InspectionSession; throw::Bool=false)
    return lock(s.lock) do
        issues = String[]
        length(s.cache) <= s.cache_limit || push!(issues, "algebra cache exceeds its limit")
        length(s.cache_order) == length(s.cache) && Set(s.cache_order) == Set(keys(s.cache)) ||
            push!(issues, "algebra cache order and entries disagree")
        length(s.slice_cache) <= s.cache_limit || push!(issues, "slice cache exceeds its limit")
        length(s.slice_cache_order) == length(s.slice_cache) &&
            Set(s.slice_cache_order) == Set(keys(s.slice_cache)) ||
            push!(issues, "slice cache order and entries disagree")
        (s.state.slice === nothing) == (s.slice_result === nothing) ||
            push!(issues, "slice selection and retained result disagree")
        if s.slice_result !== nothing
            s.slice_result.line == s.state.slice || push!(issues, "retained slice has a stale line")
            get(s.slice_result, :scope, :window) == s.state.slice_scope || push!(issues, "retained slice has a stale scope")
            get(s.snapshot.metadata, :slice_result, nothing) == s.slice_result ||
                push!(issues, "snapshot and retained slice disagree")
            s.state.interval === nothing || 1 <= s.state.interval <= length(s.slice_result.intervals) ||
                push!(issues, "selected interval is not in the current slice")
        end
        get(s.snapshot.metadata, :inspection_selection, nothing) == s.state ||
            push!(issues, "snapshot and current selection disagree")
        if s.closed && !s.busy
            isempty(s.listeners) && isempty(s.cache) && isempty(s.slice_cache) &&
                s.prepared === nothing && s.object === nothing ||
                push!(issues, "closed session retains live callbacks or prepared caches")
        elseif !s.closed
            s.prepared !== nothing && s.object !== nothing || push!(issues, "open session has no prepared input")
        end
        append!(issues, check_visual_spec(s.snapshot).issues)
        valid = isempty(issues)
        throw && !valid && _throw_invalid_visual(:check_inspection_session, issues)
        return (; kind=:inspection_session, valid, issues, closed=s.closed,
            revision=s.state.revision, cache_entries=length(s.cache))
    end
end

function _on_inspection(s::_AbstractInspectionSession, callback)
    return lock(s.lock) do
        _inspection_require_open(s)
        applicable(callback, s) || throw(ArgumentError("An inspection listener must accept the session."))
        token = s.next_listener
        s.next_listener += 1
        s.listeners[token] = callback
        token
    end
end

function _off_inspection!(s::_AbstractInspectionSession, token)
    return lock(s.lock) do
        pop!(s.listeners, token, nothing)
        nothing
    end
end

function _inspection_scene(s::InspectionSession)
    return lock(s.lock) do
        _inspection_require_open(s)
        (; region=s.prepared.region, hasse=s.prepared.graph)
    end
end

function _inspection_hover(s::InspectionSession; point=nothing, vertex=nothing, input=:pointer)
    return lock(s.lock) do
        _inspection_require_open(s)
        input isa Symbol && input in (:provided, :pointer) || throw(ArgumentError("input must be :provided or :pointer."))
        (point === nothing) != (vertex === nothing) || throw(ArgumentError("Hover requires exactly one point or vertex."))
        p = nothing
        q = if point === nothing
            id = _inspection_integer(vertex, :vertex; lower=1)
            id <= s.capabilities.nvertices || throw(ArgumentError("vertex is outside the finite poset."))
            id
        else
            s.capabilities.geometry_available || throw(ArgumentError("Parameter hover requires planar geometry."))
            p = only(_collect_query_points(; point))
            _visual_locate(_inspection_classifier(encoding_map(s.object)), p)
        end
        return (; vertex=q, dimension=q == 0 ? nothing : s.prepared.dims[q], point=p,
            input, approximate=point !== nothing && input === :pointer)
    end
end

# Exact text input for backend controls. Decimal syntax denotes its written
# rational value; expressions, identifiers, calls, and nonfinite values fail.
function _parse_inspection_coordinate(text::AbstractString)
    ncodeunits(text) <= 4096 || throw(ArgumentError("Coordinate text must contain at most 4096 bytes."))
    value = strip(text)
    if occursin('/', value)
        m = match(r"^([+-]?\d+)\s*(?://|/)\s*([+-]?\d+)$", value)
        m === nothing && throw(ArgumentError("Enter an integer, decimal, p/q, or p//q."))
        denominator = parse(BigInt, m.captures[2])
        iszero(denominator) && throw(ArgumentError("A coordinate denominator cannot be zero."))
        return parse(BigInt, m.captures[1]) // denominator
    end
    m = match(r"^([+-]?)(?:(\d+)(?:\.(\d*))?|\.(\d+))(?:[eE]([+-]?\d+))?$", value)
    m === nothing && throw(ArgumentError("Enter an integer, decimal, p/q, or p//q."))
    whole = something(m.captures[2], "0")
    fraction = something(m.captures[3], m.captures[4], "")
    numerator = parse(BigInt, whole * fraction)
    m.captures[1] == "-" && (numerator = -numerator)
    exponent = m.captures[5] === nothing ? 0 : tryparse(Int, m.captures[5])
    exponent !== nothing && -10000 <= exponent <= 10000 ||
        throw(ArgumentError("A decimal coordinate exponent must lie in -10000:10000."))
    power = Base.checked_sub(exponent, length(fraction))
    return power >= 0 ? (numerator * big(10)^power) // big(1) : numerator // big(10)^(-power)
end
