"""
    SummandInspectionSession

A checked splitting with retained geometry and bounded selected Hom spaces.
Use `summand_session`; shared inspection accessors select, reset, freeze or close
it. Selection is transactional. Closing releases algebra, geometry and Hom data
while preserving the last static snapshot.
"""
mutable struct SummandInspectionSession <: _AbstractInspectionSession
    collection::Any
    prepared::NamedTuple
    initial::NamedTuple
    state::NamedTuple
    hom_cache::Dict{Tuple{Int,Int},Any}
    snapshot::VisualizationSpec
    listeners::Dict{Int,Any}
    next_listener::Int
    listener_errors::Vector{String}
    closed::Bool
    busy::Bool
    lock::ReentrantLock
end
"""
    summand_session(result_or_collection; encoding=nothing, drawing=nothing,
        selected=nothing, focus=nothing, vertex=1, layout=:facets,
        sort_by=:index, type_filter=:all, min_value=0, limit=8, hom=false, kwargs...)

Explore a checked splitting with full/selected/complement dimensions, individual
supports and actual component matrices. An A46 result must retain witnesses.
`encoding` supplies exact classifier fibers on the identical base; `drawing`
supplies explicitly unverified box unions. Without either, finite diagrams use
schematic order positions. All panels keep the same axes while browsing.

Select original block IDs with `selected`; ranking/filtering/`limit` determine
which of those blocks are visible. `focus=0` clears details. `:interval` means an
A46-certified interval block; `:thin` means stalk dimensions at most one, not an
interval proof. `:thick` is its complement. `layout` is `:facets`, `:isolate`,
`:overlay` or `:outline` (the last two require support geometry). Quantitative
views use shared fixed dimension bins by default; `scale=:log` retains white
zero. `shared_scale=false` explicitly gives panels independent scales.

`pair=(u,v)` shows the focused block's actual structure map when ordered.
`hom=true` computes ordinary Hom dimensions for displayed blocks; `hom_pair=(a,b)`
and `basis_index` show an actual Hom basis component at `vertex`. This is neither
Hom^alpha nor Hasse adjacency. `max_hom_pairs=64` and `max_hom_entries=1_000_000`
bound each retained table and solve. `matrix_limit=(8,8)` truncates only display.

Selection changes reuse the checked splitting and prepared geometry. Call
`select_inspection!`, `reset_inspection!` and `close_inspection!` as for other
inspectors. Live controls require WGLMakie and Julia. Static snapshots and saved
finite states work without callbacks; no source reconstruction codec is supplied.
"""
function summand_session(input::_SummandInput;kwargs...)
    c=input isa SummandCollection ? input : summand_collection(input)
    prepkeys=(:encoding,:box,:drawing,:max_vertices,:max_summands,:max_hom_pairs,:max_hom_entries)
    all(k->k in (prepkeys...,_SUMMAND_OPTIONS...),keys(kwargs)) || throw(ArgumentError("Unknown summand session option."))
    prepared=_summand_prepare(c;(; (k=>v for (k,v) in kwargs if k in prepkeys)...)...)
    o=_summand_options(c,prepared;(; (k=>v for (k,v) in kwargs if k in _SUMMAND_OPTIONS)...)...)
    cache=Dict{Tuple{Int,Int},Any}();snap=_summand_snapshot(c,prepared,o;hom_cache=cache)
    SummandInspectionSession(c,prepared,deepcopy(o),merge(o,(;revision=0)),cache,snap,
        Dict{Int,Any}(),1,String[],false,false,ReentrantLock())
end
inspection_selection(s::SummandInspectionSession)=lock(s.lock) do;deepcopy(s.state);end
inspection_snapshot(s::SummandInspectionSession)=lock(s.lock) do;s.snapshot;end
inspection_summary(s::SummandInspectionSession)=lock(s.lock) do
    (;kind=:summand_inspection_session,selection=deepcopy(s.state),revision=s.state.revision,
        certification=s.snapshot.metadata.certification,geometry_status=s.snapshot.metadata.geometry_status,
        selected_dimensions=copy(s.snapshot.metadata.selected_dimensions),complement_dimensions=copy(s.snapshot.metadata.complement_dimensions),
        hom_spaces=length(s.hom_cache),closed=s.closed,updating=s.busy,listener_count=length(s.listeners),listener_errors=Tuple(s.listener_errors))
end
describe(s::SummandInspectionSession)=inspection_summary(s)
Base.show(io::IO,s::SummandInspectionSession)=print(io,"SummandInspectionSession(focus=",s.state.focus,", closed=",s.closed,")")
function _summand_candidate(s;kwargs...)
    _inspection_require_open(s)
    all(k->k in _SUMMAND_OPTIONS,keys(kwargs)) || throw(ArgumentError("Selection cannot change the prepared splitting, geometry or budgets."))
    prior=Base.structdiff(s.state,(;revision=s.state.revision,ids=s.state.ids))
    # Filters can remove the current focus. Use the first remaining block unless
    # the caller explicitly selected one. Preserve exact original IDs.
    if any(k->k in (:selected,:sort_by,:min_value,:type_filter,:limit),keys(kwargs)) && !haskey(kwargs,:focus)
        prior=merge(prior,(;focus=nothing,hom_pair=nothing))
    end
    get(kwargs,:hom,true)===false && (prior=merge(prior,(;hom_pair=nothing)))
    _summand_options(s.collection,s.prepared;merge(prior,(;kwargs...))...)
end
function check_inspection_selection(s::SummandInspectionSession;throw=false,kwargs...)
    lock(s.lock) do
        issues=String[];selection=nothing
        try;selection=_summand_candidate(s;kwargs...)
        catch err;err isa InterruptException&&rethrow();push!(issues,sprint(showerror,err));end
        valid=isempty(issues);throw&&!valid&&_throw_invalid_visual(:check_inspection_selection,issues)
        (;kind=:inspection_selection,valid,issues,selection)
    end
end
function select_inspection!(s::SummandInspectionSession;reset=false,kwargs...)
    lock(s.lock)
    try
        _inspection_require_open(s);s.busy&&throw(ArgumentError("An inspection update is already in progress."))
        reset isa Bool || throw(ArgumentError("reset must be Bool."))
        reset&&!isempty(kwargs)&&throw(ArgumentError("Reset must be a separate action."))
        o=reset ? deepcopy(s.initial) : _summand_candidate(s;kwargs...)
        s.busy=true
        try
            cache=copy(s.hom_cache)
            snap=_summand_snapshot(s.collection,s.prepared,o;hom_cache=cache)
            s.state=merge(o,(;revision=s.state.revision+1));s.snapshot=snap;s.hom_cache=cache
        catch;s.busy=false;rethrow();end
    finally;unlock(s.lock);end
    try;_inspection_notify(s)
    finally;lock(s.lock) do;s.busy=false;end;end
    s
end
reset_inspection!(s::SummandInspectionSession)=select_inspection!(s;reset=true)
function close_inspection!(s::SummandInspectionSession)
    lock(s.lock)
    try
        s.closed&&return s;s.busy&&throw(ArgumentError("Cannot close during an update."))
        s.closed=true;s.busy=true
    finally;unlock(s.lock);end
    try;_inspection_notify(s)
    finally;lock(s.lock) do;empty!(s.listeners);empty!(s.hom_cache);s.collection=nothing;s.prepared=(;);s.busy=false;end;end
    s
end
function check_inspection_session(s::SummandInspectionSession;throw=false)
    lock(s.lock) do
        issues=copy(check_visual_spec(s.snapshot).issues)
        s.snapshot.metadata.selection==Base.structdiff(s.state,(;revision=s.state.revision))||push!(issues,"Snapshot and selection disagree.")
        s.closed&&!s.busy&&!(s.collection===nothing&&isempty(s.prepared)&&isempty(s.hom_cache)&&isempty(s.listeners))&&push!(issues,"Closed session retains runtime objects.")
        valid=isempty(issues);throw&&!valid&&_throw_invalid_visual(:check_inspection_session,issues)
        (;kind=:inspection_session,valid,issues,closed=s.closed,revision=s.state.revision)
    end
end
function _visual_spec(s::SummandInspectionSession,::Symbol;kwargs...)
    isempty(kwargs)||throw(ArgumentError("Change a summand session with select_inspection!."))
    VisualizationSpec(:linked_inspector;title="Explore module summands",panels=inspection_snapshot(s).panels,
        interaction=(;hover=true,clicks=true,labels=true,widgets=(:summand_selection,:summand_gallery,:summand_hom),
            notebook=:linked_inspector,mode=:live_julia,requires_live_julia=true,offline_widgets=false),
        metadata=(;session=s,inspector_kind=:summands))
end
_visual_request_cost(::SummandInspectionSession,::Symbol)=(;work=:retained_split,geometry=:prepared_once,
    selection=:no_decomposition,hom=:bounded_cached_opt_in,requires_live_julia=true,timing=:not_measured)
