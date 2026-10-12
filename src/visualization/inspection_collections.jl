# Transactional table selection and independent pair drill-down.
"""
    CollectionInspectionSession

Linked supplied-table, embedding, and sample views from `inspection_session(c)`.
Select `view`, `sample`, `target`, or `feature` using `select_inspection!`; labels
and one-based indices are accepted. Selection never computes features, distances
or embeddings. Pair matching is a separate, explicit `collection_matching` call.
Closing releases inputs, witness references and listeners, retaining the final
static snapshot. Independent matching sessions have their own lifecycle.
"""
mutable struct CollectionInspectionSession <: _AbstractInspectionSession
    object::Union{Nothing,VisualCollection}
    initial::NamedTuple
    state::NamedTuple
    snapshot::VisualizationSpec
    snapshot_builds::Int
    listeners::Dict{Int,Any}
    next_listener::Int
    listener_errors::Vector{String}
    closed::Bool
    busy::Bool
    lock::ReentrantLock
end
function inspection_session(c::VisualCollection;kwargs...)
    o=_collection_options(c;kwargs...)
    CollectionInspectionSession(c,o,merge(o,(;revision=0)),_collection_spec(c,o),1,Dict{Int,Any}(),1,String[],false,false,ReentrantLock())
end
inspection_selection(s::CollectionInspectionSession)=lock(s.lock) do;s.state;end
inspection_snapshot(s::CollectionInspectionSession)=lock(s.lock) do;s.snapshot;end
function inspection_summary(s::CollectionInspectionSession)
    lock(s.lock) do
        (;kind=:collection_inspection_session,selection=s.state,revision=s.state.revision,
            snapshot_builds=s.snapshot_builds,all_pairs_computed=false,closed=s.closed,updating=s.busy,
            listener_count=length(s.listeners),listener_errors=Tuple(s.listener_errors))
    end
end
describe(s::CollectionInspectionSession)=inspection_summary(s)
Base.show(io::IO,s::CollectionInspectionSession)=print(io,"CollectionInspectionSession(view=",s.state.view,", closed=",s.closed,")")
function _collection_candidate(s;kwargs...)
    _inspection_require_open(s)
    all(k->k in (:view,:sample,:target,:feature),keys(kwargs)) || throw(ArgumentError("Only view, sample, target and feature are live selections."))
    state=merge(Base.structdiff(s.state,(;revision=s.state.revision)),(;kwargs...))
    _collection_options(s.object;state...)
end
function check_inspection_selection(s::CollectionInspectionSession;throw=false,kwargs...)
    lock(s.lock) do
        issues=String[];candidate=nothing
        try;candidate=_collection_candidate(s;kwargs...)
        catch err;err isa InterruptException && rethrow();push!(issues,sprint(showerror,err));end
        valid=isempty(issues)
        throw && !valid && _throw_invalid_visual(:check_inspection_selection,issues)
        (;kind=:inspection_selection,valid,issues,selection=candidate)
    end
end
function select_inspection!(s::CollectionInspectionSession;kwargs...)
    lock(s.lock)
    try
        _inspection_require_open(s)
        s.busy && throw(ArgumentError("An inspection update or callback is already in progress."))
        o=_collection_candidate(s;kwargs...);s.busy=true
        try
            snapshot=_collection_spec(s.object,o)
            s.snapshot=snapshot;s.state=merge(o,(;revision=s.state.revision+1));s.snapshot_builds+=1
        catch;s.busy=false;rethrow();end
    finally;unlock(s.lock);end
    try;_inspection_notify(s)
    finally;lock(s.lock) do;s.busy=false;end;end
    s
end
reset_inspection!(s::CollectionInspectionSession)=select_inspection!(s;s.initial...)
function close_inspection!(s::CollectionInspectionSession)
    lock(s.lock)
    try
        s.closed && return s
        s.busy && throw(ArgumentError("Cannot close during an inspection update or callback."))
        s.closed=true;s.busy=true
    finally;unlock(s.lock);end
    try;_inspection_notify(s)
    finally;lock(s.lock) do;empty!(s.listeners);s.object=nothing;s.busy=false;end;end
    s
end
function check_inspection_session(s::CollectionInspectionSession;throw=false)
    lock(s.lock) do
        issues=String[]
        s.snapshot.metadata.selection==Base.structdiff(s.state,(;revision=s.state.revision)) || push!(issues,"Snapshot and selection disagree.")
        s.closed && !s.busy && !(s.object===nothing && isempty(s.listeners)) && push!(issues,"Closed session retains inputs/listeners.")
        append!(issues,check_visual_spec(s.snapshot).issues)
        valid=isempty(issues)
        throw && !valid && _throw_invalid_visual(:check_inspection_session,issues)
        (;kind=:inspection_session,valid,issues,closed=s.closed,revision=s.state.revision)
    end
end
function collection_sample(s::CollectionInspectionSession)
    lock(s.lock) do
        _inspection_require_open(s)
        collection_sample(s.object;sample=s.state.sample)
    end
end
function collection_matching(s::CollectionInspectionSession;kwargs...)
    lock(s.lock) do
        _inspection_require_open(s)
        collection_matching(s.object;sample=s.state.sample,target=s.state.target,kwargs...)
    end
end
function _visual_spec(s::CollectionInspectionSession,::Symbol;kwargs...)
    isempty(kwargs) || throw(ArgumentError("Change a collection session with select_inspection!."))
    VisualizationSpec(:linked_inspector;title="Explore a collection",panels=inspection_snapshot(s).panels,
        interaction=(;hover=true,clicks=true,labels=true,widgets=(:collection_view,:collection_sample,:collection_feature),
            notebook=:linked_inspector,mode=:live_julia,requires_live_julia=true,offline_widgets=false),
        metadata=(;session=s,inspector_kind=:collections))
end
_visual_request_cost(::CollectionInspectionSession,::Symbol)=(;work=:supplied_tables,all_pairs=false,hover=:read_only,requires_live_julia=true,timing=:not_measured)
