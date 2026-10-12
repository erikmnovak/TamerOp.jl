# Linked signed-term selection. One retained direct answer is reused while the
# query and smoothing stay fixed; display selection never calls the owner again.
"""
    SignedInspectionSession

Linked signed contributions, created with `inspection_session(measure; ...)`.
Use `select_inspection!` to change `term`, `query`, `smoothing`, or `view`.
The query uses exact coordinates and the measure's stated endpoint convention.
A supplied `direct` callable checks one original observable per changed query;
term/view changes reuse that answer. Inputs and snapshot metadata are read-only.
`inspection_snapshot` returns a static export; `close_inspection!` releases the
input, direct callable and listeners while keeping the last snapshot.
"""
mutable struct SignedInspectionSession <: _AbstractInspectionSession
    object::Any
    options::NamedTuple
    initial::NamedTuple
    state::NamedTuple
    snapshot::VisualizationSpec
    direct_queries::Int
    snapshot_builds::Int
    listeners::Dict{Int,Any}
    next_listener::Int
    listener_errors::Vector{String}
    closed::Bool
    busy::Bool
    lock::ReentrantLock
end

"""
    inspection_session(measure; query=nothing, observable=nothing, boundary=nothing,
        direct=nothing, term=nothing, max_terms=40, smoothing=(0,0), window=nothing,
        units=("parameter 1","parameter 2"), background=nothing)

Inspect `RectSignedBarcode{2}`, `PointSignedMeasure{2}` cumulative atoms,
`PointSignedMeasure{4}` rank endpoint pairs, or `IntervalRankSummary`.
The same keywords as `visual_spec(...; kind=:signed_cancellation)` apply.
A rank query is `(p=(x,y), q=(x,y))`; a cumulative query is `(x,y)`; an interval
query is `(vertices=[...],)`. A supplied `direct` callable receives `(p,q+epsilon)`,
the point, or the finite label set respectively, and must return an integer.
Mismatches are displayed explicitly; checking one query is not a global certificate.
Use `term=0` to clear selection. `view` may be `:signed_cancellation`,
`:signed_diagonals`, or `:signed_prominence` where supported. Reset restores
initial query, smoothing and view with no selected term.
"""
function inspection_session(obj::_SignedViewInput;query=nothing,observable=nothing,boundary=nothing,
        direct=nothing,term=nothing,max_terms=40,smoothing=(0,0),window=nothing,
        units=("parameter 1","parameter 2"),background=nothing)
    options=(;observable,boundary,direct,max_terms,window,units,background)
    initial=(;query=deepcopy(query),smoothing=Tuple(smoothing),view=:signed_cancellation)
    shot=visual_spec(obj;kind=initial.view,query,smoothing,term,options...)
    state=merge(initial,(;term=shot.metadata.selected_term,revision=0))
    return SignedInspectionSession(obj,options,initial,state,shot,direct===nothing ? 0 : 1,1,
        Dict{Int,Any}(),1,String[],false,false,ReentrantLock())
end
inspection_selection(s::SignedInspectionSession)=lock(s.lock) do;s.state;end
inspection_snapshot(s::SignedInspectionSession)=lock(s.lock) do;s.snapshot;end
function inspection_summary(s::SignedInspectionSession)
    lock(s.lock) do
        (;kind=:signed_inspection_session,closed=s.closed,selection=s.state,revision=s.state.revision,
            observable=s.snapshot.metadata.observable,terms=length(s.snapshot.metadata.records),
            agreement=s.snapshot.metadata.agreement,reconstructed=s.snapshot.metadata.reconstructed,
            direct_queries=s.direct_queries,snapshot_builds=s.snapshot_builds,
            listener_count=length(s.listeners),listener_errors=Tuple(s.listener_errors),updating=s.busy)
    end
end
describe(s::SignedInspectionSession)=inspection_summary(s)
Base.show(io::IO,s::SignedInspectionSession)=print(io,"SignedInspectionSession(observable=",s.snapshot.metadata.observable,
    ", terms=",length(s.snapshot.metadata.records),", revision=",s.state.revision,", closed=",s.closed,")")
function select_inspection!(s::SignedInspectionSession;term=nothing,query=nothing,smoothing=nothing,view=nothing,reset=false)
    lock(s.lock)
    try
        _inspection_require_open(s)
        s.busy && throw(ArgumentError("Cannot update during another update or callback."))
        reset isa Bool || throw(ArgumentError("reset must be Bool."))
        reset && any(x->x!==nothing,(term,query,smoothing,view)) && throw(ArgumentError("Reset must be a separate action."))
        candidate=reset ? merge(s.initial,(;term=nothing,revision=s.state.revision+1)) :
            (;query=query===nothing ? s.state.query : deepcopy(query),
                smoothing=smoothing===nothing ? s.state.smoothing : smoothing,
                view=view===nothing ? s.state.view : view,term=term===nothing ? s.state.term : term,revision=s.state.revision+1)
        candidate.view in (:signed_cancellation,:signed_diagonals,:signed_prominence) || throw(ArgumentError("Unsupported signed view."))
        same_query=candidate.query==s.state.query && candidate.smoothing==s.state.smoothing
        options=s.options
        if same_query && options.direct!==nothing
            value=s.snapshot.metadata.direct_value
            options=merge(options,(;direct=(args...)->value))
        end
        s.busy=true
        try
            shot=visual_spec(s.object;kind=candidate.view,query=candidate.query,smoothing=candidate.smoothing,term=candidate.term,options...)
            s.state=merge(candidate,(;term=shot.metadata.selected_term,smoothing=shot.metadata.smoothing))
            s.snapshot=shot;s.snapshot_builds+=1
            !same_query && options.direct!==nothing && (s.direct_queries+=1)
        catch;s.busy=false;rethrow();end
    finally;unlock(s.lock);end
    try;_inspection_notify(s)
    finally;lock(s.lock) do;s.busy=false;end;end
    return s
end
reset_inspection!(s::SignedInspectionSession)=select_inspection!(s;reset=true)
function close_inspection!(s::SignedInspectionSession)
    lock(s.lock)
    try
        s.closed && return s
        s.busy && throw(ArgumentError("Cannot close during an update or callback."))
        s.closed=true;s.busy=true
    finally;unlock(s.lock);end
    try;_inspection_notify(s)
    finally
        lock(s.lock) do
            empty!(s.listeners);s.object=nothing;s.options=NamedTuple();s.busy=false
        end
    end
    return s
end
function check_inspection_session(s::SignedInspectionSession;throw=false)
    lock(s.lock) do
        issues=String[]
        s.snapshot.metadata.selected_term==s.state.term || push!(issues,"Snapshot and selected term disagree.")
        s.snapshot.kind==s.state.view || push!(issues,"Snapshot and selected view disagree.")
        s.closed && !s.busy && !(isempty(s.listeners) && s.object===nothing && isempty(s.options)) &&
            push!(issues,"Closed session retains inputs or listeners.")
        append!(issues,check_visual_spec(s.snapshot).issues)
        valid=isempty(issues)
        throw && !valid && _throw_invalid_visual(:check_inspection_session,issues)
        (;kind=:inspection_session,valid,issues,closed=s.closed,revision=s.state.revision)
    end
end
function _visual_spec(s::SignedInspectionSession,::Symbol;kwargs...)
    VisualizationSpec(:linked_inspector;title="Explore signed contributions",panels=inspection_snapshot(s).panels,
        interaction=(;hover=true,clicks=true,labels=true,widgets=(:signed_term,:signed_query,:signed_smoothing),
            notebook=:linked_inspector,mode=:live_julia,requires_live_julia=true,offline_widgets=false),
        metadata=(;session=s,inspector_kind=:signed))
end
_visual_request_cost(::SignedInspectionSession,::Symbol)=(;work=:scan_retained_signed_terms,direct_query=:on_query_change,
    requires_live_julia=true,timing=:not_measured)
