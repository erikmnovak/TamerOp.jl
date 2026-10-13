# Path selections reuse retained mathematics; playback belongs to each client.
"""Linked path, interval and source selection. Construct with `inspection_session(path)`.
All normal selection/reset/close and static snapshot operations are supported.
No persistence reduction is performed by selection or playback."""
mutable struct PathInspectionSession <: _AbstractInspectionSession
    object::Union{Nothing,PersistencePath}
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
    requests::_InspectionRequests
end
function inspection_session(p::PersistencePath;kwargs...)
    initial=_path_selection(p;kwargs...)
    PathInspectionSession(p,initial,merge(initial,(;revision=0)),visual_spec(p;initial...),1,
        Dict{Int,Any}(),1,String[],false,false,ReentrantLock(),_InspectionRequests())
end
inspection_selection(s::PathInspectionSession)=lock(()->s.state,s.lock)
inspection_snapshot(s::PathInspectionSession)=lock(()->s.snapshot,s.lock)
function inspection_summary(s::PathInspectionSession)
    lock(s.lock) do
        (;kind=:path_inspection_session,selection=s.state,revision=s.state.revision,closed=s.closed,updating=s.busy,
          snapshot_builds=s.snapshot_builds,listener_count=length(s.listeners),listener_errors=Tuple(s.listener_errors),
          requests=_inspection_request_summary(s),mode=s.snapshot.metadata.mode,parameter=s.snapshot.metadata.parameter)
    end
end
describe(s::PathInspectionSession)=inspection_summary(s)
Base.show(io::IO,s::PathInspectionSession)=print(io,"PathInspectionSession(frame=",s.state.frame,", closed=",s.closed,")")
function _path_candidate(s;kwargs...)
    _inspection_require_open(s)
    old=Base.structdiff(s.state,(;revision=s.state.revision))
    candidate=merge(old,(;kwargs...))
    # Clearing a selected track also clears its optional witness request.
    candidate.track in (nothing,0) && !haskey(kwargs,:representative) &&
        (candidate=merge(candidate,(;representative=false)))
    merge(_path_selection(s.object;candidate...),(;revision=s.state.revision+1))
end
function check_inspection_selection(s::PathInspectionSession;throw=false,kwargs...)
    lock(s.lock) do
        issues=String[];candidate=nothing
        try;candidate=_path_candidate(s;kwargs...)
        catch err;err isa InterruptException && rethrow();push!(issues,sprint(showerror,err));end
        valid=isempty(issues);throw && !valid && _throw_invalid_visual(:check_inspection_selection,issues)
        (;kind=:inspection_selection,valid,issues,selection=candidate)
    end
end
function select_inspection!(s::PathInspectionSession;kwargs...)
    _cancel_inspection_requests!(s)
    lock(s.lock) do
        _inspection_require_open(s);s.busy && throw(ArgumentError("A path update is in progress."))
        state=_path_candidate(s;kwargs...)
        snapshot=visual_spec(s.object;Base.structdiff(state,(;revision=state.revision))...)
        s.state=state;s.snapshot=snapshot;s.snapshot_builds+=1;s.busy=true
    end
    try;_inspection_notify(s)
    finally;lock(s.lock) do;s.busy=false;end;end
    s
end
reset_inspection!(s::PathInspectionSession)=select_inspection!(s;s.initial...)
function close_inspection!(s::PathInspectionSession)
    _cancel_inspection_requests!(s)
    lock(s.lock) do
        s.closed && return s
        s.busy && throw(ArgumentError("Cannot close during a path update."));s.closed=true
    end
    try;_inspection_notify(s)
    finally;lock(s.lock) do;empty!(s.listeners);s.object=nothing;end;end
    s
end
function check_inspection_session(s::PathInspectionSession;throw=false)
    lock(s.lock) do
        issues=copy(check_visual_spec(s.snapshot).issues)
        s.snapshot.metadata.selection==Base.structdiff(s.state,(;revision=s.state.revision)) || push!(issues,"Path selection and snapshot disagree.")
        s.closed && (s.object!==nothing || !isempty(s.listeners)) && push!(issues,"Closed path retains input/listeners.")
        valid=isempty(issues);throw && !valid && _throw_invalid_visual(:check_inspection_session,issues)
        (;kind=:inspection_session,valid,issues,closed=s.closed,revision=s.state.revision)
    end
end
function _visual_spec(s::PathInspectionSession,::Symbol;kwargs...)
    isempty(kwargs) || throw(ArgumentError("Change path selection with select_inspection!."))
    VisualizationSpec(:linked_inspector;title="Persistence along a path",panels=inspection_snapshot(s).panels,
        interaction=(;hover=true,clicks=true,labels=true,widgets=(:path_frame,:path_track,:path_playback),
            notebook=:linked_inspector,mode=:live_julia,requires_live_julia=true,offline_widgets=false),
        metadata=(;session=s,inspector_kind=:path))
end
_visual_request_cost(::PathInspectionSession,::Symbol)=(;work=:retained_history,homology=:already_computed,requires_live_julia=true,timing=:not_measured)
