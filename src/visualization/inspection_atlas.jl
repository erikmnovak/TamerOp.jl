# Selection uses retained algebra; browser callbacks never recompute homology.
"""Linked selection of atlas states, entry events and retained class witnesses.
Construct with `inspection_session(atlas)`; use the common inspection accessors,
selection, reset, close and snapshot/export operations.
"""
mutable struct AtlasInspectionSession <: _AbstractInspectionSession
    object::Union{Nothing,ParameterAtlas}
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
function inspection_session(a::ParameterAtlas;source=1,target=length(a.states)>1 ? 2 : nothing,event=:summary,witness=1)
    initial=_atlas_selection(a;source,target,event,witness)
    AtlasInspectionSession(a,initial,merge(initial,(;revision=0)),visual_spec(a;initial...),1,
        Dict{Int,Any}(),1,String[],false,false,ReentrantLock(),_InspectionRequests())
end
inspection_selection(s::AtlasInspectionSession)=lock(()->s.state,s.lock)
inspection_snapshot(s::AtlasInspectionSession)=lock(()->s.snapshot,s.lock)
function inspection_summary(s::AtlasInspectionSession)
    lock(s.lock) do
        (;kind=:atlas_inspection_session,selection=s.state,revision=s.state.revision,closed=s.closed,updating=s.busy,
          snapshot_builds=s.snapshot_builds,listener_count=length(s.listeners),listener_errors=Tuple(s.listener_errors),
          requests=_inspection_request_summary(s),transition=s.snapshot.metadata.transition)
    end
end
describe(s::AtlasInspectionSession)=inspection_summary(s)
Base.show(io::IO,s::AtlasInspectionSession)=print(io,"AtlasInspectionSession(",s.state.source," \u2192 ",s.state.target,", closed=",s.closed,")")
function _atlas_candidate(s;source=s.state.source,target=s.state.target,event=nothing,witness=nothing)
    _inspection_require_open(s)
    changed=source!=s.state.source || target!=s.state.target
    event=event===nothing ? (changed ? :summary : s.state.event) : event
    witness=witness===nothing ? (changed || event!=s.state.event ? 1 : s.state.witness) : witness
    merge(_atlas_selection(s.object;source,target,event,witness),(;revision=s.state.revision+1))
end
function check_inspection_selection(s::AtlasInspectionSession;throw=false,kwargs...)
    lock(s.lock) do
        issues=String[];candidate=nothing
        try;candidate=_atlas_candidate(s;kwargs...)
        catch err;err isa InterruptException && rethrow();push!(issues,sprint(showerror,err));end
        valid=isempty(issues);throw && !valid && _throw_invalid_visual(:check_inspection_selection,issues)
        (;kind=:inspection_selection,valid,issues,selection=candidate)
    end
end
function select_inspection!(s::AtlasInspectionSession;kwargs...)
    _cancel_inspection_requests!(s)
    lock(s.lock) do
        _inspection_require_open(s)
        s.busy && throw(ArgumentError("An atlas update is already in progress."))
        state=_atlas_candidate(s;kwargs...)
        snapshot=visual_spec(s.object;Base.structdiff(state,(;revision=state.revision))...)
        s.state=state;s.snapshot=snapshot;s.snapshot_builds+=1;s.busy=true
    end
    try;_inspection_notify(s)
    finally;lock(s.lock) do;s.busy=false;end;end
    s
end
reset_inspection!(s::AtlasInspectionSession)=select_inspection!(s;s.initial...)
function close_inspection!(s::AtlasInspectionSession)
    _cancel_inspection_requests!(s)
    lock(s.lock) do
        s.closed && return s
        s.busy && throw(ArgumentError("Cannot close during an atlas update."))
        s.closed=true
    end
    try;_inspection_notify(s)
    finally
        lock(s.lock) do;empty!(s.listeners);s.object=nothing;end
    end
    s
end
function check_inspection_session(s::AtlasInspectionSession;throw=false)
    lock(s.lock) do
        issues=copy(check_visual_spec(s.snapshot).issues)
        s.snapshot.metadata.selection==Base.structdiff(s.state,(;revision=s.state.revision)) || push!(issues,"Atlas selection and snapshot disagree.")
        s.closed && (s.object!==nothing || !isempty(s.listeners)) && push!(issues,"Closed atlas retains its input/listeners.")
        valid=isempty(issues);throw && !valid && _throw_invalid_visual(:check_inspection_session,issues)
        (;kind=:inspection_session,valid,issues,closed=s.closed,revision=s.state.revision)
    end
end
function _visual_spec(s::AtlasInspectionSession,::Symbol;kwargs...)
    isempty(kwargs) || throw(ArgumentError("Change atlas selection with select_inspection!."))
    VisualizationSpec(:linked_inspector;title="Explore parameter states",panels=inspection_snapshot(s).panels,
        interaction=(;hover=true,clicks=true,labels=true,widgets=(:atlas_states,:atlas_event,:atlas_witness),
            notebook=:linked_inspector,mode=:live_julia,requires_live_julia=true,offline_widgets=false),
        metadata=(;session=s,inspector_kind=:atlas))
end
_visual_request_cost(::AtlasInspectionSession,::Symbol)=(;work=:retained_states_and_maps,homology=:already_computed,requires_live_julia=true,timing=:not_measured)
