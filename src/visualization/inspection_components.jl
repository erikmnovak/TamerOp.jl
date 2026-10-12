"""
    ComponentInspectionSession

Linked H0 interval, merge-forest and original source membership views, created by
`inspection_session(component_history(source); ...)`. Updates select `scale`,
`component` (a birth-vertex index), or an `event` (a merge-node index). Selecting
an event moves to its exact scale; editing scale clears the selected event.
Hover reads prepared metadata. Closing releases inputs and listeners while
retaining the final static snapshot. Inputs must remain read-only while in use.
"""
mutable struct ComponentInspectionSession <: _AbstractInspectionSession
    object::Union{Nothing,OrdinaryPersistence.ComponentHistory}
    initial::NamedTuple
    options::NamedTuple
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
"""`inspection_session(history::ComponentHistory; scale, component=nothing, event=nothing, ...)` links its bars, source memberships and merge events. See `ComponentInspectionSession` for selection and lifecycle semantics."""
function inspection_session(h::OrdinaryPersistence.ComponentHistory;kwargs...)
    o=_component_view_options(h;kwargs...)
    initial=(;o.scale,o.component,o.event)
    options=Base.structdiff(o,initial)
    snapshot=visual_spec(h;o...)
    ComponentInspectionSession(h,initial,options,merge(initial,(;revision=0)),snapshot,1,Dict{Int,Any}(),1,String[],false,false,ReentrantLock())
end
inspection_selection(s::ComponentInspectionSession)=lock(s.lock) do;s.state;end
inspection_snapshot(s::ComponentInspectionSession)=lock(s.lock) do;s.snapshot;end
function inspection_summary(s::ComponentInspectionSession)
    lock(s.lock) do
        (;kind=:component_inspection_session,selection=s.state,revision=s.state.revision,
            components=length(s.snapshot.metadata.partition),closed=s.closed,updating=s.busy,
            snapshot_builds=s.snapshot_builds,listener_count=length(s.listeners),listener_errors=Tuple(s.listener_errors))
    end
end
describe(s::ComponentInspectionSession)=inspection_summary(s)
Base.show(io::IO,s::ComponentInspectionSession)=print(io,"ComponentInspectionSession(scale=",s.state.scale,", closed=",s.closed,")")
function _component_candidate(s;kwargs...)
    _inspection_require_open(s)
    all(k->k in (:scale,:component,:event),keys(kwargs)) || throw(ArgumentError("Only scale, component and event are live selections."))
    state=Base.structdiff(s.state,(;revision=s.state.revision))
    (haskey(kwargs,:scale) || haskey(kwargs,:component)) && !haskey(kwargs,:event) && (state=merge(state,(;event=nothing)))
    haskey(kwargs,:event) && kwargs[:event]!==nothing && !haskey(kwargs,:component) && (state=merge(state,(;component=nothing)))
    o=_component_view_options(s.object;merge(state,(;kwargs...))...,s.options...)
    return (;o.scale,o.component,o.event,revision=s.state.revision+1)
end
function check_inspection_selection(s::ComponentInspectionSession;throw=false,kwargs...)
    lock(s.lock) do
        issues=String[];candidate=nothing
        try;candidate=_component_candidate(s;kwargs...)
        catch err;err isa InterruptException && rethrow();push!(issues,sprint(showerror,err));end
        valid=isempty(issues)
        throw && !valid && _throw_invalid_visual(:check_inspection_selection,issues)
        (;kind=:inspection_selection,valid,issues,selection=candidate)
    end
end
function select_inspection!(s::ComponentInspectionSession;kwargs...)
    lock(s.lock)
    try
        _inspection_require_open(s)
        s.busy && throw(ArgumentError("An inspection update or callback is already in progress."))
        c=_component_candidate(s;kwargs...);s.busy=true
        try
            snapshot=visual_spec(s.object;Base.structdiff(c,(;revision=c.revision))...,s.options...)
            s.snapshot=snapshot;s.state=c;s.snapshot_builds+=1
        catch;s.busy=false;rethrow();end
    finally;unlock(s.lock);end
    try;_inspection_notify(s)
    finally;lock(s.lock) do;s.busy=false;end;end
    s
end
reset_inspection!(s::ComponentInspectionSession)=select_inspection!(s;s.initial...)
function close_inspection!(s::ComponentInspectionSession)
    lock(s.lock)
    try
        s.closed && return s
        s.busy && throw(ArgumentError("Cannot close during an inspection update or callback."))
        s.closed=true;s.busy=true
    finally;unlock(s.lock);end
    try;_inspection_notify(s)
    finally
        lock(s.lock) do;empty!(s.listeners);s.object=nothing;s.options=NamedTuple();s.busy=false;end
    end
    s
end
function check_inspection_session(s::ComponentInspectionSession;throw=false)
    lock(s.lock) do
        issues=String[]
        all(k->getproperty(s.snapshot.metadata,k)==getproperty(s.state,k),(:scale,:component,:event)) || push!(issues,"Snapshot and selection disagree.")
        s.closed && !s.busy && !(s.object===nothing && isempty(s.listeners)) && push!(issues,"Closed session retains inputs/listeners.")
        append!(issues,check_visual_spec(s.snapshot).issues)
        valid=isempty(issues)
        throw && !valid && _throw_invalid_visual(:check_inspection_session,issues)
        (;kind=:inspection_session,valid,issues,closed=s.closed,revision=s.state.revision)
    end
end
function _visual_spec(s::ComponentInspectionSession,::Symbol;kwargs...)
    isempty(kwargs) || throw(ArgumentError("Change a component session with select_inspection!."))
    VisualizationSpec(:linked_inspector;title="Components and merge history",panels=inspection_snapshot(s).panels,
        interaction=(;hover=true,clicks=true,labels=true,widgets=(:component_scale,:component_selection,:component_event),
            notebook=:linked_inspector,mode=:live_julia,requires_live_julia=true,offline_widgets=false),
        metadata=(;session=s,inspector_kind=:components))
end
_visual_request_cost(::ComponentInspectionSession,::Symbol)=
    (;work=:retained_merge_forest_cut,hover=:read_only,requires_live_julia=true,timing=:not_measured)
