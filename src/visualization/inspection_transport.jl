"""
    TransportInspectionSession

Linked finite refinement/change-of-poset inspection. Construct through
`inspection_session`, query with `inspection_summary`/`inspection_snapshot`,
select with `select_inspection!`, and release inputs with `close_inspection!`.
"""
mutable struct TransportInspectionSession <: _AbstractInspectionSession
    object::Any
    prepared::Any
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

"""
    inspection_session(result; sources=nothing, joint=nothing, comparison=nothing, kwargs...)

Explore a finite order map, joint encoding, common refinement or module
translation. Preparation validates actual modules and cover equations and may
materialize supplied lazy sources. Selection reuses that preparation. Inputs
must remain read-only. Closing releases them and preserves the last snapshot.
See `visual_spec` and the change-of-poset reference for selection and budgets.
"""
function inspection_session(x::_TransportObject;kwargs...)
    o=_transport_options(x;kwargs...)
    c=_prepare_transport(x;kwargs...)
    initial=(;vertex=o.vertex,pair=o.pair,point=o.point,side=o.side,label=o.label,view=o.view)
    options=(;matrix_limit=o.matrix_limit,max_vertices=o.max_vertices,max_pairs=o.max_pairs)
    snapshot=_transport_spec(c,_transport_selection(c;initial...);view=o.view,matrix_limit=o.matrix_limit)
    state=merge(initial,(;revision=0))
    TransportInspectionSession(x,c,initial,options,state,snapshot,1,Dict{Int,Any}(),1,String[],false,false,ReentrantLock())
end
inspection_selection(s::TransportInspectionSession)=lock(s.lock) do;s.state;end
inspection_snapshot(s::TransportInspectionSession)=lock(s.lock) do;s.snapshot;end
function inspection_summary(s::TransportInspectionSession)
    lock(s.lock) do
        m=s.snapshot.metadata
        (;kind=:transport_inspection_session,selection=s.state,revision=s.state.revision,
          comparison_validation=m.comparison_validation,selection_status=m.selection.status,
          candidates=copy(m.selection.candidates),category=m.category,
          closed=s.closed,updating=s.busy,snapshot_builds=s.snapshot_builds,
          listener_count=length(s.listeners),listener_errors=Tuple(s.listener_errors))
    end
end
describe(s::TransportInspectionSession)=inspection_summary(s)
Base.show(io::IO,s::TransportInspectionSession)=print(io,"TransportInspectionSession(view=",s.state.view,", closed=",s.closed,")")
function _transport_candidate(s;kwargs...)
    changing=any(k->haskey(kwargs,k),(:vertex,:pair,:point,:label))
    prior=changing ? (;vertex=nothing,pair=nothing,point=nothing,side=:base,label=nothing,view=s.state.view) :
        (;vertex=s.state.vertex,pair=s.state.pair,point=s.state.point,side=s.state.side,label=s.state.label,view=s.state.view)
    values=merge(prior,(;kwargs...))
    vertex,pair,point,side,label,view=values.vertex,values.pair,values.point,values.side,values.label,values.view
    all(k->k in keys(prior),keys(kwargs)) || throw(ArgumentError("Unknown transport selection keyword."))
    _inspection_require_open(s)
    o=_transport_options(s.object;vertex,pair,point,side,label,view,s.options...)
    _transport_selection(s.prepared;o...)
    return (;vertex=o.vertex,pair=o.pair,point=o.point,side=o.side,label=o.label,view=o.view,revision=s.state.revision+1)
end
function check_inspection_selection(s::TransportInspectionSession;throw=false,kwargs...)
    return lock(s.lock) do
        issues=String[];candidate=nothing
        try;candidate=_transport_candidate(s;kwargs...)
        catch err;err isa InterruptException && rethrow();push!(issues,sprint(showerror,err));end
        valid=isempty(issues)
        throw && !valid && _throw_invalid_visual(:check_inspection_selection,issues)
        (;kind=:inspection_selection,valid,issues,selection=candidate)
    end
end
function select_inspection!(s::TransportInspectionSession;kwargs...)
    lock(s.lock)
    try
        _inspection_require_open(s)
        s.busy && throw(ArgumentError("An inspection update or callback is already in progress."))
        c=_transport_candidate(s;kwargs...)
        s.busy=true
        try
            snapshot=_transport_spec(s.prepared,_transport_selection(s.prepared;c...);view=c.view,matrix_limit=s.options.matrix_limit)
            s.snapshot=snapshot;s.state=c;s.snapshot_builds+=1
        catch;s.busy=false;rethrow();end
    finally;unlock(s.lock);end
    try;_inspection_notify(s)
    finally;lock(s.lock) do;s.busy=false;end;end
    return s
end
reset_inspection!(s::TransportInspectionSession)=select_inspection!(s;s.initial...)
function close_inspection!(s::TransportInspectionSession)
    lock(s.lock)
    try
        s.closed && return s
        s.busy && throw(ArgumentError("Cannot close during an inspection update or callback."))
        s.closed=true;s.busy=true
    finally;unlock(s.lock);end
    try;_inspection_notify(s)
    finally
        lock(s.lock) do
            empty!(s.listeners);s.object=nothing;s.prepared=nothing;s.busy=false
        end
    end
    return s
end
function check_inspection_session(s::TransportInspectionSession;throw=false)
    return lock(s.lock) do
        issues=String[]
        s.snapshot.metadata.view==s.state.view && all(k->getproperty(s.snapshot.metadata.selection,k)==getproperty(s.state,k),(:pair,:point,:side,:label)) ||
            push!(issues,"Snapshot and selected view/pair disagree.")
        if s.closed && !s.busy
            isempty(s.listeners) && s.object === nothing && s.prepared === nothing || push!(issues,"Closed transport session retains callbacks or input.")
        end
        append!(issues,check_visual_spec(s.snapshot).issues)
        valid=isempty(issues)
        throw && !valid && _throw_invalid_visual(:check_inspection_session,issues)
        (;kind=:inspection_session,valid,issues,closed=s.closed,revision=s.state.revision)
    end
end

function _visual_spec(s::TransportInspectionSession,::Symbol;kwargs...)
    return VisualizationSpec(:linked_inspector;title="Explore refinement and transport",panels=inspection_snapshot(s).panels,
        interaction=(;hover=true,clicks=true,labels=true,widgets=(:transport_selection,:transport_view),
            notebook=:linked_inspector,mode=:live_julia,requires_live_julia=true,offline_widgets=false),
        metadata=(;session=s,inspector_kind=:transport))
end
_visual_request_cost(::TransportInspectionSession,::Symbol)=
    (;work=:selected_transport_maps,hover=:retained_labels,requires_live_julia=true,timing=:not_measured)
