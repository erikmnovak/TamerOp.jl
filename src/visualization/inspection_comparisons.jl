"""
    ComparisonInspectionSession

Synchronized comparison views, created with `inspection_session(comparison)`.
Selections are `view`, aligned `source`/`probe` indices, named `baseline`, and
an opt-in selected `matrix`. One rank row per case is retained; moving the target
or baseline reuses it. Invalid updates leave selection and cache unchanged.
Closing releases cases and row caches and retains the last static snapshot.
"""
mutable struct ComparisonInspectionSession <: _AbstractInspectionSession
    object::Union{Nothing,VisualComparison}
    initial::NamedTuple
    options::NamedTuple
    state::NamedTuple
    snapshot::VisualizationSpec
    row_key::Tuple{Symbol,Int}
    rows::Any
    row_builds::Int
    snapshot_builds::Int
    listeners::Dict{Int,Any}
    next_listener::Int
    listener_errors::Vector{String}
    closed::Bool
    busy::Bool
    lock::ReentrantLock
end
"""`inspection_session(comparison::VisualComparison; view=:hilbert, source=1, probe=1, ...)` synchronizes corresponding queries and retains one rank row. See `ComparisonInspectionSession` for selection and lifecycle semantics."""
function inspection_session(h::VisualComparison;kwargs...)
    o=_comparison_options(h;kwargs...)
    initial=(;o.view,o.source,o.probe,o.baseline,o.matrix)
    options=Base.structdiff(o,initial)
    data=_comparison_values(h,o)
    snapshot=_comparison_spec(h,o,data)
    ComparisonInspectionSession(h,initial,options,merge(initial,(;revision=0)),snapshot,(o.view===:rank ? :rank : :hilbert,o.view===:rank ? o.source : 0),data,1,1,Dict{Int,Any}(),1,String[],false,false,ReentrantLock())
end
inspection_selection(s::ComparisonInspectionSession)=lock(s.lock) do;s.state;end
inspection_snapshot(s::ComparisonInspectionSession)=lock(s.lock) do;s.snapshot;end
function inspection_summary(s::ComparisonInspectionSession)
    lock(s.lock) do
        (;kind=:comparison_inspection_session,selection=s.state,revision=s.state.revision,
            row_builds=s.row_builds,closed=s.closed,updating=s.busy,
            snapshot_builds=s.snapshot_builds,listener_count=length(s.listeners),listener_errors=Tuple(s.listener_errors))
    end
end
describe(s::ComparisonInspectionSession)=inspection_summary(s)
Base.show(io::IO,s::ComparisonInspectionSession)=print(io,"ComparisonInspectionSession(view=",s.state.view,", closed=",s.closed,")")
function _comparison_candidate(s;kwargs...)
    _inspection_require_open(s)
    all(k->k in (:view,:source,:probe,:baseline,:matrix),keys(kwargs)) || throw(ArgumentError("Only view, source, probe, baseline and matrix are live selections."))
    state=Base.structdiff(s.state,(;revision=s.state.revision))
    state=merge(state,(;kwargs...))
    # Leaving rank mode clears its opt-in matrix readout.
    state.view===:rank || (state=merge(state,(;matrix=false)))
    o=_comparison_options(s.object;state...,s.options...)
    return (;o.view,o.source,o.probe,o.baseline,o.matrix,revision=s.state.revision+1)
end
function check_inspection_selection(s::ComparisonInspectionSession;throw=false,kwargs...)
    lock(s.lock) do
        issues=String[];candidate=nothing
        try;candidate=_comparison_candidate(s;kwargs...)
        catch err;err isa InterruptException && rethrow();push!(issues,sprint(showerror,err));end
        valid=isempty(issues)
        throw && !valid && _throw_invalid_visual(:check_inspection_selection,issues)
        (;kind=:inspection_selection,valid,issues,selection=candidate)
    end
end
function select_inspection!(s::ComparisonInspectionSession;kwargs...)
    lock(s.lock)
    try
        _inspection_require_open(s)
        s.busy && throw(ArgumentError("An inspection update or callback is already in progress."))
        c=_comparison_candidate(s;kwargs...);s.busy=true
        try
            o=_comparison_options(s.object;Base.structdiff(c,(;revision=c.revision))...,s.options...)
            key=(c.view===:rank ? :rank : :hilbert,c.view===:rank ? c.source : 0)
            data=key==s.row_key ? s.rows : _comparison_values(s.object,o)
            snapshot=_comparison_spec(s.object,o,data)
            s.row_builds+=key!=s.row_key
            s.rows=data;s.row_key=key
            s.snapshot=snapshot;s.state=c;s.snapshot_builds+=1
        catch;s.busy=false;rethrow();end
    finally;unlock(s.lock);end
    try;_inspection_notify(s)
    finally;lock(s.lock) do;s.busy=false;end;end
    s
end
reset_inspection!(s::ComparisonInspectionSession)=select_inspection!(s;s.initial...)
function close_inspection!(s::ComparisonInspectionSession)
    lock(s.lock)
    try
        s.closed && return s
        s.busy && throw(ArgumentError("Cannot close during an inspection update or callback."))
        s.closed=true;s.busy=true
    finally;unlock(s.lock);end
    try;_inspection_notify(s)
    finally
        lock(s.lock) do;empty!(s.listeners);s.object=nothing;s.rows=nothing;s.options=NamedTuple();s.busy=false;end
    end
    s
end
function check_inspection_session(s::ComparisonInspectionSession;throw=false)
    lock(s.lock) do
        issues=String[]
        all(k->getproperty(s.snapshot.metadata.selection,k)==getproperty(s.state,k),(:view,:source,:probe,:baseline,:matrix)) || push!(issues,"Snapshot and selection disagree.")
        s.closed && !s.busy && !(s.object===nothing && s.rows===nothing && isempty(s.listeners)) && push!(issues,"Closed session retains inputs/listeners.")
        append!(issues,check_visual_spec(s.snapshot).issues)
        valid=isempty(issues)
        throw && !valid && _throw_invalid_visual(:check_inspection_session,issues)
        (;kind=:inspection_session,valid,issues,closed=s.closed,revision=s.state.revision)
    end
end
function _visual_spec(s::ComparisonInspectionSession,::Symbol;kwargs...)
    isempty(kwargs) || throw(ArgumentError("Change a comparison session with select_inspection!."))
    VisualizationSpec(:linked_inspector;title="Compare results",panels=inspection_snapshot(s).panels,
        interaction=(;hover=true,clicks=true,labels=true,widgets=(:comparison_view,:comparison_probe,:comparison_baseline),
            notebook=:linked_inspector,mode=:live_julia,requires_live_julia=true,offline_widgets=false),
        metadata=(;session=s,inspector_kind=:comparisons))
end
_visual_request_cost(::ComparisonInspectionSession,::Symbol)=
    (;work=:one_retained_rank_row,hover=:read_only,requires_live_julia=true,timing=:not_measured)
