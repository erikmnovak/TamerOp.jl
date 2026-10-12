# Transactional table selection and independent pair drill-down.
"""
    EnsembleInspectionSession

Linked prepared summary, probe and replicate/source selection from
`inspection_session(summary; samples=nothing)`. Use `select_inspection!` with
`sample` (ID or index), `probe` (Cartesian linear index) or `view`. Inference is
prepared once by `summarize_ensemble`; changing a selection never resamples.
Optional samples follow replicate order and retain static `source`/`invariant`
specifications. Their relationship to supplied values is the caller's assertion.
Invalid requests roll back. Closing releases the summary, sample references and
listeners while retaining the final static snapshot. Live controls need Julia.
"""
mutable struct EnsembleInspectionSession <: _AbstractInspectionSession
    object::Union{Nothing,_EnsembleSummary}
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
function inspection_session(c::_EnsembleSummary;kwargs...)
    o=_ensemble_options(c;kwargs...)
    EnsembleInspectionSession(c,o,merge(o,(;revision=0)),_ensemble_spec(c,o),1,Dict{Int,Any}(),1,String[],false,false,ReentrantLock())
end
inspection_selection(s::EnsembleInspectionSession)=lock(s.lock) do;s.state;end
inspection_snapshot(s::EnsembleInspectionSession)=lock(s.lock) do;s.snapshot;end
function inspection_summary(s::EnsembleInspectionSession)
    lock(s.lock) do
        (;kind=:ensemble_inspection_session,selection=s.state,revision=s.state.revision,
            snapshot_builds=s.snapshot_builds,resampling=false,closed=s.closed,updating=s.busy,
            listener_count=length(s.listeners),listener_errors=Tuple(s.listener_errors))
    end
end
describe(s::EnsembleInspectionSession)=inspection_summary(s)
Base.show(io::IO,s::EnsembleInspectionSession)=print(io,"EnsembleInspectionSession(view=",s.state.view,", closed=",s.closed,")")
function _ensemble_candidate(s;kwargs...)
    _inspection_require_open(s)
    all(k->k in (:view,:sample,:probe),keys(kwargs)) || throw(ArgumentError("Only view, sample and probe are live selections."))
    state=merge(Base.structdiff(s.state,(;revision=s.state.revision)),(;kwargs...))
    _ensemble_options(s.object;state...)
end
function check_inspection_selection(s::EnsembleInspectionSession;throw=false,kwargs...)
    lock(s.lock) do
        issues=String[];candidate=nothing
        try;candidate=_ensemble_candidate(s;kwargs...)
        catch err;err isa InterruptException && rethrow();push!(issues,sprint(showerror,err));end
        valid=isempty(issues)
        throw && !valid && _throw_invalid_visual(:check_inspection_selection,issues)
        (;kind=:inspection_selection,valid,issues,selection=candidate)
    end
end
function select_inspection!(s::EnsembleInspectionSession;kwargs...)
    lock(s.lock)
    try
        _inspection_require_open(s)
        s.busy && throw(ArgumentError("An inspection update or callback is already in progress."))
        o=_ensemble_candidate(s;kwargs...);s.busy=true
        try
            snapshot=_ensemble_spec(s.object,o)
            s.snapshot=snapshot;s.state=merge(o,(;revision=s.state.revision+1));s.snapshot_builds+=1
        catch;s.busy=false;rethrow();end
    finally;unlock(s.lock);end
    try;_inspection_notify(s)
    finally;lock(s.lock) do;s.busy=false;end;end
    s
end
reset_inspection!(s::EnsembleInspectionSession)=select_inspection!(s;sample=s.initial.sample,probe=s.initial.probe,view=s.initial.view)
function close_inspection!(s::EnsembleInspectionSession)
    lock(s.lock)
    try
        s.closed && return s
        s.busy && throw(ArgumentError("Cannot close during an inspection update or callback."))
        s.closed=true;s.busy=true
    finally;unlock(s.lock);end
    try;_inspection_notify(s)
    finally;lock(s.lock) do;empty!(s.listeners);s.object=nothing;s.initial=merge(s.initial,(;samples=nothing));s.state=merge(s.state,(;samples=nothing));s.busy=false;end;end
    s
end
function check_inspection_session(s::EnsembleInspectionSession;throw=false)
    lock(s.lock) do
        issues=String[]
        s.snapshot.metadata.selection==Base.structdiff(s.state,(;revision=s.state.revision,samples=s.state.samples)) || push!(issues,"Snapshot and selection disagree.")
        s.closed && !s.busy && !(s.object===nothing && isempty(s.listeners)) && push!(issues,"Closed session retains inputs/listeners.")
        append!(issues,check_visual_spec(s.snapshot).issues)
        valid=isempty(issues)
        throw && !valid && _throw_invalid_visual(:check_inspection_session,issues)
        (;kind=:inspection_session,valid,issues,closed=s.closed,revision=s.state.revision)
    end
end
function _visual_spec(s::EnsembleInspectionSession,::Symbol;kwargs...)
    isempty(kwargs) || throw(ArgumentError("Change an ensemble session with select_inspection!."))
    VisualizationSpec(:linked_inspector;title="Explore an ensemble",panels=inspection_snapshot(s).panels,
        interaction=(;hover=true,clicks=true,labels=true,widgets=(:ensemble_view,:ensemble_sample,:ensemble_probe),
            notebook=:linked_inspector,mode=:live_julia,requires_live_julia=true,offline_widgets=false),
        metadata=(;session=s,inspector_kind=:ensembles))
end
_visual_request_cost(::EnsembleInspectionSession,::Symbol)=(;work=:prepared_summary,resampling=false,hover=:read_only,requires_live_julia=true,timing=:not_measured)
