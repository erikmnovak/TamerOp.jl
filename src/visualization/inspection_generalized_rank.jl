# Bounded live region queries; display selections reuse exact retained results.
"""
    GeneralizedRankInspectionSession

Linked region/rank and boundary/GRIL views, made by `generalized_rank_session`.
Inputs are treated as read-only. Invalid updates preserve the previous state.
Hover and occurrence/bar/level selection reuse retained answers; changed domains
compute bounded exact queries. Closing releases the input and listeners while
retaining a static `inspection_snapshot`.
"""
mutable struct GeneralizedRankInspectionSession <: _AbstractInspectionSession
    object::Any
    initial::NamedTuple
    state::NamedTuple
    data::NamedTuple
    snapshot::VisualizationSpec
    rank_queries::Int
    landscape_queries::Int
    boundary_queries::Int
    snapshot_builds::Int
    listeners::Dict{Int,Any}
    next_listener::Int
    listener_errors::Vector{String}
    closed::Bool
    busy::Bool
    lock::ReentrantLock
end

"""
    generalized_rank_session(M; vertices, witnesses=false, boundary=false, kwargs...)
    generalized_rank_session(enc; center, width=0, length=1,
        levels=(1,2,3), lengths=(1,2,3), level=1, kwargs...)

Explore a connected convex finite-label region, or a continuous closed worm on
a positive 2D grid encoding. Uses the same queries as
`visual_spec(obj; kind=:generalized_rank, ...)`. `boundary=true` explicitly
computes a grid-index boundary barcode; `witnesses=true` retains finite-region
limit/colimit matrices. Neither is silently enabled for scalar queries.

Change finite `vertices`, `witnesses`, `boundary`, `occurrence`, or `bar` with
`select_inspection!`. Worm sessions accept `center`, `width`, `length` and `level`;
the latter two must belong to the fixed `lengths` and `levels`. Changing width
reuses the landscape; changing a layer only redraws retained results. Exact
typed entry and approximate pointer selection have the same explicit domain
contract. Budgets are `budget=Advanced.GeneralizedRankBudget()` and
`max_vertices=120` for the drawing. Live controls require WGLMakie and Julia;
use the static snapshot for Cairo or saved HTML.
"""
function generalized_rank_session(obj::_RegionRankInput;kwargs...)
    o=_region_options(obj;kwargs...);data=_region_compute(obj,o);snapshot=_region_snapshot(obj,o,data)
    GeneralizedRankInspectionSession(obj,deepcopy(o),merge(o,(;revision=0)),data,snapshot,
        Int(data.rank_computed),Int(data.landscape_computed),Int(data.boundary_computed),1,
        Dict{Int,Any}(),1,String[],false,false,ReentrantLock())
end
inspection_selection(s::GeneralizedRankInspectionSession)=lock(s.lock) do;deepcopy(s.state);end
inspection_snapshot(s::GeneralizedRankInspectionSession)=lock(s.lock) do;s.snapshot;end
function inspection_summary(s::GeneralizedRankInspectionSession)
    lock(s.lock) do
        (;kind=:generalized_rank_inspection_session,selection=deepcopy(s.state),revision=s.state.revision,
            rank=s.snapshot.metadata.rank,rank_queries=s.rank_queries,landscape_queries=s.landscape_queries,
            boundary_queries=s.boundary_queries,snapshot_builds=s.snapshot_builds,closed=s.closed,updating=s.busy,
            listener_count=length(s.listeners),listener_errors=Tuple(s.listener_errors))
    end
end
describe(s::GeneralizedRankInspectionSession)=inspection_summary(s)
Base.show(io::IO,s::GeneralizedRankInspectionSession)=print(io,"GeneralizedRankInspectionSession(mode=",s.state.mode,
    ", rank=",s.snapshot.metadata.rank,", closed=",s.closed,")")
function _region_candidate(s;kwargs...)
    _inspection_require_open(s)
    allowed=s.state.mode===:finite ? (:vertices,:witnesses,:boundary,:occurrence,:bar) : (:center,:width,:length,:level)
    all(k->k in allowed,keys(kwargs)) || throw(ArgumentError("This $(s.state.mode) session accepts selections $(allowed)."))
    prior=Base.structdiff(s.state,(;mode=s.state.mode,revision=s.state.revision))
    s.state.mode===:worm && (prior=merge(prior,(;vertices=nothing)))
    # Selecting a new region clears occurrence identities from the old path.
    changed=haskey(kwargs,:vertices) && kwargs[:vertices]!=s.state.vertices
    changed && (prior=merge(prior,(;occurrence=0,bar=0)))
    haskey(kwargs,:boundary) && kwargs[:boundary]===false && (prior=merge(prior,(;occurrence=0,bar=0)))
    _region_options(s.object;merge(prior,(;kwargs...))...)
end
function check_inspection_selection(s::GeneralizedRankInspectionSession;throw=false,kwargs...)
    lock(s.lock) do
        issues=String[];candidate=nothing
        try
            candidate=_region_candidate(s;kwargs...)
            if candidate.boundary && candidate.vertices==s.state.vertices && s.data.boundary!==nothing
                candidate.occurrence<=length(s.data.boundary.labels) || push!(issues,"occurrence exceeds the boundary path")
                candidate.bar<=length(s.data.boundary.bars) || push!(issues,"bar exceeds the boundary barcode")
            end
        catch err;err isa InterruptException && rethrow();push!(issues,sprint(showerror,err));end
        valid=isempty(issues)
        throw && !valid && _throw_invalid_visual(:check_inspection_selection,issues)
        (;kind=:inspection_selection,valid,issues,selection=candidate)
    end
end
function select_inspection!(s::GeneralizedRankInspectionSession;reset=false,kwargs...)
    lock(s.lock)
    try
        _inspection_require_open(s)
        s.busy && throw(ArgumentError("An inspection update or callback is already in progress."))
        reset isa Bool || throw(ArgumentError("reset must be Bool."))
        reset && !isempty(kwargs) && throw(ArgumentError("Reset must be a separate action."))
        o=reset ? deepcopy(s.initial) : _region_candidate(s;kwargs...)
        s.busy=true
        try
            data=_region_compute(s.object,o;previous=s.data,old=s.state)
            snapshot=_region_snapshot(s.object,o,data)
            s.state=merge(o,(;revision=s.state.revision+1));s.data=data;s.snapshot=snapshot
            s.rank_queries+=data.rank_computed;s.landscape_queries+=data.landscape_computed
            s.boundary_queries+=data.boundary_computed;s.snapshot_builds+=1
        catch;s.busy=false;rethrow();end
    finally;unlock(s.lock);end
    try;_inspection_notify(s)
    finally;lock(s.lock) do;s.busy=false;end;end
    s
end
reset_inspection!(s::GeneralizedRankInspectionSession)=select_inspection!(s;reset=true)
function close_inspection!(s::GeneralizedRankInspectionSession)
    lock(s.lock)
    try
        s.closed && return s
        s.busy && throw(ArgumentError("Cannot close during an inspection update or callback."))
        s.closed=true;s.busy=true
    finally;unlock(s.lock);end
    try;_inspection_notify(s)
    finally;lock(s.lock) do;empty!(s.listeners);s.object=nothing;s.data=(;);s.busy=false;end;end
    s
end
function check_inspection_session(s::GeneralizedRankInspectionSession;throw=false)
    lock(s.lock) do
        issues=String[]
        s.snapshot.metadata.selection==Base.structdiff(s.state,(;revision=s.state.revision)) || push!(issues,"Snapshot and selection disagree.")
        s.closed && !s.busy && !(s.object===nothing && isempty(s.listeners) && isempty(s.data)) && push!(issues,"Closed session retains inputs/listeners.")
        append!(issues,check_visual_spec(s.snapshot).issues)
        valid=isempty(issues)
        throw && !valid && _throw_invalid_visual(:check_inspection_session,issues)
        (;kind=:inspection_session,valid,issues,closed=s.closed,revision=s.state.revision)
    end
end
function _visual_spec(s::GeneralizedRankInspectionSession,::Symbol;kwargs...)
    isempty(kwargs) || throw(ArgumentError("Change a region session with select_inspection!."))
    VisualizationSpec(:linked_inspector;title="Explore generalized rank",panels=inspection_snapshot(s).panels,
        interaction=(;hover=true,clicks=true,labels=true,widgets=(:rank_region,:rank_witness,:gril_layer),
            notebook=:linked_inspector,mode=:live_julia,requires_live_julia=true,offline_widgets=false),
        metadata=(;session=s,inspector_kind=:generalized_rank))
end
_visual_request_cost(::GeneralizedRankInspectionSession,::Symbol)=(;work=:selected_exact_region,
    hover=:read_only,witnesses=:explicit_opt_in,requires_live_julia=true,timing=:not_measured)
