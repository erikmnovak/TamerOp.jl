"""A linked source-selection and relative-persistence session.
Construct with `inspection_session(spatial_pair(source; ...))`. Selection uses
`cells`, `source_ids` or `brush`, with an explicit `rule`. `scale` and `degree`
choose the query; `details=:maps` or `:representative` opts into exact bases.
`entry` sets a common earliest A-entry (latest for superlevels), clamped to
K-entry. `grades` instead supplies individual selected-cell entry grades.
Closing releases the source and algebra, retaining the final static snapshot.
"""
mutable struct RelativeInspectionSession <: _AbstractInspectionSession
    object::Union{Nothing,OrdinaryPersistence.SpatialPair}
    diagrams::Any
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
    requests::_InspectionRequests
end
function inspection_session(s::OrdinaryPersistence.SpatialPair;kwargs...)
    o=_relative_options(s;kwargs...)
    s=deepcopy(s)
    initial=(;cells=copy(s.seeds),brush=s.brush,rule=s.rule,
        grades=Dict(r.index=>r.subcomplex_grade for r in OrdinaryPersistence.spatial_cells(s) if r.selected),
        scale=o.scale,degree=o.degree,details=o.details,witness=o.witness)
    options=(;view_dims=o.view_dims,max_cells=o.max_cells,max_algebra_cells=o.max_algebra_cells,max_matrix_entries=o.max_matrix_entries)
    diagrams=_relative_diagrams(s)
    snapshot=_relative_view(s;_diagrams=diagrams,o...)
    RelativeInspectionSession(s,diagrams,deepcopy(initial),options,merge(initial,(;revision=0)),snapshot,1,
        Dict{Int,Any}(),1,String[],false,false,ReentrantLock(),_InspectionRequests())
end
inspection_selection(s::RelativeInspectionSession)=lock(()->deepcopy(s.state),s.lock)
inspection_snapshot(s::RelativeInspectionSession)=lock(()->s.snapshot,s.lock)
function inspection_summary(s::RelativeInspectionSession)
    lock(s.lock) do
        (;kind=:relative_inspection_session,selection=deepcopy(s.state),revision=s.state.revision,closed=s.closed,updating=s.busy,
          snapshot_builds=s.snapshot_builds,listener_count=length(s.listeners),listener_errors=Tuple(s.listener_errors),
          requests=_inspection_request_summary(s),counts=s.snapshot.metadata.counts,selected=s.snapshot.metadata.selected,
          added=s.snapshot.metadata.added)
    end
end
describe(s::RelativeInspectionSession)=inspection_summary(s)
Base.show(io::IO,s::RelativeInspectionSession)=print(io,"RelativeInspectionSession(scale=",s.state.scale,", closed=",s.closed,")")
function _relative_candidate(s;cells=nothing,source_ids=nothing,brush=nothing,rule=s.state.rule,
        grades=nothing,entry=missing,scale=s.state.scale,degree=s.state.degree,details=s.state.details,witness=s.state.witness)
    _inspection_require_open(s)
    changed=cells!==nothing || source_ids!==nothing || brush!==nothing || rule!=s.state.rule
    chosen=cells===nothing ? (changed ? Int[] : s.state.cells) : cells
    # Reset carries both exact seeds and the brush that generated them.
    retained_brush=brush===nothing && !changed ? s.state.brush : brush
    source=s.object.source
    result=OrdinaryPersistence.spatial_pair(source;cells=brush===nothing ? chosen : Int[],source_ids,brush,
        rule,field=s.object.pair.field,order=s.object.pair.order)
    gs=grades===nothing ? (changed ? Dict{Int,Any}() : deepcopy(s.state.grades)) : grades
    if entry!==missing
        grades===nothing || throw(ArgumentError("Choose entry or individual grades."))
        if entry===nothing
            gs=Dict{Int,Any}()
        else
            entry isa Real && !(entry isa Bool) && isfinite(entry) || throw(ArgumentError("A entry must be a finite real coordinate."))
            clamp=s.object.pair.order===:sublevel ? max : min
            gs=Dict(i=>clamp(entry,result.pair.ambient.grades[i][1]) for i in result.selected)
        end
    end
    result=OrdinaryPersistence.spatial_pair(source;cells=result.seeds,rule,grades=gs,field=s.object.pair.field,order=s.object.pair.order)
    if retained_brush!==nothing
        OrdinaryPersistence._spatial_brush(source,retained_brush,rule)==result.seeds ||
            throw(ArgumentError("Brush no longer agrees with this selection rule; select a new brush or cell list."))
        result=OrdinaryPersistence.SpatialPair(result.source,result.pair,result.seeds,result.selected,result.added,rule,deepcopy(retained_brush))
    end
    o=_relative_options(result;scale,degree,details,witness,s.options...)
    state=(;cells=copy(result.seeds),brush=retained_brush,rule,
        grades=Dict(r.index=>r.subcomplex_grade for r in OrdinaryPersistence.spatial_cells(result) if r.selected),
        scale=o.scale,degree=o.degree,details=o.details,witness=o.witness,revision=s.state.revision+1)
    result,state,o
end
function check_inspection_selection(s::RelativeInspectionSession;throw=false,kwargs...)
    lock(s.lock) do
        issues=String[];state=nothing
        try
            result,state,o=_relative_candidate(s;kwargs...)
            # A representative index is a mathematical query, not a drawing index.
            o.details===:representative && _relative_view(result;o...)
        catch err;err isa InterruptException && rethrow();push!(issues,sprint(showerror,err));end
        valid=isempty(issues);throw && !valid && _throw_invalid_visual(:check_inspection_selection,issues)
        (;kind=:inspection_selection,valid,issues,selection=state)
    end
end
function select_inspection!(s::RelativeInspectionSession;kwargs...)
    _cancel_inspection_requests!(s)
    lock(s.lock) do
        _inspection_require_open(s)
        s.busy && throw(ArgumentError("A relative query is already in progress."))
        result,state,o=_relative_candidate(s;kwargs...)
        diagrams=OrdinaryPersistence._same_filtered_pair(result.pair,s.object.pair) ? s.diagrams : _relative_diagrams(result)
        snapshot=_relative_view(result;_diagrams=diagrams,o...)
        s.object=result;s.diagrams=diagrams;s.state=state;s.snapshot=snapshot;s.snapshot_builds+=1;s.busy=true
    end
    try;_inspection_notify(s)
    finally;lock(s.lock) do;s.busy=false;end;end
    s
end
reset_inspection!(s::RelativeInspectionSession)=select_inspection!(s;s.initial...)
function close_inspection!(s::RelativeInspectionSession)
    _cancel_inspection_requests!(s)
    lock(s.lock) do
        s.closed && return s
        s.busy && throw(ArgumentError("Cannot close during a relative query."))
        s.closed=true
    end
    try;_inspection_notify(s)
    finally
        lock(s.lock) do;empty!(s.listeners);s.object=nothing;s.diagrams=nothing;end
    end
    s
end
function check_inspection_session(s::RelativeInspectionSession;throw=false)
    lock(s.lock) do
        issues=copy(check_visual_spec(s.snapshot).issues)
        s.state.cells==s.snapshot.metadata.seeds && s.state.scale==s.snapshot.metadata.options.scale || push!(issues,"Selection and snapshot disagree.")
        s.closed && (s.object!==nothing || s.diagrams!==nothing || !isempty(s.listeners)) && push!(issues,"Closed session retains source or listeners.")
        valid=isempty(issues);throw && !valid && _throw_invalid_visual(:check_inspection_session,issues)
        (;kind=:inspection_session,valid,issues,closed=s.closed,revision=s.state.revision)
    end
end
function _visual_spec(s::RelativeInspectionSession,::Symbol;kwargs...)
    isempty(kwargs) || throw(ArgumentError("Change relative selections with select_inspection!."))
    VisualizationSpec(:linked_inspector;title="Explore a relative pair",panels=inspection_snapshot(s).panels,
        interaction=(;hover=true,clicks=true,labels=true,widgets=(:relative_selection,:relative_query,:relative_witness),
            notebook=:linked_inspector,mode=:live_julia,requires_live_julia=true,offline_widgets=false),
        metadata=(;session=s,inspector_kind=:relative))
end
_visual_request_cost(::RelativeInspectionSession,::Symbol)=(;work=:relative_pair_query,
    basis=:explicit_opt_in,requires_live_julia=true,timing=:not_measured)

# Relative selection replaces the mathematical pair as well as the picture.
function _inspection_adopt!(s::RelativeInspectionSession,t::RelativeInspectionSession)
    s.object=t.object;s.diagrams=t.diagrams
    s.state=t.state;s.snapshot=t.snapshot;s.snapshot_builds=t.snapshot_builds
    nothing
end
