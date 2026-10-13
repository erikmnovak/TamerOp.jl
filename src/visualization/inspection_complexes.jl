# Live filtered-complex selection reuses the exact static activity contract.
"""
    ComplexInspectionSession

Created by `inspection_session(filtered_complex; parameter=..., ...)`.
`select_inspection!` changes parameter, cell, dimensions, activity or boundary
visibility atomically. Hover reads retained cells only. Closing releases the
input and listeners but preserves the last static snapshot. The source complex
and geometry must remain read-only while linked viewers are using them.
"""
mutable struct ComplexInspectionSession <: _AbstractInspectionSession
    object::Union{Nothing,DataIngestion.FilteredComplex}
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

function inspection_session(x::DataIngestion.FilteredComplex;kind=:filtered_complex,kwargs...)
    kind===:chain_inspector && return _chain_inspection_session(x;kwargs...)
    kind in (:filtered_complex,:geometric_filtration) || throw(ArgumentError("Choose :filtered_complex, :geometric_filtration or :chain_inspector."))
    geometric=kind===:geometric_filtration ? _geometric_options(x;kwargs...) : nothing
    o=geometric===nothing ? _complex_options(x;kwargs...) : geometric.complex
    initial=(;parameter=o.parameter,cell=o.cell,dimensions=o.dimensions,activity=o.activity,boundary=o.boundary)
    options=(;view_dims=o.view_dims,max_cells=o.max_cells,labels=o.labels,max_labels=o.max_labels,alpha=o.alpha,field_axis=o.field_axis)
    if geometric!==nothing
        options=merge(options,(;source_dims=geometric.source_dims,source_anchor=geometric.source_anchor,
            sample_resolution=geometric.sample_resolution,degree=geometric.degree,field=geometric.field,
            invariant=geometric.invariant,max_algebra_cells=geometric.max_algebra_cells,
            max_matrix_entries=geometric.max_matrix_entries,compare=geometric.compare))
    end
    options=merge(options,(;kind))
    snapshot=visual_spec(x;initial...,options...)
    state=merge(initial,(;revision=0))
    ComplexInspectionSession(x,initial,options,state,snapshot,1,Dict{Int,Any}(),1,String[],false,false,ReentrantLock(),_InspectionRequests())
end
inspection_selection(s::ComplexInspectionSession)=lock(s.lock) do;s.state;end
inspection_snapshot(s::ComplexInspectionSession)=lock(s.lock) do;s.snapshot;end
function inspection_summary(s::ComplexInspectionSession)
    lock(s.lock) do
        m=s.snapshot.metadata
        (;kind=:complex_inspection_session,selection=s.state,revision=s.state.revision,
          requests=_inspection_request_summary(s),active_counts=m.active_counts,drawn_counts=m.drawn_counts,omitted_count=m.omitted_count,
          closed=s.closed,updating=s.busy,snapshot_builds=s.snapshot_builds,
          listener_count=length(s.listeners),listener_errors=Tuple(s.listener_errors))
    end
end
describe(s::ComplexInspectionSession)=inspection_summary(s)
Base.show(io::IO,s::ComplexInspectionSession)=print(io,"ComplexInspectionSession(parameter=",s.state.parameter,", cell=",s.state.cell,", closed=",s.closed,")")
function _complex_candidate(s;parameter=s.state.parameter,cell=s.state.cell,
        dimensions=s.state.dimensions,activity=s.state.activity,boundary=s.state.boundary)
    _inspection_require_open(s)
    options=Base.structdiff(s.options,(;kind=s.options.kind))
    o=if s.options.kind===:geometric_filtration
        _geometric_options(s.object;parameter,cell,dimensions,activity,boundary,options...).complex
    else
        _complex_options(s.object;parameter,cell,dimensions,activity,boundary,options...)
    end
    return (;parameter=o.parameter,cell=o.cell,dimensions=o.dimensions,activity=o.activity,boundary=o.boundary,revision=s.state.revision+1)
end
function check_inspection_selection(s::ComplexInspectionSession;throw=false,kwargs...)
    return lock(s.lock) do
        issues=String[];candidate=nothing
        try;candidate=_complex_candidate(s;kwargs...)
        catch err;err isa InterruptException && rethrow();push!(issues,sprint(showerror,err));end
        valid=isempty(issues)
        throw && !valid && _throw_invalid_visual(:check_inspection_selection,issues)
        (;kind=:inspection_selection,valid,issues,selection=candidate)
    end
end
function select_inspection!(s::ComplexInspectionSession;kwargs...)
    _cancel_inspection_requests!(s)
    lock(s.lock)
    try
        _inspection_require_open(s)
        s.busy && throw(ArgumentError("An inspection update or callback is already in progress."))
        c=_complex_candidate(s;kwargs...)
        s.busy=true
        try
            kwargs=(;parameter=c.parameter,cell=c.cell,dimensions=c.dimensions,activity=c.activity,boundary=c.boundary)
            snapshot=if s.options.kind===:geometric_filtration
                cached=s.options.invariant===:barcode && c.parameter[2]==s.state.parameter[2] ?
                    (s.snapshot.panels[4],s.snapshot.metadata.diagram) : nothing
                _geometric_view(s.object;kwargs...,Base.structdiff(s.options,(;kind=s.options.kind))...,_cached_barcode=cached)
            else
                visual_spec(s.object;kwargs...,s.options...)
            end
            s.snapshot=snapshot;s.state=c;s.snapshot_builds+=1
        catch;s.busy=false;rethrow();end
    finally;unlock(s.lock);end
    try;_inspection_notify(s)
    finally;lock(s.lock) do;s.busy=false;end;end
    return s
end
reset_inspection!(s::ComplexInspectionSession)=select_inspection!(s;s.initial...)
function close_inspection!(s::ComplexInspectionSession)
    _cancel_inspection_requests!(s)
    lock(s.lock)
    try
        s.closed && return s
        s.busy && throw(ArgumentError("Cannot close during an inspection update or callback."))
        s.closed=true;s.busy=true
    finally;unlock(s.lock);end
    try;_inspection_notify(s)
    finally
        lock(s.lock) do
            empty!(s.listeners);s.object=nothing;s.busy=false
        end
    end
    return s
end
function check_inspection_session(s::ComplexInspectionSession;throw=false)
    return lock(s.lock) do
        issues=String[]
        s.snapshot.metadata.parameter==s.state.parameter && s.snapshot.metadata.cell==s.state.cell ||
            push!(issues,"Snapshot and selected parameter/cell disagree.")
        if s.closed && !s.busy
            isempty(s.listeners) && s.object === nothing || push!(issues,"Closed complex session retains callbacks or input.")
        end
        append!(issues,check_visual_spec(s.snapshot).issues)
        valid=isempty(issues)
        throw && !valid && _throw_invalid_visual(:check_inspection_session,issues)
        (;kind=:inspection_session,valid,issues,closed=s.closed,revision=s.state.revision)
    end
end

function _visual_spec(s::ComplexInspectionSession,::Symbol;kwargs...)
    return VisualizationSpec(:linked_inspector;title="Explore filtered cells",panels=inspection_snapshot(s).panels,
        interaction=(;hover=true,clicks=true,labels=true,widgets=(:complex_parameter,:complex_cell,:complex_dimensions),
            notebook=:linked_inspector,mode=:live_julia,requires_live_julia=true,offline_widgets=false),
        metadata=(;session=s,inspector_kind=:complex))
end
_visual_request_cost(::ComplexInspectionSession,::Symbol)=
    (;work=:cell_activity_and_geometry,hover=:retained_cells,requires_live_julia=true,timing=:not_measured)
