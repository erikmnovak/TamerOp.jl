# Source changes and parameter selections have separate lifetimes. A build owns
# all its cells and reductions; no algebraic identity crosses a rebuild.
struct _SourceLabInput
    points::Matrix{Float64}
    ids::Vector{Int}
    mass::Float64
    cap::Float64
    version::Int
end
struct _SourceLabBuild
    input::_SourceLabInput
    source::DataIngestion.FilteredComplex
end

"""A bounded planar Rips-codensity editing session, created by `source_laboratory`.
Use `edit_source!`, `rebuild_source!`, `source_draft`, `source_object`, and the
standard inspection selection/snapshot/lifecycle functions. Draft edits do not
change the committed mathematics. Undo and reset restore drafts and also need
an explicit rebuild. Point IDs persist; cell and homology IDs do not."""
mutable struct SourceLaboratorySession <: _AbstractInspectionSession
    baseline::Union{Nothing,_SourceLabBuild}
    current::Union{Nothing,_SourceLabBuild}
    draft::Union{Nothing,_SourceLabInput}
    history::Vector{_SourceLabInput}
    future::Vector{_SourceLabInput}
    next_id::Int
    next_version::Int
    options::NamedTuple
    state::NamedTuple
    diagrams::Tuple
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

function _lab_input(points,ids,mass,cap,version,options)
    points isa AbstractMatrix && size(points,2)==2 && 2<=size(points,1)<=options.max_points ||
        throw(ArgumentError("Use a planar cloud with 2:$(options.max_points) points."))
    all(v->v isa Real && !(v isa Bool) && isfinite(v),points) || throw(ArgumentError("Point coordinates must be finite real numbers."))
    X=Float64.(points)
    all(isfinite,X) || throw(ArgumentError("Point coordinates exceed Float64."))
    # Numerical geometry is explicit, but conversion must not identify distinct inputs.
    length(unique(Tuple.(eachrow(X))))==length(unique(Tuple.(eachrow(points)))) ||
        throw(ArgumentError("Distinct points coincide after Float64 conversion; rescale the source."))
    mass isa Real && !(mass isa Bool) && isfinite(mass) && 0<Float64(mass)<1 || throw(ArgumentError("dtm_mass must be strictly between 0 and 1."))
    cap isa Real && !(cap isa Bool) && isfinite(cap) && 0<Float64(cap)<Inf || throw(ArgumentError("max_diameter must be positive and finite."))
    span=maximum(abs,X;init=0.0)
    isfinite(8span+1.2Float64(cap)) && span<sqrt(floatmax(Float64))/8 || throw(ArgumentError("Coordinates leave no safe distance or plotting range; rescale the source."))
    length(ids)==size(X,1) && all(>(0),ids) && allunique(ids) || throw(ArgumentError("Source IDs must be distinct positive integers."))
    _SourceLabInput(copy(X),copy(ids),Float64(mass),Float64(cap),version)
end
_lab_data(d::_SourceLabInput)=(;points=copy(d.points),ids=copy(d.ids),dtm_mass=d.mass,max_diameter=d.cap,input_version=d.version)
_lab_same(a,b)=a.points==b.points && a.ids==b.ids && a.mass==b.mass && a.cap==b.cap
function _lab_build(d,o)
    big(length(d.ids))^2<=o.max_matrix_entries || throw(ArgumentError("Source distance table exceeds max_matrix_entries."))
    construction=Options.ConstructionOptions(budget=Options.ConstructionBudget(max_simplices=o.max_cells))
    filtration=DataIngestion.RipsCodensityFiltration(;max_dim=o.degree+1,radius=d.cap,
        dtm_mass=d.mass,nn_backend=:bruteforce,construction)
    source=DataIngestion.filtered_complex(DataTypes.PointCloud(d.points;copy=true),filtration)
    n=length(source.complex.grades)
    n<=o.max_cells && big(n)^2<=o.max_matrix_entries || throw(ArgumentError("Rebuild exceeds max_cells or max_matrix_entries; reduce the diameter or point count."))
    DataIngestion.check_filtered_complex(source;throw=true)
    _SourceLabBuild(d,source)
end
function _lab_diagram(build,threshold,o)
    G=build.source.complex;counts=DataTypes.cell_counts(G);offsets=cumsum(vcat(0,counts))
    localids=[[i for i in 1:counts[d] if G.grades[offsets[d]+i][2]<=threshold] for d in eachindex(counts)]
    cells=[G.cells_by_dim[d][localids[d]] for d in eachindex(counts)]
    boundaries=[G.boundaries[d][localids[d],localids[d+1]] for d in 1:length(counts)-1]
    grades=Tuple{Float64}[(G.grades[offsets[d]+i][1],) for d in eachindex(counts) for i in localids[d]]
    OrdinaryPersistence.persistence_diagram(DataTypes.GradedComplex(cells,boundaries,grades);field=o.field)
end
function _lab_parameter(diameter,codensity)
    for v in (diameter,codensity)
        v isa Real && !(v isa Bool) && isfinite(v) && 0<=Float64(v)<Inf || throw(ArgumentError("Diameter and codensity must be finite nonnegative numbers."))
    end
    isfinite(1.2Float64(diameter)) || throw(ArgumentError("Diameter leaves no finite plotting margin."))
    (;diameter=Float64(diameter),codensity=Float64(codensity))
end

"""
    source_laboratory(cloud; dtm_mass=0.25, max_diameter=2.0, degree=1, field=F2(), ...)

Compare a planar point cloud with explicitly rebuilt edits using the native
Rips-codensity construction. The estimator averages squared distances to
`floor(dtm_mass*n)+1` neighbours, including the point itself. Geometry uses
Float64. `max_diameter` is the inclusive Rips edge-length cap.
`select_inspection!` selects a diameter/codensity sublevel; its barcode varies
diameter at that fixed codensity, in homology degree 0 or 1 over a prime field.
Survival at a construction cap is censored, not a claim of infinite persistence.
Budgets bound points, constructed cells, reduction entries and draft history.
"""
function source_laboratory(cloud::DataTypes.PointCloud;dtm_mass=0.25,max_diameter=2.0,
        degree=1,field=CoreModules.F2(),diameter=0.0,codensity=max_diameter,
        max_points=32,max_cells=5000,max_matrix_entries=25_000_000,history_limit=16)
    degree isa Int && degree in (0,1) || throw(ArgumentError("degree must be 0 or 1."))
    field isa CoreModules.PrimeField || throw(ArgumentError("Use a prime coefficient field."))
    all(v->v isa Int && v>0,(max_points,max_cells,max_matrix_entries,history_limit)) || throw(ArgumentError("Budgets must be positive Ints."))
    options=(;degree,field,max_points,max_cells,max_matrix_entries,history_limit)
    d=_lab_input(cloud.coords,collect(1:size(cloud.coords,1)),dtm_mass,max_diameter,1,options)
    b=_lab_build(d,options);p=_lab_parameter(diameter,codensity)
    state=merge(p,(;revision=0,generation=1));diag=_lab_diagram(b,p.codensity,options)
    snapshot=_lab_snapshot(b,b,d,state,options,(diag,diag))
    SourceLaboratorySession(b,b,d,_SourceLabInput[],_SourceLabInput[],length(d.ids)+1,2,
        options,state,(diag,diag),snapshot,1,Dict{Int,Any}(),1,String[],false,false,ReentrantLock(),_InspectionRequests())
end
source_draft(s::SourceLaboratorySession)=lock(s.lock) do;_inspection_require_open(s);_lab_data(s.draft);end
"""An independent copy of the committed filtered complex. Its provenance maps
construction row IDs to stable input IDs and records the rebuild generation."""
function source_object(s::SourceLaboratorySession;baseline=false)
    baseline isa Bool || throw(ArgumentError("baseline must be Bool."))
    lock(s.lock) do
        _inspection_require_open(s);b=baseline ? s.baseline : s.current;x=deepcopy(b.source)
        DataIngestion.FilteredComplex(x.complex,x.geometry,x.sources,x.orientation,
            merge(x.provenance,(;laboratory_input_ids=copy(b.input.ids),input_version=b.input.version,
                laboratory_generation=baseline ? 1 : s.state.generation)))
    end
end
"""Bounded draft history, oldest first, including the current draft. Each entry
contains independent copies; `input_version` identifies an input, not homology."""
source_history(s::SourceLaboratorySession)=lock(s.lock) do
    _inspection_require_open(s);[_lab_data(d) for d in (s.history...,s.draft)]
end
inspection_selection(s::SourceLaboratorySession)=lock(()->s.state,s.lock)
inspection_snapshot(s::SourceLaboratorySession)=lock(()->s.snapshot,s.lock)
function inspection_summary(s::SourceLaboratorySession)
    lock(s.lock) do
        (;kind=:source_laboratory,selection=s.state,revision=s.state.revision,generation=s.state.generation,
            closed=s.closed,updating=s.busy,dirty=s.snapshot.metadata.dirty,
            input_version=s.snapshot.metadata.draft.input_version,committed_version=s.snapshot.metadata.current.input_version,
            undo_count=length(s.history),redo_count=length(s.future),requests=_inspection_request_summary(s),
            snapshot_builds=s.snapshot_builds,listener_count=length(s.listeners),listener_errors=Tuple(s.listener_errors))
    end
end
describe(s::SourceLaboratorySession)=inspection_summary(s)
Base.show(io::IO,s::SourceLaboratorySession)=print(io,"SourceLaboratorySession(generation=",s.state.generation,", dirty=",s.snapshot.metadata.dirty,", closed=",s.closed,")")
function _lab_notify(s)
    try;_inspection_notify(s)
    finally;lock(s.lock) do;s.busy=false;end;end
    s
end
function _lab_draft_commit!(s,d,history,future;next_id=s.next_id,next_version=s.next_version)
    s.busy && throw(ArgumentError("A laboratory update is in progress."))
    state=merge(s.state,(;revision=s.state.revision+1))
    snapshot=_lab_snapshot(s.baseline,s.current,d,state,s.options,s.diagrams)
    s.draft=d;s.history=history;s.future=future;s.next_id=next_id;s.next_version=next_version
    s.state=state;s.snapshot=snapshot;s.snapshot_builds+=1;s.busy=true
end
function _lab_push(history,d,limit)
    h=vcat(history,[d]);length(h)>limit && deleteat!(h,1:length(h)-limit);h
end
"""
    edit_source!(session; move=nothing, add=nothing, remove=nothing,
                 dtm_mass=..., max_diameter=...)

Change a draft using `move=(point_id, (x,y))`, `add=(x,y)`, or `remove=point_id`.
Use at most one point operation per call. IDs are never reused after undo or
reset. No construction or persistence query runs until `rebuild_source!`.
"""
function edit_source!(s::SourceLaboratorySession;move=nothing,add=nothing,remove=nothing,
        dtm_mass=nothing,max_diameter=nothing)
    _cancel_inspection_requests!(s)
    lock(s.lock) do
        _inspection_require_open(s);d=s.draft;X=copy(d.points);ids=copy(d.ids);next_id=s.next_id
        count(!isnothing,(move,add,remove))<=1 || throw(ArgumentError("Use one point operation per edit."))
        coordinate(p)=p isa Union{Tuple,AbstractVector} && length(p)==2 && all(v->v isa Real && !(v isa Bool) && isfinite(v),p)
        if move!==nothing
            move isa Tuple && length(move)==2 && move[1] isa Int && coordinate(move[2]) || throw(ArgumentError("move must be (point_id, (x,y))."))
            i=findfirst(==(move[1]),ids);i===nothing && throw(ArgumentError("Unknown source point ID."));X[i,:]=collect(move[2])
        elseif add!==nothing
            coordinate(add) || throw(ArgumentError("add must be (x,y)."));X=vcat(X,reshape(Float64.(collect(add)),1,2));push!(ids,next_id);next_id+=1
        elseif remove!==nothing
            remove isa Int || throw(ArgumentError("remove must be a source point ID."))
            i=findfirst(==(remove),ids);i===nothing && throw(ArgumentError("Unknown source point ID."));keep=setdiff(eachindex(ids),[i]);X=X[keep,:];ids=ids[keep]
        end
        candidate=_lab_input(X,ids,something(dtm_mass,d.mass),something(max_diameter,d.cap),s.next_version,s.options)
        _lab_draft_commit!(s,candidate,_lab_push(s.history,d,s.options.history_limit),_SourceLabInput[];
            next_id,next_version=s.next_version+1)
    end
    _lab_notify(s)
end
function undo_source_edit!(s::SourceLaboratorySession;redo=false)
    redo isa Bool || throw(ArgumentError("redo must be Bool."));_cancel_inspection_requests!(s)
    lock(s.lock) do
        _inspection_require_open(s);from=redo ? s.future : s.history
        isempty(from) && throw(ArgumentError(redo ? "No draft to redo." : "No draft to undo."))
        to=_lab_push(redo ? s.history : s.future,s.draft,s.options.history_limit)
        _lab_draft_commit!(s,last(from),redo ? to : from[1:end-1],redo ? from[1:end-1] : to)
    end
    _lab_notify(s)
end
function reset_inspection!(s::SourceLaboratorySession)
    _cancel_inspection_requests!(s)
    lock(s.lock) do
        _inspection_require_open(s)
        _lab_draft_commit!(s,s.baseline.input,_lab_push(s.history,s.draft,s.options.history_limit),_SourceLabInput[])
    end
    _lab_notify(s)
end
function select_inspection!(s::SourceLaboratorySession;diameter=s.state.diameter,codensity=s.state.codensity,rebuild=false)
    rebuild isa Bool || throw(ArgumentError("rebuild must be Bool."));_cancel_inspection_requests!(s)
    lock(s.lock) do
        _inspection_require_open(s);s.busy && throw(ArgumentError("A laboratory update is in progress."))
        p=_lab_parameter(diameter,codensity)
        current=rebuild ? _lab_build(s.draft,s.options) : s.current
        baseline_diagram=p.codensity==s.state.codensity ? s.diagrams[1] : _lab_diagram(s.baseline,p.codensity,s.options)
        current_diagram=!rebuild && p.codensity==s.state.codensity ? s.diagrams[2] : _lab_diagram(current,p.codensity,s.options)
        diagrams=(baseline_diagram,current_diagram)
        state=merge(p,(;revision=s.state.revision+1,generation=s.state.generation+Int(rebuild)))
        snapshot=_lab_snapshot(s.baseline,current,s.draft,state,s.options,diagrams)
        s.current=current;s.diagrams=diagrams;s.state=state;s.snapshot=snapshot;s.snapshot_builds+=1;s.busy=true
    end
    _lab_notify(s)
end
rebuild_source!(s::SourceLaboratorySession)=select_inspection!(s;rebuild=true)
function check_inspection_selection(s::SourceLaboratorySession;throw=false,diameter=s.state.diameter,codensity=s.state.codensity,rebuild=false)
    issues=String[]
    try;_inspection_require_open(s);_lab_parameter(diameter,codensity);rebuild isa Bool || Base.throw(ArgumentError("rebuild must be Bool."))
    catch err;err isa InterruptException && rethrow();push!(issues,sprint(showerror,err));end
    valid=isempty(issues);throw && !valid && _throw_invalid_visual(:check_inspection_selection,issues)
    (;kind=:inspection_selection,valid,issues)
end
function _inspection_fork(s::SourceLaboratorySession)
    SourceLaboratorySession(s.baseline,s.current,s.draft,copy(s.history),copy(s.future),s.next_id,s.next_version,
        s.options,s.state,s.diagrams,s.snapshot,s.snapshot_builds,Dict{Int,Any}(),1,String[],false,false,ReentrantLock(),_InspectionRequests())
end
function _inspection_adopt!(s::SourceLaboratorySession,t::SourceLaboratorySession)
    s.current=t.current;s.diagrams=t.diagrams;s.state=t.state;s.snapshot=t.snapshot;s.snapshot_builds=t.snapshot_builds
    s.draft=t.draft;s.history=t.history;s.future=t.future;s.next_id=t.next_id;s.next_version=t.next_version
end
function _apply_inspection_request!(s::SourceLaboratorySession;operation=:select,edit=(;),kwargs...)
    if operation===:select
        select_inspection!(s;kwargs...)
    elseif operation===:edit
        edit_source!(s;edit...)
    elseif operation===:undo || operation===:redo
        undo_source_edit!(s;redo=operation===:redo)
    elseif operation===:reset
        reset_inspection!(s)
    else
        throw(ArgumentError("Unknown source laboratory operation."))
    end
end
function close_inspection!(s::SourceLaboratorySession)
    _cancel_inspection_requests!(s)
    lock(s.lock) do
        s.closed && return s
        s.busy && throw(ArgumentError("Cannot close during a laboratory update."));s.closed=true
    end
    try;_inspection_notify(s)
    finally
        lock(s.lock) do
            empty!(s.listeners);empty!(s.history);empty!(s.future);s.baseline=nothing;s.current=nothing;s.draft=nothing;s.diagrams=()
        end
    end
    s
end
function check_inspection_session(s::SourceLaboratorySession;throw=false)
    lock(s.lock) do
        issues=copy(check_visual_spec(s.snapshot).issues)
        s.snapshot.metadata.selection==s.state || push!(issues,"Snapshot and laboratory selection disagree.")
        if s.closed
            (s.current===nothing && s.baseline===nothing && s.draft===nothing && isempty(s.listeners) && isempty(s.history) && isempty(s.diagrams)) || push!(issues,"Closed laboratory retains inputs or listeners.")
        else
            s.snapshot.metadata.dirty==!_lab_same(s.draft,s.current.input) || push!(issues,"Draft status disagrees with committed input.")
        end
        valid=isempty(issues);throw && !valid && _throw_invalid_visual(:check_inspection_session,issues)
        (;kind=:inspection_session,valid,issues,closed=s.closed,revision=s.state.revision)
    end
end
function _visual_spec(s::SourceLaboratorySession,::Symbol;kwargs...)
    isempty(kwargs) || throw(ArgumentError("Use edit_source! or select_inspection! to change the laboratory."))
    VisualizationSpec(:linked_inspector;title="Source and construction laboratory",panels=inspection_snapshot(s).panels,
        interaction=(;hover=true,clicks=true,labels=true,widgets=(:source_draft,:source_rebuild,:source_history),
            notebook=:linked_inspector,mode=:live_julia,requires_live_julia=true,offline_widgets=false),
        metadata=(;session=s,inspector_kind=:source_laboratory))
end
_visual_request_cost(::SourceLaboratorySession,::Symbol)=(;work=:explicit_source_rebuild,geometry=:float64,
    navigation=:retained_complex,fixed_codensity_barcode=:recomputed_on_threshold_change,requires_live_julia=true,timing=:not_measured)
