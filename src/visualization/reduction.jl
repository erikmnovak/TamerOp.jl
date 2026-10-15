# Reference reduction views reuse exact owner results; drawing does no algebra.
mutable struct ReductionInspectionSession <: _AbstractInspectionSession
    object::Union{Nothing,OrdinaryPersistence.PersistenceReduction}
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

function _reduction_display_budget(r,max_cells)
    max_cells isa Integer && !(max_cells isa Bool) && max_cells>=0 || throw(ArgumentError("max_cells must be a nonnegative integer."))
    length(r.cells)<=max_cells || throw(ArgumentError("The reduction view has $(length(r.cells)) cells; raise max_cells explicitly to draw this matrix."))
    nothing
end
function _reduction_selection(r;step=0,cell=0,birth=0,matrix=:R,chain=:cycle)
    matrix in (:D,:R,:V) || throw(ArgumentError("Choose matrix :D, :R or :V."))
    chain in (:cycle,:filling) || throw(ArgumentError("Choose chain :cycle or :filling."))
    cell isa Integer && !(cell isa Bool) && 0<=cell<=length(r.cells) || throw(ArgumentError("Cell must be in 0:$(length(r.cells)); zero follows the current column."))
    birth isa Integer && !(birth isa Bool) && birth>=0 || throw(ArgumentError("Birth must be zero or a finalized birth-cell index."))
    s=OrdinaryPersistence.reduction_step(r;step)
    birth==0 || any(p->p.birth_cell==birth,s.pairs) || throw(ArgumentError("Birth cell $birth is not finalized in this prefix."))
    (;step=s.step,cell=Int(cell),birth=Int(birth),matrix,chain)
end

function _reduction_matrix_panel(r,s,p,selected)
    n=length(r.cells);A=getproperty(s,p.matrix)
    layers=AbstractVisualizationLayer[];targets=NamedTuple[]
    if n>0
        # Color encodes nonzero support; exact field residues appear as text.
        push!(layers,HeatmapLayer(collect(1.:n),collect(1.:n),Float64.(.!iszero.(reverse(A;dims=1))),:Blues,0.35,"",false))
        if selected>0
            push!(layers,RectLayer([(selected-0.48,0.5,selected+0.48,n+0.5)],_VisualRole(:selected),_VisualRole(:selected),0.14,1.))
        end
        for i in 1:n,j in 1:n
            value=A[i,j];iszero(value) && continue
            point=(Float64(j),Float64(n+1-i))
            text=string(value.val)
            push!(layers,TextLayer([length(text)<=3 && n<=16 ? text : "\u2022"],[point],_VisualRole(:foreground),12.))
        end
        if p.matrix===:R && s.provisional_pivot>0
            i,j=s.provisional_pivot,s.active
            push!(layers,RectLayer([(j-0.47,n+1-i-0.47,j+0.47,n+1-i+0.47)],_VisualRole(:selected),_VisualRole(:selected),0.18,2.))
        end
        for i in 1:n,j in 1:n
            point=(Float64(j),Float64(n+1-i))
            push!(targets,(;point,segment=(point...,point...),selection=(;cell=j),
                label="$(p.matrix)[$i,$j] = $(A[i,j].val); column cell $(r.cells[j].label), dimension $(r.cells[j].dimension)"))
        end
    end
    isempty(layers) && push!(layers,TextLayer(["No cells"],[(1.,1.)],_VisualRole(:foreground),12.))
    ticks=collect(1:max(1,cld(n,16)):n)
    VisualizationSpec(:reduction_matrix;title="$(p.matrix) \u00b7 $(_inspection_field_label(r.field))",
        subtitle="Rows and columns use the cell order. Numbers are exact residues; dots stand for larger entries. Select a column for its exact values.",layers,
        axes=(;xlabel="Column",ylabel="Row",aspect=:data,xlimits=(0.5,max(1,n)+0.5),ylimits=(0.5,max(1,n)+0.5),
            xticks=(Float64.(ticks),string.(ticks)),yticks=(Float64.(n.+1 .- reverse(ticks)),string.(reverse(ticks)))),
        metadata=(;matrix=copy(A),matrix_size=size(A),pick_targets=targets,provisional_pivot=s.provisional_pivot))
end

function _reduction_source_panel(r,s,p,selected,witness)
    geometry=r.source!==nothing && r.source.provenance.ambient_dim==2 && all(c->c.dimension<=2,r.cells)
    layers=AbstractVisualizationLayer[];targets=NamedTuple[];positions=NTuple{2,Float64}[]
    support=witness===nothing ? nothing : p.chain===:cycle ? witness.cycle : witness.filling
    coefficients=support===nothing ? Dict{Int,String}() : Dict(c.index=>string(v.val) for (c,v) in zip(support.cells,support.coefficients))
    points=NTuple{2,Float64}[];exact_points=Dict{NTuple{2,Float64},Tuple}()
    for c in sort(r.cells;by=c->(-c.dimension,c.index))
        vertices=geometry ? [Tuple(Float64.(x)) for x in DataIngestion.cell_geometry(r.source,c.source_index)] : [(Float64(c.index),Float64(c.dimension))]
        all(x->all(isfinite,x),vertices) || throw(ArgumentError("Source geometry exceeds drawing precision."))
        if geometry
            for (original,drawn) in zip(DataIngestion.cell_geometry(r.source,c.source_index),vertices)
                exact=Tuple(original)
                haskey(exact_points,drawn) && exact_points[drawn]!=exact &&
                    throw(ArgumentError("Distinct source coordinates coincide at drawing precision; use the cell catalog without geometry."))
                exact_points[drawn]=exact
            end
        end
        append!(points,vertices)
        center=Tuple(sum(v[k] for v in vertices)/length(vertices) for k in 1:2)
        role=c.index==selected ? :selected : haskey(coefficients,c.index) ? :source : :foreground
        opacity=c.index<=s.prefix || c.index==s.active ? (c.dimension==2 && geometry ? 0.2 : 1.) : 0.15
        _complex_layers!(layers,vertices,geometry ? c.dimension : 0,role,opacity)
        if geometry && haskey(coefficients,c.index)
            _complex_layers!(layers,vertices,c.dimension,:source,c.dimension==2 ? 0.45 : 1.)
        end
        label=string(c.index)*(haskey(coefficients,c.index) ? " \u00d7 $(coefficients[c.index])" : "")
        push!(layers,TextLayer([label],[(center[1],center[2]+0.13)],_VisualRole(:foreground),12.))
        source=r.source===nothing ? nothing : DataIngestion.cell_source(r.source,c.source_index)
        push!(targets,(;point=center,segment=(center...,center...),selection=(;cell=c.index),
            label="Cell $(c.index): label $(c.label), dimension $(c.dimension), grade $(c.grade)"*(source===nothing ? "" : "; source $source")))
        push!(positions,center)
    end
    isempty(layers) && push!(layers,TextLayer(["Empty source"],[(0.,0.)],_VisualRole(:foreground),12.))
    xs=first.(points);ys=last.(points)
    limits(v)=isempty(v) ? (-0.5,0.5) : (minimum(v)-max(0.4,0.12*(maximum(v)-minimum(v))),maximum(v)+max(0.4,0.12*(maximum(v)-minimum(v))))
    VisualizationSpec(:reduction_source;title=geometry ? "Source cells" : "Cell order and dimension",
        subtitle=geometry ? "Labels are matrix indices; faded cells await processing. Coefficients mark the selected chain." :
            "Schematic cell catalog; spatial coordinates are not inferred from labels.",layers,
        axes=(;xlabel=geometry ? "Source x" : "Cell order",ylabel=geometry ? "Source y" : "Dimension",aspect=geometry ? :data : :auto,xlimits=limits(xs),ylimits=limits(ys)),
        metadata=(;geometry,selected_cell=selected,cells=copy(r.cells),chain_coefficients=coefficients,pick_targets=targets,
            source=r.source===nothing ? nothing : [(;geometry=deepcopy(DataIngestion.cell_geometry(r.source,c.source_index)),identity=deepcopy(DataIngestion.cell_source(r.source,c.source_index))) for c in r.cells]))
end

function _reduction_barcode_panel(r,s,p)
    n=length(r.cells);layers=AbstractVisualizationLayer[];targets=NamedTuple[]
    # A cell-order filtration displays partial grade ties without pretending that
    # every prefix is an original parameter sublevel. Grade projection is in data.
    for (row,pair) in enumerate(s.pairs)
        b=pair.birth_cell;d=pair.death_cell;y=Float64(row)
        stop=d==0 ? s.prefix+0.65 : Float64(d)
        color=p.birth==b ? _VisualRole(:selected) : _VisualRole(:categorical,pair.dimension+1)
        push!(layers,SegmentLayer([(Float64(b),y,stop,y)],color,1.,p.birth==b ? 5. : 3.,d==0 ? :dash : :solid))
        push!(layers,PointLayer([(Float64(b),y)],color,1.,7.))
        d==0 || push!(layers,PointLayer([(stop,y)],color,1.,7.))
        point=((b+stop)/2,y)
        label="H$(pair.dimension): cell $b \u2192 "*(d==0 ? "unpaired in prefix" : "cell $d")*
            "; grade $(pair.birth) \u2192 $(pair.death===nothing ? "unpaired" : pair.death)"*(pair.zero_length ? " (empty after grade projection)" : "")
        push!(targets,(;point,segment=(Float64(b),y,stop,y),selection=(;birth=b,cell=b),label))
    end
    isempty(layers) && push!(layers,TextLayer(["No finalized births yet"],[(1.,1.)],_VisualRole(:foreground),12.))
    ticks=collect(1.:n)
    VisualizationSpec(:reduction_barcode;title="Barcode of the completed cell prefix",
        subtitle="Cell-order intervals include resolved grade ties. Dashed bars are unpaired in this prefix; equal-grade pairs vanish on projection to the original grades.",layers,
        axes=(;xlabel="Cell order (exact grades in the selection)",ylabel="Degree / birth cell",aspect=:auto,xlimits=(0.5,max(1,n)+1.),ylimits=(0.3,max(1,length(s.pairs))+0.7),
            xticks=(ticks,string.(1:n)),yticks=(Float64.(collect(eachindex(s.pairs))),["H$(q.dimension) / $(q.birth_cell)" for q in s.pairs])),
        metadata=(;pairs=copy(s.pairs),prefix=s.prefix,inside_grade_tie=s.inside_grade_tie,pick_targets=targets,parameter=:cell_order,
            original_grades=[c.grade for c in r.cells],order=r.order))
end

function _reduction_snapshot(r,p)
    s=OrdinaryPersistence.reduction_step(r;step=p.step)
    selected=p.cell==0 ? (s.active==0 ? s.prefix : s.active) : p.cell
    witness=p.birth==0 ? nothing : OrdinaryPersistence.reduction_witness(r;step=p.step,birth=p.birth)
    op=s.operation
    operation=op===nothing ? "No column has been started." : op.kind===:start ? "Start column $(op.column)." :
        op.kind===:finish ? "Finalize column $(op.column)." : "Column $(op.column) \u2190 column $(op.column) + $(op.coefficient.val) \u00d7 column $(op.source), in both R and V."
    lines=["Step $(s.step) of $(length(r.operations)). $operation",
        "R = D \u00d7 V over $(_inspection_field_label(r.field)). This is the reference column reduction.",
        "Completed columns: $(s.prefix). Active column: $(s.active). Provisional pivot: $(s.provisional_pivot).",
        s.inside_grade_tie ? "This prefix ends inside a grade tie. It is a cell-order prefix, not a full parameter sublevel." : "The completed prefix ends between grade groups.",
        "$(length(s.pairs)) finalized births in this prefix; $(count(x->x.death_cell>0,s.pairs)) have a finalized death.",
        "Birth endpoints are included and death endpoints excluded in cell order. Circle markers locate endpoints; they do not change this convention."]
    if selected>0
        c=r.cells[selected]
        push!(lines,"Selected cell $(c.index): dimension $(c.dimension), original label $(c.label), source index $(c.source_index), exact grade $(c.grade).")
        A=getproperty(s,p.matrix)
        entries=["$(r.cells[i].index):$(A[i,selected].val)" for i in axes(A,1) if !iszero(A[i,selected])]
        push!(lines,"$(p.matrix) column $selected: "*(isempty(entries) ? "zero" : join(entries,", "))*".")
    end
    if witness!==nothing
        pair=witness.pair
        push!(lines,"Selected H$(pair.dimension) pair: birth cell $(pair.birth_cell) at grade $(pair.birth); "*
            (pair.death===nothing ? "unpaired in this prefix." : "death cell $(pair.death_cell) at grade $(pair.death)."))
        chain=p.chain===:cycle ? witness.cycle : witness.filling
        detail=chain===nothing ? "No filling exists in the processed prefix for this unpaired birth." :
            join(["cell $(c.index): $(v.val)" for (c,v) in zip(chain.cells,chain.coefficients)],", ")
        push!(lines,"Selected $(p.chain): $detail")
        push!(lines,"The displayed cycle has zero boundary; a retained filling has that cycle as its boundary. Representatives follow the reference basis choices.")
    end
    panels=[_reduction_source_panel(r,s,p,selected,witness),_reduction_matrix_panel(r,s,p,selected),
        _reduction_barcode_panel(r,s,p),_inspection_text_panel("Operation and exact values",lines)]
    VisualizationSpec(:persistence_reduction;title="A persistence reduction, step by step",panels,
        metadata=(;selection=p,algorithm=:reference_column_reduction,step=s.step,prefix=s.prefix,active=s.active,
            provisional_pivot=s.provisional_pivot,inside_grade_tie=s.inside_grade_tie,pairs=s.pairs,witness,
            D=s.D,R=s.R,V=s.V,operation=s.operation,field=_inspection_field_label(r.field),order=r.order,
            panel_positions=[(1:1,1:1),(1:1,2:2),(2:2,1:2),(3:3,1:2)],panel_row_weights=[500.,270.,450.],figure_size=(1350,1300)))
end

available_visuals(::OrdinaryPersistence.PersistenceReduction)=(:persistence_reduction,)
_visual_request_keywords(::OrdinaryPersistence.PersistenceReduction,::Symbol)=(:step,:cell,:birth,:matrix,:chain,:max_cells)
function _visual_spec(r::OrdinaryPersistence.PersistenceReduction,::Symbol;max_cells=32,kwargs...)
    _reduction_display_budget(r,max_cells)
    OrdinaryPersistence.check_persistence_reduction(r;throw=true)
    _reduction_snapshot(r,_reduction_selection(r;kwargs...))
end
"""
    reduction_explorer(trace; step=0, max_cells=32)

Link a checked reference reduction to its matrix, source cells, prefix barcode
and exact cycles or fillings. Pointer selections have labelled keyboard inputs.
`select_inspection!` changes the step; reverse stepping reconstructs the earlier
state without mutating the trace. Saved HTML contains explicitly prepared states.
"""
function reduction_explorer(r::OrdinaryPersistence.PersistenceReduction;max_cells=32,kwargs...)
    _reduction_display_budget(r,max_cells)
    OrdinaryPersistence.check_persistence_reduction(r;throw=true)
    owned=deepcopy(r);p=_reduction_selection(owned;kwargs...);state=merge(p,(;revision=0))
    ReductionInspectionSession(owned,p,state,_reduction_snapshot(owned,state),1,Dict{Int,Any}(),1,String[],false,false,ReentrantLock(),_InspectionRequests())
end
inspection_selection(s::ReductionInspectionSession)=lock(s.lock) do;s.state;end
inspection_snapshot(s::ReductionInspectionSession)=lock(s.lock) do;s.snapshot;end
function inspection_summary(s::ReductionInspectionSession)
    lock(s.lock) do
        (;kind=:reduction_explorer,selection=s.state,revision=s.state.revision,closed=s.closed,updating=s.busy,
          snapshot_builds=s.snapshot_builds,listener_count=length(s.listeners),listener_errors=Tuple(s.listener_errors),requests=_inspection_request_summary(s))
    end
end
describe(s::ReductionInspectionSession)=inspection_summary(s)
Base.show(io::IO,s::ReductionInspectionSession)=print(io,"ReductionInspectionSession(step=",s.state.step,", closed=",s.closed,")")
function _reduction_candidate(s;kwargs...)
    _inspection_require_open(s)
    all(k->k in keys(s.initial),keys(kwargs)) || throw(ArgumentError("Select step, cell, birth, matrix or chain."))
    p=merge(Base.structdiff(s.state,(;revision=s.state.revision)),(;kwargs...))
    if haskey(kwargs,:step) && !haskey(kwargs,:birth) && p.birth!=0
        pairs=OrdinaryPersistence.reduction_pairs(s.object;step=p.step)
        any(x->x.birth_cell==p.birth,pairs) || (p=merge(p,(;birth=0)))
    end
    merge(_reduction_selection(s.object;p...),(;revision=s.state.revision+1))
end
function check_inspection_selection(s::ReductionInspectionSession;throw=false,kwargs...)
    issues=String[]
    lock(s.lock) do
        try;_reduction_candidate(s;kwargs...)
        catch err;err isa InterruptException && rethrow();push!(issues,sprint(showerror,err));end
    end
    valid=isempty(issues);throw && !valid && _throw_invalid_visual(:check_inspection_selection,issues)
    (;kind=:inspection_selection,valid,issues)
end
function select_inspection!(s::ReductionInspectionSession;kwargs...)
    _cancel_inspection_requests!(s)
    lock(s.lock) do
        s.busy && throw(ArgumentError("A reduction selection update is in progress."))
        p=_reduction_candidate(s;kwargs...);snap=_reduction_snapshot(s.object,p)
        s.state=p;s.snapshot=snap;s.snapshot_builds+=1;s.busy=true
    end
    try;_inspection_notify(s)
    finally;lock(s.lock) do;s.busy=false;end;end
    s
end
reset_inspection!(s::ReductionInspectionSession)=select_inspection!(s;s.initial...)
function close_inspection!(s::ReductionInspectionSession)
    _cancel_inspection_requests!(s)
    lock(s.lock) do
        s.closed && return s
        s.busy && throw(ArgumentError("Wait for the current reduction update before closing."))
        s.closed=true;s.object=nothing
    end
    try;_inspection_notify(s)
    finally;lock(s.lock) do;empty!(s.listeners);end;end
    s
end
function check_inspection_session(s::ReductionInspectionSession;throw=false)
    issues=String[]
    lock(s.lock) do
        s.snapshot.metadata.selection==s.state || push!(issues,"Snapshot selection mismatch.")
        s.closed && (s.object!==nothing || !isempty(s.listeners)) && push!(issues,"Closed explorer retains input or listeners.")
        append!(issues,check_visual_spec(s.snapshot).issues)
    end
    valid=isempty(issues);throw && !valid && _throw_invalid_visual(:check_inspection_session,issues)
    (;kind=:inspection_session,valid,issues)
end
function _visual_spec(s::ReductionInspectionSession,::Symbol;kwargs...)
    isempty(kwargs) || throw(ArgumentError("Use select_inspection! to change the reduction selection."))
    VisualizationSpec(:linked_inspector;title="A persistence reduction, step by step",panels=s.snapshot.panels,
        interaction=(;hover=true,clicks=true,labels=true,widgets=(:reduction_selection,),notebook=:linked_inspector,
            mode=:live_julia,requires_live_julia=true,offline_widgets=false),metadata=(;session=s,inspector_kind=:reduction))
end
_visual_request_cost(::ReductionInspectionSession,::Symbol)=(;work=:reference_trace_replay,hover=:read_only,requires_live_julia=true,timing=:not_measured)
