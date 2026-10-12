# Geometry and exact activity are prepared independently of rendering backends.
available_visuals(x::DataIngestion.FilteredComplex)=haskey(x.provenance,:geometric) ?
    (:filtered_complex,:geometric_filtration,:chain_inspector) : (:filtered_complex,:chain_inspector)
_visual_request_keywords(::DataIngestion.FilteredComplex,kind::Symbol)=
    kind===:chain_inspector ? _CHAIN_KEYS : kind===:geometric_filtration ? _GEOMETRIC_KEYS :
    (:parameter,:cell,:dimensions,:activity,:boundary,:view_dims,:max_cells,:labels,:max_labels,:alpha,:field_axis)
_visual_request_cost(::DataIngestion.FilteredComplex,kind::Symbol)=
    kind===:chain_inspector ? _chain_cost() : kind===:geometric_filtration ? _geometric_cost() :
    (;work=:linear_cell_activity_scan,geometry=:retained_source_cells,homology=:not_computed,
      budget=:display_only,hover=:retained_readout,timing=:not_measured)

function _complex_default_parameter(x)
    G=DataIngestion.graded_complex(x)
    sum(DataTypes.cell_counts(G))==0 && return haskey(x.provenance,:geometric) ? (0,first(DataIngestion.geometric_data(x).depth_window)) : ntuple(_->0,DataTypes.parameter_dim(G))
    grades=Iterators.flatten(DataTypes.cell_grade_set(G,i) for i in 1:sum(DataTypes.cell_counts(G)))
    return ntuple(k->(x.orientation[k]==1 ? minimum(g[k] for g in grades) : maximum(g[k] for g in grades)),
        DataTypes.parameter_dim(G))
end
_complex_active(x,i,p)=any(g->all(k->x.orientation[k]==1 ? g[k]<=p[k] : g[k]>=p[k],eachindex(p)),
    DataTypes.cell_grade_set(DataIngestion.graded_complex(x),i))
function _complex_options(x;parameter=nothing,cell=nothing,dimensions=0:2,activity=:active,
        boundary=true,view_dims=nothing,max_cells=10000,labels=false,max_labels=50,alpha=0.35,field_axis=nothing)
    G=DataIngestion.graded_complex(x);N=DataTypes.parameter_dim(G);D=x.provenance.ambient_dim
    p=parameter === nothing ? _complex_default_parameter(x) : parameter isa Real && N==1 ? (parameter,) : parameter
    p isa Union{Tuple,AbstractVector} && length(p)==N && all(v->v isa Real && !(v isa Bool) && isfinite(v),p) ||
        throw(ArgumentError("parameter must contain $N finite real grade coordinates."))
    cell === nothing || (cell isa Int && 1<=cell<=sum(DataTypes.cell_counts(G))) ||
        throw(ArgumentError("cell must be a global cell index, or nothing."))
    ds=collect(dimensions)
    !isempty(ds) && all(d->d isa Int && 0<=d<=2,ds) && allunique(ds) ||
        throw(ArgumentError("dimensions selects distinct cell dimensions from 0:2 (a displayed skeleton)."))
    activity in (:active,:all) || throw(ArgumentError("activity must be :active or :all."))
    boundary isa Bool && labels isa Bool || throw(ArgumentError("boundary and labels must be Bool."))
    max_cells isa Int && max_cells>0 && max_labels isa Int && max_labels>=0 ||
        throw(ArgumentError("max_cells must be positive and max_labels nonnegative Ints."))
    alpha isa Real && !(alpha isa Bool) && isfinite(alpha) && 0<=alpha<=1 ||
        throw(ArgumentError("alpha must be a finite opacity in [0,1]."))
    vd=view_dims === nothing ? (D<=3 ? Tuple(1:D) : throw(ArgumentError("Select view_dims explicitly for geometry above dimension three."))) : view_dims
    vd isa Tuple && length(vd) in (2,3) && all(d->d isa Int && 1<=d<=D,vd) && allunique(vd) ||
        throw(ArgumentError("view_dims must select two or three distinct source coordinates."))
    field_axis === nothing || (field_axis isa Int && 1<=field_axis<=N) ||
        throw(ArgumentError("field_axis selects one actual source-grade coordinate."))
    if field_axis !== nothing
        all(i->length(DataTypes.cell_grade_set(G,i))==1,eachindex(x.geometry)) ||
            throw(ArgumentError("Multicritical cell grades do not define a single scalar field."))
    end
    return (;parameter=Tuple(p),cell,dimensions=sort(ds),activity,boundary,view_dims=vd,max_cells,labels,max_labels,alpha=Float64(alpha),field_axis)
end
function _append_visual_request_issues!(issues::Vector{String},x::DataIngestion.FilteredComplex,kind::Symbol;kwargs...)
    kind===:chain_inspector && return _append_chain_issues!(issues,x;kwargs...)
    kind===:geometric_filtration && return _append_geometric_issues!(issues,x;kwargs...)
    try;_complex_options(x;kwargs...)
    catch err;err isa InterruptException && rethrow();push!(issues,sprint(showerror,err));end
    return issues
end

function _complex_cell_record(x,i,active)
    G=x.complex;counts=DataTypes.cell_counts(G);offsets=cumsum(vcat(0,counts))
    d=G.cell_dims[i];local_id=i-offsets[d+1]
    faces=NamedTuple[]
    if d>0
        B=DataTypes.boundary_map(G;dim=d)
        for z in nzrange(B,local_id)
            iszero(nonzeros(B)[z]) && continue
            f=offsets[d]+rowvals(B)[z]
            push!(faces,(;index=f,cell_id=DataTypes.cells(G;dim=d-1)[rowvals(B)[z]],
                coefficient=nonzeros(B)[z],active=active[f],source=DataIngestion.cell_source(x,f)))
        end
    end
    return (;index=i,cell_id=DataTypes.cells(G;dim=d)[local_id],dimension=d,active=active[i],
        grades=Tuple(DataTypes.cell_grade_set(G,i)),source=DataIngestion.cell_source(x,i),
        geometry=DataIngestion.cell_geometry(x,i),boundary=faces)
end

function _complex_layers!(layers,vertices,d,role,opacity)
    D=length(first(vertices));color=_VisualRole(role)
    if D==2
        d==0 && push!(layers,PointLayer(vertices,color,opacity,9.0))
        d==1 && push!(layers,PolylineLayer([vertices],color,opacity,2.0,false))
        d==2 && push!(layers,PolygonLayer([vertices],color,color,opacity,0.0))
    else
        d==0 && push!(layers,Point3Layer(vertices,color,opacity,9.0))
        d==1 && push!(layers,Segment3Layer([Tuple((vertices[j]...,vertices[j+1]...)) for j in 1:length(vertices)-1],color,opacity,2.0))
        d==2 && push!(layers,Polygon3Layer([vertices],color,opacity))
    end
end

function _visual_spec(x::DataIngestion.FilteredComplex,kind::Symbol;kwargs...)
    kind===:chain_inspector && return _chain_view(x;kwargs...)
    kind===:geometric_filtration && return _geometric_view(x;kwargs...)
    o=_complex_options(x;kwargs...);G=x.complex;counts=DataTypes.cell_counts(G);n=sum(counts)
    active=BitVector(_complex_active(x,i,o.parameter) for i in 1:n)
    eligible=[i for i in 1:n if G.cell_dims[i] in o.dimensions && (o.activity==:all || active[i])]
    selected=o.cell === nothing ? nothing : _complex_cell_record(x,o.cell,active)
    faces=selected === nothing || !o.boundary ? Int[] : [f.index for f in selected.boundary]
    # Keep the requested selection and its boundary visible before applying a budget.
    priority=filter(i->i in eligible,selected === nothing ? Int[] : vcat([o.cell],faces))
    ids=unique(vcat(priority,eligible));resize!(ids,min(length(ids),o.max_cells))
    chosen=Set(ids);omitted=length(eligible)-length(ids)
    layers=AbstractVisualizationLayer[];targets=NamedTuple[];warnings=String[]
    points=Dict(i=>[Tuple(Float64(p[d]) for d in o.view_dims) for p in x.geometry[i]] for i in ids)
    all(p->all(isfinite,p),Iterators.flatten(Base.values(points))) || throw(ArgumentError("Source coordinates cannot be represented as finite drawing coordinates."))
    seen=Dict{Tuple,Tuple}()
    for i in ids, (original,drawn) in zip(x.geometry[i],points[i])
        exact=Tuple(original[d] for d in o.view_dims)
        if haskey(seen,drawn) && seen[drawn]!=exact
            isempty(warnings) && push!(warnings,"Distinct exact source coordinates coincide at drawing precision; inspect the retained coordinates.")
        else
            seen[drawn]=exact
        end
    end
    # Faces first, then edges and vertices. Selection never changes activity.
    for i in sort(ids;by=i->(-G.cell_dims[i],i))
        d=G.cell_dims[i];v=points[i]
        role=i==o.cell ? :selected : i in faces ? :target : !active[i] ? :muted : d==2 ? :source : :foreground
        opacity=!active[i] ? 0.12 : d==2 ? o.alpha : 1.0
        _complex_layers!(layers,v,d,role,opacity)
        push!(targets,(;index=i,dimension=d,active=active[i],source=DataIngestion.cell_source(x,i),
            point=ntuple(k->sum(p[k] for p in v)/length(v),length(o.view_dims))))
        length(unique(v))<length(v) && push!(warnings,"Cell $i has coincident projected/drawing vertices.")
    end
    if o.field_axis !== nothing
        vertices=filter(i->G.cell_dims[i]==get(x.provenance,:source_field_dimension,0),ids)
        values=Float64[first(DataTypes.cell_grade_set(G,i))[o.field_axis] for i in vertices]
        all(isfinite,values) || throw(ArgumentError("Field values cannot be displayed as finite colors."))
        coords=[ntuple(k->sum(p[k] for p in points[i])/length(points[i]),length(o.view_dims)) for i in vertices]
        push!(layers,length(o.view_dims)==2 ? PointLayer(coords,values,1.0,12.0,:viridis) : Point3Layer(coords,values,1.0,12.0,:viridis))
    end
    labelled=sort(ids)[1:min(length(ids),o.max_labels)]
    if o.labels && length(o.view_dims)==2
        positions=[only(filter(t->t.index==i,targets)).point for i in labelled]
        push!(layers,TextLayer(string.(labelled),positions,_VisualRole(:foreground),12.0))
    end
    # An empty subcomplex still has the same source coordinate frame.
    if isempty(layers)
        push!(layers,length(o.view_dims)==2 ? PointLayer(NTuple{2,Float64}[],_VisualRole(:foreground),1.0,9.0) :
            Point3Layer(NTuple{3,Float64}[],_VisualRole(:foreground),1.0,9.0))
    end
    # Fixed source bounds avoid misleading camera changes during parameter edits.
    coords=isempty(x.geometry) ? zeros(1,length(o.view_dims)) :
        reduce(vcat,[permutedims(collect(Float64(p[d]) for d in o.view_dims)) for v in x.geometry for p in v])
    axes=length(o.view_dims)==2 ? _axes_from_matrix_2d(coords;labels=Tuple("source x$d" for d in o.view_dims)) :
        _axes_from_matrix_3d(coords;labels=Tuple("source x$d" for d in o.view_dims))
    active_counts=[count(i->active[i] && G.cell_dims[i]==d,1:n) for d in 0:length(counts)-1]
    drawn_counts=[count(i->G.cell_dims[i]==d,ids) for d in 0:length(counts)-1]
    projection=length(o.view_dims)<x.provenance.ambient_dim
    skeleton=DataTypes.max_dim(G)>maximum(o.dimensions) || filter(d->d<=DataTypes.max_dim(G),o.dimensions)!=collect(0:min(2,DataTypes.max_dim(G)))
    construction_label=replace(string(x.provenance.effective),'_'=>' ')
    subtitle="$construction_label; $(x.provenance.model) cells. Parameter $(o.parameter)."
    omitted>0 && (subtitle*="\n$(length(ids)) of $(length(eligible)) eligible cells displayed; $omitted omitted by budget.")
    get(x.provenance,:sampled,false) && (subtitle*=" Constructed on a selected source subset.")
    get(x.provenance,:sparsify,:none)!=:none && (subtitle*=" Graph selection: $(x.provenance.sparsify).")
    skeleton && (subtitle*=" Only dimensions $(o.dimensions) are displayed; other cells remain in the complex.")
    projection && (subtitle*=" Coordinate projection $(o.view_dims), not a parameter restriction.")
    legend_entries=NamedTuple[]
    o.cell in chosen && push!(legend_entries,(;label="selected cell",color=_VisualRole(:selected),style=:patch))
    any(i->i in chosen,faces) && push!(legend_entries,(;label="incident boundary",color=_VisualRole(:target),style=:patch))
    chart=VisualizationSpec(:filtered_complex_geometry;title="The complex at a parameter",subtitle,layers,axes,
        legend=_default_legend(visible=!isempty(legend_entries),entries=legend_entries),
        metadata=(;pick_targets=targets,source_view_dimension=length(o.view_dims),minimal_axes=true,displayed_indices=ids,projection,warnings,
            point_colorbar_label=o.field_axis === nothing ? "" : "Source grade $(o.field_axis)",
            point_colorrange=o.field_axis === nothing ? nothing : extrema(Float64(first(DataTypes.cell_grade_set(G,i))[o.field_axis]) for i in eachindex(x.geometry))))
    lines=[join(["C$(d-1): $(active_counts[d]) active / $(counts[d]) total" for d in eachindex(counts)],";\n"),
        "Grade directions: "*join(["$k "*(x.orientation[k]==1 ? "sublevel (<=)" : "superlevel (>=)") for k in eachindex(x.orientation)],", ")]
    G isa DataTypes.MultiCriticalGradedComplex && push!(lines,"Any appearance grade can activate a cell; activity is a union of occurrences.")
    x.provenance.grade_scale==:supplied || push!(lines,"Filtration scale: $(x.provenance.grade_scale).")
    haskey(x.provenance,:units) && push!(lines,"Units: $(x.provenance.units).")
    if get(x.provenance,:requested,nothing)==:rips_codensity
        params=x.provenance.filtration_parameters
        push!(lines,"Codensity estimator: distance-to-measure; mass $(get(params,:dtm_mass,"as declared")). Values are those used in construction.")
    end
    o.activity==:all && push!(lines,"Faint cells are inactive context; they do not belong to the selected subcomplex.")
    any(x.provenance.periodic) && push!(lines,"Periodic fundamental-domain drawing: opposite faces are identified; wrap endpoints are drawn unwrapped, not as long interior edges.")
    projection && push!(lines,"Projected overlaps do not identify source cells; the source coordinates remain in metadata.")
    o.field_axis === nothing || push!(lines,"Sample colors show actual grade coordinate $(o.field_axis), at the source cells, without interpolation or a new density estimator.")
    o.labels && length(o.view_dims)==3 && push!(lines,"3D cells are selected by pointer or exact index; cell labels are in the readout.")
    append!(lines,warnings)
    if selected === nothing
        push!(lines,"Cell selection reveals its source identity, grades and signed boundary.")
    else
        push!(lines,"Cell $(selected.index): dimension $(selected.dimension), source $(selected.source), "*(selected.active ? "active" : "inactive"))
        push!(lines,"Appearance grades: $(selected.grades)")
        push!(lines,"Boundary: "*(isempty(selected.boundary) ? "0" : join(["$(f.coefficient) * cell $(f.index)" for f in selected.boundary]," + ")))
        !(o.cell in chosen) && push!(lines,"Selected cell is outside the displayed dimensions/activity/budget; its algebra is still retained.")
    end
    readout=_inspection_text_panel(selected === nothing ? "Active cells" : "Selected cell and boundary",lines)
    return VisualizationSpec(:filtered_complex;title="Inspect a filtered complex",panels=[chart,readout],
        metadata=(;options=o,parameter=o.parameter,cell=o.cell,active,active_counts,drawn_counts,
            total_counts=counts,eligible_count=length(eligible),omitted_count=omitted,displayed_indices=ids,
            selected,projection,skeleton,construction=x.provenance,warnings,
            labels_omitted=o.labels ? (length(o.view_dims)==3 ? length(ids) : max(0,length(ids)-length(labelled))) : 0,
            panel_columns=2,figure_size=(900,650),source_geometry=:retained,homology=:not_computed))
end
