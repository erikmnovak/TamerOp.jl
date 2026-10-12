available_visuals(::DataIngestion.SourceChain)=(:source_chain,:chain_inspector)
_visual_request_keywords(::DataIngestion.SourceChain,kind::Symbol)=kind===:chain_inspector ? _CHAIN_KEYS : (:view_dims,:max_cells,:max_labels,:labels)
_visual_request_cost(::DataIngestion.SourceChain,kind::Symbol)=kind===:chain_inspector ? _chain_cost() :
    (;work=:retained_source_overlay,homology=:not_computed,optimization=:not_computed,validation=:source_boundary_equations)

function _source_chain_options(c;view_dims=nothing,max_cells=10000,max_labels=50,labels=true)
    DataIngestion.check_source_chain(c;throw=true)
    c.dimension<=2 || throw(ArgumentError("Source-chain overlays currently draw degrees 0, 1 and 2; higher chain data remain available through source_chain_data."))
    o=_complex_options(c.source;parameter=c.parameter,view_dims,max_cells,max_labels,labels,dimensions=0:min(2,DataTypes.max_dim(c.source.complex)))
    return o
end
function _append_visual_request_issues!(issues::Vector{String},c::DataIngestion.SourceChain,kind::Symbol;kwargs...)
    kind===:chain_inspector && return _append_chain_issues!(issues,c;kwargs...)
    try;_source_chain_options(c;kwargs...)
    catch err;err isa InterruptException && rethrow();push!(issues,sprint(showerror,err));end
    return issues
end

function _visual_spec(c::DataIngestion.SourceChain,kind::Symbol;kwargs...)
    kind===:chain_inspector && return _chain_view(c;kwargs...)
    o=_source_chain_options(c;kwargs...)
    context=_visual_spec(c.source,:filtered_complex;parameter=c.parameter,view_dims=o.view_dims,max_cells=o.max_cells,
        dimensions=o.dimensions,labels=false)
    base=first(context.panels);layers=AbstractVisualizationLayer[]
    for layer in base.layers
        muted=_VisualRole(:muted)
        push!(layers,layer isa PointLayer ? PointLayer(layer.points,muted,0.45,5.0) :
            layer isa Point3Layer ? Point3Layer(layer.points,muted,0.45,5.0) :
            layer isa PolylineLayer ? PolylineLayer(layer.paths,muted,0.4,1.0,layer.closed) :
            layer isa Segment3Layer ? Segment3Layer(layer.segments,muted,0.4,1.0) :
            layer isa PolygonLayer ? PolygonLayer(layer.polygons,muted,muted,0.08,0.0) :
            layer isa Polygon3Layer ? Polygon3Layer(layer.polygons,muted,0.08) : layer)
    end
    data=DataIngestion.source_chain_data(c)
    field_label=c.field isa CoreModules.PrimeField ? "F$(c.field.p)" : "QQ"
    counts=DataTypes.cell_counts(c.source.complex)
    offset=sum(counts[1:c.dimension]);support=[offset+i for i in data.cell_indices]
    drawn=support[1:min(length(support),o.max_cells)]
    positions=NTuple{2,Float64}[];texts=String[];records=NamedTuple[];targets=NamedTuple[]
    for (j,i) in enumerate(drawn)
        points=[Tuple(Float64(p[d]) for d in o.view_dims) for p in c.source.geometry[i]]
        all(p->all(isfinite,p),points) || throw(ArgumentError("Source coordinates cannot be drawn as finite values."))
        if c.dimension==1
            push!(layers,length(o.view_dims)==2 ? PolylineLayer([points],_VisualRole(:selected),1.0,5.0,false) :
                Segment3Layer([Tuple((points[j]...,points[j+1]...)) for j in 1:length(points)-1],_VisualRole(:selected),1.0,5.0))
        else
            _complex_layers!(layers,points,c.dimension,:selected,c.dimension==2 ? 0.65 : 1.0)
        end
        record=_complex_cell_record(c.source,i,context.metadata.active)
        push!(records,merge(record,(;coefficient=data.coefficients[j])))
        # Chain support is drawn independently of the muted context budget.
        # Its anchors must therefore come from the actual bold geometry.
        push!(targets,(;index=i,dimension=c.dimension,active=true,source=record.source,
            point=ntuple(k->sum(p[k] for p in points)/length(points),length(o.view_dims))))
        if o.labels && length(o.view_dims)==2 && j<=o.max_labels
            push!(positions,ntuple(k->sum(p[k] for p in points)/length(points),2))
            push!(texts,"$(_matrix_coefficient_text(data.coefficients[j]))\u00b7e$(i-offset)")
        end
    end
    isempty(texts) || push!(layers,TextLayer(texts,positions,_VisualRole(:foreground),13.0))
    omitted=length(support)-length(drawn)
    title=c.kind==:cycle ? "A cycle on source cells" : c.kind==:cocycle ? "A cocycle at a parameter" : "A source chain"
    axes=base.axes
    if length(o.view_dims)==2
        padded(limits)=let (a,b)=limits; margin=0.16*(b-a);(a-margin,b+margin);end
        axes=merge(axes,(;xlimits=padded(axes.xlimits),ylimits=padded(axes.ylimits)))
    end
    chart=VisualizationSpec(:source_chain_geometry;title="Source support",
        subtitle="$field_label, degree $(c.dimension), parameter $(c.parameter). $omitted support cells omitted by budget.",
        axes,layers,metadata=(;minimal_axes=true,displayed_indices=drawn,source_chain=data,
            source_view_dimension=length(o.view_dims),pick_targets=targets))
    lines=[c.field isa CoreModules.PrimeField ? "Bold cells carry residues modulo $(c.field.p) in the retained oriented basis." :
        "Bold cells carry exact rational coefficients in the retained oriented basis.",
        "Labels e\u1d62 use dimension-local indices; signed boundaries define orientation.",
        "$(length(drawn)) / $(length(support)) support cells displayed; parameter $(c.parameter).",
        c.kind==:cycle ? "Boundary is zero on original cells." : c.kind==:cocycle ?
            "Coboundary is zero on active original cells. These are cochain values, not a homology cycle." :
            "This is a chain; no cycle equation is asserted.",
        get(c.metadata,:choice,:supplied_coordinates)==:supplied_coordinates ?
            "Choice: supplied coordinates, lifted by the checked map." : "Choice: noncanonical persistence reduction."]
    any(c.source.provenance.periodic) && push!(lines,"Periodic seams identify opposite faces. Wrap cells are drawn unwrapped, with their original source IDs.")
    if get(c.metadata,:optimization,:none)==:exact_weighted_support
        push!(lines,"Minimum weighted support at this parameter: $(c.metadata.objective), over $(c.metadata.candidates) boundary-image candidates.")
    else
        push!(lines,"This representative has no minimum-length or minimum-support guarantee.")
    end
    readout_count=min(length(records),o.max_labels,8)
    for r in records[1:readout_count]
        boundary=isempty(r.boundary) ? "0" : join(["$(f.coefficient)\u00b7cell$(f.index)" for f in r.boundary]," + ")
        push!(lines,"$(r.coefficient)\u00b7e$(r.index-offset): source $(r.source); boundary $boundary")
    end
    length(records)>readout_count && push!(lines,"$(length(records)-readout_count) further source records remain in the exact metadata.")
    readout=_inspection_text_panel("Coefficients and source correspondence",lines)
    return VisualizationSpec(:source_chain;title,panels=[chart,readout],
        metadata=(;source_chain=data,displayed_support=Tuple(drawn),omitted_support=omitted,
            readout_omitted=length(records)-readout_count,context_omitted=context.metadata.omitted_count,
            source_geometry=:verified_source_cells,representatives=:retained,optimization=:not_run_by_plot,
            projection=context.metadata.projection,panel_columns=2,figure_size=(1300,900)))
end
