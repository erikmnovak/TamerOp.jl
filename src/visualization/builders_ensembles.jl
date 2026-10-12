# Prepared ensemble summaries: sampled bands, physical support maps and sources.

function Featurizers.persistence_ensemble(fields::AbstractVector{<:ScalarField};layer=nothing,max_cells=1_000_000,kwargs...)
    _field_compatible(fields);f=first(fields)
    lid=layer===nothing ? 1 : _collection_index(f.layers,layer,:layer)
    n=length(fields);shape=length.(f.axes)
    BigInt(n)*prod(BigInt.(shape))<=max_cells || throw(ArgumentError("Ensemble exceeds max_cells."))
    X=Array{Union{Missing,Float64}}(undef,n,shape...)
    for (i,g) in enumerate(fields)
        values=selectdim(g.values,ndims(g.values),lid)
        selectdim(X,1,i).=map(v->isnan(v) ? missing : v,values)
    end
    Featurizers.persistence_ensemble(X;axes=f.axes,sample_ids=[g.id for g in fields],
        statistic="Sampled scalar field: $(f.layers[lid])",units=f.units,axis_labels=f.axis_labels,max_cells,kwargs...)
end

const _EnsembleSummary=Featurizers.EnsembleSummary
available_visuals(::_EnsembleSummary)=(:ensemble,)
_visual_request_keywords(::_EnsembleSummary,::Symbol)=(:sample,:probe,:view,:samples,:diagram_samples,:max_cells)
_visual_request_cost(::_EnsembleSummary,::Symbol)=(;work=:prepared_summary,resampling=false,timing=:not_measured)
function _ensemble_options(s;sample=1,probe=1,view=:mean,samples=nothing,diagram_samples=(),max_cells=100_000)
    e=s.ensemble;n,p=size(e.values)
    length(e.axes)<=2 || throw(ArgumentError("Ensemble figures support one or two coordinate axes; supply a declared section for higher dimensions."))
    max_cells isa Int && max_cells>0 && BigInt(n)*p<=max_cells || throw(ArgumentError("Ensemble display exceeds max_cells."))
    i=_collection_index(e.ids,sample,:sample)
    probe isa Int && 1<=probe<=p || throw(ArgumentError("probe must index the Cartesian probe grid."))
    view in (:mean,:lower,:upper,:std,:frequency,:counts) || throw(ArgumentError("Unsupported ensemble view."))
    ds=Tuple(_collection_index(e.ids,k,:diagram_sample) for k in diagram_samples)
    length(ds)<=12 && length(unique(ds))==length(ds) || throw(ArgumentError("Select at most 12 distinct diagram samples."))
    isempty(ds) || e.results!==nothing || throw(ArgumentError("Diagram facets require retained ordinary diagrams."))
    if samples!==nothing
        length(samples)==n || throw(ArgumentError("samples must follow the ensemble's sample IDs."))
        for item in samples
            item isa NamedTuple && all(k->k in (:source,:invariant,:metadata),keys(item)) || throw(ArgumentError("Sample entries accept source, invariant and metadata."))
            get(item,:metadata,NamedTuple()) isa NamedTuple || throw(ArgumentError("Sample metadata must be a NamedTuple."))
            for k in (:source,:invariant)
                panel=get(item,k,nothing);panel===nothing && continue
                panel isa VisualizationSpec && panel.kind!==:linked_inspector || throw(ArgumentError("Supply static sample specs."))
                check_visual_spec(panel;throw=true)
            end
        end
    end
    (;sample=i,probe,view,samples,diagram_samples=ds,max_cells)
end
function _append_visual_request_issues!(issues::Vector{String},s::_EnsembleSummary,::Symbol;kwargs...)
    _ensemble_options(s;kwargs...);issues
end
_visual_spec(s::_EnsembleSummary,::Symbol;kwargs...)=_ensemble_spec(s,_ensemble_options(s;kwargs...))
function _ensemble_band_label(c)
    c.band===:hoeffding && return "$(round(100c.level;digits=2))% $(c.scope) mean confidence (known bounds, IID)"
    c.band===:bootstrap && return "$(round(100c.level;digits=2))% $(c.scope) bootstrap mean band (approximate)"
    c.band===:range ? "Observed range; descriptive" : "Empirical quantiles $(c.quantiles); descriptive"
end
function _ensemble_spec(s,o)
    e=s.ensemble;d=s.data;n,p=size(e.values);coords=collect(Iterators.product(e.axes...))
    original=e.values[o.sample,o.probe];point=coords[o.probe]
    sample_metadata=o.samples===nothing ? NamedTuple() : get(o.samples[o.sample],:metadata,NamedTuple())
    field_label=e.results===nothing ? e.context.field : _inspection_field_label(OrdinaryPersistence.field(first(e.results)))
    selected=(;sample=e.ids[o.sample],probe=o.probe,coordinates=point,value=original,
        mean=d.mean[o.probe],lower=d.lower[o.probe],upper=d.upper[o.probe],
        count=d.counts[o.probe],frequency=d.frequency[o.probe],sample_metadata)
    panels=VisualizationSpec[];bandlabel=_ensemble_band_label(s.contract)
    sample_values=[ismissing(x) ? NaN : Float64(x) for x in e.values[o.sample,:]]
    if length(e.axes)==1
        x=Float64.(only(e.axes));y=d[o.view]
        ls=AbstractVisualizationLayer[]
        # Each strip is confined to two available endpoints; gaps stay gaps.
        if o.view===:mean
            polygons=Vector{NTuple{2,Float64}}[]
            for j in 1:p-1
                all(isfinite,(d.lower[j],d.lower[j+1],d.upper[j],d.upper[j+1])) || continue
                push!(polygons,[(x[j],d.lower[j]),(x[j+1],d.lower[j+1]),(x[j+1],d.upper[j+1]),(x[j],d.upper[j])])
            end
            push!(ls,PolygonLayer(polygons,_VisualRole(:source),_VisualRole(:source),0.2,0.))
            push!(ls,SegmentLayer(NTuple{4,Float64}[(x[j],d.lower[j],x[j],d.upper[j]) for j in 1:p if isfinite(d.lower[j])&&isfinite(d.upper[j])],_VisualRole(:source),0.8,1.5))
        end
        function curve(values,color)
            push!(ls,SegmentLayer(NTuple{4,Float64}[(x[j],values[j],x[j+1],values[j+1]) for j in 1:p-1 if isfinite(values[j])&&isfinite(values[j+1])],color,1.,2.))
            push!(ls,PointLayer(NTuple{2,Float64}[(x[j],values[j]) for j in 1:p if isfinite(values[j])],color,1.,8.))
        end
        curve(y,_VisualRole(:source))
        o.view in (:mean,:lower,:upper) && curve(sample_values,_VisualRole(:selected))
        isfinite(y[o.probe]) && push!(ls,PointLayer(NTuple{2,Float64}[(x[o.probe],Float64(y[o.probe]))],_VisualRole(:target),1.,17.))
        yr=Float64[v for v in vcat(y,o.view in (:mean,:lower,:upper) ? vcat(d.lower,d.upper,sample_values) : Float64[]) if isfinite(v)]
        limits(v)=isempty(v) ? (0.,1.) : begin a,b=extrema(v);pad=max((b-a)*0.08,0.05);(a-pad,b+pad);end
        push!(panels,VisualizationSpec(:ensemble_curve;title=o.view===:mean ? e.context.statistic : String(o.view),
            subtitle=o.view===:mean ? bandlabel*"; connectors are drawing guides" : "Values at supplied probes",
            layers=ls,axes=_default_axes_2d(xlabel=only(e.context.axis_labels),ylabel=o.view===:frequency ? "Fraction above $(s.contract.threshold)" : o.view===:counts ? "Available replicates" : e.context.units,
                xlimits=limits(x),ylimits=limits(yr),aspect=:auto),
            legend=_default_legend(visible=true,entries=o.view in (:mean,:lower,:upper) ? (;summary=(;label=String(o.view),color=_VisualRole(:source),style=:line),
                selected=(;label=e.ids[o.sample],color=_VisualRole(:selected),style=:line)) : (;summary=(;label=String(o.view),color=_VisualRole(:source),style=:line))),
            metadata=(;probe_coordinates=copy(x),values=copy(y),pick_targets=[(;point=(x[j],Float64(y[j])),probe=j,label="Probe $j: $(coords[j]); $(y[j])") for j in 1:p if isfinite(y[j])])) )
    else
        x,y=Float64.(e.axes[1]),Float64.(e.axes[2]);shape=length.(e.axes)
        z=permutedims(reshape(Float64.(d[o.view]),shape));selected_z=permutedims(reshape(sample_values,shape))
        vals=filter(isfinite,vcat(vec(z),vec(selected_z)));a,b=isempty(vals) ? (0.,1.) : extrema(vals)
        a==b && (a-=0.5;b+=0.5)
        for (title,values,scale) in ((String(o.view),z,o.view===:frequency ? (0.,1.) : (a,b)),
                ("Selected replicate: $(e.ids[o.sample])",selected_z,(a,b)))
            push!(panels,VisualizationSpec(:ensemble_map;title,subtitle=title==String(o.view) ? (o.view in (:mean,:lower,:upper) ? bandlabel : "Descriptive $(o.view) at supplied probes") : e.context.statistic,
                layers=AbstractVisualizationLayer[HeatmapLayer(x,y,values,:viridis,1.,title==String(o.view) && o.view===:frequency ? "fraction" : title==String(o.view) && o.view===:counts ? "replicates" : e.context.units),
                    PointLayer(NTuple{2,Float64}[(Float64(point[1]),Float64(point[2]))],_VisualRole(:selected),1.,16.)],
                axes=_default_axes_2d(xlabel=e.context.axis_labels[1],ylabel=e.context.axis_labels[2],aspect=:auto),
                metadata=(;colorrange=scale,values,probe_coordinates=deepcopy(e.axes),
                    pick_targets=[(;point=Tuple(Float64.(coords[j])),probe=j,label="Probe $j: $(coords[j]); $(title==String(o.view) ? d[o.view][j] : e.values[o.sample,j])") for j in 1:p])))
        end
    end
    lines=["Selected replicate: $(selected.sample); probe $(o.probe) at $(point); original value: $(original).",
        "Mean $(selected.mean); band [$(selected.lower), $(selected.upper)]; available $(selected.count)/$n.",
        "Fraction strictly above $(s.contract.threshold): $(selected.frequency). This describes the observed replicates.",
        bandlabel*". Domain: supplied probes only; no between-probe coverage is claimed.",
        "Protocol: $(e.context.protocol). Population: $(e.context.population). Sampling: $(e.context.sampling).",
        "Field $(field_label); degree $(e.context.degree); layer $(e.context.layer); normalization $(e.context.normalization).",
        "Essential policy: $(e.context.essential). Missing policy: $(s.contract.missing_policy)."]
    isempty(sample_metadata) || push!(lines,"Selected replicate context: $(sample_metadata).")
    s.contract.band===:bootstrap && push!(lines,"Row bootstrap: $(s.contract.bootstrap_samples) resamples; seed $(s.contract.seed). Small or degenerate samples can give unreliable bands.")
    s.contract.band===:hoeffding && push!(lines,"Coverage assumes the declared bounds hold for the population, not merely the observed sample.")
    push!(panels,_inspection_text_panel("Interpret the ensemble",lines))
    function append_panel(panel)
        isempty(panel.panels) ? push!(panels,panel) : foreach(append_panel,panel.panels)
    end
    if e.results!==nothing
        ids=isempty(o.diagram_samples) ? (o.sample,) : o.diagram_samples
        for i in ids
            panel=visual_spec(e.results[i];dim=e.context.degree)
            append_panel(_composition_copy(panel;title="Diagram: $(e.ids[i])"))
        end
    end
    if o.samples!==nothing
        item=o.samples[o.sample]
        for k in (:invariant,:source)
            panel=get(item,k,nothing);panel===nothing || append_panel(_composition_copy(panel;title="$(e.ids[o.sample]): $(panel.title)"))
        end
    end
    text_height=max(280.,26sum(max(1,cld(length(line),95)) for line in lines)+70.)
    weights=[get(panel.metadata,:panel_style,nothing)===:text_only ? text_height : 390. for panel in panels]
    selection=Base.structdiff(o,(;samples=o.samples))
    VisualizationSpec(:ensemble;title="Repeated persistence results",panels,
        metadata=(;selection,selected,contract=s.contract,context=e.context,
            panel_positions=[(i:i,1:1) for i in eachindex(panels)],panel_columns=1,panel_row_weights=weights,
            figure_size=(1050,ceil(Int,sum(weights)+160))))
end
