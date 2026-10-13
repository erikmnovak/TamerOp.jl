# A geometric snapshot separates exact cell activity, sampled source coverage,
# and the selected algebra. Renderers never infer topology from the raster.
const _GEOMETRIC_KEYS=(:parameter,:cell,:dimensions,:activity,:boundary,:view_dims,
    :max_cells,:labels,:max_labels,:alpha,:field_axis,:source_dims,:source_anchor,
    :sample_resolution,:degree,:field,:invariant,:max_algebra_cells,:max_matrix_entries,:compare)
_geometric_cost()=(;work=:retained_cells_and_sampled_source_geometry,
    homology=:selected_parameter_ranks,barcode=:explicit_fixed_depth_reduction,
    budget=:separate_drawing_and_algebra,hover=:retained_readout,timing=:not_measured)

function _geometric_options(x;parameter=nothing,source_dims=(1,2),source_anchor=nothing,
        sample_resolution=25,degree=0,field=CoreModules.F2(),invariant=:dimensions,
        max_algebra_cells=10000,max_matrix_entries=1_000_000,compare=nothing,kwargs...)
    m=DataIngestion.geometric_data(x);D=size(m.sites,2)
    p=DataIngestion.coverage_at(x,Tuple(m.sites[1,:]);parameter=parameter===nothing ? (0,clamp(1,m.depth_window...)) : parameter).parameter
    o=_complex_options(x;parameter=p,kwargs...)
    D>=2 || throw(ArgumentError("The source panel requires at least two spatial coordinates."))
    source_dims isa Tuple && length(source_dims)==2 && all(d->d isa Int && 1<=d<=D,source_dims) && allunique(source_dims) ||
        throw(ArgumentError("source_dims must select two distinct spatial coordinates."))
    anchor=source_anchor===nothing ? Tuple(sum(m.sites[:,a])/size(m.sites,1) for a in 1:D) : source_anchor
    anchor isa Union{Tuple,AbstractVector} && length(anchor)==D && all(v->v isa Real && !(v isa Bool) && isfinite(v),anchor) ||
        throw(ArgumentError("source_anchor must contain $D finite spatial coordinates."))
    sample_resolution isa Int && 2<=sample_resolution<=128 || throw(ArgumentError("sample_resolution must be an Int in 2:128."))
    degree isa Int && degree>=0 || throw(ArgumentError("degree must be a nonnegative Int."))
    field isa CoreModules.AbstractCoeffField || throw(ArgumentError("field must be a coefficient field."))
    invariant in (:none,:dimensions,:barcode) || throw(ArgumentError("invariant must be :none, :dimensions or :barcode."))
    invariant===:barcode && !(field isa CoreModules.PrimeField) && throw(ArgumentError("The ordinary barcode route requires a prime coefficient field."))
    for b in (max_algebra_cells,max_matrix_entries)
        b isa Int && b>0 || throw(ArgumentError("Algebra budgets must be positive Ints."))
    end
    if compare!==nothing
        compare isa DataIngestion.FilteredComplex || throw(ArgumentError("compare must be a retained filtered complex."))
        b=DataIngestion.geometric_data(compare)
        m.sites==b.sites && m.family==b.family && m.representation==b.representation ||
            throw(ArgumentError("Overlays require the same indexed sites, family and geometric coordinate model. Use separate views otherwise."))
        m.representation===:schematic_flags && throw(ArgumentError("Coincident schematic flag drawings are not geometric overlays; use separate views."))
        m.family===:core && (m.beta!=b.beta || m.k_values!=b.k_values) && throw(ArgumentError("Core overlays require matching beta and depth samples."))
        DataIngestion.coverage_at(compare,Tuple(b.sites[1,:]);parameter=p)
    end
    return (;complex=o,source_dims,source_anchor=Tuple(anchor),sample_resolution,degree,field,invariant,
        max_algebra_cells,max_matrix_entries,compare)
end
function _append_geometric_issues!(issues,x;kwargs...)
    try;_geometric_options(x;kwargs...)
    catch err;err isa InterruptException && rethrow();push!(issues,sprint(showerror,err));end
    return issues
end

function _geometric_source_panel(x,o)
    m=DataIngestion.geometric_data(x);X=m.sites;D=size(X,2);sd=o.source_dims
    r,k=o.complex.parameter;rf=Float64(r)
    isfinite(rf) || throw(ArgumentError("The radius cannot be represented at drawing precision."))
    extent=[extrema(Float64.(X[:,a])) for a in sd]
    padding=max(rf,maximum(b-a for (a,b) in extent)/10,0.25)
    xs=collect(range(extent[1][1]-padding,extent[1][2]+padding;length=o.sample_resolution))
    ys=collect(range(extent[2][1]-padding,extent[2][2]+padding;length=o.sample_resolution))
    values=zeros(length(ys),length(xs));counts=zeros(Int,size(values))
    for j in eachindex(ys),i in eachindex(xs)
        point=ntuple(a->a==sd[1] ? xs[i] : a==sd[2] ? ys[j] : o.source_anchor[a],D)
        record=DataIngestion.coverage_at(x,point;parameter=(r,k))
        values[j,i]=record.member;counts[j,i]=record.count
    end
    layers=AbstractVisualizationLayer[HeatmapLayer(xs,ys,values,:Blues,0.45,"",false)]
    eligible=DataIngestion.coverage_at(x,Tuple(X[1,:]);parameter=(r,k)).eligible_sites
    circles=Vector{NTuple{2,Float64}}[]
    for i in eligible
        # These are actual sphere sections, not projected disks for a 3D cloud.
        off2=sum(((X[i,a]-CoreModules.QQ(o.source_anchor[a]))^2 for a in 1:D if !(a in sd));init=zero(CoreModules.QQ))
        radius2=(r isa DataIngestion.AlgebraicReal ? r^2 : CoreModules.QQ(r)^2)-off2
        radius2<0 && continue
        cr=sqrt(Float64(radius2))
        push!(circles,[(Float64(X[i,sd[1]])+cr*cos(t),Float64(X[i,sd[2]])+cr*sin(t)) for t in range(0,2pi;length=97)])
    end
    push!(layers,PolylineLayer(circles,_VisualRole(:source),0.65,1.0,false))
    pts=[(Float64(X[i,sd[1]]),Float64(X[i,sd[2]])) for i in axes(X,1)]
    push!(layers,PointLayer(pts,_VisualRole(:foreground),1.0,8.0))
    selected_sites=Int[]
    if o.complex.cell!==nothing
        src=DataIngestion.cell_source(x,o.complex.cell)
        selected_sites=src isa Tuple && !(src isa NamedTuple) ? collect(src) :
            haskey(src,:subsets) ? sort!(unique!(collect(Iterators.flatten(src.subsets)))) : sort!(collect((src.inside...,src.on...)))
        push!(layers,PointLayer(pts[selected_sites],_VisualRole(:selected),1.0,13.0))
    end
    o.complex.labels && push!(layers,TextLayer(string.(1:min(size(X,1),o.complex.max_labels)),
        pts[1:min(size(X,1),o.complex.max_labels)],_VisualRole(:foreground),12.0))
    note=m.family===:multicover ? "Closed indexed balls; shaded samples have coverage >= $k." :
        m.backend===:core_delaunay ? "Eligible balls restricted to full-cloud Voronoi cells; shaded membership samples." :
        "Union of eligible core balls; shaded membership samples."
    note*=" Sampling is a picture, not a homology computation."
    D>2 && (note*=" Spatial section $(sd), remaining coordinates fixed at $(o.source_anchor). Sites are projected markers.")
    panel=VisualizationSpec(:geometric_coverage;title=m.family===:multicover ? "Source multicover" : "Source core region",
        subtitle=note,layers,axes=_default_axes_2d(xlabel="source x$(sd[1])",ylabel="source x$(sd[2])",
            xlimits=extrema(xs),ylimits=extrema(ys)),
        metadata=(;sampled=true,coverage_counts=counts,membership=Bool.(values),xs,ys,
            source_dims=sd,source_anchor=o.source_anchor,eligible_sites=eligible,selected_sites,closed_balls=true))
    return panel
end

function _geometric_parameter_panel(x,o)
    G=DataIngestion.graded_complex(x);m=DataIngestion.geometric_data(x)
    grades=unique([g for i in 1:sum(DataTypes.cell_counts(G)) for g in DataTypes.cell_grade_set(G,i)])
    # max_cells bounds this display too; exact event records remain in metadata.
    shown=grades[1:min(length(grades),o.complex.max_cells)]
    r,k=o.complex.parameter
    upper=m.radius_cutoff===nothing ? max(Float64(r),maximum((Float64(g[1]) for g in grades);init=0.0),1.0) : Float64(m.radius_cutoff)
    lo,hi=m.depth_window
    displayed_events=[(Float64(g[1]),Float64(g[2])) for g in shown]
    all(p->all(isfinite,p),displayed_events) || throw(ArgumentError("Cell events cannot be represented at finite drawing precision."))
    collisions=length(shown)-length(unique(displayed_events))
    layers=AbstractVisualizationLayer[
        PointLayer(displayed_events,_VisualRole(:muted),0.6,6.0),
        SegmentLayer([(0.0,Float64(k),upper,Float64(k))],_VisualRole(:source),0.7,1.5),
        PointLayer([(Float64(r),Float64(k))],_VisualRole(:selected),1.0,13.0)]
    note="Cell appearance grades; radius increases, depth decreases. Both depth caps are included."
    length(shown)<length(grades) && (note*=" $(length(grades)-length(shown)) events omitted from drawing by budget.")
    collisions>0 && (note*=" $collisions distinct events coincide at drawing precision; exact event records remain available.")
    return VisualizationSpec(:geometric_parameters;title="Radius and depth",subtitle=note,layers,
        axes=_default_axes_2d(xlabel="physical radius r",ylabel=m.family===:core ? "neighbour depth k" : "coverage depth k",xlimits=(-0.03max(upper,1),1.06max(upper,1)),
            ylimits=(lo-0.4,hi+0.4),yticks=hi-lo<=12 ? (Float64.(lo:hi),string.(lo:hi)) : nothing,
            aspect=:auto),metadata=(;events=grades,events_omitted=length(grades)-length(shown),drawing_collisions=collisions,
            radius_squared=[g[1]^2 for g in grades],closed_depth_window=m.depth_window,parameter=(r,k)))
end

function _geometric_betti(x,o)
    G=DataIngestion.graded_complex(x);counts=DataTypes.cell_counts(G);offsets=cumsum(vcat(0,counts))
    ids=[[j for j in 1:counts[d] if _complex_active(x,offsets[d]+j,o.complex.parameter)] for d in eachindex(counts)]
    ns=length.(ids)
    sum(ns)<=o.max_algebra_cells || throw(ArgumentError("Active homology exceeds max_algebra_cells; select invariant=:none or raise the budget."))
    sum((big(ns[d])*ns[d+1] for d in 1:length(ns)-1);init=big(0))<=o.max_matrix_entries ||
        throw(ArgumentError("Active boundary ranks exceed max_matrix_entries; select invariant=:none or raise the budget."))
    K=CoreModules.coeff_type(o.field)
    ranks=[FieldLinAlg.rank(o.field,K.(DataTypes.boundary_map(G;dim=d)[ids[d],ids[d+1]])) for d in 1:length(ns)-1]
    padded=vcat(0,ranks,0)
    return [ns[d]-padded[d]-padded[d+1] for d in eachindex(ns)]
end

function _geometric_barcode(x,o)
    G=DataIngestion.graded_complex(x);m=DataIngestion.geometric_data(x)
    counts=DataTypes.cell_counts(G)
    sum(counts)<=o.max_algebra_cells && big(sum(counts))^2<=o.max_matrix_entries ||
        throw(ArgumentError("Fixed-depth barcode exceeds algebra budgets; use invariant=:dimensions or raise the budgets."))
    slice=DataIngestion.radius_slice(x;depth=o.complex.parameter[2])
    diagram=OrdinaryPersistence.persistence_diagram(slice;field=o.field)
    payload=_interval_payload(diagram;dim=o.degree)
    if m.radius_cutoff!==nothing
        records=Tuple(isfinite(b.death) ? b : merge(b,(;death=m.radius_cutoff,right_closed=true,
            right_status=:censored,right_clipped=true,singleton=b.birth==m.radius_cutoff)) for b in payload.records)
        payload=merge(payload,(;records,essential_status=:survives_radius_cutoff,
            metadata=merge(payload.metadata,(;restriction_window=(0,m.radius_cutoff),
                essential_interpretation=:survives_radius_cutoff))))
    end
    panel=_interval_payload_spec(payload,:barcode;barcode_title="Radius barcode: H$(o.degree), depth $(o.complex.parameter[2])")
    positions,labels=panel.axes.xticks
    labels=[occursin("Inf",label) ? label : string(round(position;sigdigits=5)) for (position,label) in zip(positions,labels)]
    panel=VisualizationSpec(panel.kind;title=panel.title,subtitle=panel.subtitle,layers=panel.layers,panels=panel.panels,
        axes=merge(panel.axes,(;xlabel="physical radius r",xticks=(positions,labels))),
        legend=panel.legend,interaction=panel.interaction,metadata=panel.metadata)
    return panel,diagram
end

function _geometric_view(x;_cached_barcode=nothing,kwargs...)
    o=_geometric_options(x;kwargs...);m=DataIngestion.geometric_data(x)
    base=_visual_spec(x,:filtered_complex;o.complex...)
    chart=base.panels[1]
    labels=Tuple(m.representation===:source_simplices ? "source x$d" :
        m.representation===:schematic_flags ? (d==size(m.sites,2)+1 ? "subset size" : "subset mean x$d") :
        d==size(m.sites,2)+1 ? "lifted depth" : "sum of sites x$d" for d in o.complex.view_dims)
    chartaxes=merge(chart.axes,(;xlabel=labels[1],ylabel=labels[2]),length(labels)==3 ? (;zlabel=labels[3]) : (;))
    if length(labels)==2
        # A collinear source should have a readable geometric frame, not a
        # nearly zero-height axis with overlapping ticks.
        a,b=chartaxes.xlimits; c,d=chartaxes.ylimits;span=max(b-a,d-c)
        if min(b-a,d-c)<0.4span
            xr=max((b-a)/2,0.2span);yr=max((d-c)/2,0.2span)
            chartaxes=merge(chartaxes,(;xlimits=((a+b)/2-xr,(a+b)/2+xr),ylimits=((c+d)/2-yr,(c+d)/2+yr)))
        end
    end
    modelnote=m.representation===:schematic_flags ? "Schematic labelled flags; coincident positions do not identify cells." :
        m.representation===:lifted_rhomboid ? "Lifted cells (sum of sites, positive depth); a projection is not a depth slice." :
        "Constructed source simplices; core grades use numerical geometry."
    layers=copy(chart.layers);comparison=nothing
    if o.compare!==nothing
        other=_visual_spec(o.compare,:filtered_complex;o.complex...,cell=nothing)
        for i in other.metadata.displayed_indices
            vertices=[Tuple(Float64(p[d]) for d in o.complex.view_dims) for p in DataIngestion.cell_geometry(o.compare,i)]
            _complex_layers!(layers,vertices,DataIngestion.graded_complex(o.compare).cell_dims[i],:target,0.35)
        end
        b=DataIngestion.geometric_data(o.compare)
        comparison=(;backend=b.backend,relation=m.family===:multicover && m.max_dim===nothing && b.max_dim===nothing ?
            :same_multicover_on_common_window : :geometric_comparison_only,
            hypotheses=(;same_indexed_sites=true,same_coordinate_model=true,parameter_in_both_windows=true),
            active_counts=other.metadata.active_counts,omitted_count=other.metadata.omitted_count)
        modelnote*=" Comparison cells overlaid; relation: $(comparison.relation). No cell correspondence is inferred."
    end
    legend=chart.legend
    if comparison!==nothing
        legend=_default_legend(visible=true,entries=[(;label="primary model",color=_VisualRole(:foreground),style=:line),
            (;label="comparison cells",color=_VisualRole(:target),style=:line)])
    end
    chart=VisualizationSpec(:geometric_cells;title="The constructed cell model",subtitle=modelnote*"\n"*chart.subtitle,
        layers,axes=chartaxes,metadata=chart.metadata,legend)
    betti=o.invariant===:none ? nothing : _geometric_betti(x,o)
    selected_betti=betti===nothing ? nothing : o.degree<length(betti) ? betti[o.degree+1] : 0
    diagrams=nothing
    panels=[chart,_geometric_source_panel(x,o),_geometric_parameter_panel(x,o)]
    lines=["Selected physical parameter $(o.complex.parameter); $(_inspection_field_label(o.field)).",
        "Grade precision: $(m.grade_precision). Construction: $(m.backend).",
        "Drawn cells $(sum(base.metadata.drawn_counts)); active cells $(sum(base.metadata.active_counts))."]
    if o.invariant===:barcode
        panel,diagrams=_cached_barcode===nothing ? _geometric_barcode(x,o) : _cached_barcode
        push!(panels,panel)
    end
    betti===nothing ? push!(lines,"Homology not computed (invariant=:none).") :
        push!(lines,"Active complex Betti dimensions H0, H1, ...: $(betti). Selected H$(o.degree): $selected_betti.")
    m.max_dim===nothing || push!(lines,"Construction skeleton max_dim=$(m.max_dim): homology refers to this truncated model; missing cofaces can create top-degree classes.")
    if m.family===:core && m.k_values!==nothing && length(m.k_values)<size(m.sites,1)
        push!(lines,"Sampled core depths: $(m.k_values). Unlisted depths use the retained step filtration, not a newly computed neighbour slice.")
    end
    m.radius_cutoff===nothing || push!(lines,"Inclusive radius cap $(m.radius_cutoff): surviving barcode ends are censored, not certified infinite.")
    if base.metadata.selected!==nothing && base.metadata.selected.source isa NamedTuple && haskey(base.metadata.selected.source,:radius_squared)
        push!(lines,"Selected cell squared birth radius: $(base.metadata.selected.source.radius_squared).")
    end
    push!(panels,_inspection_text_panel("Read the geometry and algebra together",lines))
    push!(panels,base.panels[2])
    return VisualizationSpec(:geometric_filtration;title="Explore radius, depth and the finite model",panels,
        metadata=merge(base.metadata,(;geometric=m,geometric_options=o,betti,selected_betti,diagram=diagrams,
            comparison,homology=o.invariant,field=o.field,degree=o.degree,parameter=o.complex.parameter,
            barcode_reused=_cached_barcode!==nothing,
            panel_columns=2,figure_size=(1250,1250),source_sampling=:display_only)))
end
