# Validated ambient certificates and paper-defined pruning subquotients.
const _CertificateInput=Union{Interleavings.InterleavingCertificate,Pruning.PruningResult}
available_visuals(::_CertificateInput)=(:certificate,)
_visual_request_keywords(::_CertificateInput,::Symbol)=(:vertex,:point,:edge,:side,:view,:matrix_limit,:budget)
_visual_request_cost(::_CertificateInput,::Symbol)=(;work=:validated_certificate,
    search=:none,pruning_comparison=:constructive,hover=:retained_data,timing=:not_measured)

function _prepare_certificate(x;budget=Interleavings.InterleavingBudget())
    owned=deepcopy(x)
    pruning=owned isa Pruning.PruningResult ? owned : nothing
    cert=pruning===nothing ? owned : Pruning.pruning_certificate(owned;budget)
    Interleavings.check_interleaving(cert;throw=true,budget)
    return (;certificate=cert,pruning,budget,edges=collect(FiniteFringe.cover_edges(cert.problem.sources[1].Q)))
end

function _certificate_options(c;vertex=nothing,point=nothing,edge=1,side=:forward,view=:composition,matrix_limit=(10,10))
    p=c.certificate.problem
    if vertex===nothing && point===nothing
        vertex=something(findfirst(eachindex(p.points)) do v
            a=p.at[v]
            any(!iszero,c.certificate.forward.comps[a]) || any(!iszero,c.certificate.reverse.comps[a])
        end,1)
    end
    view in (:composition,:naturality,:domains,:pruning) || throw(ArgumentError("Unknown certificate view."))
    side in (:forward,:reverse) || throw(ArgumentError("side must be :forward or :reverse."))
    view===:pruning && c.pruning===nothing && throw(ArgumentError("This certificate has no pruning subquotient."))
    if view===:domains
        p.left isa Interleavings.RectilinearModule || throw(ArgumentError("Finite-poset certificates have no ambient parameter domains."))
        length(p.metadata.translation)<=2 || throw(ArgumentError("Domain pictures support one or two parameters; use composition view for all coordinates."))
    end
    if point!==nothing
        p.left isa Interleavings.RectilinearModule || throw(ArgumentError("A finite-poset certificate has no ambient point coordinates."))
        length(point)==length(p.metadata.translation) || throw(ArgumentError("Wrong parameter dimension."))
        all(x->x isa Union{Integer,Rational,AbstractFloat} && isfinite(x),point) || throw(ArgumentError("Use finite exact point coordinates."))
        x=CoreModules.QQ.(point)
        p.metadata.domain===:integer && !all(isinteger,x) && throw(ArgumentError("Lattice points must be integral."))
        cuts=map((a,d)->sort!(unique([a;a.-d])),p.metadata.cuts,p.metadata.translation)
        label(a,t)=p.metadata.domain===:integer ? searchsortedlast(a,t) : (searchsortedlast(a,t),t in a)
        vertex=findfirst(y->all(j->label(cuts[j],x[j])==label(cuts[j],y[j]),eachindex(x)),p.points)
    end
    vertex isa Integer && !(vertex isa Bool) && 1<=vertex<=length(p.points) || throw(ArgumentError("Equation location must lie in 1:$(length(p.points))."))
    edge isa Integer && !(edge isa Bool) && 1<=edge<=max(1,length(c.edges)) || throw(ArgumentError("Naturality edge is out of range."))
    view===:naturality && isempty(c.edges) && throw(ArgumentError("This base has no cover edges; naturality is vacuous."))
    matrix_limit isa Tuple && length(matrix_limit)==2 && all(n->n isa Integer && !(n isa Bool) && n>0,matrix_limit) ||
        throw(ArgumentError("matrix_limit must contain two positive integers."))
    return (;vertex=Int(vertex),edge=Int(edge),side,view,matrix_limit)
end

function _certificate_diagram(labels,arrows,title;subtitle="")
    positions=[(0.0,0.0),(1.5,0.75),(3.0,0.0)]
    layers=AbstractVisualizationLayer[]
    for (a,b,text) in arrows
        _spectral_arrow_layers!(layers,positions[a],positions[b],:selected)
        p=(positions[a].+positions[b])./2
        push!(layers,TextLayer([text],[(p[1],p[2]+0.15)],_VisualRole(:foreground),13.0))
    end
    push!(layers,PointLayer(positions,_VisualRole(:selected),1.0,10.0))
    push!(layers,TextLayer(labels,[(p[1],p[2]-0.22) for p in positions],_VisualRole(:foreground),14.0))
    return VisualizationSpec(:certificate_diagram;title,subtitle,layers,
        axes=_default_axes_2d(;xlabel="",ylabel="",xlimits=(-0.8,3.8),ylimits=(-0.5,1.5),aspect=:auto,
            xticks=(Float64[],String[]),yticks=(Float64[],String[])),
        metadata=(;hide_decorations=true,coordinate_space=:schematic))
end

function _certificate_chart(c,v)
    p=c.certificate.problem
    if p.left isa Interleavings.RectilinearModule && length(p.metadata.translation)<=2
        points=[(Float64(x[1]),length(x)==1 ? 0.0 : Float64(x[2])) for x in p.points]
        if !all(x->all(isfinite,x),points) || !allunique(points)
            return _certificate_index_chart(p,v;reason="Exact coordinates exceed Float64 resolution; using schematic indices.")
        end
        layers=AbstractVisualizationLayer[PointLayer(points,_VisualRole(:muted),0.45,5.0)]
        x=p.points[v];d=p.metadata.translation
        selected=[(Float64(x[1]+k*d[1]),length(x)==1 ? 0.0 : Float64(x[2]+k*d[2])) for k in 0:2]
        zero_shift=all(iszero,d)
        if !all(x->all(isfinite,x),selected) || (!zero_shift && !allunique(selected))
            return _certificate_index_chart(p,v;reason="Translated coordinates exceed Float64 resolution; using schematic indices.")
        end
        for k in 1:2
            selected[k]==selected[k+1] || _spectral_arrow_layers!(layers,selected[k],selected[k+1],:selected)
        end
        push!(layers,PointLayer(selected,_VisualRole(:selected),1.0,10.0))
        labels=zero_shift ? ["x = x+d = x+2d"] : ["x","x+d","x+2d"]
        anchors=zero_shift ? selected[1:1] : selected
        push!(layers,TextLayer(labels,anchors,_VisualRole(:foreground),12.0))
        allpoints=[points;selected];lo=minimum(first,allpoints);hi=maximum(first,allpoints)
        bottom=minimum(last,allpoints);top=maximum(last,allpoints)
        margin=max((hi-lo)/12,0.4);ymargin=max((top-bottom)/12,0.4)
        return VisualizationSpec(:certificate_locations;title="Parameters and their translations",
            subtitle="Dots represent exact strata, including exterior rays; they are not support samples.",layers,
            axes=_default_axes_2d(;xlabel="parameter 1",ylabel=length(x)==1 ? "" : "parameter 2",
                xlimits=(lo-margin,hi+margin),ylimits=(bottom-ymargin,top+ymargin),aspect=:data),
            metadata=(;pick_targets=[(;vertex=i,point=points[i],label=string(p.points[i])) for i in eachindex(points)],
                exact_points=copy(p.points),translated_points=[x.+k.*d for k in 0:2],coordinate_space=:ambient,
                rounding=:float64_display_only))
    end
    return _certificate_index_chart(p,v)
end

function _certificate_index_chart(p,v;reason="Schematic location indices; all parameter coordinates remain in the readout.")
    # The finite label view is explicitly schematic; no higher coordinates vanish.
    points=[(Float64(i),0.0) for i in eachindex(p.points)]
    return VisualizationSpec(:certificate_locations;title="Exact equation locations",
        subtitle=reason,
        layers=AbstractVisualizationLayer[PointLayer(points,_VisualRole(:muted),0.7,6.0),
            PointLayer([points[v]],_VisualRole(:selected),1.0,12.0)],
        axes=_default_axes_2d(;xlabel="equation location",ylabel="",xlimits=(0.0,length(points)+1.0),ylimits=(-1.0,1.0)),
        metadata=(;coordinate_space=:schematic,pick_targets=[(;vertex=i,point=points[i],label=string(p.points[i])) for i in eachindex(points)]))
end

function _certificate_domain_panel(M,title,box,maxdim)
    # Exact open-cell dimensions plus separate singleton/edge markers. The
    # finite plotting window is display-only: the mathematical module is global.
    cuts=encoding_axes(M);n=length(cuts)
    n<=2 || throw(ArgumentError("Domain pictures support one or two parameters; use composition view for all coordinates."))
    axes=ntuple(j->sort!(unique([box[j][1];filter(x->box[j][1]<x<box[j][2],cuts[j]);box[j][2]])),n)
    layers=AbstractVisualizationLayer[];cells=NamedTuple[]
    all(a->all(isfinite,Float64.(a)) && allunique(Float64.(a)),axes) ||
        throw(ArgumentError("Exact domain boundaries collapse or overflow in Float64 display."))
    reps=map(a->[M.domain===:integer ? a[i] : (a[i]+a[i+1])/2 for i in 1:length(a)-1],axes)
    for idx in Iterators.product(ntuple(j->1:length(reps[j]),n)...)
        x=ntuple(j->reps[j][idx[j]],n);dim=Interleavings.ambient_dimension(M,x)
        rect=n==1 ? (Float64(axes[1][idx[1]]),-0.25,Float64(axes[1][idx[1]+1]),0.25) :
            (Float64(axes[1][idx[1]]),Float64(axes[2][idx[2]]),Float64(axes[1][idx[1]+1]),Float64(axes[2][idx[2]+1]))
        push!(cells,(;point=x,dimension=dim,bounds=rect))
        dim==0 || push!(layers,RectLayer([rect],_VisualRole(:selected),_VisualRole(:selected),0.15+0.65dim/max(1,maxdim),0.0))
    end
    # Boundary strata have their own dimension, which need not agree with
    # adjacent open cells. Never infer their support from the filled rectangles.
    pts=NTuple{2,Float64}[];labels=String[];boundary=NamedTuple[]
    mixed=map((a,r)->sort!(unique([a;r])),axes,reps)
    for x in Iterators.product(mixed...)
        any(j->x[j] in cuts[j],1:n) || continue
        dim=Interleavings.ambient_dimension(M,x)
        push!(pts,(Float64(x[1]),n==1 ? 0.0 : Float64(x[2])));push!(labels,string(dim))
        push!(boundary,(;point=x,dimension=dim))
    end
    push!(layers,PointLayer(pts,_VisualRole(:background),1.0,18.0))
    push!(layers,TextLayer(labels,pts,_VisualRole(:foreground),10.0))
    return VisualizationSpec(:certificate_domain;title,subtitle="Open cells: dimension shading. Boundary labels: exact dimensions.",layers,
        axes=_default_axes_2d(;xlabel="parameter 1",ylabel=n==1 ? "" : "parameter 2",
            xlimits=Float64.(box[1]),ylimits=n==1 ? (-0.5,0.5) : Float64.(box[2]),aspect=:data),
        metadata=(;cells,boundary,window=box,exterior=:continues_globally,dimension_max=maxdim,rounding=:float64_display_only))
end

function _certificate_spec(c;kwargs...)
    o=_certificate_options(c;kwargs...);cert=c.certificate;p=cert.problem
    v=o.vertex;a=p.at[v];b=p.shifted_at[v];field=p.sources[1].field
    eq=Interleavings.interleaving_equations(cert;vertex=v)
    panels=VisualizationSpec[_certificate_chart(c,v)]
    lines=String["Location $v: $(p.points[v])", "Translation: $(p.metadata.translation)",
        "Both composition equations hold.", "Naturality holds on every cover.",
        "Field: $(_inspection_field_label(field)). Matrices use stored bases."]
    selected=nothing
    if o.view===:composition
        left=o.side===:forward
        f=left ? cert.forward.comps[a] : cert.reverse.comps[a]
        g=left ? cert.reverse.comps[b] : cert.forward.comps[b]
        e=left ? eq.forward : eq.reverse
        shifted=p.metadata.domain===:finite_poset ? "T(x)" : "x+d"
        twice=p.metadata.domain===:finite_poset ? "T(T(x))" : "x+2d"
        names=left ? ["M(x)","N($shifted)","M($twice)"] : ["N(x)","M($shifted)","N($twice)"]
        push!(panels,_certificate_diagram(names,[(1,2,"first map"),(2,3,"shifted return"),(1,3,"structure map")],
            left ? "g($shifted) f(x) = M(x -> $twice)" : "f($shifted) g(x) = N(x -> $twice)";
            subtitle="Both routes give the same exact linear map."))
        for (A,title) in ((f,"First map"),(g,"Shifted return map"),(e.lhs,"Composite"),(e.rhs,"Canonical structure map"))
            push!(panels,_algebra_matrix_panel(A,title,field,o.matrix_limit))
        end
        selected=(;first=f,second=g,lhs=e.lhs,rhs=e.rhs,valid=e.lhs==e.rhs)
    elseif o.view===:naturality
        u,w=c.edges[o.edge];f=o.side===:forward ? cert.forward : cert.reverse
        spec=_morphism_naturality_spec(f;pair=(u,w),matrix_limit=o.matrix_limit)
        append!(panels,spec.panels)
        lhs=Modules.map_leq(f.cod,u,w)*f.comps[u];rhs=f.comps[w]*Modules.map_leq(f.dom,u,w)
        selected=(;pair=(u,w),lhs,rhs,valid=lhs==rhs)
        push!(lines,"Naturality cover: $u -> $w. This square uses the once-shifted target.")
    elseif o.view===:domains
        p.left isa Interleavings.RectilinearModule || throw(ArgumentError("Finite-poset certificates have no ambient parameter domains."))
        A,B=p.left,p.right;n=length(p.metadata.translation)
        n<=2 || throw(ArgumentError("Domain pictures support one or two parameters; use composition view for all coordinates."))
        objects=(A,B,Interleavings.parameter_shift(A,p.metadata.translation;budget=c.budget),Interleavings.parameter_shift(B,p.metadata.translation;budget=c.budget))
        box=ntuple(n) do j
            values=vcat([encoding_axes(M)[j] for M in objects]...)
            isempty(values) ? (CoreModules.QQ(-1),CoreModules.QQ(1)) : (minimum(values)-1,maximum(values)+1)
        end
        maxdim=maximum(maximum(Results.encoding_module(M).dims;init=0) for M in objects)
        for (M,title) in zip(objects,("M","N","M(d): x maps to M(x+d)","N(d): x maps to N(x+d)"))
            push!(panels,_certificate_domain_panel(M,title,box,maxdim))
        end
        push!(lines,"Domain panels share coordinates and dimension shading.",
            "The display window does not truncate the certificate.")
    else
        r=c.pruning;x=p.points[v];z=x.-r.epsilon
        loc=locate(r.refined,z);maps=Pruning.pruning_maps(r)
        for (map,title) in ((maps.inclusion,"I -> M"),(maps.kernel_inclusion,"K -> I"),(maps.projection,"I -> I/K"))
            push!(panels,_algebra_matrix_panel(map.comps[loc],title,field,o.matrix_limit))
        end
        selected=(;point=z,inclusion=maps.inclusion.comps[loc],kernel=maps.kernel_inclusion.comps[loc],projection=maps.projection.comps[loc])
        push!(lines,"Pruning at x is (I/K)(x-epsilon); the displayed subquotient is at $z.")
    end
    if c.pruning!==nothing
        g=Pruning.pruning_guarantee(c.pruning)
        push!(lines,"Pruning epsilon: $(g.epsilon). Maximum stalk dimension r=$(g.max_stalk_dimension).",
            "Proved refinement bound: $(g.bound) = 2r epsilon.",
            "The displayed interleaving maps realize that bound.",
            "Pruning-distance stability is not asserted.")
    elseif p.metadata.domain===:finite_poset
        push!(lines,"The shift is a supplied finite-poset endomap; it has no physical epsilon.")
    end
    push!(panels,_inspection_text_panel("Certificate and assumptions",lines))
    return VisualizationSpec(:certificate;title="Inspect the maps behind the guarantee",panels,
        metadata=(;selection=o,selected,equations=eq,provenance=Results.provenance(cert),
            panel_columns=2,figure_size=(1400,360cld(length(panels),2)),
            guarantee=c.pruning===nothing ? :exact_interleaving : Pruning.pruning_guarantee(c.pruning),
            verified=true,scope=:global_or_declared_finite_translation))
end
function _visual_spec(x::_CertificateInput,::Symbol;budget=Interleavings.InterleavingBudget(),kwargs...)
    _certificate_spec(_prepare_certificate(x;budget);kwargs...)
end

function _append_visual_request_issues!(issues::Vector{String},x::_CertificateInput,::Symbol;kwargs...)
    view=get(kwargs,:view,:composition);side=get(kwargs,:side,:forward)
    view in (:composition,:naturality,:domains,:pruning) || push!(issues,"Unknown certificate view.")
    side in (:forward,:reverse) || push!(issues,"side must be :forward or :reverse.")
    for key in (:vertex,:edge)
        value=get(kwargs,key,1)
        value isa Integer && !(value isa Bool) && value>0 || push!(issues,"$key must be a positive integer.")
    end
    get(kwargs,:budget,Interleavings.InterleavingBudget()) isa Interleavings.InterleavingBudget || push!(issues,"Use an InterleavingBudget for certificate construction.")
    view===:pruning && !(x isa Pruning.PruningResult) && push!(issues,"This certificate has no pruning subquotient.")
    source=x isa Pruning.PruningResult ? x.source : x.problem.left
    ambient=source isa Interleavings.RectilinearModule
    view===:domains && (!ambient || length(encoding_axes(source))>2) &&
        push!(issues,"Domain pictures require one or two ambient parameters.")
    haskey(kwargs,:point) && kwargs[:point]!==nothing && !ambient &&
        push!(issues,"A finite-poset certificate has no ambient point coordinates.")
    limit=get(kwargs,:matrix_limit,(10,10))
    limit isa Tuple && length(limit)==2 && all(n->n isa Integer && !(n isa Bool) && n>0,limit) ||
        push!(issues,"matrix_limit must contain two positive integers.")
    issues
end
