# Finite transport diagrams retain actual maps, not inferred identifications.
const _TransportObject = Union{EncodingMap,ChangeOfPosets.JointEncodingResult,
    CommonRefinementTranslationResult,ModuleTranslationResult}
const _TRANSPORT_CATEGORY = "Finite Rep(P) only. No ambient identification or Ext/Tor invariance is asserted."

available_visuals(pi::EncodingMap) = (:order_map, _coordinate_map_visuals(pi)...)
available_visuals(::ChangeOfPosets.JointEncodingResult) = (:joint_encoding,)
available_visuals(::CommonRefinementTranslationResult) = (:common_refinement,)
available_visuals(r::ModuleTranslationResult) = translation_map(r) isa EncodingMap ?
    (:change_of_poset, (:pushforward_overlay in _coordinate_map_visuals(translation_map(r)) ? (:pushforward_overlay,) : ())...) : ()

_transport_keywords() = (:sources,:joint,:comparison,:direction,:vertex,:pair,:point,
    :side,:label,:view,:box,:matrix_limit,:max_vertices,:max_pairs)
_visual_request_keywords(::_TransportObject, k::Symbol) =
    k in (:regions,:region_labels,:pushforward_overlay) ? () : _transport_keywords()
_visual_request_cost(::_TransportObject,::Symbol) = (;work=:finite_transport_inspection,
    preparation=:validate_modules_and_cover_maps, lazy_sources=:materialized_when_supplied,
    selection=:selected_structure_maps_and_matrices, hover=:retained_data,
    kan_extensions=:never_computed_by_view, timing=:not_measured)

function _transport_base(x)
    x isa EncodingMap && return source_poset(x)
    x isa ModuleTranslationResult && return source_poset(translation_map(x))
    return common_poset(x)
end
function _transport_options(x; vertex=nothing,pair=nothing,point=nothing,side=:base,label=nothing,
        view=:overview,matrix_limit=(12,12),max_vertices=200,max_pairs=2500,kwargs...)
    issues=String[]
    for k in keys(kwargs)
        k in _transport_keywords() || push!(issues,"Unknown transport option $k.")
    end
    P=_transport_base(x)
    _algebra_selection_issues!(issues,P;vertex,pair,matrix_limit)
    count(!isnothing,(vertex,pair,point,label))<=1 || push!(issues,"Select only one vertex, pair, point or side label.")
    side in (:base,:left,:right) || push!(issues,"side must be :base, :left or :right.")
    label === nothing || (label isa Integer && !(label isa Bool) && label>0) || push!(issues,"label must be a positive finite vertex ID.")
    view in (:overview,:square) || push!(issues,"view must be :overview or :square.")
    for (name,value) in ((:max_vertices,max_vertices),(:max_pairs,max_pairs))
        value isa Integer && !(value isa Bool) && value>0 || push!(issues,"$name must be a positive integer.")
    end
    point === nothing || _drawing_point(point)
    get(kwargs,:direction,:left_to_right) in (:left_to_right,:right_to_left) || push!(issues,"direction must be :left_to_right or :right_to_left.")
    get(kwargs,:box,nothing) === nothing || _visual_box_2d(kwargs[:box])
    isempty(issues) || throw(ArgumentError(join(issues," ")))
    return (;vertex,pair=pair===nothing ? nothing : Tuple(Int.(pair)),point,side,label,view,
        matrix_limit=Tuple(Int.(matrix_limit)),max_vertices=Int(max_vertices),max_pairs=Int(max_pairs))
end
function _append_visual_request_issues!(issues::Vector{String},x::_TransportObject,k::Symbol;kwargs...)
    k in (:regions,:region_labels,:pushforward_overlay) && return issues
    _transport_options(x;kwargs...)
    return issues
end

_transport_module(x::Modules.PModule)=x
_transport_module(x::EncodingResult)=Results.encoding_module(x)
_transport_module(::Nothing)=nothing
function _transport_module_equal(A,B)
    A.Q === B.Q && A.field==B.field && A.dims==B.dims || return false
    return all(_algebra_equation(A.field,Modules.structure_map(A;source=p,target=q),
        Modules.structure_map(B;source=p,target=q)).valid for (p,q) in FiniteFringe.cover_edges(A.Q))
end
function _transport_validate_module(M,P)
    M===nothing && return
    M.Q === P || throw(ArgumentError("Module and displayed poset must be identical; labels or dimensions alone are insufficient."))
    Modules.check_module(M).valid || throw(ArgumentError("The supplied module violates its finite-poset relations."))
end

# The preparation boundary checks each cover once. Session navigation reuses it.
function _prepare_transport(x; sources=nothing,joint=nothing,comparison=nothing,
        direction=:left_to_right,box=nothing,max_vertices=200,max_pairs=2500,kwargs...)
    base=_transport_base(x)
    objects=(nothing,nothing);mods=(nothing,nothing);refinement=:order_map
    if x isa Union{CommonRefinementTranslationResult,ChangeOfPosets.JointEncodingResult}
        proj=projection_maps(x);maps=(proj.left,proj.right)
        if x isa CommonRefinementTranslationResult
            t=ChangeOfPosets.translated_modules(x);mods=(t.left,t.right);refinement=x.refinement
        else
            joint===nothing || joint===x || throw(ArgumentError("A joint encoding is already its own finite-source witness."))
            joint=x;refinement=:realized_joint_image
        end
        if sources!==nothing
            (sources isa Tuple || sources isa NamedTuple) && length(sources)==2 || throw(ArgumentError("sources must contain the left and right original modules or encoding results."))
            objects=Tuple(sources)
        end
    else
        pi=x isa EncodingMap ? x : translation_map(x)
        pi isa EncodingMap || throw(ArgumentError("Finite transport inspection requires an EncodingMap."))
        maps=(EncodingMap(base,base,collect(1:nvertices(base))),pi)
        if x isa ModuleTranslationResult
            sources===nothing || throw(ArgumentError("Translation results retain their source; sources is only for refinement results or order maps."))
            old=source_result(x)
            old===nothing && throw(ArgumentError("Translation result has no retained source module."))
            new=Results.translated_module(x)
            cls=encoding_map(x)
            newobj=cls===nothing ? new : EncodingResult(new.Q,new,cls)
            objects=translation_kind(x)===:restriction ? (newobj,old) : (old,newobj)
            refinement=translation_kind(x)
        elseif sources!==nothing
            (sources isa Tuple || sources isa NamedTuple) && length(sources)==2 || throw(ArgumentError("sources must contain source and target modules or encoding results."))
            objects=Tuple(sources)
        end
    end
    posets=(target_poset(maps[1]),target_poset(maps[2]))
    all(P->nvertices(P)<=max_vertices,(base,posets...)) || throw(ArgumentError("A displayed poset exceeds max_vertices=$max_vertices; increase the explicit budget."))
    for pi in maps
        source_poset(pi)===base || throw(ArgumentError("Projection sources must be the actual common poset."))
        ChangeOfPosets.check_monotone_map(pi;throw=true)
    end
    originals=map(_transport_module,objects)
    for i in 1:2
        _transport_validate_module(originals[i],posets[i]);_transport_validate_module(mods[i],base)
    end
    pullbacks=map(1:2) do i
        original=originals[i]
        original===nothing && return mods[i]
        pulled=ChangeOfPosets.restriction(maps[i],original)
        mods[i]===nothing || _transport_module_equal(pulled,mods[i]) ||
            throw(ArgumentError("Translated module $i does not equal the pullback of the supplied original maps."))
        return mods[i]===nothing ? pulled : mods[i]
    end
    if all(!isnothing,pullbacks)
        pullbacks[1].field==pullbacks[2].field || throw(ArgumentError("Comparison coefficient fields must agree."))
    end
    pairs=collect(zip(region_map(maps[1]),region_map(maps[2])))
    realized=nothing
    if joint!==nothing
        joint isa ChangeOfPosets.JointEncodingResult || throw(ArgumentError("joint must be a JointEncodingResult finite-source witness."))
        ChangeOfPosets.check_joint_encoding(joint;throw=true)
        jp=projection_maps(joint)
        target_poset(jp.left)===posets[1] && target_poset(jp.right)===posets[2] ||
            throw(ArgumentError("Joint witness targets must be the two displayed posets."))
        realized=Set(zip(region_map(jp.left),region_map(jp.right)))
        if refinement===:realized_joint_image
            length(unique(pairs))==length(pairs) && Set(pairs)==realized ||
                throw(ArgumentError("The realized-refinement labels disagree with the supplied joint witness."))
            all(leq(base,i,j)==(leq(posets[1],pairs[i][1],pairs[j][1]) &&
                leq(posets[2],pairs[i][2],pairs[j][2])) for i in eachindex(pairs),j in eachindex(pairs)) ||
                throw(ArgumentError("A realized refinement must carry the induced product order of its witnessed pairs."))
        end
    end
    f=comparison;validation=nothing
    if f!==nothing
        f isa Modules.PMorphism || throw(ArgumentError("comparison must be a supplied PMorphism."))
        all(!isnothing,pullbacks) || throw(ArgumentError("A comparison requires both actual modules."))
        a,b=direction===:left_to_right ? (1,2) : (2,1)
        _transport_module_equal(f.dom,pullbacks[a]) && _transport_module_equal(f.cod,pullbacks[b]) ||
            throw(ArgumentError("Comparison endpoints must agree with the actual pullback modules, including maps; change direction if needed."))
        issues=String[];_append_visual_request_issues!(issues,f,:morphism_inspector)
        isempty(issues) || throw(ArgumentError(join(issues," ")))
        v=_algebra_morphism_validation(f)
        ranks=[FieldLinAlg.rank(f.dom.field,Modules.component(f,q)) for q in 1:nvertices(base)]
        iso=v.valid && all(q->ranks[q]==f.dom.dims[q]==f.cod.dims[q],1:nvertices(base))
        validation=(;v...,component_ranks=ranks,isomorphism=iso)
    end
    geometry=map(objects) do o
        o isa EncodingResult && _inspection_has_geometry(o) ?
            _region_geometry_2d(_inspection_classifier(encoding_map(o));box) : nothing
    end
    return (;base,maps,posets,objects,originals,pullbacks=Tuple(pullbacks),pairs,refinement,
        joint,realized,comparison=f,direction,validation,box,geometry,max_pairs,
        hasse=map(_prepare_hasse,(base,posets...)))
end

function _transport_selection(c;vertex=nothing,pair=nothing,point=nothing,side=:base,label=nothing,kwargs...)
    candidates=Int[];query_labels=nothing
    if point!==nothing
        all(o->o isa EncodingResult && _inspection_has_geometry(o),c.objects) ||
            throw(ArgumentError("A parameter point requires two retained planar encoding classifiers."))
        query_labels=map(o->_visual_locate(_inspection_classifier(encoding_map(o)),Tuple(point)),c.objects)
        candidates=findall(==(query_labels),c.pairs)
    elseif label!==nothing
        P=side===:base ? c.base : c.posets[side===:left ? 1 : 2]
        label<=nvertices(P) || throw(ArgumentError("Selected label is outside its finite poset."))
        candidates=side===:base ? [Int(label)] : findall(==(label),region_map(c.maps[side===:left ? 1 : 2]))
    elseif vertex!==nothing
        candidates=[Int(vertex)]
    end
    selected=length(candidates)==1 ? only(candidates) : nothing
    return (;vertex=selected,pair,point,side,label,candidates,query_labels,
        status=point===nothing && label===nothing ? :finite_selection :
            isempty(candidates) ? :no_common_label : length(candidates)==1 ? :unique : :multiple_preimages)
end

function _transport_hasse(c,i,s)
    P=i==0 ? c.base : c.posets[i]
    ids=i==0 ? collect(1:nvertices(P)) : region_map(c.maps[i])
    M=i==0 ? c.pullbacks[1] : c.originals[i]
    vertex=s.vertex===nothing ? nothing : ids[s.vertex]
    pair=s.pair===nothing ? nothing : (ids[s.pair[1]],ids[s.pair[2]])
    spec=_hasse_spec(P;vertex,pair,dims=M===nothing ? nothing : M.dims,prepared=c.hasse[i+1],
        field_label=M===nothing ? nothing : _inspection_field_label(M.field))
    selected=unique(ids[s.candidates])
    labeled=nvertices(P)<=20 ? collect(1:nvertices(P)) :
        sort!(unique(vcat(selected,pair===nothing ? Int[] : collect(pair))))
    layers=AbstractVisualizationLayer[
        l isa TextLayer && length(l.labels)==nvertices(P) ?
            TextLayer(l.labels[labeled],l.positions[labeled],l.color,l.textsize) : l for l in spec.layers]
    subtitle=spec.subtitle
    nvertices(P)>20 && (subtitle*="\nLabels on selection only; all vertices retained for hover")
    isempty(selected) || push!(layers,PointLayer(spec.metadata.positions[selected],_VisualRole(:selected),0.8,16.0))
    side=i==0 ? :base : i==1 ? :left : :right
    targets=[(;point=spec.metadata.positions[q],side,label=q) for q in 1:nvertices(P)]
    return VisualizationSpec(:transport_hasse;title=i==0 ? (M===nothing ? "Common comparison base" : "Common base (left stalk dimensions)") : i==1 ? "Left finite poset" : "Right finite poset",
        subtitle,layers,axes=spec.axes,legend=spec.legend,
        metadata=merge(spec.metadata,(;side,pick_targets=targets,selected_ids=selected,labeled_vertices=labeled)))
end

function _transport_pair_panel(c,s)
    n,m=nvertices.(c.posets)
    exhaustive=big(n)*m<=c.max_pairs
    entries=exhaustive ? [(i,j) for j in 1:m for i in 1:n] : sort!(unique(c.pairs))
    layers=AbstractVisualizationLayer[]
    isempty(entries) && push!(layers,TextLayer(["No label pairs"],[(1.0,1.0)],_VisualRole(:muted),14.0))
    for pair in entries
        included=pair in c.pairs
        known=c.realized!==nothing
        actual=known && pair in c.realized
        color=known ? (actual ? _VisualRole(:source) : _VisualRole(:unrepresented)) :
            (included ? _VisualRole(:source) : _VisualRole(:muted))
        push!(layers,RectLayer([(pair[1]-0.4,pair[2]-0.4,pair[1]+0.4,pair[2]+0.4)],color,
            included ? _VisualRole(:foreground) : color,included ? 0.7 : 0.2,included ? 1.2 : 0.0))
    end
    selected=unique(c.pairs[s.candidates])
    isempty(selected) || push!(layers,PointLayer(NTuple{2,Float64}[(Float64(i),Float64(j)) for (i,j) in selected],_VisualRole(:selected),1.0,12.0))
    return VisualizationSpec(:joint_labels;title="Pairs of finite labels",
        subtitle=(c.realized===nothing ? "Outlined: in this comparison base; realization not certified" :
            "Filled: realized on supplied finite source; pale: unrealized there") *
            (exhaustive ? "" : "\nCartesian background omitted by max_pairs budget"),layers,
        axes=_default_axes_2d(xlabel="left label (not a coordinate)",ylabel="right label (not a coordinate)",
            xlimits=(0.4,max(n,1)+0.6),ylimits=(0.4,max(m,1)+0.6)),
        metadata=(;pairs=copy(c.pairs),realized_pairs=c.realized===nothing ? nothing : sort!(collect(c.realized)),
            exhaustive,cartesian_count=big(n)*m,realization_scope=c.realized===nothing ? :not_certified : :supplied_finite_source))
end

function _transport_region(c,i,s)
    geo=c.geometry[i];geo===nothing && return nothing
    selected=s.pair===nothing ? s.candidates : collect(s.pair)
    labels=region_map(c.maps[i])[selected]
    colors=Dict(q=>(q==0 ? _VisualRole(:unrepresented) : q in labels ? _VisualRole(:source) : _VisualRole(:absent)) for q in geo.region_ids)
    original=_visual_spec(_inspection_classifier(encoding_map(c.objects[i])),:region_labels;box=c.box,prepared_geometry=geo)
    layers=_region_geometry_layers(geo;colors)
    append!(layers,filter(l->l isa TextLayer,original.layers))
    if s.point!==nothing
        push!(layers,PointLayer([_drawing_point(s.point)],_VisualRole(:selected),1.0,12.0))
    end
    warnings=original.metadata.warnings
    subtitle="Selected fibers; solid/included, dashed/excluded, dotted/window cut"
    geo.geometry_kind===:nearest_lattice_tiles && (subtitle*="\nInteger fibers: nearest-lattice tiles; ties round-to-even")
    isempty(warnings) || (subtitle*="\nDrawing precision warning: inspect exact geometry metadata")
    return VisualizationSpec(:transport_regions;title=i==1 ? "Left parameter regions" : "Right parameter regions",
        subtitle,
        layers,axes=geo.axes,
        metadata=(;geometry=geo,selected_labels=copy(labels),warnings=original.metadata.warnings,side=i==1 ? :left : :right))
end

function _transport_spec(c,s;view=:overview,matrix_limit=(12,12))
    panels=VisualizationSpec[];square=nothing;matrices=NamedTuple[]
    if view===:overview
        append!(panels,[_transport_hasse(c,i,s) for i in 0:2])
        push!(panels,_transport_pair_panel(c,s))
        for i in 1:2
            region=_transport_region(c,i,s);region===nothing || push!(panels,region)
        end
    end
    if s.pair!==nothing
        p,q=s.pair
        if c.comparison!==nothing
            natural=_morphism_naturality_spec(c.comparison;pair=s.pair,matrix_limit)
            square=natural.metadata.naturality
            view===:square && append!(panels,natural.panels)
        end
        if leq(c.base,p,q)
            for i in 1:2
                M=c.pullbacks[i];M===nothing && continue
                A=copy(Modules.structure_map(M;source=p,target=q))
                a,b=region_map(c.maps[i])[[p,q]]
                push!(matrices,(;side=i==1 ? :left : :right,pair=(p,q),projected_pair=(a,b),matrix=A))
                c.comparison===nothing && push!(panels,_algebra_matrix_panel(A,"$(i==1 ? "Left" : "Right") map $a -> $b",M.field,matrix_limit))
            end
        end
    elseif s.vertex!==nothing && c.comparison!==nothing
        q=s.vertex;f=c.comparison
        push!(panels,_algebra_matrix_panel(copy(Modules.component(f,q)),"Comparison component at $q",f.dom.field,matrix_limit))
    end
    lines=String[_TRANSPORT_CATEGORY,"Construction: $(c.refinement).","Order maps: q -> (left(q), right(q))."]
    if s.pair!==nothing
        p,q=s.pair
        push!(lines,"Selected $p -> $q maps to $(c.pairs[p]) -> $(c.pairs[q]).")
        leq(c.base,p,q) || push!(lines,"No forward map on the comparison base; this is not a zero matrix.")
    elseif s.vertex!==nothing
        push!(lines,"Selected common label $(s.vertex) projects to $(c.pairs[s.vertex]).")
        for i in 1:2
            M=c.pullbacks[i];M===nothing || push!(lines,"$(i==1 ? "Left" : "Right") stalk dimension: $(M.dims[s.vertex]).")
        end
    elseif s.status!==:finite_selection
        push!(lines,"Selection: $(s.status). Common candidates: $(s.candidates).")
    end
    if c.validation===nothing
        push!(lines,"No comparison morphism supplied; equal dimensions do not identify modules.")
    else
        v=c.validation
        push!(lines,v.valid ? (v.isomorphism ? "Supplied comparison: finite natural isomorphism." : "Supplied comparison: natural, not invertible at every stalk.") : "Supplied comparison fails naturality: "*join(v.issues," "))
        push!(lines,"Direction: $(c.direction). Cover equations checked in stored coordinates.")
        square===nothing || !square.defined || push!(lines,_algebra_equation_text(square))
    end
    view===:square && s.pair===nothing && push!(lines,"Select pair=(p,q) to inspect a square and its actual matrices.")
    c.realized===nothing || push!(lines,"Realization is certified only on the supplied finite source, not an unenumerated ambient space.")
    push!(panels,_inspection_text_panel("What this comparison establishes",lines))
    layout=if view===:square
        _comparison_layout(panels;hero=first(panels).kind===:naturality_square)
    else
        # The semantic readout spans the whole last row, rather than leaving
        # two empty columns and wrapping a certificate into a narrow card.
        n=length(panels)-1;rows=cld(n,3)
        positions=[(cld(i,3):cld(i,3),mod1(i,3):mod1(i,3)) for i in 1:n]
        push!(positions,(rows+1:rows+1,1:3))
        (;panel_columns=3,panel_positions=positions,panel_row_weights=[fill(380,rows);260],
            figure_size=(1560,120+380*rows+260))
    end
    return VisualizationSpec(:change_of_poset;title="Refinement and change of poset",panels,
        metadata=(;category=:finite_poset_representations,ambient_identification=:not_asserted,
            derived_invariance=:not_asserted,refinement=c.refinement,selection=s,view,
            projection_labels=map(pi->copy(region_map(pi)),c.maps),comparison_validation=c.validation,
            naturality=square,selected_maps=matrices,realization_scope=c.realized===nothing ? :not_certified : :supplied_finite_source,
            layout...))
end

function _visual_spec(x::_TransportObject,k::Symbol;kwargs...)
    k in (:regions,:region_labels,:pushforward_overlay) && return _coordinate_map_spec(x isa ModuleTranslationResult ? translation_map(x) : x,k;kwargs...)
    o=_transport_options(x;kwargs...);c=_prepare_transport(x;kwargs...)
    s=_transport_selection(c;o...)
    return _algebra_with_title(_transport_spec(c,s;view=o.view,matrix_limit=o.matrix_limit),"Refinement and change of poset";kind=k)
end
