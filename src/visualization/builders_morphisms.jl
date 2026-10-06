# Morphisms, naturality and exact sequences on a declared common finite poset.

available_visuals(::Modules.PMorphism) =
    (:morphism_inspector, :naturality, :kernel_image_cokernel, :morphism_support)
available_visuals(::AbelianCategories.ShortExactSequence) = (:exact_sequence,)

_visual_request_keywords(::Modules.PMorphism, kind::Symbol) =
    kind === :morphism_support ? (:encoding, :box) :
    kind === :naturality ? (:pair, :matrix_limit) :
    kind === :kernel_image_cokernel ? (:vertex, :matrix_limit) :
    (:vertex, :pair, :matrix_limit)
_visual_request_keywords(::AbelianCategories.ShortExactSequence, ::Symbol) = (:vertex, :matrix_limit)

function _algebra_with_title(spec, title; subtitle=spec.subtitle, kind=spec.kind)
    return VisualizationSpec(kind; title, subtitle, layers=spec.layers,
        panels=spec.panels, axes=spec.axes, legend=spec.legend,
        interaction=spec.interaction, metadata=spec.metadata)
end

function _algebra_equation(field, lhs, rhs)
    size(lhs) == size(rhs) || throw(DimensionMismatch("Equation matrices must have the same shape."))
    if field isa CoreModules.RealField
        residual = maximum(abs, lhs-rhs; init=zero(field.atol))
        scale = max(maximum(abs, lhs; init=zero(field.atol)),
                    maximum(abs, rhs; init=zero(field.atol)))
        tolerance = field.atol + field.rtol * scale
        return (; valid=isfinite(residual) && residual <= tolerance, exact=false, residual, tolerance)
    end
    return (; valid=lhs == rhs, exact=true, residual=nothing, tolerance=nothing)
end

function _algebra_equation_text(e)
    e.exact && return e.valid ? "Equal over the coefficient field" : "Unequal over the coefficient field"
    return "Residual $(e.residual); tolerance $(e.tolerance)\n" *
        (e.valid ? "Agreement within tolerance" : "Disagreement at this tolerance")
end

function _algebra_matrix_panel(A, title, field, limit; subtitle="", metadata=NamedTuple(),
                               source="source", target="target")
    field_text = _inspection_field_label(field)
    return _presentation_matrix_panel(A, title,
        ["$target $i" for i in axes(A,1)], ["$source $j" for j in axes(A,2)], limit;
        subtitle=isempty(subtitle) ? field_text : "$subtitle\n$field_text",
        metadata=merge((; field, basis_convention=:stored_coordinates), metadata))
end

function _algebra_selection_issues!(issues, P; vertex=nothing, pair=nothing, matrix_limit=(12,12))
    n = nvertices(P)
    valid_vertex(v) = v isa Integer && !(v isa Bool) && 1 <= v <= n
    vertex === nothing || valid_vertex(vertex) || push!(issues, "vertex must be in 1:$n.")
    pair === nothing || ((pair isa Tuple || pair isa AbstractVector) && length(pair)==2 &&
        all(valid_vertex, pair)) || push!(issues, "pair must contain two vertices in 1:$n.")
    vertex === nothing || pair === nothing || push!(issues, "Select vertex or pair, not both.")
    (matrix_limit isa Tuple || matrix_limit isa AbstractVector) && length(matrix_limit)==2 &&
        all(n -> n isa Integer && !(n isa Bool) && n > 0, matrix_limit) ||
        push!(issues, "matrix_limit must contain two positive integers.")
    return issues
end

function _append_visual_request_issues!(issues::Vector{String}, f::Modules.PMorphism, kind::Symbol; kwargs...)
    if kind === :morphism_support
        enc = get(kwargs, :encoding, nothing)
        if !(enc isa EncodingResult)
            push!(issues, "morphism_support requires encoding=enc with a retained classifier.")
        else
            Results.encoding_poset(enc) === f.dom.Q ||
                push!(issues, "The classifier must have the identical finite base of the morphism; matching label numbers is insufficient.")
            _inspection_has_geometry(enc) || push!(issues, "This classifier has no supported planar region geometry.")
        end
        get(kwargs,:box,nothing) === nothing || _visual_box_2d(kwargs[:box])
    else
        _algebra_selection_issues!(issues, f.dom.Q; vertex=get(kwargs,:vertex,nothing),
            pair=get(kwargs,:pair,nothing), matrix_limit=get(kwargs,:matrix_limit,(12,12)))
        kind === :naturality && get(kwargs,:pair,nothing) === nothing &&
            push!(issues, "naturality requires pair=(source_vertex,target_vertex).")
        kind === :kernel_image_cokernel && get(kwargs,:vertex,nothing) === nothing &&
            push!(issues, "kernel_image_cokernel requires vertex=q for the inclusion/projection matrices.")
    end
    f.dom.Q === f.cod.Q || push!(issues, "Source and target must have the identical finite base.")
    f.dom.field == f.cod.field || push!(issues, "Source and target fields must agree.")
    n=nvertices(f.dom.Q)
    length(f.comps)==n || push!(issues,"Expected one component matrix per vertex.")
    for q in 1:min(n,length(f.comps))
        size(f.comps[q])==(f.cod.dims[q],f.dom.dims[q]) ||
            push!(issues,"Component $q does not have target-by-source dimensions.")
    end
    return issues
end
function _append_visual_request_issues!(issues::Vector{String}, ses::AbelianCategories.ShortExactSequence, ::Symbol; kwargs...)
    _algebra_selection_issues!(issues, ses.B.Q; vertex=get(kwargs,:vertex,nothing),
        matrix_limit=get(kwargs,:matrix_limit,(12,12)))
    ses.i.dom === ses.A && ses.i.cod === ses.B && ses.p.dom === ses.B && ses.p.cod === ses.C ||
        push!(issues, "Sequence modules must be the endpoints of its inclusion and projection.")
    _append_visual_request_issues!(issues,ses.i,:morphism_inspector)
    _append_visual_request_issues!(issues,ses.p,:morphism_inspector)
    return issues
end

# These diagnostics retain equations, including failures, rather than turn a
# constructor's name or dimensions into a certificate. No rendering is involved.
function _algebra_morphism_validation(f)
    field = f.dom.field
    issues = String[]
    equations = NamedTuple[]
    Modules.check_module(f.dom).valid || push!(issues,"The source module is invalid.")
    Modules.check_module(f.cod).valid || push!(issues,"The target module is invalid.")
    for (p,q) in FiniteFringe.cover_edges(f.dom.Q)
        lhs = Modules.structure_map(f.cod;source=p,target=q) * Modules.component(f,p)
        rhs = Modules.component(f,q) * Modules.structure_map(f.dom;source=p,target=q)
        e = _algebra_equation(field,lhs,rhs)
        push!(equations, (; pair=(p,q), e...))
        e.valid || push!(issues,"Naturality fails on cover ($p,$q).")
    end
    return (; valid=isempty(issues), issues, equations,
        exact=!(field isa CoreModules.RealField), scope=:all_cover_squares)
end

function _algebra_module_panel(M, title; vertex=nothing, pair=nothing, prepared=nothing)
    return _algebra_with_title(_hasse_spec(M.Q; dims=M.dims, vertex, pair, prepared,
        field_label=_inspection_field_label(M.field)), title)
end

function _algebra_square_diagram(p,q,ds,dt; title="Naturality square", labels=nothing, edge_labels=nothing)
    positions = [(0.0,1.6),(4.0,1.6),(0.0,0.0),(4.0,0.0)]
    names = labels === nothing ? ["M($p)  dim $(ds[1])", "N($p)  dim $(dt[1])",
                                  "M($q)  dim $(ds[2])", "N($q)  dim $(dt[2])"] : labels
    segments = NTuple{4,Float64}[]
    heads = Vector{NTuple{2,Float64}}[]
    for (a,b) in ((1,2),(3,4),(1,3),(2,4))
        _hasse_arrow!(segments,heads,positions[a],positions[b]; start_gap=0.48, end_gap=0.48)
    end
    layers = AbstractVisualizationLayer[
        SegmentLayer(segments,_VisualRole(:edge),1.0,1.6),
        PolygonLayer(heads,_VisualRole(:edge),_VisualRole(:edge),1.0,0.0),
        TextLayer(names,[(x-0.45,y+0.12) for (x,y) in positions],_VisualRole(:foreground),15.0),
        TextLayer(edge_labels === nothing ? ["f($p)","f($q)","M($p \u2264 $q)","N($p \u2264 $q)"] : edge_labels,
            [(1.8,1.85),(1.8,0.25),(-1.2,0.8),(4.15,0.8)],_VisualRole(:muted),13.0)]
    return VisualizationSpec(:naturality_square; title,
        subtitle="Both routes take a vector from the upper left to the lower right",layers,
        axes=_default_axes_2d(;xlabel="",ylabel="",xlimits=(-1.4,6.0),ylimits=(-0.35,2.35)),
        metadata=(;hide_decorations=true,layout=:schematic,semantic=:commuting_square))
end

function _morphism_naturality_spec(f; pair, matrix_limit=(12,12))
    p,q = Int.(pair)
    field = f.dom.field
    if !leq(f.dom.Q,p,q)
        relation = leq(f.dom.Q,q,p) ? :reverse_comparable : :incomparable
        return VisualizationSpec(:naturality; title="No naturality square in this direction",
            panels=[_algebra_module_panel(f.dom,"Source M";pair=(p,q)),
                _inspection_text_panel("No forward structure map",["Vertices $p and $q: $relation.",
                    "A missing square is not a square of zero maps."])],
            metadata=(;field,category=:finite_poset_representations,
                naturality=(;defined=false,relation),panel_columns=2,figure_size=(1120,580)))
    end
    A = copy(Modules.structure_map(f.dom;source=p,target=q))
    B = copy(Modules.structure_map(f.cod;source=p,target=q))
    Fp,Fq = copy(Modules.component(f,p)),copy(Modules.component(f,q))
    left,right = B*Fp,Fq*A
    equation = _algebra_equation(field,left,right)
    panels = VisualizationSpec[_algebra_square_diagram(p,q,(f.dom.dims[p],f.dom.dims[q]),
        (f.cod.dims[p],f.cod.dims[q]))]
    for (matrix,title,source,target) in ((Fp,"f($p)","M($p)","N($p)"),
            (Fq,"f($q)","M($q)","N($q)"),(A,"M($p \u2264 $q)","M($p)","M($q)"),
            (B,"N($p \u2264 $q)","N($p)","N($q)"),
            (left,"N($p \u2264 $q) f($p)","M($p)","N($q)"),
            (right,"f($q) M($p \u2264 $q)","M($p)","N($q)"))
        push!(panels,_algebra_matrix_panel(matrix,title,field,matrix_limit;source,target))
    end
    layout = _comparison_layout(panels; hero=true)
    return VisualizationSpec(:naturality; title="Naturality at $p \u2264 $q",
        subtitle=_algebra_equation_text(equation), panels,
        metadata=(; field,category=:finite_poset_representations,selection=(;pair=(p,q)),
            naturality=(;defined=true,source_map=A,target_map=B,source_component=Fp,
                target_component=Fq,left,right,equation...),
            validation=(;equation...,scope=:selected_square),layout...))
end

function _morphism_visual_spec(f,kind; vertex=nothing,pair=nothing,matrix_limit=(12,12))
    kind === :naturality && return _morphism_naturality_spec(f;pair,matrix_limit)
    kind === :kernel_image_cokernel && return _morphism_subquotient_spec(f;vertex,matrix_limit)
    pair !== nothing && return _algebra_with_title(
        _morphism_naturality_spec(f;pair,matrix_limit),"Inspecting a module morphism";kind=:morphism_inspector)
    field=f.dom.field
    prepared=_prepare_hasse(f.dom.Q)
    panels=VisualizationSpec[_algebra_module_panel(f.dom,"Source M";vertex,prepared),
        _algebra_module_panel(f.cod,"Target N";vertex,prepared)]
    validation=_algebra_morphism_validation(f)
    status=validation.valid ? (validation.exact ? "All cover squares commute." : "All cover residuals satisfy the displayed field tolerances.") :
        join(validation.issues,"\n")
    A=vertex === nothing ? nothing : copy(Modules.component(f,vertex))
    r=A === nothing ? nothing : FieldLinAlg.rank(field,A)
    if A !== nothing
        push!(panels,_algebra_matrix_panel(A,"Component f($vertex)",field,matrix_limit;
            subtitle="rank $r; kernel $(size(A,2)-r); cokernel $(size(A,1)-r)",source="M($vertex)",target="N($vertex)"))
    end
    push!(panels,_inspection_text_panel("One map between two modules",
        [status,"A component goes from M(q) to N(q).",
         "Choose vertex=q for its matrix, or pair=(p,q) for naturality.",
         "Diagram positions are shared layout, not parameter coordinates."];
        subtitle=_inspection_field_label(field)))
    return VisualizationSpec(:morphism_inspector;title="A module morphism  M \u2192 N",panels,
        metadata=(;field,category=:finite_poset_representations,selection=(;vertex,pair),
            component=A,rank=r,validation,panel_columns=2,figure_size=(1160,860)))
end

function _morphism_subquotient_spec(f;vertex,matrix_limit=(12,12))
    validation=_algebra_morphism_validation(f)
    validation.valid || throw(ArgumentError("Kernels, images and cokernels require a valid morphism. "*join(validation.issues," ")))
    K,ki=AbelianCategories.kernel_with_inclusion(f)
    I,ii=AbelianCategories.image_with_inclusion(f)
    C,cp=AbelianCategories.cokernel_with_projection(f)
    field=f.dom.field
    prepared=_prepare_hasse(f.dom.Q)
    panels=VisualizationSpec[_algebra_module_panel(M,title;vertex,prepared) for (M,title) in
        ((K,"Kernel: ker f"),(I,"Image: im f"),(C,"Cokernel: coker f"))]
    for (g,title,source,target) in ((ki,"Kernel inclusion","ker f","M"),
        (ii,"Image inclusion","im f","N"),(cp,"Cokernel projection","N","coker f"))
        push!(panels,_algebra_matrix_panel(Modules.component(g,vertex),"$title at $vertex",field,matrix_limit;source,target))
    end
    return VisualizationSpec(:kernel_image_cokernel;title="What is killed, reached and left over",
        subtitle="Computed modules with their structure maps; matrices use the returned bases. These are not claimed to be direct summands.",panels,
        metadata=(;field,category=:finite_poset_representations,vertex,validation,
            component=copy(Modules.component(f,vertex)),rank=I.dims[vertex],
            algebra=(;kernel=K,image=I,cokernel=C,inclusion_kernel=ki,inclusion_image=ii,projection_cokernel=cp),
            panel_columns=3,figure_size=(1440,940)))
end

function _sequence_visual_spec(ses;vertex=nothing,matrix_limit=(12,12))
    field=ses.B.field
    # Refresh the owner's mathematical checks: a previously checked mutable
    # sequence may have subsequently acquired different component matrices.
    fresh=AbelianCategories.short_exact_sequence(ses.i,ses.p;check=false)
    owner_validation=AbelianCategories.check_short_exact_sequence(fresh)
    # The existing morphism checker uses a fixed floating threshold. For numerical
    # inputs retain its report but validate naturality at the declared tolerance.
    inclusion_validation=_algebra_morphism_validation(ses.i)
    projection_validation=_algebra_morphism_validation(ses.p)
    issues=String[]
    append!(issues,inclusion_validation.issues)
    append!(issues,projection_validation.issues)
    AbelianCategories.is_exact(fresh) || push!(issues,"The pointwise exactness conditions fail.")
    validation=(;valid=isempty(issues),issues,owner=owner_validation,
        inclusion=inclusion_validation,projection=projection_validation,
        pointwise_exact=AbelianCategories.is_exact(fresh))
    prepared=_prepare_hasse(ses.B.Q)
    panels=VisualizationSpec[_algebra_module_panel(M,title;vertex,prepared) for (M,title) in
        ((ses.A,"A: included module"),(ses.B,"B: middle module"),(ses.C,"C: quotient module"))]
    component=nothing
    if vertex !== nothing
        A=copy(Modules.component(ses.i,vertex)); B=copy(Modules.component(ses.p,vertex))
        equation=_algebra_equation(field,B*A,zeros(eltype(A),size(B,1),size(A,2)))
        component=(;inclusion=A,projection=B,composite=B*A,equation)
        for (X,title,source,target) in ((A,"i($vertex)","A","B"),(B,"p($vertex)","B","C"),
            (B*A,"p($vertex) i($vertex)","A","C"))
            push!(panels,_algebra_matrix_panel(X,title,field,matrix_limit;source,target,
                subtitle=title=="p($vertex) i($vertex)" ? _algebra_equation_text(equation) : ""))
        end
    end
    status=validation.valid ? "Exactness checked: naturality, injectivity, image = kernel, surjectivity and zero composite." :
        "Not a verified exact sequence: "*join(validation.issues," ")
    field isa CoreModules.RealField && (status *= "\nNumerical checks use field tolerances; this is not a symbolic certificate.")
    return VisualizationSpec(:exact_sequence;title="0 \u2192 A  \u2014i\u2192  B  \u2014p\u2192  C \u2192 0",subtitle=status,panels,
        metadata=(;field,category=:finite_poset_representations,vertex,component,validation,
            exact=validation.valid,exact_arithmetic=!(field isa CoreModules.RealField),
            panel_columns=3,figure_size=(1440,vertex===nothing ? 640 : 980)))
end

# A represented zero must mask the unknown-region background completely.
# Translucent nonzero support fills retain the shared region-view appearance.
function _algebra_support_layers(geometry,colors)
    layers = _region_geometry_layers(geometry;colors)
    for (i,layer) in pairs(layers)
        if layer isa PolygonLayer && layer.fill_color === _VisualRole(:background)
            layers[i] = PolygonLayer(layer.polygons,layer.fill_color,layer.stroke_color,1.0,layer.linewidth)
        end
    end
    return layers
end

function _morphism_support_spec(f;encoding,box=nothing)
    field=f.dom.field
    validation=_algebra_morphism_validation(f)
    validation.valid || throw(ArgumentError("Support views require a valid supplied morphism."))
    pi=_inspection_classifier(encoding_map(encoding))
    geometry=_region_geometry_2d(pi;box)
    warnings=String[]
    isempty(geometry.coordinate_collisions) || push!(warnings,"Distinct exact coordinates coincide at drawing precision.")
    isempty(geometry.dimension_collapses) || push!(warnings,"A region loses dimension at drawing precision.")
    ranks=[FieldLinAlg.rank(field,Modules.component(f,q)) for q in 1:nvertices(f.dom.Q)]
    # Pull back both modules and the supplied morphism along this SAME classifier.
    # Support intersections are only eligibility for nonzero components.
    overlap=map((a,b)->a>0 && b>0,f.dom.dims,f.cod.dims)
    panels=VisualizationSpec[]
    for (title,values,role) in (("Source support",f.dom.dims,:source),
            ("Target support",f.cod.dims,:target),("Support of the image",ranks,:selected))
        colors=Dict{Int,_VisualRole}(q=>(values[q]>0 ? _VisualRole(role) : _VisualRole(:background)) for q in eachindex(values))
        colors[0]=_VisualRole(:unrepresented)
        layers=_algebra_support_layers(geometry,colors)
        push!(panels,VisualizationSpec(:morphism_support_regions;title,
            subtitle="Colored: nonzero space; uncolored: represented zero\nSolid/dashed: included/excluded; dotted: window cut",
            layers,axes=geometry.axes,metadata=(;geometry,values=copy(values),classifier=pi,minimal_axes=true,
                quantity=title=="Support of the image" ? :component_rank : :stalk_dimension)))
    end
    overlay_colors=Dict{Int,_VisualRole}(q=>_VisualRole(overlap[q] ? :both :
        f.dom.dims[q]>0 ? :source : f.cod.dims[q]>0 ? :target : :background) for q in eachindex(ranks))
    overlay_colors[0]=_VisualRole(:unrepresented)
    overlay=VisualizationSpec(:morphism_support_overlay;title="Source and target together",
        subtitle="Overlap permits a nonzero component; it does not force one\nSolid/dashed: included/excluded; dotted: window cut",
        layers=_algebra_support_layers(geometry,overlay_colors),axes=geometry.axes,
        metadata=(;geometry,overlap=copy(overlap),classifier=pi,minimal_axes=true))
    insert!(panels,3,overlay)
    subtitle="Both modules use the same classifier. Overlap alone does not determine a map or its image."
    geometry.has_unrepresented_area && (subtitle *= "\nGray denotes an unrepresented region, not a zero space.")
    geometry.geometry_kind===:nearest_lattice_tiles &&
        (subtitle *= "\nInteger fibers use nearest-lattice tiles, with ties rounded to even.")
    isempty(warnings) || (subtitle *= "\n"*join(warnings," ")*" Exact geometry is retained in metadata.")
    legend=_default_legend(;visible=true,entries=[
        (;label="source",color=_VisualRole(:source),style=:patch),
        (;label="target",color=_VisualRole(:target),style=:patch),
        (;label="overlap",color=_VisualRole(:both),style=:patch),
        (;label="image",color=_VisualRole(:selected),style=:patch)])
    return VisualizationSpec(:morphism_support;title="Supports of a supplied module morphism",subtitle,panels,legend,
        metadata=(;field,category=:finite_poset_representations,validation,classifier=pi,
            base=f.dom.Q,geometry,warnings,source_dimensions=copy(f.dom.dims),target_dimensions=copy(f.cod.dims),
            component_ranks=ranks,overlap,support_restriction=all(q->overlap[q] || iszero(ranks[q]),eachindex(ranks)),
            panel_columns=2,figure_size=(1180,990)))
end

_visual_spec(f::Modules.PMorphism,kind::Symbol;kwargs...) = kind === :morphism_support ?
    _morphism_support_spec(f;kwargs...) : _morphism_visual_spec(f,kind;kwargs...)
_visual_spec(ses::AbelianCategories.ShortExactSequence,::Symbol;kwargs...) = _sequence_visual_spec(ses;kwargs...)

function _visual_request_cost(::Modules.PMorphism,kind::Symbol)
    return (;work=kind===:kernel_image_cokernel ? :construct_kernel_image_cokernel_and_maps :
        kind===:morphism_support ? :classifier_geometry_and_component_ranks :
        kind===:naturality ? :selected_square_and_composites : :cover_naturality_and_selected_component,
        timing=:not_measured,cache_reuse=:algebra_owner,selection=:static_api_arguments,
        geometry=kind===:morphism_support ? :explicit_common_classifier : :schematic)
end
_visual_request_cost(::AbelianCategories.ShortExactSequence,::Symbol) =
    (;work=:fresh_exactness_check_and_selected_matrices,timing=:not_measured,cache_reuse=:none)
