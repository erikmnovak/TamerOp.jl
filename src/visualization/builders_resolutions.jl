# Resolution multiplicities and selected incidence on the actual finite base.
# A drawing never constructs a resolution or identifies finite-category Betti
# numbers with the invariants of an ambient multigraded free resolution.

const _ResolutionView = Union{DerivedFunctors.ProjectiveResolution,
    DerivedFunctors.InjectiveResolution, IndicatorResolutions.UpsetResolutionResult,
    IndicatorResolutions.DownsetResolutionResult}
const _PresentationView = Union{IndicatorTypes.UpsetPresentation,IndicatorTypes.DownsetCopresentation}

available_visuals(::DerivedFunctors.ProjectiveResolution) = (:betti_table, :resolution, :betti_degrees, :resolution_lift)
available_visuals(::IndicatorResolutions.UpsetResolutionResult) = (:betti_table, :resolution, :betti_degrees)
available_visuals(::Union{DerivedFunctors.InjectiveResolution,IndicatorResolutions.DownsetResolutionResult}) =
    (:bass_table, :resolution, :bass_degrees)
available_visuals(::_PresentationView) = (:presentation_incidence,)
available_visuals(res::Results.ResolutionResult) = available_visuals(Results.resolution_object(res))

_resolution_request_keywords(kind) = kind in (:betti_table, :bass_table) ? (:verify, :matrix_limit) :
    kind in (:betti_degrees, :bass_degrees) ? (:grades, :degree, :verify) :
    (:degree, :summand, :vertex, :grades, :verify, :matrix_limit, :basis_change, :support_sheets)
_visual_request_keywords(::Union{DerivedFunctors.InjectiveResolution,
    IndicatorResolutions.UpsetResolutionResult,IndicatorResolutions.DownsetResolutionResult}, kind::Symbol) =
    _resolution_request_keywords(kind)
_visual_request_keywords(::_PresentationView, ::Symbol) =
    (:degree, :summand, :vertex, :grades, :matrix_limit, :basis_change, :support_sheets)
_visual_request_keywords(res::Results.ResolutionResult, kind::Symbol) =
    _visual_request_keywords(Results.resolution_object(res), kind)
_resolution_request_cost(kind) = (; work=kind in (:betti_table,:bass_table,:betti_degrees,:bass_degrees) ? :stored_generator_counts : :selected_incidence,
    resolution_construction=false, verification=:opt_in_all_stalk_equations_and_ranks,
    grades=:supplied_order_embedding_checked, timing=:not_measured)
_visual_request_cost(::Union{DerivedFunctors.InjectiveResolution,
    IndicatorResolutions.UpsetResolutionResult,IndicatorResolutions.DownsetResolutionResult,_PresentationView}, kind::Symbol) =
    _resolution_request_cost(kind)
_visual_request_cost(res::Results.ResolutionResult, kind::Symbol) = _visual_request_cost(Results.resolution_object(res), kind)

_resolution_up(::Union{DerivedFunctors.ProjectiveResolution,IndicatorResolutions.UpsetResolutionResult,
    IndicatorTypes.UpsetPresentation}) = true
_resolution_up(::Union{DerivedFunctors.InjectiveResolution,IndicatorResolutions.DownsetResolutionResult,
    IndicatorTypes.DownsetCopresentation}) = false
_resolution_active(P, gens, q, up) = findall(g -> up ? leq(P,g,q) : leq(P,q,g), gens)

function _principal_plot_labels(P, supports, up)
    return [begin
        # These containers can be hand-built. Do not invent a birth/death label
        # for a nonprincipal support, or silently change its represented map.
        bases = [v for v in 1:nvertices(P) if all(q -> U.mask[q] ==
            (up ? leq(P,v,q) : leq(P,q,v)), 1:nvertices(P))]
        length(bases) == 1 || throw(ArgumentError("Resolution incidence requires principal supports on this finite poset."))
        only(bases)
    end for U in supports]
end

function _resolution_plot_info(res::_ResolutionView)
    up = _resolution_up(res)
    if res isa Union{DerivedFunctors.ProjectiveResolution,DerivedFunctors.InjectiveResolution}
        M = DerivedFunctors.source_module(res)
        gens = [copy(g) for g in res.gens]
        terms = DerivedFunctors.resolution_terms(res)
        maps = DerivedFunctors.resolution_differentials(res)
        aug = up ? DerivedFunctors.augmentation_map(res) : DerivedFunctors.coaugmentation_map(res)
        length(terms) == length(gens) == length(maps)+1 || throw(ArgumentError("Inconsistent resolution degree bookkeeping."))
        res isa DerivedFunctors.ProjectiveResolution && length(res.d_mat) != length(maps) &&
            throw(ArgumentError("Each projective differential needs its stored coefficient matrix."))
    else
        M = res.source_module
        presentations = IndicatorResolutions.resolution_modules(res)
        gens = [_principal_plot_labels(M.Q, up ? IndicatorTypes.generator_labels(p) :
            IndicatorTypes.cogenerator_labels(p), up) for p in presentations]
        maps = IndicatorResolutions.resolution_maps(res)
        aug = up ? IndicatorResolutions.augmentation(res) : IndicatorResolutions.coaugmentation(res)
        terms = nothing
        length(gens) == length(maps)+1 || throw(ArgumentError("Inconsistent indicator resolution degrees."))
    end
    isempty(gens) && throw(ArgumentError("A resolution must retain degree zero, including for a zero module."))
    all(g -> all(v -> 1 <= v <= nvertices(M.Q), g), gens) || throw(ArgumentError("A summand label is outside the finite poset."))
    return (; object=res, P=M.Q, field=M.field, up, gens, terms, maps, aug, module_object=M,
        presentation=false, length=length(gens)-1)
end

function _resolution_plot_info(p::_PresentationView)
    P, field, up = IndicatorTypes.ambient_poset(p), IndicatorTypes.field(p), _resolution_up(p)
    supports = up ? (IndicatorTypes.generator_labels(p),IndicatorTypes.relation_labels(p)) :
        (IndicatorTypes.cogenerator_labels(p),IndicatorTypes.corelation_labels(p))
    gens = [_principal_plot_labels(P,s,up) for s in supports]
    return (; object=p,P,field,up,gens,terms=nothing,maps=nothing,aug=nothing,module_object=nothing,
        presentation=true,length=1)
end

function _resolution_plot_matrix(data, k)
    obj = data.object
    A = if obj isa DerivedFunctors.ProjectiveResolution
        obj.d_mat[k]
    elseif obj isa DerivedFunctors.InjectiveResolution
        dom, cod = data.gens[k], data.gens[k+1]
        K = CoreModules.coeff_type(data.field)
        D = spzeros(K,length(cod),length(dom))
        # A downset map is determined by its coefficients at each target socle.
        for q in unique(cod)
            rows = _resolution_active(data.P,cod,q,false)
            cols = _resolution_active(data.P,dom,q,false)
            C = Modules.component(data.maps[k],q)
            size(C) == (length(rows),length(cols)) || throw(ArgumentError("Injective stalk coordinates disagree with their summands."))
            for (i,row) in enumerate(rows), (j,col) in enumerate(cols)
                cod[row] == q && !iszero(C[i,j]) && (D[row,col] = C[i,j])
            end
        end
        D
    elseif obj isa IndicatorResolutions.UpsetResolutionResult
        copy(transpose(data.maps[k]))
    elseif obj isa IndicatorResolutions.DownsetResolutionResult
        data.maps[k]
    elseif obj isa IndicatorTypes.UpsetPresentation
        copy(transpose(IndicatorTypes.presentation_matrix(obj)))
    else
        IndicatorTypes.copresentation_matrix(obj)
    end
    dom, cod = data.up ? (data.gens[k+1],data.gens[k]) : (data.gens[k],data.gens[k+1])
    _resolution_check_coefficients(data,A,dom,cod)
    return copy(A)
end

function _resolution_check_coefficients(data,A,dom,cod)
    A isa AbstractMatrix{CoreModules.coeff_type(data.field)} || throw(ArgumentError("Coefficient matrix has the wrong field."))
    size(A) == (length(cod),length(dom)) || throw(ArgumentError("Coefficient matrix shape disagrees with its summand labels."))
    data.field isa CoreModules.RealField && !all(isfinite,A) && throw(ArgumentError("Coefficients must be finite."))
    for j in eachindex(dom), i in eachindex(cod)
        !leq(data.P,cod[i],dom[j]) && !iszero(A[i,j]) &&
            throw(ArgumentError("A nonzero coefficient is forbidden by the principal summand order."))
    end
    return nothing
end

function _resolution_grades(P, grades)
    grades === nothing && return nothing
    grades isa AbstractVector && length(grades) == nvertices(P) ||
        throw(ArgumentError("grades must give one coordinate pair per finite vertex."))
    all(g -> (g isa Tuple || g isa AbstractVector) && length(g)==2 &&
        all(x -> x isa Real && !(x isa Bool) && isfinite(x) && isfinite(Float64(x)),g),grades) ||
        throw(ArgumentError("grades must contain finite real coordinate pairs."))
    # Compare in the supplied exact type, before converting only the drawing.
    exact = [Tuple(g) for g in grades]
    for q in eachindex(exact), p in eachindex(exact)
        leq(P,p,q) == all(exact[p][j] <= exact[q][j] for j in 1:2) ||
            throw(ArgumentError("grades must be an order embedding in the coordinatewise plane; Hasse layout coordinates are not grades."))
    end
    drawn = [_drawing_point(g) for g in exact]
    length(unique(drawn)) == length(drawn) || throw(ArgumentError("Distinct grades collide in floating display coordinates; use the exact degree-by-label table."))
    return exact
end

function _check_resolution_request!(issues,res,kind;kwargs...)
    data = _resolution_plot_info(res)
    degree = get(kwargs,:degree,0)
    _comparison_degree!(issues,degree,0:data.length)
    _comparison_matrix_limit!(issues,get(kwargs,:matrix_limit,(12,12)))
    get(kwargs,:support_sheets,false) isa Bool || push!(issues,"support_sheets must be true or false.")
    verify = get(kwargs,:verify,false)
    verify isa Bool || push!(issues,"verify must be true or false.")
    grades = _resolution_grades(data.P,get(kwargs,:grades,nothing))
    kind in (:betti_degrees,:bass_degrees) && grades === nothing && push!(issues,"A grade-plane view requires explicit grades.")
    summand = get(kwargs,:summand,nothing)
    if degree isa Integer && !(degree isa Bool) && 0 <= degree <= data.length
        summand === nothing || (summand isa Integer && !(summand isa Bool) && 1 <= summand <= length(data.gens[degree+1])) ||
            push!(issues,"summand must index a stored summand of the selected degree (empty terms have no selectable summands).")
    end
    vertex = get(kwargs,:vertex,nothing)
    vertex === nothing || _comparison_vertex!(issues,data.P,vertex)
    change = get(kwargs,:basis_change,nothing)
    if change !== nothing
        change isa NamedTuple && keys(change) == (:source,:target) || push!(issues,"basis_change must be (; source=S, target=T).")
        degree == 0 && !data.presentation && push!(issues,"Select degree >= 1 for a differential basis change; the augmentation is not a principal-summand matrix.")
    end
    return issues
end
_append_visual_request_issues!(issues::Vector{String}, res::Union{DerivedFunctors.InjectiveResolution,
    IndicatorResolutions.UpsetResolutionResult,IndicatorResolutions.DownsetResolutionResult,_PresentationView},kind::Symbol;kwargs...) =
    _check_resolution_request!(issues,res,kind;kwargs...)
_append_visual_request_issues!(issues::Vector{String},res::Results.ResolutionResult,kind::Symbol;kwargs...) =
    _append_visual_request_issues!(issues,Results.resolution_object(res),kind;kwargs...)

function _resolution_stalks(data, matrices, q)
    active = [_resolution_active(data.P,g,q,data.up) for g in data.gens]
    return [Matrix(data.up ? matrices[k][active[k],active[k+1]] : matrices[k][active[k+1],active[k]])
            for k in eachindex(matrices)], active
end

function _resolution_verify(data)
    matrices = [_resolution_plot_matrix(data,k) for k in 1:data.length]
    data.presentation && return (;minimality=:not_checked,completion=:not_applicable,checks=NamedTuple[])
    P,field,up,M = data.P,data.field,data.up,data.module_object
    Modules.check_module(M;throw=true)
    checks = NamedTuple[]
    minimal = true
    complete = true
    # Validate the augmentation and every stored term in the advertised principal
    # bases. Structural owner validators alone do not establish exactness.
    aug = data.aug
    (up ? aug.cod === M : aug.dom === M) || throw(ArgumentError("The (co)augmentation has the wrong resolved endpoint."))
    first_term = up ? aug.dom : aug.cod
    first_term.Q === P && first_term.field == field || throw(ArgumentError("The (co)augmentation uses a different poset or field."))
    if data.terms !== nothing
        first_term == first(data.terms) || throw(ArgumentError("The (co)augmentation endpoint disagrees with degree zero."))
        for k in eachindex(data.maps)
            source,target = up ? (data.terms[k+1],data.terms[k]) : (data.terms[k],data.terms[k+1])
            data.maps[k].dom == source && data.maps[k].cod == target || throw(ArgumentError("Differential endpoints disagree with stored terms."))
        end
    end
    for q in 1:nvertices(P)
        ds,active = _resolution_stalks(data,matrices,q)
        dims = length.(active)
        first_term.dims[q] == dims[1] || throw(ArgumentError("Degree-zero summand dimensions disagree with the (co)augmentation."))
        A = copy(Modules.component(aug,q))
        size(A) == (up ? (M.dims[q],dims[1]) : (dims[1],M.dims[q])) || throw(ArgumentError("(Co)augmentation shape disagrees with degree zero."))
        ranks = [FieldLinAlg.rank(field,D) for D in vcat([A],ds)]
        ranks[1] == M.dims[q] || throw(ArgumentError("(Co)augmentation is not exact at the resolved module at vertex $q."))
        for k in eachindex(ds)
            previous = k==1 ? A : ds[k-1]
            product = up ? previous*ds[k] : ds[k]*previous
            eq = _algebra_equation(field,product,zero(product))
            push!(checks,(;vertex=q,degree=k,equation=eq))
            eq.valid || throw(ArgumentError("The augmented resolution equation fails at vertex $q, degree $k."))
            ranks[k]+ranks[k+1] == dims[k] || throw(ArgumentError("The stored prefix is not exact at vertex $q, degree $(k-1)."))
        end
        complete &= ranks[end] == dims[end]
        for k in eachindex(data.gens)
            D = k==1 ? A : ds[k-1]
            g = data.gens[k][active[k]]
            if up
                old = findall(v -> v != q,g)
                minimal &= FieldLinAlg.rank(field,D)-FieldLinAlg.rank(field,D[:,old]) == count(==(q),g)
            else
                socle = findall(==(q),g)
                E = CoreModules.eye(field,length(g))[:,socle]
                minimal &= FieldLinAlg.rank(field,hcat(D,E)) == FieldLinAlg.rank(field,D)
            end
            if data.terms !== nothing
                term = data.terms[k]
                term.Q === P && term.field == field && term.dims[q] == dims[k] ||
                    throw(ArgumentError("Stored term and principal summands disagree."))
            end
        end
        if data.terms !== nothing
            for k in eachindex(ds)
                eq = _algebra_equation(field,Modules.component(data.maps[k],q),ds[k])
                eq.valid || throw(ArgumentError("Stored differential and summand coefficients disagree at vertex $q."))
            end
        end
    end
    # Check canonical indicator structure maps and augmentation naturality at covers.
    for (u,v) in FiniteFringe.cover_edges(P)
        local_active = [[_resolution_active(P,g,q,up) for g in data.gens] for q in (u,v)]
        term_maps = [CoreModules.coerce.(Ref(field),Int.(reshape(local_active[2][k],:,1) .== reshape(local_active[1][k],1,:))) for k in eachindex(data.gens)]
        _algebra_equation(field,Modules.structure_map(first_term;source=u,target=v),term_maps[1]).valid ||
            throw(ArgumentError("Degree-zero structure maps disagree with principal summands."))
        if data.terms !== nothing
            for k in eachindex(data.terms)
                _algebra_equation(field,Modules.structure_map(data.terms[k];source=u,target=v),term_maps[k]).valid ||
                    throw(ArgumentError("Term structure maps are not in the recorded principal-summand bases."))
            end
        end
        mu,mv = Modules.component(aug,u),Modules.component(aug,v)
        mapM = Modules.structure_map(M;source=u,target=v)
        eq = up ? _algebra_equation(field,mapM*mu,mv*term_maps[1]) : _algebra_equation(field,term_maps[1]*mu,mv*mapM)
        eq.valid || throw(ArgumentError("(Co)augmentation fails naturality on $u -> $v."))
    end
    return (;minimality=minimal ? :minimal : :nonminimal,completion=complete ? :complete : :truncated,checks)
end

function _resolution_counts(data)
    table = zeros(Int,length(data.gens),nvertices(data.P))
    for (k,gens) in enumerate(data.gens),g in gens; table[k,g]+=1; end
    return table
end

_resolution_label(data,g,i,grades) = "$(data.up ? "P" : "I")@$g #$i" *
    (grades===nothing ? "" : " $(grades[g])")

function _resolution_incidence_panel(data,A,dom,cod,grades,limit; selected_row=nothing,selected_column=nothing,
                                     differential_degree=1,title="Differential coefficients")
    rows,cols=collect(1:min(size(A,1),limit[1])),collect(1:min(size(A,2),limit[2]))
    # A bounded view must still include the requested summand. Retain global
    # labels when replacing the last displayed index with an out-of-window one.
    selected_row===nothing || selected_row in rows || (rows[end]=selected_row)
    selected_column===nothing || selected_column in cols || (cols[end]=selected_column)
    forced=BitMatrix([!leq(data.P,cod[i],dom[j]) for i in rows,j in cols])
    roles=[forced[i,j] ? :muted : :foreground for i in axes(forced,1),j in axes(forced,2)]
    row_roles=[i==selected_row ? :selected : :foreground for i in rows]
    column_roles=[j==selected_column ? :selected : :foreground for j in cols]
    entries=[_matrix_coefficient_text(A[i,j]) for i in rows,j in cols]
    for i in axes(forced,1),j in axes(forced,2);forced[i,j] && (entries[i,j]*="\u2020");end
    truncated=length(rows)<size(A,1) || length(cols)<size(A,2)
    k=differential_degree
    arrow=data.up ? "P_$k -> P_$(k-1)" : "I^$(k-1) -> I^$k"
    subtitle="$arrow\nRows: target; columns: source. Dagger: order-forced zero; unmarked zero: allowed.\nStored principal $(data.up ? "upset" : "downset") bases; $(_inspection_field_label(data.field))\n$(size(A,1)) x $(size(A,2))"
    truncated && (subtitle*="; showing $(length(rows)) x $(length(cols)), including the selection; full matrix in metadata")
    return VisualizationSpec(:presentation_matrix;title,subtitle,
        layers=AbstractVisualizationLayer[MatrixLayer(entries,
            [_resolution_label(data,cod[i],i,grades) for i in rows],
            [_resolution_label(data,dom[j],j,grades) for j in cols])],
        metadata=(;panel_style=:matrix,matrix=copy(A),matrix_size=size(A),displayed_rows=rows,displayed_columns=cols,truncated,
            field=data.field,source_generators=copy(dom),target_generators=copy(cod),structural_zero_mask=forced,
            structural_zero_scope=:displayed_block,cell_roles=roles,row_roles,column_roles,
            selected_row,selected_column,matrix_corner="target / source"))
end

function _resolution_grade_panel(data,counts,degree,grades;selected_vertex=nothing)
    row=counts[degree+1,:]
    vertices=findall(!iszero,row)
    points=NTuple{2,Float64}[_drawing_point(grades[q]) for q in vertices]
    # Include annotation room explicitly: text bounds do not participate in
    # Makie's point autolimits, so labels at the top/right would be clipped.
    extent=isempty(grades) ? [(0.0,0.0)] : [_drawing_point(g) for g in grades]
    xmin,xmax=extrema(first,extent);ymin,ymax=extrema(last,extent)
    dx=xmax==xmin ? max(abs(xmin),1.0) : xmax-xmin
    dy=ymax==ymin ? max(abs(ymin),1.0) : ymax-ymin
    xlimits=(xmin-0.15dx,xmax+0.4dx);ylimits=(ymin-0.15dy,ymax+0.25dy)
    all(isfinite,(xlimits...,ylimits...)) || throw(ArgumentError("Grade-plane display bounds overflow; use the exact degree-by-label table."))
    layers=AbstractVisualizationLayer[PointLayer(points,_VisualRole(:support),1.0,18.0)]
    if selected_vertex!==nothing
        push!(layers,PointLayer([_drawing_point(grades[selected_vertex])],_VisualRole(:selected),1.0,22.0))
    end
    push!(layers,TextLayer(["@$q: $(row[q])" for q in vertices],[(p[1],p[2]+0.04dy) for p in points],_VisualRole(:foreground),12.0))
    isempty(vertices) && push!(layers,TextLayer(["Zero summands in\nthis stored degree"],[(xmin,ymin+0.5dy)],_VisualRole(:foreground),14.0))
    return VisualizationSpec(:resolution_grades;title="Degree $degree: labelled multiplicities",
        subtitle="Supplied grade coordinates; finite-poset category. No ambient free-resolution claim." *
            (selected_vertex===nothing ? "" : "\nSelected finite vertex: $selected_vertex"),layers,
        axes=(;xlabel="grade 1",ylabel="grade 2",xlimits,ylimits,aspect=:data),
        metadata=(;grades=copy(grades),vertices,multiplicities=row[vertices],degree,selected_vertex))
end

function _resolution_basis_change(data,A,dom,cod,change)
    change===nothing && return nothing
    S,T=change.source,change.target
    _resolution_check_coefficients(data,S,dom,dom)
    _resolution_check_coefficients(data,T,cod,cod)
    field=data.field
    FieldLinAlg.rank(field,S)==length(dom) && FieldLinAlg.rank(field,T)==length(cod) ||
        throw(ArgumentError("Graded basis changes must be invertible over the coefficient field."))
    Sinv=FieldLinAlg.solve_fullcolumn(field,S,CoreModules.eye(field,length(dom)))
    Tinv=FieldLinAlg.solve_fullcolumn(field,T,CoreModules.eye(field,length(cod)))
    _resolution_check_coefficients(data,Sinv,dom,dom)
    _resolution_check_coefficients(data,Tinv,cod,cod)
    B=Tinv*A*S
    _resolution_check_coefficients(data,B,dom,cod)
    equation=_algebra_equation(field,A*S,T*B)
    equation.valid || throw(ArgumentError("Basis-change equation A*S = T*B failed."))
    return (;source=copy(S),target=copy(T),source_inverse=Sinv,target_inverse=Tinv,matrix=B,equation,
        scope=:selected_differential,whole_resolution_transformed=false)
end

function _resolution_view(res,kind;degree=0,summand=nothing,vertex=nothing,grades=nothing,
                          verify=false,matrix_limit=(12,12),basis_change=nothing,support_sheets=false)
    data=_resolution_plot_info(res)
    grades=_resolution_grades(data.P,grades)
    counts=_resolution_counts(data)
    verification=verify ? _resolution_verify(data) : (;minimality=:not_checked,completion=:not_checked,checks=NamedTuple[])
    side=data.up ? :projective : :injective
    convention=data.up ? :homological : :cohomological
    direction=data.up ? "... -> P_2 -> P_1 -> P_0 -> M -> 0" : "0 -> M -> I^0 -> I^1 -> I^2 -> ..."
    subtitle="$side; finite-poset representations; $(_inspection_field_label(data.field))\n$direction\nMinimality: $(verification.minimality); terminal status: $(verification.completion); stored degrees 0:$(data.length)"
    meta=(;category=:finite_poset_representations,field=data.field,side,degree_convention=convention,
        direction,computed_length=data.length,stored_degrees=0:data.length,counts,verification,
        selection=(;degree,summand,vertex),grades,ambient_free_resolution=false)
    extent_note=verification.completion===:complete ? "Termination verified; only stored degrees are displayed." :
        "Higher degrees are uncertified here; no zero padding."
    table=_presentation_matrix_panel(counts,"$(data.up ? "Projective / Betti" : "Injective / Bass") multiplicities",
        ["degree $(k-1)" for k in eachindex(data.gens)],["vertex $v" for v in 1:nvertices(data.P)],matrix_limit;
        subtitle="$(verification.minimality===:minimal ? "Verified minimal prefix" : "Stored summand counts; not certified Betti/Bass invariants").\n$extent_note",
        metadata=(;field=data.field,degree_convention=convention,matrix_corner="degree / label",matrix_row_heading="Degrees"))
    if kind in (:betti_table,:bass_table)
        return VisualizationSpec(kind;title=table.title,subtitle=subtitle*"\n"*table.subtitle,layers=table.layers,
            metadata=merge(table.metadata,meta))
    elseif kind in (:betti_degrees,:bass_degrees)
        panel=_resolution_grade_panel(data,counts,degree,grades)
        return VisualizationSpec(kind;title=panel.title,subtitle=subtitle*"\n"*panel.subtitle,layers=panel.layers,axes=panel.axes,
            metadata=merge(meta,panel.metadata))
    end
    panels=VisualizationSpec[table]
    selected=degree+1
    sheet_ids = support_sheets ? collect(1:min(length(data.gens[selected]),matrix_limit[1])) :
        summand===nothing ? Int[] : [Int(summand)]
    if support_sheets && summand!==nothing && !(summand in sheet_ids)
        push!(sheet_ids,Int(summand))
    end
    for id in sheet_ids
        g=data.gens[selected][id]
        support=Int[data.up ? leq(data.P,g,q) : leq(data.P,q,g) for q in 1:nvertices(data.P)]
        panel=_hasse_spec(data.P;dims=support,vertex=id==summand ? g : nothing)
        push!(panels,_algebra_with_title(panel,"Degree $degree summand #$id$(id==summand ? " (selected)" : "")";
            subtitle="Principal $(data.up ? "upset" : "downset") at vertex $g; 1 = support. Schematic layout, not an extra parameter."))
    end
    grades===nothing || push!(panels,_resolution_grade_panel(data,counts,degree,grades;
        selected_vertex=summand===nothing ? nothing : data.gens[selected][summand]))
    coefficient=nothing;changed=nothing;stalk=nothing;incidence=nothing;represented_dimension=nothing
    k=data.presentation ? 1 : degree
    if k>0
        coefficient=_resolution_plot_matrix(data,k)
        dom,cod=data.up ? (data.gens[k+1],data.gens[k]) : (data.gens[k],data.gens[k+1])
        selected_source=data.up ? degree==k : degree==k-1
        row=selected_source ? nothing : summand
        col=selected_source ? summand : nothing
        incidence=(;source_generators=copy(dom),target_generators=copy(cod),selected_row=row,selected_column=col)
        push!(panels,_resolution_incidence_panel(data,coefficient,dom,cod,grades,matrix_limit;
            selected_row=row,selected_column=col,differential_degree=k))
        if vertex!==nothing
            rows=_resolution_active(data.P,cod,vertex,data.up);cols=_resolution_active(data.P,dom,vertex,data.up)
            stalk=Matrix(coefficient[rows,cols])
            if data.presentation
                represented_dimension=(data.up ? length(rows) : length(cols))-FieldLinAlg.rank(data.field,stalk)
                push!(panels,_inspection_text_panel("Represented space at vertex $vertex",
                    ["$(data.up ? "Cokernel" : "Kernel") dimension: $represented_dimension",
                     "Computed from the active presentation map, not its image."]))
            end
            push!(panels,_presentation_matrix_panel(stalk,"Degree $k differential at vertex $vertex",
                [_resolution_label(data,cod[i],i,grades) for i in rows],[_resolution_label(data,dom[j],j,grades) for j in cols],matrix_limit;
                subtitle="Active summands only; rows and columns retain their global IDs.",metadata=(;active_rows=rows,active_columns=cols)))
        end
        changed=_resolution_basis_change(data,coefficient,dom,cod,basis_change)
        if changed!==nothing
            push!(panels,_resolution_incidence_panel(data,changed.matrix,dom,cod,grades,matrix_limit;
                differential_degree=k,title="After the supplied graded basis change"))
            push!(panels,_algebra_matrix_panel(changed.source,"Source change S",data.field,matrix_limit))
            push!(panels,_algebra_matrix_panel(changed.target,"Target change T",data.field,matrix_limit))
            push!(panels,_inspection_text_panel("Certified coordinate change",
                ["A*S = T*B; B = inverse(T)*A*S",_algebra_equation_text(changed.equation),
                 "Selected differential only; other resolution maps are not changed."]))
        end
    elseif vertex!==nothing
        stalk=copy(Modules.component(data.aug,vertex))
        push!(panels,_algebra_matrix_panel(stalk,data.up ? "Augmentation at vertex $vertex" : "Coaugmentation at vertex $vertex",data.field,matrix_limit))
    end
    lines=[direction,"Stored prefix: degrees 0:$(data.length). Terminal status: $(verification.completion).",
        "Minimality: $(verification.minimality). Coefficients use stored summand bases.",
        "Support sheets shown: $(length(sheet_ids))/$(length(data.gens[selected])). Separation is schematic.",
        "These are summands of the selected term, not a decomposition of the resolved module."]
    if data.presentation
        lines=[data.up ? "F_1 -> F_0 -> coker(A) -> 0" : "0 -> ker(A) -> E^0 -> E^1",
            "$(data.up ? "Cokernel presentation" : "Kernel copresentation") on this finite poset; not a fringe image.",
            "Stored generators need not be minimal. No further syzygies have been computed.",
            "Support sheets shown: $(length(sheet_ids))/$(length(data.gens[selected])). Separation is schematic.",
            "These are summands of a term, not a decomposition of the represented module."]
    end
    push!(panels,_inspection_text_panel("What the diagram represents",lines))
    layout=_comparison_layout(panels)
    return VisualizationSpec(kind;title=data.presentation ? "Presentation incidence" : "Resolution terms, support and incidence",
        subtitle=data.presentation ? "Finite-poset $(data.up ? "cokernel" : "kernel") construction; $(_inspection_field_label(data.field))" : subtitle,
        panels,metadata=merge(meta,(;coefficient_matrix=coefficient,stalk_matrix=stalk,incidence,basis_change=changed,
            represented_dimension,support_sheet_indices=sheet_ids,support_sheet_total=length(data.gens[selected]),
            construction=data.presentation ? (data.up ? :cokernel : :kernel) : :resolution,layout...)))
end

function _visual_spec(res::_ResolutionView,kind::Symbol;kwargs...)
    kind===:resolution_lift && return _resolution_lift_spec(res;kwargs...)
    return _resolution_view(res,kind;kwargs...)
end
_visual_spec(p::_PresentationView,kind::Symbol;kwargs...) = _resolution_view(p,kind;kwargs...)
_visual_spec(res::Results.ResolutionResult,kind::Symbol;kwargs...) =
    _visual_spec(Results.resolution_object(res),kind;kwargs...)
