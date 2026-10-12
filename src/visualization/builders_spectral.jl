# Spectral pages consume the numerical owner's dimensions and differentials.
# Cropping a chart never changes the complex or certifies an ambient limit.
const _SpectralVisualInput = Union{ChainComplexes.SpectralSequence,DerivedFunctors.TorSpectralSequence}

available_visuals(::_SpectralVisualInput) = (:spectral_page, :spectral_differential)
_visual_request_keywords(::_SpectralVisualInput, ::Symbol) =
    (:page, :source, :window, :compute, :truncated, :matrix_limit, :max_terms, :structure_maps)
_visual_request_cost(::_SpectralVisualInput, ::Symbol) =
    (; work=:requested_page_or_selected_differential, default=:stored_page_dimensions,
       compute_false=:stored_results_only, selection=:kernel_image_and_next_page,
       structure_maps=:supplied_matrices_only, timing=:not_measured)

_spectral_base(ss::ChainComplexes.SpectralSequence) = ss
_spectral_base(ss::DerivedFunctors.TorSpectralSequence) = DerivedFunctors.wrapped_spectral_sequence(ss)
_spectral_homological(ss) = ss isa DerivedFunctors.TorSpectralSequence
_spectral_label(r) = r === :inf ? "E_inf" : "E_$r"
_spectral_page_valid(r) = (r isa Int && r > 0) || r === :inf
_spectral_pair_valid(p) = p isa Tuple && length(p)==2 && all(x->x isa Int,p)

function _spectral_ranges(ss)
    ranges=ChainComplexes.spectral_sequence_summary(_spectral_base(ss)).bidegree_ranges
    if _spectral_homological(ss)
        return ((-ranges.a[2],-ranges.a[1]),(-ranges.b[2],-ranges.b[1]))
    end
    return (ranges.a,ranges.b)
end
_spectral_contains(ranges,p) = all(i->ranges[i][1]<=p[i]<=ranges[i][2],1:2)
function _spectral_known(ss,r; differential=false)
    info=ChainComplexes.spectral_sequence_summary(_spectral_base(ss))
    (r === :inf || r >= info.convergence_bound) && return true
    return differential ? r in info.cached_differentials : r in (1,2) || r in info.cached_pages
end
function _spectral_dimensions(ss,r,compute)
    compute || _spectral_known(ss,r) || return nothing
    return ChainComplexes.page_dims_dict(ss,r;nonzero_only=false)
end
function _spectral_target(ss,r,p;incoming=false)
    r === :inf && return nothing
    r >= ChainComplexes.spectral_sequence_summary(_spectral_base(ss)).convergence_bound && return nothing
    return incoming ? ChainComplexes.dr_source(ss,r,p) : ChainComplexes.dr_target(ss,r,p)
end
function _spectral_matrix(ss,r,p,dims)
    field=ChainComplexes.spectral_sequence_summary(_spectral_base(ss)).field
    K=CoreModules.coeff_type(field)
    n=dims[p]
    r === :inf && return zeros(K,0,n)
    # The Tor wrapper retains the same coordinate matrices and reverses degrees.
    A=Matrix(ChainComplexes.differential(ss,r,p))
    target=_spectral_target(ss,r,p)
    expected=(target === nothing ? 0 : dims[target],n)
    size(A)==expected || throw(DimensionMismatch("The retained differential at $p has shape $(size(A)); expected $expected."))
    return A
end

function _append_visual_request_issues!(issues::Vector{String},ss::_SpectralVisualInput,kind::Symbol;kwargs...)
    page=get(kwargs,:page,2);source=get(kwargs,:source,nothing)
    _spectral_page_valid(page) || push!(issues,"page must be a positive Int or :inf.")
    source === nothing || (_spectral_pair_valid(source) && _spectral_contains(_spectral_ranges(ss),source)) ||
        push!(issues,"source must be a bidegree tuple inside the supplied complex.")
    kind === :spectral_differential && source === nothing && push!(issues,"spectral_differential requires source=(p,q).")
    kind === :spectral_page && source !== nothing && push!(issues,"Use kind=:spectral_differential to select a source.")
    kind === :spectral_differential && page === :inf && push!(issues,"The associated graded limit has no selected finite-page differential.")
    for name in (:compute,:truncated)
        get(kwargs,name,name === :compute) isa Bool || push!(issues,"$name must be Bool.")
    end
    window=get(kwargs,:window,nothing)
    if window !== nothing
        valid=window isa Tuple && length(window)==2 && all(w->_spectral_pair_valid(w) && w[1]<=w[2],window)
        valid || push!(issues,"window must be ((pmin,pmax),(qmin,qmax)) with ordered Int endpoints.")
        if valid
            ranges=_spectral_ranges(ss)
            all(i->ranges[i][1]<=window[i][1]<=window[i][2]<=ranges[i][2],1:2) ||
                push!(issues,"The display window must lie inside the supplied complex; outside terms are not inferred.")
        source === nothing || !_spectral_pair_valid(source) || _spectral_contains(window,source) ||
                push!(issues,"The selected source must be in the display window.")
        end
    end
    limit=get(kwargs,:matrix_limit,(12,12))
    _spectral_pair_valid(limit) && all(>(0),limit) || push!(issues,"matrix_limit must contain two positive Ints.")
    budget=get(kwargs,:max_terms,400)
    budget isa Int && budget>0 || push!(issues,"max_terms must be a positive Int.")
    maps=get(kwargs,:structure_maps,())
    maps isa Union{Tuple,AbstractVector} || push!(issues,"structure_maps must be a tuple or vector of supplied maps.")
    return issues
end

function _spectral_supplied_maps(ss,dims,maps)
    isempty(maps) && return NamedTuple[]
    dims === nothing && throw(ArgumentError("Supplied structure maps require a computed page."))
    field=ChainComplexes.spectral_sequence_summary(_spectral_base(ss)).field
    K=CoreModules.coeff_type(field)
    records=NamedTuple[]
    for m in maps
        m isa NamedTuple && all(k->haskey(m,k),(:source,:target,:matrix,:label)) ||
            throw(ArgumentError("Each structure map requires source, target, matrix and label."))
        all(p->_spectral_pair_valid(p) && haskey(dims,p),(m.source,m.target)) ||
            throw(ArgumentError("A structure map endpoint lies outside this page."))
        m.source != m.target || throw(ArgumentError("A structure line requires distinct bidegrees."))
        m.matrix isa AbstractMatrix{K} || throw(ArgumentError("Supply structure matrices over the page's coefficient field."))
        size(m.matrix)==(dims[m.target],dims[m.source]) || throw(DimensionMismatch("Structure matrix must use target rows and source columns."))
        field isa CoreModules.RealField && !all(isfinite,m.matrix) && throw(ArgumentError("Structure coefficients must be finite."))
        m.label isa AbstractString && !isempty(strip(m.label)) || throw(ArgumentError("A structure map needs a nonempty label."))
        A=Matrix(m.matrix)
        push!(records,(;source=m.source,target=m.target,matrix=A,label=String(m.label),rank=FieldLinAlg.rank(field,A)))
    end
    return records
end

function _spectral_arrow_layers!(layers,source,target,role;dashed=false)
    segments=NTuple{4,Float64}[];heads=Vector{NTuple{2,Float64}}[]
    _hasse_arrow!(segments,heads,Float64.(source),Float64.(target))
    push!(layers,SegmentLayer(segments,_VisualRole(role),1.0,2.0,dashed ? :dash : :solid))
    push!(layers,PolygonLayer(heads,_VisualRole(role),_VisualRole(role),1.0,0.0))
end

function _spectral_chart(ss,r,dims,window,maps;source=nothing,selected=nothing)
    layers=AbstractVisualizationLayer[];records=NamedTuple[]
    for q in window[2][1]:window[2][2], p in window[1][1]:window[1][2]
        ab=(p,q);n=dims === nothing ? nothing : dims[ab]
        push!(records,(;bidegree=ab,dimension=n,status=n === nothing ? :uncomputed : n==0 ? :zero : :nonzero,
            point=(Float64(p),Float64(q))))
    end
    # Every grid site is labelled: blank never substitutes for an unknown or zero.
    zeros_=filter(x->x.dimension==0,records)
    unknown=filter(x->x.dimension === nothing,records)
    for (items,label) in ((zeros_,"0"),(unknown,"?"))
        push!(layers,TextLayer(fill(label,length(items)),[x.point for x in items],_VisualRole(:foreground),13.0))
    end
    nonzero=filter(x->x.dimension !== nothing && x.dimension>0,records)
    target=selected === nothing ? nothing : selected.target
    for x in nonzero
        role=x.bidegree==source ? :selected : x.bidegree==target ? :target : :source
        push!(layers,PointLayer([x.point],_VisualRole(role),0.9,16.0))
        push!(layers,TextLayer([string(x.dimension)],[(x.point[1]+0.16,x.point[2]+0.18)],_VisualRole(:foreground),14.0))
    end
    clipped_maps=0
    for m in maps
        if _spectral_contains(window,m.source) && _spectral_contains(window,m.target)
            m.rank==0 || _spectral_arrow_layers!(layers,m.source,m.target,:foreground;dashed=true)
            midpoint=Tuple((m.source[i]+m.target[i])/2 for i in 1:2)
            push!(layers,TextLayer(["$(m.label) (rank $(m.rank))"],
                [(midpoint[1],midpoint[2]+0.2)],_VisualRole(:foreground),12.0))
        else
            clipped_maps+=1
        end
    end
    if selected !== nothing && target !== nothing && _spectral_contains(window,target)
        _spectral_arrow_layers!(layers,source,target,:selected;dashed=selected.rank==0)
    end
    info=ChainComplexes.spectral_sequence_summary(_spectral_base(ss))
    homological=_spectral_homological(ss)
    first=info.first
    shift=homological ? (first === :vertical ? "(-r, r-1)" : "(r-1, -r)") :
        (first === :vertical ? "(r, 1-r)" : "(1-r, r)")
    subtitle=(homological ? "Homological" : "Cohomological") * "; $first first; d_r shift $shift\n" *
        "Dot label = dimension; 0 = zero term; ? = uncomputed"
    if !isempty(maps)
        subtitle *= "\nDashed structure lines: " * join((m.label for m in maps),", ")
    end
    return VisualizationSpec(:spectral_chart;title="Page $(_spectral_label(r))",subtitle,layers,
        axes=(;xlabel="p",ylabel="q",xlimits=(window[1][1]-0.5,window[1][2]+0.5),
            ylimits=(window[2][1]-0.5,window[2][2]+0.5),aspect=:data,
            xticks=collect(window[1][1]:window[1][2]),yticks=collect(window[2][1]:window[2][2])),
        metadata=(;page=r,terms=records,pick_targets=records,clipped_structure_maps=clipped_maps,
            glyph=:dimension,blank_means_zero=false,convention=homological ? :homological : :cohomological))
end

function _spectral_local_homology(ss,r,p,dims,next_dims,field)
    K=CoreModules.coeff_type(field)
    A=_spectral_matrix(ss,r,p,dims)
    incoming=_spectral_target(ss,r,p;incoming=true)
    B=incoming === nothing ? zeros(K,dims[p],0) : _spectral_matrix(ss,r,incoming,dims)
    equation=_algebra_equation(field,A*B,zeros(K,size(A,1),size(B,2)))
    equation.valid || throw(ArgumentError("Consecutive page differentials do not compose to zero at $p."))
    kernel=FieldLinAlg.nullspace(field,A)
    image=FieldLinAlg.colspace(field,B)
    quotient=ChainComplexes.subquotient_data(kernel,image;field)
    result=dimensions(quotient).quotient
    result==next_dims[p] || throw(ArgumentError("Kernel/image quotient at $p disagrees with the next page."))
    return (;bidegree=p,matrix=A,incoming_matrix=B,incoming,kernel,incoming_image=image,
        quotient_representatives=copy(ChainComplexes.basis(quotient)),
        dimension=dims[p],kernel_dimension=size(kernel,2),incoming_rank=size(image,2),
        next_dimension=result,equation)
end

# Keep the page beside its interpretation, with coefficient panels underneath.
# Size the bounded displayed matrices, allowing wide tables their own row.
function _spectral_layout(panels)
    details=panels[2:end-1]
    panel_width=maximum(_comparison_panel_width,details;init=380)
    columns=min(max(length(details),1),panel_width<=420 ? 3 : panel_width<=620 ? 2 : 1)
    width=max(1200,columns*(panel_width+20)+40)
    positions=Tuple{UnitRange{Int},UnitRange{Int}}[(1:1,1:3)]
    weights=[max(460,_comparison_panel_height(last(panels),div(width,2)-40))]
    span=div(6,columns)
    for (i,panel) in enumerate(details)
        row=2+div(i-1,columns)
        col=1+mod(i-1,columns)*span
        push!(positions,(row:row,col:(col+span-1)))
        length(weights)<row && push!(weights,175)
        weights[row]=max(weights[row],_comparison_panel_height(panel,div(width-40,columns)-20))
    end
    push!(positions,(1:1,4:6))
    return (;panel_positions=positions,panel_row_weights=weights,
        figure_size=(width,120+sum(weights)+16*length(weights)))
end

"""
    visual_spec(ss; kind=:spectral_page, page=2, compute=true, window=nothing, ...)

Inspect a computed cohomological or Tor spectral sequence. The default displays
stored E2 dimensions. `compute=false` never asks the owner to compute a missing
page/differential: those terms are labelled unknown. Select
`kind=:spectral_differential, source=(p,q)` to inspect its actual matrix,
kernel/image and both endpoints' next-page quotients. Finite-page selections
may materialize the owner's entire page of maps. `page=:inf` uses the owner's
associated graded total result, without choosing extension splittings.

`window=((pmin,pmax),(qmin,qmax))` crops only the display. `truncated=true`
declares that the supplied complex is a truncation of a larger intended input;
no convergence assertion is transferred to that larger object. `max_terms=400`
bounds drawn grid sites. `matrix_limit=(12,12)` crops coefficient panels only.
Optional `structure_maps` are labelled, field-valued matrices between terms of
this page: `(source=..., target=..., matrix=..., label=...)`. They are supplied
linear maps, not a product computation or an inferred Leibniz certificate.
"""
function _visual_spec(ss::_SpectralVisualInput,kind::Symbol;page=2,source=nothing,window=nothing,
        compute=true,truncated=false,matrix_limit=(12,12),max_terms=400,structure_maps=())
    info=ChainComplexes.spectral_sequence_summary(_spectral_base(ss))
    ranges=_spectral_ranges(ss)
    window=window === nothing ? ranges : window
    prod(BigInt(w[2])-w[1]+1 for w in window)<=max_terms ||
        throw(ArgumentError("The chart exceeds max_terms=$max_terms. Choose a smaller display window or raise the budget explicitly."))
    dims=_spectral_dimensions(ss,page,compute)
    maps=_spectral_supplied_maps(ss,dims,structure_maps)
    selected=nothing;details=VisualizationSpec[]
    next_page=(page === :inf || page >= info.convergence_bound) ? :inf : page+1
    ready=source !== nothing && dims !== nothing && (compute ||
        (_spectral_known(ss,page;differential=true) && _spectral_known(ss,next_page)))
    if ready
        ndims=_spectral_dimensions(ss,next_page,compute)
        field=info.field
        local_=_spectral_local_homology(ss,page,source,dims,ndims,field)
        target=_spectral_target(ss,page,source)
        target_data=target === nothing ? nothing : _spectral_local_homology(ss,page,target,dims,ndims,field)
        image=FieldLinAlg.colspace(field,local_.matrix)
        selected=merge(local_,(;source,target,target_data,image,rank=size(image,2)))
        push!(details,_algebra_matrix_panel(local_.matrix,"Selected d_$page: $source to $target",field,matrix_limit;
            subtitle="Target rows, source columns; chosen page coordinates"))
        push!(details,_algebra_matrix_panel(local_.kernel,"Kernel at $source",field,matrix_limit;
            subtitle="Columns lie in the source page term",source="kernel",target="E_$page $source"))
        push!(details,_algebra_matrix_panel(image,"Image at $target",field,matrix_limit;
            subtitle="Columns lie in the target page term",source="image",target="E_$page $target"))
    end
    chart=_spectral_chart(ss,page,dims,window,maps;source,selected)
    cropped=window!=ranges
    terminal=page === :inf || page >= info.convergence_bound
    # Stored equality with the owner-computed E_inf is evidence; the last cached
    # index, or one zero differential alone, is not a stabilization argument.
    stable=dims !== nothing && dims==ChainComplexes.page_dims_dict(ss,:inf;nonzero_only=false)
    lines=String["$(_inspection_field_label(info.field)); $(info.first)-first sequence.",
        truncated ? "Declared truncated input: all results concern this supplied complex only." : "Results concern the supplied bounded complex.",
        cropped ? "Display window cropped from $ranges; offscreen terms are retained in the computation." : "Full supplied bidegree window $ranges.",
        dims === nothing ? "This page has not been computed. Question marks are not zeros." :
            stable ? "Dimensions agree with the associated graded limit of this complex." : "This page has not reached the limiting dimensions.",
        "Finite-filtration bound: $(info.convergence_bound). A last computed page is not a convergence test.",
        "The limiting page gives associated graded pieces; no canonical splitting or multiplicative extensions are supplied."]
    if source !== nothing
        if selected === nothing
            push!(lines,"Selected differential unavailable without computation; use compute=true.")
        else
            push!(lines,"d_$page at $source has rank $(selected.rank); target $(selected.target).")
            for local_ in (selected,selected.target_data)
                local_ === nothing && continue
                push!(lines,"At $(local_.bidegree): next dimension = kernel $(local_.kernel_dimension) - incoming image $(local_.incoming_rank) = $(local_.next_dimension).")
            end
            push!(lines,"Kernel/image equations and next-page dimensions checked in the stated field.")
            push!(lines,"Quotient bases are chosen coordinates, not canonical identities between pages.")
            selected.target === nothing || _spectral_contains(window,selected.target) ||
                push!(lines,"The differential target is offscreen; its matrix and contribution are still computed.")
        end
    end
    isempty(maps) || push!(lines,"Supplied structure maps have checked shapes and ranks; no multiplication laws are inferred.")
    panels=[chart;details;[_inspection_text_panel("Read the page",lines)]]
    layout=_spectral_layout(panels)
    return VisualizationSpec(kind;title="Inspect a spectral sequence",panels,
        metadata=(;page,source,window,full_window=ranges,display_cropped=cropped,input_truncated=truncated,
            page_status=dims === nothing ? :uncomputed : :computed,dimensions=dims,selected,
            differential_status=source === nothing ? :not_requested : selected === nothing ? :uncomputed : :computed,
            associated_graded=terminal,limiting_dimensions=stable,convergence_bound=info.convergence_bound,
            extensions=:not_resolved,field=info.field,first=info.first,
            convention=_spectral_homological(ss) ? :homological : :cohomological,
            structure_maps=maps,compute,max_terms,layout...))
end
