# A89: selected chain algebra. All equations use the owner's coefficient field;
# renderers receive copied records and never perform quotient computations.
const _ChainAlgebraInput = Union{ChainComplexes.CochainComplex,ChainComplexes.CochainMap,
    ModuleComplexes.ModuleCochainComplex,ModuleComplexes.ModuleCochainMap,
    DataIngestion.FilteredComplex,DataIngestion.SourceChain}
const _ConnectingInput = Union{ChainComplexes.LongExactSequence,AbelianCategories.SnakeLemmaResult}
const _ChainInspectionInput = Union{_ChainAlgebraInput,_ConnectingInput}
const _CHAIN_KEYS = (:degree,:vertex,:parameter,:field,:convention,:vector,:coordinates,
    :compute,:matrix_limit,:max_terms,:view_dims,:max_cells,:homotopy)

available_visuals(::Union{ChainComplexes.CochainComplex,ChainComplexes.CochainMap,ModuleComplexes.ModuleCochainComplex})=(:chain_inspector,)
available_visuals(::_ConnectingInput)=(:connecting_map,)
_visual_request_keywords(::Union{ChainComplexes.CochainComplex,ChainComplexes.CochainMap,ModuleComplexes.ModuleCochainComplex},::Symbol)=_CHAIN_KEYS
_visual_request_keywords(::_ConnectingInput,::Symbol)=(:degree,:vertex,:map,:vector,:matrix_limit)
_chain_cost()=(;work=:validate_supplied_complex_and_copy_differentials,default=:no_quotient_bases,quotients=:explicit_compute_or_vector,
    witnesses=:checked_noncanonical_preimages,source_geometry=:retained_only,timing=:not_measured)
_visual_request_cost(::Union{ChainComplexes.CochainComplex,ChainComplexes.CochainMap,ModuleComplexes.ModuleCochainComplex},::Symbol)=_chain_cost()
_visual_request_cost(::_ConnectingInput,::Symbol)=(;work=:retained_sequence_maps,representatives=:explicit_vector_selection,timing=:not_measured)

_chain_base(f::Union{ChainComplexes.CochainMap,ModuleComplexes.ModuleCochainMap})=ChainComplexes.source(f)
_chain_base(x)=x
_chain_field(C::ChainComplexes.CochainComplex)=C.field
_chain_field(C::ModuleComplexes.ModuleCochainComplex)=ModuleComplexes.provenance(C).field
_chain_field(x::DataIngestion.SourceChain)=x.field
_chain_field(::DataIngestion.FilteredComplex)=CoreModules.QQField()
_chain_field(f::Union{ChainComplexes.CochainMap,ModuleComplexes.ModuleCochainMap})=_chain_field(_chain_base(f))
_chain_geometry(x)=x isa Union{DataIngestion.FilteredComplex,DataIngestion.SourceChain}
_chain_source(x::DataIngestion.SourceChain)=x.source
_chain_source(x::DataIngestion.FilteredComplex)=x

function _chain_parameters(x;degree=nothing,vertex=nothing,parameter=nothing,field=nothing,
        convention=nothing,vector=nothing,coordinates=:ambient,compute=false,matrix_limit=(12,12),
        max_terms=12,view_dims=nothing,max_cells=10000,homotopy=nothing)
    geometry=_chain_geometry(x);base=_chain_base(x)
    f=field === nothing ? _chain_field(x) : field
    f isa CoreModules.AbstractCoeffField || throw(ArgumentError("field must be a coefficient field."))
    !geometry && f!=_chain_field(x) && throw(ArgumentError("Use the supplied complex's coefficient field."))
    x isa DataIngestion.SourceChain && f!=x.field && throw(ArgumentError("Use the source chain's retained field."))
    conv=convention === nothing ? (geometry && !(x isa DataIngestion.SourceChain && x.kind==:cocycle) ? :homological : :cohomological) : convention
    conv in (:homological,:cohomological) || throw(ArgumentError("convention must be :homological or :cohomological."))
    !geometry && conv!=:cohomological && throw(ArgumentError("A CochainComplex retains its cohomological grading."))
    x isa DataIngestion.SourceChain && conv!=(x.kind==:cocycle ? :cohomological : :homological) &&
        throw(ArgumentError("The retained source chain determines its chain/cochain convention."))
    module_input=base isa ModuleComplexes.ModuleCochainComplex
    v=vertex === nothing ? (module_input ? 1 : nothing) : vertex
    if module_input
        v isa Int && 1<=v<=nvertices(ModuleComplexes.poset(base)) || throw(ArgumentError("vertex must be an actual finite-poset vertex."))
    else
        v === nothing || throw(ArgumentError("vertex applies to a module-valued complex only."))
    end
    p=parameter
    if geometry
        source=_chain_source(x)
        x isa DataIngestion.SourceChain && (DataIngestion.check_source_chain(x;throw=true);p===nothing && (p=x.parameter))
        opts=_complex_options(source;parameter=p,view_dims,max_cells)
        p=opts.parameter
        x isa DataIngestion.SourceChain && p!=x.parameter && throw(ArgumentError("A retained source chain belongs to its recorded parameter."))
        degrees=0:DataTypes.max_dim(source.complex)
    else
        p===nothing && view_dims===nothing || throw(ArgumentError("Parameter and geometric projection require retained source cells."))
        degrees=module_input ? ModuleComplexes.degree_range(base) : ChainComplexes.degree_range(base)
    end
    d=degree === nothing ? (x isa DataIngestion.SourceChain ? x.dimension : first(degrees)) : degree
    d isa Int && d in degrees || throw(ArgumentError("degree must belong to $degrees."))
    x isa DataIngestion.SourceChain && d!=x.dimension && throw(ArgumentError("A SourceChain has one recorded degree; inspect its source for other degrees."))
    coordinates in (:ambient,:class) || throw(ArgumentError("coordinates must be :ambient or :class."))
    compute isa Bool || throw(ArgumentError("compute must be Bool."))
    vector===nothing || vector isa Union{Tuple,AbstractVector} || throw(ArgumentError("vector must be a coefficient vector, or nothing."))
    issues=String[];_comparison_matrix_limit!(issues,matrix_limit)
    isempty(issues) || throw(ArgumentError(join(issues," ")))
    max_terms isa Int && max_terms>=3 || throw(ArgumentError("max_terms must be an Int of at least three."))
    max_cells isa Int && max_cells>0 || throw(ArgumentError("max_cells must be a positive Int."))
    homotopy===nothing || (x isa ModuleComplexes.ModuleCochainMap && homotopy isa ModuleComplexes.ModuleCochainHomotopy) ||
        throw(ArgumentError("homotopy requires a supplied module cochain map and ModuleCochainHomotopy."))
    # Coefficients are data, not a tuple whose length creates another compiled view.
    coefficients=vector===nothing ? nothing : _chain_vector(f,vector,length(vector))
    return (;degree=d,vertex=v,parameter=p,field=f,convention=conv,vector=coefficients,
        coordinates,compute,matrix_limit=Tuple(matrix_limit),max_terms,view_dims,max_cells,homotopy)
end

function _chain_stalk(C::ModuleComplexes.ModuleCochainComplex,q)
    _comparison_checked_report(ModuleComplexes.check_module_complex(C),"Module complex")
    degrees=ModuleComplexes.degree_range(C);field=_chain_field(C);K=CoreModules.coeff_type(field)
    for t in degrees
        _comparison_checked_report(Modules.check_module(ModuleComplexes.component(C,t)),"Complex term")
        _comparison_checked_report(_algebra_morphism_validation(ModuleComplexes.differential(C,t)),"Complex differential")
    end
    dims=[Modules.dim_at(ModuleComplexes.component(C,t),q) for t in degrees]
    ds=SparseMatrixCSC{K,Int}[sparse(Modules.component(ModuleComplexes.differential(C,t),q)) for t in first(degrees):last(degrees)-1]
    return ChainComplexes.CochainComplex{K}(first(degrees),last(degrees),dims,ds;field)
end
function _chain_model(x,o)
    base=_chain_base(x)
    if !_chain_geometry(x)
        C=base isa ModuleComplexes.ModuleCochainComplex ? _chain_stalk(base,o.vertex) : base
        ChainComplexes.check_complex(C;throw=true)
        return (;complex=C,t=o.degree,sign=1,source=nothing,active_ids=nothing,parameter=nothing,
            category=base isa ModuleComplexes.ModuleCochainComplex ? :finite_poset_stalk : :vector_spaces)
    end
    source=_chain_source(x);G=source.complex;counts=DataTypes.cell_counts(G);offsets=cumsum(vcat(0,counts))
    active=[[i for i in 1:counts[d+1] if _complex_active(source,offsets[d+1]+i,o.parameter)] for d in 0:length(counts)-1]
    K=CoreModules.coeff_type(o.field);n=length(counts)-1
    boundaries=SparseMatrixCSC{K,Int}[sparse(K.(DataTypes.boundary_map(G;dim=d)[active[d],active[d+1]])) for d in 1:n]
    if o.convention==:homological
        C=ChainComplexes.CochainComplex{K}(-n,0,reverse(length.(active)),reverse(boundaries);
            labels=reverse(active),field=o.field)
        sign=-1
    else
        C=ChainComplexes.CochainComplex{K}(0,n,length.(active),SparseMatrixCSC{K,Int}[sparse(transpose(b)) for b in boundaries];labels=active,field=o.field)
        sign=1
    end
    ChainComplexes.check_complex(C;throw=true)
    return (;complex=C,t=sign*o.degree,sign,source,active_ids=active,parameter=o.parameter,category=:active_source_cells)
end

function _chain_vector(field,v,n)
    length(v)==n || throw(DimensionMismatch("Expected $n coefficients; received $(length(v))."))
    K=CoreModules.coeff_type(field)
    all(a->!(a isa Bool) && (a isa K || a isa Integer || (field isa Union{CoreModules.QQField,CoreModules.PrimeField} && a isa Rational) ||
        (field isa CoreModules.RealField && a isa Real && isfinite(a))),v) ||
        throw(ArgumentError("Coefficients must belong to the selected field; exact fields do not silently round decimals."))
    result=K[CoreModules.coerce(field,a) for a in v]
    field isa CoreModules.RealField && !all(isfinite,result) && throw(ArgumentError("Numerical coefficients must be finite."))
    return result
end

# The owner's field-aware solve returns a particular original-basis preimage.
# Check the displayed equation as well, including its numerical tolerance.
function _chain_preimage(field,A,z)
    K=eltype(A)
    _algebra_equation(field,z,zeros(K,length(z))).valid && return zeros(K,size(A,2))
    w=vec(ChainComplexes.solve_particular(field,A,reshape(z,:,1)))
    _algebra_equation(field,A*w,z).valid || throw(ArgumentError("The proposed preimage fails the boundary equation."))
    return w
end

function _chain_local(C,t;compute=false,vector=nothing,coordinates=:ambient)
    field=C.field;K=CoreModules.coeff_type(field)
    A=copy(ChainComplexes.differential(C,t));B=copy(ChainComplexes.differential(C,t-1))
    eq=_algebra_equation(field,A*B,zeros(K,size(A,1),size(B,2)))
    eq.valid || throw(ArgumentError("Consecutive differentials do not compose to zero."))
    H=compute || vector!==nothing ? ChainComplexes.cohomology_data(C,t) : nothing
    selected=nothing
    if vector!==nothing
        z=coordinates==:class ? ChainComplexes.cohomology_representative(H,_chain_vector(field,vector,H.dimH)) :
            _chain_vector(field,vector,size(A,2))
        dz=A*z;cycle_equation=_algebra_equation(field,dz,zeros(K,size(A,1)))
        if cycle_equation.valid
            coords=vec(ChainComplexes.cohomology_coordinates(H,z))
            representative=ChainComplexes.cohomology_representative(H,coords)
            w=_chain_preimage(field,B,z-representative)
            boundary=_algebra_equation(field,coords,zeros(K,length(coords))).valid
            selected=(;vector=z,differential=dz,is_cycle=true,is_boundary=boundary,coordinates=coords,
                representative,boundary_adjustment=w,cycle_equation,
                quotient_equation=_algebra_equation(field,z,representative+B*w))
        else
            selected=(;vector=z,differential=dz,is_cycle=false,is_boundary=nothing,coordinates=nothing,
                representative=nothing,boundary_adjustment=nothing,cycle_equation,quotient_equation=nothing)
        end
    end
    quotient=H===nothing ? nothing : (;cycles=copy(H.K),boundaries=copy(H.B),representatives=copy(ChainComplexes.basis(H)),
        dimensions=(;ambient=H.dimC,cycles=H.dimZ,boundaries=H.dimB,quotient=H.dimH))
    return (;outgoing=A,incoming=B,equation=eq,quotient,selected,data=H)
end

function _chain_chart(model,o)
    C=model.complex;degrees=collect(ChainComplexes.degree_range(C));t=model.t
    kept=degrees
    if length(degrees)>o.max_terms
        start=clamp(t-div(o.max_terms,2),first(degrees),last(degrees)-o.max_terms+1)
        kept=collect(start:start+o.max_terms-1)
    end
    layers=AbstractVisualizationLayer[];targets=NamedTuple[]
    for (i,d) in enumerate(kept)
        n=ChainComplexes.component(C,d).dimension;point=(Float64(i),0.0);actual=model.sign*d
        push!(targets,(;degree=actual,dimension=n,point,selection=(;degree=actual)))
        push!(layers,PointLayer([point],_VisualRole(d==t ? :selected : :source),1.0,18.0))
        label=(model.sign==-1 ? "C_$actual" : "C^$actual")*" : $n"
        push!(layers,TextLayer([label],[(point[1],0.35)],_VisualRole(:foreground),14.0))
        if i<length(kept)
            _spectral_arrow_layers!(layers,point,(point[1]+1,0.0),:foreground)
            push!(layers,TextLayer(["d"],[(point[1]+0.5,-0.2)],_VisualRole(:foreground),13.0))
        end
    end
    return VisualizationSpec(:chain_overview;title="The complex and selected degree",
        subtitle="Labels are vector-space dimensions. Arrows follow the differential.",layers,
        axes=(;xlabel="",ylabel="",xlimits=(0.4,length(kept)+0.6),ylimits=(-0.5,0.7),aspect=:auto),
        metadata=(;hide_decorations=true,pick_targets=targets,omitted_terms=length(degrees)-length(kept)))
end
function _chain_layout(panels)
    layout=_comparison_layout(panels;hero=true)
    width=max(layout.figure_size[1],160*(length(first(panels).metadata.pick_targets)+1))
    return merge(layout,(;figure_size=(width,layout.figure_size[2])))
end

function _chain_selection_panels!(panels,local_data,field,limit;prefix="")
    s=local_data.selected;s===nothing && return
    push!(panels,_algebra_matrix_panel(reshape(s.vector,:,1),prefix*"Selected vector",field,limit))
    push!(panels,_algebra_matrix_panel(reshape(s.differential,:,1),prefix*"Its differential",field,limit))
    if s.is_cycle
        push!(panels,_algebra_matrix_panel(reshape(s.coordinates,:,1),prefix*"Quotient coordinates",field,limit))
        push!(panels,_algebra_matrix_panel(reshape(s.representative,:,1),prefix*"Chosen class representative",field,limit))
        push!(panels,_algebra_matrix_panel(reshape(s.boundary_adjustment,:,1),prefix*"Boundary adjustment w",field,limit;
            subtitle="z = representative + d(w); this choice need not be minimal"))
    end
end
function _chain_source_overlay(model,o,z;kind=:chain,degree=o.degree)
    model.source===nothing && return nothing
    o.field isa Union{CoreModules.QQField,CoreModules.PrimeField} || return nothing
    degree<=2 || return nothing
    counts=DataTypes.cell_counts(model.source.complex);K=CoreModules.coeff_type(o.field)
    full=zeros(K,counts[degree+1]);full[model.active_ids[degree+1]]=z
    c=DataIngestion.SourceChain(model.source,full,degree,model.parameter,o.field,kind,(;choice=:supplied_coordinates))
    DataIngestion.check_source_chain(c;throw=true)
    return first(_visual_spec(c,:source_chain;view_dims=o.view_dims,max_cells=o.max_cells).panels)
end

function _chain_map_data(x,model,o,local_data)
    x isa Union{ChainComplexes.CochainMap,ModuleComplexes.ModuleCochainMap} || return nothing
    if x isa ChainComplexes.CochainMap
        ChainComplexes.is_cochain_map(x) || throw(ArgumentError("The supplied map does not commute with differentials."))
        D=ChainComplexes.target(x);F=copy(ChainComplexes.component(x,o.degree))
    else
        _cochain_checked_map(x)
        D=_chain_stalk(ModuleComplexes.target(x),o.vertex)
        F=copy(Modules.component(ModuleComplexes.component(x,o.degree),o.vertex))
    end
    ChainComplexes.check_complex(D;throw=true)
    # Pad only genuine zero terms outside a supplied target's finite support.
    t=o.degree
    if !(t in ChainComplexes.degree_range(D))
        K=CoreModules.coeff_type(o.field)
        D=ChainComplexes.CochainComplex{K}(t,t,[0],SparseMatrixCSC{K,Int}[];field=o.field)
    end
    s=local_data.selected
    image=s===nothing ? nothing : F*s.vector
    target=_chain_local(D,t;compute=o.compute || s!==nothing,vector=image)
    induced=local_data.data===nothing ? nothing : ChainComplexes.induced_map_on_cohomology(local_data.data,target.data,F)
    witness=nothing
    if o.homotopy!==nothing
        H=o.homotopy
        ModuleComplexes.source_map(H)===x || throw(ArgumentError("homotopy must compare this map as its first map."))
        _cochain_checked_map(ModuleComplexes.target_map(H))
        _homotopy_equations(H)
        G=Modules.component(ModuleComplexes.component(ModuleComplexes.target_map(H),t),o.vertex)
        h=Modules.component(ModuleComplexes.component(H,t),o.vertex)
        hnext=Modules.component(ModuleComplexes.component(H,t+1),o.vertex)
        difference=F-G;rhs=target.incoming*h+hnext*local_data.outgoing
        eq=_algebra_equation(o.field,difference,rhs)
        eq.valid || throw(ArgumentError("The homotopy equation fails."))
        witness=(;matrix=copy(h),next=copy(hnext),difference,rhs,equation=eq,
            vector=s===nothing ? nothing : h*s.vector,
            image_difference=s===nothing ? nothing : difference*s.vector,
            noncycle_correction=s===nothing ? nothing : hnext*s.differential)
    end
    return (;matrix=F,induced,target,witness)
end

function _chain_view(x::_ChainAlgebraInput;kwargs...)
    o=_chain_parameters(x;kwargs...);model=_chain_model(x,o)
    vector=o.vector
    if x isa DataIngestion.SourceChain && vector===nothing
        vector=x.coefficients[model.active_ids[o.degree+1]]
        o.coordinates==:ambient || throw(ArgumentError("A retained chain uses ambient coordinates; supply a vector to select class coordinates."))
    end
    local_data=_chain_local(model.complex,model.t;compute=o.compute,vector,coordinates=o.coordinates)
    map_data=_chain_map_data(x,model,o,local_data)
    panels=VisualizationSpec[_chain_chart(model,o),
        _algebra_matrix_panel(local_data.incoming,"Incoming differential",o.field,o.matrix_limit),
        _algebra_matrix_panel(local_data.outgoing,"Outgoing differential",o.field,o.matrix_limit)]
    if o.compute && local_data.quotient!==nothing
        q=local_data.quotient
        push!(panels,_algebra_matrix_panel(q.cycles,"Kernel: cycles",o.field,o.matrix_limit))
        push!(panels,_algebra_matrix_panel(q.boundaries,"Image: boundaries",o.field,o.matrix_limit))
        push!(panels,_algebra_matrix_panel(q.representatives,"Chosen quotient basis",o.field,o.matrix_limit))
    end
    _chain_selection_panels!(panels,local_data,o.field,o.matrix_limit)
    s=local_data.selected
    if s!==nothing
        overlay=_chain_source_overlay(model,o,s.vector;kind=s.is_cycle ? (o.convention==:homological ? :cycle : :cocycle) : :chain)
        overlay===nothing || push!(panels,overlay)
        if model.source!==nothing && s.is_cycle && o.convention==:homological && o.degree<DataTypes.max_dim(model.source.complex)
            filling=_chain_source_overlay(model,o,s.boundary_adjustment;degree=o.degree+1)
            filling===nothing || push!(panels,_algebra_with_title(filling,"Source boundary adjustment"))
        end
    end
    if map_data!==nothing
        push!(panels,_algebra_matrix_panel(map_data.target.incoming,"Target incoming differential",o.field,o.matrix_limit))
        push!(panels,_algebra_matrix_panel(map_data.target.outgoing,"Target outgoing differential",o.field,o.matrix_limit))
        push!(panels,_algebra_matrix_panel(map_data.matrix,"Supplied chain-map component",o.field,o.matrix_limit))
        map_data.induced===nothing || push!(panels,_algebra_matrix_panel(map_data.induced,"Induced quotient map",o.field,o.matrix_limit;
            subtitle="Coordinates use the selected local cycle and quotient bases"))
        _chain_selection_panels!(panels,map_data.target,o.field,o.matrix_limit;prefix="Image: ")
        map_data.witness===nothing || push!(panels,_algebra_matrix_panel(map_data.witness.matrix,"Supplied homotopy h",o.field,o.matrix_limit))
        if map_data.witness!==nothing && s!==nothing
            w=map_data.witness
            push!(panels,_algebra_matrix_panel(reshape(w.vector,:,1),"Homotopy vector h(z)",o.field,o.matrix_limit))
            push!(panels,_algebra_matrix_panel(reshape(w.image_difference,:,1),"Difference f(z) - g(z)",o.field,o.matrix_limit;
                subtitle=s.is_cycle ? "This difference is d(h(z))" : "This difference is d(h(z)) + h(d(z))"))
        end
    end
    lines=["$(o.convention) degree $(o.degree); field $(_inspection_field_label(o.field)).",
        "Incoming followed by outgoing is zero in the declared field.",
        "Matrix rows are target coordinates; columns are source coordinates."]
    o.vertex===nothing || push!(lines,"Finite-poset vertex $(o.vertex). Quotient bases here are local to this stalk.")
    model.source===nothing || push!(lines,"Active source-cell basis (first 12 local IDs): $(first(model.active_ids[o.degree+1],12)); $(length(model.active_ids[o.degree+1])) total. Parameter $(o.parameter).")
    model.source===nothing || !(o.field isa CoreModules.RealField) || push!(lines,"Source overlays require exact rational or prime coefficients; this numerical view shows algebraic coordinates.")
    model.source===nothing || o.degree<=2 || push!(lines,"Source geometry is retained; cells above dimension two are shown in algebraic coordinates.")
    if local_data.quotient===nothing
        push!(lines,"Quotient bases have not been requested. Select a vector or request compute=true.")
    else
        q=local_data.quotient.dimensions
        push!(lines,"$(q.cycles) cycle dimensions / $(q.boundaries) boundary dimensions = $(q.quotient) quotient dimensions.")
        push!(lines,"Bases and representatives are choices; neither uniqueness nor shortest support is asserted.")
    end
    if s!==nothing
        push!(lines,!s.is_cycle ? "The selected vector is not a cycle, so it has no quotient class." :
            s.is_boundary ? "The selected cycle is a boundary: its quotient class is zero." : "The selected cycle has a nonzero quotient class.")
        s.is_cycle && push!(lines,"Verified: z = representative + d(w).")
    end
    o.field isa CoreModules.RealField && push!(lines,"Numerical membership uses atol=$(o.field.atol), rtol=$(o.field.rtol); identities are tolerance checks.")
    map_data===nothing || push!(lines,"The supplied map was checked against the differentials before computing its induced map.")
    map_data===nothing || map_data.witness===nothing || push!(lines,"The supplied homotopy checks f-g = d h + h d. On cycles, h(z) witnesses equal image classes.")
    push!(panels,_inspection_text_panel("How to read this selection",lines))
    layout=_chain_layout(panels)
    return VisualizationSpec(:chain_inspector;title="Cycles, boundaries and their classes",panels,
        metadata=merge(layout,(;options=Base.structdiff(o,(homotopy=o.homotopy,)),degree=o.degree,vertex=o.vertex,parameter=o.parameter,field=o.field,
            convention=o.convention,category=model.category,active_ids=model.active_ids,
            local_data=Base.structdiff(local_data,(data=local_data.data,)),map_data=map_data===nothing ? nothing :
                merge(map_data,(;target=Base.structdiff(map_data.target,(data=map_data.target.data,)))),
            quotient_computed=local_data.quotient!==nothing,representative_choice=:noncanonical,
            source_geometry=model.source===nothing ? :unavailable : :retained,selection=s)))
end

function _append_chain_issues!(issues,x;kwargs...)
    try;_chain_parameters(x;kwargs...)
    catch err;err isa InterruptException && rethrow();push!(issues,sprint(showerror,err));end
    return issues
end
function _append_visual_request_issues!(issues::Vector{String},x::Union{ChainComplexes.CochainComplex,ChainComplexes.CochainMap,ModuleComplexes.ModuleCochainComplex},::Symbol;kwargs...)
    _append_chain_issues!(issues,x;kwargs...)
end
_visual_spec(x::Union{ChainComplexes.CochainComplex,ChainComplexes.CochainMap,ModuleComplexes.ModuleCochainComplex},::Symbol;kwargs...)=_chain_view(x;kwargs...)

function _connecting_options(x;degree=nothing,vertex=nothing,map=:delta,vector=nothing,matrix_limit=(12,12))
    issues=String[];_comparison_matrix_limit!(issues,matrix_limit)
    isempty(issues) || throw(ArgumentError(join(issues," ")))
    vector===nothing || vector isa Union{Tuple,AbstractVector} || throw(ArgumentError("vector must be a coefficient vector."))
    if x isa ChainComplexes.LongExactSequence
        d=degree===nothing ? first(ChainComplexes.degree_range(x)) : degree
        d isa Int && d in ChainComplexes.degree_range(x) || throw(ArgumentError("degree lies outside the retained long exact sequence."))
        vertex===nothing || throw(ArgumentError("A vector-space long exact sequence has no finite-poset vertex."))
        map in (:f,:i,:delta) || throw(ArgumentError("map must be :f, :i or :delta."))
        v=nothing
    else
        degree===nothing || throw(ArgumentError("The snake sequence is indexed by vertex, not cohomological degree."))
        v=vertex===nothing ? 1 : vertex;d=nothing
        P=AbelianCategories.connecting_map(x).dom.Q
        v isa Int && 1<=v<=nvertices(P) || throw(ArgumentError("vertex must be a finite-poset vertex."))
        map in (:k1,:k2,:delta,:c1,:c2) || throw(ArgumentError("map must be :k1, :k2, :delta, :c1 or :c2."))
    end
    field=x isa ChainComplexes.LongExactSequence ? x.triangle.C.field : AbelianCategories.kernel_objects(x).A.field
    coefficients=vector===nothing ? nothing : _chain_vector(field,vector,length(vector))
    return (;degree=d,vertex=v,map,vector=coefficients,matrix_limit=Tuple(matrix_limit))
end
function _append_visual_request_issues!(issues::Vector{String},x::_ConnectingInput,::Symbol;kwargs...)
    try;_connecting_options(x;kwargs...)
    catch err;err isa InterruptException && rethrow();push!(issues,sprint(showerror,err));end
    return issues
end

function _connecting_records(les::ChainComplexes.LongExactSequence,o)
    t=o.degree;entry=ChainComplexes.sequence_entry(les,t);tri=les.triangle;field=tri.C.field
    records=NamedTuple[]
    for (name,S,T,F,A) in ((:f,entry.C,entry.D,ChainComplexes.component(tri.f,t),entry.f),
            (:i,entry.D,entry.cone,ChainComplexes.component(tri.i,t),entry.i),
            (:delta,entry.cone,entry.Cshift,ChainComplexes.component(tri.p,t),entry.delta))
        expected=ChainComplexes.induced_map_on_cohomology(S,T,F)
        equation=_algebra_equation(field,A,expected)
        equation.valid || throw(ArgumentError("Retained $name disagrees with its inducing chain map."))
        push!(records,(;name,matrix=copy(A),source_data=S,target_data=T,chain_matrix=copy(F),equation))
    end
    for f in (tri.f,tri.i,tri.p)
        ChainComplexes.is_cochain_map(f) || throw(ArgumentError("A retained triangle map is not a cochain map."))
    end
    return (;field,records,dimensions=[entry.C.dimH,entry.D.dimH,entry.cone.dimH,entry.Cshift.dimH],
        labels=["H^$t(C)","H^$t(D)","H^$t(Cone)","H^$(t+1)(C)"],kind=:cone_sequence)
end
function _connecting_records(sn::AbelianCategories.SnakeLemmaResult,o)
    AbelianCategories.check_snake_lemma(sn;throw=true)
    k=AbelianCategories.kernel_objects(sn);c=AbelianCategories.cokernel_objects(sn)
    objects=(k.A,k.B,k.C,c.A,c.B,c.C);field=k.A.field
    records=[(;name,matrix=copy(Modules.component(f,o.vertex)),source_data=nothing,target_data=nothing,
        chain_matrix=nothing,equation=nothing) for (name,f) in ((:k1,sn.k1),(:k2,sn.k2),(:delta,sn.delta),(:c1,sn.c1),(:c2,sn.c2))]
    return (;field,records,dimensions=[Modules.dim_at(M,o.vertex) for M in objects],
        labels=["ker(alpha)","ker(beta)","ker(gamma)","coker(alpha)","coker(beta)","coker(gamma)"],kind=:snake_sequence)
end
function _connecting_chart(data,o)
    layers=AbstractVisualizationLayer[];targets=NamedTuple[]
    for (i,label) in enumerate(data.labels)
        point=(Float64(i),0.0)
        push!(layers,PointLayer([point],_VisualRole(:source),1.0,17.0))
        push!(layers,TextLayer(["$label : $(data.dimensions[i])"],[(point[1],0.35)],_VisualRole(:foreground),13.0))
        if i<=length(data.records)
            r=data.records[i];role=r.name==o.map ? :selected : :foreground
            _spectral_arrow_layers!(layers,point,(point[1]+1,0.0),role)
            midpoint=(point[1]+0.5,-0.08)
            push!(targets,(;map=r.name,point=midpoint,dimension=size(r.matrix),selection=(;map=r.name)))
            push!(layers,TextLayer([string(r.name)],[(midpoint[1],-0.2)],_VisualRole(role),14.0))
        end
    end
    return VisualizationSpec(:connecting_sequence;title=data.kind==:cone_sequence ? "The mapping-cone exact sequence" : "The snake-lemma exact sequence",
        subtitle="Select an arrow; every displayed matrix belongs to this supplied sequence.",layers,
        axes=(;xlabel="",ylabel="",xlimits=(0.4,length(data.labels)+0.6),ylimits=(-0.5,0.7),aspect=:auto),
        metadata=(;hide_decorations=true,pick_targets=targets))
end
function _visual_spec(x::_ConnectingInput,::Symbol;kwargs...)
    o=_connecting_options(x;kwargs...);data=_connecting_records(x,o);field=data.field;K=CoreModules.coeff_type(field)
    records=data.records;idx=only(findall(r->r.name==o.map,records));r=records[idx]
    checks=NamedTuple[]
    for i in 1:length(records)-1
        A,B=records[i].matrix,records[i+1].matrix
        eq=_algebra_equation(field,B*A,zeros(K,size(B,1),size(A,2)))
        exact=FieldLinAlg.rank(field,A)+FieldLinAlg.rank(field,B)==size(A,1)
        eq.valid && exact || throw(ArgumentError("The supplied sequence is not exact at $(data.labels[i+1])."))
        push!(checks,(;term=data.labels[i+1],equation=eq,exact))
    end
    panels=VisualizationSpec[_connecting_chart(data,o),_algebra_matrix_panel(r.matrix,"Selected map $(o.map)",field,o.matrix_limit;
        source=data.labels[idx],target=data.labels[idx+1])]
    selected=nothing
    if o.vector!==nothing
        v=_chain_vector(field,o.vector,size(r.matrix,2));w=r.matrix*v
        source_rep,target_rep,image,adjustment=nothing,nothing,nothing,nothing
        equation=nothing
        if x isa ChainComplexes.LongExactSequence
            source_rep=ChainComplexes.cohomology_representative(r.source_data,v)
            image=r.chain_matrix*source_rep
            target_rep=ChainComplexes.cohomology_representative(r.target_data,w)
            target_complex=o.map==:f ? x.triangle.D : o.map==:i ? x.triangle.cone : x.triangle.Cshift
            B=ChainComplexes.differential(target_complex,o.degree-1)
            adjustment=_chain_preimage(field,B,image-target_rep)
            equation=_algebra_equation(field,image,target_rep+B*adjustment)
            for (values,title) in ((source_rep,"Source cocycle"),(image,"Image under the chain map"),
                    (target_rep,"Target class representative"),(adjustment,"Target boundary adjustment"))
                push!(panels,_algebra_matrix_panel(reshape(values,:,1),title,field,o.matrix_limit))
            end
        end
        selected=(;vector=v,image_coordinates=w,source_representative=source_rep,chain_image=image,
            target_representative=target_rep,boundary_adjustment=adjustment,equation)
        push!(panels,_algebra_matrix_panel(reshape(v,:,1),"Selected source coordinates",field,o.matrix_limit))
        push!(panels,_algebra_matrix_panel(reshape(w,:,1),"Image coordinates",field,o.matrix_limit))
    end
    lines=["Field $(_inspection_field_label(field)). Target rows and source columns use the retained bases.",
        "Consecutive maps compose to zero, and image/kernel dimensions agree at the displayed interior terms."]
    if data.kind==:cone_sequence
        push!(lines,"delta is the cone projection to C[1], read as H^t(Cone) -> H^(t+1)(C).")
        push!(lines,"Cone(f)^t = D^t + C^(t+1); d = [d_D f; 0 -d_C]. The shift uses -d_C.")
        selected===nothing || push!(lines,"The chain-level image and target representative differ by the displayed boundary; this verifies the selected class image.")
    else
        push!(lines,"Vertex $(o.vertex). These are kernel/cokernel coordinates of the supplied snake result.")
        push!(lines,"The original two-row diagram is not retained by this result; no diagram-chase lifts are invented.")
    end
    push!(lines,"Choose a vector explicitly to inspect its image. Basis choices are not canonical.")
    field isa CoreModules.RealField && push!(lines,"Numerical equations and ranks use atol=$(field.atol), rtol=$(field.rtol).")
    push!(panels,_inspection_text_panel("Connecting-map interpretation",lines))
    return VisualizationSpec(:connecting_map;title="Follow a connecting map",panels,
        metadata=merge(_chain_layout(panels),(;options=o,degree=o.degree,vertex=o.vertex,map=o.map,
            field,records=[Base.structdiff(r,(source_data=r.source_data,target_data=r.target_data)) for r in records],checks,selection=selected,sequence_kind=data.kind,quotient_computed=true)))
end
