# Finite-category derived spaces, selected quotient classes and real products.
# No multiplication or quotient basis is requested by the dimension view.
const _DerivedSpace = Union{DerivedFunctors.ExtSpaceProjective,DerivedFunctors.ExtSpaceInjective,
    DerivedFunctors.ExtSpace,DerivedFunctors.TorSpace,DerivedFunctors.TorSpaceSecond}
const _DerivedAlgebra = Union{DerivedFunctors.ExtAlgebra,DerivedFunctors.TorAlgebra}
const _DerivedElement = Union{DerivedFunctors.ExtElement,DerivedFunctors.TorElement}
const _DerivedVisualInput = Union{_DerivedSpace,_DerivedAlgebra,_DerivedElement}

available_visuals(::_DerivedSpace) = (:graded_dimensions,:derived_class)
available_visuals(::DerivedFunctors.ExtSpaceProjective) = (:graded_dimensions,:derived_class,:yoneda_product)
available_visuals(::_DerivedAlgebra) = (:graded_dimensions,:derived_class,:product_table)
available_visuals(::_DerivedElement) = (:derived_class,)
_derived_space(x::_DerivedSpace) = x
_derived_space(x::DerivedFunctors.HomSpace) = x
_derived_space(x::DerivedFunctors.ExtAlgebra) = DerivedFunctors.underlying_ext_space(x)
_derived_space(x::DerivedFunctors.TorAlgebra) = DerivedFunctors.underlying_tor_space(x)
_derived_space(x::_DerivedElement) = _derived_space(DerivedFunctors.parent_algebra(x))
_derived_tor(x) = _derived_space(x) isa Union{DerivedFunctors.TorSpace,DerivedFunctors.TorSpaceSecond}
_derived_name(x) = x isa DerivedFunctors.HomSpace ? "Hom" : _derived_tor(x) ? "Tor" : "Ext"

function _visual_request_keywords(x::_DerivedVisualInput,kind::Symbol)
    kind===:graded_dimensions && return (:matrix_limit,)
    kind===:derived_class && return x isa _DerivedElement ? (:matrix_limit,) : (:degree,:vector,:matrix_limit)
    kind===:product_table && return (:p,:q,:left,:right,:matrix_limit,:max_products,:max_entries,:validate_algebra)
    return (:p,:q,:right,:left_vector,:right_vector,:target,:matrix_limit)
end
_visual_request_cost(x::_DerivedVisualInput,kind::Symbol) =
    (;work=kind===:graded_dimensions ? :stored_dimensions : kind===:derived_class ? :selected_representative :
        kind===:product_table ? :one_degree_pair_table : :selected_yoneda_composition,
      quotient_bases=kind!==:graded_dimensions,all_degree_products=:never_implicit,
      crop=:display_only,timing=:not_measured)

function _derived_degree(x,t)
    t isa Int && t in DerivedFunctors.degree_range(_derived_space(x)) ||
        throw(ArgumentError("degree must be in $(DerivedFunctors.degree_range(_derived_space(x)))."))
    t
end
function _derived_coordinates(x,t,v)
    field=DerivedFunctors.provenance(x).field;n=DerivedFunctors.dim(_derived_space(x),t)
    v isa Union{Tuple,AbstractVector} || throw(ArgumentError("Class coordinates must be a vector."))
    _chain_vector(field,v,n)
end
function _derived_options(x,kind;matrix_limit=(12,12),kwargs...)
    issues=String[];_comparison_matrix_limit!(issues,matrix_limit)
    isempty(issues) || throw(ArgumentError(join(issues," ")))
    kind===:graded_dimensions && return nothing
    if kind===:derived_class
        if x isa _DerivedElement
            t=DerivedFunctors.element_degree(x);v=DerivedFunctors.element_coordinates(x)
        else
            t=get(kwargs,:degree,0);v=get(kwargs,:vector,nothing)
        end
        _derived_degree(x,t)
        v===nothing || _derived_coordinates(x,t,v)
    elseif kind===:product_table
        p=get(kwargs,:p,0);q=get(kwargs,:q,0)
        _derived_degree(x,p);_derived_degree(x,q)
        p<=last(DerivedFunctors.degree_range(x))-q || throw(ArgumentError("Product exceeds the computed degree range."))
        for (key,t) in ((:left,p),(:right,q))
            get(kwargs,key,nothing)===nothing || _derived_coordinates(x,t,kwargs[key])
        end
        (get(kwargs,:left,nothing)===nothing)==(get(kwargs,:right,nothing)===nothing) ||
            throw(ArgumentError("Supply both left and right coordinates to select a product."))
        for key in (:max_products,:max_entries)
            b=get(kwargs,key,key===:max_products ? 4096 : 1_000_000)
            b isa Int && b>0 || throw(ArgumentError("$key must be a positive Int."))
        end
        pairs=big(DerivedFunctors.dim(x,p))*DerivedFunctors.dim(x,q)
        pairs<=get(kwargs,:max_products,4096) || throw(ArgumentError("Product table exceeds max_products; choose smaller degrees or raise the explicit budget."))
        pairs*DerivedFunctors.dim(x,p+q)<=get(kwargs,:max_entries,1_000_000) ||
            throw(ArgumentError("Structure constants exceed max_entries; raise the explicit budget."))
        a=get(kwargs,:validate_algebra,false)
        a isa Bool || throw(ArgumentError("validate_algebra must be Bool."))
        a && !(x isa DerivedFunctors.TorAlgebra) && throw(ArgumentError("validate_algebra is for supplied Tor algebra data."))
    else
        R=get(kwargs,:right,nothing);T=get(kwargs,:target,nothing)
        R isa DerivedFunctors.ExtSpaceProjective || throw(ArgumentError("right must be a projective Ext(L,M) model."))
        T===nothing || T isa DerivedFunctors.ExtSpaceProjective || throw(ArgumentError("target must be a projective Ext(L,N) model."))
        p=get(kwargs,:p,0);q=get(kwargs,:q,0)
        _derived_degree(x,p);_derived_degree(R,q)
        p<=last(DerivedFunctors.degree_range(R))-q || throw(ArgumentError("The right resolution must be computed through p+q."))
        T===nothing || _derived_degree(T,p+q)
        _derived_coordinates(x,p,get(kwargs,:left_vector,nothing))
        _derived_coordinates(R,q,get(kwargs,:right_vector,nothing))
        DerivedFunctors.provenance(x).field==DerivedFunctors.provenance(R).field ||
            throw(ArgumentError("Both Ext models must use the same coefficient field."))
    end
    nothing
end
function _append_visual_request_issues!(issues::Vector{String},x::_DerivedVisualInput,kind::Symbol;kwargs...)
    try;_derived_options(x,kind;kwargs...)
    catch err;err isa InterruptException && rethrow();push!(issues,sprint(showerror,err));end
    issues
end

function _derived_context(x)
    E=_derived_space(x);p=DerivedFunctors.provenance(x);tor=_derived_tor(x)
    source=copy(dimensions(DerivedFunctors.source_module(E)).stalks)
    target=copy(dimensions(DerivedFunctors.target_module(E)).stalks)
    short(ds)=join(ds[1:min(length(ds),16)],", ")*(length(ds)>16 ? ", ... (full vector in metadata)" : "")
    lines=["Field: $(_inspection_field_label(p.field)). Finite base: $(nvertices(p.base_poset)) vertices.",
        tor ? "Right module on P^op; left module on P. Homological degrees." : "Representations of the recorded finite P. Cohomological degrees; Ext^0 = Hom.",
        "Model: $(p.model). No ambient category identification is asserted.",
        "$(tor ? "Right" : "Source") stalk dimensions: $(short(source)).",
        "$(tor ? "Left" : "Target") stalk dimensions: $(short(target))."]
    haskey(p,:canonical_model) && push!(lines,"Canonical coordinate model: $(p.canonical_model).")
    return (;provenance=p,source_dimensions=source,target_dimensions=target,lines)
end
function _derived_dimensions(x;matrix_limit=(12,12))
    E=_derived_space(x);ctx=_derived_context(x);ts=collect(DerivedFunctors.degree_range(E))
    ds=[DerivedFunctors.dim(E,t) for t in ts]
    panel=_presentation_matrix_panel(reshape(ds,1,:),"$(_derived_name(x)) dimensions",["dimension"],
        ["degree $t" for t in ts],matrix_limit;
        subtitle="Computed degrees only; zero entries are retained",metadata=(;matrix_corner="dimension / degree",matrix_row_heading="Dimension",matrix_column_heading="Degree"))
    lines=[ctx.lines...,"No representatives or products were requested."]
    x isa DerivedFunctors.TorAlgebra && push!(lines,"Multiplication requires the supplied chain products; dimensions alone do not define it.")
    VisualizationSpec(:graded_dimensions;title="Graded $(_derived_name(x)) over a finite category",
        panels=[panel,_inspection_text_panel("The represented computation",lines)],
        metadata=(;provenance=ctx.provenance,degrees=ts,dimensions=ds,
            source_dimensions=ctx.source_dimensions,target_dimensions=ctx.target_dimensions,
            representatives_computed=false,products_computed=false,panel_columns=1,
            panel_positions=[(1:1,1:1),(2:2,1:1)],
            panel_row_weights=[140,_derived_text_height(lines)],figure_size=(1150,220+_derived_text_height(lines))))
end

_derived_matrix_height(n,limit)=max(150,100+22*min(n,limit[1]))
_derived_text_height(lines)=max(230,60+26*length(lines))
function _derived_column(v,title,field,limit;basis="coordinate",column="selected",empty="the coordinate space has dimension zero")
    isempty(v) && return _inspection_text_panel(title,
        ["Empty coordinate vector $column (0 x 1).",uppercasefirst(empty)*"."];
        subtitle=_inspection_field_label(field),metadata=(;field,matrix_size=(0,1),coordinates=copy(v)))
    _presentation_matrix_panel(reshape(v,:,1),title,["$basis $i" for i in eachindex(v)],[column],limit;
        subtitle=_inspection_field_label(field),metadata=(;field,matrix_corner="$basis / vector",empty_matrix_reason=empty))
end

# Always express the selected class in its owner's quotient basis. A new
# cohomology calculation on the same complex need not choose that same basis.
function _derived_class_data(x,t,v;representative=nothing)
    E=_derived_space(x);field=DerivedFunctors.provenance(E).field
    c=_derived_coordinates(E,t,v);C=DerivedFunctors.representative_complex(E)
    ct=_derived_tor(E) ? -t : t;K=CoreModules.coeff_type(field)
    z=representative===nothing ? DerivedFunctors.representative(E,t,c) : copy(representative)
    A=copy(ChainComplexes.differential(C,ct));B=copy(ChainComplexes.differential(C,ct-1))
    chain=_algebra_equation(field,A*B,zeros(K,size(A,1),size(B,2)))
    cycle=_algebra_equation(field,A*z,zeros(K,size(A,1)))
    chain.valid && cycle.valid || throw(ArgumentError("The supplied representative complex or selected cycle is invalid."))
    recovered=DerivedFunctors.coordinates(E,t,z)
    equation=_algebra_equation(field,recovered,c)
    equation.valid || throw(ArgumentError("The representative does not recover its requested quotient coordinates."))
    canonical=DerivedFunctors.representative(E,t,c)
    adjustment=_chain_preimage(field,B,z-canonical)
    return (;degree=t,complex_degree=ct,coordinates=copy(c),representative=copy(z),canonical=copy(canonical),
        incoming=B,outgoing=A,boundary_adjustment=adjustment,cycle_equation=cycle,coordinate_equation=equation,
        quotient_equation=_algebra_equation(field,z,canonical+B*adjustment),
        zero_class=_algebra_equation(field,c,zeros(K,length(c))).valid)
end
function _derived_class_view(x;degree=0,vector=nothing,matrix_limit=(12,12))
    E=_derived_space(x);ctx=_derived_context(x)
    if x isa _DerivedElement
        degree=DerivedFunctors.element_degree(x);vector=DerivedFunctors.element_coordinates(x)
    elseif vector===nothing
        K=CoreModules.coeff_type(ctx.provenance.field)
        vector=zeros(K,DerivedFunctors.dim(E,degree));isempty(vector) || (vector[1]=one(K))
    end
    s=_derived_class_data(E,degree,vector);field=ctx.provenance.field
    panels=[_derived_column(s.coordinates,"Class coordinates",field,matrix_limit;basis="class basis",column="c"),
        _derived_column(s.representative,"Selected representative",field,matrix_limit;column="z"),
        _algebra_matrix_panel(s.incoming,"Incoming boundary",field,matrix_limit),
        _algebra_matrix_panel(s.outgoing,"Outgoing differential",field,matrix_limit)]
    lines=[ctx.lines...,"Degree $degree. $(s.zero_class ? "Zero quotient class." : "Nonzero quotient class.")",
        "The differential annihilates the representative; quotient coordinates are checked.",
        "Bases and representatives are choices. Adding an incoming boundary preserves the class."]
    _derived_tor(E) && push!(lines,"Chain degree $degree is cochain degree $(-degree) in representative_complex.")
    push!(panels,_inspection_text_panel("Reading the class",lines))
    heights=[_derived_matrix_height(max(length(s.coordinates),length(s.representative)),matrix_limit),
        _derived_matrix_height(max(size(s.incoming,1),size(s.outgoing,1)),matrix_limit),_derived_text_height(lines)]
    VisualizationSpec(:derived_class;title="A selected $(_derived_name(x)) class",panels,
        metadata=(;provenance=ctx.provenance,source_dimensions=ctx.source_dimensions,target_dimensions=ctx.target_dimensions,
            selection=s,basis_convention=:owner_quotient_basis,
            panel_columns=2,panel_positions=[(1:1,1:1),(1:1,2:2),(2:2,1:1),(2:2,2:2),(3:3,1:2)],
            panel_row_weights=heights,figure_size=(1250,sum(heights)+80)))
end

function _derived_table(x;p=0,q=0,left=nothing,right=nothing,matrix_limit=(12,12),
        max_products=4096,max_entries=1_000_000,validate_algebra=false)
    ctx=_derived_context(x);field=ctx.provenance.field
    report=validate_algebra ? DerivedFunctors.check_tor_algebra(x;algebraic=true,throw=true) : nothing
    mu=copy(DerivedFunctors.multiplication_matrix(x,p,q))
    pairs=[(i,j) for i in 1:DerivedFunctors.dim(x,p) for j in 1:DerivedFunctors.dim(x,q)]
    panels=VisualizationSpec[_presentation_matrix_panel(mu,"Multiplication structure constants",
        ["result $k" for k in axes(mu,1)],["b$i * a$j" for (i,j) in pairs],matrix_limit;
        subtitle="Degrees ($p, $q) -> $(p+q); right index varies fastest",
        metadata=(;field,matrix_corner="result / pair",matrix_row_heading="Target basis",matrix_column_heading="Left * right basis"))]
    selected=nothing
    if left!==nothing
        l=_derived_coordinates(x,p,left);r=_derived_coordinates(x,q,right);c=mu*kron(l,r)
        selected=_derived_class_data(x,p+q,c)
        for (v,title) in ((l,"Left class, degree $p"),(r,"Right class, degree $q"),(c,"Product class, degree $(p+q)"))
            push!(panels,_derived_column(v,title,field,matrix_limit;basis="class basis"))
        end
    end
    lines=[ctx.lines...,"$(length(pairs)) basis products computed/reused. Cropping changes display only.",
        "Target rows and ordered input pairs use chosen quotient bases.",
        _derived_tor(x) ? "Supplied chain products checked for cycle preservation and independence modulo boundaries." :
        "Yoneda composition: left after right. No extra Koszul sign is inserted."]
    _derived_tor(x) && push!(lines,report===nothing ? "Global Leibniz, associativity and unit checks were not requested." :
        "Leibniz, associativity and supplied unit checked through the computed range.")
    selected===nothing || push!(lines,selected.zero_class ? "The selected product is zero in the quotient." : "The selected product is nonzero in the quotient.")
    push!(panels,_inspection_text_panel("What the table establishes",lines))
    positions=selected===nothing ? [(1:1,1:3),(2:2,1:3)] :
        [(1:1,1:3),(2:2,1:1),(2:2,2:2),(2:2,3:3),(3:3,1:3)]
    heights=selected===nothing ? [_derived_matrix_height(size(mu,1),matrix_limit),_derived_text_height(lines)] :
        [_derived_matrix_height(size(mu,1),matrix_limit),
         _derived_matrix_height(max(length(left),length(right),length(selected.coordinates)),matrix_limit),_derived_text_height(lines)]
    VisualizationSpec(:product_table;title="$(_derived_name(x)) products in chosen bases",panels,
        metadata=(;provenance=ctx.provenance,source_dimensions=ctx.source_dimensions,target_dimensions=ctx.target_dimensions,
            degrees=(p,q),matrix=mu,basis_pairs=pairs,selection=selected,
            algebra_validation=report,basis_convention=:owner_quotient_basis,panel_columns=3,panel_positions=positions,
            panel_row_weights=heights,figure_size=(1350,sum(heights)+80)))
end

function _derived_yoneda(E;p=0,q=0,right,left_vector,right_vector,target=nothing,matrix_limit=(12,12))
    b=_derived_coordinates(E,p,left_vector);a=_derived_coordinates(right,q,right_vector)
    T,c,z=DerivedFunctors.yoneda_product(E,p,b,right,q,a;ELN=target,return_cocycle=true)
    ctx=_derived_context(T);field=ctx.provenance.field
    l=_derived_class_data(E,p,b);r=_derived_class_data(right,q,a)
    s=_derived_class_data(T,p+q,c;representative=z)
    middle=copy(dimensions(DerivedFunctors.source_module(E)).stalks)
    middle_label=join(middle[1:min(length(middle),16)],", ")*(length(middle)>16 ? ", ... (full vector in metadata)" : "")
    panels=VisualizationSpec[]
    for (v,title,label,basis) in ((b,"beta: Ext^$p(M,N)","beta","class basis"),
            (a,"alpha: Ext^$q(L,M)","alpha","class basis"),
            (c,"beta after alpha: Ext^$(p+q)(L,N)","product","class basis"),
            (z,"Computed composite cocycle","z","coordinate"),
            (s.boundary_adjustment,"Boundary adjustment to chosen representative","w","coordinate"))
        push!(panels,_derived_column(v,title,field,matrix_limit;basis,column=label))
    end
    lines=[ctx.lines...,"Source L -> intermediate M -> target N describes Ext composition, not an extension sequence.",
        "Intermediate M stalk dimensions: $middle_label.",
        "Middle-module maps and coordinates must agree, not merely their dimensions.",
        "No extra Koszul sign. Representatives depend on comparison lifts.",
        "Composite = chosen representative + incoming boundary * adjustment.",
        s.zero_class ? "This product is zero in Ext." : "This product is nonzero in Ext."]
    push!(panels,_inspection_text_panel("Composition and its witness",lines))
    heights=[_derived_matrix_height(max(length(b),length(a),length(c)),matrix_limit),
        _derived_matrix_height(max(length(z),length(s.boundary_adjustment)),matrix_limit),_derived_text_height(lines)]
    VisualizationSpec(:yoneda_product;title="Composing two Ext classes",panels,
        metadata=(;provenance=ctx.provenance,source_dimensions=ctx.source_dimensions,target_dimensions=ctx.target_dimensions,
            target_space=T,left=l,right=r,selection=s,
            intermediate_dimensions=middle,
            sign_convention=:unshifted_comparison_composition,extension_sequence=:not_constructed,
            panel_columns=3,panel_positions=[(1:1,1:1),(1:1,2:2),(1:1,3:3),(2:2,1:1),(2:2,2:3),(3:3,1:3)],
            panel_row_weights=heights,figure_size=(1350,sum(heights)+80)))
end

"""
    visual_spec(E; kind=:graded_dimensions, matrix_limit=(12,12))
    visual_spec(E; kind=:derived_class, degree=0, vector=nothing, matrix_limit=(12,12))
    visual_spec(A; kind=:product_table, p=0, q=0, left=nothing, right=nothing,
                max_products=4096, max_entries=1_000_000, validate_algebra=false,
                matrix_limit=(12,12))
    visual_spec(E_MN; kind=:yoneda_product, right=E_LM, p=0, q=0,
                left_vector, right_vector, target=nothing, matrix_limit=(12,12))

Inspect finite-category Ext/Tor spaces and their algebras. The default only
reads dimensions. A class request materializes one chosen representative;
`vector` means coordinates in the owner's quotient basis (default: first
basis class, or the zero class when the group is zero). Elements carry their
own degree and coordinates. Tor keeps homological degree labels.

An explicit product-table request computes/reuses one pair of degrees. Input
pairs index columns with the right basis index fastest; target basis vectors
index rows. Both table budgets are checked before multiplication. Optional
`left` and `right` coordinate vectors select a product. Tor requires actual
supplied chain products; `validate_algebra=true` additionally requests global
algebraic checks, potentially generating products beyond the displayed pair.

Yoneda composition accepts projective Ext models and computes left after right;
the right model must extend through p+q. A supplied target fixes its output
coordinates. Copied matrices and checked cocycle/quotient equations are retained
in metadata. No extension sequence is inferred from a class or dimension table.
All figures are static specifications and work with installed render backends.
"""
function _visual_spec(x::_DerivedVisualInput,kind::Symbol;kwargs...)
    kind===:graded_dimensions && return _derived_dimensions(x;kwargs...)
    kind===:derived_class && return _derived_class_view(x;kwargs...)
    kind===:product_table && return _derived_table(x;kwargs...)
    _derived_yoneda(x;kwargs...)
end
